"""
Reproduce Dinomaly2 on MVTec-AD or VisA (multi-class / "uni" setting).

Mirrors Dinomaly2's own dinomaly_2D.py: StableAdamW with the bottleneck's first
(compress) layer at a 10x lower lr than the rest, warmup then constant lr
(final_ratio=1.0 is their own script default -- no decay), 40k iterations,
batch 16, hard-mining cosine loss ramped to p=0.9 over the first 1000 iters,
eval every eval_every iters averaging image+pixel metrics across every
category. Config matches their README's example commands for these two
datasets exactly (image_size=448, crop_size=392, all other args left default).

    python -m src.training.train_dinomaly2 --dataset mvtec --smoke
    python -m src.training.train_dinomaly2 --dataset visa
"""

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.mvtec_visa import MVTEC_CLASSES, VISA_CLASSES, load_multiclass, resolve_mvtec_dir, resolve_visa_dir
from src.eval.pixel import evaluate_category
from src.models._ref2 import StableAdamW
from src.models.dinomaly2 import build_dinomaly2
from src.models.losses import global_cosine_hm_percent
from src.models.lr_schedule import WarmupCosineRatioSchedule

REPO_ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "mvtec": (resolve_mvtec_dir, MVTEC_CLASSES),
    "visa": (resolve_visa_dir, VISA_CLASSES),
}


@dataclass
class Dinomaly2Config:
    dataset: str = "mvtec"          # "mvtec" | "visa"
    backbone: str = "vit_base"
    image_size: int = 448
    crop_size: int = 392
    total_iters: int = 40000
    batch_size: int = 16
    dropout: float = 0.4
    ll_ratio: float = 0.9           # hard-mining p, ramped over hm_ramp_iters
    ll_factor: float = 0.1
    hm_ramp_iters: int = 1000
    lr: float = 2e-3
    warmup_iters: int = 100
    final_ratio: float = 1.0        # Dinomaly2's own script default: no LR decay
    grad_clip: float = 0.1
    max_ratio: float = 0.01
    resize_mask: int = 256
    eval_every: int = 5000
    num_workers: int = 4
    seed: int = 1
    log_every: int = 50
    out_dir: str | None = None

    @classmethod
    def smoke(cls, **overrides):
        base = dict(total_iters=150, eval_every=150, warmup_iters=20, hm_ramp_iters=150)
        base.update(overrides)
        return cls(**base)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _endless(loader):
    while True:
        for batch in loader:
            yield batch


def _mean_metrics(per_class: dict) -> dict:
    keys = next(iter(per_class.values())).keys()
    return {k: float(np.mean([m[k] for m in per_class.values()])) for k in keys}


def train(config: Dinomaly2Config, device=None, progress=True, on_eval=None):
    """
    Returns a history dict: iters, loss, lr, evals (list of {iter, epoch, mean,
    per_class}), batches_per_epoch, epochs, out_dir, elapsed_min.
    """
    set_seed(config.seed)
    device = device or pick_device()

    resolve_dir, classes = DATASETS[config.dataset]
    root = resolve_dir()
    train_data, test_sets = load_multiclass(root, classes, config.image_size, config.crop_size)

    train_loader = DataLoader(
        train_data, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, drop_last=True, persistent_workers=config.num_workers > 0,
    )
    batches_per_epoch = len(train_loader)
    total_epochs = math.ceil(config.total_iters / batches_per_epoch)

    model, trainable, param_groups = build_dinomaly2(config.backbone, dropout=config.dropout)
    model = model.to(device)

    optimizer = StableAdamW(param_groups, lr=config.lr, betas=(0.9, 0.999),
                            weight_decay=1e-4, amsgrad=False, eps=1e-10)
    schedule = WarmupCosineRatioSchedule(
        optimizer, total_iters=config.total_iters, warmup_iters=config.warmup_iters,
        final_ratio=config.final_ratio,
    )

    out_dir = Path(config.out_dir) if config.out_dir else \
        REPO_ROOT / "runs" / time.strftime(f"dinomaly2_{config.dataset}_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))

    history = {
        "iters": [], "loss": [], "lr": [], "evals": [],
        "batches_per_epoch": batches_per_epoch, "epochs": total_epochs,
        "classes": classes, "out_dir": str(out_dir),
    }

    def say(msg):
        (tqdm.write if progress else print)(msg)

    n_train_params = sum(p.numel() for p in trainable.parameters())
    say(f"device {device} | dataset {config.dataset} ({len(classes)} classes) | "
        f"backbone {config.backbone} | trainable {n_train_params/1e6:.1f}M")
    say(f"train images {len(train_data)} | {batches_per_epoch} batches/epoch -> {total_epochs} epochs")
    say(f"-> {out_dir}")

    model.train()
    model.encoder.eval()

    batches = _endless(train_loader)
    t0 = time.time()
    bar = tqdm(range(1, config.total_iters + 1), disable=not progress, dynamic_ncols=True)
    for it in bar:
        img, _ = next(batches)
        img = img.to(device)

        en, de = model(img)
        p = min(config.ll_ratio * it / config.hm_ramp_iters, config.ll_ratio)
        loss = global_cosine_hm_percent(en, de, p=p, factor=config.ll_factor)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=config.grad_clip)
        optimizer.step()
        lrs = schedule.step()

        epoch = (it - 1) // batches_per_epoch + 1
        if it % config.log_every == 0 or it == 1:
            loss_v = loss.item()
            history["iters"].append(it)
            history["loss"].append(loss_v)
            history["lr"].append(lrs[-1])  # decoder's lr (the un-scaled one)
            bar.set_postfix(epoch=f"{epoch}/{total_epochs}", loss=f"{loss_v:.4f}", lr=f"{lrs[-1]:.1e}")

        if it % config.eval_every == 0 or it == config.total_iters:
            per_class = {}
            for cls, test_data in test_sets.items():
                # num_workers=0: MVTecDataset is loaded dynamically (see _ref2.py /
                # mvtec_visa.py) to avoid the `models`-package collision with the
                # Dinomaly (v1) submodule, which means it can't be pickled to a
                # DataLoader worker subprocess. Eval runs far less often than
                # training, so this costs little.
                loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, num_workers=0)
                per_class[cls] = evaluate_category(
                    model, loader, device, max_ratio=config.max_ratio, resize_mask=config.resize_mask,
                )
            mean = _mean_metrics(per_class)
            history["evals"].append({"iter": it, "epoch": epoch, "mean": mean, "per_class": per_class})
            say(f"[eval @ iter {it} / epoch {epoch}] mean I-AUROC {mean['auroc_sp']:.4f}  "
                f"I-AP {mean['ap_sp']:.4f}  I-F1 {mean['f1_sp']:.4f}  P-AUROC {mean['auroc_px']:.4f}  "
                f"P-AUPRO {mean['aupro_px']:.4f}")
            torch.save(
                {"trainable": trainable.state_dict(), "iter": it, "mean_metrics": mean,
                 "config": asdict(config)},
                out_dir / "checkpoint.pt",
            )
            model.train()
            model.encoder.eval()
            if on_eval is not None:
                on_eval(history)

    history["elapsed_min"] = (time.time() - t0) / 60
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    say(f"done in {history['elapsed_min']:.1f} min  ->  {out_dir}")
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="mvtec", choices=list(DATASETS))
    parser.add_argument("--backbone", default="vit_base", choices=["vit_small", "vit_base", "vit_large"])
    parser.add_argument("--total-iters", type=int, default=40000)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--smoke", action="store_true", help="short run: 150 iters")
    args = parser.parse_args()

    if args.smoke:
        config = Dinomaly2Config.smoke(
            dataset=args.dataset, backbone=args.backbone, num_workers=args.num_workers,
            seed=args.seed, out_dir=args.out_dir,
        )
    else:
        config = Dinomaly2Config(
            dataset=args.dataset, backbone=args.backbone, total_iters=args.total_iters,
            eval_every=args.eval_every, num_workers=args.num_workers, seed=args.seed, out_dir=args.out_dir,
        )

    print(f"config: {config}", flush=True)
    train(config)


if __name__ == "__main__":
    sys.exit(main())
