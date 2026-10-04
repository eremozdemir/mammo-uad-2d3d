"""
Applies the reproduced Dinomaly2 recipe (src/models/dinomaly2.py) to each
BMAD modality individually. Unlike MVTec-AD/VisA, a BMAD modality has no
sub-category dimension; it is already a single normal-vs-abnormal problem.
This module therefore trains and evaluates one modality per run rather than
building a multi-class ConcatDataset the way train_dinomaly2.py does.

Uses the same optimizer, schedule, and loss recipe as train_dinomaly2.py:
StableAdamW with the bottleneck's compress layer at a 10x lower learning
rate, warmup followed by a constant rate, and hard-mining cosine loss ramped
to p=0.9 over the first hm_ramp_iters.

Each modality ships its own val and test splits (BMAD's own split, not one
carved out here). At every eval checkpoint the model is scored on both: val
selects the checkpoint to keep (`best_checkpoint.pt`, by val I-AUROC, the
one metric available for every modality), and test is the number reported.
This keeps checkpoint selection from being graded against the same numbers
it is judged on.

    python -m src.training.train_bmad --modality brain --smoke
    python -m src.training.train_bmad --modality chest
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

from src.data.bmad import MODALITIES, load_modality
from src.eval.bmad import evaluate_modality
from src.models._ref2 import StableAdamW
from src.models.dinomaly2 import build_dinomaly2
from src.models.losses import global_cosine_hm_percent
from src.models.lr_schedule import WarmupCosineRatioSchedule

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class BMADConfig:
    modality: str = "brain"         # one of src.data.bmad.MODALITIES
    backbone: str = "vit_base"
    image_size: int = 448
    crop_size: int = 392
    total_iters: int = 40000
    batch_size: int = 16
    dropout: float = 0.4
    target_layers: list[int] | None = None  # None: build_dinomaly2's per-backbone default
    ll_ratio: float = 0.9           # hard-mining p, ramped over hm_ramp_iters
    ll_factor: float = 0.1
    hm_ramp_iters: int = 1000
    lr: float = 2e-3
    warmup_iters: int = 100
    final_ratio: float = 1.0        # no LR decay, matching Dinomaly2's own default
    grad_clip: float = 0.1
    max_ratio: float = 0.01
    resize_mask: int = 256
    eval_every: int = 2000
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


def train(config: BMADConfig, device=None, progress=True, on_eval=None):
    """
    Returns a history dict: iters, loss, lr, evals (list of {iter, epoch,
    val, test}), batches_per_epoch, epochs, has_masks, best_val_iter,
    out_dir, elapsed_min.
    """
    set_seed(config.seed)
    device = device or pick_device()

    train_data, val_data, test_data, has_masks = load_modality(
        config.modality, config.image_size, config.crop_size,
    )

    train_loader = DataLoader(
        train_data, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, drop_last=True, persistent_workers=config.num_workers > 0,
    )
    batches_per_epoch = len(train_loader)
    total_epochs = math.ceil(config.total_iters / batches_per_epoch)

    model, trainable, param_groups = build_dinomaly2(
        config.backbone, dropout=config.dropout, target_layers=config.target_layers,
    )
    model = model.to(device)

    optimizer = StableAdamW(param_groups, lr=config.lr, betas=(0.9, 0.999),
                            weight_decay=1e-4, amsgrad=False, eps=1e-10)
    schedule = WarmupCosineRatioSchedule(
        optimizer, total_iters=config.total_iters, warmup_iters=config.warmup_iters,
        final_ratio=config.final_ratio,
    )

    out_dir = Path(config.out_dir) if config.out_dir else \
        REPO_ROOT / "runs" / time.strftime(f"bmad_dinomaly2_{config.modality}_seed{config.seed}_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))

    history = {
        "iters": [], "loss": [], "lr": [], "evals": [],
        "batches_per_epoch": batches_per_epoch, "epochs": total_epochs,
        "modality": config.modality, "has_masks": has_masks, "best_val_iter": None,
        "out_dir": str(out_dir),
    }
    best_val_auroc = -1.0

    def say(msg):
        (tqdm.write if progress else print)(msg)

    n_train_params = sum(p.numel() for p in trainable.parameters())
    say(f"device {device} | modality {config.modality} (pixel masks: {has_masks}) | "
        f"backbone {config.backbone} | trainable {n_train_params/1e6:.1f}M")
    say(f"train images {len(train_data)} | val images {len(val_data)} | "
        f"test images {len(test_data)} | {batches_per_epoch} batches/epoch -> {total_epochs} epochs")
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
            history["lr"].append(lrs[-1])
            bar.set_postfix(epoch=f"{epoch}/{total_epochs}", loss=f"{loss_v:.4f}", lr=f"{lrs[-1]:.1e}")

        if it % config.eval_every == 0 or it == config.total_iters:
            def run_eval(dataset):
                loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
                return evaluate_modality(
                    model, loader, device, has_masks, max_ratio=config.max_ratio, resize_mask=config.resize_mask,
                )

            val_metrics = run_eval(val_data)
            test_metrics = run_eval(test_data)
            history["evals"].append({"iter": it, "epoch": epoch, "val": val_metrics, "test": test_metrics})

            def fmt(metrics):
                s = f"I-AUROC {metrics['auroc_sp']:.4f}  I-AP {metrics['ap_sp']:.4f}  I-F1 {metrics['f1_sp']:.4f}"
                if has_masks:
                    s += (f"  P-AUROC {metrics['auroc_px']:.4f}  P-AP {metrics['ap_px']:.4f}  "
                          f"P-AUPRO {metrics['aupro_px']:.4f}")
                return s

            say(f"[eval @ iter {it} / epoch {epoch}] val: {fmt(val_metrics)}")
            say(f"{' ' * len(f'[eval @ iter {it} / epoch {epoch}] ')}test: {fmt(test_metrics)}")

            checkpoint = {"trainable": trainable.state_dict(), "iter": it,
                          "val_metrics": val_metrics, "test_metrics": test_metrics,
                          "config": asdict(config)}
            torch.save(checkpoint, out_dir / "checkpoint.pt")
            if val_metrics["auroc_sp"] > best_val_auroc:
                best_val_auroc = val_metrics["auroc_sp"]
                history["best_val_iter"] = it
                torch.save(checkpoint, out_dir / "best_checkpoint.pt")

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
    parser.add_argument("--modality", default="brain", choices=list(MODALITIES))
    parser.add_argument("--backbone", default="vit_base", choices=["vit_small", "vit_base", "vit_large"])
    parser.add_argument("--total-iters", type=int, default=40000)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--smoke", action="store_true", help="short run: 150 iters")
    args = parser.parse_args()

    if args.smoke:
        config = BMADConfig.smoke(
            modality=args.modality, backbone=args.backbone, num_workers=args.num_workers,
            seed=args.seed, out_dir=args.out_dir,
        )
    else:
        config = BMADConfig(
            modality=args.modality, backbone=args.backbone, total_iters=args.total_iters,
            eval_every=args.eval_every, num_workers=args.num_workers, seed=args.seed, out_dir=args.out_dir,
        )

    print(f"config: {config}", flush=True)
    train(config)


if __name__ == "__main__":
    sys.exit(main())
