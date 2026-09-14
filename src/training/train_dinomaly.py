"""
Train Dinomaly on CBIS-DDSM normal (benign) mammograms.

Only the bottleneck and decoder train; the DINOv2 encoder is frozen. Defaults
follow Dinomaly's dinomaly_mvtec_uni.py: StableAdamW at lr 2e-3 -> 2e-4 on a
warmup-cosine schedule, 10k iterations, batch 16, the hard-mining cosine loss
with p ramped 0 -> 0.9 over the first 1000 iters.

`train(TrainConfig(...))` is the entry point and returns a history dict; it's
what notebooks/dinomaly_train_local.ipynb calls. `main()` is the CLI wrapper:

    python -m src.training.train_dinomaly --smoke
    python -m src.training.train_dinomaly --iters 10000 --eval-every 2000
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
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

from src.data.cbis_ddsm import load_datasets
from src.eval.anomaly import evaluate
from src.models._ref import StableAdamW
from src.models.dinomaly import build_dinomaly
from src.models.losses import global_cosine_hm_percent
from src.models.lr_schedule import WarmCosineSchedule

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainConfig:
    backbone: str = "vit_base"
    iters: int = 10000
    batch_size: int = 16
    lr: float = 2e-3
    final_lr: float = 2e-4
    warmup_iters: int = 100
    eval_every: int = 2000
    num_workers: int = 4
    seed: int = 1
    log_every: int = 50
    hm_ramp_iters: int = 1000     # iters to ramp the hard-mining p from 0 to hm_p
    hm_p: float = 0.9
    hm_factor: float = 0.1
    grad_clip: float = 0.1
    out_dir: str | None = None    # None -> runs/dinomaly_<timestamp>

    @classmethod
    def smoke(cls, **overrides):
        base = dict(iters=150, eval_every=150, warmup_iters=20, hm_ramp_iters=150)
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


def train(config: TrainConfig, datasets=None, device=None, progress=True, on_eval=None):
    """
    Returns a history dict:
        iters, loss, lr            per-iteration lists (sampled every log_every)
        evals                      list of {iter, auroc, ap, f1_max}
        batches_per_epoch, epochs  epoch bookkeeping
        out_dir, elapsed_min
    `on_eval(history)` is called after each evaluation, for live plotting.
    """
    set_seed(config.seed)
    device = device or pick_device()

    if datasets is None:
        datasets = load_datasets()
    train_loader = DataLoader(
        datasets["train_normal"], batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, drop_last=True,
        persistent_workers=config.num_workers > 0,
    )
    test_loader = DataLoader(
        ConcatDataset([datasets["test_normal"], datasets["test_anomaly"]]),
        batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers,
    )
    batches_per_epoch = len(train_loader)
    total_epochs = math.ceil(config.iters / batches_per_epoch)

    model, trainable = build_dinomaly(config.backbone)
    model = model.to(device)

    optimizer = StableAdamW(
        trainable.parameters(), lr=config.lr, betas=(0.9, 0.999),
        weight_decay=1e-4, amsgrad=True, eps=1e-10,
    )
    schedule = WarmCosineSchedule(
        optimizer, base_lr=config.lr, final_lr=config.final_lr,
        total_iters=config.iters, warmup_iters=config.warmup_iters,
    )

    out_dir = Path(config.out_dir) if config.out_dir else REPO_ROOT / "runs" / time.strftime("dinomaly_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))

    history = {
        "iters": [], "loss": [], "lr": [], "evals": [],
        "batches_per_epoch": batches_per_epoch,
        "epochs": total_epochs,
        "out_dir": str(out_dir),
    }

    def say(msg):
        (tqdm.write if progress else print)(msg)

    say(f"device {device} | backbone {config.backbone} | "
        f"trainable {sum(p.numel() for p in trainable.parameters())/1e6:.1f}M")
    say(f"train_normal {len(datasets['train_normal'])} | "
        f"test_normal {len(datasets['test_normal'])} | test_anomaly {len(datasets['test_anomaly'])} | "
        f"{batches_per_epoch} batches/epoch -> {total_epochs} epochs")
    say(f"-> {out_dir}")

    model.train()
    model.encoder.eval()  # frozen backbone stays in eval mode

    batches = _endless(train_loader)
    t0 = time.time()
    bar = tqdm(range(1, config.iters + 1), disable=not progress, dynamic_ncols=True)
    for it in bar:
        img, _ = next(batches)
        img = img.to(device)

        en, de = model(img)
        p = min(config.hm_p * it / config.hm_ramp_iters, config.hm_p)
        loss = global_cosine_hm_percent(en, de, p=p, factor=config.hm_factor)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=config.grad_clip)
        optimizer.step()
        lr = schedule.step()

        epoch = (it - 1) // batches_per_epoch + 1
        if it % config.log_every == 0 or it == 1:
            loss_v = loss.item()
            history["iters"].append(it)
            history["loss"].append(loss_v)
            history["lr"].append(lr)
            bar.set_postfix(epoch=f"{epoch}/{total_epochs}", loss=f"{loss_v:.4f}", lr=f"{lr:.1e}")

        if it % config.eval_every == 0 or it == config.iters:
            metrics = evaluate(model, test_loader, device)
            metrics["iter"] = it
            metrics["epoch"] = epoch
            history["evals"].append(metrics)
            say(f"[eval @ iter {it} / epoch {epoch}]  AUROC {metrics['auroc']:.4f}  "
                f"AP {metrics['ap']:.4f}  F1max {metrics['f1_max']:.4f}")
            torch.save(
                {"trainable": trainable.state_dict(), "iter": it, "metrics": metrics,
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
    parser.add_argument("--backbone", default="vit_base", choices=["vit_small", "vit_base", "vit_large"])
    parser.add_argument("--iters", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--final-lr", type=float, default=2e-4)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--smoke", action="store_true", help="short run: 150 iters")
    args = parser.parse_args()

    if args.smoke:
        config = TrainConfig.smoke(
            backbone=args.backbone, batch_size=args.batch_size, num_workers=args.num_workers,
            seed=args.seed, out_dir=args.out_dir,
        )
    else:
        config = TrainConfig(
            backbone=args.backbone, iters=args.iters, batch_size=args.batch_size, lr=args.lr,
            final_lr=args.final_lr, warmup_iters=args.warmup_iters, eval_every=args.eval_every,
            num_workers=args.num_workers, seed=args.seed, out_dir=args.out_dir,
        )

    print(f"config: {config}", flush=True)
    train(config)


if __name__ == "__main__":
    sys.exit(main())
