"""
Fits an AnomalyDINO-DPMM model (Schulthess & Konukoglu, MICCAI 2025; see
src/models/dpmm.py for the full citation) on each BMAD modality, one model
per modality -- same single-class setting as src/training/train_bmad.py.

Architecturally this is a different family of method from Dinomaly2
(src/training/train_bmad.py): there is no trainable encoder/decoder and no
backprop. A frozen DINOv2 backbone (src/models/anomaly_dpmm.py) extracts
per-patch embeddings; "training" fits a truncated stick-breaking Dirichlet
Process Mixture (src/models/dpmm.py) to the normal training patches via
online EM, one `dpmm.step()` per image batch. At test time, a patch's
anomaly score is its negative log-likelihood under the fitted mixture.

Uses the same val/test protocol as train_bmad.py: val picks the checkpoint
(`best_checkpoint.pt`, by val I-AUROC), test is what gets reported.

    python -m src.training.train_bmad_dpmm --modality liver --smoke
    python -m src.training.train_bmad_dpmm --modality brain
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.bmad import MODALITIES, load_modality
from src.eval.bmad_dpmm import evaluate_dpmm_modality
from src.models.anomaly_dpmm import embedding_dim, extract_patch_features, load_dpmm_encoder
from src.models.dpmm import DPMM
from src.training.train_bmad import pick_device, set_seed

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class BMADDPMMConfig:
    modality: str = "brain"           # one of src.data.bmad.MODALITIES
    backbone: str = "dinov2_vits14"   # matches the method's own reference configs
    image_size: int = 448
    crop_size: int = 392              # 392 = 28 * 14, an integer number of dinov2 patches
    normalize_embeddings: bool = True
    max_num_components: int = 500     # DPMM truncation level K
    update_rate: float = 0.2
    schedule: str = "ema"             # "ema" or "exp"
    cov_type: str = "diag"            # "full", "diag" or "spherical"
    reg_covar: float = 1e-6
    alpha: float = 2.0
    alpha_fixed: bool = False
    epochs: int = 40
    eval_every_epochs: int = 5
    batch_size: int = 8               # image batch for feature extraction (no backward pass)
    max_ratio: float = 0.01
    resize_mask: int = 256
    num_workers: int = 4
    seed: int = 1
    out_dir: str | None = None

    @classmethod
    def smoke(cls, **overrides):
        base = dict(epochs=2, eval_every_epochs=1, max_num_components=20)
        base.update(overrides)
        return cls(**base)


def train(config: BMADDPMMConfig, device=None, progress=True, on_eval=None):
    """
    Returns a history dict with the same shape as train_bmad.train's: iters,
    loss, evals (list of {iter, epoch, val, test}), batches_per_epoch, epochs,
    has_masks, best_val_iter, out_dir, elapsed_min. `iters`/`evals[i]['iter']`
    count epochs here (there is no gradient-step notion for this method).
    """
    set_seed(config.seed)
    device = device or pick_device()

    train_data, val_data, test_data, has_masks = load_modality(
        config.modality, config.image_size, config.crop_size,
    )
    train_loader = DataLoader(
        train_data, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, drop_last=False, persistent_workers=config.num_workers > 0,
    )
    batches_per_epoch = len(train_loader)

    encoder = load_dpmm_encoder(config.backbone, device=device)
    D = embedding_dim(encoder)
    dpmm = DPMM(
        K=config.max_num_components, D=D, update_rate=config.update_rate,
        schedule=config.schedule, alpha=config.alpha, alpha_fixed=config.alpha_fixed,
        cov_type=config.cov_type, reg_covar=config.reg_covar, device=str(device),
    )

    out_dir = Path(config.out_dir) if config.out_dir else \
        REPO_ROOT / "runs" / time.strftime(f"bmad_dpmm_{config.modality}_seed{config.seed}_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))

    history = {
        "iters": [], "loss": [], "evals": [],
        "batches_per_epoch": batches_per_epoch, "epochs": config.epochs,
        "modality": config.modality, "has_masks": has_masks, "best_val_iter": None,
        "out_dir": str(out_dir), "method": "anomalydino_dpmm",
    }
    best_val_auroc = -1.0

    def say(msg):
        (tqdm.write if progress else print)(msg)

    n_active = sum(p.numel() for p in encoder.parameters())
    say(f"device {device} | modality {config.modality} (pixel masks: {has_masks}) | "
        f"backbone {config.backbone} (frozen, {n_active/1e6:.1f}M params) | D={D} | K={config.max_num_components}")
    say(f"train images {len(train_data)} | val images {len(val_data)} | "
        f"test images {len(test_data)} | {batches_per_epoch} batches/epoch -> {config.epochs} epochs")
    say(f"-> {out_dir}")

    def run_eval(dataset):
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
        return evaluate_dpmm_modality(
            encoder, dpmm, loader, device, has_masks,
            normalize=config.normalize_embeddings, max_ratio=config.max_ratio, resize_mask=config.resize_mask,
        )

    def fmt(metrics):
        s = f"I-AUROC {metrics['auroc_sp']:.4f}  I-AP {metrics['ap_sp']:.4f}  I-F1 {metrics['f1_sp']:.4f}"
        if has_masks:
            s += f"  P-AUROC {metrics['auroc_px']:.4f}  P-AP {metrics['ap_px']:.4f}  P-AUPRO {metrics['aupro_px']:.4f}"
        return s

    t0 = time.time()
    initialized = False
    epoch_bar = tqdm(range(1, config.epochs + 1), disable=not progress, dynamic_ncols=True)
    for epoch in epoch_bar:
        running_loss, n_batches = 0.0, 0
        for img, _ in tqdm(train_loader, disable=not progress, leave=False, dynamic_ncols=True):
            img = img.to(device)
            features, _ = extract_patch_features(encoder, img, normalize=config.normalize_embeddings)
            B, N, Dd = features.shape
            flat = features.reshape(B * N, Dd)

            if not initialized:
                dpmm.initialize(flat)
                initialized = True

            dpmm.step(flat)
            with torch.no_grad():
                running_loss += (-dpmm.score(flat)).item()
            n_batches += 1

        mean_loss = running_loss / max(n_batches, 1)
        history["iters"].append(epoch)
        history["loss"].append(mean_loss)
        epoch_bar.set_postfix(epoch=f"{epoch}/{config.epochs}", loss=f"{mean_loss:.4f}")

        if epoch % config.eval_every_epochs == 0 or epoch == config.epochs:
            val_metrics = run_eval(val_data)
            test_metrics = run_eval(test_data)
            history["evals"].append({"iter": epoch, "epoch": epoch, "val": val_metrics, "test": test_metrics})

            say(f"[eval @ epoch {epoch}] val: {fmt(val_metrics)}")
            say(f"{' ' * len(f'[eval @ epoch {epoch}] ')}test: {fmt(test_metrics)}")

            checkpoint = {"dpmm": dpmm.state_dict(), "epoch": epoch,
                          "val_metrics": val_metrics, "test_metrics": test_metrics,
                          "config": asdict(config)}
            torch.save(checkpoint, out_dir / "checkpoint.pt")
            if val_metrics["auroc_sp"] > best_val_auroc:
                best_val_auroc = val_metrics["auroc_sp"]
                history["best_val_iter"] = epoch
                torch.save(checkpoint, out_dir / "best_checkpoint.pt")

            if on_eval is not None:
                on_eval(history)

    history["elapsed_min"] = (time.time() - t0) / 60
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    say(f"done in {history['elapsed_min']:.1f} min  ->  {out_dir}")
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", default="brain", choices=list(MODALITIES))
    parser.add_argument("--backbone", default="dinov2_vits14")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--eval-every-epochs", type=int, default=5)
    parser.add_argument("--max-num-components", type=int, default=500)
    parser.add_argument("--update-rate", type=float, default=0.2)
    parser.add_argument("--schedule", default="ema", choices=["ema", "exp"])
    parser.add_argument("--cov-type", default="diag", choices=["full", "diag", "spherical"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--smoke", action="store_true", help="short run: 2 epochs, K=20")
    args = parser.parse_args()

    overrides = dict(
        modality=args.modality, backbone=args.backbone, update_rate=args.update_rate,
        schedule=args.schedule, cov_type=args.cov_type, num_workers=args.num_workers,
        seed=args.seed, out_dir=args.out_dir,
    )
    if args.smoke:
        config = BMADDPMMConfig.smoke(**overrides)
    else:
        config = BMADDPMMConfig(
            epochs=args.epochs, eval_every_epochs=args.eval_every_epochs,
            max_num_components=args.max_num_components, **overrides,
        )

    print(f"config: {config}", flush=True)
    train(config)


if __name__ == "__main__":
    sys.exit(main())
