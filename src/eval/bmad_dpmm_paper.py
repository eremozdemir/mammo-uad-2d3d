"""
Re-evaluation of already-fitted AnomalyDINO-DPMM checkpoints (Schulthess &
Konukoglu, MICCAI 2025; see src/models/dpmm.py for the citation) under two
protocols, without refitting anything:

  - "paper": the official code's own evaluation (anomalydino-dpmm/, config
    *_diag.yaml + src/{data,backbones,utils,evaluate,test}.py). Images resized
    so the smaller edge is 448 (bicubic, antialiased; no center crop), DINOv2
    patch tokens L2-normalized and bilinearly resampled to a 32x32 grid, one
    patch score per token (likelihood / Euclidean / cosine to the nearest
    component with pi > 1e-6), the 32x32 map bilinearly upsampled to each
    mask's original resolution, no smoothing (masks binarized per LABEL_RULES:
    the official code's own rule, and every nonzero lesion pixel), and pixel AUROC / AUPR computed
    over every test pixel (anomalib 0.6.0 `AUROC`/`AUPR`: trapezoidal area
    under the ROC and under the precision-recall curve). Dice at 10/5/1% FPR
    uses thresholds taken from the normal validation patch scores (test.py
    `get_threshold`), then torchmetrics `Dice` (micro, foreground) over all
    test pixels.
  - "ours": section 8's pipeline (src/eval/bmad_dpmm.py): 448x448 resize +
    392 center crop, 28x28 patch map, bilinear to 392 then 256, Gaussian blur
    (k=5, sigma=4), masks bilinear-resized/cropped then nearest to 256, image
    score = mean of the top 1% of the map. Only the patch score is swapped.

Checkpoint selection follows the official train.py: the epoch with the lowest
mean negative log-likelihood (pi > 1e-6) on the *normal* validation images.

Pixel curves are accumulated as a 2^24-bin histogram of the score per class
rather than by sorting every pixel (RESC alone has ~0.95e9 test pixels at its
original 512x1024 resolution). With 2^24 bins the bin width is at the float32
resolution of the normalized score, so ties introduced by binning are
negligible; `exact_check` in the notebook compares it against sklearn on one
full run.

    python -m src.eval.bmad_dpmm_paper --modality brain
"""

import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import gaussian_blur
from tqdm.auto import tqdm

from src.data.bmad import MODALITIES, load_modality, resolve_bmad_root
from src.models.anomaly_dpmm import PATCH_SIZE, extract_patch_features, load_dpmm_encoder
from src.models.dpmm import DPMM

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "runs" / "bmad_dpmm_paper_eval"

PAPER_RESOLUTION = 448     # config/*_diag.yaml `resolution`
FEATURE_RESOLUTION = 32    # config `feature_resolution_eval`, applied by utils.resample_features
WEIGHT_THRESHOLD = 1e-6    # src/evaluate.py:44
PAPER_BATCH_SIZE = 12      # config `embedding_batch_size`
FPR_LEVELS = (10, 5, 1)    # src/test.py:185-189
N_BINS = 2 ** 24

# Names follow the paper's Table 3; official map keys in test.py:177-184.
#   likelihood -> "anomaly_map" (-log p, pi > 1e-6)
#   euclidean  -> "distance_map" (distance_to_nearest_cluster, covariance_weighted_norm=False)
#   cosine     -> "cosine_distance_map" (1 - cosine similarity to the nearest mean; the paper's score)
# nll_all is section 8's original score: -log p over all components (no pi threshold).
PAPER_SCORES = ("likelihood", "euclidean", "cosine")

# Brain (BraTS) masks in this copy of BMAD hold three tumor sub-region values
# {58, 122, 255} and RESC masks {128, 191, 255}; Liver is binary {0, 255}.
#   official: what the official code counts as anomalous. data.py ToTensor maps
#             255 -> 1.0 and the rest into (0, 1); torchmetrics 0.10.3's
#             _binary_clf_curve keeps only `target == pos_label (1)`, and
#             test.py:44 casts Dice targets with `.int()` -> only value 255.
#   nonzero:  every nonzero (lesion) pixel, at original resolution, no dilation.
LABEL_RULES = {
    "official": lambda raw: raw == 255,
    "nonzero": lambda raw: raw > 0,
}


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------

def paper_image_transform():
    """src/data.py:49-53 (and backbones.py:128-132)."""
    return transforms.Compose([
        transforms.Resize(size=PAPER_RESOLUTION, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


class PaperSplit(Dataset):
    """Normal-only or anomalous-only images of one BMAD val/test split, as the
    official main.py loads them (separate normal/anomalous datasets). Labels are
    kept at the original image resolution (`resize_labels: False`, data.py:60)
    as raw uint8 values; normal images get an all-zero mask (data.py:73).
    Binarization happens later, per LABEL_RULES."""

    def __init__(self, modality: str, split: str, kind: str):
        cfg = MODALITIES[modality]
        root = resolve_bmad_root() / cfg.subpath
        split_dir = cfg.val_dir if split == "val" else "test"
        sub = "good" if kind == "normal" else cfg.abnormal_dir
        self.img_paths = sorted((root / split_dir / sub / "img").glob("*.png"))
        self.label_dir = root / split_dir / sub / "label" if (cfg.has_masks and kind == "anomalous") else None
        self.transform = paper_image_transform()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.label_dir is not None:
            label = Image.open(self.label_dir / path.name).convert("L")
        else:
            label = Image.new("L", image.size)
        label = torch.from_numpy(np.array(label, dtype=np.uint8))[None]
        return self.transform(image), label


# ----------------------------------------------------------------------------
# features + patch scores
# ----------------------------------------------------------------------------

@torch.inference_mode()
def extract_paper_features(encoder, images: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """[B, 3, H, W] -> [B, 32*32, D]. backbones.py crop_image (top-left crop to a
    multiple of 14) + extract_features (L2 norm after the last block), then
    utils.resample_features (bilinear to 32x32; the identity for square inputs)."""
    _, _, height, width = images.shape
    images = images[:, :, :height - height % PATCH_SIZE, :width - width % PATCH_SIZE]
    grid = (images.shape[2] // PATCH_SIZE, images.shape[3] // PATCH_SIZE)
    tokens = encoder.get_intermediate_layers(images)[0]
    if normalize:
        tokens = F.normalize(tokens, dim=2)
    B, _, D = tokens.shape
    feats = tokens.reshape(B, grid[0], grid[1], D).permute(0, 3, 1, 2)
    feats = F.interpolate(feats, (FEATURE_RESOLUTION, FEATURE_RESOLUTION), mode="bilinear")
    return feats.permute(0, 2, 3, 1).reshape(B, FEATURE_RESOLUTION * FEATURE_RESOLUTION, D)


@torch.inference_mode()
def score_patches(dpmm: DPMM, flat: torch.Tensor, scores, chunk: int = 4096) -> dict:
    """flat: [M, D] -> {score name: [M]}, higher = more anomalous, plus
    'loglik' (log p with pi > 1e-6, what train.py's validation loss averages)."""
    out = {name: [] for name in list(scores) + ["loglik"]}
    for batch in torch.split(flat, chunk, dim=0):
        loglik = dpmm.sample_score(batch, weight_threshold=WEIGHT_THRESHOLD)
        out["loglik"].append(loglik)
        for name in scores:
            if name == "likelihood":
                out[name].append(-loglik)
            elif name == "nll_all":
                out[name].append(-dpmm.sample_score(batch))
            elif name == "euclidean":
                out[name].append(dpmm.distance_to_nearest_cluster(batch, WEIGHT_THRESHOLD))
            elif name == "cosine":
                out[name].append(dpmm.cosine_distance_to_nearest_cluster(batch, WEIGHT_THRESHOLD))
            else:
                raise ValueError(name)
    return {k: torch.cat(v) for k, v in out.items()}


def load_dpmm(state: dict, device) -> DPMM:
    dpmm = DPMM(K=state["mean"].shape[0], D=state["mean"].shape[1], cov_type="diag", device=str(device))
    dpmm.load_state_dict({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in state.items()})
    return dpmm


# ----------------------------------------------------------------------------
# pixel curves
# ----------------------------------------------------------------------------

class PixelCurve:
    """Per-class histogram of scores in [lo, hi] -> AUROC, AUPR (trapezoidal, as
    anomalib 0.6.0 `AUPR` = torchmetrics `auc(recall, precision)`), AP (sklearn's
    step-wise average_precision_score), plus exact Dice counts at fixed thresholds."""

    def __init__(self, lo: float, hi: float, device, thresholds: dict | None = None, n_bins: int = N_BINS):
        self.lo, self.hi, self.n_bins = float(lo), float(hi), n_bins
        self.counts = torch.zeros(2 * n_bins, dtype=torch.int64, device=device)
        self.thresholds = thresholds or {}
        self.dice_counts = {k: [0, 0, 0] for k in self.thresholds}  # tp, fp, fn

    def update(self, pred: torch.Tensor, label: torch.Tensor):
        pred, label = pred.reshape(-1), label.reshape(-1)
        scale = (self.n_bins - 1) / max(self.hi - self.lo, 1e-12)
        idx = ((pred - self.lo) * scale).floor().clamp_(0, self.n_bins - 1).long()
        idx += label.long() * self.n_bins
        self.counts += torch.bincount(idx, minlength=2 * self.n_bins)
        for key, t in self.thresholds.items():
            hit = pred > t
            c = self.dice_counts[key]
            c[0] += int((hit & label).sum())
            c[1] += int((hit & ~label).sum())
            c[2] += int((~hit & label).sum())

    def compute(self) -> dict:
        counts = self.counts.cpu().numpy().astype(np.float64)
        neg, pos = counts[:self.n_bins][::-1], counts[self.n_bins:][::-1]   # descending score
        keep = (neg + pos) > 0                                             # distinct thresholds only
        tps, fps = np.cumsum(pos)[keep], np.cumsum(neg)[keep]
        P, N = tps[-1], fps[-1]

        tpr = np.concatenate([[0.0], tps / P])
        fpr = np.concatenate([[0.0], fps / N])
        auroc = float(np.trapezoid(tpr, fpr))

        # torchmetrics 0.10.3 precision_recall_curve: stop at full recall, append (r=0, p=1)
        last = int(np.searchsorted(tps, P))
        precision = tps[:last + 1] / (tps[:last + 1] + fps[:last + 1])
        recall = tps[:last + 1] / P
        aupr = float(np.trapezoid(np.concatenate([[1.0], precision]), np.concatenate([[0.0], recall])))
        ap = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))

        out = {"auroc": auroc, "aupr": aupr, "ap": ap, "n_pos": int(P), "n_neg": int(N)}
        for key, (tp, fp, fn) in self.dice_counts.items():
            out[f"dice_{key}"] = 2 * tp / max(2 * tp + fp + fn, 1)
        return out


def _max_f1(labels, scores):
    prec, rec, _ = precision_recall_curve(labels, scores)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(np.nanmax(f1))


# ----------------------------------------------------------------------------
# protocols
# ----------------------------------------------------------------------------

def _loader(dataset, modality, batch_size):
    # OCT2017 images differ in size, so the aspect-preserving paper resize cannot be batched.
    bs = 1 if modality == "retina_oct" else batch_size
    return DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=4)


@torch.inference_mode()
def val_normal_paper(encoder, dpmms: dict, modality, device, scores=PAPER_SCORES, progress=True):
    """Normal val images under the paper protocol -> per checkpoint: mean
    log-likelihood (selection criterion) and Dice thresholds per score."""
    loader = _loader(PaperSplit(modality, "val", "normal"), modality, PAPER_BATCH_SIZE)
    loglik = {k: [] for k in dpmms}
    maps = {k: {s: [] for s in scores} for k in dpmms}
    for images, _ in tqdm(loader, disable=not progress, leave=False, desc=f"{modality} val normal"):
        flat = extract_paper_features(encoder, images.to(device)).reshape(-1, dpmms[next(iter(dpmms))].D)
        for key, dpmm in dpmms.items():
            sc = score_patches(dpmm, flat, scores)
            loglik[key].append(sc["loglik"].cpu())
            for s in scores:
                maps[key][s].append(sc[s].cpu())
    out = {}
    for key in dpmms:
        entry = {"val_normal_loglik": float(torch.cat(loglik[key]).double().mean())}
        thresholds = {}
        for s in scores:
            vals = torch.cat(maps[key][s])
            # test.py get_threshold: quantile of the normal val score maps (patch level)
            thresholds[s] = {fpr: float(torch.quantile(vals, 1 - fpr / 100)) for fpr in FPR_LEVELS}
        entry["dice_thresholds"] = thresholds
        out[key] = entry
    return out


@torch.inference_mode()
def test_pixel_paper(encoder, dpmms: dict, modality, device, thresholds: dict, scores=PAPER_SCORES,
                     label_rules=tuple(LABEL_RULES), progress=True, return_maps=False):
    """Pixel AUROC / AUPR / AP / Dice@FPR over all test pixels under the paper
    protocol -> out[checkpoint][label_rule][score]."""
    patch_maps = {k: {s: [] for s in scores} for k in dpmms}
    labels = []
    for kind in ("normal", "anomalous"):
        loader = _loader(PaperSplit(modality, "test", kind), modality, PAPER_BATCH_SIZE)
        for images, label in tqdm(loader, disable=not progress, leave=False, desc=f"{modality} test {kind}"):
            B = images.shape[0]
            flat = extract_paper_features(encoder, images.to(device)).reshape(-1, dpmms[next(iter(dpmms))].D)
            for key, dpmm in dpmms.items():
                sc = score_patches(dpmm, flat, scores)
                for s in scores:
                    patch_maps[key][s].append(sc[s].reshape(B, 1, FEATURE_RESOLUTION, FEATURE_RESOLUTION).cpu())
            labels.append(label)
    labels = torch.cat(labels)                   # [N, 1, H, W] uint8, original resolution
    target_shape = labels.shape[-2:]

    out = {}
    for key in dpmms:
        out[key] = {rule: {} for rule in label_rules}
        for s in scores:
            maps = torch.cat(patch_maps[key][s])
            curves = {rule: PixelCurve(maps.min(), maps.max(), device, thresholds=thresholds[key][s])
                      for rule in label_rules}
            for i in range(0, maps.shape[0], 64):
                # test.py calculate_metric: F.interpolate(pred, target_shape, mode="bilinear")
                pred = F.interpolate(maps[i:i + 64].to(device), target_shape, mode="bilinear")
                raw = labels[i:i + 64].to(device)
                for rule, curve in curves.items():
                    curve.update(pred, LABEL_RULES[rule](raw))
            for rule, curve in curves.items():
                out[key][rule][s] = curve.compute()
            if return_maps:
                out[key].setdefault("_maps", {})[s] = maps
    if return_maps:
        out["_labels"] = labels
    return out


@torch.inference_mode()
def test_ours(encoder, dpmms: dict, modality, device, scores=("nll_all", "cosine"), progress=True,
              max_ratio=0.01, resize_mask=256, batch_size=8):
    """Section 8's pipeline (src/eval/bmad_dpmm.py) with a choice of patch score.
    Returns per checkpoint/score: image-level metrics and, for masked modalities,
    pixel AUROC/AP/AUPR at 256x256."""
    _, _, test_data, has_masks = load_modality(modality, 448, 392)
    loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    patch_maps = {k: {s: [] for s in scores} for k in dpmms}
    gts, labels = [], []
    for img, gt, label, _ in tqdm(loader, disable=not progress, leave=False, desc=f"{modality} test (ours)"):
        img = img.to(device)
        features, grid = extract_patch_features(encoder, img, normalize=True)
        B = img.shape[0]
        flat = features.reshape(-1, features.shape[-1])
        for key, dpmm in dpmms.items():
            sc = score_patches(dpmm, flat, scores)
            for s in scores:
                patch_maps[key][s].append(sc[s].reshape(B, 1, *grid).cpu())
        if has_masks:
            gts.append(F.interpolate(gt, size=resize_mask, mode="nearest").bool()[:, 0])
        labels.append(label)
    labels = torch.cat(labels).numpy()
    gts = torch.cat(gts) if has_masks else None
    crop = test_data[0][0].shape[-1]

    out = {}
    for key in dpmms:
        out[key] = {}
        for s in scores:
            maps = torch.cat(patch_maps[key][s])
            curve = PixelCurve(maps.min(), maps.max(), device) if has_masks else None
            image_scores = []
            for i in range(0, maps.shape[0], 64):
                amap = F.interpolate(maps[i:i + 64].to(device), size=crop, mode="bilinear", align_corners=False)
                amap = F.interpolate(amap, size=resize_mask, mode="bilinear", align_corners=False)
                amap = gaussian_blur(amap, kernel_size=5, sigma=4.0)
                flat = amap.flatten(1)
                k = max(int(flat.shape[1] * max_ratio), 1)
                image_scores.append(torch.topk(flat, k=k, dim=1).values.mean(dim=1).cpu())
                if has_masks:
                    curve.update(amap[:, 0], gts[i:i + 64].to(device))
            image_scores = torch.cat(image_scores).numpy()
            entry = {
                "auroc_sp": float(roc_auc_score(labels, image_scores)),
                "ap_sp": float(average_precision_score(labels, image_scores)),
                "f1_sp": _max_f1(labels, image_scores),
            }
            if has_masks:
                px = curve.compute()
                entry.update({"auroc_px": px["auroc"], "ap_px": px["ap"], "aupr_px": px["aupr"]})
            out[key][s] = entry
    return out


@torch.inference_mode()
def test_ours_original_masks(encoder, dpmms: dict, modality, device, scores=("cosine",),
                             label_rules=tuple(LABEL_RULES), progress=True, batch_size=8):
    """Section 8's features/patch maps (448x448 resize, 392 center crop, 28x28),
    but scored like the official code: the map is bilinearly upsampled straight
    to the crop's footprint in the original-resolution mask (no 256 resize, no
    blur, no bilinear mask resize), and only that footprint is evaluated. Isolates
    mask handling from preprocessing without refitting."""
    _, _, test_data, _ = load_modality(modality, 448, 392)
    loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    patch_maps = {k: {s: [] for s in scores} for k in dpmms}
    for img, _, _, _ in tqdm(loader, disable=not progress, leave=False, desc=f"{modality} test (ours feats)"):
        features, grid = extract_patch_features(encoder, img.to(device), normalize=True)
        flat = features.reshape(-1, features.shape[-1])
        for key, dpmm in dpmms.items():
            sc = score_patches(dpmm, flat, scores)
            for s in scores:
                patch_maps[key][s].append(sc[s].reshape(img.shape[0], 1, *grid).cpu())

    labels = []
    for path, gt_path in zip(test_data.img_paths, test_data.gt_paths):
        size = Image.open(path).size[::-1]
        raw = np.array(Image.open(gt_path).convert("L"), dtype=np.uint8) if gt_path else np.zeros(size, np.uint8)
        labels.append(torch.from_numpy(raw)[None])
    labels = torch.stack(labels)                                   # [N, 1, H, W]
    H, W = labels.shape[-2:]
    # 392/448 center crop of a 448x448 resize covers the central 7/8 of each original axis
    top, left = round(H * (1 - 392 / 448) / 2), round(W * (1 - 392 / 448) / 2)
    labels = labels[..., top:H - top, left:W - left]
    footprint = labels.shape[-2:]

    out = {}
    for key in dpmms:
        out[key] = {rule: {} for rule in label_rules}
        for s in scores:
            maps = torch.cat(patch_maps[key][s])
            curves = {rule: PixelCurve(maps.min(), maps.max(), device) for rule in label_rules}
            for i in range(0, maps.shape[0], 64):
                pred = F.interpolate(maps[i:i + 64].to(device), footprint, mode="bilinear")
                raw = labels[i:i + 64].to(device)
                for rule, curve in curves.items():
                    curve.update(pred, LABEL_RULES[rule](raw))
            for rule, curve in curves.items():
                out[key][rule][s] = curve.compute()
    return out


def evaluate_ours_original_masks(modality: str, seeds=(1, 2, 3), device=None, progress=True, out_dir: Path = OUT_DIR):
    """Cached driver for test_ours_original_masks (section 8's I-AUROC checkpoint only)."""
    from src.training.train_bmad import pick_device
    device = device or pick_device()
    encoder = load_dpmm_encoder("dinov2_vits14", device=device)
    for seed in seeds:
        out_path = out_dir / f"{modality}_seed{seed}_ours_features_original_masks.json"
        if out_path.exists():
            continue
        cand = candidate_checkpoints(find_run(modality, seed))["iauroc_best"]
        res = test_ours_original_masks(encoder, {"iauroc_best": load_dpmm(cand["state"], device)}, modality,
                                       device, progress=progress)
        out_path.write_text(json.dumps({"modality": modality, "seed": seed, "epoch": cand["epoch"], **res}, indent=2))
        print(f"{modality} seed {seed} -> {out_path.name}")


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------

def find_run(modality: str, seed: int) -> Path:
    matches = sorted(glob.glob(str(REPO_ROOT / "runs" / f"bmad_dpmm_{modality}_seed{seed}_*")))
    if not matches:
        raise FileNotFoundError(f"no run for {modality} seed {seed}")
    return Path(matches[-1])


def candidate_checkpoints(run_dir: Path) -> dict:
    """The two checkpoints section 8's training saved: best_checkpoint.pt (picked
    by val I-AUROC) and checkpoint.pt (epoch 40). Deduplicated when they coincide."""
    out = {}
    for name, fname in (("iauroc_best", "best_checkpoint.pt"), ("last", "checkpoint.pt")):
        ck = torch.load(run_dir / fname, map_location="cpu", weights_only=False)
        if any(v["epoch"] == ck["epoch"] for v in out.values()):
            continue
        out[name] = {"epoch": ck["epoch"], "state": ck["dpmm"], "stored_test_metrics": ck["test_metrics"]}
    return out


def evaluate_modality(modality: str, seeds=(1, 2, 3), device=None, progress=True, out_dir: Path = OUT_DIR):
    from src.training.train_bmad import pick_device
    device = device or pick_device()
    out_dir.mkdir(parents=True, exist_ok=True)
    has_masks = MODALITIES[modality].has_masks
    encoder = load_dpmm_encoder("dinov2_vits14", device=device)

    for seed in seeds:
        out_path = out_dir / f"{modality}_seed{seed}.json"
        if out_path.exists():
            print(f"{out_path.name} exists, skipping")
            continue
        t0 = time.time()
        run_dir = find_run(modality, seed)
        cands = candidate_checkpoints(run_dir)
        dpmms = {k: load_dpmm(v["state"], device) for k, v in cands.items()}

        result = {
            "modality": modality, "seed": seed, "run_dir": str(run_dir.relative_to(REPO_ROOT)),
            "has_masks": has_masks, "checkpoints": {},
        }
        val = val_normal_paper(encoder, dpmms, modality, device, progress=progress)
        for key, cand in cands.items():
            n_active = int((dpmms[key].calculate_pi() > WEIGHT_THRESHOLD).sum())
            result["checkpoints"][key] = {
                "epoch": cand["epoch"], "n_active": n_active,
                "stored_test_metrics": cand["stored_test_metrics"], **val[key],
            }
        result["selected"] = max(val, key=lambda k: val[k]["val_normal_loglik"])

        if has_masks:
            thresholds = {k: val[k]["dice_thresholds"] for k in dpmms}
            paper = test_pixel_paper(encoder, dpmms, modality, device, thresholds, progress=progress)
            for key in dpmms:
                result["checkpoints"][key]["paper_pixel"] = paper[key]
        ours = test_ours(encoder, dpmms, modality, device, progress=progress)
        for key in dpmms:
            result["checkpoints"][key]["ours"] = ours[key]

        result["elapsed_min"] = (time.time() - t0) / 60
        out_path.write_text(json.dumps(result, indent=2))
        print(f"{modality} seed {seed}: selected {result['selected']} "
              f"(epoch {cands[result['selected']]['epoch']}) in {result['elapsed_min']:.1f} min -> {out_path.name}")
        del dpmms
        if device.type == "mps":
            torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", nargs="+", default=list(MODALITIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--ours-original-masks", action="store_true",
                        help="run evaluate_ours_original_masks instead of evaluate_modality")
    args = parser.parse_args()
    for modality in args.modality:
        if args.ours_original_masks:
            evaluate_ours_original_masks(modality, seeds=args.seeds)
        else:
            evaluate_modality(modality, seeds=args.seeds)


if __name__ == "__main__":
    sys.exit(main())
