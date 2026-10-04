"""
Image + pixel level evaluation for the AnomalyDINO-DPMM method (Schulthess &
Konukoglu, MICCAI 2025; see src/models/dpmm.py for the citation).

Same map -> score pipeline and metric set as src/eval/pixel.py /
src/eval/bmad.py (I-AUROC, I-AP, I-F1, and P-AUROC/P-AP/P-F1/P-AUPRO when
pixel masks exist), but the anomaly map comes from the DPMM's per-patch
negative log-likelihood under the fitted mixture rather than a Dinomaly
encoder/decoder cosine-distance map. Reuses compute_pro (AUPRO) from
src/eval/pixel.py unmodified.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torchvision.transforms.functional import gaussian_blur

from src.models.anomaly_dpmm import extract_patch_features

from .pixel import compute_pro


@torch.inference_mode()
def dpmm_anomaly_map(encoder, dpmm, images: torch.Tensor, normalize: bool, out_size: int):
    """images: [B, 3, H, W] -> [B, 1, out_size, out_size] negative-log-likelihood map."""
    features, grid_size = extract_patch_features(encoder, images, normalize=normalize)
    B, N, D = features.shape
    flat = features.reshape(B * N, D)

    nll = -dpmm.sample_score(flat)  # [B*N], higher = more anomalous
    amap = nll.reshape(B, 1, grid_size[0], grid_size[1])
    amap = F.interpolate(amap, size=out_size, mode="bilinear", align_corners=False)
    return amap


@torch.inference_mode()
def evaluate_dpmm_modality(
    encoder,
    dpmm,
    dataloader,
    device,
    has_masks: bool,
    normalize: bool = True,
    max_ratio: float = 0.01,
    resize_mask: int = 256,
):
    """dataloader yields (img, gt, label, path) -- src.data.bmad.BMADSplitDataset format.

    Pixel-level maps are only accumulated when `has_masks` (three of BMAD's six
    modalities have no real pixel masks -- see src/data/bmad.py -- and one of the
    other three, Chest, has 17k test images, so skipping this for the unmasked
    modalities avoids holding tens of thousands of full-resolution maps in memory
    for a metric that would raise on an all-zero mask anyway).
    """
    gt_px, pr_px, gt_sp, pr_sp = [], [], [], []

    for img, gt, label, _ in dataloader:
        img = img.to(device)
        amap = dpmm_anomaly_map(encoder, dpmm, img, normalize, out_size=img.shape[-1])

        if resize_mask is not None:
            amap = F.interpolate(amap, size=resize_mask, mode="bilinear", align_corners=False)
        amap = gaussian_blur(amap, kernel_size=5, sigma=4.0)

        if has_masks:
            if resize_mask is not None:
                gt = F.interpolate(gt, size=resize_mask, mode="nearest")
            gt_px.append(gt.bool().cpu())
            pr_px.append(amap.cpu())
        gt_sp.append(label)

        flat = amap.flatten(1)
        if max_ratio == 0:
            sp = flat.max(dim=1).values
        else:
            k = max(int(flat.shape[1] * max_ratio), 1)
            sp = torch.topk(flat, k=k, dim=1).values.mean(dim=1)
        pr_sp.append(sp.cpu())

    gt_sp = torch.cat(gt_sp).flatten().numpy()
    pr_sp = torch.cat(pr_sp).flatten().numpy()

    metrics = {
        "auroc_sp": float(roc_auc_score(gt_sp, pr_sp)),
        "ap_sp": float(average_precision_score(gt_sp, pr_sp)),
        "f1_sp": float(_max_f1(gt_sp, pr_sp)),
    }

    if has_masks:
        gt_px = torch.cat(gt_px, dim=0)[:, 0].numpy()
        pr_px = torch.cat(pr_px, dim=0)[:, 0].numpy()
        gt_px_flat, pr_px_flat = gt_px.ravel(), pr_px.ravel()

        metrics["auroc_px"] = float(roc_auc_score(gt_px_flat, pr_px_flat))
        metrics["ap_px"] = float(average_precision_score(gt_px_flat, pr_px_flat))
        metrics["f1_px"] = float(_max_f1(gt_px_flat, pr_px_flat))
        metrics["aupro_px"] = compute_pro(gt_px.astype(int), pr_px)

    return metrics


def _max_f1(labels, scores):
    prec, rec, _ = precision_recall_curve(labels, scores)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return np.nanmax(f1)
