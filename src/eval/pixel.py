"""
Image + pixel level evaluation for MVTec-AD / VisA, matching Dinomaly2's own
evaluation_batch (utils.py) closely enough to compare against the paper's
table columns: I-AUROC, I-AP, I-F1, P-AUROC, P-AP, P-F1, P-AUPRO.

Not imported from the reference utils.py itself -- that module does
`from adeval import EvalAccumulatorCuda`, which fails to import on any
non-CUDA machine (see _ref2.py). AUPRO (compute_pro below) is a direct port of
theirs (pure numpy/pandas/skimage, no CUDA involved); everything else is our
own eval/anomaly.py plumbing applied per-category.
"""

from statistics import mean

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skimage import measure
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from torchvision.transforms.functional import gaussian_blur

from .anomaly import anomaly_map


def compute_pro(masks: np.ndarray, amaps: np.ndarray, num_th: int = 200) -> float:
    """Area under the per-region-overlap curve, FPR in [0, 0.3]. Ported from
    Dinomaly2's utils.py::compute_pro -- pure numpy/pandas/skimage, no CUDA."""
    assert amaps.shape == masks.shape
    assert set(np.unique(masks).tolist()) <= {0, 1}

    binary_amaps = np.zeros_like(amaps)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / num_th

    rows = []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                y, x = region.coords[:, 0], region.coords[:, 1]
                pros.append(binary_amap[y, x].sum() / region.area)

        inverse_masks = 1 - masks
        fpr = np.logical_and(inverse_masks, binary_amaps).sum() / inverse_masks.sum()
        rows.append({"pro": mean(pros) if pros else 0.0, "fpr": fpr})

    df = pd.DataFrame(rows)
    df = df[df["fpr"] < 0.3]
    if df["fpr"].max() == 0:
        return 0.0
    df["fpr"] = df["fpr"] / df["fpr"].max()
    return float(auc(df["fpr"], df["pro"]))


@torch.no_grad()
def evaluate_category(model, dataloader, device, max_ratio=0.01, resize_mask=256):
    """dataloader yields (img, gt, label, img_type) -- Dinomaly2's MVTecDataset format."""
    model.eval()
    gt_px, pr_px, gt_sp, pr_sp = [], [], [], []

    for img, gt, label, _ in dataloader:
        img = img.to(device)
        en, de = model(img)
        amap = anomaly_map(en, de, img.shape[-1])

        if resize_mask is not None:
            amap = F.interpolate(amap, size=resize_mask, mode="bilinear", align_corners=False)
            gt = F.interpolate(gt, size=resize_mask, mode="nearest")
        amap = gaussian_blur(amap, kernel_size=5, sigma=4.0)

        gt = gt.bool()
        gt_px.append(gt.cpu())
        pr_px.append(amap.cpu())
        gt_sp.append(label)

        flat = amap.flatten(1)
        if max_ratio == 0:
            sp = flat.max(dim=1).values
        else:
            k = max(int(flat.shape[1] * max_ratio), 1)
            sp = torch.topk(flat, k=k, dim=1).values.mean(dim=1)
        pr_sp.append(sp.cpu())

    gt_px = torch.cat(gt_px, dim=0)[:, 0].numpy()
    pr_px = torch.cat(pr_px, dim=0)[:, 0].numpy()
    gt_sp = torch.cat(gt_sp).flatten().numpy()
    pr_sp = torch.cat(pr_sp).flatten().numpy()

    aupro_px = compute_pro(gt_px.astype(int), pr_px)

    gt_px_flat, pr_px_flat = gt_px.ravel(), pr_px.ravel()
    prec_px, rec_px, _ = precision_recall_curve(gt_px_flat, pr_px_flat)
    f1_px_curve = 2 * prec_px * rec_px / np.clip(prec_px + rec_px, 1e-12, None)
    prec_sp, rec_sp, _ = precision_recall_curve(gt_sp, pr_sp)
    f1_sp_curve = 2 * prec_sp * rec_sp / np.clip(prec_sp + rec_sp, 1e-12, None)

    return {
        "auroc_sp": float(roc_auc_score(gt_sp, pr_sp)),
        "ap_sp": float(average_precision_score(gt_sp, pr_sp)),
        "f1_sp": float(np.nanmax(f1_sp_curve)),
        "auroc_px": float(roc_auc_score(gt_px_flat, pr_px_flat)),
        "ap_px": float(average_precision_score(gt_px_flat, pr_px_flat)),
        "f1_px": float(np.nanmax(f1_px_curve)),
        "aupro_px": aupro_px,
    }
