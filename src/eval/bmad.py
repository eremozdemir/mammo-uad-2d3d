"""
BMAD evaluation. Every modality has image-level good/abnormal labels, so
image-level metrics (I-AUROC, I-AP, I-F1) are always computed. Pixel-level
metrics (P-AUROC, P-AP, P-F1, P-AUPRO) only make sense for the three
modalities that ship real per-pixel masks (Brain, Liver, RESC). For the
other three (Chest, Histopathology, OCT2017), every ground-truth mask is the
all-zero placeholder from src/data/bmad.py, so sklearn's roc_auc_score would
have only one class to score against and would raise.

For has_masks modalities this is exactly src/eval/pixel.py::evaluate_category
(same fused-map extraction, same AUPRO); this module adds the guard and an
image-only fallback built from src/eval/anomaly.py's primitives.
"""

import torch

from .anomaly import anomaly_map, image_scores, metrics_from_scores
from .pixel import evaluate_category


@torch.no_grad()
def _evaluate_image_only(model, dataloader, device, top_ratio=0.01):
    """dataloader yields (img, gt, label, path); gt is unused here.

    Keys are renamed to match evaluate_category's `_sp` (image/"specimen"-
    level) suffix convention, so callers can read `metrics['auroc_sp']`
    regardless of whether pixel masks were available.
    """
    model.eval()
    scores, labels = [], []
    for img, _gt, label, _ in dataloader:
        img = img.to(device)
        en, de = model(img)
        amap = anomaly_map(en, de, img.shape[-1])
        scores.append(image_scores(amap, top_ratio).cpu())
        labels.append(label)

    scores = torch.cat(scores).numpy()
    labels = torch.cat(labels).numpy()
    m = metrics_from_scores(scores, labels)
    return {"auroc_sp": m["auroc"], "ap_sp": m["ap"], "f1_sp": m["f1_max"]}


@torch.no_grad()
def evaluate_modality(model, dataloader, device, has_masks, max_ratio=0.01, resize_mask=256):
    if has_masks:
        return evaluate_category(model, dataloader, device, max_ratio=max_ratio, resize_mask=resize_mask)
    return _evaluate_image_only(model, dataloader, device, top_ratio=max_ratio)
