"""
Image-level anomaly scoring and metrics.

Same recipe as Dinomaly's evaluation_batch (utils.py), trimmed to the
image-level case since CBIS-DDSM full mammograms have no per-pixel ground truth
for the normal class:

  1. cosine distance between encoder and decoder feature maps, per fused group
  2. upsample each to the input size, average the groups into one anomaly map
  3. light Gaussian blur
  4. image score = mean of the top `top_ratio` fraction of map values
  5. AUROC / average precision / best-threshold F1 over normal vs. anomaly
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from torchvision.transforms.functional import gaussian_blur

LABEL_TO_INT = {"normal": 0, "anomaly": 1}


def anomaly_map(en, de, out_size):
    """en, de: lists of [B, C, H, W] feature maps -> [B, 1, out_size, out_size]."""
    maps = []
    for fs, ft in zip(en, de):
        m = 1 - F.cosine_similarity(fs, ft)  # [B, H, W]
        m = F.interpolate(m.unsqueeze(1), size=out_size, mode="bilinear", align_corners=True)
        maps.append(m)
    return torch.cat(maps, dim=1).mean(dim=1, keepdim=True)


def image_scores(amap, top_ratio=0.01):
    """amap: [B, 1, H, W] -> [B] score = mean of the top `top_ratio` map values."""
    amap = gaussian_blur(amap, kernel_size=5, sigma=4.0)
    flat = amap.flatten(1)
    k = max(int(flat.shape[1] * top_ratio), 1)
    topk = torch.topk(flat, k=k, dim=1).values
    return topk.mean(dim=1)


def metrics_from_scores(scores, labels):
    """scores, labels: 1-D arrays -> dict of image-level metrics."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    prec, rec, _ = precision_recall_curve(labels, scores)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "f1_max": float(np.nanmax(f1)),
        "n_normal": int((labels == 0).sum()),
        "n_anomaly": int((labels == 1).sum()),
    }


@torch.no_grad()
def score_dataset(model, dataloader, device, top_ratio=0.01, return_maps=False):
    """
    Run the model over a loader and return per-image results:
        scores  [N]   image anomaly scores
        labels  [N]   0 normal / 1 anomaly
        maps    [N, h, w]  downsampled anomaly maps, only if return_maps
    """
    model.eval()
    all_scores, all_labels, all_maps = [], [], []
    for img, label in dataloader:
        img = img.to(device)
        en, de = model(img)
        amap = anomaly_map(en, de, img.shape[-1])
        all_scores.append(image_scores(amap, top_ratio).cpu())
        all_labels.extend(LABEL_TO_INT[l] for l in label)
        if return_maps:
            small = F.interpolate(amap, size=64, mode="bilinear", align_corners=False)
            all_maps.append(small[:, 0].cpu())

    scores = torch.cat(all_scores).numpy()
    labels = np.array(all_labels)
    if return_maps:
        return scores, labels, torch.cat(all_maps).numpy()
    return scores, labels


@torch.no_grad()
def evaluate(model, dataloader, device, top_ratio=0.01):
    scores, labels = score_dataset(model, dataloader, device, top_ratio)
    return metrics_from_scores(scores, labels)


@torch.no_grad()
def anomaly_map_for_images(model, imgs, device):
    """imgs: [B, 3, H, W] tensor -> [B, H, W] anomaly maps (numpy), for visualisation."""
    model.eval()
    imgs = imgs.to(device)
    en, de = model(imgs)
    amap = anomaly_map(en, de, imgs.shape[-1])
    amap = gaussian_blur(amap, kernel_size=5, sigma=4.0)
    return amap[:, 0].cpu().numpy()
