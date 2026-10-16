"""
PDD scoring, as written in the official code.

Anomaly map (Sec. 4.4, "cos(t1, s1) + cos(t2, s2)"): for each of the three
decoder levels, 1 - cosine similarity between Student 2 and the InA-fused
teacher features f_b, plus the same for Student 1, each map bilinearly
upsampled to 256x256 (align_corners=True) and summed (6 maps in total).
The image score is the max of that map after a Gaussian filter with sigma=4.
That is `cal_stu_ano_map(...) x 2 -> gaussian_filter(sigma=4) -> max` in
utils/evaluation.py and `batch_anomaly_scores` in the multi-class script.

Metrics are image-level AUROC, AP and F1-max (`compute_f1max` from the
official utils/evaluation.py, i.e. the max of 2PR/(P+R) over the PR curve).

Two scoring paths exist upstream, and they disagree for single-class runs:

* Multi-class script (Uni-Medical): every image gets its own score. This is
  `per_image` below.

* Single-class script (HeadCT, BrainMRI, ZhangLab, CheXpert):
  utils/evaluation.py::evaluation runs the test loader with batch_size=10
  but `cal_stu_ano_map` keeps only `a_map[0, 0]`, the first image of each
  batch, and the label is `np.max` over the batch. So each batch of 10
  becomes one sample: the first image's score against "does the batch hold
  any abnormal image". HeadCT's 140 test images become 14 samples. This is
  `official_single_class` below; it needs the per-image scores in the
  loader's order (PDD's MVTecDataset order, shuffle=False).

Both are reported for the single-class datasets: per-image is the
evaluation the paper describes in Sec. 4.1, the batch version is what the
released script prints.

The student-student map "cos(s1, s2)" of Table 4 (1 - cos between Student 1
and Student 2, summed over the three levels) is computed alongside.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score, roc_auc_score


def _compute_f1max():
    from src.models.pdd import upstream
    return upstream().evaluation.compute_f1max


def _sum_maps(pairs, out_size, accumulate):
    """Sum of upsampled (1 - cos) maps over the given (a, b) feature pairs,
    as numpy (batch, out_size, out_size). `accumulate` is the dtype of the
    running sum: float64 in the single-class code (np.zeros default),
    float32 in the multi-class code."""
    total = None
    for a, b in pairs:
        m = (1 - F.cosine_similarity(a, b)).unsqueeze(1)
        m = F.interpolate(m, size=out_size, mode="bilinear", align_corners=True)
        m = m[:, 0].detach().cpu().numpy()
        total = np.zeros(m.shape, dtype=accumulate) if total is None else total
        total += m
    return total


def batch_scores(outputs, out_size=256, accumulate=np.float32):
    """Image scores for one batch of decoder outputs.

    Returns {"ts": teacher-student scores (the PDD score),
             "ss": student-student scores (Table 4's cos(s1, s2))}.
    """
    s2, fb, s1 = outputs[0:3], outputs[3:6], outputs[6:9]
    ts = (_sum_maps(zip(s2, fb), out_size, accumulate)
          + _sum_maps(zip(s1, fb), out_size, accumulate))
    ss = _sum_maps(zip(s2, s1), out_size, accumulate)
    return {
        "ts": np.array([gaussian_filter(m, sigma=4).max() for m in ts]),
        "ss": np.array([gaussian_filter(m, sigma=4).max() for m in ss]),
    }


def metrics(labels, scores):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "f1": float(_compute_f1max()(labels, scores)),
    }


def official_single_class(labels, scores, batch_size=10):
    """The single-class script's evaluation: one sample per test batch, the
    first image's score against the max label of the batch. `labels` and
    `scores` must be in the test loader's order."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    starts = range(0, len(labels), batch_size)
    batch_labels = [labels[i:i + batch_size].max() for i in starts]
    batch_scores_ = [scores[i] for i in starts]
    out = metrics(batch_labels, batch_scores_)
    out["n_samples"] = len(batch_labels)
    out["n_abnormal"] = int(np.sum(batch_labels))
    return out


def summarize(labels, scores_ts, scores_ss, class_names, single_class, val_batch_size=10):
    """All metrics for one evaluation. Per class for Uni-Medical (with the
    mean over classes, Table 2's "Mean" row, and the pooled "global" numbers
    the multi-class script prints); per image, plus the official batch
    version, for single-class datasets."""
    labels = np.asarray(labels)
    class_names = np.asarray(class_names)
    out = {"per_class": {}, "per_class_ss": {}}
    for cls in dict.fromkeys(class_names.tolist()):
        idx = class_names == cls
        out["per_class"][cls] = metrics(labels[idx], scores_ts[idx])
        out["per_class_ss"][cls] = metrics(labels[idx], scores_ss[idx])
    out["mean"] = {k: float(np.mean([m[k] for m in out["per_class"].values()])) for k in ("auroc", "ap", "f1")}
    out["global"] = metrics(labels, scores_ts)
    if single_class:
        out["official_batch"] = official_single_class(labels, scores_ts, val_batch_size)
        out["official_batch_ss"] = official_single_class(labels, scores_ss, val_batch_size)
    return out
