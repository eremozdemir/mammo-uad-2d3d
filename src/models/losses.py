"""
Feature-reconstruction losses from Dinomaly (utils.py in the reference repo).

Both compare encoder features `a` against decoder features `b` with cosine
distance. The encoder side is detached, so only the decoder learns to
reconstruct. `global_cosine_hm_percent` adds the hard-mining trick from the
paper: after computing the loss it registers a backward hook that scales down
the gradient from the easiest (1 - p) fraction of spatial positions, so the
decoder isn't dominated by the bulk of already-well-reconstructed patches.
"""

from functools import partial

import torch
import torch.nn.functional as F


def _scale_easy_grads(grad, keep_mask, factor):
    """Backward hook: multiply gradient at easy positions by `factor` (0 = drop)."""
    grad = grad.clone()
    grad[keep_mask.expand_as(grad)] *= factor
    return grad


def global_cosine(en, de):
    """Mean cosine distance between (detached) encoder and decoder feature maps."""
    loss = 0.0
    for a, b in zip(en, de):
        a = a.reshape(a.shape[0], -1).detach()
        b = b.reshape(b.shape[0], -1)
        loss = loss + torch.mean(1 - F.cosine_similarity(a, b))
    return loss / len(en)


def global_cosine_hm_percent(en, de, p=0.9, factor=0.1):
    """
    `global_cosine` plus per-position hard mining. `p` is the fraction of
    positions treated as "easy" and down-weighted in the backward pass; the
    reference ramps it from 0 to 0.9 over the first 1000 iterations.
    """
    loss = 0.0
    for a, b in zip(en, de):
        a = a.detach()
        with torch.no_grad():
            point_dist = (1 - F.cosine_similarity(a, b)).unsqueeze(1)
            k = max(int(point_dist.numel() * (1 - p)), 1)
            thresh = torch.topk(point_dist.reshape(-1), k=k)[0][-1]

        loss = loss + torch.mean(
            1 - F.cosine_similarity(a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1))
        )
        b.register_hook(partial(_scale_easy_grads, keep_mask=point_dist < thresh, factor=factor))
    return loss / len(en)
