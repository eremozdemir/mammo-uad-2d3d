"""
Linear attention for the Dinomaly decoder.

This is the reference LinearAttention2 (Dinomaly's models/vision_transformer.py)
with one change: the `1 / z` normaliser gets an eps guard. The Dinomaly README
recommends exactly this for datasets outside the standard industrial benchmarks,
where the un-guarded reciprocal can divide by ~0 and send the loss to NaN.
Grayscale mammograms fed through an RGB-pretrained backbone are that kind of
out-of-distribution input, so we use the guarded version from the start.

ELU+1 feature map keeps q, k positive; attention is computed as
(q phi) (k phi)^T v in linear time. Per the paper, this is deliberately
"unable to focus", and that softness is what makes reconstruction loose.
"""

import torch
import torch.nn as nn


class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.eps = eps
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = nn.functional.elu(q) + 1.0
        k = nn.functional.elu(k) + 1.0

        kv = torch.einsum("...sd,...se->...de", k, v)
        z = 1.0 / (torch.einsum("...sd,...d->...s", q, k.sum(dim=-2)) + self.eps)
        x = torch.einsum("...de,...sd,...s->...se", kv, q, z)
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, kv
