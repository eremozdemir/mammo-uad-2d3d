"""
Dinomaly for CBIS-DDSM.

Frozen DINOv2 encoder, a small MLP bottleneck, and an 8-block linear-attention
decoder that reconstructs the encoder's mid-level features. At test time the
cosine distance between encoder and decoder features is the anomaly signal.

ViTill and the transformer Block/bMlp come straight from the vendored Dinomaly
repo; the decoder attention is our eps-guarded LinearAttention (see attention.py).
This module loads the backbone, assembles the pieces with the paper's MVTec
settings, and initialises the trainable parts.
"""

import types
from functools import partial

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_

from . import _ref  # noqa: F401  (registers third_party/Dinomaly on sys.path)
from models.uad import ViTill
from models.vision_transformer import Block, bMlp

from .attention import LinearAttention

# name -> (torch.hub entrypoint, embed_dim, num_heads)
BACKBONES = {
    "vit_small": ("dinov2_vits14_reg", 384, 6),
    "vit_base": ("dinov2_vitb14_reg", 768, 12),
    "vit_large": ("dinov2_vitl14_reg", 1024, 16),
}

# Dinomaly's MVTec multi-class config (dinomaly_mvtec_uni.py).
TARGET_LAYERS = [2, 3, 4, 5, 6, 7, 8, 9]
TARGET_LAYERS_LARGE = [4, 6, 8, 10, 12, 14, 16, 18]
FUSE_GROUPS = [[0, 1, 2, 3], [4, 5, 6, 7]]
N_DECODER_BLOCKS = 8
BOTTLENECK_DROP = 0.2


def load_encoder(backbone="vit_base"):
    """Frozen DINOv2 backbone with a `prepare_tokens` alias for ViTill."""
    hub_name = BACKBONES[backbone][0]
    encoder = torch.hub.load("facebookresearch/dinov2", hub_name, pretrained=True, verbose=False)

    # Match Dinomaly's vit_encoder.py rather than the newer torch.hub reg-model
    # defaults (offset 0.0, antialias True). These control how the pretrained
    # pos embedding is resized from the 37x37 training grid to our 28x28 grid;
    # both are read at forward time, so setting them after construction is fine.
    encoder.interpolate_offset = 0.1
    encoder.interpolate_antialias = False

    encoder.prepare_tokens = types.MethodType(
        lambda self, x: self.prepare_tokens_with_masks(x), encoder
    )
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    return encoder


def _init_trainable(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


def build_dinomaly(backbone="vit_base"):
    """
    Returns (model, trainable) where `trainable` is the ModuleList of parameters
    the optimizer should see (bottleneck + decoder; the encoder stays frozen).
    """
    _, embed_dim, num_heads = BACKBONES[backbone]
    target_layers = TARGET_LAYERS_LARGE if backbone == "vit_large" else TARGET_LAYERS

    encoder = load_encoder(backbone)

    bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=BOTTLENECK_DROP)])
    decoder = nn.ModuleList([
        Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-8),
            attn=LinearAttention,
        )
        for _ in range(N_DECODER_BLOCKS)
    ])

    trainable = nn.ModuleList([bottleneck, decoder])
    _init_trainable(trainable)

    model = ViTill(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        mask_neighbor_size=0,
        fuse_layer_encoder=FUSE_GROUPS,
        fuse_layer_decoder=FUSE_GROUPS,
    )
    return model, trainable
