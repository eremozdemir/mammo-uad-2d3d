"""
Dinomaly2 model builder, for reproducing the paper's MVTec-AD/VisA numbers.

Same frozen DINOv2 encoder as v1 (src/models/dinomaly.py::load_encoder). Two
things changed in Dinomaly2's own architecture, both reused verbatim from the
submodule via _ref2.py:

  - a two-stage "noisy bottleneck": compress to 256 (dropout), then expand
    back through embed_dim*4 to embed_dim (GELU, dropout at every stage),
    replacing v1's single bMlp.
  - context-aware recentering: the encoder-side reconstruction target has
    its own class token subtracted from every patch token, then LayerNorm,
    before the cosine loss compares it against the decoder's output. This is
    built into Dinomaly2's `Dinomaly.forward` and enabled by passing
    context_aware_recenter=True.

The decoder blocks (linear attention, eps-guarded) and the Dinomaly wrapper
class are Dinomaly2's own code, not v1's attention.py. Fidelity to the
original math matters more than module reuse for a reproduction.
"""

from functools import partial

import torch.nn as nn

from ._ref2 import Attention, Block, Dinomaly, LinearAttention2
from .dinomaly import BACKBONES, load_encoder

# dinomaly_2D.py's --lc 2 default (also v1's MVTec config): two fused groups.
FUSE_GROUPS = [[0, 1, 2, 3], [4, 5, 6, 7]]
N_DECODER_BLOCKS = 8


def build_dinomaly2(backbone="vit_base", dropout=0.4, linear_attention=True,
                     context_aware_recenter=True, fuse_groups=None, target_layers=None):
    """
    Returns (model, trainable, param_groups). `trainable` is the ModuleList of
    parameters the optimizer should see; `param_groups` is the list of dicts
    StableAdamW expects, with the bottleneck's first (compress) layer at its
    own lr and everything else at whatever base lr the caller sets on top.

    `target_layers` overrides the default per-backbone depth selection. The
    default ([2..9] of 12 blocks for vit_base) follows Dinomaly2's MVTec-AD
    config and favors mid-depth, semantically structured features, which
    suits anomalies defined by object or organ shape. For texture-dominated
    domains without a canonical spatial layout, shallower layers preserve
    more of the low-level statistics the anomaly signal depends on; see
    `notebooks/bmad/bmad_dinomaly2_train.ipynb` section 7.
    """
    _, embed_dim, num_heads = BACKBONES[backbone]
    default_layers = [4, 6, 8, 10, 12, 14, 16, 18] if backbone == "vit_large" else [2, 3, 4, 5, 6, 7, 8, 9]
    target_layers = target_layers or default_layers
    fuse_groups = fuse_groups or FUSE_GROUPS

    encoder = load_encoder(backbone)

    bottleneck = nn.ModuleList([
        nn.Sequential(nn.Linear(embed_dim, 256), nn.Dropout(p=dropout)),
        nn.Sequential(
            nn.Linear(256, embed_dim * 4), nn.GELU(), nn.Dropout(p=dropout),
            nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(p=dropout),
        ),
    ])

    attn = partial(LinearAttention2, eps=1e-8) if linear_attention else Attention
    decoder = nn.ModuleList([
        Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True,
              norm_layer=partial(nn.LayerNorm, eps=1e-8), attn=attn)
        for _ in range(N_DECODER_BLOCKS)
    ])

    model = Dinomaly(
        encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
        fuse_layer_encoder=fuse_groups, fuse_layer_decoder=fuse_groups,
        fuse_layer_bottleneck=list(range(len(target_layers))),
        context_aware_recenter=context_aware_recenter,
    )
    model.init_weights()

    param_groups = [
        {"params": bottleneck[0].parameters(), "lr": 2e-4},
        {"params": bottleneck[1].parameters()},
        {"params": decoder.parameters()},
    ]
    trainable = nn.ModuleList([bottleneck, decoder])
    return model, trainable, param_groups
