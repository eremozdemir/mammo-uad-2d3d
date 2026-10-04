"""
Frozen DINOv2 patch-embedding extractor for the AnomalyDINO-DPMM method
(Schulthess & Konukoglu, MICCAI 2025 -- see src/models/dpmm.py for the full
citation). Unlike src/models/dinomaly.py's encoder (wrapped for Dinomaly's
trained bottleneck/decoder), nothing here is trained: the backbone is used
as-is to produce per-patch tokens that src/models/dpmm.py clusters directly.

Uses the plain (no-register) `dinov2_vits14` backbone and single-layer
`get_intermediate_layers` output, matching the method's own reference
configs (anomalydino-dpmm/config/*.yaml all use dinov2_vits14). Images are
expected already resized/cropped/imagenet-normalized by
src.data.bmad.load_modality (crop_size=392 is exactly 28x14, an integer
number of dinov2's 14px patches).
"""

import torch
import torch.nn.functional as F

PATCH_SIZE = 14


def load_dpmm_encoder(model_name: str = "dinov2_vits14", device=None):
    """Frozen DINOv2 backbone (no registers, matching the paper's own configs)."""
    encoder = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True, verbose=False)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad_(False)
    if device is not None:
        encoder = encoder.to(device)
    return encoder


def embedding_dim(encoder) -> int:
    return encoder.norm.normalized_shape[0]


@torch.inference_mode()
def extract_patch_features(encoder, images: torch.Tensor, normalize: bool = True):
    """
    images: [B, 3, H, W], H and W multiples of PATCH_SIZE.
    Returns (features [B, N, D], grid_size (H//14, W//14)).
    """
    _, _, height, width = images.shape
    assert height % PATCH_SIZE == 0 and width % PATCH_SIZE == 0, (
        f"image size {(height, width)} must be a multiple of patch size {PATCH_SIZE}"
    )
    grid_size = (height // PATCH_SIZE, width // PATCH_SIZE)

    tokens = encoder.get_intermediate_layers(images)[0]  # [B, N, D], last block, no class token
    if normalize:
        tokens = F.normalize(tokens, dim=-1)
    return tokens, grid_size
