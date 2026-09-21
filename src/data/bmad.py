"""
BMAD (DorisBao/BMAD, "Benchmark for Medical Anomaly Detection") loading.

Six single-class modality datasets, each already split into train/good
(normal only), val/good+<abnormal>, and test/good+<abnormal> by the benchmark
itself. Unlike MVTec-AD/VisA, there is no category dimension inside a
modality, so each one gets its own model (single-class setting) rather than
a multi-class ConcatDataset; see src/training/train_bmad.py.

The benchmark's on-disk layout is inconsistent across modalities, since each
comes from a different upstream source dataset with its own preparation
script. This module hardcodes the per-modality quirks rather than inferring
them:

  - train/good is always a flat directory of images (ImageFolder-compatible:
    one "good" class folder under train/).
  - val/good, val/<abnormal>, test/good, and test/<abnormal> are all `img/`
    (plus `label/` when pixel masks exist) subdirectories, unlike train,
    which has no such nesting.
  - the top-level validation folder is named "valid" for Brain/Liver/
    Histopathology and "val" for Chest/RESC/OCT2017.
  - the abnormal split directory is capitalized inconsistently ("Ungood" for
    Brain/Chest/Histopathology/OCT2017, "ungood" for Liver/RESC).
  - only Brain, Liver, and RESC ship pixel-level masks; Chest, Histopathology,
    and OCT2017 are image-level-only benchmarks.

Reuses Dinomaly2's own get_data_transforms (via src.data.mvtec_visa, which
already isolates the import) and its MVTecDataset.__getitem__ convention
(img, gt, label, path; gt is an all-zero mask when label==0 or no mask
exists), so src/eval/pixel.py and src/eval/bmad.py work unmodified.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from src.data.mvtec_visa import get_data_transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


@dataclass(frozen=True)
class ModalityConfig:
    subpath: str        # relative to data/BMAD/
    abnormal_dir: str    # name of the abnormal split folder under val/ and test/
    has_masks: bool      # whether <split>/<abnormal>/label/ ships pixel masks
    val_dir: str         # name of the top-level validation folder ("val" | "valid")


MODALITIES: dict[str, ModalityConfig] = {
    "brain": ModalityConfig("Brain_AD/BraTS2021_slice", "Ungood", True, "valid"),
    "liver": ModalityConfig("Liver_AD/hist_DIY", "ungood", True, "valid"),
    "retina_resc": ModalityConfig("Retina_RESC_AD/RESC", "ungood", True, "val"),
    "chest": ModalityConfig("Chest_AD/Chest-RSNA", "Ungood", False, "val"),
    "histopathology": ModalityConfig("Histopathology_AD/camelyon16_256", "Ungood", False, "valid"),
    "retina_oct": ModalityConfig("Retina_OCT2017_AD/OCT2017", "Ungood", False, "val"),
}


def resolve_bmad_root(data_dir: Path = DATA_DIR) -> Path:
    p = data_dir / "BMAD"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Download BMAD (https://github.com/DorisBao/BMAD) and "
            f"place/symlink it there."
        )
    return p


class BMADSplitDataset(Dataset):
    """One modality's val or test split: <split_dir>/good + <split_dir>/<abnormal>,
    each holding an img/ subdirectory (and a label/ one when cfg.has_masks)."""

    def __init__(self, root: Path, cfg: ModalityConfig, split_dir: str, transform, gt_transform):
        self.transform = transform
        self.gt_transform = gt_transform
        self.has_masks = cfg.has_masks

        img_paths, gt_paths, labels = [], [], []
        for sub, label in [("good", 0), (cfg.abnormal_dir, 1)]:
            paths = sorted((root / split_dir / sub / "img").glob("*.png"))
            img_paths.extend(paths)
            labels.extend([label] * len(paths))
            if cfg.has_masks and label == 1:
                label_dir = root / split_dir / sub / "label"
                gt_paths.extend([label_dir / p.name for p in paths])
            else:
                gt_paths.extend([None] * len(paths))

        self.img_paths = img_paths
        self.gt_paths = gt_paths
        self.labels = labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        img = self.transform(img)
        label = self.labels[idx]

        gt_path = self.gt_paths[idx]
        if gt_path is None:
            gt = torch.zeros([1, img.shape[-2], img.shape[-1]])
        else:
            gt = Image.open(gt_path).convert("L")
            gt = self.gt_transform(gt)

        return img, gt, label, str(self.img_paths[idx])


def load_modality(name: str, image_size: int = 448, crop_size: int = 392, data_dir: Path = DATA_DIR):
    """Returns (train_data, val_data, test_data, has_masks). train_data is an
    ImageFolder over train/good; val_data/test_data are BMADSplitDataset."""
    cfg = MODALITIES[name]
    root = resolve_bmad_root(data_dir) / cfg.subpath
    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_data = ImageFolder(root=str(root / "train"), transform=data_transform)
    val_data = BMADSplitDataset(root, cfg, cfg.val_dir, data_transform, gt_transform)
    test_data = BMADSplitDataset(root, cfg, "test", data_transform, gt_transform)
    return train_data, val_data, test_data, cfg.has_masks
