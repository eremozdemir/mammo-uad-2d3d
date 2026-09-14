"""
MVTec-AD / VisA loading, for reproducing Dinomaly2's published numbers before
applying the method to CBIS-DDSM.

Both datasets ship in the same per-category train/test/ground_truth layout, and
Dinomaly2's own dataset.py already has the loader for it (MVTecDataset) plus the
resize/crop/normalize transform (get_data_transforms) -- reused directly here for
fidelity, via the same isolated-load pattern as _ref2.py (dataset.py is
self-contained aside from tifffile/natsort, both in requirements.txt).

Multi-class ("uni") setup: one ConcatDataset of every category's train/good
images to train on, one MVTecDataset per category (each returns (img, gt, label,
img_type)) to evaluate separately and average -- matching dinomaly_2D.py.
"""

import importlib.util
from pathlib import Path

from torchvision.datasets import ImageFolder
from torch.utils.data import ConcatDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DINOMALY2_DIR = REPO_ROOT / "third_party" / "Dinomaly2"

MVTEC_CLASSES = [
    "carpet", "grid", "leather", "tile", "wood", "bottle", "cable", "capsule",
    "hazelnut", "metal_nut", "pill", "screw", "toothbrush", "transistor", "zipper",
]
VISA_CLASSES = [
    "candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1", "macaroni2",
    "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]


def _load_dataset_module():
    spec = importlib.util.spec_from_file_location("_dinomaly2_dataset", DINOMALY2_DIR / "dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ds = _load_dataset_module()
get_data_transforms = _ds.get_data_transforms
MVTecDataset = _ds.MVTecDataset


def resolve_mvtec_dir(data_dir: Path = DATA_DIR) -> Path:
    p = data_dir / "mvtec_anomaly_detection"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Download MVTec-AD (e.g. via kagglehub 'ipythonx/mvtec-ad') "
            f"and symlink/place it there, one folder per category."
        )
    return p


def resolve_visa_dir(data_dir: Path = DATA_DIR) -> Path:
    p = data_dir / "VisA_pytorch" / "1cls"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Download VisA and run Dinomaly2's "
            f"prepare_data/prepare_visa.py --split-type 1cls to build this layout."
        )
    return p


def load_multiclass(root: Path, classes: list[str], image_size: int = 448, crop_size: int = 392):
    """
    Returns (train_data, test_sets) where train_data is a ConcatDataset of every
    class's train/good images and test_sets is {class_name: MVTecDataset}.
    """
    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_sets, test_sets = [], {}
    for i, cls in enumerate(classes):
        train_path = root / cls / "train"
        train_data = ImageFolder(root=str(train_path), transform=data_transform)
        train_data.classes = cls
        train_data.class_to_idx = {cls: i}
        train_data.samples = [(sample[0], i) for sample in train_data.samples]
        train_sets.append(train_data)

        test_sets[cls] = MVTecDataset(
            root=str(root / cls), transform=data_transform, gt_transform=gt_transform, phase="test"
        )

    return ConcatDataset(train_sets), test_sets
