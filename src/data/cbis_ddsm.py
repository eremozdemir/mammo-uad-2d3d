"""
CBIS-DDSM data loading for the benign/normal-vs-malignant/anomaly framing.

Reads pathology labels from the mass/calc case description CSVs (same logic as
notebooks/breast_cancer_eda_local.ipynb section 3), links each case to its full
mammogram image via dicom_info.csv, and builds a patient-level train/test split
where the anomaly model only ever sees benign images during training.

The CBIS-DDSM copy this project uses (the awsaf49 Kaggle mirror) ships
pre-converted JPEGs, not raw .dcm files. dicom_info.csv maps the original
DICOM-style paths onto the actual .jpg files on disk, and this loader reads
the JPEGs directly. If dataset_dir ever points at a raw-DICOM source instead,
swap _load_image over to pydicom.
"""

import os
import random
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Match Dinomaly's own MVTec config: third_party/Dinomaly/dinomaly_mvtec_uni.py
IMAGE_SIZE = 448
CROP_SIZE = 392
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

KAGGLE_DATASET_SLUG = "awsaf49/cbis-ddsm-breast-cancer-image-dataset"

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


def resolve_dataset_dir(data_dir: Path = DATA_DIR) -> str:
    """
    Points at the local CBIS-DDSM copy. If data/archive isn't there yet, downloads
    it via kagglehub into data/ instead of kagglehub's default ~/.cache location,
    so the whole dataset cache stays inside this repo's (gitignored) data/ dir.
    """
    local_copy = data_dir / "archive"
    if local_copy.exists():
        return str(local_copy)

    import kagglehub

    os.environ.setdefault("KAGGLEHUB_CACHE", str(data_dir))
    return kagglehub.dataset_download(KAGGLE_DATASET_SLUG)


def load_case_metadata(dataset_dir: str) -> pd.DataFrame:
    """Same concat + pathology_binary logic as the EDA notebook, section 3."""
    csv_dir = f"{dataset_dir}/csv"
    mass_train = pd.read_csv(f"{csv_dir}/mass_case_description_train_set.csv")
    mass_test = pd.read_csv(f"{csv_dir}/mass_case_description_test_set.csv")
    calc_train = pd.read_csv(f"{csv_dir}/calc_case_description_train_set.csv")
    calc_test = pd.read_csv(f"{csv_dir}/calc_case_description_test_set.csv")

    cases = pd.concat(
        [
            mass_train.assign(abnormality="mass", split="train"),
            mass_test.assign(abnormality="mass", split="test"),
            calc_train.assign(abnormality="calcification", split="train"),
            calc_test.assign(abnormality="calcification", split="test"),
        ],
        ignore_index=True,
    )

    cases["pathology_binary"] = cases["pathology"].map(
        lambda p: "MALIGNANT" if p == "MALIGNANT" else "BENIGN"
    )
    return cases


def _series_uid_from_case_path(case_path: str) -> Optional[str]:
    """
    "image file path" values look like
    Mass-Training_P_00001_LEFT_CC/<StudyUID>/<SeriesUID>/000000.dcm
    dicom_info.csv keys the full-mammogram JPEGs by that SeriesUID (the third
    path segment), not the StudyUID.
    """
    parts = case_path.strip().split("/")
    return parts[2] if len(parts) > 2 else None


def build_image_index(dataset_dir: str, cases: pd.DataFrame) -> pd.DataFrame:
    """
    Links each case row to its on-disk full mammogram JPEG and collapses to one
    row per image. Around 15% of images carry more than one abnormality (e.g.
    two masses on the same mammogram); those collapse to a single row, labeled
    anomaly if any of their abnormalities is malignant.

    A small fraction of case rows (~8% here) don't resolve to a full-mammogram
    entry in dicom_info.csv, a known gap in this Kaggle mirror. Those are
    dropped, with a count printed rather than silently discarded.
    """
    dicom_info = pd.read_csv(f"{dataset_dir}/csv/dicom_info.csv")
    full_mammo = dicom_info[dicom_info["SeriesDescription"] == "full mammogram images"]
    series_to_path = dict(zip(full_mammo["SeriesInstanceUID"], full_mammo["image_path"]))

    cases = cases.copy()
    cases["series_uid"] = cases["image file path"].map(_series_uid_from_case_path)
    cases["image_path"] = cases["series_uid"].map(series_to_path)

    n_unresolved = int(cases["image_path"].isna().sum())
    if n_unresolved:
        print(
            f"build_image_index: dropping {n_unresolved} abnormality rows with "
            f"no matching full-mammogram entry in dicom_info.csv"
        )
    cases = cases.dropna(subset=["image_path"]).copy()
    cases["image_path"] = cases["image_path"].str.replace("CBIS-DDSM", dataset_dir, regex=False)

    is_malignant = cases.groupby("series_uid")["pathology_binary"].transform(
        lambda s: (s == "MALIGNANT").any()
    )
    cases["label"] = is_malignant.map({True: "anomaly", False: "normal"})

    images = (
        cases.drop_duplicates(subset="series_uid")[["series_uid", "patient_id", "image_path", "label"]]
        .reset_index(drop=True)
    )
    return images


def build_splits(images: pd.DataFrame, train_ratio: float = 0.85, seed: int = 42) -> dict:
    """
    Patient-level split: train_ratio of benign/normal patients' images go to
    training (what the anomaly model learns from); the rest become a held-out
    normal test set. Splitting by patient_id, not by image, keeps a patient's
    other breast/view from leaking across train and test. Every malignant
    image is kept out of training entirely and goes to the anomaly test set.
    """
    normal = images[images["label"] == "normal"]
    anomaly = images[images["label"] == "anomaly"]

    normal_patients = sorted(normal["patient_id"].unique())
    rng = random.Random(seed)
    rng.shuffle(normal_patients)

    n_train = int(len(normal_patients) * train_ratio)
    train_patients = set(normal_patients[:n_train])

    train_normal = normal[normal["patient_id"].isin(train_patients)].reset_index(drop=True)
    test_normal = normal[~normal["patient_id"].isin(train_patients)].reset_index(drop=True)
    test_anomaly = anomaly.reset_index(drop=True)

    return {
        "train_normal": train_normal,
        "test_normal": test_normal,
        "test_anomaly": test_anomaly,
    }


def get_transform(image_size: int = IMAGE_SIZE, crop_size: int = CROP_SIZE) -> transforms.Compose:
    """Resize -> tensor -> center crop -> ImageNet normalize, matching Dinomaly's get_data_transforms."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.CenterCrop(crop_size),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class CBISDDSMDataset(Dataset):
    """
    Returns (image, label) where label is the string 'normal' or 'anomaly'.
    Build one instance per split from build_splits()'s output.
    """

    def __init__(self, image_df: pd.DataFrame, transform: Optional[transforms.Compose] = None):
        self.image_paths = image_df["image_path"].tolist()
        self.labels = image_df["label"].tolist()
        self.transform = transform or get_transform()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        # Mammograms are grayscale; convert('RGB') replicates to 3 channels,
        # which is what the ImageNet-pretrained DINOv2 backbone expects.
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.labels[idx]


def load_datasets(
    data_dir: Path = DATA_DIR, train_ratio: float = 0.85, seed: int = 42
) -> dict:
    """Convenience entry point: dataset_dir -> {train_normal, test_normal, test_anomaly} datasets."""
    dataset_dir = resolve_dataset_dir(data_dir)
    cases = load_case_metadata(dataset_dir)
    images = build_image_index(dataset_dir, cases)
    splits = build_splits(images, train_ratio=train_ratio, seed=seed)
    return {name: CBISDDSMDataset(df) for name, df in splits.items()}
