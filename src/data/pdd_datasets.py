"""
Datasets for the PDD reproduction (Lu et al., CVPR 2026, Sec. 4.1), built
under data/PDD/ in the layouts the official scripts read:

  single-class (scripts/train_two_twins_ddp.py)
    <name>/train/<one folder>/*.png         read with torchvision ImageFolder
    <name>/test/<normal>/*.png              read with PDD's MVTecDataset:
    <name>/test/<abnormal>/*.png            "good" or "normal_256" -> label 0,
                                            every other folder -> label 1

  multi-class (scripts/train_two_twins_ddp_multi.py)
    uni_medical/meta.json                   {"train"|"test": {cls: [records]}}
    uni_medical/{brain,liver,retinal}/...   records hold img_path + anomaly

Sources and splits, matched against the counts in Sec. 4.1:

  HeadCT     60 train / 40 normal + 100 abnormal test.
  BrainMRI   58 train / 40 normal + 154 abnormal test.
      Both come from the MVTec-style "head_ct" / "brainMRI" folders of the
      Kaggle dataset arktis2022/medical-anomaly-detection (HeadCT: 100 normal
      + 100 hemorrhage, Kitamura 2018; BrainMRI: 98 normal + 154 tumour,
      Chakrabarty 2019). That pack ships 90/88 train normals and 10 test
      normals. The paper uses 60/58 train and 40 test normals, so 30 train
      normals are moved to test. Which 30 the authors moved is not
      published; here they are drawn with np.random.default_rng(0) from the
      sorted file list, so the split is fixed but not guaranteed to be the
      authors'.
  ZhangLab   1249 train / 234 normal + 390 pneumonia test.
  CheXpert   4499 train / 250 normal + 250 abnormal test.
      Both are SQUID's (Xiang et al., CVPR 2023) released splits, downloaded
      from the Google Drive links in github.com/tiangexiang/SQUID. The
      "normal_256" folder name in PDD's MVTecDataset comes from these zips.
      Their val splits are not used: the official scripts never read val.
  Uni-Medical  brain 7500/3715, liver 1542/1493, retinal 4297/1805.
      The BMAD (Bao et al., CVPR 2024) train and test splits of
      BraTS2021_slice, hist_DIY and RESC, already in data/BMAD/. Uni-Medical
      (Zhang et al., arXiv 2312.07495) is those three, so meta.json only
      points at the local BMAD files through relative symlinks.
"""

import json
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
PDD_DATA = DATA_DIR / "PDD"
DOWNLOADS = PDD_DATA / "_downloads"

SINGLE_CLASS = ["head_ct", "brain_mri", "zhanglab", "chexpert"]
UNI_MEDICAL_CLASSES = ["brain", "liver", "retinal"]

# Sec. 4.1 of the paper: (train normal, test normal, test abnormal)
PAPER_COUNTS = {
    "head_ct": (60, 40, 100),
    "brain_mri": (58, 40, 154),
    "zhanglab": (1249, 234, 390),
    "chexpert": (4499, 250, 250),
}
# (train, test) per Uni-Medical class
PAPER_COUNTS_UNI = {"brain": (7500, 3715), "liver": (1542, 1493), "retinal": (4297, 1805)}

KAGGLE_PACK = "arktis2022/medical-anomaly-detection"
SQUID_DRIVE_IDS = {
    "zhanglab": "1kgYtvVvyfPnQnrPhhLt50ZK9SnxJpriC",
    "chexpert": "14pEg9ch0fsice29O8HOjnyJ7Zg4GYNXM",
}
# inner zip -> (folder inside that zip, destination under data/PDD/<name>/)
SQUID_LAYOUT = {
    "zhanglab": [("train.zip", "train", "train"), ("test.zip", "test", "test")],
    "chexpert": [("train_256_pa.zip", "train_256_pa", "train"),
                 ("our_test_256_pa.zip", "our_test_256_pa", "test")],
}
UNI_MEDICAL_SOURCES = {
    # class -> (path under data/BMAD, abnormal folder name)
    "brain": ("Brain_AD/BraTS2021_slice", "Ungood"),
    "liver": ("Liver_AD/hist_DIY", "ungood"),
    "retinal": ("Retina_RESC_AD/RESC", "ungood"),
}
MOVED_TO_TEST = 30
SPLIT_SEED = 0
IMG_EXTENSIONS = (".png", ".jpg", ".jpeg")


def _images(folder):
    return sorted(p for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS and not p.name.startswith("."))


def _tidy(root):
    # MVTecDataset treats every entry of test/ as a class folder, so a loose
    # file there crashes it. Hidden files (.DS_Store) are deleted; anything
    # else (CheXpert's test/diseases.txt, the per-image finding list) is
    # moved up next to train/ and test/.
    root = Path(root)
    for p in root.rglob(".*"):
        if p.is_file():
            p.unlink()
    for split in ["train", "test"]:
        for p in (root / split).iterdir():
            if p.is_file():
                shutil.move(str(p), str(root / f"{split}_{p.name}"))


def _prepare_kaggle(name, folder):
    import kagglehub

    out = PDD_DATA / name
    if (out / "test").exists():
        return out
    src = Path(kagglehub.dataset_download(KAGGLE_PACK)) / folder

    train_good = _images(src / "train" / "good")
    rng = np.random.default_rng(SPLIT_SEED)
    moved = set(rng.choice(len(train_good), size=MOVED_TO_TEST, replace=False).tolist())

    for sub in ["train/good", "test/good", "test/bad"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(train_good):
        shutil.copy2(p, out / ("test/good" if i in moved else "train/good") / p.name)
    for p in _images(src / "test" / "good"):
        shutil.copy2(p, out / "test/good" / p.name)
    for p in _images(src / "test" / "bad"):
        shutil.copy2(p, out / "test/bad" / p.name)
    (out / "moved_to_test.txt").write_text("\n".join(sorted(train_good[i].name for i in moved)) + "\n")
    return out


def _prepare_squid(name):
    import gdown

    out = PDD_DATA / name
    if (out / "test").exists():
        return out
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    outer = DOWNLOADS / f"{name}.zip"
    if not outer.exists():
        gdown.download(id=SQUID_DRIVE_IDS[name], output=str(outer), quiet=False)

    staging = DOWNLOADS / name
    with zipfile.ZipFile(outer) as z:
        z.extractall(staging)
    for inner, folder, dest in SQUID_LAYOUT[name]:
        with zipfile.ZipFile(staging / name / inner) as z:
            z.extractall(staging / "unpacked")
        shutil.move(str(staging / "unpacked" / folder), str(out / dest))
    shutil.rmtree(staging)
    _tidy(out)
    return out


def _prepare_uni_medical():
    out = PDD_DATA / "uni_medical"
    meta_path = out / "meta.json"
    if meta_path.exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    bmad = DATA_DIR / "BMAD"
    if not bmad.exists():
        raise FileNotFoundError(f"{bmad} not found; Uni-Medical is built from the local BMAD copy.")

    meta = {"train": {}, "test": {}}
    for cls, (subpath, abnormal) in UNI_MEDICAL_SOURCES.items():
        link = out / cls
        if not link.exists():
            link.symlink_to(os.path.relpath(bmad / subpath, out))

        meta["train"][cls] = [
            {"img_path": f"{cls}/train/good/{p.name}", "mask_path": "", "cls_name": cls,
             "specie_name": "good", "anomaly": 0}
            for p in _images(link / "train" / "good")
        ]
        test = []
        for sub, anomaly in [("good", 0), (abnormal, 1)]:
            for p in _images(link / "test" / sub / "img"):
                test.append({
                    "img_path": f"{cls}/test/{sub}/img/{p.name}",
                    "mask_path": f"{cls}/test/{sub}/label/{p.name}" if anomaly else "",
                    "cls_name": cls, "specie_name": sub, "anomaly": anomaly,
                })
        meta["test"][cls] = test
    meta_path.write_text(json.dumps(meta, indent=1))
    return out


def prepare(name):
    """Builds data/PDD/<name>/ if it is not there yet and returns its path.
    name: head_ct, brain_mri, zhanglab, chexpert or uni_medical."""
    if name == "head_ct":
        return _prepare_kaggle(name, "head_ct")
    if name == "brain_mri":
        return _prepare_kaggle(name, "brainMRI")
    if name in SQUID_DRIVE_IDS:
        return _prepare_squid(name)
    if name == "uni_medical":
        return _prepare_uni_medical()
    raise ValueError(name)


def split_counts(name):
    """Counts read back from disk, in the same shape as PAPER_COUNTS /
    PAPER_COUNTS_UNI."""
    root = PDD_DATA / name
    if name == "uni_medical":
        meta = json.loads((root / "meta.json").read_text())
        return {cls: (len(meta["train"][cls]), len(meta["test"][cls])) for cls in UNI_MEDICAL_CLASSES}
    train = sum(len(_images(d)) for d in (root / "train").iterdir() if d.is_dir())
    normal = abnormal = 0
    for d in (root / "test").iterdir():
        n = len(_images(d))
        if d.name in ("good", "normal_256"):
            normal += n
        else:
            abnormal += n
    return train, normal, abnormal


def build_datasets(name, image_size=256):
    """(train_dataset, test_dataset, class_names) built with the official
    code's own classes and transform:

      single-class: ImageFolder(<root>/train) and MVTecDataset(<root>, "test"),
                    both yielding (img, label)
      uni_medical:  build_multi_class_dataset(meta.json, phase="train"/"test"),
                    yielding (img, label, cls_name)

    The transform is PDD's get_data_transforms(256, 256): Resize to 256x256,
    ToTensor, CenterCrop(256) (a no-op at this size), ImageNet Normalize.
    No augmentation.
    """
    from torchvision.datasets import ImageFolder

    from src.models.pdd import upstream

    up = upstream()
    root = PDD_DATA / name
    if name == "uni_medical":
        transform, _ = up.dataset_full.get_data_transforms(image_size, image_size)
        meta = str(root / "meta.json")
        train = up.dataset_full.build_multi_class_dataset(meta, str(root), transform, "train", UNI_MEDICAL_CLASSES)
        test = up.dataset_full.build_multi_class_dataset(meta, str(root), transform, "test", UNI_MEDICAL_CLASSES)
        return train, test, list(UNI_MEDICAL_CLASSES)

    transform, _ = up.dataset.get_data_transforms(image_size, image_size)
    train = ImageFolder(root=str(root / "train"), transform=transform)
    test = up.dataset.MVTecDataset(root=str(root), transform=transform, phase="test")
    return train, test, [name]
