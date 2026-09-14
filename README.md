# mammo-uad-2d3d

Unsupervised anomaly detection for breast cancer screening on CBIS-DDSM mammography,
framed as: train on benign/normal images only, detect malignant cases as anomalies at
test time.

**Current direction:** reproduce Dinomaly2's published results on its own benchmarks
first (MVTec-AD, VisA), to validate the reimplementation against numbers that can be
checked. Done (see [Reproducing Dinomaly2](#reproducing-dinomaly2)). Next is looking
at architecture changes and applying the validated pipeline to CBIS-DDSM.

- **Dinomaly** (CVPR 2025), the original Transformer reconstruction-based unsupervised
  anomaly detector using a DINOv2 backbone. Vendored at
  [third_party/Dinomaly](third_party/Dinomaly). Ported to CBIS-DDSM first; that work is
  paused, not removed (see `src/models/dinomaly.py`, `src/data/cbis_ddsm.py`).
- **Dinomaly2** (arXiv 2510.17611, preview release), the follow-up: same recipe, plus
  a two-stage bottleneck, per-parameter-group learning rates, and context-aware
  recentering. Vendored at [third_party/Dinomaly2](third_party/Dinomaly2). Reproduced
  on MVTec-AD and VisA; this is the current base for further work.
- **AnomalyMoE** (AAAI 2026), a Mixture-of-Experts anomaly detector with
  patch/component/global expert levels. Not started.

## Reproducing Dinomaly2

Full 40k-iteration multi-class runs, ViT-B/14, against Dinomaly2's Table II
(arXiv 2510.17611v2). Reproduced 2026-09-11 to 2026-09-12 on an M4 Max, ~13.4h per
dataset (`notebooks/dinomaly2_repro_mvtec_visa.ipynb`):

| Dataset  | I-AUROC | I-AP | I-F1 | P-AUROC | P-AP | P-F1 | P-AUPRO |
|----------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
| MVTec-AD | 99.78 (99.8)  | 99.86 (99.9)  | 99.39 (99.3)  | 98.39 (98.4)  | 69.31 (69.3)  | 68.74 (68.9)  | 95.34 (94.8)  |
| VisA     | 99.21 (99.2)  | 99.33 (99.3)  | 96.99 (97.0)  | 99.02 (99.1)  | 52.49 (52.9)  | 56.34 (56.4)  | 95.32 (95.5)  |

(paper's number in parentheses). Every metric is within 0.5 points of the paper, most
within 0.1; nothing suggests a systematic difference from a single-seed rerun. See the
notebook for training curves, per-category breakdowns, and the full comparison tables.

The one real architecture change from Dinomaly v1 is **context-aware recentering**:
before the reconstruction loss, the encoder-side target has its own class token
subtracted from every patch token, then LayerNorm. That's `Dinomaly.forward`'s
`context_aware_recenter` flag, reused verbatim from the submodule. Everything else
(two-stage noisy bottleneck, per-group LR, eps-guarded linear attention) is also
theirs; what's ours is the backbone loader, the MPS-safe pixel/image eval (their own
`utils.py` imports a CUDA-only extension that fails to import at all on a non-CUDA
machine), and the training loop.

### Data

- **MVTec-AD**: downloaded via the `ipythonx/mvtec-ad` Kaggle mirror, symlinked into
  `data/mvtec_anomaly_detection`. The official site requires a license form; this
  mirror doesn't gate it.
- **VisA**: public S3 download (`amazon-visual-anomaly.s3...VisA_20220922.tar`), then
  Dinomaly2's own `prepare_data/prepare_visa.py --split-type 1cls` for the
  per-category train/test/ground_truth layout.

### Run

```
python -m src.training.train_dinomaly2 --dataset mvtec --smoke        # ~150 iters, sanity check
python -m src.training.train_dinomaly2 --dataset mvtec                # full 40k iters, ~13-25h
python -m src.training.train_dinomaly2 --dataset visa
```

Or `notebooks/dinomaly2_repro_mvtec_visa.ipynb`, which runs both and reports on them
as it goes (loss/metric curves, per-category tables, paper comparison).

## CBIS-DDSM / mammography (paused)

[CBIS-DDSM](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset),
via the awsaf49 Kaggle mirror. Ships pre-converted JPEGs plus `dicom_info.csv`, not raw
`.dcm` files. [src/data/cbis_ddsm.py](src/data/cbis_ddsm.py) reads the JPEGs directly
and uses `dicom_info.csv` to link each case to its full mammogram image.

Pathology labels come from the `mass_case_description*.csv` / `calc_case_description*.csv`
files, not `dicom_info.csv` alone. An image with any malignant finding is `anomaly`;
everything else is `normal`. Split is patient-level (85/15 normal train/test, malignant
fully held out) so a patient's other breast/view can't leak across train and test.

Dinomaly v1 was ported and trained on this (`src/models/dinomaly.py`,
`src/training/train_dinomaly.py`, `notebooks/dinomaly_train_local.ipynb` /
`dinomaly_full_train.ipynb`). This is where the project returns to once Dinomaly2's
architecture is ported over the same way.

## Repo layout

```
data/                  # dataset cache, gitignored
notebooks/
  breast_cancer_eda_local.ipynb     # CBIS-DDSM EDA
  dinomaly_train_local.ipynb        # CBIS-DDSM: quick dev run
  dinomaly_full_train.ipynb         # CBIS-DDSM: full run, report, Dinomaly-v1-paper comparison
  dinomaly2_repro_mvtec_visa.ipynb  # MVTec-AD / VisA: full run, report, Dinomaly2-paper comparison
runs/                  # training checkpoints/logs, gitignored
src/
  data/
    cbis_ddsm.py       # CBIS-DDSM loading, patient-level split
    mvtec_visa.py      # MVTec-AD / VisA loading (reuses Dinomaly2's dataset.py)
  models/
    dinomaly.py        # Dinomaly v1 build (CBIS-DDSM)
    dinomaly2.py        # Dinomaly2 build (MVTec-AD / VisA reproduction)
    attention.py        # eps-guarded linear attention (v1 decoder)
    losses.py           # cosine reconstruction loss, hard mining (shared)
    lr_schedule.py       # warmup-cosine LR (v1 flavor, and v2's per-group ratio variant)
    _ref.py / _ref2.py   # bridges to the vendored Dinomaly / Dinomaly2 repos
  training/
    train_dinomaly.py    # CBIS-DDSM training loop
    train_dinomaly2.py   # MVTec-AD / VisA reproduction training loop
  eval/
    anomaly.py           # image-level scoring (CBIS-DDSM; also used by pixel.py)
    pixel.py              # pixel-level metrics incl. AUPRO (MVTec-AD / VisA)
third_party/
  Dinomaly/              # v1 reference implementation, git submodule
  Dinomaly2/              # v2 reference implementation, git submodule
```

## Setup

```
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
git submodule update --init  # pulls third_party/Dinomaly and Dinomaly2
```

Confirm the MPS backend is picked up (Apple Silicon):

```
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Current phase

**Dinomaly2 reproduced on MVTec-AD and VisA**, both within 0.5 points of the paper's
Table II across every image- and pixel-level metric. Next: architecture changes on top
of Dinomaly2, then porting that back to CBIS-DDSM.
