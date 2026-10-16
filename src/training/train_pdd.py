"""
Trains PDD on one dataset with the official recipe (PDD/scripts/
train_two_twins_ddp.py for single-class datasets, train_two_twins_ddp_multi.py
for Uni-Medical), run as one process on one device, matching the paper's
"single RTX A6000" setup (Sec. 4.1).

Recipe, taken from the scripts' defaults (layerloss=4, res=10):
  - input 256x256, ImageNet normalisation, no augmentation
  - batch size 10, 100 epochs, shuffled every epoch by DistributedSampler
  - Adam, lr 2e-3, betas (0.5, 0.999)
  - CosineAnnealingLR(T_max=40, eta_min=2e-4), stepped once per epoch.
    With 100 epochs and T_max=40 the rate falls to 2e-4 at epoch 40, rises
    back to 2e-3 at epoch 80 and falls again; that is what the scripts do
  - loss = l_prp + l_kr + l_div, with
        l_prp = 0.02 * MSE(S2, f_b) + 0.05 * CosineLoss(S2, f_b)
        l_kr  = 0.02 * MSE(f_b, S1)
        l_div = 0.5 * DiversityLoss(S2, S1, low_dim_layers=1,
                                    tau_low=0.75, tau_high=0.30)
    (S2 = student with the MLP prior, S1 = student without, f_b = InA-fused
    teacher features; MSE_loss, CosineLoss and DiversityLoss are imported
    from PDD/utils/losses.py unchanged)
  - test-set evaluation every 10 epochs

Where the paper text and the code disagree, the code is followed, because
it is what the authors released for these results:
  - Eq. (7) has beta * (1 - cos); CosineLoss returns +cos (the 1 - cos
    version is commented out in losses.py), so the code pushes S2 away from
    f_b in angle while the MSE term pulls it closer.
  - Sec. 4.3 says tau_low = 0.30, tau_high = 0.75. The code passes
    tau_low=0.75, tau_high=0.30, the pair Table 5 marks best on ZhangLab
    and HeadCT (and the "tao_07503" checkpoint name).
  - The README launches with 2 GPUs (2 x 10 images per step); the paper says
    one GPU. One device, batch 10, is used here.

Two additions that do not change what is computed:
  - Teacher cache (`cache_teacher=True`). Both teachers are frozen and in
    eval mode, and the transform has no randomness, so VMamba-Tiny returns
    the same features for an image every epoch. They are computed once per
    split and stored in data/PDD/_teacher_cache/, then read back in place
    of the forward pass. Without the CUDA scan kernel VMamba is ~95% of a
    training step on Apple silicon (0.95 s of 1.15 s per batch);
    Wide-ResNet50 is cheap and still runs live. `verify_cache` recomputes
    a few images and reports the largest difference.
  - Seeds. The official scripts never seed anything. Here `seed` sets the
    torch/numpy/random seeds before the decoder is built and is passed to
    DistributedSampler (whose own default seed is 0), so each seed has its
    own initialisation and batch order.

Outputs, in runs/pdd_<dataset>_seed<seed>[_<tag>]/:
  config.json, history.json (per-epoch losses and lr, every evaluation),
  checkpoint.pt (latest full state, for resuming), decoder_final.pt,
  scores_epoch<k>.npz (per-image labels and scores at each evaluation).
"""

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from src.data.pdd_datasets import PDD_DATA, build_datasets, prepare
from src.eval.pdd import batch_scores, summarize
from src.models.pdd import build_pdd, pdd_forward, upstream, vmamba_features

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PDD_DATA / "_teacher_cache"
VMAMBA_SHAPES = [(192, 32, 32), (384, 16, 16), (768, 8, 8), (768, 8, 8)]  # for 256x256 input


@dataclass
class PDDConfig:
    dataset: str = "head_ct"        # head_ct | brain_mri | zhanglab | chexpert | uni_medical
    image_size: int = 256
    epochs: int = 100
    lr: float = 2e-3
    betas: tuple = (0.5, 0.999)
    t_max: int = 40
    eta_min_ratio: float = 0.1      # eta_min = lr * 0.1
    res: int = 10
    batch_size: int = 10
    val_batch_size: int = 10
    eval_every: int = 10            # print_epoch in the scripts
    # layerloss == 4
    w_prp_mse: float = 0.02
    w_prp_cos: float = 0.05
    prp_cos: str = "code"           # "code": +CosineLoss as released; "paper": 1 - cos, Eq. (7)
    w_kr: float = 0.02
    w_div: float = 0.5
    low_dim_layers: int = 1
    tau_low: float = 0.75
    tau_high: float = 0.30
    seed: int = 1
    num_workers: int = 8
    cache_teacher: bool = True
    tag: str = ""                   # appended to the run folder name, for ablations
    out_dir: str | None = None

    @property
    def run_name(self):
        name = f"pdd_{self.dataset}_seed{self.seed}"
        return f"{name}_{self.tag}" if self.tag else name

    @classmethod
    def smoke(cls, **overrides):
        base = dict(epochs=2, eval_every=1, tag="smoke")
        base.update(overrides)
        return cls(**base)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def dataset_paths(ds):
    """Image paths of a dataset in index order (ImageFolder, PDD's
    MVTecDataset, or a ConcatDataset of PDD's MetaMVTecDataset)."""
    if hasattr(ds, "samples"):
        return [p for p, _ in ds.samples]
    if hasattr(ds, "img_paths"):
        return list(ds.img_paths)
    if hasattr(ds, "datasets"):
        return [str(Path(d.root_dir) / it["img_path"]) for d in ds.datasets for it in d.items]
    raise TypeError(type(ds))


class TeacherCache:
    """VMamba-Tiny features for every image of one split, one float32 memmap
    per stage under data/PDD/_teacher_cache/<key>/. `paths.json` records
    the image order, so a cache built for a different file list is rebuilt
    instead of read."""

    def __init__(self, key):
        self.dir = CACHE_DIR / key
        self._maps = None

    def _files(self):
        return [self.dir / f"stage{i}.f32" for i in range(len(VMAMBA_SHAPES))]

    def is_valid(self, paths):
        meta = self.dir / "paths.json"
        return meta.exists() and json.loads(meta.read_text()) == paths and all(f.exists() for f in self._files())

    @torch.no_grad()
    def build(self, ds, encoder_ssm, device, batch_size=10, num_workers=8):
        """Computes and stores the features unless a valid cache exists."""
        paths = dataset_paths(ds)
        if self.is_valid(paths):
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        n = len(paths)
        maps = [np.lib.format.open_memmap(f, mode="w+", dtype=np.float32, shape=(n, *s))
                for f, s in zip(self._files(), VMAMBA_SHAPES)]
        loader = DataLoader(PDDSamples(ds, self.dir.name), batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
        i = 0
        for batch in tqdm(loader, desc=f"teacher cache {self.dir.name}", dynamic_ncols=True):
            feats = vmamba_features(encoder_ssm, batch[0].to(device))
            b = feats[0].shape[0]
            for m, f in zip(maps, feats):
                m[i:i + b] = f.cpu().numpy()
            i += b
        for m in maps:
            m.flush()
        (self.dir / "paths.json").write_text(json.dumps(paths))

    def get(self, idx):
        if self._maps is None:  # opened lazily, once per DataLoader worker
            self._maps = [np.load(f, mmap_mode="r") for f in self._files()]
        return [torch.from_numpy(np.array(m[idx])) for m in self._maps]

    def __getstate__(self):
        return {"dir": self.dir, "_maps": None}


class PDDSamples(Dataset):
    """Wraps a PDD dataset so every item is (img, label, class_name, feats):
    single-class datasets yield (img, label) and get their dataset name as
    class; feats are the cached VMamba features, or [] without a cache."""

    def __init__(self, base, name, cache=None):
        self.base, self.name, self.cache = base, name, cache

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        img, label = item[0], int(item[1])
        cls = item[2] if len(item) > 2 else self.name
        feats = self.cache.get(idx) if self.cache is not None else []
        return img, label, cls, feats


@torch.no_grad()
def verify_cache(cache, ds, encoder_ssm, device, batch_size=10):
    """Recomputes the first cached batch (images 0..batch_size-1, the same
    batch the cache was built from) and returns the largest absolute
    difference from the cached features. It should be exactly 0. Running
    the same image in a batch of a different size changes the features at
    the level of float rounding (about 1e-5 relative on M4 Max MPS), the
    same as it would in a live run."""
    n = min(batch_size, len(ds))
    live = vmamba_features(encoder_ssm, torch.stack([ds[i][0] for i in range(n)]).to(device))
    worst = 0.0
    for i in range(n):
        for a, b in zip(cache.get(i), live):
            worst = max(worst, (a - b[i].cpu()).abs().max().item())
    return worst


@torch.no_grad()
def evaluate(encoder_ssm, encoder_res, decoder, loader, device, res, single_class, val_batch_size):
    """Scores the whole test loader. Returns (summary dict, raw arrays)."""
    decoder.eval()
    labels, ts, ss, classes = [], [], [], []
    accumulate = np.float64 if single_class else np.float32
    for img, label, cls, feats in loader:
        img = img.to(device)
        feats = [f.to(device) for f in feats] or None
        outputs = pdd_forward(encoder_ssm, encoder_res, decoder, img, res=res, ssm_feats=feats)
        s = batch_scores(outputs, img.shape[-1], accumulate)
        ts.append(s["ts"])
        ss.append(s["ss"])
        labels.extend(label.tolist())
        classes.extend(cls)
    raw = {"labels": np.asarray(labels), "scores_ts": np.concatenate(ts),
           "scores_ss": np.concatenate(ss), "classes": np.asarray(classes)}
    summary = summarize(raw["labels"], raw["scores_ts"], raw["scores_ss"], raw["classes"],
                        single_class, val_batch_size)
    return summary, raw


def train(config: PDDConfig, device=None, progress=True):
    """Trains (or resumes) one run and returns its history dict. A finished
    run with at least config.epochs epochs is loaded, not retrained; an
    unfinished or shorter one continues from checkpoint.pt."""
    device = device or pick_device()
    out_dir = Path(config.out_dir) if config.out_dir else REPO_ROOT / "runs" / config.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        if history.get("finished") and len(history["epochs"]) >= config.epochs:
            print(f"{config.run_name}: finished run found, loading {history_path}")
            return history
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))

    def say(msg):
        (tqdm.write if progress else print)(msg)

    set_seed(config.seed)
    up = upstream()
    L = up.losses
    single_class = config.dataset != "uni_medical"

    prepare(config.dataset)
    train_base, test_base, class_names = build_datasets(config.dataset, config.image_size)
    encoder_ssm, encoder_res, decoder = build_pdd(device)

    train_cache = test_cache = None
    if config.cache_teacher:
        train_cache = TeacherCache(f"{config.dataset}_train_{config.image_size}")
        test_cache = TeacherCache(f"{config.dataset}_test_{config.image_size}")
        train_cache.build(train_base, encoder_ssm, device, config.batch_size, config.num_workers)
        test_cache.build(test_base, encoder_ssm, device, config.val_batch_size, config.num_workers)

    train_data = PDDSamples(train_base, config.dataset, train_cache)
    test_data = PDDSamples(test_base, config.dataset, test_cache)
    sampler = DistributedSampler(train_data, num_replicas=1, rank=0, shuffle=True, seed=config.seed)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, sampler=sampler,
                              num_workers=config.num_workers, persistent_workers=config.num_workers > 0)
    test_loader = DataLoader(test_data, batch_size=config.val_batch_size, shuffle=False,
                             num_workers=config.num_workers, persistent_workers=config.num_workers > 0)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=config.lr, betas=tuple(config.betas))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.t_max, eta_min=config.lr * config.eta_min_ratio)

    history = {
        "config": asdict(config), "classes": class_names, "single_class": single_class,
        "n_train": len(train_data), "n_test": len(test_data), "batches_per_epoch": len(train_loader),
        "epochs": [], "loss": [], "l_prp": [], "l_kr": [], "l_div": [], "lr": [],
        "evals": [], "elapsed_min": 0.0, "finished": False, "out_dir": str(out_dir),
    }
    start_epoch = 0
    ckpt_path = out_dir / "checkpoint.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        decoder.load_state_dict(ckpt["decoder"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        history = ckpt["history"]
        history["config"], history["finished"] = asdict(config), False
        start_epoch = ckpt["epoch"] + 1
        say(f"resumed from epoch {start_epoch} ({ckpt_path})")

    n_params = sum(p.numel() for p in decoder.parameters())
    say(f"device {device} | {config.dataset} | train {len(train_data)} | test {len(test_data)} | "
        f"{len(train_loader)} batches/epoch | decoder {n_params / 1e6:.1f}M params | "
        f"teacher cache {'on' if config.cache_teacher else 'off'}")
    if config.cache_teacher and start_epoch == 0:
        diff = verify_cache(train_cache, train_base, encoder_ssm, device, config.batch_size)
        say(f"cached vs live VMamba features, max abs diff: {diff:.2e}")
        history["cache_max_abs_diff"] = diff
    say(f"-> {out_dir}")

    t0 = time.time() - history["elapsed_min"] * 60
    for epoch in range(start_epoch, config.epochs):
        decoder.train()
        sampler.set_epoch(epoch)
        sums = np.zeros(4)
        n_batches = 0
        bar = tqdm(train_loader, disable=not progress, dynamic_ncols=True, leave=False,
                   desc=f"epoch {epoch + 1}/{config.epochs}")
        for img, _, _, feats in bar:
            img = img.to(device)
            feats = [f.to(device) for f in feats] or None
            outputs = pdd_forward(encoder_ssm, encoder_res, decoder, img, res=config.res, ssm_feats=feats)

            # losses.loss_fucntion is the 1 - cos version left in utils/losses.py
            cos_term = L.CosineLoss if config.prp_cos == "code" else L.loss_fucntion
            l_prp = config.w_prp_mse * L.MSE_loss(outputs[0:3], outputs[3:6]) \
                + config.w_prp_cos * cos_term(outputs[0:3], outputs[3:6])
            l_kr = config.w_kr * L.MSE_loss(outputs[3:6], outputs[6:9])
            if config.w_div:
                l_div = config.w_div * L.DiversityLoss(
                    features_student1=outputs[0:3], features_student2=outputs[6:9],
                    low_dim_layers=config.low_dim_layers, tau_low=config.tau_low, tau_high=config.tau_high)
            else:
                l_div = torch.zeros((), device=device)
            loss = l_prp + l_kr + l_div

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sums += [loss.item(), l_prp.item(), l_kr.item(), l_div.item()]
            n_batches += 1
            bar.set_postfix(loss=f"{sums[0] / n_batches:.4f}")

        lr_used = optimizer.param_groups[0]["lr"]
        scheduler.step()
        means = sums / max(n_batches, 1)
        history["epochs"].append(epoch + 1)
        history["lr"].append(lr_used)
        for k, v in zip(["loss", "l_prp", "l_kr", "l_div"], means):
            history[k].append(float(v))

        if (epoch + 1) % config.eval_every == 0:
            summary, raw = evaluate(encoder_ssm, encoder_res, decoder, test_loader, device,
                                    config.res, single_class, config.val_batch_size)
            history["evals"].append({"epoch": epoch + 1, **summary})
            np.savez(out_dir / f"scores_epoch{epoch + 1}.npz", **raw)
            if single_class:
                m, o = summary["global"], summary["official_batch"]
                say(f"[epoch {epoch + 1}] loss {means[0]:.4f} (prp {means[1]:.4f} kr {means[2]:.4f} "
                    f"div {means[3]:.4f}) | per-image AUROC {m['auroc']:.4f} AP {m['ap']:.4f} "
                    f"F1 {m['f1']:.4f} | official-batch AUROC {o['auroc']:.4f}")
            else:
                per = "  ".join(f"{c} {v['auroc']:.4f}/{v['ap']:.4f}/{v['f1']:.4f}"
                                for c, v in summary["per_class"].items())
                say(f"[epoch {epoch + 1}] loss {means[0]:.4f} | AUROC/AP/F1  {per}  "
                    f"| mean {summary['mean']['auroc']:.4f}/{summary['mean']['ap']:.4f}/"
                    f"{summary['mean']['f1']:.4f}")

        history["elapsed_min"] = (time.time() - t0) / 60
        torch.save({"decoder": decoder.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "epoch": epoch, "history": history}, ckpt_path)
        history_path.write_text(json.dumps(history, indent=1))

    torch.save(decoder.state_dict(), out_dir / "decoder_final.pt")
    history["finished"] = True
    history_path.write_text(json.dumps(history, indent=1))
    say(f"done in {history['elapsed_min']:.1f} min -> {out_dir}")
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="head_ct",
                        choices=["head_ct", "brain_mri", "zhanglab", "chexpert", "uni_medical"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--no-cache", action="store_true", help="run VMamba live every step")
    parser.add_argument("--smoke", action="store_true", help="2 epochs, evaluated after each")
    args = parser.parse_args()
    if args.smoke:
        config = PDDConfig.smoke(dataset=args.dataset, seed=args.seed, cache_teacher=not args.no_cache)
    else:
        config = PDDConfig(dataset=args.dataset, seed=args.seed, epochs=args.epochs,
                           cache_teacher=not args.no_cache)
    print(f"config: {config}", flush=True)
    train(config)


if __name__ == "__main__":
    sys.exit(main())
