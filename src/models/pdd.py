"""
PDD (Lu et al., CVPR 2026, "Manifold-Prior Diverse Distillation for Medical
Anomaly Detection"), loaded straight from the official code in PDD/ rather
than re-written. The teachers (VMamba-Tiny, Wide-ResNet50-2), the decoder
(InA, MMU, MLP prior and the shared student layers) and the losses are the
upstream files, imported by path:

    PDD/models/vmamba.py          VMamba-Tiny teacher
    PDD/models/resnet_encoder.py  Wide-ResNet50-2 teacher
    PDD/models/mamba_decoder.py   pdd_decoder (both students)
    PDD/utils/losses.py           MSE_loss, CosineLoss, DiversityLoss
    PDD/utils/dataset.py          MVTecDataset (single-class test set)
    PDD/utils/dataset_full.py     meta.json datasets (Uni-Medical)
    PDD/utils/evaluation.py       compute_f1max

They are imported as `pdd_upstream.<name>` through a small import hook
instead of putting PDD/ on sys.path, because PDD's top-level packages are
called `models` and `utils` and would shadow anything else with those names.

Three things in the upstream code do not run on a Mac. All three are
handled here without editing the upstream files:

1. vmamba.py decorates its Triton kernels with `@triton.jit` even when the
   `import triton` above it failed, so the module raises NameError on any
   machine without Triton (Triton is CUDA-only). A stub `triton` name is
   put into the module's namespace before it runs. The stubbed kernels are
   only used by the "v2"/"v3" forward types; vanilla VMamba-Tiny uses
   forward_type="v0", which never calls them.

2. Without the `selective_scan_cuda` extension, vmamba.py falls back to
   `selective_scan_torch`, which runs the state-space recurrence one token
   at a time in Python: 4096 sequential steps per block at stage 1 for a
   256x256 input, about 2.6 s per batch of 10 on an M4 Max.
   `chunked_selective_scan` computes the same recurrence in 2*T + L/T
   sequential steps (T ~ sqrt(L)) and replaces `selective_scan_fn`. The
   math is identical; only the order of floating-point additions between
   chunks changes. `check_selective_scan` measures the difference against
   the upstream reference.

3. VMamba returns its features as permute(0, 3, 1, 2) views. On MPS the
   weight gradient of a conv fed such a view fails with a stride error, so
   `pdd_forward` copies them to contiguous memory first (same values).
"""

import importlib.abc
import importlib.machinery
import importlib.util
import math
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
PDD_ROOT = REPO_ROOT / "PDD"

if not PDD_ROOT.exists():
    raise FileNotFoundError(f"{PDD_ROOT} not found. Clone https://github.com/OxygenLu/PDD there.")


def _triton_stub():
    """Stand-ins for the `triton` / `tl` names vmamba.py touches at import
    time (decorators and type annotations). They go into vmamba.py's own
    namespace only, never into sys.modules: torch probes for a real Triton
    install, and a fake package in sys.modules breaks that probe."""

    class _Anything:
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return _Anything()

        def __call__(self, *args, **kwargs):
            return _Anything()

    triton = types.SimpleNamespace(
        jit=lambda fn=None, **kwargs: fn if fn is not None else (lambda f: f),
        language=_Anything(),
    )
    return triton, triton.language


UPSTREAM_FILES = {
    "vmamba": "models/vmamba.py",
    "resnet_encoder": "models/resnet_encoder.py",
    "mamba_decoder": "models/mamba_decoder.py",
    "losses": "utils/losses.py",
    "dataset": "utils/dataset.py",
    "dataset_full": "utils/dataset_full.py",
    "evaluation": "utils/evaluation.py",
}


class _UpstreamLoader(importlib.machinery.SourceFileLoader):
    def exec_module(self, module):
        short = module.__name__.split(".", 1)[1]
        if short == "vmamba" and importlib.util.find_spec("triton") is None:
            # vmamba.py's own `import triton` still fails and sets
            # WITH_TRITON = False; the stub only satisfies `@triton.jit`.
            module.triton, module.tl = _triton_stub()
        super().exec_module(module)
        if short == "vmamba":
            module.selective_scan_reference = module.selective_scan_torch
            if not module.WITH_SELECTIVESCAN_MAMBA:
                module.selective_scan_fn = chunked_selective_scan


class _UpstreamFinder(importlib.abc.MetaPathFinder):
    """Makes `import pdd_upstream.<name>` load PDD/<UPSTREAM_FILES[name]>.
    A real import path (rather than loading files by hand) is needed so that
    DataLoader worker processes, which unpickle PDD's dataset classes by
    module name, can find them too."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pdd_upstream":
            spec = importlib.machinery.ModuleSpec(fullname, None, is_package=True)
            spec.submodule_search_locations = []
            return spec
        if fullname.startswith("pdd_upstream.") and fullname.split(".", 1)[1] in UPSTREAM_FILES:
            file = str(PDD_ROOT / UPSTREAM_FILES[fullname.split(".", 1)[1]])
            return importlib.util.spec_from_file_location(fullname, file, loader=_UpstreamLoader(fullname, file))
        return None


if not any(isinstance(f, _UpstreamFinder) for f in sys.meta_path):
    sys.meta_path.append(_UpstreamFinder())

vmamba = None  # set by upstream()


def upstream():
    """The upstream modules as a namespace: vmamba, resnet_encoder,
    mamba_decoder, losses, dataset, dataset_full, evaluation."""
    global vmamba
    mods = {name: importlib.import_module(f"pdd_upstream.{name}") for name in UPSTREAM_FILES}
    vmamba = mods["vmamba"]
    return types.SimpleNamespace(**mods)


def chunked_selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True,
                           oflex=True, backend=None, chunk=None):
    """Same arguments and output as vmamba.selective_scan_torch:

        h_t = exp(delta_t * A) * h_{t-1} + delta_t * B_t * u_t
        y_t = <h_t, C_t> + D * u_t

    u, delta: (batch, K*C, L); A: (K*C, N); B, C: (batch, K, N, L).

    The length-L sequence is cut into L/T chunks of length T and solved in
    three passes:
      1. every chunk runs the recurrence from h = 0, all chunks at once
         (T steps), giving its local end state;
      2. the state entering each chunk is carried across chunks (L/T steps),
         H_c = decay_c * H_{c-1} + local_end_{c-1}, where decay_c is the
         chunk's total decay exp(sum_t delta_t * A);
      3. every chunk reruns the recurrence from its entering state (T steps)
         and y is read out from all states at once.

    Everything is laid out time-major, (T, batch, K*C, L/T, N), so each step
    touches one contiguous slice, and pass 3 writes the states in place
    over delta*B*u.
    """
    dtype_in = u.dtype
    Bt, K, N, L = B.shape
    KC = u.shape[1]
    Cd = KC // K

    if delta_bias is not None:
        delta = delta + delta_bias[..., None]
    if delta_softplus:
        delta = F.softplus(delta)
    u, delta, A, B, C = u.float(), delta.float(), A.float(), B.float(), C.float()

    T = min(chunk or 2 ** round(math.log2(math.sqrt(L))), L)
    pad = (-L) % T
    if pad:
        # delta = 0 and u = 0 after the end: decay exp(0) = 1 and no input,
        # so the padded steps leave the state unchanged.
        u, delta = F.pad(u, (0, pad)), F.pad(delta, (0, pad))
        B, C = F.pad(B, (0, pad)), F.pad(C, (0, pad))
    nc = (L + pad) // T

    def time_major(x):  # (..., L) -> (T, ..., nc)
        return x.view(*x.shape[:-1], nc, T).movedim(-1, 0).contiguous()

    d_tm = time_major(delta)                       # (T, Bt, KC, nc)
    B_tm = time_major(B).transpose(-1, -2)         # (T, Bt, K, nc, N)
    C_tm = time_major(C).transpose(-1, -2)
    dA = torch.exp(d_tm.unsqueeze(-1) * A.view(1, 1, KC, 1, N))
    H = (time_major(delta * u).view(T, Bt, K, Cd, nc, 1) * B_tm.unsqueeze(3)).view(T, Bt, KC, nc, N)

    # pass 1
    h = H[0].clone()
    for t in range(1, T):
        h = torch.addcmul(H[t], dA[t], h)
    decay = torch.exp(d_tm.sum(0).unsqueeze(-1) * A.view(1, KC, 1, N))

    # pass 2
    h_in = torch.zeros_like(h)
    for c in range(1, nc):
        h_in[:, :, c] = torch.addcmul(h[:, :, c - 1], decay[:, :, c - 1], h_in[:, :, c - 1])

    # pass 3
    H[0].addcmul_(dA[0], h_in)
    for t in range(1, T):
        H[t].addcmul_(dA[t], H[t - 1])
    y = torch.einsum("tbkcqn,tbkqn->bkcqt", H.view(T, Bt, K, Cd, nc, N), C_tm)
    y = y.reshape(Bt, KC, nc * T)[..., :L]

    out = y if D is None else y + u[..., :L] * D.unsqueeze(-1)
    return out if oflex else out.to(dtype=dtype_in)


@torch.no_grad()
def check_selective_scan(device, batch=2, seed=0):
    """Runs the upstream per-token scan and chunked_selective_scan on the same
    random inputs, at the four sequence lengths VMamba-Tiny sees for a
    256x256 image, and returns the max absolute difference per length."""
    upstream()
    g = torch.Generator().manual_seed(seed)
    results = {}
    for c, length in [(192, 4096), (384, 1024), (768, 256), (1536, 64)]:
        kc, n = 4 * c, 16
        u = torch.randn(batch, kc, length, generator=g)
        delta = torch.randn(batch, kc, length, generator=g) * 0.5 - 2.0
        A = -torch.exp(torch.rand(kc, n, generator=g) * 2.8)       # [-16, -1], like mamba_init
        Bm = torch.randn(batch, 4, n, length, generator=g)
        Cm = torch.randn(batch, 4, n, length, generator=g)
        D = torch.ones(kc)
        bias = torch.randn(kc, generator=g) * 0.1
        args = [t.to(device) for t in (u, delta, A, Bm, Cm, D)]
        bias = bias.to(device)
        ref = vmamba.selective_scan_reference(*args, delta_bias=bias, delta_softplus=True)
        out = chunked_selective_scan(*args, delta_bias=bias, delta_softplus=True)
        results[length] = {"max_abs_diff": (ref - out).abs().max().item(),
                           "max_abs_value": ref.abs().max().item()}
    return results


def build_pdd(device):
    """The model set the official training scripts build:

        encoder_res = wide_resnet50_2(pretrained=True).eval()
        encoder_ssm = create_model("vanilla_vmamba_tiny", pretrained=True).eval()
        decoder     = pdd_decoder(pretrained=False)

    `create_model("vanilla_vmamba_tiny")` resolves to vmamba.vanilla_vmamba_tiny,
    which is called directly. Teacher weights download through torch.hub on
    first use (timm's wide_resnet50_racm-8234f177.pth and VMamba's
    vssmtiny_dp01_ckpt_epoch_292.pth). Returns (encoder_ssm, encoder_res, decoder).
    """
    up = upstream()
    encoder_res = up.resnet_encoder.wide_resnet50_2(pretrained=True).to(device).eval()
    encoder_ssm = up.vmamba.vanilla_vmamba_tiny(pretrained=True).to(device).eval()
    for p in list(encoder_res.parameters()) + list(encoder_ssm.parameters()):
        p.requires_grad_(False)
    decoder = up.mamba_decoder.pdd_decoder(pretrained=False).to(device)
    return encoder_ssm, encoder_res, decoder


@torch.no_grad()
def vmamba_features(encoder_ssm, img):
    """VMamba-Tiny's four stage outputs, contiguous (see note 3 above)."""
    return [f.contiguous() for f in encoder_ssm(img, feature_list=True)]


def pdd_forward(encoder_ssm, encoder_res, decoder, img, res=10, ssm_feats=None):
    """One forward pass as written in the training scripts. `ssm_feats` can
    hold VMamba features computed earlier for the same images (the teacher
    cache in src/training/train_pdd.py); otherwise they are computed here.
    Returns the decoder's 9-feature list:

        [0:3]  Student 2 (with MLP prior, "vssm" path), 64/32/16 px
        [3:6]  InA-fused teacher features f_b,           64/32/16 px
        [6:9]  Student 1 (no prior, "res" path),         64/32/16 px
    """
    if ssm_feats is None:
        ssm_feats = vmamba_features(encoder_ssm, img)
    with torch.no_grad():
        inputs_res = encoder_res(img, feature_list=True)
    return decoder(inputs_res[3], inputs_res, ssm_feats, res=res, feature_list=True)
