"""
Bridges to the vendored Dinomaly2 repo (third_party/Dinomaly2, a git submodule),
the same way _ref.py does for the original Dinomaly.

Loaded in isolation rather than via sys.path -- both submodules define a
top-level `models` package, and Python would cache whichever one is imported
first under that name, silently shadowing the other in any process that
touches both. `models/vision_transformer.py` also has an internal relative
import (`from .utils import ...`), so it's registered as a real package under
a name unique to this submodule (load_isolated_package) rather than loaded as
a standalone file.

utils.py (loss/scheduler/eval helpers) is NOT importable this way: it does
`from adeval import EvalAccumulatorCuda`, and adeval/iterative_cuda.py imports
a CUDA-only extension unconditionally, so the module fails to import on any
non-CUDA machine, MPS included. We don't use it -- global_cosine_hm_percent
already lives in src/models/losses.py, and the LR schedule/pixel metrics are
reimplemented in lr_schedule.py / eval/pixel.py.
"""

import importlib
import importlib.util
import sys
from pathlib import Path

REF_DIR = Path(__file__).resolve().parents[2] / "third_party" / "Dinomaly2"

if not (REF_DIR / "models" / "uad.py").exists():
    raise RuntimeError(
        f"Dinomaly2 submodule not found at {REF_DIR}. Run: git submodule update --init"
    )


def load_isolated(relpath, module_name):
    """Load a single self-contained file (no relative imports) under a private name."""
    spec = importlib.util.spec_from_file_location(module_name, REF_DIR / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_isolated_package(dirname, package_name):
    """
    Like load_isolated, but for a directory whose modules use relative imports
    (models/vision_transformer.py does `from .utils import ...`). Registers it
    in sys.modules as a real package under `package_name` so those resolve,
    without colliding with third_party/Dinomaly's own package of the same
    on-disk name.
    """
    pkg_dir = REF_DIR / dirname
    spec = importlib.util.spec_from_file_location(
        package_name, pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)]
    )
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = pkg
    spec.loader.exec_module(pkg)
    return pkg


_models = load_isolated_package("models", "_dinomaly2_models")
_uad = importlib.import_module("_dinomaly2_models.uad")
_vit = importlib.import_module("_dinomaly2_models.vision_transformer")

Dinomaly = _uad.Dinomaly
Block = _vit.Block
Attention = _vit.Attention
LinearAttention2 = _vit.LinearAttention2

StableAdamW = load_isolated("optimizers/StableAdamW.py", "_dinomaly2_stable_adamw").StableAdamW
