"""
Puts the vendored Dinomaly repo (third_party/Dinomaly, a git submodule) on the
import path so we can reuse its architecture code verbatim instead of
reimplementing attention math that's easy to get subtly wrong.

Only a few pieces are pulled in from there:
  - models.uad.ViTill                        the encoder/bottleneck/decoder wrapper
  - models.vision_transformer.{Block, ...}   the decoder transformer blocks
  - optimizers.StableAdamW.StableAdamW       the optimizer the paper trains with

Everything else (loss, LR schedule, anomaly scoring, the training loop) is our
own code under src/, because the repo's utils.py isn't importable in isolation.
"""

import sys
from pathlib import Path

REF_DIR = Path(__file__).resolve().parents[2] / "third_party" / "Dinomaly"

if not (REF_DIR / "models" / "uad.py").exists():
    raise RuntimeError(
        f"Dinomaly submodule not found at {REF_DIR}. Run: git submodule update --init"
    )

# Appended, not inserted at front: our own packages keep priority, and nothing
# here is named to collide with src/.
if str(REF_DIR) not in sys.path:
    sys.path.append(str(REF_DIR))


def load_isolated(relpath, module_name):
    """
    Import a single file from the repo without running its package __init__.
    optimizers/__init__.py pulls in every optimizer (and third-party deps we
    don't want, like tabulate), so StableAdamW has to be loaded this way.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, REF_DIR / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


StableAdamW = load_isolated("optimizers/StableAdamW.py", "_dinomaly_stable_adamw").StableAdamW
