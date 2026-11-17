"""Author-reference settings and utilities shared by both experiments."""

import hashlib
import importlib.util
from pathlib import Path

from gcot.alignment import SolverConfig


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]
AUTHOR = WORKSPACE / "HiWA"
AUTHOR_PANEL_MAPPING = {"e": "e", "f": "f"}


def shared_config():
    """Return the fixed iteration budgets used by the HiWA authors."""
    return SolverConfig(sinkhorn_tol=0)


def author_utils():
    """Load the unmodified plotting helpers from the local HiWA checkout."""
    path = AUTHOR / "PyHiWA/src/utils.py"
    spec = importlib.util.spec_from_file_location("author_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load author utilities from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
