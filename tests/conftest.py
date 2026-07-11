"""Keep migrated source and experiment modules importable during tests."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for path in (
    ROOT / "src",
    ROOT / "src" / "cc_hiwa",
    ROOT / "experiments" / "hiwa",
    ROOT / "experiments" / "roca",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
