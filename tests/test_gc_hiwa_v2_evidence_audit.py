from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIRECTORY = ROOT / "experiments" / "synthetic"
if str(SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIRECTORY))

from verify_gc_hiwa_v2_evidence import audit  # noqa: E402


def test_frozen_gc_hiwa_v2_evidence_chain_is_intact() -> None:
    result = audit()
    assert result["synthetic_independent_holdout"]["recoverable_core_accepted"] == 15
    assert result["synthetic_independent_holdout"]["declared_violations_core_abstained"] == 25
    assert result["isospectral_nonorthogonal_counterexample"]["core_abstained"] == 5
    assert result["panoptic_controlled_recovery"]["accepted"]
