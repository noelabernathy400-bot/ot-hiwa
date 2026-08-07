from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "synthetic" / "summarize_qualification_calibration.py"
SPEC = importlib.util.spec_from_file_location("qualification_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _core(scenario: str, accepted: bool) -> dict[str, object]:
    return {
        "scenario": scenario,
        "method": "soft_transport_oracle_component",
        "rotation_frobenius_error": 0.01,
        "applicability_qualification": {"accepted": accepted},
    }


def _roca(scenario: str, selected: bool) -> dict[str, object]:
    return {
        "scenario": scenario,
        "method": "soft_roca" if selected else "soft_roca_abstained",
        "rotation_frobenius_error": 0.01 if selected else None,
        "roca_selected_correct_component": selected,
    }


def test_audit_reports_fixed_decisions_without_using_truth_for_selection() -> None:
    payload = {
        "experiment": "fixture",
        "records": [
            _core("ideal_rotation", True),
            _roca("ideal_rotation", True),
            _core("cross_modal_independent", False),
            _roca("cross_modal_independent", False),
        ],
    }

    result = MODULE.audit(payload, {"ideal_rotation"})

    aggregate = result["aggregate"]
    assert aggregate["recoverable_runs"] == 1
    assert aggregate["recoverable_core_accepted"] == 1
    assert aggregate["recoverable_roca_accepted"] == 1
    assert aggregate["nonrecoverable_runs"] == 1
    assert aggregate["nonrecoverable_core_abstained"] == 1
    assert aggregate["nonrecoverable_roca_abstained"] == 1
    assert aggregate["roca_nonrecovery_abstain_rate"] == 1.0
