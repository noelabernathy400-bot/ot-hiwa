from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "experiments" / "roca"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from analyze_roca_candidate_disagreement import audit_payloads, summarize  # noqa: E402


def _row(sign: int, accuracy: float, objective: float) -> dict:
    return {
        "method": f"soft_gcot_roca_candidate_det_{sign:+d}",
        "seed": 9,
        "stage": 2,
        "rotation_determinant": float(sign),
        "transport_objective": objective,
        "after_direction_accuracy": accuracy,
        "after_velocity_r2": accuracy,
    }


def test_disagreement_is_computed_without_evaluation_metric() -> None:
    payload = {
        "roca": [{"seed": 9, "selected_determinant_sign": -1, "volume_product_margin": 3.0, "warning": False}],
        "stage_results": [_row(-1, 0.2, 1.2), _row(1, 0.8, 0.8)],
    }
    record = audit_payloads([payload])[0]
    assert record["geometric_sign"] == -1
    assert record["objective_sign"] == 1
    assert record["geometry_objective_disagree"] is True
    assert record["objective_conflict_abstain"] is True
    assert record["geometry_matches_oracle_evaluation_only"] is False


def test_summary_separates_unlabeled_rejection_from_evaluation() -> None:
    payload = {
        "roca": [
            {"seed": 9, "selected_determinant_sign": -1, "volume_product_margin": 3.0, "warning": False},
            {"seed": 10, "selected_determinant_sign": 1, "volume_product_margin": 3.0, "warning": False},
        ],
        "stage_results": [
            _row(-1, 0.2, 1.2),
            _row(1, 0.8, 0.8),
            {**_row(-1, 0.3, 1.1), "seed": 10},
            {**_row(1, 0.9, 0.7), "seed": 10},
        ],
    }
    result = summarize(audit_payloads([payload]))
    assert result["rejected_by_material_conflict"] == 1
    assert result["accepted_without_material_conflict"] == 1
    assert result["wrong_selections_rejected_evaluation_only"] == 1


def test_audit_keeps_data_subset_identity_separate_from_solver_seed() -> None:
    payload = {
        "sampling": {"mode": "unlabeled_random_without_replacement", "seed": 1001},
        "roca": [{"seed": 0, "selected_determinant_sign": -1, "volume_product_margin": 3.0, "warning": False}],
        "stage_results": [_row(-1, 0.8, 0.8), _row(1, 0.2, 1.2)],
    }
    payload["stage_results"] = [{**row, "seed": 0} for row in payload["stage_results"]]
    record = audit_payloads([payload])[0]
    assert record["case_id"] == "subset_1001_solver_0"
    assert record["subset_seed"] == 1001
    assert record["sampling_mode"] == "unlabeled_random_without_replacement"


def test_summary_falsifies_marker_when_it_misses_a_wrong_selection() -> None:
    payload = {
        "roca": [{"seed": 9, "selected_determinant_sign": -1, "volume_product_margin": 3.0, "warning": False}],
        "stage_results": [_row(-1, 0.2, 1.0), _row(1, 0.8, 1.0)],
    }
    result = summarize(audit_payloads([payload]))
    assert result["wrong_selections_rejected_evaluation_only"] == 0
    assert "falsifies" in result["interpretation"]
