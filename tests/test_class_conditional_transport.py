from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "cc_hiwa"))

from class_conditional_transport import source_label_assignments, target_prediction_assignments


def test_source_label_assignments_are_semantic_and_row_stochastic() -> None:
    assignments = source_label_assignments(np.asarray([0, 2, 1]), n_classes=3, smoothing=0.1)
    np.testing.assert_allclose(assignments.sum(axis=1), 1.0)
    np.testing.assert_allclose(assignments[0], np.asarray([0.9, 0.05, 0.05]))
    np.testing.assert_allclose(assignments[1], np.asarray([0.05, 0.05, 0.9]))


def test_target_prediction_assignments_follow_logits_and_temperature() -> None:
    logits = np.asarray([[4.0, 0.0], [0.0, 4.0]])
    cool = target_prediction_assignments(logits, temperature=0.5)
    warm = target_prediction_assignments(logits, temperature=2.0)
    np.testing.assert_allclose(cool.sum(axis=1), 1.0)
    assert cool[0, 0] > warm[0, 0] > 0.5
    assert cool[1, 1] > warm[1, 1] > 0.5


def test_source_label_assignments_reject_out_of_range_labels() -> None:
    try:
        source_label_assignments(np.asarray([0, 3]), n_classes=3)
    except ValueError as error:
        assert "[0, n_classes)" in str(error)
    else:
        raise AssertionError("out-of-range source labels must be rejected")
