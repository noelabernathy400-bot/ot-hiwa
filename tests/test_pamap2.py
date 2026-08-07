from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets.pamap2 import PAMAP2SplitCounts, build_paired_windows, build_temporal_domain_adaptation_split, window_features
from audit_multi_subject_eligibility import assess_subject


def _rows() -> np.ndarray:
    rows = np.zeros((24, 54), dtype=float)
    rows[:, 0] = np.arange(24) / 100.0
    rows[:12, 1] = 1
    rows[12:, 1] = 2
    for column in range(54):
        rows[:, column] += np.linspace(0.0, 1.0, 24) * (column + 1)
    rows[:, 1] = np.repeat((1.0, 2.0), 12)
    return rows


def test_window_features_have_the_documented_dimension() -> None:
    assert window_features(np.ones((8, 6))).shape == (48,)


def test_paired_windows_use_source_labels_and_preserve_hidden_metadata() -> None:
    prepared = build_paired_windows(
        _rows(),
        subject=101,
        activities=(1, 2),
        window_samples=4,
        windows_per_activity=2,
    )
    assert prepared.source_features.shape == (4, 48)
    assert prepared.target_features.shape == (4, 48)
    assert prepared.source_labels.tolist() == [1, 1, 2, 2]
    assert prepared.target_labels.tolist() == [1, 1, 2, 2]
    assert len(prepared.pair_ids) == 4


def test_temporal_split_has_disjoint_partitions_and_hides_adaptation_metadata() -> None:
    rows = np.zeros((64, 54), dtype=float)
    rows[:, 0] = np.arange(64) / 100.0
    rows[:32, 1] = 1
    rows[32:, 1] = 2
    for column in range(54):
        rows[:, column] += np.linspace(0.0, 1.0, 64) * (column + 1)
    rows[:, 1] = np.repeat((1.0, 2.0), 32)
    split = build_temporal_domain_adaptation_split(
        rows,
        subject=101,
        activities=(1, 2),
        window_samples=4,
        counts=PAMAP2SplitCounts(source_train=2, source_validation=2, target_adaptation=2, target_test=2),
    )
    assert split.source_train_features.shape == (4, 48)
    assert split.source_train_paired_target_features.shape == (4, 48)
    assert split.source_validation_features.shape == (4, 48)
    assert split.target_adaptation_features.shape == (4, 48)
    assert split.evaluation_target_features.shape == (4, 48)
    assert split.source_train_labels.tolist() == [1, 1, 2, 2]
    assert split.evaluation_target_labels.tolist() == [1, 1, 2, 2]
    assert len(split.evaluation_pair_ids) == 4


def test_multi_subject_eligibility_reports_missing_protocol_file(tmp_path: Path) -> None:
    record = assess_subject(
        tmp_path,
        102,
        window_samples=4,
        counts=PAMAP2SplitCounts(source_train=1, source_validation=1, target_adaptation=1, target_test=1),
    )
    assert record["available"] is False
    assert record["eligible"] is False
    assert record["reason"] == "protocol_file_missing"
