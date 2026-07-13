from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets.pamap2 import build_paired_windows, window_features


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
