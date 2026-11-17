import numpy as np

from experiments.dandi_000688.compare import evaluate
from experiments.dandi_000688.config import N_GROUPS, PREPROCESSING
from experiments.dandi_000688.prepare import _movement_onsets


def test_movement_onset_uses_sustained_post_go_speed():
    timestamps = np.arange(0.0, 2.0, 0.01)
    velocity = np.zeros((len(timestamps), 2))
    velocity[(timestamps >= 0.72) & (timestamps < 0.90), 0] = 10.0
    onsets = _movement_onsets(
        np.array([0.50]),
        np.array([1.20]),
        velocity,
        timestamps,
        threshold_fraction=0.10,
        sustain_ms=30.0,
    )
    assert np.isfinite(onsets[0])
    assert 0.60 <= onsets[0] <= 0.75


def test_perfect_alignment_has_perfect_direction_metrics():
    labels = np.repeat(np.arange(N_GROUPS), 3)
    points = np.eye(N_GROUPS, 4, k=0)[labels % 4]
    # Add a direction-specific offset so all eight centroids are distinct in 4D.
    points = points + labels[:, None] * np.array([0.01, 0.02, 0.03, 0.04])
    metrics = evaluate(points, labels, points.copy(), labels.copy())
    assert metrics["cluster_matching_accuracy"] == 1.0
    assert metrics["direction_accuracy_1nn"] == 1.0
    assert metrics["direction_accuracy_5nn"] == 1.0
    assert metrics["centroid_nrmse"] == 0.0


def test_accepted_preprocessing_contract():
    assert PREPROCESSING.latent_dimension == 4
    assert PREPROCESSING.window_ms == (-50.0, 400.0)
    assert N_GROUPS == 8
