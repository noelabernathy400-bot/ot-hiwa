import numpy as np

from experiments.panoptic.compare import evaluate
from experiments.panoptic.prepare import _camera_coordinates


def test_camera_rotation_preserves_body_centered_pose_distances():
    rng = np.random.default_rng(3)
    poses = rng.normal(size=(5, 19, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    camera = {"R": q.tolist()}
    transformed = _camera_coordinates(poses, camera)
    np.testing.assert_allclose(
        np.linalg.norm(poses[:, 1] - poses[:, 7], axis=1),
        np.linalg.norm(transformed[:, 1] - transformed[:, 7], axis=1),
        atol=1e-12,
    )


def test_perfect_panoptic_alignment_scores_perfectly():
    points = np.column_stack((np.arange(12), np.arange(12) ** 2)).astype(float)
    labels = np.repeat(np.arange(3), 4)
    metrics = evaluate(points, points.copy(), labels)
    assert metrics["paired_rmse"] == 0.0
    assert metrics["paired_recall_at_1"] == 1.0
    assert metrics["paired_recall_at_5"] == 1.0
    assert metrics["three_group_accuracy_5nn"] == 1.0
