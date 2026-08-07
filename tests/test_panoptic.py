from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from datasets.panoptic import load_panoptic_camera_pose_pair  # noqa: E402


def _write_fixture(root: Path) -> None:
    pose_dir = root / "hdPose3d_stage1_coco19"
    pose_dir.mkdir(parents=True)
    rotation = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    calibration = {
        "cameras": [
            {"name": "00_00", "R": np.eye(3).tolist()},
            {"name": "00_01", "R": rotation},
        ]
    }
    (root / "calibration_fixture.json").write_text(json.dumps(calibration), encoding="utf-8")
    for frame in range(5):
        coordinates = np.arange(57, dtype=float).reshape(19, 3) + frame
        joints = np.column_stack([coordinates, np.ones(19)]).reshape(-1).tolist()
        payload = {"bodies": [{"id": 0, "joints19": joints}]}
        (pose_dir / f"body3DScene_{frame:08d}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_panoptic_adapter_hides_pairing_but_preserves_exact_camera_rotation(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    pair = load_panoptic_camera_pose_pair(tmp_path, shuffle_seed=7)
    np.testing.assert_allclose(
        pair.target_paired_truth,
        pair.source @ pair.feature_rotation_truth.T,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        pair.feature_rotation_truth.T @ pair.feature_rotation_truth,
        np.eye(pair.source.shape[1]),
        atol=1e-12,
    )
    assert not np.array_equal(pair.target_frame_ids, pair.source_frame_ids)
    assert pair.source.shape == pair.target.shape == (5, 57)


def test_panoptic_adapter_can_select_a_temporally_disjoint_frame_window(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    pair = load_panoptic_camera_pose_pair(tmp_path, minimum_frame_id=1, max_frames=4, shuffle_seed=8)
    assert pair.source.shape == (4, 57)
    np.testing.assert_array_equal(pair.source_frame_ids, np.array([1, 2, 3, 4]))
