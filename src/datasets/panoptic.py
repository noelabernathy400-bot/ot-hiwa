"""Leakage-safe Panoptic Studio pose-coordinate alignment adapter.

The official Panoptic sample provides reconstructed 3D skeletons in a common
world coordinate system and calibrated camera extrinsics.  This adapter forms
two *camera-coordinate* pose domains from the real skeletons.  Each skeleton
is centred before the coordinate change, so camera translation cancels and the
two flattened pose vectors differ by one block-diagonal orthogonal matrix.

Frame pairing and the calibration-derived transform are retained only for
post-fit scoring.  They never need to enter soft grouping or OT fitting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PanopticCameraPosePair:
    """Two shuffled unpaired camera-coordinate pose domains with hidden truth."""

    source: np.ndarray
    target: np.ndarray
    target_paired_truth: np.ndarray
    feature_rotation_truth: np.ndarray
    source_frame_ids: np.ndarray
    target_frame_ids: np.ndarray
    target_permutation: np.ndarray
    source_camera: str
    target_camera: str
    common_scale: float


def _camera_rotation(calibration: dict[str, object], name: str) -> np.ndarray:
    cameras = calibration.get("cameras")
    if not isinstance(cameras, list):
        raise ValueError("Panoptic calibration is missing its cameras list")
    matches = [camera for camera in cameras if isinstance(camera, dict) and camera.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"camera {name!r} was not found exactly once in calibration")
    rotation = np.asarray(matches[0].get("R"), dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError(f"camera {name!r} has an invalid rotation")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-5):
        raise ValueError(f"camera {name!r} rotation is not orthogonal within calibration tolerance")
    return rotation


def _choose_body(bodies: object, person_id: int | None) -> dict[str, object]:
    if not isinstance(bodies, list) or not bodies:
        raise ValueError("frame has no reconstructed bodies")
    candidates = [body for body in bodies if isinstance(body, dict)]
    if person_id is not None:
        candidates = [body for body in candidates if body.get("id") == person_id]
    if len(candidates) != 1:
        raise ValueError("frame does not contain exactly one requested body")
    return candidates[0]


def _root_centered_pose(body: dict[str, object], *, minimum_joint_confidence: float) -> np.ndarray:
    values = np.asarray(body.get("joints19"), dtype=float)
    if values.ndim != 1 or values.size != 19 * 4:
        raise ValueError("body must contain exactly 19 Panoptic COCO joints")
    joints = values.reshape(19, 4)
    coordinates = joints[:, :3]
    confidence = joints[:, 3]
    valid = np.isfinite(coordinates).all(axis=1) & np.isfinite(confidence) & (confidence >= minimum_joint_confidence)
    if valid.sum() < 4:
        raise ValueError("fewer than four confident 3D joints")
    weights = np.where(valid, confidence, 0.0)
    centre = (weights[:, None] * coordinates).sum(axis=0) / weights.sum()
    centered = coordinates - centre
    centered[~valid] = 0.0
    return centered


def _feature_rotation(rotation: np.ndarray, joints: int = 19) -> np.ndarray:
    return np.kron(np.eye(joints, dtype=float), rotation)


def load_panoptic_camera_pose_pair(
    raw_directory: str | Path,
    *,
    source_camera: str = "00_00",
    target_camera: str = "00_01",
    person_id: int | None = 0,
    minimum_joint_confidence: float = 0.2,
    max_frames: int | None = None,
    minimum_frame_id: int | None = None,
    shuffle_seed: int = 0,
) -> PanopticCameraPosePair:
    """Create a shuffled, unpaired two-camera pose problem from Panoptic data.

    ``raw_directory`` is the extracted official sequence directory containing
    ``calibration_*.json`` and ``hdPose3d_stage1_coco19/body3DScene_*.json``.
    Both domains are scaled by one common scalar, preserving the exact
    orthogonal relation.  The returned frame IDs and paired target array are
    evaluation-only fields.
    """
    root = Path(raw_directory)
    calibration_paths = sorted(root.glob("calibration_*.json"))
    pose_directory = root / "hdPose3d_stage1_coco19"
    frame_paths = sorted(pose_directory.glob("body3DScene_*.json"))
    if len(calibration_paths) != 1:
        raise ValueError("expected exactly one Panoptic calibration JSON")
    if not frame_paths:
        raise ValueError("no extracted hdPose3d_stage1_coco19 frame JSON files found")
    if source_camera == target_camera:
        raise ValueError("source_camera and target_camera must differ")
    if not 0 <= minimum_joint_confidence <= 1:
        raise ValueError("minimum_joint_confidence must lie in [0, 1]")
    if max_frames is not None and max_frames < 4:
        raise ValueError("max_frames must be at least four when specified")
    if minimum_frame_id is not None and minimum_frame_id < 0:
        raise ValueError("minimum_frame_id must be non-negative when specified")

    calibration = json.loads(calibration_paths[0].read_text(encoding="utf-8"))
    source_rotation = _camera_rotation(calibration, source_camera)
    target_rotation = _camera_rotation(calibration, target_camera)
    if minimum_frame_id is not None:
        frame_paths = [
            path
            for path in frame_paths
            if int(path.stem.rsplit("_", maxsplit=1)[-1]) >= minimum_frame_id
        ]
    selected = frame_paths if max_frames is None else frame_paths[:max_frames]

    poses: list[np.ndarray] = []
    frame_ids: list[int] = []
    for path in selected:
        try:
            frame = json.loads(path.read_text(encoding="utf-8"))
            centred = _root_centered_pose(
                _choose_body(frame.get("bodies"), person_id),
                minimum_joint_confidence=minimum_joint_confidence,
            )
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        poses.append(centred)
        frame_ids.append(int(path.stem.rsplit("_", maxsplit=1)[-1]))
    if len(poses) < 4:
        raise ValueError("fewer than four valid Panoptic poses after confidence filtering")

    world_centered = np.stack(poses, axis=0)
    source = (world_centered @ source_rotation.T).reshape(len(poses), -1)
    target_paired = (world_centered @ target_rotation.T).reshape(len(poses), -1)
    common_scale = float(np.sqrt(np.mean(source**2)))
    if not np.isfinite(common_scale) or common_scale <= 0:
        raise ValueError("Panoptic poses have invalid common scale")
    source /= common_scale
    target_paired /= common_scale

    rotation = target_rotation @ source_rotation.T
    feature_rotation = _feature_rotation(rotation)
    if not np.allclose(target_paired, source @ feature_rotation.T, atol=2e-5):
        raise RuntimeError("camera-coordinate pair violates the expected orthogonal relation")
    permutation = np.random.default_rng(shuffle_seed).permutation(len(poses))
    frame_array = np.asarray(frame_ids, dtype=int)
    return PanopticCameraPosePair(
        source=source,
        target=target_paired[permutation],
        target_paired_truth=target_paired,
        feature_rotation_truth=feature_rotation,
        source_frame_ids=frame_array,
        target_frame_ids=frame_array[permutation],
        target_permutation=permutation,
        source_camera=source_camera,
        target_camera=target_camera,
        common_scale=common_scale,
    )
