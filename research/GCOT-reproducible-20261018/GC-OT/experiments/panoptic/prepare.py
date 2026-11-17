"""Create two camera-coordinate pose domains from Panoptic world skeletons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from .config import (
    FIRST_FRAME,
    FIXED_INITIALIZATION,
    LAST_FRAME,
    N_GROUPS,
    PCA_DIMENSION,
    PERSON_ID,
    SEQUENCE,
    SOURCE_CAMERA,
    TARGET_CAMERA,
    TSNE_INPUT_DIMENSION,
)


def _camera_map(calibration: dict) -> dict[str, dict]:
    return {camera["name"]: camera for camera in calibration["cameras"]}


def _load_world_poses(raw_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pose_dir = raw_dir / "hdPose3d_stage1_coco19"
    poses, confidences, frame_ids = [], [], []
    for frame_id in range(FIRST_FRAME, LAST_FRAME + 1):
        path = pose_dir / f"body3DScene_{frame_id:08d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Panoptic pose frame: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        body = next((item for item in payload["bodies"] if item["id"] == PERSON_ID), None)
        if body is None:
            raise ValueError(f"Person {PERSON_ID} is absent from frame {frame_id}")
        joints = np.asarray(body["joints19"], dtype=float).reshape(19, 4)
        # Panoptic sometimes marks occluded joints with confidence -1 while
        # retaining a reconstructed 3-D position. Keep all 199 synchronized frames.
        poses.append(joints[:, :3] - joints[2:3, :3])
        confidences.append(joints[:, 3])
        frame_ids.append(frame_id)
    return np.asarray(poses), np.asarray(confidences), np.asarray(frame_ids)


def _camera_coordinates(world_centered: np.ndarray, camera: dict) -> np.ndarray:
    rotation = np.asarray(camera["R"], dtype=float)
    return world_centered @ rotation.T


def _pca_domain(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flattened = values.reshape(len(values), -1)
    model = PCA(n_components=PCA_DIMENSION, svd_solver="full").fit(flattened)
    points = model.transform(flattened)
    rms = np.sqrt(np.mean(np.sum(points**2, axis=1)))
    return points / rms, model.explained_variance_ratio_


def _tsne_domain(values: np.ndarray) -> np.ndarray:
    flattened = values.reshape(len(values), -1)
    reduced = PCA(n_components=TSNE_INPUT_DIMENSION, svd_solver="full").fit_transform(flattened)
    return TSNE(
        n_components=2,
        perplexity=20,
        init="pca",
        learning_rate="auto",
        max_iter=1500,
        random_state=FIXED_INITIALIZATION,
    ).fit_transform(reduced)


def run(raw_dir: Path, output: Path) -> None:
    calibration_path = raw_dir / f"calibration_{SEQUENCE}.json"
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Missing calibration: {calibration_path}")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    cameras = _camera_map(calibration)
    for name in (SOURCE_CAMERA, TARGET_CAMERA):
        if name not in cameras:
            raise ValueError(f"Camera {name} not found in calibration")

    world, confidence, frame_ids = _load_world_poses(raw_dir)
    source_pose = _camera_coordinates(world, cameras[SOURCE_CAMERA])
    target_pose = _camera_coordinates(world, cameras[TARGET_CAMERA])
    source, source_variance = _pca_domain(source_pose)
    target, target_variance = _pca_domain(target_pose)
    source_tsne = _tsne_domain(source_pose)
    target_tsne = _tsne_domain(target_pose)

    # Three anonymous visual/evaluation groups are inferred only after feature
    # construction. They never enter any alignment call.
    labels = KMeans(
        n_clusters=N_GROUPS,
        n_init=20,
        random_state=FIXED_INITIALIZATION,
    ).fit_predict(source_tsne)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        source=source,
        target=target,
        source_tsne=source_tsne,
        target_tsne=target_tsne,
        frame_ids=frame_ids,
        labels=labels,
        confidence=confidence,
    )
    counts = {str(int(label)): int(np.sum(labels == label)) for label in np.unique(labels)}
    manifest = {
        "dataset": "CMU Panoptic Studio",
        "sequence": SEQUENCE,
        "source_camera": SOURCE_CAMERA,
        "target_camera": TARGET_CAMERA,
        "person_id": PERSON_ID,
        "frame_window_inclusive": [FIRST_FRAME, LAST_FRAME],
        "samples": int(len(frame_ids)),
        "world_pose": "19 COCO joints; body-centred at joint 2",
        "camera_transform": "X_camera = R_camera X_world; translation cancels after body centring",
        "alignment_input": "independent 2-D PCA coordinates, centred and RMS-normalized",
        "source_pca_explained_variance_ratio": source_variance.tolist(),
        "target_pca_explained_variance_ratio": target_variance.tolist(),
        "visual_groups": "KMeans(k=3) on source t-SNE; evaluation and colouring only",
        "group_counts": counts,
        "source_tsne_silhouette": float(silhouette_score(source_tsne, labels)),
        "labels_or_frame_pairing_used_for_alignment": False,
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.raw_dir, args.output)


if __name__ == "__main__":
    main()
