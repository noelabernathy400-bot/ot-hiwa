"""Controlled downstream transfer using a prequalified Panoptic alignment.

This is intentionally not an action-recognition claim. It evaluates the
mechanism needed by the intended application: a frozen classifier trained in a
source camera coordinate system should become usable on unlabelled target
camera poses after GC-HiWA alignment.

The labels are source-defined pose states: K-means is fit on chronological
source-training frames only. The corresponding target labels are retained
solely for scoring through hidden frame identity; they never enter alignment,
restart selection, state discovery, standardization, or classifier fitting.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT / "src", ROOT / "src" / "cc_hiwa"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import RESULTS_DIR, ensure_output_dirs, write_json  # noqa: E402
from datasets.panoptic import load_panoptic_camera_pose_pair  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-result", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--states", type=int, default=4)
    parser.add_argument("--tag", default="panoptic_source_pose_state_transfer")
    args = parser.parse_args()
    if not 0.2 <= args.train_fraction <= 0.8:
        parser.error("train-fraction must lie in [0.2, 0.8]")
    if args.states < 2:
        parser.error("states must be at least two")
    return args


def _load_selected_alignment(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment") not in {
        "panoptic_coordinate_alignment_restart_aggregation",
        "panoptic_uncalibrated_cross_view_pose_coordinate_alignment",
    }:
        raise ValueError("alignment-result must be a restart-qualified Panoptic alignment result")
    stability = payload.get("restart_stability_qualification")
    records = payload.get("records")
    if not isinstance(stability, dict) or not bool(stability.get("accepted")):
        raise ValueError("alignment result was not restart-stable; downstream transfer must abstain")
    if not isinstance(records, list):
        raise ValueError("alignment result has no records")
    selected_index = stability.get("selected_index")
    if not isinstance(selected_index, int) or not 0 <= selected_index < len(records):
        raise ValueError("alignment result has no valid selected record")
    record = records[selected_index]
    parameters = payload.get("parameters")
    if not isinstance(record, dict) or not isinstance(parameters, dict):
        raise ValueError("alignment result has invalid records or parameters")
    qualification = record.get("applicability_qualification")
    if not isinstance(qualification, dict) or qualification.get("accepted") is not True:
        raise ValueError("selected alignment failed its individual applicability qualification")
    return parameters, record


def _chronological_split(frame_ids: np.ndarray, train_fraction: float) -> np.ndarray:
    order = np.argsort(np.asarray(frame_ids, dtype=int))
    count = int(np.floor(train_fraction * order.size))
    if count < 2 or count >= order.size:
        raise ValueError("train fraction leaves an empty chronological split")
    mask = np.zeros(order.size, dtype=bool)
    mask[order[:count]] = True
    return mask


def _metrics(classifier: LogisticRegression, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    prediction = classifier.predict(features)
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
    }


def main() -> None:
    args = parse_args()
    parameters, selected = _load_selected_alignment(args.alignment_result)
    required = ("raw_directory", "source_camera", "target_camera", "max_frames")
    missing = [key for key in required if key not in parameters]
    if missing:
        raise ValueError(f"alignment result lacks {missing}")
    pair = load_panoptic_camera_pose_pair(
        ROOT / str(parameters["raw_directory"]),
        source_camera=str(parameters["source_camera"]),
        target_camera=str(parameters["target_camera"]),
        person_id=int(parameters.get("person_id", 0)),
        minimum_frame_id=(
            None
            if parameters.get("minimum_frame_id") is None
            else int(parameters["minimum_frame_id"])
        ),
        max_frames=None if parameters["max_frames"] is None else int(parameters["max_frames"]),
        shuffle_seed=int(selected["seed"]),
    )
    if pair.source.shape[0] != int(selected["samples"]):
        raise RuntimeError("reloaded Panoptic window differs from the qualified alignment result")
    rotation_3d = np.asarray(selected["estimated_physical_rotation_3x3"], dtype=float)
    feature_rotation = np.kron(np.eye(pair.source.shape[1] // 3), rotation_3d)
    if not np.allclose(feature_rotation.T @ feature_rotation, np.eye(pair.source.shape[1]), atol=1e-6):
        raise RuntimeError("selected physical rotation is not a valid repeated-block orthogonal map")

    source_train = _chronological_split(pair.source_frame_ids, args.train_fraction)
    source_state_model = KMeans(n_clusters=args.states, n_init=20, random_state=0)
    source_state_model.fit(pair.source[source_train])
    source_states = source_state_model.predict(pair.source)
    target_states_evaluation_only = source_states[pair.target_permutation]
    target_test = ~source_train[pair.target_permutation]
    if np.unique(source_states[source_train]).size != args.states:
        raise RuntimeError("source training split did not populate every pose state")
    if not np.any(target_test):
        raise RuntimeError("target test split is empty")

    scaler = StandardScaler().fit(pair.source[source_train])
    classifier = LogisticRegression(
        max_iter=2_000,
        C=10.0,
        random_state=0,
    )
    classifier.fit(scaler.transform(pair.source[source_train]), source_states[source_train])
    target_mapped_to_source = pair.target @ feature_rotation
    metrics = {
        "source_train": _metrics(classifier, scaler.transform(pair.source[source_train]), source_states[source_train]),
        "source_holdout": _metrics(classifier, scaler.transform(pair.source[~source_train]), source_states[~source_train]),
        "target_raw_coordinate": _metrics(classifier, scaler.transform(pair.target[target_test]), target_states_evaluation_only[target_test]),
        "target_after_gc_hiwa": _metrics(classifier, scaler.transform(target_mapped_to_source[target_test]), target_states_evaluation_only[target_test]),
    }
    metrics["aligned_minus_raw_accuracy"] = float(
        metrics["target_after_gc_hiwa"]["accuracy"] - metrics["target_raw_coordinate"]["accuracy"]
    )
    metrics["aligned_minus_raw_balanced_accuracy"] = float(
        metrics["target_after_gc_hiwa"]["balanced_accuracy"]
        - metrics["target_raw_coordinate"]["balanced_accuracy"]
    )
    payload = {
        "experiment": "panoptic_controlled_source_pose_state_transfer",
        "scope": "controlled downstream coordinate-transfer proxy, not a semantic action-recognition claim; source-defined pose-state labels fit only on chronological source-training frames",
        "leakage_policy": "target frame identity and paired source state are evaluation-only; no target state labels, frame IDs, pairing, camera calibration, or truth rotation enters alignment or classifier fitting",
        "alignment_result": str(args.alignment_result),
        "selected_alignment_seed": int(selected["seed"]),
        "alignment_restart_stable": True,
        "parameters": {
            "train_fraction": args.train_fraction,
            "states": args.states,
            "source_samples": int(source_train.sum()),
            "source_holdout_samples": int((~source_train).sum()),
            "target_holdout_samples": int(target_test.sum()),
            "source_frame_first": int(pair.source_frame_ids.min()),
            "source_frame_last": int(pair.source_frame_ids.max()),
            "source_camera": pair.source_camera,
            "target_camera": pair.target_camera,
        },
        "metrics": metrics,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    ensure_output_dirs()
    output_path = RESULTS_DIR / f"{args.tag}.json"
    write_json(output_path, payload)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
