"""Aggregate independently persisted Panoptic restarts without refitting.

Long full-support OT jobs may be scheduled one seed at a time.  This utility
validates that those files share the same data and solver contract, then applies
the same label-free restart-stability gate used by the main runner.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT / "src", ROOT / "src" / "cc_hiwa"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from alignment_qualification import qualify_restart_stability  # noqa: E402
from common import RESULTS_DIR, ensure_output_dirs, write_json  # noqa: E402


COMPARABLE_PARAMETER_KEYS = (
    "raw_directory",
    "source_camera",
    "target_camera",
    "person_id",
    "minimum_frame_id",
    "max_frames",
    "groups",
    "temperature",
    "entropy_weight",
    "maxiter",
    "tol",
    "mu",
    "local_gamma",
    "local_iterations",
    "local_sinkhorn_iterations",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--minimum-restarts", type=int, default=3)
    parser.add_argument("--max-restart-disagreement-degrees", type=float, default=5.0)
    parser.add_argument("--minimum-restart-consensus", type=float, default=0.8)
    parser.add_argument("--tag", required=True)
    return parser.parse_args()


def _load_single(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
        raise ValueError(f"{path} must contain exactly one Panoptic record")
    if payload.get("experiment") != "panoptic_uncalibrated_cross_view_pose_coordinate_alignment":
        raise ValueError(f"{path} is not a Panoptic coordinate-alignment result")
    return payload, records[0]


def main() -> None:
    args = parse_args()
    loaded = [_load_single(path) for path in args.inputs]
    reference_parameters = loaded[0][0]["parameters"]
    if not isinstance(reference_parameters, dict):
        raise ValueError("result parameters must be an object")
    for path, (payload, _) in zip(args.inputs[1:], loaded[1:], strict=True):
        parameters = payload.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError(f"{path} has invalid parameters")
        differences = [
            key
            for key in COMPARABLE_PARAMETER_KEYS
            if parameters.get(key) != reference_parameters.get(key)
        ]
        if differences:
            raise ValueError(f"{path} differs from the first input on {differences}")
    records = [record for _, record in loaded]
    stability = qualify_restart_stability(
        [np.asarray(record["estimated_physical_rotation_3x3"], dtype=float) for record in records],
        [bool(record["applicability_qualification"]["accepted"]) for record in records],
        minimum_restarts=args.minimum_restarts,
        max_rotation_disagreement_degrees=args.max_restart_disagreement_degrees,
        min_consensus_fraction=args.minimum_restart_consensus,
    )
    selected_index = stability["selected_index"]
    selected = records[selected_index] if selected_index is not None else None
    summary = {
        "restart_count": len(records),
        "individually_accepted_count": int(
            sum(bool(record["applicability_qualification"]["accepted"]) for record in records)
        ),
        "admm_converged_count": int(sum(bool(record["admm_converged"]) for record in records)),
        "rotation_frobenius_error_mean_evaluation_only": float(
            np.mean([float(record["rotation_frobenius_error"]) for record in records])
        ),
        "rotation_frobenius_error_max_evaluation_only": float(
            np.max([float(record["rotation_frobenius_error"]) for record in records])
        ),
        "selected_seed": None if selected is None else int(selected["seed"]),
        "selected_rotation_frobenius_error_evaluation_only": (
            None if selected is None else float(selected["rotation_frobenius_error"])
        ),
        "selected_paired_recall_at_1_evaluation_only": (
            None if selected is None else float(selected["paired_recall_at_1"])
        ),
    }
    output = {
        "experiment": "panoptic_coordinate_alignment_restart_aggregation",
        "scope": "post-fit aggregation of independently persisted unlabelled restarts; truth-based metrics remain evaluation-only",
        "input_files": [str(path) for path in args.inputs],
        "parameters": {key: reference_parameters.get(key) for key in COMPARABLE_PARAMETER_KEYS},
        "records": records,
        "restart_stability_qualification": stability,
        "summary": summary,
    }
    ensure_output_dirs()
    output_path = RESULTS_DIR / f"{args.tag}.json"
    write_json(output_path, output)
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
