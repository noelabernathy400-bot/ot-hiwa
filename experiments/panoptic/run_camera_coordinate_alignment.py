"""Frozen GC-HiWA benchmark for uncalibrated cross-view 3D pose coordinates.

This is a controlled real-data validation: Panoptic's reconstructed 3D
skeletons are expressed in two calibrated camera coordinate systems, then the
frame pairing and camera extrinsics are hidden from soft grouping and OT.
Camera calibration is used only after fitting to score rotation recovery.
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.decomposition import PCA


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT / "src", ROOT / "src" / "cc_hiwa"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import RESULTS_DIR, ensure_output_dirs, write_json  # noqa: E402
from datasets.panoptic import load_panoptic_camera_pose_pair  # noqa: E402
from alignment_qualification import (  # noqa: E402
    normalized_covariance_identifiability_margin,
    normalized_covariance_spectral_mismatch,
    qualify_alignment_fit,
    qualify_restart_stability,
)
from soft_groups import learn_soft_groups  # noqa: E402
from soft_hiwa import SoftHiWA  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-directory",
        type=Path,
        default=ROOT / "data" / "raw" / "panoptic" / "171204_pose1_sample",
    )
    parser.add_argument(
        "--min-covariance-identifiability-margin",
        type=float,
        default=0.02,
        help="minimum physical 3-D covariance eigenvalue-separation required to output a rotation",
    )
    parser.add_argument("--source-camera", default="00_00")
    parser.add_argument("--target-camera", default="00_01")
    parser.add_argument("--person-id", type=int, default=0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="maximum chronological pose JSON files to inspect; use a fixed window for tractable full-support OT",
    )
    parser.add_argument(
        "--minimum-frame-id",
        type=int,
        default=None,
        help="start a temporally disjoint fixed evaluation window at this frame ID",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[1301])
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.50)
    parser.add_argument("--entropy-weight", type=float, default=0.05)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=0.02)
    parser.add_argument("--mu", type=float, default=0.05)
    parser.add_argument("--local-gamma", type=float, default=0.40)
    parser.add_argument("--local-iterations", type=int, default=24)
    parser.add_argument("--local-sinkhorn-iterations", type=int, default=60)
    parser.add_argument("--tag", default="panoptic_camera_coordinate_alignment")
    parser.add_argument("--minimum-restarts", type=int, default=3)
    parser.add_argument("--max-restart-disagreement-degrees", type=float, default=5.0)
    parser.add_argument("--minimum-restart-consensus", type=float, default=0.8)
    parser.add_argument(
        "--max-covariance-spectral-mismatch",
        type=float,
        default=0.05,
        help="necessary global-spectrum compatibility gate for a single orthogonal coordinate map",
    )
    args = parser.parse_args()
    if args.groups < 2 or args.temperature <= 0 or args.local_gamma <= 0:
        parser.error("groups must exceed one; temperature and local-gamma must be positive")
    if args.max_frames is not None and args.max_frames < 4:
        parser.error("max-frames must be at least four when specified")
    if args.minimum_frame_id is not None and args.minimum_frame_id < 0:
        parser.error("minimum-frame-id must be non-negative when specified")
    if args.minimum_restarts < 2 or args.max_restart_disagreement_degrees <= 0:
        parser.error("restart count must be at least two and disagreement angle positive")
    if not 0 < args.minimum_restart_consensus <= 1:
        parser.error("minimum-restart-consensus must lie in (0, 1]")
    if args.max_covariance_spectral_mismatch <= 0:
        parser.error("max-covariance-spectral-mismatch must be positive")
    if args.min_covariance_identifiability_margin <= 0:
        parser.error("min-covariance-identifiability-margin must be positive")
    return args


def _recall_at_k(aligned: np.ndarray, target: np.ndarray, permutation: np.ndarray, k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    # Avoid materialising an n_source x n_target x d tensor: independent
    # Panoptic windows can contain hundreds of 57-D poses.
    distances = (
        np.sum(aligned**2, axis=1, keepdims=True)
        + np.sum(target**2, axis=1)[None, :]
        - 2.0 * (aligned @ target.T)
    )
    distances = np.maximum(distances, 0.0)
    ranked = np.argsort(distances, axis=1)[:, : min(k, target.shape[0])]
    target_position_for_source = np.argsort(permutation)
    return float(np.mean(np.any(ranked == target_position_for_source[:, None], axis=1)))


def _physical_coordinates(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] % 3:
        raise ValueError("flattened 3-D pose features must have a feature dimension divisible by three")
    return array.reshape(-1, 3)


def _fit_once(args: argparse.Namespace, seed: int) -> dict[str, object]:
    pair = load_panoptic_camera_pose_pair(
        args.raw_directory,
        source_camera=args.source_camera,
        target_camera=args.target_camera,
        person_id=args.person_id,
        max_frames=args.max_frames,
        minimum_frame_id=args.minimum_frame_id,
        shuffle_seed=seed,
    )
    source_groups = learn_soft_groups(
        pair.source,
        args.groups,
        args.temperature,
        args.entropy_weight,
        seed=seed,
        scaling_mode="global_scalar",
    )
    target_groups = learn_soft_groups(
        pair.target,
        args.groups,
        args.temperature,
        args.entropy_weight,
        seed=seed,
        scaling_mode="global_scalar",
    )
    model = SoftHiWA(
        dim_red_method=PCA(n_components=pair.source.shape[1]),
        normalize=False,
        maxiter=args.maxiter,
        tol=args.tol,
        mu=args.mu,
        shorn_maxiter=200,
        shorn_gamma=0.20,
        sa_maxiter=args.local_iterations,
        sa_tol=1e-2,
        sa_shorn_maxiter=args.local_sinkhorn_iterations,
        sa_shorn_gamma=args.local_gamma,
        support_mode="full",
        random_state=seed,
        warm_start_local=True,
        consensus_weighting="transport",
        inner_entropy_mode="fixed",
        determinant_sign=1,
        rotation_structure="repeated_3d_blocks",
    )
    aligned = model.fit_transform(
        pair.source,
        source_groups.assignments,
        pair.target,
        target_groups.assignments,
        X_transform=np.eye(pair.source.shape[1]),
        Y_transform=np.eye(pair.target.shape[1]),
    )
    diagnostics = model.diagnostics
    covariance_spectral_mismatch = normalized_covariance_spectral_mismatch(pair.source, pair.target)
    covariance_identifiability_margin = normalized_covariance_identifiability_margin(
        _physical_coordinates(pair.source)
    )
    qualification_diagnostics = {
        **diagnostics,
        "covariance_spectral_mismatch": covariance_spectral_mismatch,
        "covariance_identifiability_margin": covariance_identifiability_margin,
    }
    dimension = pair.source.shape[1]
    return {
        "seed": int(seed),
        "samples": int(pair.source.shape[0]),
        "feature_dimension": int(dimension),
        "source_camera": pair.source_camera,
        "target_camera": pair.target_camera,
        "source_frame_first": int(pair.source_frame_ids.min()),
        "source_frame_last": int(pair.source_frame_ids.max()),
        "common_scale": pair.common_scale,
        "rotation_frobenius_error": float(np.linalg.norm(model.Rg - pair.feature_rotation_truth, "fro")),
        "rotation_frobenius_error_per_dimension": float(np.linalg.norm(model.Rg - pair.feature_rotation_truth, "fro") / np.sqrt(dimension)),
        "paired_alignment_mse": float(np.mean((aligned - pair.target_paired_truth) ** 2)),
        "paired_recall_at_1": _recall_at_k(aligned, pair.target, pair.target_permutation, 1),
        "paired_recall_at_5": _recall_at_k(aligned, pair.target, pair.target_permutation, 5),
        "unaligned_recall_at_1": _recall_at_k(pair.source, pair.target, pair.target_permutation, 1),
        "unaligned_recall_at_5": _recall_at_k(pair.source, pair.target, pair.target_permutation, 5),
        "admm_converged": bool(diagnostics["admm_converged"]),
        "iterations": int(len(diagnostics["Rg_norm"])),
        "final_global_residual": float(diagnostics["Rg_norm"][-1]),
        "final_primal_residual": float(diagnostics["admm_primal_residual"][-1]),
        "final_local_marginal_error": float(diagnostics["max_sinkhorn_marginal_error"][-1]),
        "rotation_orthogonality_error": float(diagnostics["rotation_orthogonality_error"]),
        "estimated_physical_rotation_3x3": model.Rg[:3, :3].tolist(),
        "transport_objective": float(diagnostics["transport_objective"]),
        "relative_global_fit_ratio": float(diagnostics["relative_global_fit_ratio"]),
        "covariance_spectral_mismatch": float(covariance_spectral_mismatch),
        "covariance_identifiability_margin": float(covariance_identifiability_margin),
        "applicability_qualification": qualify_alignment_fit(
            qualification_diagnostics,
            max_covariance_spectral_mismatch=args.max_covariance_spectral_mismatch,
            min_covariance_identifiability_margin=args.min_covariance_identifiability_margin,
            required_rotation_structure="repeated_3d_blocks",
        ),
        "source_soft_group_diagnostics": source_groups.diagnostics,
        "target_soft_group_diagnostics": target_groups.diagnostics,
    }


def main() -> None:
    args = parse_args()
    records = []
    for seed in args.seeds:
        records.append(_fit_once(args, seed))
        print(f"completed seed={seed}", flush=True)
    restart_stability = qualify_restart_stability(
        [np.asarray(record["estimated_physical_rotation_3x3"], dtype=float) for record in records],
        [bool(record["applicability_qualification"]["accepted"]) for record in records],
        minimum_restarts=args.minimum_restarts,
        max_rotation_disagreement_degrees=args.max_restart_disagreement_degrees,
        min_consensus_fraction=args.minimum_restart_consensus,
    )
    payload = {
        "experiment": "panoptic_uncalibrated_cross_view_pose_coordinate_alignment",
        "scope": "controlled real-data coordinate-frame validation; camera extrinsics and frame pairing are withheld from fitting and used only for final scoring",
        "truth_usage": "calibration-derived relative rotation and synchronised frame identity are evaluation-only; no action labels, frame IDs, pair IDs, or camera extrinsics enter group learning or OT",
        "parameters": {**vars(args), "raw_directory": str(args.raw_directory)},
        "fixed_contract": {
            "support_mode": "full",
            "consensus_weighting": "transport",
            "inner_entropy_mode": "fixed",
            "soft_group_scaling_mode": "global_scalar (rotation-equivariant)",
            "determinant_component": "+1, because physical camera-coordinate changes are proper rotations",
            "rotation_structure": "I_19 kron R_3 (one physical camera rotation shared by all joints)",
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "records": records,
        "restart_stability_qualification": restart_stability,
    }
    ensure_output_dirs()
    path = RESULTS_DIR / f"{args.tag}.json"
    write_json(path, payload)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
