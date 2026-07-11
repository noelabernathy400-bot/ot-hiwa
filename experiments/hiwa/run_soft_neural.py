from __future__ import annotations

import argparse
import platform
import time

import numpy as np
import scipy
import sklearn
from sklearn.decomposition import FactorAnalysis
from sklearn.manifold import Isomap

from common import RESULTS_DIR, HiWA, ensure_output_dirs, nearest_neighbor_accuracy, original_r2, write_json
from run_neural import (
    least_squares_rotation,
    load_demo,
    movement_to_3d,
    remove_constant_columns,
)
from soft_groups import learn_soft_groups
from soft_hiwa import SoftHiWA


PROFILES = {
    "quick": dict(
        maxiter=12,
        tol=1e-2,
        mu=2e-2,
        shorn_maxiter=100,
        sa_maxiter=12,
        sa_shorn_maxiter=30,
    ),
    "pilot": dict(
        maxiter=80,
        tol=1e-1,
        mu=5e-3,
        shorn_maxiter=300,
        sa_maxiter=40,
        sa_shorn_maxiter=80,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--entropy-weight", type=float, default=0.05)
    parser.add_argument("--retain-mass", type=float, default=0.90)
    parser.add_argument("--max-support-factor", type=float, default=1.5)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("oracle_hard", "prototype_hard", "prototype_soft", "prototype_soft_warm"),
        default=("oracle_hard", "prototype_hard", "prototype_soft"),
    )
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def evaluate(
    aligned: np.ndarray,
    neural_3d: np.ndarray,
    movement_3d: np.ndarray,
    movement_xy: np.ndarray,
    neural_labels: np.ndarray,
    movement_labels: np.ndarray,
) -> dict[str, float]:
    return {
        "before_direction_accuracy": nearest_neighbor_accuracy(
            neural_3d,
            neural_labels,
            movement_3d,
            movement_labels,
        ),
        "after_direction_accuracy": nearest_neighbor_accuracy(
            aligned,
            neural_labels,
            movement_3d,
            movement_labels,
        ),
        "before_velocity_r2": original_r2(neural_3d[:, :2], movement_xy),
        "after_velocity_r2": original_r2(aligned[:, :2], movement_xy),
    }


def make_hiwa(profile: str) -> HiWA:
    return HiWA(
        dim_red_method=Isomap(n_components=2, n_neighbors=12),
        normalize=True,
        shorn_gamma=2e-1,
        sa_tol=1e-2,
        sa_shorn_gamma=1e-1,
        **PROFILES[profile],
    )


def run_hard(
    method: str,
    neural_3d: np.ndarray,
    neural_groups: np.ndarray,
    movement_3d: np.ndarray,
    movement_groups: np.ndarray,
    target_transform: np.ndarray,
    oracle_rotation: np.ndarray,
    seed: int,
    profile: str,
    evaluation_args: dict,
) -> tuple[dict, np.ndarray]:
    np.random.seed(seed)
    model = make_hiwa(profile)
    started = time.perf_counter()
    aligned = model.fit_transform(
        neural_3d,
        neural_groups,
        movement_3d,
        movement_groups,
        Y_transform=target_transform,
        Rgt=oracle_rotation,
    )
    elapsed = time.perf_counter() - started
    residuals = model.diagnostics["Rg_norm"]
    result = {
        "method": method,
        "seed": seed,
        **evaluate(aligned=aligned, neural_3d=neural_3d, movement_3d=movement_3d, **evaluation_args),
        "iterations": int(len(residuals)),
        "final_residual": float(residuals[-1]),
        "converged": bool(residuals[-1] <= PROFILES[profile]["tol"]),
        "elapsed_seconds": elapsed,
        "transport_P": model.P,
        "rotation_R": model.Rg,
    }
    return result, aligned


def run_soft(
    neural_3d: np.ndarray,
    neural_assignments: np.ndarray,
    movement_3d: np.ndarray,
    movement_assignments: np.ndarray,
    target_transform: np.ndarray,
    oracle_rotation: np.ndarray,
    seed: int,
    profile: str,
    retain_mass: float,
    max_support_factor: float,
    evaluation_args: dict,
    method: str = "prototype_soft",
    initial_rotation: np.ndarray | None = None,
    initial_transport: np.ndarray | None = None,
    warm_start_local: bool = False,
    rotation_anchor: np.ndarray | None = None,
    rotation_anchor_weight: float = 0.0,
    determinant_sign: int | None = None,
    source_transform: np.ndarray | None = None,
    representative_guidance_weight: float = 0.0,
    representative_rotation_weight: float = 0.0,
    component_conditioning_weight: float = 0.0,
) -> tuple[dict, np.ndarray]:
    model = SoftHiWA(
        dim_red_method=Isomap(n_components=2, n_neighbors=12),
        normalize=True,
        shorn_gamma=2e-1,
        sa_tol=1e-2,
        sa_shorn_gamma=1e-1,
        retain_mass=retain_mass,
        max_support_factor=max_support_factor,
        random_state=seed,
        warm_start_local=warm_start_local,
        rotation_anchor_weight=rotation_anchor_weight,
        determinant_sign=determinant_sign,
        representative_guidance_weight=representative_guidance_weight,
        representative_rotation_weight=representative_rotation_weight,
        component_conditioning_weight=component_conditioning_weight,
        **PROFILES[profile],
    )
    started = time.perf_counter()
    aligned = model.fit_transform(
        neural_3d,
        neural_assignments,
        movement_3d,
        movement_assignments,
        X_transform=source_transform,
        Y_transform=target_transform,
        Rgt=oracle_rotation,
        initial_rotation=initial_rotation,
        initial_transport=initial_transport,
        rotation_anchor=rotation_anchor,
    )
    elapsed = time.perf_counter() - started
    residuals = model.diagnostics["Rg_norm"]
    marginal_errors = model.diagnostics["max_sinkhorn_marginal_error"]
    result = {
        "method": method,
        "seed": seed,
        **evaluate(aligned=aligned, neural_3d=neural_3d, movement_3d=movement_3d, **evaluation_args),
        "iterations": int(len(residuals)),
        "final_residual": float(residuals[-1]),
        "converged": bool(residuals[-1] <= PROFILES[profile]["tol"]),
        "elapsed_seconds": elapsed,
        "max_sinkhorn_marginal_error": float(np.max(marginal_errors)),
        "source_support_sizes": model.diagnostics["source_support_sizes"],
        "target_support_sizes": model.diagnostics["target_support_sizes"],
        "source_retained_mass": model.diagnostics["source_retained_mass"],
        "target_retained_mass": model.diagnostics["target_retained_mass"],
        "transport_P": model.P,
        "rotation_R": model.Rg,
        "rotation_anchor_weight": rotation_anchor_weight,
        "rotation_anchor_distance": model.diagnostics["rotation_anchor_distance"],
        "determinant_sign_constraint": determinant_sign,
        "rotation_determinant": model.diagnostics["rotation_determinant"],
        "rotation_orthogonality_error": model.diagnostics["rotation_orthogonality_error"],
        "transport_objective": model.diagnostics["transport_objective"],
        "guided_transport_objective": model.diagnostics["guided_transport_objective"],
        "representative_guidance_weight": model.diagnostics[
            "representative_guidance_weight"
        ],
        "representative_rotation_weight": model.diagnostics[
            "representative_rotation_weight"
        ],
        "representative_rotation_cross_norm": model.diagnostics[
            "representative_rotation_cross_norm"
        ],
        "component_conditioning_weight": model.diagnostics[
            "component_conditioning_weight"
        ],
        "component_conditioning_cost_mean": model.diagnostics[
            "component_conditioning_cost_mean"
        ],
        "component_conditioning_cost_max": model.diagnostics[
            "component_conditioning_cost_max"
        ],
        "representative_cost_mean": model.diagnostics["representative_cost_mean"],
        "representative_cost_max": model.diagnostics["representative_cost_max"],
        "local_global_consensus_mean": model.diagnostics["local_global_consensus_mean"],
        "local_global_consensus_max": model.diagnostics["local_global_consensus_max"],
        "local_global_consensus_weighted_rms": model.diagnostics[
            "local_global_consensus_weighted_rms"
        ],
    }
    return result, aligned


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    data = load_demo()
    test_neural = remove_constant_columns(data["test_neural"])
    neural_3d = FactorAnalysis(n_components=3, random_state=0).fit_transform(test_neural)
    train_movement_3d = movement_to_3d(data["train_movement"])
    test_movement_3d = movement_to_3d(data["test_movement"])
    target_transform = np.linalg.pinv(train_movement_3d) @ data["train_movement"]
    oracle_rotation = least_squares_rotation(test_movement_3d, neural_3d)
    evaluation_args = {
        "movement_xy": data["test_movement"],
        "neural_labels": data["test_labels"],
        "movement_labels": data["train_labels"],
    }

    all_results: list[dict] = []
    prototype_records: list[dict] = []
    representative: dict[str, np.ndarray] | None = None

    for seed in args.seeds:
        neural_groups = learn_soft_groups(
            neural_3d,
            n_groups=args.groups,
            temperature=args.temperature,
            entropy_weight=args.entropy_weight,
            seed=seed,
        )
        movement_groups = learn_soft_groups(
            train_movement_3d,
            n_groups=args.groups,
            temperature=args.temperature,
            entropy_weight=args.entropy_weight,
            seed=seed,
        )
        prototype_records.append(
            {
                "seed": seed,
                "neural": neural_groups.diagnostics,
                "movement": movement_groups.diagnostics,
                "neural_prototypes": neural_groups.prototypes,
                "movement_prototypes": movement_groups.prototypes,
            }
        )

        seed_results: dict[str, dict] = {}
        aligned_results: dict[str, np.ndarray] = {}
        if "oracle_hard" in args.methods:
            seed_results["oracle_hard"], aligned_results["oracle_hard"] = run_hard(
                "oracle_hard",
                neural_3d,
                data["test_labels"],
                train_movement_3d,
                data["train_labels"],
                target_transform,
                oracle_rotation,
                seed,
                args.profile,
                evaluation_args,
            )
        if "prototype_hard" in args.methods:
            seed_results["prototype_hard"], aligned_results["prototype_hard"] = run_hard(
                "prototype_hard",
                neural_3d,
                np.argmax(neural_groups.assignments, axis=1),
                train_movement_3d,
                np.argmax(movement_groups.assignments, axis=1),
                target_transform,
                oracle_rotation,
                seed,
                args.profile,
                evaluation_args,
            )
        if "prototype_soft_warm" in args.methods and "prototype_hard" not in seed_results:
            seed_results["prototype_hard"], aligned_results["prototype_hard"] = run_hard(
                "prototype_hard",
                neural_3d,
                np.argmax(neural_groups.assignments, axis=1),
                train_movement_3d,
                np.argmax(movement_groups.assignments, axis=1),
                target_transform,
                oracle_rotation,
                seed,
                args.profile,
                evaluation_args,
            )
        if "prototype_soft" in args.methods:
            seed_results["prototype_soft"], aligned_results["prototype_soft"] = run_soft(
                neural_3d,
                neural_groups.assignments,
                train_movement_3d,
                movement_groups.assignments,
                target_transform,
                oracle_rotation,
                seed,
                args.profile,
                args.retain_mass,
                args.max_support_factor,
                evaluation_args,
            )
        if "prototype_soft_warm" in args.methods:
            hard_result = seed_results["prototype_hard"]
            seed_results["prototype_soft_warm"], aligned_results["prototype_soft_warm"] = run_soft(
                neural_3d,
                neural_groups.assignments,
                train_movement_3d,
                movement_groups.assignments,
                target_transform,
                oracle_rotation,
                seed,
                args.profile,
                args.retain_mass,
                args.max_support_factor,
                evaluation_args,
                method="prototype_soft_warm",
                initial_rotation=np.asarray(hard_result["rotation_R"]),
                initial_transport=np.asarray(hard_result["transport_P"]),
                warm_start_local=True,
            )
        all_results.extend(seed_results.values())
        if representative is None:
            representative = {
                "neural_3d": neural_3d,
                "movement_3d": train_movement_3d,
                "neural_assignments": neural_groups.assignments,
                "movement_assignments": movement_groups.assignments,
                "neural_prototypes": neural_groups.prototypes,
                "movement_prototypes": movement_groups.prototypes,
                "neural_labels": data["test_labels"],
                "movement_labels": data["train_labels"],
            }
            for method, aligned in aligned_results.items():
                representative[f"{method}_aligned"] = aligned

        summary = " ".join(
            f"{method}={result['after_direction_accuracy']:.3f}/R2={result['after_velocity_r2']:.3f}"
            for method, result in seed_results.items()
        )
        print(f"seed={seed} {summary}", flush=True)

    payload = {
        "experiment": "mihi_soft_group_pilot",
        "profile": args.profile,
        "parameters": {
            "groups": args.groups,
            "temperature": args.temperature,
            "entropy_weight": args.entropy_weight,
            "retain_mass": args.retain_mass,
            "max_support_factor": args.max_support_factor,
            "methods": list(args.methods),
            "profile_parameters": PROFILES[args.profile],
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "results": all_results,
        "prototype_diagnostics": prototype_records,
        "interpretation_scope": "exploratory paired-seed pilot; labels are used only by oracle_hard and evaluation",
    }
    suffix = f"_{args.tag}" if args.tag else ""
    output = RESULTS_DIR / f"soft_neural_{args.profile}{suffix}.json"
    write_json(output, payload)
    if representative is not None:
        np.savez_compressed(
            RESULTS_DIR / f"soft_neural_{args.profile}{suffix}_representative.npz",
            **representative,
        )
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
