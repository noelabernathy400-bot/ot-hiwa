"""Mechanism test for rotation-stabilized Soft-HiWA.

The experiment separates two interventions that were previously confounded:

1. hard-start continuation: initialize the first soft stage from prototype-hard;
2. rotation anchoring: add a chordal penalty around the preceding stage rotation.

For a consensus matrix A_bar, the anchored global update is

    R = Polar(A_bar + lambda_R * R_anchor),

which exactly minimizes the original chordal consensus objective plus
lambda_R / 2 * ||R - R_anchor||_F^2 over orthogonal R.
"""

from __future__ import annotations

import argparse
import platform
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
from sklearn.decomposition import FactorAnalysis

from common import FIGURES_DIR, RESULTS_DIR, ensure_output_dirs, write_json
from run_neural import (
    least_squares_rotation,
    load_demo,
    movement_to_3d,
    remove_constant_columns,
)
from run_soft_neural import PROFILES, run_hard, run_soft
from soft_groups import assignments_from_prototypes, learn_soft_groups


ANNEALING_PATH = [0.25, 0.35, 0.50]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--anchor-weights", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    parser.add_argument("--profile", choices=PROFILES, default="pilot")
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--entropy-weight", type=float, default=0.05)
    parser.add_argument("--retain-mass", type=float, default=0.90)
    parser.add_argument("--max-support-factor", type=float, default=1.5)
    parser.add_argument("--tag", default="seed7_screen")
    args = parser.parse_args()
    if any(weight < 0 for weight in args.anchor_weights):
        parser.error("anchor weights must be non-negative")
    return args


def _standardize(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    return (values - mean) / np.maximum(scale, 1e-12)


def _rotation_separation(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    chordal = float(np.linalg.norm(reference - candidate, "fro"))
    relative = reference.T @ candidate
    same_component = bool(np.linalg.det(relative) > 0)
    angle = None
    if reference.shape == (3, 3) and same_component:
        cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
        angle = float(np.degrees(np.arccos(cosine)))
    return {
        "chordal_distance": chordal,
        "same_orthogonal_component": same_component,
        "angle_degrees_if_proper": angle,
    }


def _load_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
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
    return neural_3d, train_movement_3d, target_transform, oracle_rotation, evaluation_args


def _fixed_assignments(
    values: np.ndarray,
    groups: int,
    entropy_weight: float,
    seed: int,
) -> dict[float, np.ndarray]:
    learned = learn_soft_groups(
        values,
        n_groups=groups,
        temperature=ANNEALING_PATH[0],
        entropy_weight=entropy_weight,
        seed=seed,
    )
    standardized = _standardize(values)
    return {
        tau: assignments_from_prototypes(
            standardized,
            learned.prototypes_standardized,
            tau,
        )
        for tau in ANNEALING_PATH
    }


def _run_chain(
    *,
    method: str,
    neural_3d: np.ndarray,
    movement_3d: np.ndarray,
    neural_assignments: dict[float, np.ndarray],
    movement_assignments: dict[float, np.ndarray],
    hard_result: dict,
    target_transform: np.ndarray,
    oracle_rotation: np.ndarray,
    evaluation_args: dict,
    seed: int,
    profile: str,
    retain_mass: float,
    max_support_factor: float,
    hard_start: bool,
    anchor_weight: float,
    determinant_sign: int | None = None,
    source_transform: np.ndarray | None = None,
    representative_guidance_weight: float = 0.0,
    representative_rotation_weight: float = 0.0,
    component_conditioning_weight: float = 0.0,
) -> list[dict]:
    previous_rotation = np.asarray(hard_result["rotation_R"]) if hard_start else None
    previous_transport = np.asarray(hard_result["transport_P"]) if hard_start else None
    stages: list[dict] = []

    for stage, tau in enumerate(ANNEALING_PATH):
        has_previous = previous_rotation is not None
        result, _ = run_soft(
            neural_3d=neural_3d,
            neural_assignments=neural_assignments[tau],
            movement_3d=movement_3d,
            movement_assignments=movement_assignments[tau],
            target_transform=target_transform,
            oracle_rotation=oracle_rotation,
            seed=seed,
            profile=profile,
            retain_mass=retain_mass,
            max_support_factor=max_support_factor,
            evaluation_args=evaluation_args,
            method=method,
            initial_rotation=previous_rotation,
            initial_transport=previous_transport,
            warm_start_local=has_previous,
            rotation_anchor=previous_rotation if has_previous and anchor_weight > 0 else None,
            rotation_anchor_weight=anchor_weight if has_previous else 0.0,
            determinant_sign=determinant_sign,
            source_transform=source_transform,
            representative_guidance_weight=representative_guidance_weight,
            representative_rotation_weight=representative_rotation_weight,
            component_conditioning_weight=component_conditioning_weight,
        )
        result["stage"] = stage
        result["temperature"] = tau
        result["hard_start"] = hard_start
        result["requested_anchor_weight"] = anchor_weight
        result["requested_determinant_sign"] = determinant_sign
        result["requested_representative_guidance_weight"] = representative_guidance_weight
        result["requested_representative_rotation_weight"] = representative_rotation_weight
        result["requested_component_conditioning_weight"] = component_conditioning_weight
        result["rotation_from_hard"] = _rotation_separation(
            np.asarray(hard_result["rotation_R"]),
            np.asarray(result["rotation_R"]),
        )
        if previous_rotation is not None:
            result["rotation_from_previous_stage"] = _rotation_separation(
                previous_rotation,
                np.asarray(result["rotation_R"]),
            )
        stages.append(result)
        previous_rotation = np.asarray(result["rotation_R"])
        previous_transport = np.asarray(result["transport_P"])
    return stages


def _summary(records: list[dict]) -> dict[str, Any]:
    final = [record for record in records if record.get("temperature") == ANNEALING_PATH[-1]]
    methods = sorted({record["method"] for record in final})
    output: dict[str, Any] = {}
    for method in methods:
        chosen = sorted((record for record in final if record["method"] == method), key=lambda x: x["seed"])
        acc = np.asarray([record["after_direction_accuracy"] for record in chosen])
        r2 = np.asarray([record["after_velocity_r2"] for record in chosen])
        angle = [record["rotation_from_hard"]["angle_degrees_if_proper"] for record in chosen]
        output[method] = {
            "seeds": [record["seed"] for record in chosen],
            "direction_accuracy": {
                "values": acc.tolist(),
                "mean": float(acc.mean()),
                "sample_std": float(acc.std(ddof=1)) if len(acc) > 1 else None,
            },
            "movement_r2": {
                "values": r2.tolist(),
                "mean": float(r2.mean()),
                "sample_std": float(r2.std(ddof=1)) if len(r2) > 1 else None,
            },
            "rotation_from_hard_degrees": angle,
            "all_converged": bool(all(record["converged"] for record in chosen)),
        }
    return output


def _plot(records: list[dict], baselines: list[dict], out_path: Path) -> None:
    final = [record for record in records if record.get("temperature") == ANNEALING_PATH[-1]]
    available = {record["method"] for record in final}
    preferred_order = [
        "pure_unanchored",
        "hard_start_only",
        "anchored_lambda_0.1",
        "anchored_lambda_1",
        "anchored_lambda_10",
    ]
    methods = [method for method in preferred_order if method in available]
    methods.extend(sorted(available - set(methods)))
    labels = {
        "pure_unanchored": "Unanchored",
        "hard_start_only": "Hard-start",
        "anchored_lambda_0.1": "Anchor 0.1",
        "anchored_lambda_1": "Anchor 1",
        "anchored_lambda_10": "Anchor 10",
    }
    seeds = sorted({record["seed"] for record in final})
    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.8), constrained_layout=True, facecolor="white"
    )
    x = np.arange(len(methods))
    for seed in seeds:
        selected = [next(record for record in final if record["seed"] == seed and record["method"] == method) for method in methods]
        axes[0].plot(x, [record["after_direction_accuracy"] for record in selected], marker="o", linewidth=1.2, alpha=0.75, label=f"seed {seed}")
        axes[1].plot(x, [record["after_velocity_r2"] for record in selected], marker="s", linewidth=1.2, alpha=0.75, label=f"seed {seed}")
    hard_acc = float(np.mean([record["after_direction_accuracy"] for record in baselines]))
    hard_r2 = float(np.mean([record["after_velocity_r2"] for record in baselines]))
    axes[0].axhline(hard_acc, color="0.3", linestyle="--", label=f"hard mean ({hard_acc:.3f})")
    axes[1].axhline(hard_r2, color="0.3", linestyle="--", label=f"hard mean ({hard_r2:.3f})")
    for axis, title, ylabel in [
        (axes[0], "Direction accuracy", "accuracy"),
        (axes[1], "Movement geometry", "R²"),
    ]:
        axis.set_facecolor("white")
        axis.set_xticks(x, [labels.get(method, method) for method in methods])
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, frameon=False)
    fig.savefig(out_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    neural_3d, movement_3d, target_transform, oracle_rotation, evaluation_args = _load_problem()
    all_records: list[dict] = []
    hard_baselines: list[dict] = []

    for seed in args.seeds:
        print(f"\nSeed {seed}")
        neural_fixed = _fixed_assignments(neural_3d, args.groups, args.entropy_weight, seed)
        movement_fixed = _fixed_assignments(movement_3d, args.groups, args.entropy_weight, seed)

        # Preserve the earlier baseline definition: hard groups come from tau=0.50 prototypes.
        neural_050 = learn_soft_groups(
            neural_3d,
            n_groups=args.groups,
            temperature=0.50,
            entropy_weight=args.entropy_weight,
            seed=seed,
        )
        movement_050 = learn_soft_groups(
            movement_3d,
            n_groups=args.groups,
            temperature=0.50,
            entropy_weight=args.entropy_weight,
            seed=seed,
        )
        hard, _ = run_hard(
            "prototype_hard",
            neural_3d,
            np.argmax(neural_050.assignments, axis=1),
            movement_3d,
            np.argmax(movement_050.assignments, axis=1),
            target_transform,
            oracle_rotation,
            seed,
            args.profile,
            evaluation_args,
        )
        hard_baselines.append(hard)
        print(
            f"  {'prototype_hard':22s} acc={hard['after_direction_accuracy']:.4f} "
            f"R2={hard['after_velocity_r2']:.4f}"
        )

        configurations = [("pure_unanchored", False, 0.0), ("hard_start_only", True, 0.0)]
        configurations.extend((f"anchored_lambda_{weight:g}", True, weight) for weight in args.anchor_weights)
        for method, hard_start, weight in configurations:
            stages = _run_chain(
                method=method,
                neural_3d=neural_3d,
                movement_3d=movement_3d,
                neural_assignments=neural_fixed,
                movement_assignments=movement_fixed,
                hard_result=hard,
                target_transform=target_transform,
                oracle_rotation=oracle_rotation,
                evaluation_args=evaluation_args,
                seed=seed,
                profile=args.profile,
                retain_mass=args.retain_mass,
                max_support_factor=args.max_support_factor,
                hard_start=hard_start,
                anchor_weight=weight,
            )
            all_records.extend(stages)
            final = stages[-1]
            angle = final["rotation_from_hard"]["angle_degrees_if_proper"]
            print(
                f"  {method:22s} acc={final['after_direction_accuracy']:.4f} "
                f"R2={final['after_velocity_r2']:.4f} angle_from_hard={angle}"
            )

    suffix = f"_{args.tag}" if args.tag else ""
    raw_path = RESULTS_DIR / f"rotation_stabilized_soft_hiwa{suffix}.json"
    summary_path = RESULTS_DIR / f"rotation_stabilized_soft_hiwa_summary{suffix}.json"
    figure_path = FIGURES_DIR / f"rotation_stabilized_soft_hiwa{suffix}.png"
    payload = {
        "experiment": "rotation_stabilized_soft_hiwa",
        "status": "development_mechanism_test",
        "parameters": {
            "seeds": args.seeds,
            "profile": args.profile,
            "profile_parameters": PROFILES[args.profile],
            "annealing_path": ANNEALING_PATH,
            "anchor_weights": args.anchor_weights,
            "groups": args.groups,
            "entropy_weight": args.entropy_weight,
            "retain_mass": args.retain_mass,
            "max_support_factor": args.max_support_factor,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "baseline_hard": hard_baselines,
        "stage_results": all_records,
    }
    write_json(raw_path, payload)
    write_json(summary_path, {"experiment": payload["experiment"], "summary": _summary(all_records)})
    _plot(all_records, hard_baselines, figure_path)
    print(f"\nSaved {raw_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
