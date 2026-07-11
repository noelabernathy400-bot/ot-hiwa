"""Representative-guided transport plus ROCA branch selection.

This experiment keeps ROCA as the label-free determinant selector and adds
TACO-style representative terms inside Soft-HiWA. `representative_guidance_weight`
lets soft representatives control the group-level transport cost, while
`representative_rotation_weight` lets matched representatives directly enter
the global orthogonal rotation update. `component_conditioning_weight` adds a
TACO-style soft component compatibility penalty to the existing local sample OT
cost. Setting all weights to 0 recovers the previous ROCA/Soft-HiWA pipeline.
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
from sklearn.manifold import Isomap

from common import FIGURES_DIR, RESULTS_DIR, ensure_output_dirs, write_json
from run_component_aware import (
    _hard_baseline,
    _representative_orientation_selector,
)
from run_rotation_stabilized import ANNEALING_PATH, _fixed_assignments, _load_problem, _run_chain
from run_soft_neural import PROFILES
from soft_hiwa import _normal


RESULT_DIR = RESULTS_DIR / "rgca_roca"
FIGURE_DIR = FIGURES_DIR / "rgca_roca"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[50, 51])
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--entropy-weight", type=float, default=0.05)
    parser.add_argument("--retain-mass", type=float, default=0.90)
    parser.add_argument("--max-support-factor", type=float, default=1.5)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument(
        "--guidance-weights",
        nargs="+",
        type=float,
        default=[0.0, 0.25],
        help="Representative guidance lambda values. 0.0 is the ROCA baseline.",
    )
    parser.add_argument(
        "--rotation-weights",
        nargs="+",
        type=float,
        default=[0.0],
        help="Representative-to-rotation lambda values. 0.0 disables this term.",
    )
    parser.add_argument(
        "--component-weights",
        nargs="+",
        type=float,
        default=[0.0],
        help="Soft component-conditioned sample OT beta values. 0.0 disables this term.",
    )
    parser.add_argument("--tag", default="smoke")
    return parser.parse_args()


def _transport_entropy(transport: np.ndarray) -> float:
    values = np.asarray(transport, dtype=float)
    positive = values[values > 0]
    if positive.size == 0:
        return 0.0
    return float(-np.sum(positive * np.log(positive)))


def _source_transform(values: np.ndarray) -> np.ndarray:
    fit_values = _normal(values)
    return np.linalg.pinv(fit_values) @ Isomap(n_components=2, n_neighbors=12).fit_transform(
        fit_values
    )


def _summarize(final_records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    weight_pairs = sorted(
        {
            (
                float(record["representative_guidance_weight"]),
                float(record["representative_rotation_weight"]),
                float(record["component_conditioning_weight"]),
            )
            for record in final_records
        }
    )
    for guidance_weight, rotation_weight, component_weight in weight_pairs:
        summary_key = (
            f"transport_{guidance_weight:g}__rotation_{rotation_weight:g}"
            f"__component_{component_weight:g}"
        )
        weight_records = [
            record for record in final_records
            if float(record["representative_guidance_weight"]) == guidance_weight
            and float(record["representative_rotation_weight"]) == rotation_weight
            and float(record["component_conditioning_weight"]) == component_weight
        ]
        seeds = sorted({int(record["seed"]) for record in weight_records})
        chosen = []
        branch_pairs = []
        for seed in seeds:
            candidates = [record for record in weight_records if int(record["seed"]) == seed]
            predicted_sign = int(candidates[0]["representative_orientation_sign"])
            selected = next(
                record for record in candidates
                if int(record["requested_determinant_sign"]) == predicted_sign
            )
            chosen.append(selected)
            high = max(candidates, key=lambda record: record["after_direction_accuracy"])
            branch_pairs.append(
                {
                    "seed": seed,
                    "selected_sign": selected["requested_determinant_sign"],
                    "high_accuracy_sign": high["requested_determinant_sign"],
                    "selected_is_high_accuracy_branch": bool(
                        selected["requested_determinant_sign"]
                        == high["requested_determinant_sign"]
                    ),
                    "selected_accuracy": selected["after_direction_accuracy"],
                    "high_branch_accuracy": high["after_direction_accuracy"],
                }
            )
        accuracy = np.asarray([record["after_direction_accuracy"] for record in chosen])
        r2 = np.asarray([record["after_velocity_r2"] for record in chosen])
        entropy = np.asarray([record["transport_entropy"] for record in chosen])
        consensus = np.asarray(
            [record["local_global_consensus_weighted_rms"] for record in chosen]
        )
        output[summary_key] = {
            "representative_guidance_weight": guidance_weight,
            "representative_rotation_weight": rotation_weight,
            "component_conditioning_weight": component_weight,
            "seeds": seeds,
            "selected_methods": [record["method"] for record in chosen],
            "selected_determinant_signs": [
                int(record["requested_determinant_sign"]) for record in chosen
            ],
            "selected_branch_matches_high_accuracy_count": int(
                sum(row["selected_is_high_accuracy_branch"] for row in branch_pairs)
            ),
            "n_seeds": len(seeds),
            "mean_direction_accuracy": float(accuracy.mean()),
            "min_direction_accuracy": float(accuracy.min()),
            "mean_movement_r2": float(r2.mean()),
            "min_movement_r2": float(r2.min()),
            "mean_transport_entropy": float(entropy.mean()),
            "mean_local_global_consensus_weighted_rms": float(consensus.mean()),
            "branch_pairs": branch_pairs,
        }
    baseline_key = "transport_0__rotation_0"
    if baseline_key in output:
        baseline = output[baseline_key]
        for key, value in output.items():
            value["delta_mean_direction_accuracy_vs_lambda0"] = (
                value["mean_direction_accuracy"] - baseline["mean_direction_accuracy"]
            )
            value["delta_mean_movement_r2_vs_lambda0"] = (
                value["mean_movement_r2"] - baseline["mean_movement_r2"]
            )
    return output


def _plot(summary: dict[str, Any], out_path: Path) -> None:
    weights = sorted(float(key) for key in summary)
    labels = [str(weight) for weight in weights]
    acc = [summary[str(weight)]["mean_direction_accuracy"] for weight in weights]
    r2 = [summary[str(weight)]["mean_movement_r2"] for weight in weights]
    entropy = [summary[str(weight)]["mean_transport_entropy"] for weight in weights]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor="white", constrained_layout=True)
    axes[0].plot(labels, acc, marker="o")
    axes[0].set(title="ROCA-selected direction accuracy", xlabel="RGCA λ", ylabel="accuracy")
    axes[1].plot(labels, r2, marker="o", color="#2ca02c")
    axes[1].set(title="ROCA-selected movement R²", xlabel="RGCA λ", ylabel="R²")
    axes[2].plot(labels, entropy, marker="o", color="#9467bd")
    axes[2].set(title="Selected group transport entropy", xlabel="RGCA λ", ylabel="entropy")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_facecolor("white")
    fig.savefig(out_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _plot(summary: dict[str, Any], out_path: Path) -> None:
    """Plot paired transport/rotation representative weights.

    This definition intentionally replaces the earlier single-lambda plotting
    helper while keeping the same function name for the script entry point.
    """
    keys = sorted(summary)
    labels = [
        f"T={summary[key]['representative_guidance_weight']:g}\n"
        f"R={summary[key]['representative_rotation_weight']:g}\n"
        f"C={summary[key]['component_conditioning_weight']:g}"
        for key in keys
    ]
    acc = [summary[key]["mean_direction_accuracy"] for key in keys]
    r2 = [summary[key]["mean_movement_r2"] for key in keys]
    entropy = [summary[key]["mean_transport_entropy"] for key in keys]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor="white", constrained_layout=True)
    axes[0].plot(labels, acc, marker="o")
    axes[0].set(
        title="ROCA-selected direction accuracy",
        xlabel="representative/component weights",
        ylabel="accuracy",
    )
    axes[1].plot(labels, r2, marker="o", color="#2ca02c")
    axes[1].set(
        title="ROCA-selected movement R2",
        xlabel="representative/component weights",
        ylabel="R2",
    )
    axes[2].plot(labels, entropy, marker="o", color="#9467bd")
    axes[2].set(
        title="Selected group transport entropy",
        xlabel="representative/component weights",
        ylabel="entropy",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_facecolor("white")
    fig.savefig(out_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    neural_3d, movement_3d, target_transform, oracle_rotation, evaluation_args = _load_problem()
    source_transform = _source_transform(neural_3d)

    hard_results: list[dict[str, Any]] = []
    stage_results: list[dict[str, Any]] = []

    for seed in args.seeds:
        print(f"\nSeed {seed}")
        hard = _hard_baseline(
            neural_3d,
            movement_3d,
            target_transform,
            oracle_rotation,
            evaluation_args,
            seed,
            args,
        )
        hard_results.append(hard)
        neural_assignments = _fixed_assignments(
            neural_3d, args.groups, args.entropy_weight, seed
        )
        movement_assignments = _fixed_assignments(
            movement_3d, args.groups, args.entropy_weight, seed
        )

        for weight in args.guidance_weights:
            for rotation_weight in args.rotation_weights:
                for component_weight in args.component_weights:
                    seed_final_records: list[dict[str, Any]] = []
                    for sign in (-1, 1):
                        method = (
                            f"rgca_roca_transport_{weight:g}"
                            f"_rotation_{rotation_weight:g}"
                            f"_component_{component_weight:g}_det_{sign:+d}"
                        )
                        stages = _run_chain(
                            method=method,
                            neural_3d=neural_3d,
                            movement_3d=movement_3d,
                            neural_assignments=neural_assignments,
                            movement_assignments=movement_assignments,
                            hard_result=hard,
                            target_transform=target_transform,
                            oracle_rotation=oracle_rotation,
                            evaluation_args=evaluation_args,
                            seed=seed,
                            profile=args.profile,
                            retain_mass=args.retain_mass,
                            max_support_factor=args.max_support_factor,
                            hard_start=True,
                            anchor_weight=args.anchor_weight,
                            determinant_sign=sign,
                            source_transform=source_transform,
                            representative_guidance_weight=weight,
                            representative_rotation_weight=rotation_weight,
                            component_conditioning_weight=component_weight,
                        )
                        final = stages[-1]
                        final["transport_entropy"] = _transport_entropy(
                            np.asarray(final["transport_P"])
                        )
                        seed_final_records.append(final)
                        stage_results.extend(stages)
                        print(
                            f"  transport_lambda={weight:g} "
                            f"rotation_lambda={rotation_weight:g} "
                            f"component_beta={component_weight:g} det={sign:+d} "
                            f"acc={final['after_direction_accuracy']:.4f} "
                            f"R2={final['after_velocity_r2']:.4f} "
                            f"obj={final['transport_objective']:.6f} "
                            f"guided_obj={final['guided_transport_objective']:.6f}"
                        )

                    representative_selector = _representative_orientation_selector(
                        neural_3d,
                        neural_assignments[ANNEALING_PATH[-1]],
                        movement_3d,
                        movement_assignments[ANNEALING_PATH[-1]],
                        [np.asarray(record["transport_P"]) for record in seed_final_records],
                    )
                    for record in seed_final_records:
                        record.update(representative_selector)
                    print(
                        f"  transport_lambda={weight:g} rotation_lambda={rotation_weight:g} "
                        f"component_beta={component_weight:g} ROCA selects "
                        f"det={representative_selector['representative_orientation_sign']:+d}"
                    )

    final_records = [
        record for record in stage_results
        if record["temperature"] == ANNEALING_PATH[-1]
    ]
    summary = _summarize(final_records)
    suffix = f"_{args.tag}" if args.tag else ""
    raw_path = RESULT_DIR / f"rgca_roca_raw{suffix}.json"
    summary_path = RESULT_DIR / f"rgca_roca_summary{suffix}.json"
    figure_path = FIGURE_DIR / f"rgca_roca_summary{suffix}.png"
    payload = {
        "experiment": "rgca_roca",
        "status": "controlled_development",
        "label_usage": (
            "labels are excluded from fitting and ROCA branch selection; labels are used "
            "only after selection for accuracy/R2 evaluation"
        ),
        "parameters": {
            "seeds": args.seeds,
            "profile": args.profile,
            "profile_parameters": PROFILES[args.profile],
            "annealing_path": ANNEALING_PATH,
            "determinant_signs": [-1, 1],
            "representative_guidance_weights": args.guidance_weights,
            "representative_rotation_weights": args.rotation_weights,
            "component_conditioning_weights": args.component_weights,
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
        "hard_baselines": hard_results,
        "stage_results": stage_results,
    }
    write_json(raw_path, payload)
    write_json(
        summary_path,
        {
            "experiment": "rgca_roca",
            "summary": summary,
        },
    )
    _plot(summary, figure_path)
    print(f"\nSaved {raw_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
