"""Coordinate sign-flip validation for the representative-orientation selector.

The representative-orientation selector should respond to a coordinate-system
reflection.  If one source coordinate is multiplied by -1, the valid orthogonal
component should flip sign while the selected candidate should keep high
direction accuracy and movement R².

This script is intentionally separate from the original confirmation run so the
original result files stay immutable and the sign-flip stress test has its own
traceable evidence chain.
"""

from __future__ import annotations

import argparse
import platform
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
from run_rotation_stabilized import (
    ANNEALING_PATH,
    _fixed_assignments,
    _load_problem,
    _run_chain,
)
from run_soft_neural import PROFILES
from soft_hiwa import _normal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(70, 80)))
    parser.add_argument("--axes", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--profile", choices=PROFILES, default="pilot")
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--entropy-weight", type=float, default=0.05)
    parser.add_argument("--retain-mass", type=float, default=0.90)
    parser.add_argument("--max-support-factor", type=float, default=1.5)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--include-unflipped-control", action="store_true")
    parser.add_argument("--tag", default="source_axes_seeds_70_79")
    args = parser.parse_args()
    if any(axis not in (0, 1, 2) for axis in args.axes):
        parser.error("this validation currently supports 3D axes 0, 1, and 2")
    return args


def _source_condition(
    neural_3d: np.ndarray,
    base_source_transform: np.ndarray,
    axis: int | None,
) -> tuple[str, np.ndarray, np.ndarray, int]:
    transformed = neural_3d.copy()
    if axis is None:
        return "source_unflipped_control", transformed, base_source_transform, -1
    transformed[:, axis] *= -1.0
    flip = np.eye(transformed.shape[1])
    flip[axis, axis] = -1.0
    transformed_source_transform = flip @ base_source_transform
    return f"source_axis_{axis}_flipped", transformed, transformed_source_transform, 1


def _source_transform(values: np.ndarray) -> np.ndarray:
    fit_values = _normal(values)
    return np.linalg.pinv(fit_values) @ Isomap(n_components=2, n_neighbors=12).fit_transform(
        fit_values
    )


def _select_records(final_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for record in final_records:
        if record["requested_determinant_sign"] == record["representative_orientation_sign"]:
            selected.append(record)
    return selected


def _summarize(final_records: list[dict[str, Any]]) -> dict[str, Any]:
    selected = _select_records(final_records)
    selected_acc = np.asarray([record["after_direction_accuracy"] for record in selected])
    selected_r2 = np.asarray([record["after_velocity_r2"] for record in selected])
    condition_summary = {}
    for condition in sorted({record["condition"] for record in final_records}):
        condition_records = [
            record for record in final_records if record["condition"] == condition
        ]
        condition_selected = _select_records(condition_records)
        acc = np.asarray([record["after_direction_accuracy"] for record in condition_selected])
        r2 = np.asarray([record["after_velocity_r2"] for record in condition_selected])
        expected = sorted({record["expected_selected_determinant_sign"] for record in condition_records})
        predicted = [record["representative_orientation_sign"] for record in condition_selected]
        condition_summary[condition] = {
            "expected_selected_determinant_signs": expected,
            "selected_determinant_signs": predicted,
            "selector_matches_expected_count": int(
                sum(
                    record["representative_orientation_sign"]
                    == record["expected_selected_determinant_sign"]
                    for record in condition_selected
                )
            ),
            "cases": len(condition_selected),
            "mean_direction_accuracy": float(acc.mean()),
            "minimum_direction_accuracy": float(acc.min()),
            "mean_movement_r2": float(r2.mean()),
            "minimum_movement_r2": float(r2.min()),
            "high_branch_count_accuracy_gt_0.5": int(np.sum(acc > 0.5)),
        }
    return {
        "cases": len(selected),
        "selector_matches_expected_count": int(
            sum(
                record["representative_orientation_sign"]
                == record["expected_selected_determinant_sign"]
                for record in selected
            )
        ),
        "mean_direction_accuracy": float(selected_acc.mean()),
        "minimum_direction_accuracy": float(selected_acc.min()),
        "sample_std_direction_accuracy": (
            float(selected_acc.std(ddof=1)) if len(selected_acc) > 1 else None
        ),
        "mean_movement_r2": float(selected_r2.mean()),
        "minimum_movement_r2": float(selected_r2.min()),
        "sample_std_movement_r2": (
            float(selected_r2.std(ddof=1)) if len(selected_r2) > 1 else None
        ),
        "high_branch_count_accuracy_gt_0.5": int(np.sum(selected_acc > 0.5)),
        "condition_summary": condition_summary,
    }


def _plot(final_records: list[dict[str, Any]], out_path) -> None:
    selected = _select_records(final_records)
    conditions = sorted({record["condition"] for record in selected})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), facecolor="white", constrained_layout=True)
    x = np.arange(len(conditions))
    acc_data = [
        [record["after_direction_accuracy"] for record in selected if record["condition"] == condition]
        for condition in conditions
    ]
    r2_data = [
        [record["after_velocity_r2"] for record in selected if record["condition"] == condition]
        for condition in conditions
    ]
    sign_data = [
        [record["representative_orientation_sign"] for record in selected if record["condition"] == condition]
        for condition in conditions
    ]
    axes[0].boxplot(acc_data, tick_labels=conditions, patch_artist=True)
    axes[1].boxplot(r2_data, tick_labels=conditions, patch_artist=True)
    axes[2].boxplot(sign_data, tick_labels=conditions, patch_artist=True)
    for axis, title, ylabel in [
        (axes[0], "Selected branch accuracy after source-axis reflection", "direction accuracy"),
        (axes[1], "Selected branch movement geometry", "R²"),
        (axes[2], "Representative selector determinant sign", "selected det sign"),
    ]:
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_facecolor("white")
        axis.grid(alpha=0.2)
        axis.tick_params(axis="x", rotation=25)
    axes[0].axhline(0.5, color="0.4", linestyle="--", linewidth=1.0)
    axes[2].set_yticks([-1, 1])
    fig.savefig(out_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    neural_3d, movement_3d, target_transform, oracle_rotation, evaluation_args = _load_problem()
    base_source_transform = _source_transform(neural_3d)
    axes: list[int | None] = list(args.axes)
    if args.include_unflipped_control:
        axes = [None] + axes

    hard_results = []
    stage_results = []
    final_records = []

    for seed in args.seeds:
        for axis in axes:
            condition, condition_neural_3d, condition_source_transform, expected_sign = (
                _source_condition(neural_3d, base_source_transform, axis)
            )
            print(f"\nSeed {seed} | {condition}")
            hard = _hard_baseline(
                condition_neural_3d,
                movement_3d,
                target_transform,
                oracle_rotation,
                evaluation_args,
                seed,
                args,
            )
            hard["condition"] = condition
            hard["flipped_source_axis"] = axis
            hard_results.append(hard)
            neural_assignments = _fixed_assignments(
                condition_neural_3d, args.groups, args.entropy_weight, seed
            )
            movement_assignments = _fixed_assignments(
                movement_3d, args.groups, args.entropy_weight, seed
            )

            seed_condition_final = []
            for sign in (-1, 1):
                method = f"component_{'negative' if sign == -1 else 'positive'}"
                stages = _run_chain(
                    method=method,
                    neural_3d=condition_neural_3d,
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
                    source_transform=condition_source_transform,
                )
                for record in stages:
                    record["condition"] = condition
                    record["flipped_source_axis"] = axis
                    record["expected_selected_determinant_sign"] = expected_sign
                final = stages[-1]
                seed_condition_final.append(final)
                stage_results.extend(stages)
                print(
                    f"  det={sign:+d} acc={final['after_direction_accuracy']:.4f} "
                    f"R2={final['after_velocity_r2']:.4f} "
                    f"objective={final['transport_objective']:.6f}"
                )

            representative_selector = _representative_orientation_selector(
                condition_neural_3d,
                neural_assignments[ANNEALING_PATH[-1]],
                movement_3d,
                movement_assignments[ANNEALING_PATH[-1]],
                [np.asarray(record["transport_P"]) for record in seed_condition_final],
            )
            for record in seed_condition_final:
                record.update(representative_selector)
                final_records.append(record)
            print(
                "  representative orientation selects "
                f"det={representative_selector['representative_orientation_sign']:+d}; "
                f"expected det={expected_sign:+d}"
            )

    suffix = f"_{args.tag}" if args.tag else ""
    raw_path = RESULTS_DIR / f"coordinate_flip_validation{suffix}.json"
    summary_path = RESULTS_DIR / f"coordinate_flip_validation_summary{suffix}.json"
    figure_path = FIGURES_DIR / f"coordinate_flip_validation{suffix}.png"
    payload = {
        "experiment": "coordinate_flip_validation",
        "status": "source_axis_reflection_stress_test",
        "label_usage": "labels are excluded from fitting and selector choice; used only after fitting for evaluation",
        "parameters": {
            "seeds": args.seeds,
            "source_axes": args.axes,
            "include_unflipped_control": args.include_unflipped_control,
            "profile": args.profile,
            "profile_parameters": PROFILES[args.profile],
            "annealing_path": ANNEALING_PATH,
            "determinant_signs": [-1, 1],
            "anchor_weight": args.anchor_weight,
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
        "final_records": final_records,
    }
    write_json(raw_path, payload)
    write_json(
        summary_path,
        {
            "experiment": payload["experiment"],
            "summary": _summarize(final_records),
        },
    )
    _plot(final_records, figure_path)
    print(f"\nSaved {raw_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
