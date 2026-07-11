"""Create traceable statistics and mechanism figures for the confirmation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from common import FIGURES_DIR, RESULTS_DIR, write_json
from run_rotation_stabilized import _plot


METHODS = ["prototype_hard", "pure_unanchored", "hard_start_only", "anchored_lambda_1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=RESULTS_DIR / "rotation_stabilized_soft_hiwa_confirm_seeds_10_29.json",
    )
    parser.add_argument("--tag", default="confirm_seeds_10_29")
    return parser.parse_args()


def _bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    indices = rng.integers(0, len(values), size=(50_000, len(values)))
    means = values[indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def _describe(records: list[dict]) -> dict[str, Any]:
    accuracy = np.asarray([record["after_direction_accuracy"] for record in records])
    r2 = np.asarray([record["after_velocity_r2"] for record in records])
    return {
        "n": len(records),
        "converged": int(sum(bool(record["converged"]) for record in records)),
        "direction_accuracy": {
            "mean": float(accuracy.mean()),
            "sample_std": float(accuracy.std(ddof=1)),
            "median": float(np.median(accuracy)),
            "high_branch_count_accuracy_gt_0.5": int(np.sum(accuracy > 0.5)),
        },
        "movement_r2": {
            "mean": float(r2.mean()),
            "sample_std": float(r2.std(ddof=1)),
            "median": float(np.median(r2)),
            "catastrophic_count_r2_lt_0": int(np.sum(r2 < 0)),
        },
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    hard = {int(record["seed"]): record for record in payload["baseline_hard"]}
    final = [record for record in payload["stage_results"] if record["temperature"] == 0.5]
    by_method = {"prototype_hard": sorted(hard.values(), key=lambda record: record["seed"])}
    for method in METHODS[1:]:
        by_method[method] = sorted(
            (record for record in final if record["method"] == method),
            key=lambda record: record["seed"],
        )

    rng = np.random.default_rng(20260706)
    statistics: dict[str, Any] = {
        "experiment": "rotation_stabilized_confirmation_analysis",
        "source": str(args.input),
        "method_summaries": {method: _describe(records) for method, records in by_method.items()},
        "paired_vs_prototype_hard": {},
        "orthogonal_component_diagnostic": {},
    }
    for method in METHODS[1:]:
        records = by_method[method]
        accuracy_diff = np.asarray([
            record["after_direction_accuracy"] - hard[int(record["seed"])]["after_direction_accuracy"]
            for record in records
        ])
        r2_diff = np.asarray([
            record["after_velocity_r2"] - hard[int(record["seed"])]["after_velocity_r2"]
            for record in records
        ])
        without_seed24 = np.asarray([record["seed"] != 24 for record in records])
        statistics["paired_vs_prototype_hard"][method] = {
            "direction_accuracy": {
                "mean_difference": float(accuracy_diff.mean()),
                "median_difference": float(np.median(accuracy_diff)),
                "bootstrap_95_ci": _bootstrap_ci(accuracy_diff, rng),
                "noninferiority_rate_margin_minus_0.01": float(np.mean(accuracy_diff >= -0.01)),
                "mean_difference_excluding_seed24": float(accuracy_diff[without_seed24].mean()),
            },
            "movement_r2": {
                "mean_difference": float(r2_diff.mean()),
                "median_difference": float(np.median(r2_diff)),
                "bootstrap_95_ci": _bootstrap_ci(r2_diff, rng),
                "positive_seed_count": int(np.sum(r2_diff > 0)),
                "mean_difference_excluding_seed24": float(r2_diff[without_seed24].mean()),
            },
        }

    component_points = []
    for method, records in by_method.items():
        for record in records:
            determinant_sign = int(np.sign(np.linalg.det(np.asarray(record["rotation_R"]))))
            component_points.append(
                {
                    "method": method,
                    "seed": int(record["seed"]),
                    "determinant_sign": determinant_sign,
                    "direction_accuracy": record["after_direction_accuracy"],
                }
            )
        statistics["orthogonal_component_diagnostic"][method] = {
            str(sign): {
                "n": len(chosen := [
                    point for point in component_points
                    if point["method"] == method and point["determinant_sign"] == sign
                ]),
                "mean_direction_accuracy": (
                    float(np.mean([point["direction_accuracy"] for point in chosen]))
                    if chosen else None
                ),
            }
            for sign in (-1, 1)
        }

    oracle_rows = []
    for seed in sorted(hard):
        candidates = [hard[seed]] + [record for record in final if int(record["seed"]) == seed]
        best = max(candidates, key=lambda record: record["after_direction_accuracy"])
        oracle_rows.append(float(best["after_direction_accuracy"]))
    statistics["label_using_oracle_upper_bound_not_a_valid_method"] = {
        "mean_direction_accuracy": float(np.mean(oracle_rows)),
        "minimum_direction_accuracy": float(np.min(oracle_rows)),
        "high_branch_count_accuracy_gt_0.5": int(np.sum(np.asarray(oracle_rows) > 0.5)),
    }

    stats_path = RESULTS_DIR / f"rotation_confirmation_statistics_{args.tag}.json"
    write_json(stats_path, statistics)
    _plot(
        payload["stage_results"],
        payload["baseline_hard"],
        FIGURES_DIR / f"rotation_confirmation_methods_{args.tag}.png",
    )

    fig, ax = plt.subplots(figsize=(8.5, 5), facecolor="white", constrained_layout=True)
    offsets = dict(zip(METHODS, np.linspace(-0.18, 0.18, len(METHODS))))
    for method in METHODS:
        points = [point for point in component_points if point["method"] == method]
        ax.scatter(
            [point["determinant_sign"] + offsets[method] for point in points],
            [point["direction_accuracy"] for point in points],
            alpha=0.75,
            label=method,
        )
    ax.set_xticks([-1, 1], ["det(R) = −1", "det(R) = +1"])
    ax.set_ylabel("direction accuracy")
    ax.set_title("Orthogonal component and direction-accuracy branch")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(
        FIGURES_DIR / f"rotation_component_accuracy_{args.tag}.png",
        dpi=180,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
