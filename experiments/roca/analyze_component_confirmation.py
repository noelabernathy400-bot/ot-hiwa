"""Analyze a frozen bidirectional selector on component-aware confirmation seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import FIGURES_DIR, RESULTS_DIR, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=RESULTS_DIR / "component_aware_soft_hiwa_confirm_seeds_30_49.json",
    )
    parser.add_argument("--tag", default="confirm_seeds_30_49")
    return parser.parse_args()


def _describe(records: list[dict]) -> dict:
    accuracy = np.asarray([record["after_direction_accuracy"] for record in records])
    r2 = np.asarray([record["after_velocity_r2"] for record in records])
    return {
        "n": len(records),
        "mean_direction_accuracy": float(accuracy.mean()),
        "sample_std_direction_accuracy": float(accuracy.std(ddof=1)),
        "minimum_direction_accuracy": float(accuracy.min()),
        "high_branch_count_accuracy_gt_0.5": int(np.sum(accuracy > 0.5)),
        "mean_movement_r2": float(r2.mean()),
        "sample_std_movement_r2": float(r2.std(ddof=1)),
        "minimum_movement_r2": float(r2.min()),
        "catastrophic_count_r2_lt_0": int(np.sum(r2 < 0)),
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    final = [record for record in payload["stage_results"] if record["temperature"] == 0.5]
    hard = {int(record["seed"]): record for record in payload["hard_baselines"]}
    seeds = sorted(hard)
    negative = [
        next(record for record in final if int(record["seed"]) == seed and record["requested_determinant_sign"] == -1)
        for seed in seeds
    ]
    positive = [
        next(record for record in final if int(record["seed"]) == seed and record["requested_determinant_sign"] == 1)
        for seed in seeds
    ]
    selected = []
    oracle = []
    rows = []
    for seed, neg, pos in zip(seeds, negative, positive):
        winner = min([neg, pos], key=lambda record: record["bidirectional_transport_objective"])
        best = max([neg, pos], key=lambda record: record["after_direction_accuracy"])
        selected.append(winner)
        oracle.append(best)
        rows.append(
            {
                "seed": seed,
                "negative_accuracy": neg["after_direction_accuracy"],
                "positive_accuracy": pos["after_direction_accuracy"],
                "negative_bidirectional_objective": neg["bidirectional_transport_objective"],
                "positive_bidirectional_objective": pos["bidirectional_transport_objective"],
                "objective_margin_positive_minus_negative": (
                    pos["bidirectional_transport_objective"]
                    - neg["bidirectional_transport_objective"]
                ),
                "selected_sign": winner["requested_determinant_sign"],
                "selected_accuracy": winner["after_direction_accuracy"],
                "selected_r2": winner["after_velocity_r2"],
                "oracle_sign_for_diagnosis_only": best["requested_determinant_sign"],
                "selector_matches_oracle": bool(
                    winner["requested_determinant_sign"] == best["requested_determinant_sign"]
                ),
            }
        )

    selected_summary = _describe(selected)
    success = {
        "at_least_16_of_20_high_branch": (
            selected_summary["high_branch_count_accuracy_gt_0.5"] >= 16
        ),
        "mean_accuracy_at_least_0.55": (
            selected_summary["mean_direction_accuracy"] >= 0.55
        ),
        "mean_r2_at_least_0.59": selected_summary["mean_movement_r2"] >= 0.59,
        "no_catastrophic_r2": selected_summary["catastrophic_count_r2_lt_0"] == 0,
    }
    output = {
        "experiment": "component_aware_confirmation_analysis",
        "source": str(args.input),
        "frozen_selector": "minimum bidirectional_transport_objective",
        "negative_candidate": _describe(negative),
        "positive_candidate": _describe(positive),
        "selected": selected_summary,
        "prototype_hard": _describe(list(hard.values())),
        "selector_oracle_agreement_count": int(sum(row["selector_matches_oracle"] for row in rows)),
        "label_using_oracle_not_a_valid_method": _describe(oracle),
        "pre_registered_success_checks": success,
        "all_success_checks_pass": bool(all(success.values())),
        "per_seed": rows,
    }
    stats_path = RESULTS_DIR / f"component_aware_confirmation_analysis_{args.tag}.json"
    write_json(stats_path, output)

    x = np.arange(len(seeds))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor="white", constrained_layout=True)
    axes[0].plot(x, [record["after_direction_accuracy"] for record in negative], "o-", label="det(R) = −1")
    axes[0].plot(x, [record["after_direction_accuracy"] for record in positive], "s-", label="det(R) = +1")
    axes[0].set(title="Two constrained candidates", ylabel="direction accuracy")
    margins = np.asarray([row["objective_margin_positive_minus_negative"] for row in rows])
    colours = ["#2ca02c" if row["selector_matches_oracle"] else "#d62728" for row in rows]
    axes[1].bar(x, margins, color=colours)
    axes[1].axhline(0, color="0.25", linewidth=1)
    axes[1].set(
        title="Frozen selector margin",
        ylabel="S(+1) − S(−1)\n(positive selects det=-1)",
    )
    for axis in axes:
        axis.set_facecolor("white")
        axis.set_xticks(x, seeds, rotation=45)
        axis.set_xlabel("seed")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    fig.savefig(
        FIGURES_DIR / f"component_aware_confirmation_{args.tag}.png",
        dpi=180,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
