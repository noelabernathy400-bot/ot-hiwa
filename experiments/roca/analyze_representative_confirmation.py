"""Analyze the frozen representative-orientation selector confirmation run."""

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
        default=RESULTS_DIR / "component_aware_soft_hiwa_representative_confirm_seeds_50_69.json",
    )
    parser.add_argument("--tag", default="representative_confirm_seeds_50_69")
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


def _paired_summary(
    selected: list[dict],
    hard: dict[int, dict],
    key: str,
    rng: np.random.Generator,
) -> dict:
    differences = np.asarray([
        record[key] - hard[int(record["seed"])][key] for record in selected
    ])
    indices = rng.integers(0, len(differences), size=(50_000, len(differences)))
    bootstrap_means = differences[indices].mean(axis=1)
    return {
        "differences": differences.tolist(),
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "bootstrap_95_ci": np.quantile(bootstrap_means, [0.025, 0.975]).tolist(),
        "positive_seed_count": int(np.sum(differences > 0)),
        "noninferiority_count_margin_minus_0.01": int(np.sum(differences >= -0.01)),
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    final = [record for record in payload["stage_results"] if record["temperature"] == 0.5]
    hard = {int(record["seed"]): record for record in payload["hard_baselines"]}
    seeds = sorted(hard)
    negative = []
    positive = []
    selected = []
    oracle = []
    rows = []
    for seed in seeds:
        candidates = [record for record in final if int(record["seed"]) == seed]
        neg = next(record for record in candidates if record["requested_determinant_sign"] == -1)
        pos = next(record for record in candidates if record["requested_determinant_sign"] == 1)
        predicted_sign = int(candidates[0]["representative_orientation_sign"])
        winner = next(
            record for record in candidates
            if int(record["requested_determinant_sign"]) == predicted_sign
        )
        best = max(candidates, key=lambda record: record["after_direction_accuracy"])
        negative.append(neg)
        positive.append(pos)
        selected.append(winner)
        oracle.append(best)
        rows.append(
            {
                "seed": seed,
                "representative_orientation_sign": predicted_sign,
                "representative_source_oriented_volume": candidates[0][
                    "representative_source_oriented_volume"
                ],
                "representative_target_oriented_volume": candidates[0][
                    "representative_target_oriented_volume"
                ],
                "representative_orientation_product": candidates[0][
                    "representative_orientation_product"
                ],
                "representative_target_order": candidates[0]["representative_target_order"],
                "negative_accuracy": neg["after_direction_accuracy"],
                "positive_accuracy": pos["after_direction_accuracy"],
                "selected_accuracy": winner["after_direction_accuracy"],
                "selected_r2": winner["after_velocity_r2"],
                "oracle_sign_for_diagnosis_only": best["requested_determinant_sign"],
                "selector_matches_oracle": bool(
                    winner["requested_determinant_sign"] == best["requested_determinant_sign"]
                ),
            }
        )

    selected_summary = _describe(selected)
    degenerate_count = int(
        sum(abs(row["representative_orientation_product"]) <= 1e-10 for row in rows)
    )
    checks = {
        "at_least_16_of_20_high_branch": (
            selected_summary["high_branch_count_accuracy_gt_0.5"] >= 16
        ),
        "mean_accuracy_at_least_0.55": selected_summary["mean_direction_accuracy"] >= 0.55,
        "mean_r2_at_least_0.59": selected_summary["mean_movement_r2"] >= 0.59,
        "no_catastrophic_r2": selected_summary["catastrophic_count_r2_lt_0"] == 0,
        "no_degenerate_representative_simplex": degenerate_count == 0,
    }
    rng = np.random.default_rng(20260706)
    output = {
        "experiment": "representative_orientation_confirmation_analysis",
        "source": str(args.input),
        "frozen_selector": "sign(det(X_rep) * det(Y_rep)) after Hungarian representative matching",
        "negative_candidate": _describe(negative),
        "positive_candidate": _describe(positive),
        "selected": selected_summary,
        "prototype_hard": _describe(list(hard.values())),
        "paired_selected_vs_prototype_hard": {
            "direction_accuracy": _paired_summary(
                selected, hard, "after_direction_accuracy", rng
            ),
            "movement_r2": _paired_summary(
                selected, hard, "after_velocity_r2", rng
            ),
        },
        "selector_oracle_agreement_count": int(sum(row["selector_matches_oracle"] for row in rows)),
        "label_using_oracle_not_a_valid_method": _describe(oracle),
        "degenerate_representative_count": degenerate_count,
        "pre_registered_success_checks": checks,
        "all_success_checks_pass": bool(all(checks.values())),
        "per_seed": rows,
    }
    write_json(
        RESULTS_DIR / f"representative_orientation_confirmation_analysis_{args.tag}.json",
        output,
    )

    x = np.arange(len(seeds))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor="white", constrained_layout=True)
    axes[0].plot(x, [record["after_direction_accuracy"] for record in negative], "o-", label="det(R) = −1")
    axes[0].plot(x, [record["after_direction_accuracy"] for record in positive], "s-", label="det(R) = +1")
    axes[0].set(title="Two constrained candidates", ylabel="direction accuracy")
    products = [row["representative_orientation_product"] for row in rows]
    colours = ["#2ca02c" if row["selector_matches_oracle"] else "#d62728" for row in rows]
    axes[1].bar(x, products, color=colours)
    axes[1].axhline(0, color="0.25", linewidth=1)
    axes[1].set(title="Representative oriented-volume product", ylabel="det(Xrep) · det(Yrep)")
    for axis in axes:
        axis.set_facecolor("white")
        axis.set_xticks(x, seeds, rotation=45)
        axis.set_xlabel("seed")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False)
    fig.savefig(
        FIGURES_DIR / f"representative_orientation_confirmation_{args.tag}.png",
        dpi=180,
        facecolor="white",
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
