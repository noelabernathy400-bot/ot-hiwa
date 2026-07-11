"""Aggregate coordinate sign-flip validation batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from common import FIGURES_DIR, RESULTS_DIR, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--tag", default="source_axes_seeds_70_79")
    return parser.parse_args()


def _selected(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record["requested_determinant_sign"] == record["representative_orientation_sign"]
    ]


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    selected = _selected(records)
    acc = np.asarray([record["after_direction_accuracy"] for record in selected])
    r2 = np.asarray([record["after_velocity_r2"] for record in selected])
    high_candidates = [
        max(
            [
                candidate
                for candidate in records
                if candidate["seed"] == record["seed"]
                and candidate["condition"] == record["condition"]
            ],
            key=lambda candidate: candidate["after_direction_accuracy"],
        )
        for record in selected
    ]
    condition_summary: dict[str, Any] = {}
    for condition in sorted({record["condition"] for record in records}):
        condition_selected = [
            record for record in selected if record["condition"] == condition
        ]
        condition_acc = np.asarray(
            [record["after_direction_accuracy"] for record in condition_selected]
        )
        condition_r2 = np.asarray(
            [record["after_velocity_r2"] for record in condition_selected]
        )
        condition_summary[condition] = {
            "cases": len(condition_selected),
            "selector_matches_expected_count": int(
                sum(
                    record["representative_orientation_sign"]
                    == record["expected_selected_determinant_sign"]
                    for record in condition_selected
                )
            ),
            "selected_determinant_signs": [
                int(record["representative_orientation_sign"])
                for record in condition_selected
            ],
            "mean_direction_accuracy": float(condition_acc.mean()),
            "minimum_direction_accuracy": float(condition_acc.min()),
            "mean_movement_r2": float(condition_r2.mean()),
            "minimum_movement_r2": float(condition_r2.min()),
            "high_branch_count_accuracy_gt_0.5": int(np.sum(condition_acc > 0.5)),
        }
    return {
        "cases": len(selected),
        "seeds": sorted({int(record["seed"]) for record in selected}),
        "conditions": sorted({record["condition"] for record in selected}),
        "selector_matches_expected_count": int(
            sum(
                record["representative_orientation_sign"]
                == record["expected_selected_determinant_sign"]
                for record in selected
            )
        ),
        "selector_matches_high_accuracy_candidate_count": int(
            sum(
                selected_record["requested_determinant_sign"]
                == high_candidate["requested_determinant_sign"]
                for selected_record, high_candidate in zip(selected, high_candidates)
            )
        ),
        "mean_direction_accuracy": float(acc.mean()),
        "minimum_direction_accuracy": float(acc.min()),
        "sample_std_direction_accuracy": float(acc.std(ddof=1)),
        "mean_movement_r2": float(r2.mean()),
        "minimum_movement_r2": float(r2.min()),
        "sample_std_movement_r2": float(r2.std(ddof=1)),
        "high_branch_count_accuracy_gt_0.5": int(np.sum(acc > 0.5)),
        "condition_summary": condition_summary,
    }


def _plot(records: list[dict[str, Any]], out_path: Path) -> None:
    selected = _selected(records)
    conditions = sorted({record["condition"] for record in selected})
    labels = {
        "source_axis_0_flipped": "source x₁ flipped",
        "source_axis_1_flipped": "source x₂ flipped",
        "source_axis_2_flipped": "source x₃ flipped",
    }
    fig, axes = plt.subplots(
        1, 3, figsize=(13.8, 4.6), facecolor="white", constrained_layout=True
    )
    for axis, metric, title, ylabel in [
        (axes[0], "after_direction_accuracy", "Direction accuracy after selected branch", "accuracy"),
        (axes[1], "after_velocity_r2", "Movement geometry after selected branch", "R²"),
        (axes[2], "representative_orientation_sign", "Selected determinant sign", "det sign"),
    ]:
        data = [
            [record[metric] for record in selected if record["condition"] == condition]
            for condition in conditions
        ]
        axis.boxplot(
            data,
            tick_labels=[labels.get(condition, condition) for condition in conditions],
            patch_artist=True,
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_facecolor("white")
        axis.grid(alpha=0.2)
        axis.tick_params(axis="x", rotation=20)
    axes[0].axhline(0.5, color="0.4", linestyle="--", linewidth=1.0)
    axes[2].set_yticks([-1, 1])
    fig.savefig(out_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    payloads = [
        json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))
        for name in args.inputs
    ]
    final_records = [
        record for payload in payloads for record in payload["final_records"]
    ]
    combined = {
        "experiment": "coordinate_flip_validation",
        "status": "aggregated_source_axis_reflection_stress_test",
        "inputs": args.inputs,
        "final_records": final_records,
        "summary": _summary(final_records),
    }
    raw_path = RESULTS_DIR / f"coordinate_flip_validation_combined_{args.tag}.json"
    summary_path = (
        RESULTS_DIR / f"coordinate_flip_validation_combined_summary_{args.tag}.json"
    )
    figure_path = FIGURES_DIR / f"coordinate_flip_validation_combined_{args.tag}.png"
    write_json(raw_path, combined)
    write_json(
        summary_path,
        {"experiment": combined["experiment"], "summary": combined["summary"]},
    )
    _plot(final_records, figure_path)
    print(f"Saved {raw_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {figure_path}")
    print(json.dumps(combined["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
