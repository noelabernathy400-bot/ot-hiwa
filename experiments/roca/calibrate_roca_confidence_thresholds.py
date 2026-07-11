"""Calibrate and apply synthetic-only ROCA confidence warnings.

Calibration uses only synthetic determinant-validation diagnostics.  The
selected threshold rule is then applied unchanged to existing real ROCA result
files to estimate warning rates.  Real direction accuracy is not used for
threshold calibration.
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn

from analyze_roca_degeneracy_diagnostics import (
    _assignment_diagnostics,
    _matching_diagnostics,
    _representatives,
    _simplex_diagnostics,
)
from common import json_ready
from run_coordinate_flip_validation import _source_condition
from run_rotation_stabilized import ANNEALING_PATH, _fixed_assignments, _load_problem


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = SCRIPT_ROOT / "results" / "roca_confidence_calibration"
FIGURE_ROOT = SCRIPT_ROOT / "figures" / "roca_confidence_calibration"

SYNTHETIC_DIAGNOSTICS = (
    SCRIPT_ROOT
    / "results"
    / "roca_degeneracy_diagnostics"
    / "roca_degeneracy_diagnostics_raw.json"
)
REAL_REPRESENTATIVE_RESULTS = (
    SCRIPT_ROOT
    / "results"
    / "component_aware_soft_hiwa_representative_confirm_seeds_50_69.json"
)
REAL_COORDINATE_FLIP_RESULTS = (
    SCRIPT_ROOT
    / "results"
    / "coordinate_flip_validation_combined_source_axes_seeds_70_79.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-diagnostics", type=Path, default=SYNTHETIC_DIAGNOSTICS)
    parser.add_argument("--representative-results", type=Path, default=REAL_REPRESENTATIVE_RESULTS)
    parser.add_argument("--coordinate-flip-results", type=Path, default=REAL_COORDINATE_FLIP_RESULTS)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--entropy-weight", type=float, default=0.05)
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    for key in ("records", "results", "stage_results", "final_records"):
        if key in payload:
            return payload[key]
    raise KeyError("could not find records in JSON payload")


def _warning_with_thresholds(record: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    reasons = []
    if record["volume_product_margin"] < thresholds["volume_product_margin_min"]:
        reasons.append("low_oriented_volume_margin")
    if (
        record["source_condition_number"] > thresholds["simplex_condition_number_max"]
        or record["target_condition_number"] > thresholds["simplex_condition_number_max"]
    ):
        reasons.append("ill_conditioned_simplex")
    if (
        record["source_assignment_mean_entropy"] > thresholds["assignment_entropy_max"]
        or record["target_assignment_mean_entropy"] > thresholds["assignment_entropy_max"]
    ):
        reasons.append("high_assignment_entropy")
    if record["matching_score_relative_margin"] < thresholds["matching_relative_margin_min"]:
        reasons.append("ambiguous_group_matching")
    return {
        "warning": bool(reasons),
        "warning_reasons": reasons,
    }


def _evaluate_thresholds(
    records: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    failures = [record for record in records if not record["selected_is_correct"]]
    successes = [record for record in records if record["selected_is_correct"]]
    warned = []
    caught = []
    false_warnings = []
    for record in records:
        warning = _warning_with_thresholds(record, thresholds)["warning"]
        if warning:
            warned.append(record)
        if warning and not record["selected_is_correct"]:
            caught.append(record)
        if warning and record["selected_is_correct"]:
            false_warnings.append(record)
    return {
        "thresholds": thresholds,
        "cases": len(records),
        "failures": len(failures),
        "warning_count": len(warned),
        "caught_failures": len(caught),
        "false_warning_count": len(false_warnings),
        "failure_recall": float(len(caught) / len(failures)) if failures else None,
        "precision": float(len(caught) / len(warned)) if warned else None,
    }


def _calibrate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Select a simple OR-rule with full synthetic failure recall.

    The grid is deliberately coarse and interpretable.  Among rules with full
    failure recall, choose the one with the fewest warnings, then highest
    precision, then less restrictive volume/matching and more restrictive
    entropy/condition limits to avoid overly broad warnings.
    """
    volume_mins = [0.0, 0.25, 0.5, 0.75, 1.0]
    condition_maxes = [30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 1e12]
    entropy_maxes = [0.35, 0.40, 0.45, 0.48, 0.50, 1e12]
    matching_mins = [0.0, 0.01, 0.02, 0.03, 0.05]
    candidates = []
    for values in itertools.product(volume_mins, condition_maxes, entropy_maxes, matching_mins):
        thresholds = {
            "volume_product_margin_min": values[0],
            "simplex_condition_number_max": values[1],
            "assignment_entropy_max": values[2],
            "matching_relative_margin_min": values[3],
        }
        candidates.append(_evaluate_thresholds(records, thresholds))
    feasible = [item for item in candidates if item["failure_recall"] == 1.0]
    if not feasible:
        raise RuntimeError("no candidate threshold rule achieved full synthetic failure recall")
    feasible.sort(
        key=lambda item: (
            item["warning_count"],
            -float(item["precision"] or 0.0),
            item["thresholds"]["volume_product_margin_min"],
            -item["thresholds"]["simplex_condition_number_max"],
            -item["thresholds"]["assignment_entropy_max"],
            item["thresholds"]["matching_relative_margin_min"],
        )
    )
    return {
        "selected": feasible[0],
        "top_candidates": feasible[:10],
        "grid_size": len(candidates),
        "feasible_count": len(feasible),
    }


def _real_diagnostic_record(
    *,
    source: np.ndarray,
    source_assignments: np.ndarray,
    target: np.ndarray,
    target_assignments: np.ndarray,
    candidate_transports: list[np.ndarray],
    selected_det: int,
    metadata: dict[str, Any],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    source_reps = _representatives(source, source_assignments)
    target_reps = _representatives(target, target_assignments)
    source_simplex = _simplex_diagnostics(source_reps)
    target_simplex = _simplex_diagnostics(target_reps)
    source_assignment = _assignment_diagnostics(source_assignments)
    target_assignment = _assignment_diagnostics(target_assignments)
    mean_transport = np.mean(np.asarray(candidate_transports), axis=0)
    matching = _matching_diagnostics(mean_transport)
    record: dict[str, Any] = {
        **metadata,
        "selected_det": int(selected_det),
        "source_volume": source_simplex["volume"],
        "target_volume": target_simplex["volume"],
        "volume_product_margin": float(source_simplex["abs_volume"] * target_simplex["abs_volume"]),
        "source_condition_number": source_simplex["condition_number"],
        "target_condition_number": target_simplex["condition_number"],
        "source_assignment_mean_entropy": source_assignment["mean_entropy"],
        "target_assignment_mean_entropy": target_assignment["mean_entropy"],
        "source_assignment_mean_max_probability": source_assignment["mean_max_probability"],
        "target_assignment_mean_max_probability": target_assignment["mean_max_probability"],
        "source_group_mass_ratio": source_assignment["group_mass_ratio"],
        "target_group_mass_ratio": target_assignment["group_mass_ratio"],
        **matching,
    }
    warning = _warning_with_thresholds(record, thresholds)
    record["warning_v1_calibrated"] = warning["warning"]
    record["warning_v1_reasons"] = warning["warning_reasons"]
    return record


def _selected_by_seed(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("temperature") != ANNEALING_PATH[-1]:
            continue
        by_seed.setdefault(int(record["seed"]), []).append(record)
    return by_seed


def _apply_to_representative_confirm(
    path: Path,
    thresholds: dict[str, float],
    *,
    groups: int,
    entropy_weight: float,
) -> list[dict[str, Any]]:
    payload = _load_json(path)
    records = _as_records(payload)
    by_seed = _selected_by_seed(records)
    neural_3d, movement_3d, _, _, _ = _load_problem()
    output = []
    for seed, seed_records in sorted(by_seed.items()):
        if len(seed_records) != 2:
            continue
        neural_assignments = _fixed_assignments(neural_3d, groups, entropy_weight, seed)[ANNEALING_PATH[-1]]
        movement_assignments = _fixed_assignments(movement_3d, groups, entropy_weight, seed)[ANNEALING_PATH[-1]]
        selected_det = int(seed_records[0]["representative_orientation_sign"])
        output.append(
            _real_diagnostic_record(
                source=neural_3d,
                source_assignments=neural_assignments,
                target=movement_3d,
                target_assignments=movement_assignments,
                candidate_transports=[np.asarray(record["transport_P"]) for record in seed_records],
                selected_det=selected_det,
                metadata={"experiment": "representative_confirm", "seed": seed},
                thresholds=thresholds,
            )
        )
    return output


def _apply_to_coordinate_flip(
    path: Path,
    thresholds: dict[str, float],
    *,
    groups: int,
    entropy_weight: float,
) -> list[dict[str, Any]]:
    payload = _load_json(path)
    final_records = payload.get("final_records", [])
    neural_3d, movement_3d, _, _, _ = _load_problem()
    base_source_transform = None
    by_condition: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for record in final_records:
        key = (int(record["seed"]), str(record["condition"]))
        by_condition.setdefault(key, []).append(record)
    output = []
    for (seed, condition), condition_records in sorted(by_condition.items()):
        if len(condition_records) != 2:
            continue
        axis_value = condition_records[0].get("flipped_source_axis")
        axis = None if axis_value is None else int(axis_value)
        if base_source_transform is None:
            # The warning metrics only use transformed source values and not this
            # transform, so an identity matrix is sufficient for _source_condition.
            base_source_transform = np.eye(neural_3d.shape[1])
        _, condition_neural_3d, _, _ = _source_condition(neural_3d, base_source_transform, axis)
        neural_assignments = _fixed_assignments(condition_neural_3d, groups, entropy_weight, seed)[ANNEALING_PATH[-1]]
        movement_assignments = _fixed_assignments(movement_3d, groups, entropy_weight, seed)[ANNEALING_PATH[-1]]
        selected_det = int(condition_records[0]["representative_orientation_sign"])
        output.append(
            _real_diagnostic_record(
                source=condition_neural_3d,
                source_assignments=neural_assignments,
                target=movement_3d,
                target_assignments=movement_assignments,
                candidate_transports=[np.asarray(record["transport_P"]) for record in condition_records],
                selected_det=selected_det,
                metadata={
                    "experiment": "coordinate_flip",
                    "seed": seed,
                    "condition": condition,
                    "flipped_source_axis": axis,
                },
                thresholds=thresholds,
            )
        )
    return output


def _warning_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    warning_records = [record for record in records if record["warning_v1_calibrated"]]
    by_experiment: dict[str, Any] = {}
    for experiment in sorted({record["experiment"] for record in records}):
        subset = [record for record in records if record["experiment"] == experiment]
        warned = [record for record in subset if record["warning_v1_calibrated"]]
        by_experiment[experiment] = {
            "cases": len(subset),
            "warning_count": len(warned),
            "warning_rate": float(len(warned) / len(subset)) if subset else None,
            "warning_records": warned,
        }
    return {
        "cases": len(records),
        "warning_count": len(warning_records),
        "warning_rate": float(len(warning_records) / len(records)) if records else None,
        "by_experiment": by_experiment,
    }


def _plot_real(records: list[dict[str, Any]], thresholds: dict[str, float], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for experiment, marker in [("representative_confirm", "o"), ("coordinate_flip", "s")]:
        subset = [record for record in records if record["experiment"] == experiment]
        colors = ["tab:red" if record["warning_v1_calibrated"] else "tab:blue" for record in subset]
        axes[0].scatter(
            [record["volume_product_margin"] for record in subset],
            [record["source_assignment_mean_entropy"] for record in subset],
            c=colors,
            marker=marker,
            alpha=0.8,
            label=experiment,
        )
        axes[1].scatter(
            [record["source_condition_number"] for record in subset],
            [record["target_condition_number"] for record in subset],
            c=colors,
            marker=marker,
            alpha=0.8,
        )
        axes[2].scatter(
            [record["matching_score_relative_margin"] for record in subset],
            [record["volume_product_margin"] for record in subset],
            c=colors,
            marker=marker,
            alpha=0.8,
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("|det(Xrep)| · |det(Yrep)|")
    axes[0].set_ylabel("source assignment mean entropy")
    axes[0].set_title("Real warning check: volume vs entropy")
    axes[0].legend()
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    condition_limit = thresholds["simplex_condition_number_max"]
    axes[1].axvline(condition_limit, color="tab:red", linestyle="--", linewidth=1.5)
    axes[1].axhline(condition_limit, color="tab:red", linestyle="--", linewidth=1.5)
    max_condition = max(
        condition_limit * 1.2,
        max(record["source_condition_number"] for record in records) * 1.2,
        max(record["target_condition_number"] for record in records) * 1.2,
    )
    axes[1].set_xlim(left=1.0, right=max_condition)
    axes[1].set_ylim(bottom=1.0, top=max_condition)
    axes[1].set_xlabel("source simplex condition number")
    axes[1].set_ylabel("target simplex condition number")
    axes[1].set_title("Real simplex conditioning; red line = calibrated threshold")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("relative matching margin")
    axes[2].set_ylabel("volume product margin")
    axes[2].set_title("Real matching vs volume")
    fig.suptitle("Fixed synthetic-calibrated ROCA warning on real runs")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _write_report(
    calibration: dict[str, Any],
    real_summary: dict[str, Any],
    real_records: list[dict[str, Any]],
    output: Path,
) -> None:
    thresholds = calibration["selected"]["thresholds"]
    lines = [
        "# ROCA confidence calibration report",
        "",
        "Calibration uses synthetic diagnostics only. Real data are checked after thresholds are fixed.",
        "",
        "## Selected thresholds",
        "",
        f"- volume_product_margin_min: {thresholds['volume_product_margin_min']}",
        f"- simplex_condition_number_max: {thresholds['simplex_condition_number_max']}",
        f"- assignment_entropy_max: {thresholds['assignment_entropy_max']}",
        f"- matching_relative_margin_min: {thresholds['matching_relative_margin_min']}",
        "",
        "## Synthetic calibration performance",
        "",
        f"- cases: {calibration['selected']['cases']}",
        f"- failures: {calibration['selected']['failures']}",
        f"- warning_count: {calibration['selected']['warning_count']}",
        f"- caught_failures: {calibration['selected']['caught_failures']}",
        f"- false_warning_count: {calibration['selected']['false_warning_count']}",
        f"- failure_recall: {calibration['selected']['failure_recall']}",
        f"- precision: {calibration['selected']['precision']}",
        "",
        "## Fixed-threshold real warning check",
        "",
        f"- cases: {real_summary['cases']}",
        f"- warning_count: {real_summary['warning_count']}",
        f"- warning_rate: {real_summary['warning_rate']}",
        "",
        "| experiment | cases | warning_count | warning_rate |",
        "|---|---:|---:|---:|",
    ]
    for experiment, item in real_summary["by_experiment"].items():
        lines.append(
            f"| {experiment} | {item['cases']} | {item['warning_count']} | {item['warning_rate']} |"
        )
    lines.extend(
        [
            "",
            "## Real warning records",
            "",
            "| experiment | seed | condition | warning reasons | volume margin | src cond | tgt cond | src entropy | match rel margin |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    warned = [record for record in real_records if record["warning_v1_calibrated"]]
    if not warned:
        lines.append("| none |  |  |  |  |  |  |  |  |")
    for record in warned:
        lines.append(
            "| {experiment} | {seed} | {condition} | {warning_v1_reasons} | {volume_product_margin:.3e} | {source_condition_number:.3f} | {target_condition_number:.3f} | {source_assignment_mean_entropy:.3f} | {matching_score_relative_margin:.3e} |".format(
                condition=record.get("condition", ""),
                **record,
            )
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    synthetic_records = _as_records(_load_json(args.synthetic_diagnostics))
    calibration = _calibrate(synthetic_records)
    thresholds = calibration["selected"]["thresholds"]
    representative_records = _apply_to_representative_confirm(
        args.representative_results,
        thresholds,
        groups=args.groups,
        entropy_weight=args.entropy_weight,
    )
    coordinate_records = _apply_to_coordinate_flip(
        args.coordinate_flip_results,
        thresholds,
        groups=args.groups,
        entropy_weight=args.entropy_weight,
    )
    real_records = representative_records + coordinate_records
    real_summary = _warning_summary(real_records)
    payload = {
        "experiment": "roca_confidence_calibration",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "parameters": {
            "synthetic_diagnostics": str(args.synthetic_diagnostics),
            "representative_results": str(args.representative_results),
            "coordinate_flip_results": str(args.coordinate_flip_results),
            "groups": args.groups,
            "entropy_weight": args.entropy_weight,
        },
        "calibration": calibration,
        "real_warning_summary": real_summary,
        "real_warning_records": real_records,
    }
    suffix = f"_{args.tag}" if args.tag else ""
    raw_path = RESULT_ROOT / f"roca_confidence_calibration_raw{suffix}.json"
    summary_path = RESULT_ROOT / f"roca_confidence_calibration_summary{suffix}.json"
    report_path = RESULT_ROOT / f"roca_confidence_calibration_report{suffix}.md"
    figure_path = FIGURE_ROOT / f"roca_confidence_calibration_real_warning_check{suffix}.png"
    raw_path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            json_ready(
                {
                    "experiment": payload["experiment"],
                    "selected_thresholds": thresholds,
                    "synthetic_selected_performance": calibration["selected"],
                    "real_warning_summary": real_summary,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(calibration, real_summary, real_records, report_path)
    _plot_real(real_records, thresholds, figure_path)
    print(json.dumps(json_ready({
        "selected_thresholds": thresholds,
        "synthetic_selected_performance": calibration["selected"],
        "real_warning_summary": real_summary,
    }), indent=2))
    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
