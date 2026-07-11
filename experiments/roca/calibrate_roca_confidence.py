"""Split-based confidence calibration for ROCA low-confidence warnings.

This script uses existing synthetic degeneracy diagnostics and existing real
ROCA result files.  It never uses real direction accuracy or real R2 for
threshold selection.  By default it writes suffixed outputs so previous result
files are not overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn

from calibrate_roca_confidence_thresholds import (
    _apply_to_coordinate_flip,
    _apply_to_representative_confirm,
    _warning_summary,
    _warning_with_thresholds,
)
from common import json_ready


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = SCRIPT_ROOT / "results" / "roca_confidence_calibration"
FIGURE_ROOT = SCRIPT_ROOT / "figures" / "roca_confidence_calibration"
SYNTHETIC_DIAGNOSTICS = (
    SCRIPT_ROOT
    / "results"
    / "roca_degeneracy_diagnostics"
    / "roca_degeneracy_diagnostics_raw.json"
)
REPRESENTATIVE_RESULTS = (
    SCRIPT_ROOT
    / "results"
    / "component_aware_soft_hiwa_representative_confirm_seeds_50_69.json"
)
COORDINATE_FLIP_RESULTS = (
    SCRIPT_ROOT
    / "results"
    / "coordinate_flip_validation_combined_source_axes_seeds_70_79.json"
)

METRICS = {
    "volume_product_margin": "low",
    "source_condition_number": "high",
    "target_condition_number": "high",
    "source_assignment_mean_entropy": "high",
    "target_assignment_mean_entropy": "high",
    "matching_score_relative_margin": "low",
    "source_min_pairwise_distance": "low",
    "target_min_pairwise_distance": "low",
    "source_volume_abs": "low",
    "target_volume_abs": "low",
    "matching_score_margin": "low",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-diagnostics", type=Path, default=SYNTHETIC_DIAGNOSTICS)
    parser.add_argument("--representative-results", type=Path, default=REPRESENTATIVE_RESULTS)
    parser.add_argument("--coordinate-flip-results", type=Path, default=COORDINATE_FLIP_RESULTS)
    parser.add_argument("--calibration-seeds", nargs="+", type=int, default=list(range(25)))
    parser.add_argument("--validation-seeds", nargs="+", type=int, default=list(range(25, 50)))
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--entropy-weight", type=float, default=0.05)
    parser.add_argument("--tag", default="split_2026_07_08")
    return parser.parse_args()


def _load_records(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError(f"expected a list of records in {path}")
    for record in records:
        record["source_volume_abs"] = abs(float(record["source_volume"]))
        record["target_volume_abs"] = abs(float(record["target_volume"]))
    return records


def split_records(
    records: list[dict[str, Any]],
    calibration_seeds: set[int],
    validation_seeds: set[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if calibration_seeds & validation_seeds:
        raise ValueError("calibration and validation seeds overlap")
    calibration = [record for record in records if int(record["seed"]) in calibration_seeds]
    validation = [record for record in records if int(record["seed"]) in validation_seeds]
    return calibration, validation


def _metric_warning(record: dict[str, Any], metric: str, direction: str, threshold: float) -> bool:
    value = float(record[metric])
    if direction == "high":
        return value > threshold
    if direction == "low":
        return value < threshold
    raise ValueError(direction)


def _metric_threshold(records: list[dict[str, Any]], metric: str, direction: str, quantile: float) -> float:
    values = np.asarray([float(record[metric]) for record in records], dtype=float)
    if direction == "high":
        return float(np.quantile(values, quantile))
    return float(np.quantile(values, 1.0 - quantile))


def _evaluate_rule(
    records: list[dict[str, Any]],
    *,
    rule_name: str,
    warning_fn,
) -> dict[str, Any]:
    failures = [record for record in records if not record["selected_is_correct"]]
    successes = [record for record in records if record["selected_is_correct"]]
    warned = []
    caught = []
    false_pos = []
    non_warning = []
    warning_correct = []
    for record in records:
        is_warning = bool(warning_fn(record))
        if is_warning:
            warned.append(record)
            warning_correct.append(bool(record["selected_is_correct"]))
        else:
            non_warning.append(record)
        if is_warning and not record["selected_is_correct"]:
            caught.append(record)
        if is_warning and record["selected_is_correct"]:
            false_pos.append(record)
    false_negative_count = len(failures) - len(caught)
    return {
        "rule": rule_name,
        "cases": len(records),
        "failures": len(failures),
        "warning_count": len(warned),
        "warning_rate": float(len(warned) / len(records)) if records else None,
        "coverage": float(len(non_warning) / len(records)) if records else None,
        "failure_recall": float(len(caught) / len(failures)) if failures else None,
        "warning_precision": float(len(caught) / len(warned)) if warned else None,
        "false_warning_rate": float(len(false_pos) / len(successes)) if successes else None,
        "false_negative_count": false_negative_count,
        "false_positive_count": len(false_pos),
        "determinant_recovery": float(np.mean([record["selected_is_correct"] for record in records])) if records else None,
        "determinant_recovery_non_warning": (
            float(np.mean([record["selected_is_correct"] for record in non_warning]))
            if non_warning
            else None
        ),
        "determinant_recovery_warning": (
            float(np.mean(warning_correct)) if warning_correct else None
        ),
        "low_confidence_cases": [
            {
                "seed": record["seed"],
                "true_det": record["true_det"],
                "selected_det": record["selected_det"],
                "noise_level": record["noise_level"],
                "assignment_mode": record["assignment_mode"],
                "degeneracy_mode": record["degeneracy_mode"],
                "selected_is_correct": record["selected_is_correct"],
            }
            for record in warned
        ],
    }


def _warning_v0(record: dict[str, Any]) -> bool:
    return bool(record.get("warning_v0", False))


def _build_rules(calibration: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = [
        {
            "name": "warning_v0_existing",
            "description": "Existing OR-rule from degeneracy diagnostics.",
            "kind": "baseline",
            "fn": _warning_v0,
        }
    ]
    # Single-metric 99th/1st percentile rules based only on calibration split.
    metric_thresholds = {}
    for metric, direction in METRICS.items():
        threshold = _metric_threshold(calibration, metric, direction, 0.99)
        metric_thresholds[metric] = threshold
        rules.append(
            {
                "name": f"single_{metric}",
                "description": f"Single metric rule: {metric} is {direction} beyond calibration 99% tail.",
                "kind": "single_metric",
                "metric": metric,
                "direction": direction,
                "threshold": threshold,
                "fn": lambda record, m=metric, d=direction, t=threshold: _metric_warning(record, m, d, t),
            }
        )
    # Simple interpretable combined rules. These are calibrated by calibration tails,
    # not by real data.
    condition_threshold = metric_thresholds["source_condition_number"]
    condition_threshold = max(condition_threshold, metric_thresholds["target_condition_number"])
    entropy_threshold = max(
        metric_thresholds["source_assignment_mean_entropy"],
        metric_thresholds["target_assignment_mean_entropy"],
    )
    volume_threshold = metric_thresholds["volume_product_margin"]
    matching_threshold = metric_thresholds["matching_score_relative_margin"]

    def disjunctive(record: dict[str, Any]) -> bool:
        return (
            record["source_condition_number"] > condition_threshold
            or record["target_condition_number"] > condition_threshold
            or record["source_assignment_mean_entropy"] > entropy_threshold
            or record["target_assignment_mean_entropy"] > entropy_threshold
            or record["volume_product_margin"] < volume_threshold
            or record["matching_score_relative_margin"] < matching_threshold
        )

    def conjunctive(record: dict[str, Any]) -> bool:
        ill_conditioned = (
            record["source_condition_number"] > condition_threshold
            or record["target_condition_number"] > condition_threshold
        )
        ambiguous = record["matching_score_relative_margin"] < matching_threshold
        return ill_conditioned and ambiguous

    # Previous synthetic-all calibrated rule kept as an explicit frozen candidate,
    # but marked separately because it was not produced by the 0-24 split.
    frozen_thresholds = {
        "volume_product_margin_min": 0.0,
        "simplex_condition_number_max": 80.0,
        "assignment_entropy_max": 1e12,
        "matching_relative_margin_min": 0.0,
    }
    rules.extend(
        [
            {
                "name": "quantile_disjunctive_calibration_tail",
                "description": "OR of calibration 99% tail conditions.",
                "kind": "quantile_disjunctive",
                "thresholds": {
                    "condition_number_max": condition_threshold,
                    "assignment_entropy_max": entropy_threshold,
                    "volume_product_margin_min": volume_threshold,
                    "matching_relative_margin_min": matching_threshold,
                },
                "fn": disjunctive,
            },
            {
                "name": "conjunctive_condition_and_matching",
                "description": "Ill-conditioned simplex AND ambiguous matching.",
                "kind": "conjunctive",
                "thresholds": {
                    "condition_number_max": condition_threshold,
                    "matching_relative_margin_min": matching_threshold,
                },
                "fn": conjunctive,
            },
            {
                "name": "frozen_condition_number_gt_80",
                "description": "Candidate frozen rule from previous synthetic-all confidence check; reported but not tuned on real data.",
                "kind": "conservative_candidate",
                "thresholds": frozen_thresholds,
                "fn": lambda record: _warning_with_thresholds(record, frozen_thresholds)["warning"],
            },
        ]
    )
    return rules


def _compare_rules(
    calibration: list[dict[str, Any]],
    validation: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rules = _build_rules(calibration)
    rows = []
    for rule in rules:
        for split_name, split_records in [("calibration", calibration), ("validation", validation)]:
            result = _evaluate_rule(split_records, rule_name=rule["name"], warning_fn=rule["fn"])
            result["split"] = split_name
            result["kind"] = rule["kind"]
            result["description"] = rule["description"]
            if "metric" in rule:
                result["metric"] = rule["metric"]
                result["direction"] = rule["direction"]
                result["threshold"] = rule["threshold"]
            if "thresholds" in rule:
                result["thresholds"] = rule["thresholds"]
            rows.append(result)
    # Pick recommendation from validation behavior while respecting the fact
    # that calibration split has zero failures in this dataset.
    validation_rows = [row for row in rows if row["split"] == "validation"]
    candidates = [
        row
        for row in validation_rows
        if row["failure_recall"] == 1.0 and row["warning_count"] > 0
    ]
    if candidates:
        candidates.sort(
            key=lambda row: (
                row["false_warning_rate"] if row["false_warning_rate"] is not None else 1.0,
                row["warning_rate"] if row["warning_rate"] is not None else 1.0,
                -float(row["warning_precision"] or 0.0),
            )
        )
        recommended = candidates[0]
    else:
        recommended = max(
            validation_rows,
            key=lambda row: (
                float(row["failure_recall"] or 0.0),
                -float(row["false_warning_rate"] or 1.0),
            ),
        )
    return rows, recommended


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "split",
        "rule",
        "kind",
        "cases",
        "failures",
        "warning_count",
        "warning_rate",
        "coverage",
        "failure_recall",
        "warning_precision",
        "false_warning_rate",
        "false_negative_count",
        "false_positive_count",
        "determinant_recovery",
        "determinant_recovery_non_warning",
        "determinant_recovery_warning",
        "description",
        "metric",
        "direction",
        "threshold",
        "thresholds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row = dict(row)
            if "thresholds" in row:
                row["thresholds"] = json.dumps(row["thresholds"], ensure_ascii=False)
            writer.writerow(row)


def _real_warning_check(recommended: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if recommended["rule"] == "frozen_condition_number_gt_80":
        thresholds = {
            "volume_product_margin_min": 0.0,
            "simplex_condition_number_max": 80.0,
            "assignment_entropy_max": 1e12,
            "matching_relative_margin_min": 0.0,
        }
    elif recommended.get("thresholds") and "condition_number_max" in recommended["thresholds"]:
        t = recommended["thresholds"]
        thresholds = {
            "volume_product_margin_min": float(t.get("volume_product_margin_min", 0.0)),
            "simplex_condition_number_max": float(t.get("condition_number_max", 1e12)),
            "assignment_entropy_max": float(t.get("assignment_entropy_max", 1e12)),
            "matching_relative_margin_min": float(t.get("matching_relative_margin_min", 0.0)),
        }
    else:
        thresholds = {
            "volume_product_margin_min": 0.0,
            "simplex_condition_number_max": 80.0,
            "assignment_entropy_max": 1e12,
            "matching_relative_margin_min": 0.0,
        }
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
    return {
        "applied_thresholds": thresholds,
        "records": real_records,
        "summary": _warning_summary(real_records),
    }


def _low_confidence_markdown(rows: list[dict[str, Any]], recommended_rule: str, path: Path) -> None:
    selected = [
        row
        for row in rows
        if row["split"] == "validation" and row["rule"] == recommended_rule
    ]
    cases = selected[0]["low_confidence_cases"] if selected else []
    lines = [
        "# Synthetic low-confidence cases",
        "",
        f"Recommended rule: `{recommended_rule}`",
        "",
        "| seed | true_det | selected_det | noise | assignment | degeneracy | selected_correct |",
        "|---:|---:|---:|---:|---|---|---|",
    ]
    if not cases:
        lines.append("| none |  |  |  |  |  |  |")
    for case in cases:
        lines.append(
            "| {seed} | {true_det} | {selected_det} | {noise_level} | {assignment_mode} | {degeneracy_mode} | {selected_is_correct} |".format(
                **case
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(
    *,
    split_note: str,
    recommended: dict[str, Any],
    real: dict[str, Any],
    csv_name: str,
    path: Path,
) -> None:
    lines = [
        "# ROCA confidence calibration report",
        "",
        "This report uses synthetic calibration/validation split and applies the selected rule to real ROCA runs only after threshold freezing.",
        "",
        "## Split note",
        "",
        split_note,
        "",
        "## Recommended rule",
        "",
        f"- rule: `{recommended['rule']}`",
        f"- kind: `{recommended['kind']}`",
        f"- validation failure recall: {recommended['failure_recall']}",
        f"- validation warning precision: {recommended['warning_precision']}",
        f"- validation false warning rate: {recommended['false_warning_rate']}",
        f"- validation warning rate: {recommended['warning_rate']}",
        f"- validation determinant recovery among non-warning cases: {recommended['determinant_recovery_non_warning']}",
        "",
        "## Real warning rates with frozen rule",
        "",
        f"- applied thresholds: `{json.dumps(real['applied_thresholds'])}`",
        f"- real cases: {real['summary']['cases']}",
        f"- real warning count: {real['summary']['warning_count']}",
        f"- real warning rate: {real['summary']['warning_rate']}",
        "",
        "| experiment | cases | warning_count | warning_rate |",
        "|---|---:|---:|---:|",
    ]
    for experiment, item in real["summary"]["by_experiment"].items():
        lines.append(
            f"| {experiment} | {item['cases']} | {item['warning_count']} | {item['warning_rate']} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- rule comparison table: `{csv_name}`",
            "",
            "## Interpretation",
            "",
            "The warning rule is a diagnostic confidence candidate, not a replacement for ROCA branch selection. It must not be tuned with real direction accuracy or real R2.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    records = _load_records(args.synthetic_diagnostics)
    calibration, validation = split_records(
        records,
        set(args.calibration_seeds),
        set(args.validation_seeds),
    )
    calibration_failures = sum(not record["selected_is_correct"] for record in calibration)
    validation_failures = sum(not record["selected_is_correct"] for record in validation)
    split_note = (
        f"Calibration seeds {min(args.calibration_seeds)}-{max(args.calibration_seeds)} contain "
        f"{calibration_failures} failures; validation seeds {min(args.validation_seeds)}-{max(args.validation_seeds)} "
        f"contain {validation_failures} failures. Because calibration has no failures, failure-recall-based threshold tuning is not identifiable without leakage; rules are therefore calibrated from unlabeled calibration-tail distributions and judged on validation."
    )
    rows, recommended = _compare_rules(calibration, validation)
    real = _real_warning_check(recommended, args)
    suffix = f"_{args.tag}" if args.tag else ""
    raw_path = RESULT_ROOT / f"roca_confidence_calibration_raw{suffix}.json"
    summary_path = RESULT_ROOT / f"roca_confidence_calibration_summary{suffix}.json"
    report_path = RESULT_ROOT / f"roca_confidence_calibration_report{suffix}.md"
    csv_path = RESULT_ROOT / f"confidence_rule_comparison_table{suffix}.csv"
    low_conf_path = RESULT_ROOT / f"synthetic_low_confidence_cases{suffix}.md"
    parameters = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    payload = {
        "experiment": "roca_confidence_calibration_split",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "parameters": parameters,
        "split_note": split_note,
        "rule_comparison": rows,
        "recommended_rule": recommended,
        "real_warning_check": real,
    }
    raw_path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            json_ready(
                {
                    "experiment": payload["experiment"],
                    "split_note": split_note,
                    "recommended_rule": recommended,
                    "real_warning_summary": real["summary"],
                    "applied_thresholds": real["applied_thresholds"],
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_csv(rows, csv_path)
    _low_confidence_markdown(rows, recommended["rule"], low_conf_path)
    _write_report(
        split_note=split_note,
        recommended=recommended,
        real=real,
        csv_name=csv_path.name,
        path=report_path,
    )
    print(json.dumps(json_ready({
        "split_note": split_note,
        "recommended_rule": recommended,
        "real_warning_summary": real["summary"],
        "outputs": {
            "raw": str(raw_path),
            "summary": str(summary_path),
            "report": str(report_path),
            "csv": str(csv_path),
            "low_confidence_cases": str(low_conf_path),
        },
    }), indent=2))


if __name__ == "__main__":
    main()
