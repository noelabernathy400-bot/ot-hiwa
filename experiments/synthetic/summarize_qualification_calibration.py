"""Summarize fixed-threshold recovery/abstention calibration results.

This reader never re-fits a model and never changes a threshold. It converts
per-seed benchmark records into auditable counts for the oracle-determinant
core diagnostic and the label-free ROCA decision. Ground-truth condition
labels are used only after fitting, for reporting a fixed decision.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_RECOVERABLE = ("ideal_rotation", "noise_05", "noise_10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="boundary benchmark JSON")
    parser.add_argument(
        "--recoverable-scenarios",
        nargs="+",
        default=DEFAULT_RECOVERABLE,
        help="predeclared conditions assumed to contain one allowed global rotation",
    )
    parser.add_argument("--output", type=Path, default=None, help="optional output JSON")
    return parser.parse_args()


def _mean_or_none(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _core_row_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [bool(row["applicability_qualification"]["accepted"]) for row in rows]
    errors = [
        row.get("rotation_frobenius_error")
        for row in rows
        if bool(row["applicability_qualification"]["accepted"])
    ]
    return {
        "runs": len(rows),
        "accepted_count": int(sum(accepted)),
        "abstained_count": int(len(rows) - sum(accepted)),
        "accepted_rotation_frobenius_error_mean": _mean_or_none(errors),
    }


def _roca_row_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if row["method"] == "soft_roca"]
    abstained = [row for row in rows if row["method"] == "soft_roca_abstained"]
    if len(selected) + len(abstained) != len(rows):
        raise ValueError("ROCA records must be either soft_roca or soft_roca_abstained")
    return {
        "runs": len(rows),
        "accepted_count": len(selected),
        "abstained_count": len(abstained),
        "selected_correct_component_count": int(
            sum(bool(row.get("roca_selected_correct_component", False)) for row in selected)
        ),
        "accepted_rotation_frobenius_error_mean": _mean_or_none(
            [row.get("rotation_frobenius_error") for row in selected]
        ),
    }


def audit(payload: dict[str, Any], recoverable_scenarios: set[str]) -> dict[str, Any]:
    records = payload["records"]
    by_scenario_method: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_scenario_method[(str(row["scenario"]), str(row["method"]))].append(row)

    scenarios = sorted({str(row["scenario"]) for row in records})
    scenario_rows: list[dict[str, Any]] = []
    total = {
        "recoverable_runs": 0,
        "recoverable_core_accepted": 0,
        "recoverable_roca_accepted": 0,
        "nonrecoverable_runs": 0,
        "nonrecoverable_core_abstained": 0,
        "nonrecoverable_roca_abstained": 0,
    }
    for scenario in scenarios:
        expected_recoverable = scenario in recoverable_scenarios
        core = _core_row_summary(by_scenario_method[(scenario, "soft_transport_oracle_component")])
        roca_rows = [
            *by_scenario_method[(scenario, "soft_roca")],
            *by_scenario_method[(scenario, "soft_roca_abstained")],
        ]
        roca = _roca_row_summary(roca_rows)
        runs = core["runs"]
        if roca["runs"] != runs:
            raise ValueError(f"scenario={scenario}: unequal core and ROCA run counts")
        if expected_recoverable:
            total["recoverable_runs"] += runs
            total["recoverable_core_accepted"] += core["accepted_count"]
            total["recoverable_roca_accepted"] += roca["accepted_count"]
        else:
            total["nonrecoverable_runs"] += runs
            total["nonrecoverable_core_abstained"] += core["abstained_count"]
            total["nonrecoverable_roca_abstained"] += roca["abstained_count"]
        scenario_rows.append(
            {
                "scenario": scenario,
                "expected_single_rotation": expected_recoverable,
                "core_oracle_component": core,
                "roca_label_free": roca,
            }
        )

    def rate(numerator: int, denominator: int) -> float | None:
        return float(numerator / denominator) if denominator else None

    return {
        "source_experiment": payload.get("experiment"),
        "truth_usage": (
            "scenario truth is used only for this post-fit audit; it was not supplied "
            "to fitting, qualification, or ROCA selection"
        ),
        "predeclared_recoverable_scenarios": sorted(recoverable_scenarios),
        "per_scenario": scenario_rows,
        "aggregate": {
            **total,
            "core_recovery_accept_rate": rate(total["recoverable_core_accepted"], total["recoverable_runs"]),
            "roca_recovery_accept_rate": rate(total["recoverable_roca_accepted"], total["recoverable_runs"]),
            "core_nonrecovery_abstain_rate": rate(
                total["nonrecoverable_core_abstained"], total["nonrecoverable_runs"]
            ),
            "roca_nonrecovery_abstain_rate": rate(
                total["nonrecoverable_roca_abstained"], total["nonrecoverable_runs"]
            ),
        },
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = audit(payload, set(args.recoverable_scenarios))
    output = args.output or args.input.with_name(f"{args.input.stem}_qualification_audit.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
