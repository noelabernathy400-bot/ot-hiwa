"""Audit ROCA geometry/objective disagreement without using evaluation labels.

The rule under study is deliberately an abstention signal, not a replacement
selector: if the representative-orientation sign and the lower transport-
objective candidate disagree, mark the branch choice as ambiguous.  Labels are
read only after that marker is fixed, to evaluate its coverage and errors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"
FIGURES_DIR = ROOT / "experiments" / "figures"


def _candidate_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return final-stage determinant candidates, including the mutated chosen row."""
    rows: list[dict[str, Any]] = []
    for row in payload["stage_results"]:
        method = str(row.get("method", ""))
        if row.get("stage") != 2:
            continue
        if method.startswith("soft_gcot_roca_candidate") or method == "soft_gcot_roca":
            candidate = dict(row)
            candidate["candidate_sign"] = int(round(float(candidate["rotation_determinant"])))
            rows.append(candidate)
    return rows


def audit_payloads(payloads: list[dict[str, Any]], min_relative_gap: float = 0.01) -> list[dict[str, Any]]:
    """Build per-run diagnostics; only final fields are evaluation-only."""
    output: list[dict[str, Any]] = []
    for payload_index, payload in enumerate(payloads):
        sampling = dict(payload.get("sampling", {}))
        subset_seed = sampling.get("seed")
        sampling_mode = sampling.get("mode", "not_recorded")
        roca_by_seed = {int(item["seed"]): item for item in payload["roca"]}
        by_seed: dict[int, list[dict[str, Any]]] = {}
        for row in _candidate_rows(payload):
            by_seed.setdefault(int(row["seed"]), []).append(row)
        for seed, candidates in sorted(by_seed.items()):
            if len(candidates) != 2:
                raise ValueError(f"seed {seed} needs exactly two final ROCA candidates, got {len(candidates)}")
            signs = {int(row["candidate_sign"]) for row in candidates}
            if signs != {-1, 1}:
                raise ValueError(f"seed {seed} candidates must have determinant signs -1 and +1, got {signs}")
            by_sign = {int(row["candidate_sign"]): row for row in candidates}
            geometric_sign = int(roca_by_seed[seed]["selected_determinant_sign"])
            objective_sign = min(by_sign, key=lambda sign: float(by_sign[sign]["transport_objective"]))
            objective_values = {sign: float(by_sign[sign]["transport_objective"]) for sign in (-1, 1)}
            low = min(objective_values.values())
            high = max(objective_values.values())
            oracle_sign = max(
                by_sign,
                key=lambda sign: float(by_sign[sign]["after_direction_accuracy"]),
            )
            relative_gap = float((high - low) / max(abs(low), 1e-12))
            case_id = (
                f"subset_{subset_seed}_solver_{seed}"
                if subset_seed is not None
                else f"input_{payload_index}_solver_{seed}"
            )
            output.append(
                {
                    "case_id": case_id,
                    "seed": seed,
                    "subset_seed": subset_seed,
                    "sampling_mode": sampling_mode,
                    "geometric_sign": geometric_sign,
                    "objective_sign": int(objective_sign),
                    "objective_relative_gap": relative_gap,
                    "geometry_objective_disagree": bool(geometric_sign != objective_sign),
                    "objective_conflict_abstain": bool(
                        geometric_sign != objective_sign and relative_gap >= min_relative_gap
                    ),
                    "geometric_transport_objective": objective_values[geometric_sign],
                    "alternative_transport_objective": objective_values[-geometric_sign],
                    "selected_volume_product_margin": float(roca_by_seed[seed]["volume_product_margin"]),
                    "selected_warning": bool(roca_by_seed[seed]["warning"]),
                    "oracle_sign_evaluation_only": int(oracle_sign),
                    "geometry_matches_oracle_evaluation_only": bool(geometric_sign == oracle_sign),
                    "selected_accuracy_evaluation_only": float(by_sign[geometric_sign]["after_direction_accuracy"]),
                    "alternative_accuracy_evaluation_only": float(by_sign[-geometric_sign]["after_direction_accuracy"]),
                    "selected_r2_evaluation_only": float(by_sign[geometric_sign]["after_velocity_r2"]),
                    "alternative_r2_evaluation_only": float(by_sign[-geometric_sign]["after_velocity_r2"]),
                }
            )
    return output


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("no candidate records")
    accepted = [row for row in records if not row["objective_conflict_abstain"]]
    rejected = [row for row in records if row["objective_conflict_abstain"]]
    wrong = [row for row in records if not row["geometry_matches_oracle_evaluation_only"]]
    rejected_wrong = int(sum(row["objective_conflict_abstain"] for row in wrong))
    if wrong and rejected_wrong == 0:
        interpretation = (
            "The frozen material-conflict marker failed to reject at least one wrong geometric selection. "
            "This result falsifies its use as a general-purpose ROCA confidence mechanism on this protocol."
        )
    elif wrong:
        interpretation = (
            "The marker rejected some wrong geometric selections, but this finite result does not establish "
            "calibrated selective performance or justify replacing ROCA with the objective branch."
        )
    else:
        interpretation = (
            "No wrong geometric selection occurred in these runs, so this result cannot evaluate whether the "
            "marker detects errors or establish calibrated selective performance."
        )
    return {
        "n_runs": len(records),
        "n_seeds": len(records),  # compatibility with earlier result readers
        "accepted_without_material_conflict": len(accepted),
        "rejected_by_material_conflict": len(rejected),
        "coverage_without_material_conflict": len(accepted) / len(records),
        "geometry_accuracy_all_evaluation_only": float(
            np.mean([row["geometry_matches_oracle_evaluation_only"] for row in records])
        ),
        "geometry_accuracy_accepted_evaluation_only": float(
            np.mean([row["geometry_matches_oracle_evaluation_only"] for row in accepted])
        ) if accepted else None,
        "wrong_geometry_selections_evaluation_only": len(wrong),
        "wrong_selections_rejected_evaluation_only": rejected_wrong,
        "warning_rejected": int(sum(row["selected_warning"] for row in rejected)),
        "interpretation": interpretation,
    }


def _write_report(
    path: Path,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    inputs: list[Path],
    min_relative_gap: float,
    study_phase: str,
) -> None:
    if study_phase == "post_hoc":
        scope_text = (
            "This is a post-hoc diagnostic of an existing result set. The disagreement marker is "
            "computed from unlabeled outputs only; direction accuracy and $R^2$ are shown only after "
            "the marker is fixed. This result set must not be used to claim calibrated selective performance."
        )
    elif study_phase == "blind":
        scope_text = (
            "This is a blind evaluation of a diagnostic frozen before these seeds were inspected. "
            "The disagreement marker is computed from unlabeled outputs only; direction accuracy and "
            "$R^2$ are shown only after the marker is fixed."
        )
    elif study_phase == "frozen_initialization_repeat":
        scope_text = (
            "This is a frozen initialization-repeat analysis: the diagnostic was fixed before these "
            "runs, but the data subset and evaluation split are held fixed while only solver "
            "initialization varies. The disagreement marker is computed from unlabeled outputs only; "
            "direction accuracy and $R^2$ are shown only after the marker is fixed. These runs are "
            "repeatability evidence, not an independent generalization test."
        )
    else:
        raise ValueError(f"unknown study_phase: {study_phase}")
    lines = [
        "# ROCA geometry--objective disagreement audit",
        "",
        scope_text,
        "",
        "## Frozen diagnostic",
        "",
        "- Geometric selector: representative signed-volume sign.",
        "- Objective selector: determinant candidate with lower final transport objective.",
        f"- Abstention marker: the two signs disagree and the relative objective gap is at least {min_relative_gap:.3f}.",
        "- This marker is not a replacement selector and was not tuned with labels.",
        "",
        "## Aggregate result",
        "",
        f"- Runs: {summary['n_runs']}",
        f"- Geometry selector accuracy (evaluation only): {summary['geometry_accuracy_all_evaluation_only']:.3f}",
        f"- Material-conflict rejects: {summary['rejected_by_material_conflict']}/{summary['n_seeds']} (coverage {summary['coverage_without_material_conflict']:.3f})",
        f"- Geometry accuracy among non-rejected seeds (evaluation only): {summary['geometry_accuracy_accepted_evaluation_only']:.3f}",
        f"- Wrong geometric selections rejected (evaluation only): {summary['wrong_selections_rejected_evaluation_only']}/{summary['wrong_geometry_selections_evaluation_only']}",
        "",
        "## Per-run diagnostics",
        "",
        "| case | solver seed | subset seed | geometry sign | objective sign | material conflict? | relative objective gap | geometry correct (evaluation only) | selected accuracy | alternative accuracy |",
        "| --- | ---: | ---: | ---: | ---: | :---: | ---: | :---: | ---: | ---: |",
    ]
    for row in records:
        lines.append(
            "| {case_id} | {seed} | {subset_seed} | {geometric_sign:+d} | {objective_sign:+d} | {disagree} | {gap:.4f} | {correct} | {selected:.4f} | {alternative:.4f} |".format(
                case_id=row["case_id"],
                seed=row["seed"],
                subset_seed="--" if row["subset_seed"] is None else row["subset_seed"],
                geometric_sign=row["geometric_sign"],
                objective_sign=row["objective_sign"],
                disagree="yes" if row["objective_conflict_abstain"] else "no",
                gap=row["objective_relative_gap"],
                correct="yes" if row["geometry_matches_oracle_evaluation_only"] else "no",
                selected=row["selected_accuracy_evaluation_only"],
                alternative=row["alternative_accuracy_evaluation_only"],
            )
        )
    lines += [
        "",
        "## Interpretation",
        "",
        summary["interpretation"],
        "Lower transport objective is not a justified ROCA replacement: any disagreement can contain either a geometric error or an objective-preference error.",
        "",
        "## Inputs",
        "",
        *[f"- `{item.as_posix()}`" for item in inputs],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(path: Path, records: list[dict[str, Any]], min_relative_gap: float) -> None:
    fig, axis = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
    case_labels = [row["case_id"] for row in records]
    colors = ["#d97706" if row["objective_conflict_abstain"] else "#0f766e" for row in records]
    axis.bar(case_labels, [row["objective_relative_gap"] for row in records], color=colors)
    axis.set(
        xlabel="run identity",
        ylabel="relative objective gap",
        title="ROCA geometry--objective disagreement diagnostic",
    )
    axis.axhline(min_relative_gap, color="#6b7280", linestyle="--", linewidth=1, label="material-gap threshold")
    axis.legend(fontsize=8, frameon=False)
    axis.tick_params(axis="x", labelrotation=35, labelsize=8)
    axis.text(0.01, 0.98, "orange = disagreement / abstain", transform=axis.transAxes, va="top", fontsize=9)
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--tag", default="roca_candidate_disagreement_audit_20260720")
    parser.add_argument("--min-relative-gap", type=float, default=0.01)
    parser.add_argument(
        "--study-phase",
        choices=["post_hoc", "blind", "frozen_initialization_repeat"],
        default="post_hoc",
        help="Whether inputs developed the rule or remained held out until it was frozen.",
    )
    args = parser.parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    if args.min_relative_gap < 0:
        raise ValueError("min_relative_gap must be non-negative")
    records = audit_payloads(payloads, args.min_relative_gap)
    summary = summarize(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"{args.tag}.json"
    report_path = RESULTS_DIR / f"{args.tag}.md"
    figure_path = FIGURES_DIR / f"{args.tag}.png"
    json_path.write_text(json.dumps({"inputs": [str(path) for path in args.inputs], "min_relative_gap": args.min_relative_gap, "study_phase": args.study_phase, "summary": summary, "records": records}, indent=2) + "\n", encoding="utf-8")
    _write_report(report_path, records, summary, args.inputs, args.min_relative_gap, args.study_phase)
    _plot(figure_path, records, args.min_relative_gap)
    print(f"saved: {json_path}\nsaved: {report_path}\nsaved: {figure_path}")


if __name__ == "__main__":
    main()
