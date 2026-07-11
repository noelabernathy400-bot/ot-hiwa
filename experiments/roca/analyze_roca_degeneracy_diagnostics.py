"""Reliability diagnostics for ROCA synthetic determinant validation.

This script does not change the frozen ROCA selector.  It recomputes the
synthetic determinant-validation grid and records label-free warning features
that may explain when the representative-oriented branch selector is fragile.

Ground-truth determinant is used only after selection to summarize diagnostic
separation between successful and failed cases.
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

from common import json_ready
from run_component_aware import _representative_orientation_selector
from run_roca_synthetic_determinant_validation import (
    _assignments,
    _candidate_transport,
    _make_cloud,
    _representatives,
)
from soft_groups import assignment_entropy


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = SCRIPT_ROOT / "results" / "roca_degeneracy_diagnostics"
FIGURE_ROOT = SCRIPT_ROOT / "figures" / "roca_degeneracy_diagnostics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(50)))
    parser.add_argument("--noise-levels", nargs="+", type=float, default=[0.0, 0.02, 0.05, 0.10])
    parser.add_argument("--true-dets", nargs="+", type=int, default=[1, -1])
    parser.add_argument("--assignment-modes", nargs="+", choices=["oracle_soft", "learned_soft"], default=["oracle_soft", "learned_soft"])
    parser.add_argument("--degeneracy-modes", nargs="+", choices=["normal", "near_degenerate"], default=["normal", "near_degenerate"])
    parser.add_argument("--points-per-cluster", type=int, default=40)
    parser.add_argument("--cluster-std", type=float, default=0.06)
    parser.add_argument("--oracle-confidence", type=float, default=0.97)
    parser.add_argument("--learned-temperature", type=float, default=0.50)
    parser.add_argument("--learned-entropy-weight", type=float, default=0.05)
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def _simplex_matrix(representatives: np.ndarray) -> np.ndarray:
    return (representatives[1:] - representatives[0]).T


def _simplex_diagnostics(representatives: np.ndarray) -> dict[str, float]:
    simplex = _simplex_matrix(representatives)
    singular_values = np.linalg.svd(simplex, compute_uv=False)
    pairwise = []
    for i, j in itertools.combinations(range(len(representatives)), 2):
        pairwise.append(float(np.linalg.norm(representatives[i] - representatives[j])))
    min_singular = float(np.min(singular_values))
    max_singular = float(np.max(singular_values))
    return {
        "volume": float(np.linalg.det(simplex)),
        "abs_volume": float(abs(np.linalg.det(simplex))),
        "min_singular_value": min_singular,
        "max_singular_value": max_singular,
        "condition_number": float(max_singular / max(min_singular, 1e-12)),
        "min_pairwise_distance": float(min(pairwise)),
        "median_pairwise_distance": float(np.median(pairwise)),
    }


def _assignment_diagnostics(assignments: np.ndarray) -> dict[str, float]:
    masses = assignments.sum(axis=0)
    max_probs = assignments.max(axis=1)
    entropy = assignment_entropy(assignments)
    return {
        "mean_entropy": float(entropy.mean()),
        "median_entropy": float(np.median(entropy)),
        "mean_max_probability": float(max_probs.mean()),
        "min_group_mass": float(masses.min()),
        "max_group_mass": float(masses.max()),
        "group_mass_ratio": float(masses.max() / max(masses.min(), 1e-12)),
    }


def _matching_diagnostics(transport: np.ndarray) -> dict[str, Any]:
    scores = []
    for permutation in itertools.permutations(range(transport.shape[1])):
        score = float(sum(transport[row, col] for row, col in enumerate(permutation)))
        scores.append((score, permutation))
    scores.sort(reverse=True, key=lambda item: item[0])
    best_score, best_perm = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else float("nan")
    return {
        "best_matching_score": best_score,
        "second_best_matching_score": second_score,
        "matching_score_margin": float(best_score - second_score),
        "matching_score_relative_margin": float(
            (best_score - second_score) / max(abs(best_score), 1e-12)
        ),
        "best_target_order_by_transport": [int(item) for item in best_perm],
    }


def _warning_v0(record: dict[str, Any]) -> dict[str, Any]:
    """A diagnostic-only warning rule, not a frozen selector.

    The thresholds are intentionally conservative and interpretable.  They are
    not tuned into the ROCA selector and should be validated on future synthetic
    or held-out data before being used as an automatic rejection rule.
    """
    reasons = []
    if record["volume_product_margin"] < 1.0:
        reasons.append("low_oriented_volume_margin")
    if record["source_condition_number"] > 50 or record["target_condition_number"] > 50:
        reasons.append("ill_conditioned_simplex")
    if record["source_assignment_mean_entropy"] > 0.45 or record["target_assignment_mean_entropy"] > 0.45:
        reasons.append("high_assignment_entropy")
    if record["matching_score_relative_margin"] < 0.05:
        reasons.append("ambiguous_group_matching")
    return {
        "warning_v0": bool(reasons),
        "warning_v0_reasons": reasons,
    }


def analyze_case(
    *,
    seed: int,
    true_det: int,
    noise_level: float,
    assignment_mode: str,
    degeneracy_mode: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cloud = _make_cloud(
        seed=seed,
        true_det=true_det,
        noise_level=noise_level,
        degeneracy_mode=degeneracy_mode,
        points_per_cluster=args.points_per_cluster,
        cluster_std=args.cluster_std,
    )
    source = cloud["source"]
    target = cloud["target"]
    labels = cloud["labels"]
    source_assignments = _assignments(source, labels, assignment_mode, seed, args)
    target_assignments = _assignments(target, labels, assignment_mode, seed + 100_000, args)
    transport = _candidate_transport(source_assignments, target_assignments)

    source_reps = _representatives(source, source_assignments)
    target_reps = _representatives(target, target_assignments)
    source_simplex = _simplex_diagnostics(source_reps)
    target_simplex = _simplex_diagnostics(target_reps)
    source_assignment = _assignment_diagnostics(source_assignments)
    target_assignment = _assignment_diagnostics(target_assignments)
    matching = _matching_diagnostics(transport)

    try:
        selector = _representative_orientation_selector(
            source,
            source_assignments,
            target,
            target_assignments,
            [transport, transport],
        )
        selected_det = int(selector["representative_orientation_sign"])
        selector_error = None
    except Exception as exc:  # noqa: BLE001 - preserved as diagnostic evidence
        selector = {}
        selected_det = None
        selector_error = str(exc)

    record: dict[str, Any] = {
        "seed": seed,
        "true_det": true_det,
        "selected_det": selected_det,
        "selected_is_correct": bool(selected_det == true_det) if selected_det is not None else False,
        "noise_level": noise_level,
        "assignment_mode": assignment_mode,
        "degeneracy_mode": degeneracy_mode,
        "selector_error": selector_error,
        "source_volume": source_simplex["volume"],
        "target_volume": target_simplex["volume"],
        "volume_product_margin": float(source_simplex["abs_volume"] * target_simplex["abs_volume"]),
        "source_min_singular_value": source_simplex["min_singular_value"],
        "target_min_singular_value": target_simplex["min_singular_value"],
        "source_condition_number": source_simplex["condition_number"],
        "target_condition_number": target_simplex["condition_number"],
        "source_min_pairwise_distance": source_simplex["min_pairwise_distance"],
        "target_min_pairwise_distance": target_simplex["min_pairwise_distance"],
        "source_assignment_mean_entropy": source_assignment["mean_entropy"],
        "target_assignment_mean_entropy": target_assignment["mean_entropy"],
        "source_assignment_mean_max_probability": source_assignment["mean_max_probability"],
        "target_assignment_mean_max_probability": target_assignment["mean_max_probability"],
        "source_group_mass_ratio": source_assignment["group_mass_ratio"],
        "target_group_mass_ratio": target_assignment["group_mass_ratio"],
        **matching,
        "selector_target_order": selector.get("representative_target_order"),
    }
    record.update(_warning_v0(record))
    return record


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "q25": None, "median": None, "q75": None, "max": None}
    array = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [record for record in records if not record["selected_is_correct"]]
    successes = [record for record in records if record["selected_is_correct"]]
    warned = [record for record in records if record["warning_v0"]]
    caught_failures = [record for record in failures if record["warning_v0"]]
    false_warnings = [record for record in successes if record["warning_v0"]]
    metrics = [
        "volume_product_margin",
        "source_condition_number",
        "target_condition_number",
        "source_assignment_mean_entropy",
        "target_assignment_mean_entropy",
        "matching_score_relative_margin",
    ]
    return {
        "cases": len(records),
        "failures": len(failures),
        "successes": len(successes),
        "warning_v0_count": len(warned),
        "warning_v0_failure_recall": float(len(caught_failures) / len(failures)) if failures else None,
        "warning_v0_precision": float(len(caught_failures) / len(warned)) if warned else None,
        "warning_v0_false_warning_count": len(false_warnings),
        "metric_quantiles": {
            metric: {
                "success": _quantiles([record[metric] for record in successes]),
                "failure": _quantiles([record[metric] for record in failures]),
            }
            for metric in metrics
        },
        "failure_cases": [
            {
                key: record[key]
                for key in [
                    "seed",
                    "true_det",
                    "selected_det",
                    "noise_level",
                    "assignment_mode",
                    "degeneracy_mode",
                    "volume_product_margin",
                    "source_condition_number",
                    "target_condition_number",
                    "source_assignment_mean_entropy",
                    "target_assignment_mean_entropy",
                    "matching_score_relative_margin",
                    "warning_v0_reasons",
                ]
            }
            for record in failures
        ],
    }


def _plot(records: list[dict[str, Any]], output: Path) -> None:
    colors = ["tab:red" if not record["selected_is_correct"] else "tab:blue" for record in records]
    markers = ["x" if record["warning_v0"] else "o" for record in records]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for marker in sorted(set(markers)):
        selected = [i for i, value in enumerate(markers) if value == marker]
        axes[0].scatter(
            [records[i]["volume_product_margin"] for i in selected],
            [records[i]["target_assignment_mean_entropy"] for i in selected],
            c=[colors[i] for i in selected],
            marker=marker,
            alpha=0.7,
            label=f"warning_v0={marker == 'x'}",
        )
        axes[1].scatter(
            [records[i]["source_condition_number"] for i in selected],
            [records[i]["target_condition_number"] for i in selected],
            c=[colors[i] for i in selected],
            marker=marker,
            alpha=0.7,
        )
        axes[2].scatter(
            [records[i]["matching_score_relative_margin"] for i in selected],
            [records[i]["volume_product_margin"] for i in selected],
            c=[colors[i] for i in selected],
            marker=marker,
            alpha=0.7,
        )

    axes[0].set_xscale("log")
    axes[0].set_xlabel("|det(Xrep)| · |det(Yrep)|")
    axes[0].set_ylabel("target assignment mean entropy")
    axes[0].set_title("Volume margin vs assignment entropy")
    axes[0].legend()

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("source simplex condition number")
    axes[1].set_ylabel("target simplex condition number")
    axes[1].set_title("Simplex conditioning")

    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("relative matching margin")
    axes[2].set_ylabel("volume product margin")
    axes[2].set_title("Matching margin vs volume margin")

    fig.suptitle("ROCA synthetic reliability diagnostics")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _write_report(records: list[dict[str, Any]], summary: dict[str, Any], output: Path) -> None:
    lines = [
        "# ROCA degeneracy diagnostics report",
        "",
        "This report is diagnostic only. It does not modify the frozen ROCA selector.",
        "",
        "## Summary",
        "",
        f"- Cases: {summary['cases']}",
        f"- Selector failures: {summary['failures']}",
        f"- warning_v0 flagged cases: {summary['warning_v0_count']}",
        f"- warning_v0 failure recall: {summary['warning_v0_failure_recall']}",
        f"- warning_v0 precision: {summary['warning_v0_precision']}",
        f"- warning_v0 false-warning count: {summary['warning_v0_false_warning_count']}",
        "",
        "## Interpretation",
        "",
        "`warning_v0` is not a final rejection rule. It is a conservative diagnostic proposal based on low oriented-volume margin, ill-conditioned representative simplex, high assignment entropy, or ambiguous group matching.",
        "",
        "## Failure cases",
        "",
        "| seed | true_det | selected_det | noise | assignment | degeneracy | volume_margin | src_cond | tgt_cond | src_entropy | tgt_entropy | match_rel_margin | warning reasons |",
        "|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in summary["failure_cases"]:
        lines.append(
            "| {seed} | {true_det} | {selected_det} | {noise_level} | {assignment_mode} | {degeneracy_mode} | {volume_product_margin:.3e} | {source_condition_number:.3f} | {target_condition_number:.3f} | {source_assignment_mean_entropy:.3f} | {target_assignment_mean_entropy:.3f} | {matching_score_relative_margin:.3e} | {warning_v0_reasons} |".format(
                **record
            )
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in args.seeds:
        for true_det in args.true_dets:
            for noise_level in args.noise_levels:
                for assignment_mode in args.assignment_modes:
                    for degeneracy_mode in args.degeneracy_modes:
                        records.append(
                            analyze_case(
                                seed=seed,
                                true_det=true_det,
                                noise_level=noise_level,
                                assignment_mode=assignment_mode,
                                degeneracy_mode=degeneracy_mode,
                                args=args,
                            )
                        )

    summary = {
        "experiment": "roca_degeneracy_diagnostics",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "parameters": vars(args),
        "summary": _summarize(records),
    }
    suffix = f"_{args.tag}" if args.tag else ""
    raw_path = RESULT_ROOT / f"roca_degeneracy_diagnostics_raw{suffix}.json"
    summary_path = RESULT_ROOT / f"roca_degeneracy_diagnostics_summary{suffix}.json"
    report_path = RESULT_ROOT / f"roca_degeneracy_diagnostics_report{suffix}.md"
    figure_path = FIGURE_ROOT / f"roca_degeneracy_diagnostics{suffix}.png"
    raw_path.write_text(json.dumps(json_ready(records), indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(json_ready(summary), indent=2), encoding="utf-8")
    _write_report(records, summary["summary"], report_path)
    _plot(records, figure_path)
    print(json.dumps(json_ready(summary["summary"]), indent=2))
    print(f"Wrote {raw_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()

