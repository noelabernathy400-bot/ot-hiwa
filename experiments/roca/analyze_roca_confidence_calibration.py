"""Plot split-based ROCA confidence calibration outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = SCRIPT_ROOT / "results" / "roca_confidence_calibration"
FIGURE_ROOT = SCRIPT_ROOT / "figures" / "roca_confidence_calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=RESULT_ROOT / "roca_confidence_calibration_raw_split_2026_07_08.json",
    )
    parser.add_argument("--tag", default="split_2026_07_08")
    return parser.parse_args()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validation_rows(payload: dict) -> list[dict]:
    return [row for row in payload["rule_comparison"] if row["split"] == "validation"]


def _plot_recall_precision(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for row in rows:
        recall = row["failure_recall"]
        precision = row["warning_precision"]
        if recall is None or precision is None:
            continue
        ax.scatter(recall, precision, s=60)
        if row["rule"] in {"warning_v0_existing", "frozen_condition_number_gt_80"}:
            ax.annotate(row["rule"], (recall, precision), fontsize=8)
    ax.set_xlabel("failure recall")
    ax.set_ylabel("warning precision")
    ax.set_title("Validation recall-precision by warning rule")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_warning_rate(rows: list[dict], out_path: Path) -> None:
    rows = sorted(rows, key=lambda row: row["warning_rate"] if row["warning_rate"] is not None else 0)
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [row["rule"].replace("_", "\n") for row in rows]
    values = [row["warning_rate"] or 0.0 for row in rows]
    ax.bar(range(len(rows)), values)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("warning rate")
    ax.set_title("Validation warning rate by rule")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_metric_distributions(payload: dict, out_path: Path) -> None:
    records = []
    diagnostics_path = (
        SCRIPT_ROOT
        / "results"
        / "roca_degeneracy_diagnostics"
        / "roca_degeneracy_diagnostics_raw.json"
    )
    if diagnostics_path.exists():
        records = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    metrics = [
        "volume_product_margin",
        "source_condition_number",
        "matching_score_relative_margin",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes, metrics):
        success = [r[metric] for r in records if r["selected_is_correct"]]
        failure = [r[metric] for r in records if not r["selected_is_correct"]]
        ax.hist(success, bins=30, alpha=0.7, label="success")
        if failure:
            ax.hist(failure, bins=10, alpha=0.9, label="failure")
        ax.set_title(metric)
        ax.legend()
    fig.suptitle("Synthetic metric distributions")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    payload = _load(args.raw)
    rows = _validation_rows(payload)
    suffix = f"_{args.tag}" if args.tag else ""
    outputs = {
        "recall_precision": FIGURE_ROOT / f"confidence_recall_precision_curve{suffix}.png",
        "warning_rate": FIGURE_ROOT / f"confidence_warning_rate_by_rule{suffix}.png",
        "metric_distributions": FIGURE_ROOT / f"confidence_metric_distributions{suffix}.png",
    }
    _plot_recall_precision(rows, outputs["recall_precision"])
    _plot_warning_rate(rows, outputs["warning_rate"])
    _plot_metric_distributions(payload, outputs["metric_distributions"])
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()

