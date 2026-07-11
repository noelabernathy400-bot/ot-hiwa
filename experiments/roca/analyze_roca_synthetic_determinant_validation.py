"""Plot and summarize ROCA synthetic determinant validation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import json_ready


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = SCRIPT_ROOT / "results" / "roca_synthetic_determinant_validation"
FIGURE_ROOT = SCRIPT_ROOT / "figures" / "roca_synthetic_determinant_validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="roca_synthetic_determinant_validation_raw.json")
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def _load_records(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["records"]


def _group_accuracy(records: list[dict], fields: list[str]) -> list[dict]:
    groups = {}
    for record in records:
        key = tuple(record[field] for field in fields)
        groups.setdefault(key, []).append(record)
    output = []
    for key, selected in groups.items():
        row = {field: value for field, value in zip(fields, key)}
        row["cases"] = len(selected)
        row["accuracy"] = float(np.mean([item["selected_is_correct"] for item in selected]))
        row["degenerate_count"] = int(sum(item["degeneracy_flag"] for item in selected))
        row["median_margin"] = float(np.median([item["oriented_volume_margin"] for item in selected]))
        output.append(row)
    return sorted(output, key=lambda row: tuple(str(row[field]) for field in fields))


def _plot_accuracy_by_noise(records: list[dict], out_path: Path) -> None:
    rows = _group_accuracy(records, ["degeneracy_mode", "assignment_mode", "noise_level"])
    fig, axis = plt.subplots(figsize=(8.2, 5.0), facecolor="white", constrained_layout=True)
    for degeneracy in sorted({row["degeneracy_mode"] for row in rows}):
        for assignment in sorted({row["assignment_mode"] for row in rows}):
            selected = [
                row for row in rows
                if row["degeneracy_mode"] == degeneracy and row["assignment_mode"] == assignment
            ]
            axis.plot(
                [row["noise_level"] for row in selected],
                [row["accuracy"] for row in selected],
                marker="o",
                label=f"{assignment}, {degeneracy}",
            )
    axis.set_title("Synthetic determinant recovery by noise")
    axis.set_xlabel("noise level")
    axis.set_ylabel("selected_det == true_det")
    axis.set_ylim(-0.05, 1.05)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.savefig(out_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _plot_accuracy_by_assignment(records: list[dict], out_path: Path) -> None:
    rows = _group_accuracy(records, ["assignment_mode", "degeneracy_mode"])
    labels = [f"{row['assignment_mode']}\n{row['degeneracy_mode']}" for row in rows]
    fig, axis = plt.subplots(figsize=(7.2, 4.8), facecolor="white", constrained_layout=True)
    axis.bar(labels, [row["accuracy"] for row in rows], color="#4c78a8")
    axis.set_title("Synthetic determinant recovery by assignment mode")
    axis.set_ylabel("accuracy")
    axis.set_ylim(0, 1.05)
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _plot_margin(records: list[dict], out_path: Path) -> None:
    normal = [record["oriented_volume_margin"] for record in records if record["degeneracy_mode"] == "normal"]
    near = [record["oriented_volume_margin"] for record in records if record["degeneracy_mode"] == "near_degenerate"]
    fig, axis = plt.subplots(figsize=(7.0, 4.8), facecolor="white", constrained_layout=True)
    axis.boxplot([normal, near], tick_labels=["normal", "near_degenerate"], patch_artist=True)
    axis.set_yscale("log")
    axis.set_title("Oriented-volume margin")
    axis.set_ylabel("|det(X_rep) det(Y_rep)|")
    axis.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    records = _load_records(RESULT_ROOT / args.raw)
    suffix = f"_{args.tag}" if args.tag else ""
    _plot_accuracy_by_noise(records, FIGURE_ROOT / f"synthetic_det_accuracy_by_noise{suffix}.png")
    _plot_accuracy_by_assignment(records, FIGURE_ROOT / f"synthetic_det_accuracy_by_assignment_mode{suffix}.png")
    _plot_margin(records, FIGURE_ROOT / f"synthetic_oriented_volume_margin{suffix}.png")
    analysis = {
        "by_noise": _group_accuracy(records, ["degeneracy_mode", "assignment_mode", "noise_level"]),
        "by_assignment": _group_accuracy(records, ["assignment_mode", "degeneracy_mode"]),
        "by_true_det": _group_accuracy(records, ["true_det", "degeneracy_mode", "assignment_mode"]),
    }
    (RESULT_ROOT / f"roca_synthetic_determinant_validation_analysis{suffix}.json").write_text(
        json.dumps(json_ready(analysis), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(json_ready(analysis), ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
