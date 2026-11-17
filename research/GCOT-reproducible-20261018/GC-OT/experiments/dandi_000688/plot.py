"""HiWA neural Figure 2a-style panels and metrics for DANDI 000688."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .config import N_GROUPS


METHODS = ["GCOT", "WA", "SA", "ICP", "CORAL"]


def _plot_2d_clusters(ax, points: np.ndarray, labels: np.ndarray) -> None:
    """Match the HiWA neural demo's ``plot_2d_clusters`` convention."""
    for label in np.unique(labels):
        selected = labels == label
        ax.plot(
            points[selected, 0],
            points[selected, 1],
            linestyle="",
            marker=".",
            markersize=15,
        )


def run(aligned_path: Path, metrics_path: Path, output_dir: Path) -> None:
    if not aligned_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError("Run the preparation and comparison stages before plotting")
    data = np.load(aligned_path)
    target = data["target"]
    source_labels, target_labels = data["source_labels"], data["target_labels"]
    output_dir.mkdir(parents=True, exist_ok=True)

    names = ["Source", "Target", *METHODS]
    clouds = [
        data["source"],
        target,
        *[data[f"aligned_{name.lower()}"] for name in METHODS],
    ]
    labels = [source_labels, target_labels, *([source_labels] * len(METHODS))]
    fig, axes = plt.subplots(1, len(names), figsize=(25.2, 3.7), layout="constrained")
    for ax, name, cloud, cloud_labels in zip(axes, names, clouds, labels):
        _plot_2d_clusters(ax, cloud, cloud_labels)
        ax.set_title(name)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"figure2a_dandi.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    with metrics_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    names = [r["method"] for r in rows]
    acc = [float(r["direction_accuracy_5nn"]) for r in rows]
    match = [float(r["cluster_matching_accuracy"]) for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar(x - 0.18, acc, width=0.36, label="5-NN direction accuracy")
    ax.bar(x + 0.18, match, width=0.36, label="8-cluster matching accuracy")
    ax.axhline(
        1 / N_GROUPS,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"chance (1/{N_GROUPS})",
    )
    ax.set_ylim(0, 1.02)
    ax.set_xticks(x, names)
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.set_title("Unsupervised cross-day alignment (direction labels used only here)")
    fig.savefig(output_dir / "alignment_accuracy.png", dpi=220)
    fig.savefig(output_dir / "alignment_accuracy.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("aligned", type=Path)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    run(args.aligned, args.metrics, args.output_dir)


if __name__ == "__main__":
    main()
