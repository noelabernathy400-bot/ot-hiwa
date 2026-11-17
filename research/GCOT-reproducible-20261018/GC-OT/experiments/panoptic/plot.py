"""Render Panoptic cluster diagnostics and HiWA-style alignment panels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = ["GCOT", "WA", "SA", "ICP", "CORAL"]


def _plot_clusters(ax, points: np.ndarray, labels: np.ndarray) -> None:
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
    data = np.load(aligned_path)
    labels = data["labels"]
    frame_ids = data["frame_ids"]
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), layout="constrained")
    diagnostic = [
        (data["source"], "Source domain: camera 00_00 · PCA", "PC 1", "PC 2"),
        (data["source_tsne"], "Source domain: camera 00_00 · t-SNE", "t-SNE 1", "t-SNE 2"),
        (data["target"], "Target domain: camera 00_01 · PCA", "PC 1", "PC 2"),
        (data["target_tsne"], "Target domain: camera 00_01 · t-SNE", "t-SNE 1", "t-SNE 2"),
    ]
    for ax, (points, title, xlabel, ylabel) in zip(axes.flat, diagnostic):
        scatter = ax.scatter(points[:, 0], points[:, 1], c=frame_ids, cmap="cividis", s=26)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(scatter, ax=ax, label="Frame ID (early → late)")
    fig.suptitle("CMU Panoptic 171204_pose2 · person 0 · frames 141–339", fontweight="bold")
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"pca_tsne_three_groups.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    names = ["Source", "Target", *METHODS]
    clouds = [
        data["source"],
        data["target"],
        *[data[f"aligned_{name.lower()}"] for name in METHODS],
    ]
    fig, axes = plt.subplots(1, len(names), figsize=(25.2, 3.7), layout="constrained")
    for ax, name, cloud in zip(axes, names, clouds):
        _plot_clusters(ax, cloud, labels)
        ax.set_title(name)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"figure2a_panoptic.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    with metrics_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    names = [row["method"] for row in rows]
    recall = [float(row["paired_recall_at_1"]) for row in rows]
    group_accuracy = [float(row["three_group_accuracy_5nn"]) for row in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    ax.bar(x - 0.18, recall, width=0.36, label="Paired frame Recall@1")
    ax.bar(x + 0.18, group_accuracy, width=0.36, label="3-group 5-NN accuracy")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Panoptic cross-camera alignment")
    ax.legend()
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"alignment_metrics.{extension}", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aligned", type=Path)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    run(args.aligned, args.metrics, args.output_dir)


if __name__ == "__main__":
    main()
