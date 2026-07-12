"""Summarize paired Hard/Soft/ROCA experiment artifacts without label-based selection."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
FIGURES = ROOT / "experiments" / "figures"
METHODS = ("hard_hiwa", "soft_gcot_full", "soft_gcot_roca")


def _summary(delta: np.ndarray) -> dict[str, float | int | list[float]]:
    rng = np.random.default_rng(0)
    draws = rng.choice(delta, size=(10_000, len(delta)), replace=True).mean(axis=1)
    return {
        "mean": float(delta.mean()),
        "median": float(np.median(delta)),
        "sample_std": float(delta.std(ddof=1)),
        "positive_count": int(np.sum(delta > 0)),
        "n": int(len(delta)),
        "bootstrap_95_ci": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--no-sync-results", action="store_true")
    args = parser.parse_args()
    rows: list[dict] = []
    for path in args.inputs:
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["results"])
    by_method = {method: {int(row["seed"]): row for row in rows if row["method"] == method} for method in METHODS}
    seeds = sorted(set.intersection(*(set(by_method[method]) for method in METHODS)))
    if not seeds:
        raise ValueError("no seeds with all paired methods")
    metrics = ("after_direction_accuracy", "after_velocity_r2")
    comparisons: dict[str, dict] = {}
    for method in METHODS[1:]:
        comparisons[method] = {}
        for metric in metrics:
            delta = np.asarray([by_method[method][seed][metric] - by_method["hard_hiwa"][seed][metric] for seed in seeds])
            comparisons[method][metric] = _summary(delta)
    output = {
        "inputs": [str(path) for path in args.inputs],
        "seeds": seeds,
        "comparisons_vs_hard": comparisons,
        "label_usage": "labels appear only in stored final metrics; this script performs no model, branch, or hyperparameter selection",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / f"paired_analysis_{args.tag}.json"
    figure_path = FIGURES / f"paired_analysis_{args.tag}.png"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    for axis, metric, label in zip(axes, metrics, ("direction accuracy", "movement R²")):
        hard = np.asarray([by_method["hard_hiwa"][seed][metric] for seed in seeds])
        for method, color, name in (("soft_gcot_full", "#377eb8", "Soft-GCOT HiWA"), ("soft_gcot_roca", "#984ea3", "Soft-GCOT HiWA + ROCA")):
            values = np.asarray([by_method[method][seed][metric] for seed in seeds])
            axis.scatter(hard, values, color=color, label=name, alpha=0.8)
        lo, hi = np.nanmin(hard), np.nanmax(hard)
        axis.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1)
        axis.set(xlabel=f"Hard HiWA {label}", ylabel=label)
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    if not args.no_sync_results:
        subprocess.run([sys.executable, str(ROOT / "tools" / "sync_result_artifacts.py"), str(json_path), str(figure_path), "--message", "record paired soft-gcot analysis"], check=True)


if __name__ == "__main__":
    main()
