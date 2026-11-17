"""Render synthetic GCOT comparisons from saved results only."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from experiments.synthetic.core import CONFIGS, METHODS, ROOT, SYNTHETIC_OUT, sha


COLORS = ("#0072BD", "#D95319", "#7E2F8E", "#77AC30", "#EDB120")
MARKERS = ("o", "d", "p", "h", "<")


def save(fig, name):
    folder = SYNTHETIC_OUT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(folder / f"{name}.{extension}", dpi=220,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def cdf(ax, errors, groups, dimension, fontsize=12, markersize=7):
    probability = np.arange(1, len(errors) + 1) / len(errors)
    for index, method in enumerate(METHODS):
        ax.plot(np.sort(errors[:, index]), probability, label=method,
                color=COLORS[index], marker=MARKERS[index], markevery=3,
                linewidth=2, markersize=markersize)
    ax.set(xlim=(0, 2), ylim=(0, 1), xlabel="Alignment error",
           ylabel="Cumulative probability", title=f"S={groups}, D={dimension}")
    ax.grid(True)
    ax.tick_params(labelsize=fontsize)
    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)
    ax.title.set_size(fontsize)


def collect():
    results, statistics, rows = {}, [], []
    for groups, dimension in CONFIGS:
        collected = []
        for seed in range(1, 51):
            path = SYNTHETIC_OUT / "runs" / f"S{groups:02d}_D{dimension:02d}_seed{seed:02d}.npz"
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved["methods"], METHODS)
                error = saved["errors"]
                if error.shape != (len(METHODS),) or not np.isfinite(error).all():
                    raise ValueError(f"Invalid saved errors: {path}")
            collected.append(error)
            rows.append(dict(S=groups, D=dimension, seed=seed,
                             **dict(zip(METHODS, error.tolist()))))
        errors = np.asarray(collected)
        results[groups, dimension] = errors
        np.save(SYNTHETIC_OUT / f"figure1_S{groups}_D{dimension}_errors.npy", errors)
        savemat(SYNTHETIC_OUT / f"figure1_S{groups}_D{dimension}_errors.mat",
                dict(rMSE_results=errors.T, methods=np.array(METHODS, dtype=object),
                     S=groups, D=dimension, seeds=np.arange(1, 51)))
        statistics.append(dict(
            S=groups, D=dimension, n=len(errors),
            method_means=dict(zip(METHODS, errors.mean(axis=0).tolist())),
            method_medians=dict(zip(METHODS, np.median(errors, axis=0).tolist())),
            errors_above_plot_limit=dict(zip(METHODS, (errors > 2).sum(axis=0).tolist())),
        ))
    with (SYNTHETIC_OUT / "per_trial_errors.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return results, statistics


def render(results):
    for (groups, dimension), errors in results.items():
        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
        cdf(ax, errors, groups, dimension, fontsize=20, markersize=10)
        ax.legend(loc="upper right", fontsize=11, frameon=True, framealpha=1)
        save(fig, f"figure1_S{groups}_D{dimension}")
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), layout="constrained")
    for ax, ((groups, dimension), errors) in zip(axes.flat, results.items()):
        cdf(ax, errors, groups, dimension)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(METHODS),
               frameon=False, fontsize=13)
    save(fig, "figure1_all_six")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, key, panel in zip(axes, ((5, 6), (2, 2)), ("e", "f")):
        cdf(ax, results[key], *key)
        ax.set_title(f"({panel}) S={key[0]}, D={key[1]}")
    fig.legend(handles, labels, loc="outside lower center", ncol=len(METHODS), frameon=False)
    save(fig, "figure1ef")


def main():
    results, statistics = collect()
    render(results)
    payload = {"synthetic": statistics}
    (SYNTHETIC_OUT / "statistics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = dict(
        experiment="synthetic", configs=CONFIGS, seeds=list(range(1, 51)), methods=METHODS,
        scaling="X/sqrt(D), penalty mu/D, matching the HiWA synthetic protocol.",
        source_hashes={str(path.relative_to(ROOT)): sha(path) for path in (
            ROOT / "experiments/synthetic/core.py", Path(__file__),
            ROOT / "gcot/alignment.py", ROOT / "gcot/grouping.py", ROOT / "gcot/numerics.py")})
    (SYNTHETIC_OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    lines = ["# 合成实验", "", "六组 S/D 配置，每组 50 个随机种子。", "",
             "| S | D | GCOT 平均误差 | GCOT 中位误差 |", "|---:|---:|---:|---:|"]
    for row in statistics:
        lines.append(f"| {row['S']} | {row['D']} | {row['method_means']['GCOT']:.6f} | "
                     f"{row['method_medians']['GCOT']:.6f} |")
    lines += ["", "完整统计见 `statistics.json` 和 `per_trial_errors.csv`。"]
    (SYNTHETIC_OUT / "REPORT_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    main()
