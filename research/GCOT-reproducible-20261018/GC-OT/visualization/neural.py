"""Render the neural GCOT comparison from saved results only."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.linalg import sqrtm

from evaluation.metrics import nearest_label_accuracy, paired_neural_truth, regression_metrics
from experiments.neural.core import OUT, ROOT
from experiments.reference import author_utils


def main():
    data_path = ROOT / "data/processed/neural/neural_input.npz"
    archive = ROOT / "results/neural/author_reference/tpami"
    with np.load(data_path) as data, np.load(OUT / "GCOT.npz") as gcot:
        wa = loadmat(archive / "WA.mat", simplify_cells=True)
        dad = loadmat(archive / "DAD.mat", simplify_cells=True)
        clouds = [data["source_raw"], data["Xtr"], data["source_raw"] @ gcot["R"].T,
                  wa["aligned"], dad["aligned"]]
        names = ["Source", "Target", "GCOT", "WA", "DAD"]
        fig, axes = plt.subplots(1, 5, figsize=(18, 3.7), layout="constrained")
        utils = author_utils()
        for index, (ax, name, cloud) in enumerate(zip(axes, names, clouds)):
            labels = data["target_labels"] if name == "Target" else data["source_labels"]
            plt.sca(ax)
            utils.plot_2d_clusters(cloud, labels)
            ax.set_title(name)
        folder = OUT / "figures"
        folder.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "svg"):
            fig.savefig(folder / f"figure2a.{extension}", dpi=220, bbox_inches="tight")
        plt.close(fig)

        fits = {"GCOT": gcot["aligned"][:, :2],
                "WA": wa["aligned_objective"][:, :2], "DAD": dad["aligned"][:, :2]}
        accuracy = {name: nearest_label_accuracy(points, data["target"][:, :2],
                    data["source_labels"], data["target_labels"])
                    for name, points in fits.items()}
        target_raw = np.column_stack((data["Xtr"], np.linalg.norm(data["Xtr"], axis=1)))
        mean = target_raw.mean(axis=0)
        whitening = np.linalg.inv(sqrtm(np.cov(target_raw, rowvar=False)))
        truth = paired_neural_truth(data["Xte"], mean, whitening)
        regression = {name: regression_metrics(points, truth) for name, points in fits.items()}

    result = dict(seed=1, accuracy_1nn_shared_whitened_xy=accuracy,
                  paired_decoding_metrics=regression,
                  labels_policy="Labels are used only after fitting for evaluation and color.")
    (OUT / "statistics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# 神经点云实验", "",
             "Figure 2a 展示 Source、Target、GCOT、WA 和 DAD。真实对应和标签不参与拟合。", "",
             "| 方法 | 1NN准确率 | 配对解码rMSE | 配对解码R² |", "|---|---:|---:|---:|"]
    for name, value in accuracy.items():
        metric = regression[name]
        lines.append(f"| {name} | {value:.4%} | {metric['rmse_relative']:.6f} | {metric['r2']:.6f} |")
    (OUT / "REPORT_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
