"""Rebuild Seed 7 confusion diagnostics from saved rotations without refitting.

This script repairs the label-axis and report-embedding problems in the first
diagnostic report while preserving the original experiment outputs.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import scipy
import sklearn

from common import FIGURES_DIR, RESULTS_DIR, write_json
from diagnose_seed7 import _confusion_fig, _load_data, _nn_error_analysis, _nn_predict


SEED = 7
OUT_DIR = RESULTS_DIR / "soft_neural_pure_annealing_v2"
FIG_DIR = FIGURES_DIR / "soft_neural_pure_annealing_v2"


def main() -> None:
    source_path = RESULTS_DIR / "soft_neural_annealing.json"
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    (_, neural_3d, movement_3d, _, _, _, _) = _load_data()
    from run_neural import load_demo

    data = load_demo()
    true_labels = data["test_labels"].ravel().astype(int)
    movement_labels = data["train_labels"].ravel().astype(int)
    methods = []
    for record in payload["results"]:
        if int(record["seed"]) != SEED:
            continue
        method = record["method"]
        if method == "annealed_soft":
            method = f"annealed_tau{float(record['temperature']):g}"
        if method not in {
            "prototype_hard",
            "fixed_soft_warm",
            "annealed_tau0.25",
            "annealed_tau0.35",
            "annealed_tau0.5",
        }:
            continue
        methods.append((method, record))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    analyses = {}
    result_index = {}
    for method, record in methods:
        aligned = (np.asarray(record["rotation_R"]) @ neural_3d.T).T
        predicted = _nn_predict(aligned, movement_3d, movement_labels)
        analysis = _nn_error_analysis(true_labels, predicted)
        analyses[method] = analysis
        result_index[method] = {
            "direction_accuracy": record["after_direction_accuracy"],
            "movement_r2": record["after_velocity_r2"],
        }
        _confusion_fig(
            true_labels,
            predicted,
            f"Seed 7 — {method} (accuracy={record['after_direction_accuracy']:.3f})",
            FIG_DIR / f"seed7_confusion_{method}.png",
        )

    write_json(
        OUT_DIR / "seed7_nn_error_analysis_v2.json",
        {
            "experiment": "seed7_diagnostic_rebuild_from_saved_rotations",
            "source_results": str(source_path),
            "seed": SEED,
            "class_labels": np.unique(true_labels).astype(int).tolist(),
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "metrics": result_index,
            "analyses": analyses,
        },
    )

    image_lines = "\n\n".join(
        f"### {method}\n\n"
        f"![](../../figures/soft_neural_pure_annealing_v2/seed7_confusion_{method}.png)"
        for method, _ in methods
    )
    failed = analyses["annealed_tau0.5"]
    top = failed["top_errors"][0]
    report = f"""# Seed 7 诊断修正版

## 修正范围

本报告不重新拟合模型，而是读取 `soft_neural_annealing.json` 中保存的旋转矩阵，
重建对齐结果和最近邻预测。原报告把混淆矩阵行号 0–3 当成真实方向标签；
本版明确使用原始方向标签 `{failed['class_labels']}`。

## 核心结论

- prototype-hard 准确率：{result_index['prototype_hard']['direction_accuracy']:.4f}
- fixed-soft-warm 准确率：{result_index['fixed_soft_warm']['direction_accuracy']:.4f}
- annealed-$\\tau=0.5$ 准确率：{result_index['annealed_tau0.5']['direction_accuracy']:.4f}
- 最大单项错误：真实方向 {top['true_direction']} 被预测成方向
  {top['predicted_as']}，共 {top['count']} 个样本。

因此旧报告中的“方向 0 → 方向 1”只是矩阵索引解释；实际语义是
“方向 {top['true_direction']} → 方向 {top['predicted_as']}”。

## 混淆矩阵

{image_lines}

## 证据文件

- 原始模型输出：`../soft_neural_annealing.json`
- 修正后的结构化诊断：`seed7_nn_error_analysis_v2.json`
- 图：`../../figures/soft_neural_pure_annealing_v2/`
"""
    (OUT_DIR / "seed7_diagnostic_report_v2.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
