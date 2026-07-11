# 2026-07-08 ROCA-HiWA阶段任务split校准文献时间线组会提纲报告

## 目标

按照用户提供的阶段任务，把 ROCA-HiWA 从“有效实验现象”推进为“可解释、可复现、有边界的研究方法”。本次重点包括：

- split 版 confidence calibration；
- 系统文献检索与新颖性定位；
- 项目总时间线与结果总表；
- 10–15 分钟组会汇报提纲；
- 后续探索建议更新；
- 修复第二个 `最优传输` vault 内图片显示问题。

## 新增/修改文件

新增脚本：

- `Projects/最优传输/Experiments/hiwa_python_reproduction/scripts/calibrate_roca_confidence.py`
- `Projects/最优传输/Experiments/hiwa_python_reproduction/scripts/analyze_roca_confidence_calibration.py`

修改测试：

- `Projects/最优传输/Experiments/hiwa_python_reproduction/tests/test_diagnostics.py`

新增结果：

- `Projects/最优传输/Experiments/hiwa_python_reproduction/results/roca_confidence_calibration/roca_confidence_calibration_raw_split_2026_07_08.json`
- `Projects/最优传输/Experiments/hiwa_python_reproduction/results/roca_confidence_calibration/roca_confidence_calibration_summary_split_2026_07_08.json`
- `Projects/最优传输/Experiments/hiwa_python_reproduction/results/roca_confidence_calibration/roca_confidence_calibration_report_split_2026_07_08.md`
- `Projects/最优传输/Experiments/hiwa_python_reproduction/results/roca_confidence_calibration/confidence_rule_comparison_table_split_2026_07_08.csv`
- `Projects/最优传输/Experiments/hiwa_python_reproduction/results/roca_confidence_calibration/synthetic_low_confidence_cases_split_2026_07_08.md`

新增图像：

- `Projects/最优传输/Experiments/hiwa_python_reproduction/figures/roca_confidence_calibration/confidence_recall_precision_curve_split_2026_07_08.png`
- `Projects/最优传输/Experiments/hiwa_python_reproduction/figures/roca_confidence_calibration/confidence_warning_rate_by_rule_split_2026_07_08.png`
- `Projects/最优传输/Experiments/hiwa_python_reproduction/figures/roca_confidence_calibration/confidence_metric_distributions_split_2026_07_08.png`

新增/更新中文笔记：

- `Projects/最优传输/最优传输/文本笔记/代码方面/ROCA-HiWA系统文献检索与新颖性定位.md`
- `Projects/最优传输/最优传输/文本笔记/代码方面/Soft-Prototype-HiWA到ROCA-HiWA项目总时间线与结果总表.md`
- `Projects/最优传输/最优传输/文本笔记/代码方面/ROCA-HiWA阶段性组会汇报提纲.md`
- `Projects/最优传输/最优传输/文本笔记/代码方面/ROCA-HiWA后续探索建议.md`

图片显示修复：

- 将关键 PNG 复制到 `Projects/最优传输/最优传输/文本笔记/代码方面/_attachments/`；
- 将笔记图片引用改为 Obsidian 内部文件名嵌入。

## 关键结果

- split 设置：calibration seeds 0-24；validation seeds 25-49。
- calibration split 中 failure 数为 0，因此不能在 calibration split 中按 failure recall 调阈值。
- validation split 中 failure 数为 8，全部为 seed 42、learned_soft、near_degenerate。
- 推荐候选规则：`frozen_condition_number_gt_80`，即 representative simplex condition number > 80。
- validation split：
  - failure recall = 1.0；
  - warning precision = 1.0；
  - false warning rate = 0.0；
  - warning rate = 0.01；
  - non-warning determinant recovery = 1.0。
- 真实固定阈值检查：
  - seeds 50-69：20 cases，warning rate = 0.0；
  - coordinate flip seeds 70-79：30 cases，warning rate = 0.0。

## 文献定位结论

- determinant-constrained Procrustes、Kabsch reflection correction、signed volume / oriented simplex、condition number degeneracy diagnostic 都是已有数学或算法工具。
- 当前未发现“soft representatives + oriented simplex/signed volume + HiWA/hierarchical OT/neural alignment determinant branch selection”的完全同型方法。
- TACO 原文仍需准确题名和全文核验；不能夸大 novelty。

## 验证

- `python -m py_compile scripts\calibrate_roca_confidence.py scripts\analyze_roca_confidence_calibration.py`：通过。
- `python scripts\calibrate_roca_confidence.py`：完成。
- `python scripts\analyze_roca_confidence_calibration.py`：完成。
- `python -m unittest discover -s tests -v`：24/24 通过。
- 已检查第二个 `最优传输` vault 内所有 PNG 嵌入均能找到对应文件。

## 风险与下一步

- 阈值 80 仍是当前 synthetic / validation 下的候选规则，不是普适理论阈值。
- TACO、neural manifold alignment、BMI alignment、Wasserstein Procrustes 的精读仍待补。
- 下一步建议优先做外部数据或 session-to-session 最小验证，而不是 K=5、聚类消融或追分。

