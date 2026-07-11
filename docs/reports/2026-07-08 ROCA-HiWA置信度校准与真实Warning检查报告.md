# 2026-07-08 ROCA-HiWA置信度校准与真实Warning检查报告

## 目标

在不改变 ROCA selector、不扩大真实 seeds 追分的前提下，只用 synthetic diagnostics 校准低置信 warning 阈值，并固定阈值检查真实 ROCA seeds 50-69 与 coordinate flip seeds 70-79 的 warning rate。

## 改动文件

新增脚本：

- `Projects/最优传输/Experiments/hiwa_python_reproduction/scripts/calibrate_roca_confidence_thresholds.py`

修改测试：

- `Projects/最优传输/Experiments/hiwa_python_reproduction/tests/test_diagnostics.py`

新增结果：

- `Projects/最优传输/Experiments/hiwa_python_reproduction/results/roca_confidence_calibration/roca_confidence_calibration_raw.json`
- `Projects/最优传输/Experiments/hiwa_python_reproduction/results/roca_confidence_calibration/roca_confidence_calibration_summary.json`
- `Projects/最优传输/Experiments/hiwa_python_reproduction/results/roca_confidence_calibration/roca_confidence_calibration_report.md`
- `Projects/最优传输/Experiments/hiwa_python_reproduction/figures/roca_confidence_calibration/roca_confidence_calibration_real_warning_check.png`

新增中文笔记：

- `Projects/最优传输/最优传输/文本笔记/代码方面/ROCA-HiWA置信度校准与真实Warning检查-2026-07-08.md`

## 关键结果

synthetic 校准选出的固定规则：

- `max(source_condition_number, target_condition_number) > 80`

synthetic calibration：

- cases = 1600
- failures = 8
- warning_count = 8
- caught_failures = 8
- false_warning_count = 0
- failure_recall = 1.0
- precision = 1.0

真实固定阈值检查：

- representative_confirm seeds 50-69：20 cases，warning_count = 0
- coordinate_flip seeds 70-79：30 cases，warning_count = 0
- 合计：50 cases，warning_rate = 0.0

## 关键判断

当前真实 ROCA 成功证据没有落入 synthetic 校准出的低置信区域。这个结果增强了 ROCA 的可信边界，但不构成普适性证明。

## 标签泄漏检查

- 阈值校准只使用 synthetic ground truth。
- 真实 warning check 不使用 direction label、accuracy 或 $R^2$。
- 真实数据只使用 representatives、assignments、transport matrices 的无标签几何指标。

## 验证

- `python -m py_compile scripts\calibrate_roca_confidence_thresholds.py`：通过。
- `python scripts\calibrate_roca_confidence_thresholds.py`：完成。
- `python -m unittest discover -s tests -v`：20/20 通过。

## 风险

- 阈值 80 来自当前 synthetic grid，不能直接当普适理论阈值。
- 目前还没有外部真实数据集验证 warning rule。
- 后续若使用 confidence-aware ROCA，应把它写成“诊断/拒绝选择候选机制”，而不是替代主 selector 的性能调参工具。

## 下一步建议

1. 在更多 synthetic degeneracy 形态上验证 $\kappa>80$。
2. 在外部真实数据或 session-to-session alignment 中报告 fixed warning rate。
3. 在相关工作中补充 registration / Procrustes 的 degeneracy rejection 与 condition-number 讨论。

