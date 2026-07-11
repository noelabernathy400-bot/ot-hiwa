# ROCA-HiWA置信度校准与真实Warning检查-2026-07-08

## 1. 这一步做什么

上一份笔记 [[ROCA-HiWA退化检测与低置信诊断-2026-07-08]] 发现，ROCA synthetic validation 的 8 个失败案例都有明显无标签低置信信号。

本次继续做一件更干净的事情：

1. 只用 synthetic diagnostics 校准一个固定 warning 阈值；
2. 固定后，再拿到真实 ROCA seeds 50-69 和 coordinate flip seeds 70-79 上检查 warning rate；
3. 不使用真实 direction accuracy 或 $R^2$ 调阈值；
4. 不改变 ROCA selector。

这一步回答的是：真实数据上 ROCA 成功的那些 case，是否落在 synthetic 发现的低置信区域里？

## 2. 新增脚本

脚本：

`D:\ai\数学ai\Projects\最优传输\Experiments\hiwa_python_reproduction\scripts\calibrate_roca_confidence_thresholds.py`

输入：

- synthetic diagnostics：`results/roca_degeneracy_diagnostics/roca_degeneracy_diagnostics_raw.json`
- 真实确认结果：`results/component_aware_soft_hiwa_representative_confirm_seeds_50_69.json`
- 坐标翻转结果：`results/coordinate_flip_validation_combined_source_axes_seeds_70_79.json`

输出：

- `results/roca_confidence_calibration/roca_confidence_calibration_raw.json`
- `results/roca_confidence_calibration/roca_confidence_calibration_summary.json`
- `results/roca_confidence_calibration/roca_confidence_calibration_report.md`
- `figures/roca_confidence_calibration/roca_confidence_calibration_real_warning_check.png`

运行命令：

```bash
python scripts\calibrate_roca_confidence_thresholds.py
```

## 3. 校准方式

校准只在 synthetic diagnostics 上做。候选 warning 仍然是 OR-rule：

- volume product margin 太低；
- simplex condition number 太高；
- assignment entropy 太高；
- matching relative margin 太低。

网格搜索目标：

1. synthetic failure recall 必须为 1.0；
2. 在满足 1.0 recall 的规则里，选择 warning_count 最少；
3. 若并列，再选择 precision 更高；
4. 规则保持简单、可解释。

最终选出的规则非常简单：

$$
\max(\kappa_X,\kappa_Y)>80.
$$

也就是说，当源域或目标域 representative simplex 的 condition number 超过 80 时，标记为低置信。

这里：

$$
\kappa=\frac{\sigma_1}{\sigma_3},
$$

其中 $\sigma_1$ 和 $\sigma_3$ 是四面体 simplex 矩阵的最大和最小奇异值。

## 4. Synthetic 校准结果

| 指标 | 数值 |
|---|---:|
| synthetic cases | 1600 |
| synthetic failures | 8 |
| warning_count | 8 |
| caught_failures | 8 |
| false_warning_count | 0 |
| failure_recall | 1.0 |
| precision | 1.0 |

解释：

- 8 个 synthetic 失败全部被 $\kappa>80$ 抓住；
- synthetic 上没有误警告成功 case；
- 这比上一版 `warning_v0` 更干净。

但必须谨慎：这个阈值是在当前 synthetic grid 上校准的，还不是普适理论常数。

## 5. 固定阈值后的真实数据检查

固定阈值后，检查两个真实结果集合：

1. representative confirm seeds 50-69；
2. coordinate flip seeds 70-79，三个 source-axis flip 条件。

结果：

| 数据集合 | cases | warning_count | warning_rate |
|---|---:|---:|---:|
| representative_confirm | 20 | 0 | 0.0 |
| coordinate_flip | 30 | 0 | 0.0 |
| 合计 | 50 | 0 | 0.0 |

这意味着：当前真实 ROCA 成功证据没有落入 synthetic 校准出的低置信区域。

## 6. 图像解释

![[ROCA置信度真实Warning检查-2026-07-08.png]]

图中蓝点是 fixed threshold 下未 warning 的真实 case。没有红点，说明真实 seeds 50-69 和 coordinate flip seeds 70-79 都没有触发 $\kappa>80$ 的低置信警告。

## 7. 这增强了什么证据

这一步增强的是 ROCA-HiWA 的“可靠性边界”：

- synthetic 失败可以被一个无标签 condition-number warning 识别；
- 真实成功证据没有落入这个 warning 区域；
- 因此真实 ROCA 结果不是靠接近退化的四面体勉强选对；
- 但这仍然不是普适性证明。

换句话说，ROCA 当前不只是“跑出来 20/20 和 30/30”，而是：

1. 知道什么情况下可能失败；
2. synthetic 上能提前识别失败；
3. 真实当前证据没有触发这类失败信号。

## 8. 标签泄漏检查

校准阶段：

- 使用 synthetic true_det / selected_is_correct 来选择 warning 阈值；
- 这是 synthetic ground-truth calibration，允许作为方法开发诊断。

真实检查阶段：

- 不使用真实 direction label；
- 不使用真实 direction accuracy；
- 不使用真实 $R^2$；
- 只使用 source/target representatives、assignments、transport matrices 的无标签几何指标。

因此真实 warning rate 检查没有标签泄漏。

## 9. 当前不能声称什么

不能说：

- $\kappa>80$ 是普适阈值；
- confidence-aware ROCA 已经是最终方法；
- 所有真实数据都不会触发 warning；
- 只要不触发 warning 就一定正确；
- 这个结果已经替代外部数据集验证。

更准确的表述是：

在当前 synthetic calibration 和真实 ROCA 检查中，simplex condition number 是一个非常强的失败预警信号；固定 $\kappa>80$ 后，当前真实 ROCA 证据未触发低置信 warning。

## 10. 下一步建议

现在可以把 confidence-aware ROCA 写成一个“候选扩展”：

- 主方法仍是 ROCA branch selector；
- 附加报告 confidence diagnostics；
- 如果 $\kappa$ 过大，则提示 representative simplex 退化，不应强行解释分支选择。

下一步最值得做：

1. 在更多 synthetic degeneracy settings 上验证 $\kappa>80$ 是否稳定；
2. 在其他真实数据/跨 session 数据上只报告 warning rate；
3. 文献检索中补充 registration / Procrustes 中的 condition-number 或 degeneracy rejection 相关工作。
