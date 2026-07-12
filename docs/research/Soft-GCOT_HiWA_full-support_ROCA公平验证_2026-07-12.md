# Soft-GCOT HiWA full-support ROCA 公平验证

_冻结同-prototype 对照与 full-support ROCA；seeds 80–89；2026-07-12_

---

## 结论

此验证修复了此前 Hard HiWA 与 Soft-GCOT HiWA 分别学习 prototypes 的归因问题：Hard labels 现在直接由最终 soft assignments 的 `argmax` 得到。ROCA 两个 determinant 候选均使用 full support。seeds 80–89 在这些新规则确定后才运行，所有方法 10/10 收敛。

| 方法 | 方向准确率 | movement $R^2$ |
|---|---:|---:|
| Hard HiWA | 0.4188 | 0.1908 |
| Soft-GCOT HiWA | 0.4271 | 0.1947 |
| Sparse approximation | 0.4271 | 0.1937 |
| Soft-GCOT HiWA + ROCA | 0.4792 | 0.3364 |

成对分析显示，Soft-GCOT HiWA 相对 Hard HiWA 的 direction accuracy 平均变化为 +0.0083，bootstrap 95% CI 为 [+0.0042, +0.0135]；movement $R^2$ 平均变化为 +0.0039，CI 为 [+0.0025, +0.0052]。

## ROCA 边界

Soft-GCOT HiWA + ROCA 的 accuracy 平均变化为 +0.0604，bootstrap 95% CI 为 [+0.0083, +0.1271]；$R^2$ 平均变化为 +0.1456，CI 为 [+0.0033, +0.4287]。但这两个均值受少数大提升 seed 影响，accuracy 的中位数变化仅为 +0.0104。

ROCA 在 8/10 个 seed 选择了两个 determinant 候选中事后 accuracy 更高的分支。seeds 85、86 均无 warning 却选择了较低 accuracy 分支，因此：

> 当前 warning 是退化风险提示，不是 branch correctness 的保证。

这轮验证支持“ROCA 在 full-support Soft-GCOT HiWA 上有平均增益”，但不支持“ROCA 是无错的无标签 selector”。

## 结果来源

- [seeds 80–84 JSON](../../experiments/results/taco_faithful_baseline_full_support_roca_seeds_80_84.json)
- [seeds 85–89 JSON](../../experiments/results/taco_faithful_baseline_full_support_roca_seeds_85_89.json)
- [paired analysis JSON](../../experiments/results/paired_analysis_full_support_roca_seeds_80_89.json)
- [paired scatter figure](../../experiments/figures/paired_analysis_full_support_roca_seeds_80_89.png)

所有 artifact 已由运行器或统计工具自动提交至 GitHub `main`；本说明记录其解释边界。
