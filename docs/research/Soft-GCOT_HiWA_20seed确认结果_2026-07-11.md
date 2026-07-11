# Soft-GCOT HiWA 20-seed 确认结果

_固定 staged 使用条件：seeds 50–69、96×96 神经子集、2026-07-11_

---

## 结论

在预先固定的 temperature path $0.25\rightarrow0.35\rightarrow0.50$、Hard warm start、阶段状态继承和统一 0.01 global-plus-primal 停止规则下，全部 20 个 seed 收敛。

| 方法 | 收敛 | 方向准确率（均值 ± SD） | movement $R^2$（均值 ± SD） |
|---|---:|---:|---:|
| Hard HiWA | 20/20 | 0.3865 ± 0.1284 | 0.0487 ± 0.7096 |
| Soft-GCOT HiWA | 20/20 | 0.3990 ± 0.1383 | 0.0529 ± 0.7125 |
| Sparse approximation | 20/20 | 0.3990 ± 0.1383 | 0.0521 ± 0.7114 |
| Soft-GCOT HiWA + ROCA | 20/20 | 0.4375 ± 0.1459 | 0.1196 ± 0.6973 |

Soft-GCOT HiWA 相对 Hard HiWA 的平均变化为 accuracy +0.0125、$R^2$ +0.0043；20 个 seed 中有 12 个 accuracy 更高。full support 与 sparse approximation 的均值几乎一致，故本确认集不支持“sparse 近似本身带来提升”。

Soft-GCOT HiWA + ROCA 相对 Hard HiWA 的平均变化为 accuracy +0.0510、$R^2$ +0.0709；20 个 seed 中有 16 个 accuracy 更高。

## ROCA 边界

- 20/20 无 warning；代表点 condition number 为 2.32–4.74，oriented-volume margin 约 49.42。
- 代表点方向规则选择了更高 accuracy 分支 19/20 次。
- 唯一失配为 seed 53：规则选择 det=-1（accuracy 0.208），而 det=+1 为 0.427。

因此，ROCA 在这个固定使用条件下有强的平均稳定化证据，但不是无错 branch selector。seed 53 不是 simplex 退化案例，说明仍存在未被当前有向体积规则捕获的分支歧义。

## 与直接纯基线的关系

直接 temperature=1、无 warm start 的纯 Soft-GCOT HiWA 审计虽可收敛，却性能较差并触发高 assignment entropy warning。恢复 staged 使用条件后，Soft-GCOT HiWA 与 ROCA 的结果恢复。这表明使用条件是方法结论的一部分，不能把“直接 soft assignment”与“staged Soft-GCOT HiWA”混成同一实验设置。

这不引入新模型名称：论文或报告中仍只使用 Hard HiWA、Soft-GCOT HiWA、Soft-GCOT HiWA + ROCA；temperature path 和 warm start 仅是后两者的固定运行配置。

## 结果来源

- [seeds 50–54 JSON](../../experiments/results/taco_faithful_baseline_historical_protocol_seeds_50_54.json)
- [seeds 55–59 JSON](../../experiments/results/taco_faithful_baseline_historical_protocol_seeds_55_59.json)
- [seeds 60–64 JSON](../../experiments/results/taco_faithful_baseline_historical_protocol_seeds_60_64.json)
- [seeds 65–69 JSON](../../experiments/results/taco_faithful_baseline_historical_protocol_seeds_65_69.json)

每个 JSON 均包含逐 seed 的指标、收敛曲线、$P$、group cost、Sinkhorn 误差、旋转与 ROCA 诊断；四批 artifact 均在运行完成后自动推送至 GitHub。
