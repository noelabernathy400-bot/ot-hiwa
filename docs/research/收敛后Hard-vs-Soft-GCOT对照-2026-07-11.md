# 收敛后 Hard HiWA 与 Soft-GCOT HiWA 对照

_固定 `audit` 协议、seeds 50–54、96×96 神经子集；2026-07-11_

---

## 结论

五个 seed 的 Hard HiWA、Soft-GCOT HiWA、Sparse approximation 与 Soft-GCOT HiWA + ROCA 均达到统一的 global-plus-primal ADMM 停止条件。因而这是一轮可比较的负结果：

| 方法 | 收敛 | 方向准确率 | movement $R^2$ |
|---|---:|---:|---:|
| Hard HiWA | 5/5 | 0.3375 | -0.2319 |
| Soft-GCOT HiWA | 5/5 | 0.3083 | -0.4903 |
| Sparse approximation | 5/5 | 0.2208 | -0.9399 |
| Soft-GCOT HiWA + ROCA | 5/5 | 0.0833 | -1.6571 |

相对 Hard HiWA，Soft-GCOT HiWA 的平均方向准确率为 -0.0292、平均 $R^2$ 为 -0.2584。Sparse approximation 更差，不能替代 full support。

## 收敛与数值检查

- Soft-GCOT HiWA 的最终 primal residual 为 0.00940–0.00990；最大 Sinkhorn marginal error 为 $4.3\times10^{-14}$ 至 $9.4\times10^{-14}$。
- Hard HiWA 的最终 global residual 为 $1.0\times10^{-4}$ 至 $1.3\times10^{-4}$。
- 所有全局旋转的正交误差约为 $10^{-15}$。
- JSON 保存每个 seed 的 global/primal/dual residual、transport objective、Sinkhorn error、$P$、group cost 和 ROCA 几何诊断；收敛图显示各外层迭代轨迹。

## ROCA 适用性

本次纯 Soft-GCOT HiWA 基线上的 ROCA 不满足可用条件：5/5 触发 `high_assignment_entropy` warning，尽管 simplex condition number 仅为 2.33–5.23、oriented-volume margin 约为 26.79，几何并未退化。

因此，这里的失败不是代表点 simplex 退化，而是当前纯 soft grouping 接口下的 branch selection 与历史已验证的代表点/分量流程不等价。不能把此前 ROCA 的成功直接移植为“纯 Soft-GCOT HiWA + ROCA 已成功”。

## 可保留与不可混合的结论

- 可保留：历史的代表点方向选择在其原始流程中有 20-seed 确认和 30-case 坐标翻转证据。
- 可保留：当前纯 Soft-GCOT HiWA full-support 代码在统一 ADMM 协议下可收敛。
- 不可混合：两套结果的初始化、代表点/分量处理和运行协议不同；不能合并为单一性能结论。

### 已核对的历史流程差异

历史 ROCA 成功实验不是当前纯基线的直接复跑。它使用从 Hard HiWA 的 $R,P$ warm start、三阶段 temperature path $0.25\rightarrow0.35\rightarrow0.50$、每阶段继承前一阶段状态、`pilot` 的 0.1 停止阈值，以及 sparse support。当前审计使用直接 temperature=1.0、full support、无 warm start、96×96 固定子集、0.01 的统一停止阈值和 $μ=0.05$。

这些差异足以解释为什么“历史 ROCA 成功”不能成为当前纯 Soft-GCOT HiWA + ROCA 的性能担保；它们不是小的实现细节，也不意味着 ROCA 数学公式被修改。

## 历史使用条件的最小统一复现

在同一 96×96 子集、同一 0.01 收敛阈值下，恢复温度路径、Hard warm start 和阶段继承后，seeds 50–54 的结果如下。ROCA 在这一轮使用历史的 sparse support；full-support Soft-GCOT HiWA 仍同时报告，作为对照。

| 方法 | 收敛 | 方向准确率 | movement $R^2$ |
|---|---:|---:|---:|
| Hard HiWA | 5/5 | 0.3896 | 0.0488 |
| Soft-GCOT HiWA | 5/5 | 0.4021 | 0.0529 |
| Sparse approximation | 5/5 | 0.4021 | 0.0522 |
| Soft-GCOT HiWA + ROCA | 5/5 | 0.4167 | 0.0393 |

这证明“温度路径 + warm start + 阶段继承”足以恢复 Soft-GCOT HiWA 的小幅正向效果；full 和 sparse 的均值几乎相同，因此 sparse support 不是这次恢复的主要解释。ROCA 的方向准确率较 Hard HiWA 高 0.0271，但 $R^2$ 低 0.0096，不能据此宣称它已经在统一协议下稳定提高所有指标。

所有 ROCA 运行均无 warning，condition number 为 2.32–4.74，oriented-volume margin 约 49.42。这和直接纯基线的 `high_assignment_entropy` warning 相反，说明温度路径也改变了 ROCA 的适用性诊断。

该最小复现的结果文件：[JSON](../../experiments/results/taco_faithful_baseline_historical_protocol_seeds_50_54.json)，[accuracy figure](../../experiments/figures/taco_faithful_baseline_historical_protocol_seeds_50_54.png)，[convergence figure](../../experiments/figures/taco_faithful_convergence_historical_protocol_seeds_50_54.png)。

下一步不是调参或增加模块，而是做实现谱系核对：将历史成功流程逐项映射到当前 runner，确认 ROCA 接口在何处发生改变，再决定是否存在一个可复现的 `Soft-GCOT HiWA + ROCA` 条目。

结果文件：[audit JSON](../../experiments/results/taco_faithful_baseline_audit_seeds_50_54.json)，[accuracy figure](../../experiments/figures/taco_faithful_baseline_audit_seeds_50_54.png)，[convergence figure](../../experiments/figures/taco_faithful_convergence_audit_seeds_50_54.png)。
