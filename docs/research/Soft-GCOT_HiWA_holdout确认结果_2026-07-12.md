# Soft-GCOT HiWA holdout 确认结果

_冻结配置的未参与确认 seeds 70–79；2026-07-12_

---

## 结论

seeds 70–79 未用于 50–69 的确认汇总，也未用于任何参数或分支规则选择。以同一固定 temperature path、Hard warm start、阶段继承和 0.01 global-plus-primal 停止规则运行后，所有方法均为 10/10 收敛。

| 方法 | 方向准确率 | movement $R^2$ |
|---|---:|---:|
| Hard HiWA | 0.4802 | 0.4722 |
| Soft-GCOT HiWA | 0.4990 | 0.4782 |
| Sparse approximation | 0.4990 | 0.4768 |
| Soft-GCOT HiWA + ROCA | 0.5125 | 0.4767 |

相对 Hard HiWA，Soft-GCOT HiWA 的平均变化为 accuracy +0.0188、$R^2$ +0.0060，10 个 seed 中 9 个 accuracy 更高。Soft-GCOT HiWA + ROCA 的平均变化为 accuracy +0.0323、$R^2$ +0.0045，10 个 seed 全部 accuracy 更高。

## ROCA 检查

- 10/10 无 warning。
- 10/10 选择了两条 determinant 候选中 accuracy 更高的分支；标签仅用于该事后核验。
- 该 holdout 支持 50–69 中观察到的平均增益，但不消除已知的 seed 53 失配边界。

## 结果来源

- [seeds 70–74 JSON](../../experiments/results/taco_faithful_baseline_holdout_seeds_70_74.json)
- [seeds 75–79 JSON](../../experiments/results/taco_faithful_baseline_holdout_seeds_75_79.json)

两批的 JSON 和图均由 runner 在成功完成后自动提交到 GitHub `main`。
