# Transport-consistent 联合优化首轮结果

_2026-07-12 · 固定 MiHiA 96×96 子集，seeds 110–112_

---

## 结果

本轮以固定 Soft-GCOT 的 \(P,Q,R\) 构造 \(\Pi=\sum_{kl}P_{kl}Q_{kl}\)，并最小化

$$
L_{TC}=\sum_{ij}\Pi_{ij}\|S_i-T_jM^\top\|^2,
$$

其中 \(M\) 为组传输的行归一化映射。更新阶段不对 Sinkhorn 或 ADMM 求梯度。

| 方法 | 平均方向准确率 | 平均 movement \(R^2\) | 收敛 |
| --- | ---: | ---: | ---: |
| Hard HiWA | 0.4896 | 0.6148 | 3/3 |
| Fixed Soft-GCOT | 0.5035 | 0.6200 | 3/3 |
| Transport-consistent Joint | 0.5035 | 0.6192 | 3/3 |

第一次更新使源 assignment 的 Frobenius 差异约为 0.256，且每个 seed 有 1/96 个样本改变 argmax 组；但组传输变化仅约 \(8.4\times10^{-5}\)，全局样本 coupling 变化约 \(8.6\times10^{-4}\)。第二次更新后所有变化进一步显著下降。

## 结论

该损失比 prototype-only 更新更直接作用于样本 coupling，也确实导致更明显的 assignment 与局部 \(Q\) 重分配；但它仍未改变最终 accuracy，且只带来很小的 \(R^2\) 下降。

因此，当前不应继续扫描 \(\lambda_{TC}\) 或增加外循环。prototype-only 与单向 transport-consistent 两个联合目标都应冻结为负结果对照。后续若继续联合优化，需要新的、可验证的双向或质量感知约束，而非继续调整这一目标的权重。

结果文件：[JSON](../../experiments/results/alternating_joint_soft_gcot_transport_consistent_joint_seeds_110_112.json)。
