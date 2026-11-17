# 指标修订 v2 · 2026-11-17

**优先适用：** 在原有数据与求解器参数不变的条件下，本项目以后统一使用 **HiWA rMSE（真实数据必须标注代理值）** 和 **标准 held-out R²**，旧版 NRMSE 只保存用于历史追溯。

- rMSE 来源：压缩包原版 `hiwa_original.py::_rMSE`。真实 DANDI 无已知的 `R_gt`，因此只有以训练同步配对 Procrustes 作为 `R_ref` 的**代理值**；这是明确受限的旋转参考误差，绝非真实旋转误差。
- 标准决定系数：**R² = 1 − SSE/SST**，其中 `SSE = sum_test ||y_i − R x_i||²`，`SST = sum_test ||y_i − mean_test(y)||²`。修正原版 `_eval_R2` 未平方残差的实现缺陷，其他求解代码完全不动。
- 固定所有 seed、四维表示、trial-disjoint holdout、归一化及已保存的 `R`；使用 `code/evaluate_hiwa_two_metrics.py` 重新评价即可。校准的有监督 `R_ref` **不进入无监督训练**。
- 依旧检查 ADMM global/local 共识等数值可靠性，但它们是求解器审计，而不是预测效果的第三/第四指标。

---

# 冻结协议 v1 · 2026-11-17

## 数据处理
DANDI:000138 / MC_Maze_Large / 0.220113.0407 官方 train NWB。162 units 的 spiking → 20ms bins → sqrt → 40ms 平滑 → 训练集通道标准化 → PCA(4)；行为域为 hand_pos x/y、hand_vel x/y 四个字段，**不做行为 PCA**，只用训练集列尺度标准化。以 move_onset_time 为参考窗口 [-100,600]ms；每试次 35 个窗口。500 trials，随机种子 42 将试次划为 400 train / 100 test，分别 14000 / 3500 个窗口。PCA、标准化、KMeans 只用训练集拟合。

两域在四维各自独立 KMeans(K=4,n_init=20,seed=42)。相同簇编号/颜色不表示域间匹配。

## 对齐输入与原版求解器
将两个域分别按训练集均值中心化并除以**各自的一个全局 RMS 半径** \`s_D=sqrt(mean_train ||z-m_D||²)\`。这种各域不同的尺度变换不是在原始物理单位下的严格正交对齐；但它保留各域内部四维相对几何。禁止逐轴二次白化或行为 PCA。原版 HiWA 设置 \`normalize=False\`，\`X_transform=Y_transform=I4\`，dimension=4。

各域从训练集中按簇**独立**抽 40 个点/簇（各 160 点），随机种子 42–46。配对时间索引不供 HiWA 拟合。原版算法包含局部 Sinkhorn Q_ij、簇间 P、局部 R_ij、全局 R 及 ADMM；仅不改返回值地捕获最终局部旋转以审计。

| 固定参数 | 数值 |
|---|---:|
| seeds | 42 43 44 45 46 |
| outer maxiter | 140 |
| 原版 stop tol | 1e-9 |
| 额外收敛审计阈值（global / primal） | 1e-2 |
| mu | 0.05 |
| group Sinkhorn gamma / iters | 0.2 / 200 |
| local Sinkhorn gamma / iters | 0.1 / 45 |
| local alternating iters / tol | 20 / 0.01 |
| holdout unpaired OT sample count | 每域独立 200（固定 RNG seed 20261117） |

严格通过条件：\`||R_t-R_{t-1}||_F <= 0.01\` 且 \`max_ij ||R_ij-R||_F <= 0.01\` 且 \`max(marginal_error(P)) < 1e-5\`；同时检查 \`||R^T R-I||_F < 1e-7\`、无 NaN、P 无负质量。原版仅 global-delta 的停止不能视为足够的 ADMM 证明。

## 同一误差口径
统一 **held-out 100 trials**、统一各域全局尺度、统一测试目标的 RMS 作分母：
- **配对误差**：\`NRMSE_pair(R)=sqrt(mean_test ||R x_i-y_i||²)/sqrt(mean_test ||y_i||²)\`，测试时窗的已知同步索引仅作评估。
- **无配对 OT 误差**：在预先固定的独立测试各 200 个点上，\`NRMSE_OT(R)=sqrt(min_perm mean ||R x_j-y_perm(j)||²)/sqrt(mean_test ||y_i||²)\`，匈牙利精确一对一匹配。
- identity(R=I)、无监督 Hard HiWA、仅供诊断的监督 Procrustes（仅使用**训练**时窗配对）都用同两个指标。

两种误差不可相互解释为同一任务。神经肌肉反应有时间滞后可能；无配对分布匹配成功不蕴含逐窗预测成功。
