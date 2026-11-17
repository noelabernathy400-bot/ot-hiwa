# Hyser Hard HiWA：原神经实验 R² 与相对误差补算（2026-11-17）

> 本文数值**不是重新拟合**：使用已经保存的真实 Hyser sample1 点云与 10 个 Hard HiWA 数值转录版 seed 的最终旋转矩阵，对同步窗口做事后评价。实际 Hard HiWA 求解来自前次的仓库公式转录版，不是直接执行 `src/hiwa/hiwa.py` 的原始文件。不能把本报告当作原文件直接执行的结果。

## 1. 严格区分三个指标

### 神经实验中的 `original_r2`
来自本仓库 `src/cc_hiwa/common.py`；两组分别白化后，计算

\[
R^2_{\rm neural}=1-\frac{\sum_i\|\operatorname{white}(X R^\top)_i-\operatorname{white}(Y_{\pi(i)})\|^2}{\sum_i\|\operatorname{white}(Y_{\pi(i)})_i-\overline{\operatorname{white}(Y)}\|^2}.
\]

神经实验通常只计算二维运动分量；这里使用 Hyser 的**全部三个力量坐标**。评价时用同步索引 \(\pi\)；HiWA 拟合时不允许使用。

### 配对样本的相对平方误差（实际可算）

\[
E_{\rm relMSE}=\frac{\|XR^\top-Y_{\pi}\|_F^2}{\|Y_{\pi}\|_F^2},
\quad E_{\rm rel}=\sqrt{E_{\rm relMSE}}.
\]

这里 \(X,Y\) 是用**训练数据单独拟合协方差白化**变换后的三维点，因此误差是无量纲的。这是与任务直接相关的**同步配对预测相对误差**，并不是下面的原版旋转 rMSE。

### HiWA `_rMSE`（当前无真实值）

`src/hiwa/hiwa.py` 定义的是 \(\|R_{\rm gt}X^\top-RX^\top\|_F^2/\|R_{\rm gt}X^\top\|_F^2\)。它要求**真实旋转矩阵 \(R_{\rm gt}\)**。Hyser 没有提供该矩阵，不能直接声称有原版旋转 rMSE。若把“训练配对拟合的 oracle 旋转”代入作参照，仅是代理误差，8 个收敛 seed 的训练均值 **2.5913 ± 0.1360**，不能与真正的已知旋转误差混用。

此外，`src/hiwa/hiwa.py::_eval_R2` 的公式使用 **均值差**而不是**平方差**，不是标准的 \(R^2\)。本报告严格按神经实验的 `original_r2` 定义评价。

## 2. 结果

真实 Hyser `subject01_session1/sample1`：0.1 秒窗口，前 200 对训练、后 50 对作为同一试次内探索性留出；每边 3D，K=4，参数 \(\mu=.05\)，ADMM tol=.01，最大 200 轮。随机 seed0–9，其中 **8/10** 双残差满足条件。Hard HiWA 均值与标准差仅汇总已收敛的 8 个种子。

| 指标 | 对齐前 | Hard HiWA (mean ± sample SD) | 训练配对 oracle |
|---|---:|---:|---:|
| **训练神经式 R² ↑** | -0.6372 | **-1.1459 ± 0.1084** | -0.1073 |
| **留出神经式 R² ↑** | -1.2411 | **-1.1222 ± 0.5045** | -0.4869 |
| 训练配对相对平方误差 ↓ | 1.6372 | **2.1459 ± 0.1084** | 1.1073 |
| 留出配对相对平方误差 ↓ | 3.8316 | **3.7832 ± 0.7863** | 2.9487 |
| 训练相对 Frobenius 误差 ↓ | 1.2795 | **1.4645 ± 0.0369** | 1.0523 |
| 留出相对 Frobenius 误差 ↓ | 1.9574 | **1.9362 ± 0.1984** | 1.7172 |

**为什么训练 `original_r2=1-relMSE`，留出却不相等？** 训练已在固定白化三维空间中中心化，预测是正交旋转，因而训练的原神经式再次白化恰好与固定空间一致；但留出样本再次白化会用留出自己的均值/协方差，而固定训练白化不等于留出白化，所以两种指标不同。不能混用。

## 3. 结论与限制

- Hard HiWA **数值收敛不等于对齐成功**：训练 R² 变差；同一次试次的最后 5 秒 R² 仍小于 0，且结果高度依赖 seed。
- 相对平方误差远高于 1，意味着目标的原点参照总平方能量尺度上，误差很大；不是“100% 以上分类错误”。
- Oracle 是训练同步配对求出的**最佳正交旋转**，只用于诊断，而不是实际可用的无监督算法。
- 这个留出**不是独立试次**，还不能据此评价跨试次或跨被试泛化。
- 需要先用仓库原 `HiWA` 类直接执行复验，再与 Soft-GCOT 做严格同条件比较。

## 4. 逐种子结果

见 [CSV](hard_hiwa_r2_relative_error.csv)。具体复算程序见 [scripts/evaluate_original_r2.py](../scripts/evaluate_original_r2.py)，以 `scripts/run_hard_distance_multiseed.py` 生成的原类运行结果 JSON 与本地 `domains.npz` 为输入，未来可重复计算。

**来源**：本仓库 `src/cc_hiwa/common.py::original_r2`、`src/hiwa/hiwa.py::_rMSE` 和 `_eval_R2`，真实数据处理记录见 `../EXPERIMENT_LOG.md`。
