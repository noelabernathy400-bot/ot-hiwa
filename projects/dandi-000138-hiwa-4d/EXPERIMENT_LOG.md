# 实验日志（持续更新）

## 2026-11-17 指标口径纠正：以后只使用 HiWA 的 rMSE 与 R² 作为主评价

**本节取代下文 2026-11-17 旧记录中两类 NRMSE 的性能判定。** 使用原有 400/100 trial 分割、相同四维表示和固定归一化、seed 42–46，**没有重新训练 HiWA**。原始 `code/hiwa_original.py` 保持不变。

- 原版 `_rMSE(X,R,Rgt)=||Rgt Xᵀ-R Xᵀ||²_F/||Rgt Xᵀ||²_F` 需要实际的真值旋转 `Rgt`。DANDI 000138 不提供 `Rgt`，因此仅报告**rMSE 代理值**：用训练试次已知同步配对拟合的有监督正交 Procrustes 作为参照 `R_ref`；这个参照**不参与 HiWA 无监督拟合**，且**不能等同真实 rMSE**。
- 原版 `_eval_R2` 存在公式实现问题（用了有符号平均误差而非平方残差）。不修改原版求解器，但按标准多元 **R² = 1 − SSE/SST**（`SSE=sum_test ||y_i−R x_i||²`，`SST=sum_test ||y_i−mean_test(y)||²`）在留出测试集评分。
- ADMM 共识/正交性/OT 边缘检查保留为**数值合格性检验**，不再作为额外的“预测评价指标”。

| 参照方法 | rMSE（代理，越低越好） | 标准测试 R²（越高越好） |
|---|---:|---:|
| Identity（未对齐） | 1.96470 | -0.79649 |
| Hard HiWA，5 seeds 均值 ± 样本标准差 | **1.91621 ± 0.37054** | **-0.89472 ± 0.22508** |
| 有监督训练配对 Procrustes（只作为诊断） | 0（定义所致） | -0.06044 |

| HiWA seed | rMSE 代理值 | 标准 R² |
|---:|---:|---:|
| 42 | 2.11824 | -0.80478 |
| 43 | 1.26308 | -0.54862 |
| 44 | 2.09887 | -1.09933 |
| 45 | 1.97163 | -0.95505 |
| 46 | 2.12925 | -1.06582 |

**结论：** rMSE 代理值五次平均稍微降低，但随机性很大；真正可解释为测试集运动预测的 R² 五次都是负的，平均比未对齐还差。监督训练参照的测试 R² 也稍低于 0，提示四维全局正交假设值得优先检验。不能断言找到真实旋转，也不能从这次实验得出 Soft 优于 Hard。

**复现**：`python code/evaluate_hiwa_two_metrics.py --clusters results/clustering/clustered_4d.npz --alignment results/hard_hiwa/hard_hiwa_results.json --output results/hard_hiwa/hiwa_rmse_r2_evaluation.json`。新实验推荐 `code/run_hard_hiwa_2metric.py`，从原版 runner 的保存结果自动计算这两项。旧的 NRMSE JSON/图仅作为历史记录保留，**不用于以后的主结论**。

---


**最新：2026-11-17**。本页用于快速恢复项目进度。原始 NWB 不入仓库；所有固定输入、中间图和 Hard HiWA 原始数值见 \`results/\`。实验只回答原版硬分组基线，尚未测 Soft-GCOT。

## 明确研究问题
1. 四维神经表征和原生四维手部运动是否具备可识别的簇结构？
2. Hard HiWA 在数值上**真正收敛**后，能否降低两个域的**无配对分布**误差？
3. 这种分布匹配能否恢复**逐时刻神经—行为对应**？需要哪些几何/动态条件？

## 实验 A：KMeans(4)，真实 NWB
DANDI 000138 (0.220113.0407), \`sub-Jenkins_ses-large_desc-train_behavior+ecephys.nwb\`，500 trials，训练/测试 400/100（窗口 14000/3500）。神经 162 units → PCA4（前四轴累计解释约 13.06%）；行为保留 (pos_x,pos_y,vel_x,vel_y) 四维，无行为 PCA。两域独立 KMeans，train silhouette 神经 **0.33961** / 行为 **0.37032**。四维拟合、六张二维投影，不要把簇颜色视作跨域对应。

[神经簇图](results/clustering/neural_4d_pairwise_train.png) · [手部簇图](results/clustering/hand_4d_pairwise_train.png)

## 实验 B：原版 Hard HiWA，2026-11-17
精确使用用户上传的 2026-10-18 zip 原版 PyHiWA；关闭求解器内部额外归一化和降维；每域每簇独立随机抽 40 点；固定参数和统一误差见 [PROTOCOL](PROTOCOL.md)。

| seed | 外循环轮数 | 最终 global Δ | 最大 local-global 共识残差 | 审计通过 | 配对 NRMSE | 无配对 OT NRMSE |
|---:|---:|---:|---:|:---:|---:|---:|
| 42 | 140 | 2.71e-9 | 4.13e-9 | 是 | 1.34199 | 0.64336 |
| 43 | 140 | 4.61e-9 | 6.48e-9 | 是 | 1.24311 | 0.61718 |
| 44 | 140 | 9.37e-10 | 1.67e-9 | 是 | 1.44736 | 0.61489 |
| 45 | 130 | 9.75e-10 | 1.45e-9 | 是 | 1.39674 | 0.62637 |
| 46 | 140 | 4.59e-7 | 5.17e-7 | 是 | 1.43576 | 0.65457 |
| **Hard mean ± sample SD** | — | — | — | **5/5** | **1.37299 ± 0.08347** | **0.63127 ± 0.01717** |
| **identity, before** | — | — | — | — | **1.33890** | **0.66162** |
| **有监督训练配对 Procrustes（仅诊断）** | — | — | — | — | **1.02868** | **0.58886** |

所有簇级传输矩阵均满足行列边缘 1/4，最大偏差量级 \`1e-17\`；旋转正交检验通过。**Hard 对无配对分布误差平均降低约 4.59%，但逐窗配对误差没有稳定改善（均值略差）。** 不能将分布对齐说成神经运动解码成功。seed 影响明显，五次为探索性重复，并非显著性检验。

[误差图](results/hard_hiwa/heldout_error_comparison.png) · [收敛与组间 OT](results/hard_hiwa/admm_and_group_transport.png) · [对齐前后投影](results/hard_hiwa/hard_hiwa_alignment_xy.png) · [逐种子 P、R、残差](results/hard_hiwa/hard_hiwa_results.json)

## 遇到并解决的问题
- 官方 DANDI 000138 NWB hand Series 存放位置与 000688 不同；读取脚本按 NWB 实际 series 结构查找，禁止用 cursor 代替 hand。
- 原版停止条件仅看 \`||R_t-R_{t-1}||\`，短迭代预算会在局部共识尚未满足时提前停。将原版 stop tol 设为 1e-9，**另用 0.01 同时审核 global 与 primal**；不修改原版实现。
- 保留同一测试集/归一化/误差定义；区分评价配对关系的 NRMSE 与只评价分布几何的 OT NRMSE。


**复现证据**：[GitHub Actions 完整成功运行 #44059421834](https://github.com/noelabernathy400-bot/ot-hiwa/actions/runs/44059421834)。该运行重新下载官方 NWB、执行完整聚类和 Hard HiWA、验证原版源码 SHA-256 与五次共识收敛，并将处理后数据、图表及 JSON 提交到 `results/`。已比对与本地运行的两类 NRMSE，差异仅为浮点舍入级别。

## 下一步，尚未执行
1. 分析五个 R 和 P 的结构稳定性；测量相对旋转与群组软/硬对应歧义。
2. 仅在**评价环节**扫描神经—手运动时间滞后，检验 4D PCA（累计解释方差 13.06%）的代表性。
3. 在相同数据拆分、相同四维及同两个误差指标下再引入 Soft-GCOT，严格比较对齐质量及收敛性。

**限制：** 只代表此被试、此会话的探索性实验。正交可恢复性与语义预测泛化尚无证据。
