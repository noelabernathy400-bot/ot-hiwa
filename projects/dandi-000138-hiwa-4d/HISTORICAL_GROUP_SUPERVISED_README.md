# 历史组级监督诊断 README（2026-11-17 前的研究记录）

**此文件为历史快照，内容中“最新正式实验”等措辞已经过时。其源域分组向目标域传播，不能作为无监督 Hard/Soft 基线。请转到 [现行无监督首页](README.md)。**

---

> [!CAUTION]
> **重要分组纠错（2026-11-17）：后续「唯一源域 K-means」的 Hard / Soft / 边界自适应实验曾将用于同步着色的源域分组或软成员权重复制到目标域训练。相关结果保留为「组级监督诊断」，不能作为严格无监督性能结论。** 最初源、目标各自独立 K-means 的 Hard 流程不属于这一类；请勿混作同一无监督基线。原版求解器的 P 是均匀初始化后经 Sinkhorn 更新，并未被固定为 I₄/4。详见 [GROUP_LABEL_LEAKAGE_AUDIT.md](GROUP_LABEL_LEAKAGE_AUDIT.md)。
# DANDI 000138 × HiWA：唯一源域硬聚类四维对齐

**当前正式实验：只在源域 K-means 聚类一次，目标域按同步对应继承源域标签。对齐前、目标域、对齐后的源域使用从始至终相同的组。**

- [最新正式 XeLaTeX PDF（outer≤500，五次数值审计通过）](四维HiWA统一源域分组实验.pdf)
- [实验日志（按最新研究排列）](EXPERIMENT_LOG.md)
- [500 次扩展的最终数值、R、P、全部图片](results/source_only_fixed_groups_outer500/)
- [源域唯一 K-means 标签、原始两域图片及 140 次初始对齐（保留供复核）](results/source_only_fixed_groups/)
- [源域唯一聚类及绘图程序](code/source_only_fixed_group_pipeline.py)
- [原版 HiWA 运行器](code/run_hard_hiwa.py) · [原版算法源码](code/hiwa_original.py)
- [全维度评价、标签一致性校验及出图](code/evaluate_fixed_group_alignment.py)

## 实验协议（冻结）

1. 使用 DANDI:000138、MC_Maze_Large、v0.220113.0407 的已验证 NWB 派生数据。162 个神经放电通道→20ms 计数→平方根与平滑→训练统计标准化→PCA(4)。目标域是 (hand_pos_x,hand_pos_y,hand_vel_x,hand_vel_y) 原生四维，**不降维**。
2. 500 个有效试次按 trial 独立划分为 400 训练/100 测试，对应 14000/3500 个窗口。只在神经域训练点云上 **KMeans(K=4, seed=42,n_init=20)**；四个训练簇分别为 (4137,4316,3146,2401)。
3. 目标域每一个点的标签直接采用同一时间窗对应神经点的标签。测试神经点用训练源聚类器 predict，然后沿一一对应索引赋给测试目标。**不对目标域独立聚类，不在最优传输后再次聚类；颜色在全部可视化中不变。**
4. 在保持上述组不动的前提下，原版 HiWA 对源域学习全局正交矩阵，目标域不旋转。仅输入源/目标相同的组编号，不直接以配对回归误差训练 HiWA。原版 HiWA 内每个域/簇独立采样 40 点。
5. 固定 seeds 42–46，最终将最大外循环数从最初 140 扩至 **500**，其余原超参不变：mu=.05、局部 Sinkhorn gamma=.1、组间 Sinkhorn gamma=.2、局部最多 20 次。5/5 seed 通过 global-change≤.01、local-global residual≤.01、正交性和 OT 边缘审计（不是全局最优证明）。

**重要方法学限制：** 虽然 HiWA 仍利用分组上的非配对 OT 计算变换，但目标域组标签来自真实同步配对，因此这是**带有已知对应信息的组定义协议**，不是完全无监督的跨域聚类发现。1NN 也是用继承标签计算的固定组邻域一致性，不应冒充独立运动类别准确率。

## 最新补充：最优传输的留出测试误差（2026-11-17）

**HiWA 不能只用旋转后的配对欧氏距离评价。** 已对同一次唯一源 KMeans、目标继承标签、HiWA 500 次实验的冻结全局 R / 簇间 P 进行独立 OT 评价：

| 指标（归一化四维平方距离，越小越好） | 未对齐 | HiWA 后 5 seeds 均值 ± 样本标准差 | 代价降低 |
|---|---:|---:|---:|
| 无约束全域 OT W₂²，200×200 测试点 | 0.43948 | **0.40482 ± 0.04367** | **7.89%** |
| 同编号固定组内部 OT W₂²（经验质量），每组 128×128 | 1.23175 | **0.96642 ± 0.29007** | **21.54%** |
| 已学习簇间 P 加权组间 OT 代价（P 在比较前后固定） | 1.20303 | **0.95542 ± 0.03338** | **20.58%** |

[全部公式、随机采样协议、5 seeds 明细与图表](results/source_only_fixed_groups_outer500/ot_evaluation/OT_EVALUATION.md) · [原始结果 JSON](results/source_only_fixed_groups_outer500/ot_evaluation/OT_HELDOUT_METRICS.json) · [可复现评估代码](code/evaluate_hierarchical_ot.py)

![冻结分组下三种 OT 代价的传输前后比较](results/source_only_fixed_groups_outer500/ot_evaluation/ot_three_costs_before_after.png)

**研究上必须区分：**第 2 项强制同编号组匹配，是最贴近「从始至终同一组」的评价协议；第 3 项保留原版 HiWA 所学的非对角 P，仍允许跨编号组传输。后者是利用保存的 global R 和 P 在留出测试集重新算的代理 OT 代价，**不是**原始算法含局部旋转与熵正则的完整训练目标，因为原始局部 R_ij 和 Q_ij 未保存。已学 P 的对角总质量均值约 0.3456，因此「固定聚类标签」并不意味着「算法已限制只能同组传输」。OT 分布误差改善不能直接证明同步运动预测 R² 成功，两者回答不同问题。

---

## 边界自适应软分组正式实验（2026-11-17）：320/80 内验证 + 100 外测试

**结论：**选出的边界自适应 Soft 略微改善同编号固定组 OT，但整体分层总 OT 没有超过 Hard；没有证据表明其整体更优。

在唯一源域 K-means(K=4) 的四维神经空间中，定义 Δ = 次近中心距离平方 − 最近中心距离平方，α = exp(−Δ/h)，最终隶属度 A = (1−α)×onehot + α×softmax(−距离平方/(2τ²))。目标域直接从同步源时间窗继承隶属度，不独立聚类，变换后组标签完全不变。

在旧 400 个训练试次中分出 320/80 试次拟合/验证；h 在训练边界间隔 10%、30%、60% 分位数中选，τ∈{0.20,0.40}；三种种子做内部验证，预先约定按固定同组 OT 最小选参。最终在完整 400 训练试次重新拟合 5 seeds，对独立的 100 测试试次报告结果。PCA、标准化、唯一源域 K-means 原型沿用已在全部 400 训练试次拟合的缓存，故内部验证不是完全嵌套的表征预处理验证，但原 100 测试试次没有参与选参。

**内部验证选择 h=0.209933、τ=0.20（最窄过渡）。**该方案的训练平均软隶属度熵 0.03328，平均 α 为 0.10370；内部验证组 OT 为 0.84835，而 Hard 为 0.84638，尚未超过 Hard。

| 同 GCOT 求解器、5 seeds | P-Q 总 OT↓ | 固定同组 OT↓ | 全域 OT↓ | MAE↓ | R²↑ | 1NN↑ |
|---|---:|---:|---:|---:|---:|---:|
| Hard one-hot | **1.65257** | 0.98655 | **0.41854** | 0.52131 | -0.74185 | 36.19% |
| 固定 Soft τ=0.20 | 1.65883 | **0.96980** | 0.42092 | **0.51901** | **-0.72735** | **36.48%** |
| 边界自适应 Soft | 1.65493 | 0.97898 | 0.41977 | 0.52023 | -0.73633 | 36.22% |

五种随机初始化全部通过数值审计。与 Hard 配对比较，边界自适应在 4/5 seed 上降低了固定同组 OT，但均值降幅仅约 0.77%；P-Q 总 OT 反而升高约 0.14%。改进幅度远小于不同种子的离散程度，不能宣称具有统计意义。不同隶属度的 P-Q 代价涉及不同组条件分布；固定同组与全域 OT 是重要的共同分布检验。

![训练内部验证参数比较](results/boundary_adaptive_trial_cv/boundary_validation_grid.png)

![Hard、固定 Soft、自适应 Soft 的测试误差](results/boundary_adaptive_trial_cv/boundary_vs_hard_soft_all_errors.png)

![源域变换前、自适应变换后、目标域四维六投影](results/boundary_adaptive_trial_cv/boundary_before_after_target_six_views.png)

[详细报告](results/boundary_adaptive_trial_cv/BOUNDARY_ADAPTIVE_REPORT.md) · [原始数据与全部种子 R/P/误差](results/boundary_adaptive_trial_cv/BOUNDARY_ADAPTIVE_RESULTS.json) · [可复现 Python](code/run_boundary_adaptive_ablation.py) · [成功实验记录](https://github.com/noelabernathy400-bot/ot-hiwa/actions/runs/44096230862)

---
## 最新正式控制变量研究：同一求解器下 Soft 温度与总分层 OT

此次使用**完全相同的用户 ZIP 中 GCOT 分层 OT 求解器**（不是把原版 PyHiWA 直接与 GCOT 相比），同一 DANDI 000138 四维数据、唯一源域 K-means 四个原型、相同训练/测试划分、相同 160 点训练池、5 个种子与相同初始化。仅改变组成员关系：**Hard one-hot 或源原型距离 softmax（τ=0.20/0.40/0.80）**。目标软隶属度按真实同步索引继承，没有额外目标域 K-means；变换后也不重新分组。

本次首次对同一求解器统一计算**完整分层总传输几何代价**：\(\sum_{ij}P_{ij}\langle Q_{ij},\|Rx-y\|^2\rangle\)。测试时冻结训练求得的组级 \(P\)，在同一批留出测试点上通过原版 Sinkhorn 重新优化各组对 \(Q_{ij}\)，**不把熵项混入几何代价**；该指标允许跨组输运，区别于独立计算的「固定同编号组 OT」。总传输代价单独采用每域四组各32个测试样本；固定同组 OT 使用每组128个、全域 OT 使用200个点，三种问题及采样设置不同，**不可直接比较数值大小**。

| 相同 GCOT 求解器 | 总 P-Q OT 传输前 ↓ | 总 P-Q OT 传输后 ↓ | 固定同组 OT ↓ | 全域 OT ↓ | R² ↑ | 数值审计 |
|---|---:|---:|---:|---:|---:|:---:|
| Hard one-hot | 1.84457 ± 0.11774 | **1.65257 ± 0.02705** | 0.98655 ± 0.17106 | 0.41854 ± 0.01722 | -0.74185 ± 0.29195 | 5/5 |
| Soft τ=0.20 | 1.83751 ± 0.10973 | **1.65883 ± 0.02489** | 0.96980 ± 0.16222 | 0.42092 ± 0.01838 | -0.72735 ± 0.26579 | 5/5 |
| Soft τ=0.40 | 1.85479 ± 0.06809 | **1.72925 ± 0.01924** | 0.97591 ± 0.21394 | 0.42786 ± 0.03653 | -0.73686 ± 0.16337 | 5/5 |
| Soft τ=0.80 | 1.93353 ± 0.00516 | **1.90621 ± 0.01131** | 1.08109 ± 0.20117 | 0.41710 ± 0.03261 | -0.80493 ± 0.21280 | 5/5 |

**熵项不能替代传输误差。** 原作者目标函数带有 \(\sum Q\log Q\) 形式的熵项，Hard/Soft 有效支持不同，原始负熵基准也不同。下面将训练几何代价、原始含熵目标及额外的 KL 修正指标分别列出，避免将原始目标「越负」误判为实际传输「越好」：

| 模式 | 训练 P-Q 几何代价 ↓ | 训练原始含熵目标 | 测试 KL 修正目标（scaled） | 软分组归一化熵 |
|---|---:|---:|---:|---:|
| Hard one-hot | 1.55427 ± 0.05576 | -11.91206 ± 0.02141 | 0.47300 ± 0.00917 | 0.0000 |
| Soft τ=0.20 | 1.56126 ± 0.05733 | -12.11974 ± 0.05099 | 0.47332 ± 0.00841 | 0.0486 |
| Soft τ=0.40 | 1.63363 ± 0.06929 | -13.05755 ± 0.06791 | 0.48184 ± 0.00730 | 0.2677 |
| Soft τ=0.80 | 1.84350 ± 0.07503 | -15.58207 ± 0.04603 | 0.49778 ± 0.00104 | 0.8403 |

**阶段性结论：**五个种子全部通过预设的最终数值残差审计。在**测试 P-Q 总传输代价**上 Hard one-hot 最低，Soft τ=0.20 非常接近，较大的温度 τ=0.40/0.80 反而明显增大代价。若只看固定同组 OT，Soft τ=0.20（0.96980）比同求解器 Hard（0.98655）略低，但五种初始化不足以证明统计优势。

**严格限制：**温度网格是探索性敏感性分析，不能将测试集上最优温度当作经独立验证的超参数；本方法由于沿同步试次传播目标组隶属度，仍含有组信息监督；这一 Soft 是固定源 K-means 原型的连续分配，而非原版 TPAMI 分别学习两域原型的完全无配对 GCOT。

![控制变量实验：分层总 OT、同组 OT、全域 OT、R²](results/controlled_hard_vs_soft_PQ/controlled_soft_hard_all_OT_errors.png)

![原始标签始终不变：同一求解器的 Hard/Soft/Target 四维六投影](results/controlled_hard_vs_soft_PQ/hard_soft_target_six_pairs.png)

[完整对照报告](results/controlled_hard_vs_soft_PQ/CONTROLLED_ABLATION.md) · [每种方法每个种子的 R、P、分层 Q 传输代价与各种误差](results/controlled_hard_vs_soft_PQ/CONTROLLED_ABLATION.json) · [可复现运行代码](code/run_controlled_soft_hard_ablation.py) · [原始 ZIP GCOT 源码快照](code/zip_gcot/)

---

## 历史方法级对照：Soft-GCOT vs Hard HiWA（五种初始化）

复用相同 DANDI 000138 四维特征、trial 划分、唯一源域 K-means 四个中心、目标继承标签与同一留出测试样本。**Soft 使用固定源原型的距离 softmax（τ=0.40）生成连续隶属度，目标按配对索引继承；不对目标单独聚类，不改变原硬标签的 argmax 身份。** 使用上传压缩包原版 GCOT 分层 OT 求解器，保持对齐的全局正交输出，五种种子，outer=500、inner=20、sample Sinkhorn=45、40/组采样。Soft 与 Hard 的实现及每组有效支持不同，因而是方法级对照，不能将差异完全归因于软化本身。

| 归一化测试指标 | 未对齐 | Hard（均值 ± SD） | Soft（均值 ± SD） |
|---|---:|---:|---:|
| 全域自由 OT W₂² ↓ | 0.43948 | **0.40482 ± 0.04367** | 0.42786 ± 0.03653 |
| **同编号固定组 OT W₂² ↓** | 1.23175 | **0.96642 ± 0.29007** | 0.97591 ± 0.21394 |
| MAE ↓ | 0.53149 | **0.49767 ± 0.05548** | 0.52215 ± 0.02929 |
| R² ↑ | -0.79649 | **-0.60738 ± 0.30314** | -0.73686 ± 0.16337 |
| 共享标签 1NN ↑ | 23.14% | **37.74% ± 21.62%** | 34.57% ± 17.19% |

**阶段性判断：** Soft 对未对齐基线也能降低固定同组 OT（1.23175→0.97591，约 20.8%），但没有超越 Hard 平均值（0.96642）。这不意味着软分组一般无效；仅有五个随机种子，结论受当前温度、优化器、数据表示及结构假设限制。全部五个 Soft seed 通过预设数值审计，结果如实保留。

![Hard/Soft/Target：相同标签三域直接散点比较](results/soft_fixedgroup_vs_hard/hard_soft_target_xy.png)

![Hard Soft OT/R2 numerical comparison](results/soft_fixedgroup_vs_hard/soft_hard_global_groupOT_R2.png)

[完整六投影三域散点图](results/soft_fixedgroup_vs_hard/hard_soft_target_six_pairs.png) · [详细对照报告](results/soft_fixedgroup_vs_hard/SOFT_HARD_COMPARISON.md) · [原始逐种子 R/P/指标 JSON](results/soft_fixedgroup_vs_hard/SOFT_HARD_COMPARISON.json) · [Soft-GCOT 运行器](code/run_soft_fixedgroup.py) · [上传压缩包原始 GCOT 源码快照](code/zip_gcot/)

**方法学限定：**目标软隶属度基于已知同步配对索引继承，因此不是 TPAMI 原版中两域分别学习原型的无监督 Soft-GCOT；不能将本对照视为原论文 Soft-GCOT 性能的直接复现。

---

## 当前正式结果：500 次以内运行的五个种子

相同测试集、相同归一化、相同四维表示：

| 评价 | 对齐前 Identity | Hard HiWA 五次均值 ± 样本 SD |
|---|---:|---:|
| 固定组标签的 1NN ↑ | 23.14% | **37.74% ± 21.62%** |
| 平均绝对误差 MAE ↓ | 0.53149 | **0.49767 ± 0.05548** |
| 配对相对平方误差 ↓ | 1.79266 | **1.60396 ± 0.30249** |
| 旋转 rMSE 代理 ↓ | 1.96470 | **1.49268 ± 0.78664** |
| 标准测试 R² ↑ | -0.79649 | **-0.60738 ± 0.30314** |

各种子 test R² 分别是 -0.75816、-0.56025、-0.10161、-0.86298、-0.75392（seed 42–46）。**尽管平均改善且都通过数值审计，测试 R² 仍全为负，跨初始化变化很大**。最佳 seed 44 不能单独代表整个算法。

rMSE 代理并非真实旋转恢复误差，因为此真实神经运动数据集没有旋转真值 Rgt；用训练集同步对应拟合的正交 Procrustes 作为参考。配对相对平方误差则是 SSE / sum(||target||²)，二者公式不同，均不取平方根。R² 使用标准 1−SSE/SST，独立评价器纠正了原版内部不标准的 _eval_R2 公式，原版算法代码没有修改。

## 源域 → 对齐后源域 → 目标域：相同标签的图

![固定组的传输前后以及目标域：第一组二维投影](results/source_only_fixed_groups_outer500/before_after_target_xy.png)

[查看四维的全部六组二维投影（三列直接比较）](results/source_only_fixed_groups_outer500/before_after_target_six_pairs.png) · [对齐前源](results/source_only_fixed_groups_outer500/source_before_test_six.png) · [对齐后源](results/source_only_fixed_groups_outer500/source_after_test_six.png) · [目标域](results/source_only_fixed_groups_outer500/target_shared_test_six.png)

所有图均显示相同的 3500 个留出测试时窗；主图固定画 seed 42，不按测试结果择优选择画图种子。目标点颜色来自其配对源点的 K-means 标签。二维仅为可视化，K-means 与 HiWA 均在四维完成。

## 代码来源与复现

算法：用户上传 GCOT-reproducible-20261018.zip 中的 reference/HiWA/PyHiWA/src/hiwa.py 原版，项目目录里的 code/hiwa_original.py 的 SHA256 为 4419e19004b06c7840d187ddb12a19b2ad97974dd28ad1dc9510d8ead741da6a；其他旧版 src/hiwa 代码不参与本轮实验。

重现时先复用冻结的四维观测缓存，只重新分组、重新训练 HiWA：

- Python：安装 numpy、scipy、scikit-learn、matplotlib。
- 只聚类：python code/source_only_fixed_group_pipeline.py --input results/clustering/clustered_4d.npz --output results/source_only_fixed_groups --seed 42 --k 4
- 原版 HiWA：python code/run_hard_hiwa.py --clusters results/source_only_fixed_groups/fixed_groups_4d.npz --output results/source_only_fixed_groups_outer500 --seeds 42 43 44 45 46 --outer 500
- 评价与图：python code/evaluate_fixed_group_alignment.py --input results/source_only_fixed_groups/fixed_groups_4d.npz --alignment results/source_only_fixed_groups_outer500/hard_hiwa_results.json --output results/source_only_fixed_groups_outer500
- 生成论文数值：python code/write_converged_summary.py

上述命令相对于当前项目文件夹执行。GitHub Actions 的实际成功运行与全部计算工件保存在仓库历史及工作流运行记录中。

## 历史实验（不再作为当前协议）

- [旧实验：两域分别 K-means、140 次外循环](results/hard_hiwa/)——已废弃的分组设定，仅保留供方法论对比。
- [新单次源聚类，但仅 140 次外循环](results/source_only_fixed_groups/)——三种子未满足额外残差审计，现已由 500 次实验替代。
- [旧版技术诊断 PDF](四维HardHiWA实验与问题诊断.pdf)——仍描述原来的两个域独立分组实验，属于历史文档；现行报告是本 README 开头所链接的统一分组 PDF。
