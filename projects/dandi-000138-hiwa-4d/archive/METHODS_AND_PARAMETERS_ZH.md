# DANDI 方法、模型、数值参数与评价定义台账

**状态基准：2026-11-19。** 所有表内数值只属于 `projects/dandi-000138-hiwa-4d/`；Hyser、合成正向对照及本项目真实指标不得混写。完整中文解读见 [最终总结](FINAL_EXPERIMENT_SUMMARY_ZH.md)。

## 1. 数据与输入设定

- **DANDI:000138**, version `0.220113.0407`, MC_Maze_Large / `sub-Jenkins_ses-large_desc-train_behavior+ecephys.nwb`；原始 148,590,536 字节；SHA256 `1188ddf5b822dd9ac49afb8916f4b42a85867613cf7f169d0b04a1d1aeecd700`。
- 500 usable trials，运动起始相对窗口 \([-100,600]\) ms，每个 trial 35 个不重叠 20ms 窗：共 17,500 个同步神经与运动观测。**同步关系存在于原始记录中，但训练的无配对 OT 不读取它**。
- 神经 162 units 的放电率→sqrt→时间高斯平滑 \(\sigma=40/20=2\) bins→仅训练集通道 StandardScaler→PCA / FA 潜在维度。
- 目标原始手部四个物理量 \(y=(p_x,p_y,v_x,v_y)\)，非 cursor。4D 用仅训练集逐通道 z-score；3D 在这四个标准化量上**仅由训练集拟合 PCA3**。该三维不是“三个单独的物理量”，在现有 320/80 实验中保留原手部运动标准化方差 **83.6148%**。
- 原始可用 trial 划分：`seed=42` 的 400 train (14,000 windows) /100 test (3,500 windows)；**原100 test 已经被早期反复探索，不能再充当全新盲测**。最近可比评估：原400个 train 依 `seed=20261119` 拆 320 inner-fit (11,200) /80 inner-valid (2,800)；这80后续也被反复分析，只是内部探索。
- 两域在训练集分别做中心化后除各自全局 RMS 半径 \(s_D=\sqrt{n^{-1}\sum_i\|z_i-\bar z\|_2^2}\)；算法拟合没有使用同步样本索引。**4D 手部位置和速度的原始数值单位尚未完整核定，禁止随意标厘米或厘米/秒。**

## 2. 模型台账：哪些实际在 DANDI 跑过？

| 方法 | 目的/数学骨架 | DANDI 实际状态 | 公平比较边界 |
|---|---|---|---|
| Identity / Random matching | 不做变换/随机目标，确定零信息水平 | **实测** | 必须对比真实匹配，而非仅 OT |
| 原版 **PyHiWA Hard** | 分别独立硬聚类 + 局部 \(R_{k\ell}\)、\(Q^{k\ell}\)、组间 \(P\)、ADMM 共识 | **4D、原400/100历史5种子已运行** | 数值实现与同核 ZIP 不是纯单因素对照；原始最后局部Q有缺档 |
| 同核 **Hard-HiWA** | 与Soft同一 GCOT P/Q/R 优化器、One-hot 组权重 | **4D、3D全运行** | 与 Soft 同特征、同种子、同求解器才可归因分组 |
| **Soft-GCOT** | 源/目标分别独立软原型概率 \(A,B\)、P自由运输、局部Q与R、共识ADMM | **4D、3D全运行** | 不同组权重导致原生T/J不可直接按绝对值比较；部分4D未数值收敛 |
| **SA** / Subspace Alignment | full-basis PCA子空间基相乘得到正交全局变换 | **FA3、FA4同内验真实运行** | 不使用P/Q；作者式 full-basis 转写，单个训练点集确定性结果 |
| **WA** / Wasserstein Alignment | 无分组全局正交R与Sinkhorn Q交替；\(\gamma:100\to0.1\)退火 | **FA3、FA4同内验真实运行3 seeds** | 无P/Q层级；与HiWA对比共同外部 W2和物理匹配 |
| **DAD-like / modified DAD** | 密度 \(k\)-NN KL 对称代理 + 三维旋转/反射粗网格与局部搜索 | **FA3同内验真实运行3 sets** | **不是 KordingLab 的原始 MATLAB DAD，不能冒充官方复现；FA4未运行** |
| CORAL | 协方差对齐，非正交变换 | **未在当前DANDI 320/80协议独立比较** | 未来若需增强基线再补 |
| ICP | 最近点迭代形变，可能含平移 | **未在当前DANDI 320/80协议独立比较** | 未来预注册；不能与同结构算法混作纯分组对照 |
| HiWA-SSC | SSC 硬聚类 + HiWA | **未在当前 DANDI 同内验协议运行** | 不是Soft-GCOT同因素比较 |
| HiWA-GMM | GMM 独立组 + HiWA，可伴随作者式投影 | **未在当前 DANDI 同内验协议运行** | 可能同时改变聚类与特征映射 |
| 有监督 Procrustes | 使用已知同步配对学习正交R | **历史仅作诊断参考** | **不能参加正式无监督训练/调参** |

**基线来源**：SA/WA 依照该 GitHub 项目历史研究源 `research/GCOT-reproducible-20261018/GC-OT/gcot/baselines.py` 的公式做 row-vector 约定转写；DAD 的官方项目为 `https://github.com/KordingLab/DAD`，本项目实现为**明确变体**：仅在3D下使用训练点的 \(k=5\) 邻域密度距离比、对称密度KL代理、每轴 \(\{0,\pi/2,\pi,3\pi/2\}\) × 2种镜像的**128个初值**，再 Powell 局部搜索≤45次迭代。不含原官方 MATLAB 的平移、网格密度处理等完整细节。实测源码：[SA/WA/DAD 对照实现](../code/closeout_dandi_sa_wa_dad_baselines.py)、[结果](../results/closeout_baselines_real_DANDI/DANDI_SA_WA_DAD_ADAPTED_RESULTS.json)。

## 3. 算法的精确目标与输出

两域行向量均归一化到相同维度 \(d\)。对齐后样本对代价（ZIP `scale=True`）：

\[
D^{k\ell}_{ij}(R)=\left\|\frac{x_iR^\top-y_j}{\sqrt d}\right\|_2^2.
\]

每个簇对源/目标条件组质量为 \(a^k,b^\ell\)，\(Q^{k\ell}\) 的行列边缘分别为 \(a^k,b^\ell\)；\(P\in\mathbb R_+^{K\times K}\) 行列边缘在现有协议为 \(1/K\)。两层原生**纯运输**是

\[
C_{k\ell}=\langle Q^{k\ell},D^{k\ell}(R)\rangle,\qquad
T_{\rm hier}=\sum_{k,\ell}P_{k\ell}C_{k\ell}.
\]

ZIP `retain_plans=True` 对应的**完整带熵目标**为

\[
J_{\rm hier}=T_{\rm hier}
  +\epsilon_s\sum_{k,\ell}H(Q^{k\ell})
  +\epsilon_g H(P),\qquad
H(M)=\sum_{ij}M_{ij}\log M_{ij}.
\]

其中 \(\epsilon_s=0.1,\epsilon_g=0.2\)；条件簇内Sinkhorn使用 \(\epsilon_s/P_{k\ell}\)。代码层的“无缩放” \(T\) 有时乘 \(d\) 把代价写回原潜空间的 \(\|Rx-y\|^2\)；必须查具体数据字段，**严禁把归一化和未归一化值混表**。\(\mu\) 是 ADMM 共识罚参数，**不是 J 的直接加项**。原生 Hard/Soft 的组质量/熵基线不同，\(T/J\) 数值不完全是共同目标。后验用新Q重算的\(J\)不是训练末轮实际局部Q构造的\(J\)。

不分组 SA / WA / DAD-like 不产生 \(P\) 与 16组 \(Q^{k\ell}\)，因此不存在与 HiWA 同义的训练层级运输 \(T/J\)。它们只比较**相同维度、相同验证采样**的自由 \(W_2^2\) 与同一原始物理检索。

## 4. 冻结参数表：不要混淆不同运行

| 参数 | 早期 4D 400/100 | 内部 4D PCA4/FA4 | 内部 3D PCA3/FA3 |
|---|---|---|---|
| 原始数据 | 同一DANDI NWB，17,500窗 | 同一NWB，内训320/内验80 trials | 同左 |
| 源表示 | 神经 PCA4 | 神经 PCA4 / FA4 | 神经 PCA3 / FA3 |
| 目标表示 | 原始位置/速度4D | 原始位置/速度4D | 训练拟合目标 PCA3 |
| OT潜维 \(d\) | 4 | 4 | 3 |
| 组数 \(K\) | 4 | 4 | 4 |
| 每组样本 / 每域OT样本 | 40 / 160 | 40 / 160 | 40 / 160 |
| 模型重复seed | 42–46 | 42,43,44 | 42,43,44 |
| 独立训练组 | 源/目标分别KMeans或Soft原型 | 同左 | 同左 |
| \(\epsilon_s\), \(\epsilon_g\), \(\mu\) | 0.1 / 0.2 / 0.05 | 0.1 / 0.2 / 0.05 | 0.1 / 0.2 / 0.05 |
| 外层max / 内层max | 500 / 20（ZIP同核） | 500 / 20 | 500 / 20 |
| 样本/组Sinkhorn最大迭代 | 45 / 200 | 45 / 200 | 45 / 200 |
| 软原型熵权重 / 原型max | 1.0 / 300 | 1.0 / 300 | 1.0 / 300 |
| 验证全局 \(W_2^2\) | 从原3500 heldout 独立抽200+200 | 从内部2800独立抽200+200 | 与4D相同抽样索引 |
| 层级 heldout Q | 独立128+128样本; 重新求Q | 独立128+128样本; 重新求Q | 同左 |
| 原始物理验证 | 历史原3500测试探索 | 内部2800，排除同trial | 同左，原四维手部量评分 |
| 完整P/Q/R是否归档 | 随具体实验而异 | **每次完整16组Q/P/R归档** | **每次完整16组Q/P/R归档** |

**原版 PyHiWA** 的旧初版只用过 outer140、stop-tol 1e−9 + 单独共识检查，绝不能与 ZIP 的 outer500 误称为完全相同求解器。**早期“温度”tau0.2/0.4/0.8**采用了目标继承源分组，故是**组级监督诊断**；真正无配对 Soft 是使用分别独立拟合的 prototype `soft_groups(...entropy_weight=1.0)`，不是同一套 tau0.2/0.4/0.8 参数变化。另一个目标几何消融使用 \(\lambda_{shrink}=0.15\)，四种事先设定的目标模式是 identity / block_zca / full_zca / arcsinh。

**SA**：原作者式 full-basis PCA 重新标准化矩阵，内训每个seed的各域 160样本；确定性列单次 seed42，不能当作有意义的种子稳定性统计。

**WA**：200 outer，组边缘均匀，各域160输入，\(\gamma_0=100\)，每轮当\(\gamma>0.1\)乘0.95；每轮 Sinkhorn 上限1000，近似正交Procrustes更新，随机种子42–44。

**DAD-like 3D**：训练各域160点，三维、\(k=5\) Nearest-neighbor density-KL 代理、128旋转/反射候选与局部Powell优化；这不是官方DAD可核查的完整实验，也未在FA4运行。

## 5. 评价口径与统计单位

| 指标 | 定义、阈值与解释 |
|---|---|
| 自由经验 \(W_2^2\) | 独立抽取源与目标各200个验证点、同权精确匈牙利最佳一对一距离平方均值；只评价边缘分布，不评价真配对 |
| 组级 \(P\)、簇对 \(C\)、样本 \(Q\) | 双层模型核心；必须核对行列边缘、组质量及所有16簇对；不能只看P对角（组编号可任意置换） |
| 原始位置误差 | \(e_i^{pos}=\|p_{j^*(i)}-p_i^{true}\|_2\)；最近邻在潜空间寻找\(j^*\)，但评价在原始手部位置单位 |
| 原始位置 Hit@10% | \(e_i^{pos}\le0.1\times\mathrm{median}_{train}\)每trial路径长度；当前内训半径约18.4567（原始存储单位）。另报5%/20%容差曲线、随机检索 |
| 原始速度方向 Hit@30° | 候选与真实速度夹角≤30°；真实速度低于训练20%速度分位者预先排除；额外报45°和随机 |
| 排除同trial目标 | 20ms相邻时窗极相关，为防止伪近邻同trial目标不允许候选 |
| Identity与随机 | 同一物理候选池、同一验证试次；随机内验位置10%=10.00%、方向30°=13.20% |
| 数值稳定性 | 保存P/Q边缘、\(R^TR-I\)、全局变化、局部-全局共识、ADMM对偶残差、是否到预算上限；**任何一个失败不能报严格收敛** |
| 重复统计单位 | 3 seeds是**同一**320/80分割上的优化重复；2800窗也不是2800个独立实验。正式组间显著性需独立trials/subjects及重复划分 |
| 3D↔4D | 只能直接比较同一原始四维物理匹配；不同潜空间的OT原生数值不可简单排名 |
| 新增基线 | FA3 SA/WA/DAD-like 与 FA4 SA/WA 在同一内验协议；SA只一个确定性解，WA/DAD-like不同样本训练3份 |

## 6. 曲线及尚未进行的敏感性实验

**已经真实计算并绘图**：[位置5/10/20%](figures/POSITION_TOLERANCE_3D_4D.png)、[速度30/45°](figures/DIRECTION_TOLERANCE_3D_4D.png)、[目标变换](figures/DANDI_4D_TARGET_TRANSFORM_ABLATION.png)、[SA/WA/DAD-like与Hard/Soft](figures/DANDI_FA3_HARD_SOFT_SA_WA_DAD_BASELINES.png)、[旧tau泄漏**警告图**](figures/LEGACY_TAU_SWEEP_SUPERVISED_WARNING.png)。每张图的数值源见 [逐行CSV](tables/DANDI_3D4D_REAL_PHYSICAL_TOLERANCE_CURVES.csv) 和 [图表manifest](FIGURES_EVIDENCE_MANIFEST.json)。

**没有开展的严格无监督参数曲线**：\(\tau\) 温度、\(K\) 簇数、\(\mu\) ADMM罚、\(\epsilon_s/\epsilon_g\) 熵，以及用户关心的**真实组重叠程度—匹配准确率**曲线。这些只能列为下一数据研究问题与预注册计划，不能画假数据、也不能把旧监督泄漏图当成成功消融。完整下一轮研究协议见 [实验工作流](NEXT_EXPERIMENT_WORKFLOW_ZH.md)。
