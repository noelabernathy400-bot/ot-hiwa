> **当前主目标（2026-11-17确认）：完全无监督的跨模态分布对齐质量。** 主要评价是 Hard 和 Soft 共用同一经验点云的 W₂²；同步配对 R² 是二级指标。注意：下方原始“源域聚类→目标继承标签”实验仅属于组级监督诊断，不可称为正式无监督训练。完整定义与全部结果归类见 [研究目标及统一评价规范](RESEARCH_OBJECTIVE_AND_EVALUATION.md)，无监督训练结果见 [独立分组实验](unsupervised_control/RESULTS.md)。

# Hyser：源域唯一 K-means → 目标继承标签 → Hard HiWA

**版本：2026-11-17。此目录取代原先 Hyser 的各域独立聚类实验及其图片。**

真实 PhysioNet Hyser 受试者01、session1 的 Random Task sample1。256通道肌电2048Hz、五指力量100Hz，同步25秒。以0.1秒切窗获得250对样本。肌电做窗口RMS、log1p、通道标准化、PCA3；力量选thumb/middle/index三指窗口均值。两域分别协方差白化成3D，均值约0、协方差为单位阵。

**只在X上做一次K-means**（K=4，seed=42，n_init=30），组大小46、53、91、60。目标域Y的第i个点继承X第i个点的标签，绝不再在Y上聚类。因此使用了真实同步对应关系，**是组级监督诊断，不是完全无监督HiWA。**

## 最新结果：全部250对点

| 方案 | R² ↑ | 相对平方误差 ↓ | 同步配对平均欧氏距离 ↓ |
|---|---:|---:|---:|
| 对齐前 | -0.674071 | 1.674071 | 1.977467 |
| Hard HiWA：自由簇级 OT | -0.169051 | 1.169051 | 1.697697 |
| Hard HiWA：固定已知组对应 | -0.153907 | 1.153907 | 1.688001 |
| 配对最佳正交Oracle | -0.136928 | 1.136928 | 1.688373 |

自由OT使用 μ=0.005，75次迭代满足原作者的全局旋转变化准则，但局部共识残差仍约0.99。固定组对应是P=I4/4的**修改版算法**，μ=0.05，60次迭代后的旋转变化0.00339、局部共识残差0.00551。二者并非同一μ的严格消融。

## 实验文件与图（所有旧图已替换）

- [中文LaTeX PDF](Hyser_HardHiWA_最新实验.pdf) · [源文件](Hyser_HardHiWA_最新实验.tex)
- [完整过程笔记](EXPERIMENT_LOG.md) · [数值结果、旋转矩阵、P矩阵](results/latest.json)
- [对齐前源域/目标域3D及2D](figures/01_before_hiwa.png)
- [自由组级OT对齐后的3D及2D](figures/02_after_hiwa_free.png)
- [固定已知组对应的3D及2D](figures/03_after_hiwa_fixed.png)
- [并排总览](figures/00_complete_comparison.png) · [自由组P矩阵](figures/04_free_group_transport.png)
- [图像生成代码](scripts/render_figures.py) · [本地HiWA数值移植与有约束版本](scripts/solver.py) · [WFDB原始数据预处理说明](scripts/prepare_from_wfdb.py)

工作流会从保存在data中的**真实250点可视化量化坐标**生成图片并编译PDF。该图像用点云的有限精度近似仅用于绘图；数值表来自以前真实点云**原始全精度**运行，不能以量化数据重新拟合冒充原实验。

本轮求解是按用户上传GCOT项目更新公式的本地移植执行，并**非远程原文件直接导入**。两种HiWA的结果仅限同一25秒试次的拟合内诊断。
## 2026-11-17：Hard OT 代价复核及源域软分组新实验

- [Hard OT 代价审计](results/HARD_OT_AUDIT.md)：自由组 OT 的全局旋转成本 **1.385820 → 0.843677**；固定对应P的成本 **1.384391 → 0.835982**；另有无熵 W₂² 对照。
- [Soft-GCOT式软分组完整报告](results/SOFT_EXPERIMENT_20261117.md) 与 [JSON结果](results/soft_experiment_20261117.json)。
- [5种子、3方法的15条原始记录](results/soft_multiseed_40_44.csv)。
- [软分组实验复算脚本](scripts/run_soft_inherited.py)：直接读取官方WFDB预处理脚本得到的全精度数据；目标域**继承源域软隶属度，不单独分组**。
- [更新的中文 PDF](Hyser_HardHiWA_最新实验.pdf)：原报告及新增硬软对照与软隶属度图（自动构建更新）。

本轮固定组对应且同 μ=0.05：Hard R²=-0.153907；Soft K-means τ=.5 为 -0.150957；Soft原型Eq2 λ=1 为 **-0.142562**；Oracle=-0.136928。5个种子40-44均验证这个方向。注意Soft的概率组分布与Hard不同，**各方法的OT绝对代价不是同一固定目标函数**。自由P的Soft原型没有获得稳定改善，还出现了组匹配错误且ADMM共识未收敛。

目标继承隶属度通过真实同步配对完成，故这仍然是**已知组对应诊断实验**，不是论文中完全无监督的双域Soft-GCOT。求解代码是GC-OT公式的本地移植，而非直接import原始远程代码。


## 2026-11-17：已完成真正独立分组的无监督控制实验

保留此前唯一源域K-means／目标继承标签的已知组对应诊断实验，并另设 [unsupervised_control](unsupervised_control/RESULTS.md) 作为不向目标传递任何真实分组关系的严格控制。K=4、三维白化、μ=.05、相同HIWA缩放、样本/组熵参数一致。4×4组级OT改用数学等价的高精度熵正则对偶根求解，15/15运行通过旋转变化、共识和边缘质量阈值检查（不等于证明全局最优）。

5种子40–44的平均真实配对R²：独立Hard **-0.92003 ±0.16675**、独立Soft K-means **-0.86820±0.04808**、独立Soft原型 **-1.04793±0.21158**。Soft K-means只在3/5个种子优于Hard；没有发现稳定恢复真实配对的证据。seed42时，Hard自由组P的OT成本2.36079→0.39606，但其真实配对R²仍为-0.86443。**低运输代价与恢复同步物理对应不同**。原始数据处理、真正分组对应和评估信息边界见独立报告。


## 已完成：统一无监督分布距离评价（2026-11-17）

**主指标**：完全相同的250个三维点、同一白化尺度、相同样本权重上的无熵精确W₂²。对齐前共同0.241479；独立分组5种子对齐后均值 Hard **0.199080**、Soft K-means **0.207529**、Soft原型 **0.231263**，平均降幅分别17.56%、14.06%、4.23%。Soft K-means 4/5 种子微弱胜出，但种子44失败，平均Hard更好。这是已有15个真实无监督R的**新公平评价**，不是重新训练模型。见 [15次全量统计、误差定义及审计](unsupervised_control/COMMON_W2_RESULTS.md)、[逐seed CSV](unsupervised_control/common_w2_unpaired_5seed.csv)、[15个R矩阵](unsupervised_control/fitted_R_15runs.csv)、[复算代码](unsupervised_control/evaluate_common_w2.py)。本轮不使用真实同步标签训练/选择。


## 2026-11-17：新训练的完全无监督Soft K-means温度消融

在前述统一W₂²主指标确定后，进一步**新运行10次独立两域软分组训练**：τ=0.25和τ=1.0，各seed40–44；其余条件固定（K=4、μ=.05、εs=.1、εg=.2、三维白化、自由P、ADMM/OT共同残差阈值）。加上此前已存在的τ=0.5五次训练得到：Hard平均W₂² **0.19907977**（改善17.56%）、Soft τ=.25 **0.19907906**（17.56%）、Soft τ=.5 **0.20752896**（14.06%）、Soft τ=1 **0.20093204**（16.79%）。全部新增10/10次通过本轮数值残差检查；τ=.25与Hard平均差只有约7×10^-7，不能视作实质性改进。τ=.5种子44退化、τ=1种子40退化，说明软温度与初始化的相互作用仍待研究。

已保存 [完整温度试验报告](unsupervised_control/SOFT_TEMPERATURE_SWEEP.md)、[10条逐seed数值](unsupervised_control/temperature_sweep_unpaired_10new.csv)、[10个R矩阵](unsupervised_control/temperature_sweep_R_10new.csv)、[10个簇级P矩阵](unsupervised_control/temperature_sweep_P_10new.csv) 与 [可运行新训练程序](unsupervised_control/run_temperature_sweep.py)。另外重复τ=.5、seed42结果与历史W₂²=0.18870520630894314**完全相同**。训练与共同W₂²均未使用真实同步配对，R²仍只是辅助评价。这些是同一试次的拟合内数据，不提供跨试次泛化证明；依赖本地数值移植的代码来源限制仍在。
