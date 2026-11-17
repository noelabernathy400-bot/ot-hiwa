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
