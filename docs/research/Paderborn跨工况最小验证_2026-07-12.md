# Paderborn 跨工况最小验证

_日期：2026-07-12；状态：数据接入成功，当前最小任务不进入扩展实验。_

---

## 🎯 目的

建立一个不同于 MiHiA 的真实工业应用：在源工况的有标签轴承振动数据上训练最终故障分类器，在目标工况上只做无监督 Hard HiWA 或 Fixed Soft-GCOT 对齐，最后才读取目标标签评价。

Paderborn 官方数据包含同步振动、电流、转速、扭矩、径向载荷与温度，并为每个轴承状态和工况提供 20 次、每次 4 秒的测量。[^1] 本次使用健康 K001/K002、真实外圈损伤 KA04/KA15、真实内圈损伤 KI04/KI14；该三类对应关系可由公开的数据集说明交叉核对。[^2]

## ⚙️ 构造

| 项目 | 设置 |
| --- | --- |
| 源工况 | `N15_M07_F10`：1500 rpm、0.7 Nm、1000 N |
| 目标工况 | `N09_M07_F10`：900 rpm、0.7 Nm、1000 N |
| 每域样本 | 96：每个类别 4 条记录、每条记录 8 个不重叠窗口 |
| 表示 | `vibration_1` → log FFT 幅值谱 → 256 频带均值 → 合并标准化与 PCA（20 维） |
| 分组 | 4 个无监督组；同 prototype Hard 与 full-support Fixed Soft-GCOT |
| 禁止使用 | 目标标签不参与特征、分组、对齐、参数选择或分支选择 |

## 📊 两个必要诊断

| 任务设计 | 未对齐 accuracy | Hard HiWA | Fixed Soft-GCOT | 解释 |
| --- | ---: | ---: | ---: | --- |
| 同一实体轴承，仅转速改变 | 0.9375 | 0.3333 | 0.3229（未收敛） | 原始 FFT 特征已可诊断，对齐没有必要 |
| 不同实体轴承，同时转速改变 | 0.3333 | 0.3333 | 0.3333（未收敛） | 三分类随机水平；每类只有一个实体轴承，任务对实体差异过度敏感 |

第二个任务中 Hard HiWA 达到当前收敛阈值（31 次迭代；全局残差 0.0937；primal 残差 0.0738），但下游诊断仍为随机水平。Fixed Soft-GCOT 达到最大 80 次迭代仍未满足共识要求（global/primal 残差约 2.0）。

## 🔎 结论

Paderborn 是合适的工业应用数据源，数据读取、信号字段、工况命名、无标签特征构造和目标域评价协议都已打通。

但当前每类只放入一个源实体和一个目标实体的 96 样本任务不能作为方法优劣证据：它要么过于容易、无需对齐；要么同时混入实体泛化与工况迁移，且样本量不足以稳定估计类别结构。因此不扫描参数、不增加 seed，也不以本次结果否定或修改 Soft-GCOT。

## ✅ 下一步决定

Paderborn 分支暂时冻结在“数据与任务可构造”阶段。若后续继续，应先预先定义一个更完整的实体级协议：每个健康／外圈／内圈类别在源与目标域各包含多个不同轴承实体，并严格按实体划分源、目标与最终测试。之后才值得重新运行 Hard 与 Soft 的单一对照。

本次可复现实验输出：

- `experiments/results/paderborn_soft_gcot_speed_shift_96_seed130.json`
- `experiments/results/paderborn_soft_gcot_speed_and_bearing_shift_96_seed130.json`

[^1]: Paderborn University, [Data Sets and Download](https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download); [Operating Conditions](https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/operating-conditions).
[^2]: P. N. Sai et al., [A Siamese Vision Transformer for Bearings Fault Diagnosis](https://pmc.ncbi.nlm.nih.gov/articles/PMC9607027/), Table 13.
