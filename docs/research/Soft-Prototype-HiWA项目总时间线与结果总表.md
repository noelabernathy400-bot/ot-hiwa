# Soft-Prototype HiWA 项目总时间线与结果总表

日期：2026-07-07

本文是 Soft-Prototype HiWA / ROCA-HiWA 项目的阶段性总览。它不是最终论文结果表，而是把已经发生的实验、正结果、负结果和自然演化路线串起来，方便后续复现、写作和继续扩展。

## 1. 项目主线一句话

本项目从复现 [[HiWA]] 在猕猴神经元—运动数据上的无标签对齐开始，尝试借鉴 TACO 式 soft prototype / soft assignment 思想改进自动分组；随后发现 soft prototype 本身未直接显著提高方向分类 accuracy，但它提供的 soft representatives 暴露并解决了一个更关键的问题：HiWA / Soft-HiWA 在三维表示中实际存在 $O(3)$ 正交分支选择不稳定，ROCA-HiWA 用无标签的代表点有向体积选择 $\det(R)=+1$ 或 $\det(R)=-1$ 分支。

## 2. 阶段时间线

| 阶段 | 想验证什么 | 实验设置 | 结果 | 结论 | 局限 | 下一阶段为什么出现 |
|---|---|---|---|---|---|---|
| HiWA 官方代码理解 | HiWA 是否适合神经—运动无标签对齐 | 阅读论文、MATLAB/Python 实现与数据组织 | 确认可运行层级 OT + Procrustes 对齐 | HiWA 是可复现基线 | 对代码细节依赖强 | 需要在本地 Python 管线跑通 |
| HiWA 猴神经复现 | 官方/移植管线能否在猕猴数据上复现 | 三维 neural representation 与三维 movement representation | 连续运动 $R^2$ 较稳定，direction accuracy 对 seed 敏感 | 主要问题不是完全对齐失败，而是离散方向语义不稳定 | 数据集单一 | 需要找出 accuracy 不稳定来源 |
| Soft-Prototype HiWA 设计 | TACO 式软分组能否改善自动聚类 | 学习 soft prototypes 与 soft assignments，接入 HiWA | 方法实现可运行 | soft grouping 是可行结构 | 初版未证明性能提升 | 进入软分组实验 |
| soft grouping first experiment | soft assignment 是否直接提高 accuracy | soft random init / soft warm start 对比 | random init 不稳定；warm start 稳定但最初没有明显提高 direction accuracy | soft assignment 本身不是简单“加了就涨分” | 可能受温度和初始化影响 | 需要做 temperature sensitivity |
| temperature sensitivity | 温度是否控制 prototype 软硬程度并影响结果 | 多温度小规模对比 | 未稳定解决 direction accuracy 下降 | 温度不是核心旋钮 | 不宜继续大规模调参追分 | 进入退火实验 |
| held-out seeds 5-9 | 初步基线在保留 seeds 上是否稳定 | hard prototype / held-out seeds 5-9 | acc $0.544141\pm0.0745$，$R^2=0.575595\pm0.01685$ | hard baseline 有一定方向解码能力 | seed 数仍小 | 需要解释 soft 方法何时失败 |
| annealing | 温度退火能否兼顾稳定与软分组 | 从软到硬或温度渐变 | 未明显提升 accuracy | prototype softness 不是唯一矛盾 | 可能是 prototype drift | 做 pure annealing 排除 |
| pure annealing | 排除 prototype drift 后退火是否有效 | 固定/纯退火变体 | seed 7 acc $0.4093$，$R^2=0.6031$；hard seed 7 acc $0.5746$，$R^2=0.5511$ | 高 $R^2$ 与低 direction accuracy 可以同时出现 | 只展示关键 seed 诊断 | 需要诊断连续几何与离散语义分离 |
| seed 7 diagnosis | 为什么 $R^2$ 高但 accuracy 低 | 对 seed 7 的 candidate / branch 诊断 | 连续轨迹对齐好，方向语义错 | 问题可能在坐标方向、旋转或反射分支 | 单 seed 诊断不能泛化 | 进入 rotation-stabilized Soft-HiWA |
| rotation-stabilized Soft-HiWA | 固定旋转或分支能否稳定语义 | 对正交矩阵分支做显式控制 | 发现搜索空间实际为 $O(3)$，不是只含 $SO(3)$ | determinant branch 是核心不稳定来源 | 不是最终方法 | 需要显式比较 $\det(R)=+1/-1$ |
| O(3) determinant branch discovery | 两个正交分支是否对应不同 accuracy | determinant-constrained Procrustes 生成双候选 | seeds 10-29 中 $\det=-1$ 高，$\det=+1$ 低；hard-start acc 约 $0.51027$，$R^2$ 约 $0.60125$ | 分支选择确实决定方向语义 | 不能用测试标签选分支 | 需要无标签 selector |
| 双向运输目标选分支 | transport objective 能否无标签选分支 | seeds 30-49 | 高分支只选中 7/20；selected mean acc $0.47111$ | transport cost 不是可靠语义分支选择器 | 负结果，但很重要 | 需要使用几何方向信息 |
| representative-oriented selector | soft representatives 能否选择分支 | 用 K=4 soft representatives 构造有向四面体 | seeds 50-69：20/20 选择高 accuracy 分支；selected mean acc $0.577528$，min acc $0.573034$，mean $R^2=0.599344$ | soft prototypes 的新用途成立：无标签分支选择 | 仍只是真实单数据集 | 需要验证不是坐标巧合 |
| coordinate sign flip validation | selector 是否随坐标手性翻转而翻转 | seeds 70-79，source 坐标单轴符号翻转 | 30/30 按手性改变 determinant，30/30 选中高 accuracy 分支；mean acc $0.576404$，mean $R^2=0.599123$ | 不是机械固定选某个 determinant | 仍是真实数据派生验证 | 需要合成真值验证 |
| synthetic determinant validation | 已知 true determinant 时能否恢复真值 | seeds 0-49；true_det $\pm1$；noise 0/0.02/0.05/0.10；oracle/learned soft；normal/near_degenerate | 1600 cases，总准确率 0.995；normal 全部 1.0；8 个失败均为 learned_soft + near_degenerate + seed 42 | 支持 selector 能恢复已知 determinant；失败与软分组在弱体积边界附近有关 | near_degenerate 在标准化后未触发 hard degeneracy flag | 需要 refinement 和更多数据集 |
| literature search | 是否已有同样方法 | 初步检索 Procrustes、Kabsch、Wasserstein Procrustes、HiWA、signed volume、chirality | 目前未找到“soft representatives + signed volume + HiWA/neural alignment determinant branch”的完全相同方法 | 可定位为经典几何事实在 HiWA 稳定性问题中的新组合/新应用 | 仍需系统全文检索 TACO 和神经对齐文献 | 进入论文定位与扩展验证 |

## 3. 总结果表

| 阶段 | 方法 | Seeds | Direction accuracy | $R^2$ | 结论 |
|---|---|---|---|---|---|
| held-out baseline | hard prototype HiWA | 5-9 | $0.544141\pm0.0745$ | $0.575595\pm0.01685$ | baseline 可跑通，但方向 accuracy 有随机性 |
| seed 7 diagnosis | pure annealing | 7 | 0.4093 | 0.6031 | 高 $R^2$ 不保证方向语义正确 |
| seed 7 diagnosis | hard baseline | 7 | 0.5746 | 0.5511 | 低一点 $R^2$ 反而有更好 direction accuracy |
| determinant branch discovery | hard-start / O(3) candidate | 10-29 | 约 0.51027 | 约 0.60125 | 正交分支是主要不稳定来源之一 |
| negative selector | bidirectional transport objective | 30-49 | selected mean 0.47111；高分支 7/20 | 见原始结果 | transport objective 不足以选语义分支 |
| ROCA independent confirmation | representative-oriented selector | 50-69 | mean 0.577528；min 0.573034；20/20 高分支 | mean 0.599344 | 无标签 selector 在真实数据固定管线中稳定 |
| coordinate flip validation | ROCA selector with source-axis sign flips | 70-79，3 axes | mean 0.576404；30/30 高分支 | mean 0.599123 | selector 随手性变化自动改变 determinant |
| synthetic validation | ROCA selector | 0-49 synthetic | determinant recovery 0.995 | 不适用 | 已知 determinant 可被恢复；失败集中在 learned_soft near-degenerate |

## 4. 关键负结果

- soft random init 不稳定：说明 soft prototypes 如果没有良好初始化，会引入新的不稳定源。
- warm start 稳定但最初不提升 direction accuracy：说明 soft assignment 不是直接涨分模块。
- temperature sensitivity 没有解决 accuracy 下降：不应继续围绕温度无限调参。
- annealing 不提升 accuracy：软硬过渡本身不是关键机制。
- pure annealing 排除 prototype drift 后仍失败：说明问题更接近几何分支，而非单纯 prototype 漂移。
- seed 7 诊断显示 $R^2$ 高而 accuracy 低：连续运动拟合与离散方向语义可以分离。
- 双向运输目标选分支失败：transport cost 不等价于方向语义正确。
- rotation anchoring 不是主要贡献：最终贡献应是 representative-oriented determinant branch selection，而不是人为固定旋转。

## 5. 关键正结果

- TACO 式 prototypes 可以形成有结构的自动分组。
- Soft-Prototype HiWA 已经实现，并能作为 HiWA 的可扩展分组层。
- hard-start 提高了 soft grouping 的稳定性。
- HiWA / Soft-HiWA 在当前三维表示中实际搜索 $O(3)$，而不仅是 $SO(3)$。
- $\det(R)=+1$ 和 $\det(R)=-1$ 分支在方向 accuracy 上有系统差异。
- representative-oriented selector 在 seeds 50-69 上 20/20 选择高 accuracy 分支。
- coordinate flip validation 中 30/30 随手性变化自动翻转选择，并 30/30 选中高 accuracy 分支。
- synthetic determinant validation 在 1600 个合成条件中总体 0.995 恢复 true determinant。

## 6. 当前可信结论

当前证据支持 ROCA-HiWA 能在当前神经—运动数据与固定管线中，无标签地选择高准确率正交分支，并显著降低随机初始化带来的方向语义不稳定。但其普适性、新颖性和高维推广仍需进一步验证。

更谨慎地说，ROCA-HiWA 当前最强的贡献不是“全面提高所有 HiWA 指标”，而是发现并处理了一个原本隐藏在 $O(3)$ 对齐中的 determinant branch ambiguity：连续几何对齐可以很好，但离散方向语义会因为 reflection / handedness 分支而错位。soft representatives 提供了一个无标签、有向几何的分支选择依据。

## 7. 还不能声称什么

- 不能声称 ROCA-HiWA 已经普遍解决所有 HiWA 分支问题。
- 不能声称在所有神经数据集、所有 $K$、所有维度 $d$ 上有效。
- 不能声称 soft prototypes 本身必然提高 direction accuracy。
- 不能声称 related work 中绝对没有类似思想；目前只是初步检索未找到高度相同方法。
- 不能把测试标签用于训练、分支选择或调参；标签只用于最终评价。

