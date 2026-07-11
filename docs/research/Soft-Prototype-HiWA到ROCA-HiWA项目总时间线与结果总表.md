# Soft-Prototype-HiWA到ROCA-HiWA项目总时间线与结果总表

日期：2026-07-08

## 1. 一句话研究故事

本项目从复现 HiWA 的 macaque neural-movement 对齐开始，最初尝试借鉴 TACO 的 soft prototype / soft assignment 改进自动分组。实验发现 soft assignment 并不直接提升 direction accuracy，但它提供的 soft representatives 暴露并解决了更关键的问题：HiWA / Soft-HiWA 在三维表示中存在 $O(3)$ determinant branch instability。ROCA-HiWA 用 soft representatives 的有向几何结构，在无标签条件下选择 determinant branch，并用 condition number 做低置信诊断。

## 2. 阶段时间线

| 阶段 | 想验证什么 | 做了什么 | 结果 | 结论 | 局限 | 下一步为什么自然出现 |
|---|---|---|---|---|---|---|
| HiWA 官方代码理解 | HiWA 是否适合本项目 | 阅读论文、代码、数据结构 | 明确 hierarchical OT + Procrustes 管线 | 可作为基线 | MATLAB/Python 差异需处理 | 做 Python 复现 |
| HiWA 猴神经复现 | 主链路能否跑通 | macaque neural-movement 数据 | 连续 $R^2$ 较稳，accuracy 对 seed 敏感 | 问题不是完全对齐失败 | 不稳定原因不明 | 解释指标 |
| 指标解释 | accuracy、$R^2$、seed stability 各代表什么 | 对照不同 seed 与候选 | $R^2$ 高不等于 direction accuracy 高 | 连续几何与方向语义可分离 | 指标关系复杂 | 尝试 soft grouping |
| Soft-Prototype HiWA 设计 | TACO 思想能否迁移 | soft prototypes / assignments 接入 HiWA | 方法可实现 | soft grouping 可作为结构层 | 初期未涨分 | 做 first experiment |
| soft grouping first experiment | soft assignment 是否直接提升 | soft random / warm start | random init 不稳定；warm start 稳定但初期不提升 | soft 不是直接涨分按钮 | 受初始化影响 | hard / random / warm 对比 |
| prototype hard / soft random / warm | 初始化如何影响稳定性 | 多 seed 对照 | hard-start 改善稳定性 | 初始化重要 | 还未解释 accuracy 下降 | 查 temperature |
| temperature sensitivity | 温度是否是关键 | 温度扫描 | 未解决 accuracy 下降 | 不应继续盲调 temperature | 可能是退火路径问题 | 做 annealing |
| annealing | 软硬过渡能否改善 | 温度退火 | 未提升 direction accuracy | 退火不是核心解 | prototype drift 未排除 | 做 pure annealing |
| pure annealing | 排除 prototype drift | 固定/纯退火诊断 | seed 7：acc 0.4093，$R^2=0.6031$；hard acc 0.5746，$R^2=0.5511$ | $R^2$ 高可伴随 accuracy 低 | 关键 seed 现象 | seed 7 diagnosis |
| seed 7 diagnosis | 为什么高 $R^2$ 低 accuracy | 诊断候选分支 | 连续几何对齐与方向语义错位 | 可能是正交分支问题 | 单 seed | rotation-stabilized |
| rotation-stabilized Soft-HiWA | 旋转锚定能否稳定 | hard-start、anchor、分支诊断 | 发现算法实际搜索 $O(3)$ | determinant branch 是核心不稳定来源之一 | anchoring 不是贡献 | 构造 det 候选 |
| $O(3)$ determinant branch discovery | $\det=+1/-1$ 是否影响语义 | determinant-constrained candidates | 两个分支 accuracy 明显不同 | 需要无标签选分支 | 不能用标签选择 | representative selector |
| determinant-constrained candidate construction | 能否公平生成双候选 | Procrustes 更新加 determinant constraint | det=±1 candidates 可生成 | 分支问题可显式化 | 仍缺 selector | ROCA |
| representative-oriented branch selector | soft reps 能否无标签选分支 | 有向四面体体积 + Hungarian matching | seeds 50-69：20/20 选高分支 | ROCA 主发现成立 | 单真实数据 | 坐标翻转 |
| seeds 50-69 independent confirmation | 开发外 seeds 是否稳定 | 独立 seeds 50-69 | mean acc 0.577528，mean $R^2=0.599344$ | 无标签 selector 稳定 | 仍可能固定选 det | coordinate flip |
| coordinate sign flip validation | 是否只是固定选 det | source 单轴反射 seeds 70-79 | 30/30 随手性翻转；30/30 高分支 | selector 用有向几何而非固定偏好 | 仍是真实数据派生 | synthetic truth |
| synthetic determinant validation | 能否恢复 true_det | 1600 synthetic cases | determinant recovery 0.995 | 支持机制解释 | 8 failures | low-confidence diagnostics |
| degeneracy / low-confidence diagnostics | 失败是否可解释 | volume、condition、entropy、matching margin | warning_v0 抓 8/8，但误报 25 | 可预警，但不能最终拒绝 | false warning 多 | confidence calibration |
| confidence calibration | 固定 low-confidence rule | split 0-24/25-49；规则比较 | calibration 无 failure；validation 推荐 condition >80，抓 8/8，误报 0；真实 warning rate 0 | 真实结果未落入低置信区 | 阈值不普适 | 文献与外部验证 |
| literature search | 新颖性边界 | Procrustes、OT、HiWA、registration、chirality | 基础工具已有；组合未见同型 | 贡献是组合/迁移 | TACO 与 neural alignment 待补 | next-stage plan |
| next-stage plan | 下一步做什么 | 记录 proposal | 先文献/TACO/外部数据，再 K 扩展 | 主线保持清楚 | 仍需更多验证 | 准备组会/论文 |

## 3. 总结果表

| 阶段 | 方法 | Seeds / cases | Direction accuracy | $R^2$ | determinant recovery | 结论 |
|---|---|---:|---:|---:|---:|---|
| HiWA baseline | hard prototype | seeds 5-9 | $0.544141\pm0.0745$ | $0.575595\pm0.01685$ | 不适用 | 主链路跑通 |
| pure annealing diagnosis | pure annealing | seed 7 | 0.4093 | 0.6031 | 不适用 | $R^2$ 高不等于 accuracy 高 |
| hard seed 7 | hard baseline | seed 7 | 0.5746 | 0.5511 | 不适用 | direction 语义可与 $R^2$ 分离 |
| determinant branch | hard-start / O(3) | seeds 10-29 | 约 0.51027 | 约 0.60125 | 不适用 | determinant branch 是不稳定来源 |
| bidirectional selector | transport objective | seeds 30-49 | selected mean 0.47111 | 见原始结果 | 高分支 7/20 | 负结果，不能可靠选分支 |
| ROCA confirm | representative selector | seeds 50-69 | mean 0.577528；min 0.573034 | mean 0.599344 | 20/20 高分支 | 无标签 selector 稳定 |
| coordinate flip | ROCA + source reflection | 30 cases | mean 0.576404；min 0.534510 | mean 0.599123 | 30/30 预期翻转 | 通过手性压力测试 |
| synthetic truth | ROCA selector | 1600 cases | 不适用 | 不适用 | 0.995 | 支持真值 determinant 恢复 |
| warning_v0 | diagnostic warning | 1600 cases | 不适用 | 不适用 | 抓 8/8 failure | 误报 25，不能最终拒绝 |
| confidence split | condition > 80 candidate | validation 800 cases | 不适用 | 不适用 | non-warning 1.0；warning 0.0 | 抓 8/8，误报 0；真实 warning 0/50 |

## 4. 必须保留的负结果

- soft random init 不稳定；
- soft warm start 初期没有明显 accuracy 增益；
- annealing 没有提升 direction accuracy；
- pure annealing 排除 prototype drift，但仍然不能救 accuracy；
- $R^2$ 高不等于 direction accuracy 高；
- 双向运输目标不能可靠选分支；
- rotation anchoring 不是主要贡献；
- warning_v0 误报较多，不能作为最终拒绝规则。

## 5. 必须突出但谨慎的正结果

- HiWA 主链路跑通；
- TACO prototypes / soft grouping 有结构；
- Soft-HiWA 可实现；
- hard-start 改善稳定性；
- $O(3)$ determinant branch 是核心不稳定来源之一；
- ROCA selector 20/20 选择高 accuracy 分支；
- coordinate flip 30/30 随手性翻转；
- synthetic determinant recovery 总体 0.995；
- failure cases 有无标签低置信信号。

## 6. 当前最可信结论

当前证据支持：在当前 macaque neural-movement 数据与固定 HiWA / Soft-HiWA 管线中，ROCA-HiWA 能利用 soft representatives 的有向几何结构，在不使用方向标签、测试 accuracy 或测试 $R^2$ 的条件下选择高方向语义稳定性的 determinant branch。该方法在真实数据、坐标手性翻转和合成真值实验中得到初步支持，但其普适性、高维推广、自动 $K$、外部数据验证和文献新颖性仍需进一步确认。

