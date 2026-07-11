# ROCA-HiWA后续探索建议

日期：2026-07-07

本文是研究备忘录，不是实验结果，也不是当前协议的一部分。本阶段主线仍冻结为 ROCA-HiWA：用 soft representatives 的有向体积无标签选择 HiWA / Soft-HiWA 的 determinant branch。下面方向值得后续探索，但不应在当前阶段直接大规模展开。

## 1. $K=5$ 或自动 $K$

- 为什么值得做：当前 $K=4,d=3$ 刚好形成一个四面体，数学解释清楚，但真实神经结构未必正好四组。
- 回答什么问题：ROCA 是否依赖 $K=d+1$ 的巧合？如果自动分更多簇，selector 是否仍稳定？
- 和当前主线关系：这是 ROCA 从“漂亮的四面体机制”走向更一般 prototype structure 的关键。
- 什么时候做：先完成当前 $K=4$ 的合成真值、文献定位和退化检测后再做。
- 第一版干净实验：固定其他参数，只比较 $K=4$ 与 $K=5$；$K=5$ 不直接用所有点，而用 simplex subset voting。
- 风险：如果直接改 $K$，会把 group matching、simplex selection、assignment entropy 和 determinant branch 混在一起，解释会变脏。

## 2. GMM / spherical k-means / clustering ablation

- 为什么值得做：当前 learned_soft 的失败集中在 near_degenerate 条件，说明 assignment 质量会影响 representative。
- 回答什么问题：ROCA 成功来自“soft prototype 思想”，还是只要有稳定 representatives 就可以？
- 和当前主线关系：用于拆分“代表点质量”和“ROCA orientation rule”的贡献。
- 什么时候做：等当前方法冻结后，作为 ablation 进行。
- 第一版干净实验：同一 X/Y、同一 K=4、同一 selector，分别输入 hard k-means、spherical k-means、GMM posterior、当前 learned_soft assignments。
- 风险：容易变成聚类方法横评，偏离 determinant branch 机制。

## 3. $P_{ij}$ 加权全局旋转共识

- 为什么值得做：当前 ROCA 先做 group matching，再用四面体 orientation 选分支；也许可以把 group transport matrix $P$ 中的软对应权重直接用于全局共识。
- 回答什么问题：能否不用单一 Hungarian matching，而用所有 group-pair weights 更稳健地估计 determinant？
- 和当前主线关系：这是 group-conditioned OT 与 ROCA selector 的更紧密融合。
- 什么时候做：当 Hungarian matching 出现不稳定或多个近似匹配时。
- 第一版干净实验：在 synthetic 数据中制造两个代表点接近、matching margin 小的情形，对比 Hungarian-ROCA 与 $P$-weighted simplex / covariance orientation。
- 风险：公式可能复杂化，且 $P$ 本身可能受错误 branch 影响，需要避免循环依赖。

## 4. $K>d+1$ 时 representative subset 或 simplex voting

- 为什么值得做：$d$ 维空间中最小有向 simplex 需要 $d+1$ 个点；当 $K>d+1$ 时，可以从多个 simplex 投票。
- 回答什么问题：ROCA 能否自然推广到更多 prototypes？
- 和当前主线关系：这是 $K=5$/自动 $K$ 的数学基础。
- 什么时候做：在 $K=4$ 论文故事稳定之后。
- 第一版干净实验：$d=3,K=5$，枚举 5 个四面体，按体积绝对值和 matching confidence 加权投票。
- 风险：subset voting 可能引入选择偏差；如果用结果好坏挑 subset，会形成隐性调参。

## 5. $d$ 维一般化

- 为什么值得做：当前神经和运动表示被降到三维；更高维神经 manifold alignment 可能需要 $d>3$。
- 回答什么问题：oriented simplex selector 是否能从 $d=3,K=4$ 推广到 $d$ 维 $K=d+1$ 或 $K>d+1$？
- 和当前主线关系：决定 ROCA 是三维特例，还是通用 orientation branch selector。
- 什么时候做：完成 $K>d+1$ simplex voting 后。
- 第一版干净实验：合成 $d=2,3,5$，每个维度用 $K=d+1$，true_det $\pm1$，检查 determinant recovery。
- 风险：高维体积数值不稳定，representatives 更容易退化。

## 6. 其他神经数据集

- 为什么值得做：当前真实数据只有一个猕猴神经—运动数据集。
- 回答什么问题：ROCA 是否是数据集特例？
- 和当前主线关系：验证普适性和论文可信度。
- 什么时候做：当前代码和文档冻结后。
- 第一版干净实验：选择一个有连续行为变量、可构造三维运动表示的数据集；不改 selector，只迁移预处理。
- 风险：不同数据集可能没有同样方向标签或同样几何结构，评价协议需要重新设计。

## 7. session-to-session neural alignment

- 为什么值得做：神经流形跨天/跨 session 对齐常有无标签或弱标签需求。
- 回答什么问题：ROCA 能否作为 session alignment 的 determinant branch selector？
- 和当前主线关系：把“神经—运动”对齐扩展到“神经—神经”对齐。
- 什么时候做：当 ROCA 在当前任务上的方法故事成型后。
- 第一版干净实验：同一动物不同 session 的低维 neural manifold，用 held-out behavior 只做最终评价。
- 风险：session drift 可能不是纯正交/reflection 问题，ROCA 可能只解释一部分误差。

## 8. unbalanced OT / partial OT

- 为什么值得做：真实神经和运动样本可能存在不完全对应、噪声样本或分布质量不均。
- 回答什么问题：允许质量不守恒是否改善 group transport 与 representatives？
- 和当前主线关系：属于 HiWA/OT 层面的鲁棒性增强。
- 什么时候做：只有在发现失败主要来自 outlier 或 mass mismatch 时。
- 第一版干净实验：synthetic 中加入 outlier clusters / missing clusters，对比 balanced 与 unbalanced selector 的 determinant recovery。
- 风险：会引入新的正则化参数，容易变成调参故事。

## 9. 原型与 HiWA 联合优化

- 为什么值得做：当前 prototypes 与 HiWA 多是分阶段处理；联合优化可能让 representatives 更适合对齐。
- 回答什么问题：prototype learning 是否可以被 alignment objective 反向修正？
- 和当前主线关系：这是 Soft-Prototype HiWA 的更强版本。
- 什么时候做：必须等当前分阶段方法清楚后，否则无法判断改进来自哪里。
- 第一版干净实验：只在 synthetic 上做 joint objective，比较 fixed prototype 与 joint prototype 的 determinant recovery 和 assignment entropy。
- 风险：容易引入标签泄漏或复杂神经训练，偏离当前无标签、可解释主线。

## 10. ROCA selector 作为 HiWA 通用初始化或 model selection 组件

- 为什么值得做：如果 ROCA 只是选择 determinant branch，它可能不局限于 Soft-HiWA，也能用于普通 HiWA、Wasserstein Procrustes 或点云 registration。
- 回答什么问题：ROCA 是任务特定技巧，还是通用 orientation selector？
- 和当前主线关系：决定贡献范围。
- 什么时候做：文献检索确认没有同类方法，并完成 synthetic + 真实数据核心验证后。
- 第一版干净实验：在纯 Wasserstein Procrustes 合成任务中接入 ROCA representatives，不使用神经数据。
- 风险：贡献边界可能扩大过快，导致论文主线失焦。

## 11. 退化检测 refinement

- 为什么值得做：合成 near_degenerate 在标准化后未触发 hard degeneracy flag，但出现低 margin 和 learned_soft 失败。
- 回答什么问题：selector 什么时候应该拒绝选择，而不是强行输出 determinant？
- 和当前主线关系：这是 ROCA 可靠性边界的核心。
- 什么时候做：下一步优先做。
- 第一版干净实验：记录 raw volume、standardized volume、representative pairwise distance、simplex condition number、assignment entropy、Hungarian margin，并用合成失败案例校准 warning 规则。
- 风险：如果用测试结果调阈值，会形成隐性标签选择；阈值必须在 synthetic 或 validation protocol 中确定。

## 12. 结论性建议

下一步最值得做的是退化检测 refinement 和系统文献检索，而不是立刻开大规模新模型。ROCA 当前的研究价值在于“发现并无标签解决 determinant branch ambiguity”，后续所有探索都应服务于这个机制是否真实、稳健、可推广。

## 13. 本轮新增信息后的优先级更新

### 13.1 confidence-aware ROCA 应成为近期优先级

- 为什么值得做：warning_v0 抓住 8/8 synthetic failure，但误报 25 个成功 case；split 版 confidence calibration 进一步显示 `condition_number > 80` candidate rule 在 validation split 抓住 8/8 failure 且误报 0。
- 回答什么问题：ROCA 什么时候应该低置信，而不是强行解释分支选择？
- 和当前主线关系：这是把 ROCA 从“有效 selector”推进成“有边界的方法”的关键。
- 什么时候做最合适：现在就应继续做，但只做 calibration / validation，不改变主 selector。
- 第一版干净实验：更多 synthetic degeneracy 形态；固定阈值后只报告真实 warning rate。
- 风险：如果用真实 accuracy 或 $R^2$ 调 warning 阈值，会形成标签泄漏。

### 13.2 K=5 / 自动 K 暂缓

- 为什么值得做：真实结构未必正好 $K=4$。
- 回答什么问题：ROCA 是否依赖 $K=d+1$？
- 和当前主线关系：属于泛化，不是当前主结果。
- 什么时候做最合适：confidence-aware ROCA 稳定后。
- 第一版干净实验：先 synthetic $d=3,K=5$，用 simplex voting，不碰真实主实验。
- 风险：现在做会把 K、matching、confidence、accuracy 混在一起。

### 13.3 clustering ablation 暂缓

- 为什么值得做：learned_soft near_degenerate failure 说明代表点质量是边界。
- 回答什么问题：ROCA 需要哪类 representatives？
- 和当前主线关系：用于拆分 soft prototype 与 orientation rule 的贡献。
- 什么时候做最合适：literature 和当前 method story 稳定后。
- 第一版干净实验：同一 synthetic 数据比较 k-means、GMM posterior、当前 learned_soft。
- 风险：容易变成聚类横评，偏离 determinant branch 机制。

### 13.4 外部数据验证升为下一阶段重要目标

- 为什么值得做：当前真实证据只来自一个 macaque neural-movement 数据集。
- 回答什么问题：ROCA 是否是数据集特例？
- 和当前主线关系：决定论文说服力。
- 什么时候做最合适：组会确认主线后。
- 第一版干净实验：找一个 session-to-session 或 neural-behavior alignment 场景，固定 ROCA selector 和 warning rule，只报告 branch stability 与 warning rate。
- 风险：外部数据预处理差异大，不能把失败简单解释为方法无效。

### 13.5 $d$ 维一般化与 $K>d+1$ simplex voting

- 为什么值得做：当前 $d=3,K=4$ 是四面体特例。
- 回答什么问题：ROCA 是三维巧合还是一般 orientation selector？
- 和当前主线关系：理论推广。
- 什么时候做最合适：外部验证或更多 synthetic 后。
- 第一版干净实验：synthetic $d=2,3,5$；$K=d+1$ 先验证，再做 $K>d+1$ voting。
- 风险：高维 simplex volume 数值不稳，可能需要 log-volume 或 SVD-based orientation confidence。

### 13.6 learned_soft near_degenerate failure 的解释价值

- 为什么值得做：失败案例说明代表点质量是 ROCA 的方法边界。
- 回答什么问题：ROCA failure 是随机噪声，还是代表点退化？
- 和当前主线关系：增强可解释性和诚实边界。
- 什么时候做最合适：写论文方法边界和实验讨论时。
- 第一版干净实验：固定 seed 42，系统改变 cluster separation / degeneracy / assignment entropy，只记录 warning，不追 accuracy。
- 风险：不能围绕 seed 42 过拟合最终规则。
