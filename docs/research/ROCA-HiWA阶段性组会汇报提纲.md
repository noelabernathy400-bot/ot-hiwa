# ROCA-HiWA阶段性组会汇报提纲

适用时长：10–15 分钟

## Slide 1. 研究问题：HiWA 为什么方向语义不稳定？

- Main message：HiWA 连续几何对齐可以稳定，但 direction accuracy 对 seed / branch 敏感。
- Figure/table：总结果表中的 HiWA baseline 与 seed 7 diagnosis。
- Speaker notes：先说我们不是从一开始就有 ROCA，而是复现中发现了稳定性问题。
- Potential question：是不是只是 accuracy 指标噪声？
- Suggested answer：不是；seed 7 中 $R^2$ 高但 accuracy 低，说明连续几何和离散方向语义可分离。

## Slide 2. 初始尝试：TACO soft prototype 迁移

- Main message：我们借鉴 soft prototype / soft assignment / group-conditioned OT，希望改善自动分组。
- Figure/table：soft grouping structure 图。
- Speaker notes：强调 TACO 是思想来源，不是直接复刻。
- Potential question：为什么不用硬聚类？
- Suggested answer：硬聚类稳定但僵硬，soft assignments 可提供更细的 representative 结构。

## Slide 3. 负结果：soft assignment 没有直接提升

- Main message：soft random init 不稳定，warm start 稳定但初期不涨 accuracy。
- Figure/table：soft heldout comparison。
- Speaker notes：负结果很重要，说明不能把 soft 分组写成简单提分模块。
- Potential question：是不是 temperature 没调好？
- Suggested answer：做过 temperature sensitivity、annealing、pure annealing，未解决核心问题。

## Slide 4. 关键诊断：$R^2$ 和 accuracy 分离

- Main message：$R^2$ 高不代表 direction accuracy 高。
- Figure/table：seed 7 pure annealing vs hard baseline。
- Speaker notes：这是转向几何分支诊断的关键理由。
- Potential question：那我们优化目标是不是错了？
- Suggested answer：不是简单错，而是连续轨迹对齐与离散方向语义对应不同层面，需要 branch-level 解释。

## Slide 5. 新发现：$O(3)$ determinant branch

- Main message：HiWA / Soft-HiWA 实际可能搜索 $O(3)$，包含 $\det=+1$ 和 $\det=-1$ 两个分支。
- Figure/table：rotation component accuracy seeds 10-29。
- Speaker notes：解释 $SO(3)$ 和 $O(3)$。
- Potential question：为什么不能直接选高 accuracy 分支？
- Suggested answer：测试标签不能用于选择；只能最终评价。

## Slide 6. 方法：ROCA-HiWA

- Main message：用 soft representatives 构造有向四面体，无标签选择 determinant branch。
- Figure/table：方法流程图或公式。
- Speaker notes：给出 $r_k^X$、$r_l^Y$ 和 $\operatorname{sign}(\det X_{\mathrm{rep}}\det Y_{\mathrm{rep}})$。
- Potential question：signed volume 是新的吗？
- Suggested answer：不是；新点是把它和 soft representatives 结合，用于 HiWA determinant branch selection。

## Slide 7. 实验 1：真实数据 seeds 50-69

- Main message：ROCA 在独立 seeds 上 20/20 选择高 accuracy 分支。
- Figure/table：ROCA代表点定向确认图。
- Speaker notes：强调标签只用于最终评价。
- Potential question：是不是碰巧固定选择 det=-1？
- Suggested answer：下一页 coordinate flip 专门排除这个担心。

## Slide 8. 实验 2：坐标符号翻转

- Main message：源端单轴反射后，selector 30/30 随手性翻转。
- Figure/table：![[坐标符号翻转验证总图-2026-07-07.png]]
- Speaker notes：这是比原始 seeds 更强的机制检验。
- Potential question：翻转是否改变了距离结构？
- Suggested answer：单轴反射是正交变换，不改变欧氏距离，但改变手性。

## Slide 9. 实验 3：synthetic determinant truth

- Main message：在已知 true_det 的 synthetic 数据中，ROCA recovery 总体 0.995。
- Figure/table：synthetic accuracy by noise / assignment。
- Speaker notes：强调合成真值回答真实数据无法回答的 determinant ground truth。
- Potential question：失败在哪里？
- Suggested answer：8 个失败集中在 learned_soft + near_degenerate + seed 42。

## Slide 10. 实验 4：低置信诊断与 calibration

- Main message：失败有无标签预警信号，condition number > 80 candidate rule 在 validation 抓 8/8，真实 warning 0/50。
- Figure/table：![[ROCA置信规则召回精度曲线-split-2026-07-08.png]]
- Speaker notes：这不是为了提高 accuracy，而是给方法可靠性边界。
- Potential question：阈值 80 是否普适？
- Suggested answer：不是；它是当前 synthetic calibration candidate，需要更多数据验证。

## Slide 11. 当前结论

- Main message：ROCA 解决的是一个具体 instability，不是随意追分。
- Figure/table：总结果表。
- Speaker notes：三条证据：真实 seeds、坐标手性、synthetic truth。
- Potential question：能写成论文贡献吗？
- Suggested answer：可以作为机制贡献雏形，但还需要文献精读和外部验证。

## Slide 12. 局限

- Main message：当前只在一个真实数据集、$d=3,K=4$ 下验证。
- Figure/table：局限清单。
- Speaker notes：主动讲边界会更可信。
- Potential question：自动 K 呢？
- Suggested answer：应该等 confidence-aware ROCA 稳定后再做，否则变量太多。

## Slide 13. 下一步

- Main message：补 TACO/neural alignment 文献、外部数据、再做 K/d 推广。
- Figure/table：后续路线图。
- Speaker notes：把下一步从“追分”转为“验证边界和泛化”。
- Potential question：最优先做什么？
- Suggested answer：文献精读 + 外部/跨 session 最小验证。

