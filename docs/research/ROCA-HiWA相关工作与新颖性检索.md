# ROCA-HiWA 相关工作与新颖性检索

## 1. 检索目标

本检索服务于 ROCA-HiWA 当前阶段的定位：

> 是否已有方法用 soft prototypes / representatives 的有向几何结构，在无标签条件下解决 HiWA / Soft-HiWA / Wasserstein-Procrustes 类方法中的 determinant branch selection 问题？

本文件区分：

- 经典数学事实；
- 已有算法；
- 我们的方法组合；
- 我们在 HiWA / neural alignment 场景中的新应用。

检索日期：2026-07-07。

## 2. 已检索关键词

- Orthogonal Procrustes problem
- determinant constrained Procrustes
- Kabsch algorithm reflection correction
- reflection ambiguity in point set registration
- orientation-preserving registration
- chirality in point cloud alignment
- oriented simplex
- signed volume for orientation detection
- optimal transport with Procrustes alignment
- Wasserstein Procrustes
- hierarchical optimal transport alignment
- neural manifold alignment
- brain-machine interface alignment
- unsupervised neural decoding alignment
- TACO soft prototype group-conditioned OT

## 3. 初步结论

目前检索到的证据支持如下谨慎判断：

1. determinant-constrained Procrustes / Kabsch reflection correction 是经典数学与算法事实；
2. point-set registration 中 reflection / chirality / orientation-preserving registration 是相关主题；
3. Wasserstein Procrustes 类方法会联合估计 transport / permutation 和正交变换；
4. HiWA 使用 hierarchical OT 处理 clustered / multimodal alignment，并应用到 macaque neural decoding；
5. oriented simplex / signed volume 是经典有向几何工具；
6. 当前尚未检索到完全相同的方法：用 soft cluster representatives 构成有向四面体，基于 signed volume 在无标签条件下选择 HiWA / Soft-HiWA 的 $O(3)$ determinant branch。

这个结论仍是初步检索，不是最终查新。正式论文前必须精读全文并核验引用。

## 4. 文献与方法表

| 主题 | 文献/来源 | 年份 | 核心方法 | 是否处理 determinant branch | 是否使用标签 | 与 ROCA-HiWA 的关系 |
|---|---|---:|---|---|---|---|
| Orthogonal Procrustes | Schönemann, A generalized solution of the orthogonal Procrustes problem | 1966 | SVD 求解正交 Procrustes | 一般在 $O(d)$ 中求解 | 需要给定对应 | 经典基础；ROCA 使用其 determinant-constrained 版本 |
| Kabsch algorithm | Kabsch, A solution for the best rotation to relate two sets of vectors, Acta Cryst. | 1976 | 最小二乘最佳旋转 | 处理 proper rotation；后续常见 reflection correction | 需要给定对应 | determinant correction 的经典来源 |
| Kabsch discussion | Kabsch, A discussion of the solution for the best rotation to relate two sets of vectors | 1978 | proper rotation 求解讨论 | 明确 proper rotation | 需要给定对应 | 与我们 Procrustes 分支构造相关 |
| 3D point-set least squares | Arun, Huang, Blostein, Least-Squares Fitting of Two 3-D Point Sets | 1987 | SVD 求解刚体配准 | 关注旋转矩阵 | 需要给定对应 | point registration 基础 |
| Procrustes 工具文档 | MATLAB Procrustes | 持续维护 | 支持 reflection true / false / best | 明确可强制或禁止 reflection | 需要输入点集 | 说明 reflection choice 是标准 Procrustes 参数 |
| Point-set registration | Besl and McKay, ICP | 1992 | 迭代最近点配准 | 通常关注刚体旋转和平移 | 最近邻对应迭代 | 与无监督配准相关，但不是 HiWA 分支选择 |
| Wasserstein Procrustes | Grave, Joulin, Berthet, Unsupervised Alignment of Embeddings with Wasserstein Procrustes | 2019 | 联合估计 orthogonal matrix 和 permutation / transport | 主要优化 orthogonal alignment，未见 signed-volume branch selector | 无监督词向量对齐 | 与 OT + Procrustes 相关 |
| Two-sided Wasserstein Procrustes | Jin, Liu, Xia, Two-sided Wasserstein Procrustes Analysis | 2021 | 双侧变换 + OT coupling | 讨论全局变换，不是 soft representative signed volume | 无监督 | 说明 OT + Procrustes 是成熟方向 |
| Procrustes-Wasserstein distance | Procrustes-Wasserstein 相关论文 | 2025 | 在分布距离中对 orthogonal group 优化 | 允许 rotations/reflections | 无监督 | 与 $O(d)$ 优化相关，但不是 ROCA selector |
| HiWA | Lee, Dabagia, Dyer, Rozell, Hierarchical Optimal Transport for Multimodal Distribution Alignment | 2019 | hierarchical OT alignment, clustered structure | 使用 unitary / orthogonal 类变换；未见 signed volume branch selector | 神经任务评价用标签 | ROCA 直接基于 HiWA 问题 |
| HiWA 官方代码 | siplab-gt/hiwa-matlab | 2019 | 官方 MATLAB 代码 | 未提供 ROCA selector | 无标签 alignment，标签评价 | 本项目复现基础 |
| Oriented simplex | algebraic topology / simplicial complex 教材 | 经典 | simplex orientation 由顶点顺序决定 | 不是 Procrustes 算法 | 不涉及标签 | ROCA 的 signed tetrahedron 工具来源 |
| Signed volume / chirality | Chirality / signed volume 相关几何资料 | 经典 | 用 orientation / signed measure 区分手性 | 可检测 reflection | 不涉及标签 | ROCA 将其用于 soft representatives |
| TACO / soft prototype | 用户提供的 TACO 思想与待核验文献 | 待核验 | soft prototype / soft assignment / group-conditioned OT | 未确认处理 determinant branch | 待核验 | ROCA 的代表点思想来源，但不是简单复刻 |

## 5. 关键问题回答

### 5.1 是否已有方法用 oriented simplex / signed volume 解决 Procrustes reflection ambiguity？

初步判断：经典几何中 signed volume / oriented simplex 本来就可以判断手性；Procrustes / Kabsch 中也有 determinant correction 或 reflection option。因此“用有向体积判断反射”不是全新数学事实。

但目前未检索到与 ROCA 完全相同的组合：

> 由 soft assignments 形成跨域 representatives，再用 matched representative tetrahedron 的 signed volume product 作为 HiWA / Soft-HiWA determinant branch selector。

### 5.2 是否已有方法把 signed volume branch selection 用到 hierarchical OT 或 HiWA？

初步未发现。

HiWA 原文强调 clustered / multimodal / hierarchical OT alignment，并在 macaque neural decoding 上验证，但检索到的摘要与官方代码说明没有显示 signed-volume determinant branch selection。

### 5.3 是否已有方法把 soft prototypes 用于 determinant branch selection？

初步未发现。

soft prototypes 常用于 representation / prototype learning / group-level alignment，但把它们用于 $O(3)$ determinant branch selection 目前没有检索到完全同型工作。

### 5.4 如果已有类似思想，ROCA 应如何重新定位？

如果后续精读发现已有 signed-volume branch correction，那么 ROCA 的定位应降级为：

> 将 signed-volume / chirality correction 与 TACO-style soft representatives 结合，并迁移到 HiWA neural alignment 的无标签正交分支选择问题。

也就是说，贡献从“新算法思想”变为“面向 HiWA/neural alignment 的组合式机制和实证验证”。

### 5.5 如果没有高度相同方法，贡献可如何表述？

谨慎贡献可以写成：

1. 诊断 HiWA / Soft-HiWA 在 macaque neural--movement alignment 中存在 $O(3)$ determinant branch instability；
2. 提出 ROCA-HiWA：用 soft cluster representatives 的有向四面体体积，在无标签条件下选择 determinant branch；
3. 在真实数据、坐标手性翻转和合成 determinant 真值实验中验证该选择器的可解释性和稳定性。

## 6. 与经典方法的边界

ROCA-HiWA 不声称发明了：

- Orthogonal Procrustes；
- determinant-constrained SVD；
- Kabsch reflection correction；
- signed volume；
- oriented simplex；
- optimal transport alignment；
- HiWA。

ROCA-HiWA 的潜在新意在组合：

$$
\text{soft representatives}
\quad+\quad
\text{group transport matching}
\quad+\quad
\text{oriented tetrahedron}
\quad+\quad
\text{HiWA determinant branch selection}.
$$

## 7. 当前引用线索

以下来源已经在线核验到题名或官方页面：

- Kabsch, W. 1976. A solution for the best rotation to relate two sets of vectors. Acta Crystallographica. DOI 页面：https://journals.iucr.org/paper?a12999=
- Kabsch, W. 1978. A discussion of the solution for the best rotation to relate two sets of vectors. Acta Crystallographica. https://journals.iucr.org/paper?s0567739478001680=
- Arun, K. S., Huang, T. S., Blostein, S. D. 1987. Least-Squares Fitting of Two 3-D Point Sets. IEEE TPAMI. IEEE 页面：https://ieeexplore.ieee.org/document/4767965
- Grave, E., Joulin, A., Berthet, Q. 2019. Unsupervised Alignment of Embeddings with Wasserstein Procrustes. PMLR PDF：https://proceedings.mlr.press/v89/grave19a/grave19a.pdf
- Jin, K., Liu, C., Xia, C. 2021. Two-Sided Wasserstein Procrustes Analysis. IJCAI PDF：https://www.ijcai.org/proceedings/2021/0484.pdf
- Lee, J., Dabagia, M., Dyer, E., Rozell, C. 2019. Hierarchical Optimal Transport for Multimodal Distribution Alignment. arXiv：https://arxiv.org/abs/1906.11768
- HiWA MATLAB official repository：https://github.com/siplab-gt/hiwa-matlab
- MATLAB Procrustes reflection option：https://www.mathworks.com/help/stats/procrustes.html

## 8. 待继续核验

后续需要继续：

1. 精读 HiWA 原文和 supplement，确认其 unitary / orthogonal 更新是否讨论 determinant；
2. 精读 Wasserstein Procrustes 相关论文，确认是否有 determinant branch 处理；
3. 检索 computational geometry 中 signed volume / chirality correction 是否已有同型算法；
4. 找到用户所说 TACO 方法的准确论文题名、作者和链接；
5. 检索 neural manifold alignment / BMI session alignment 中是否已有 reflection ambiguity 处理。

## 9. 推荐检索式

- `"signed volume" "Procrustes" reflection`
- `"oriented simplex" "point set registration"`
- `"chirality" "point cloud registration" determinant`
- `"Wasserstein Procrustes" determinant`
- `"hierarchical optimal transport" determinant`
- `"neural manifold alignment" Procrustes reflection`
- `"soft prototype" "optimal transport" alignment`
- `"group-conditioned optimal transport" prototype`

## 10. 当前新颖性风险等级

当前风险：中等。

理由：

- 基础数学和 Procrustes determinant correction 都是经典；
- OT + Procrustes 也是已有方向；
- HiWA 是已有方法；
- 但用 soft representatives 的有向体积做 HiWA branch selector，目前没有检索到完全同型方法。

因此当前不应说“完全原创”，应说：

> 初步检索未发现完全同型方法；ROCA-HiWA 的潜在贡献在于把 soft prototype representatives 的有向几何结构用于 HiWA / Soft-HiWA 的无标签 determinant branch selection。
