# ROCA-HiWA系统文献检索与新颖性定位

日期：2026-07-08

## 1. 检索目标

本笔记用于定位 [[ROCA-HiWA]] 的新颖性边界。重点问题不是“基础数学是否新”，而是：

- determinant-constrained Procrustes / Kabsch reflection correction 是否已有；
- signed volume / oriented simplex 判断 orientation 是否是经典事实；
- point set registration 中 reflection ambiguity 和 degeneracy warning 是否已有；
- Wasserstein Procrustes / HiWA / neural alignment 是否已经处理过类似 determinant branch；
- TACO / soft prototype 是否把 soft representatives 用于 determinant branch selection；
- 是否已有“soft representatives + signed volume + HiWA/neural alignment”的完全同型组合。

## 2. 总结性结论

当前检索支持三层定位：

### A. 已知经典数学事实

- $O(d)$ 与 $SO(d)$ 的 determinant branch 区分是经典事实；
- signed volume / oriented simplex 可用于判断 orientation / handedness，是经典几何；
- tetrahedral signed volume 在 chirality 判断中也有成熟使用。

### B. 已有算法或常规技巧

- Orthogonal Procrustes、Kabsch、Umeyama、Arun 等已有 SVD / determinant correction 路线；
- Procrustes 工具通常允许或禁止 reflection；
- point cloud / ICP registration 文献已有 degeneracy-aware registration，常用 Hessian eigenvalue、condition number、TSVD、regularization 或 constraints 标记不稳定方向；
- Wasserstein Procrustes 已将 OT 与 Procrustes 对齐结合。

### C. ROCA-HiWA 的可能新组合 / 新应用

目前未发现完全同型方法：

> soft representatives + oriented simplex / signed volume + HiWA / hierarchical OT / neural-movement alignment determinant branch selection。

谨慎贡献表述：

> signed volume 与 determinant-constrained Procrustes 都是已有数学和算法工具；ROCA-HiWA 的潜在贡献在于把 TACO 式 soft representatives 作为无标签跨域有向几何锚点，用于解决 HiWA / Soft-HiWA 在 neural-movement alignment 中暴露出的 $O(3)$ determinant branch instability。

## 3. 文献表

| 主题 | Title | Authors | Year | Venue | Link / DOI | 核心方法 | determinant branch | signed volume / chirality | OT | neural alignment | soft prototypes | 与 ROCA 相似度 | 可放段落 |
|---|---|---:|---:|---|---|---|---|---|---|---|---|---|---|
| Orthogonal Procrustes | Least-Squares Fitting of Two 3-D Point Sets | Arun, Huang, Blostein | 1987 | IEEE TPAMI | https://experts.illinois.edu/en/publications/least-squares-fitting-of-two-3-d-point-sets/ | SVD 解 3D 点集刚体拟合 | 涉及 reflection/determinant 问题 | 否 | 否 | 否 | 否 | 基础算法，不同问题 | Procrustes |
| Similarity Procrustes | Least-Squares Estimation of Transformation Parameters Between Two Point Patterns | Umeyama | 1991 | IEEE TPAMI | DOI 待核验：`10.1109/34.88573`; PDF: https://graphics.stanford.edu/courses/cs164-09-spring/Handouts/paper_Umeyama.pdf | SVD 估计 similarity transform | 有 determinant correction 背景 | 否 | 否 | 否 | 否 | 基础算法 | Procrustes |
| Kabsch | A Purely Algebraic Justification of the Kabsch-Umeyama Algorithm | Lawrence, Bernal, Witzgall | 2019 | NIST JRES | https://www.nist.gov/publications/purely-algebraic-justification-kabsch-umeyama-algorithm | constrained orthogonal Procrustes 的代数证明 | 是，orientation-preserving | 否 | 否 | 否 | 否 | 支持数学基础 | Procrustes |
| Procrustes software | MATLAB Procrustes documentation | MathWorks | 2026访问 | 官方文档 | https://www.mathworks.com/help/stats/procrustes.html | 支持/禁止 reflection | 是，文档明确 reflection | 否 | 否 | 否 | 否 | 说明 branch 是常规选项 | Procrustes |
| Wasserstein Procrustes | Unsupervised Alignment of Embeddings with Wasserstein Procrustes | Grave et al. | 2019 | AISTATS | https://proceedings.mlr.press/v89/grave19a/grave19a.pdf | OT + Procrustes 对齐 embedding | 未作为 neural determinant branch 讨论 | 否 | 是 | 否 | 否 | 有 OT+Procrustes，但非 ROCA | OT alignment |
| Two-sided WP | Two-Sided Wasserstein Procrustes Analysis | Chowdhury et al. | 2021 | IJCAI | https://www.ijcai.org/proceedings/2021/0484.pdf | 两侧 Wasserstein-Procrustes | 待精读 | 否 | 是 | 否 | 否 | 相近 OT+Procrustes 背景 | OT alignment |
| HiWA | Hierarchical Optimal Transport for Multimodal Distribution Alignment | Lee et al. | 2019 | NeurIPS | https://arxiv.org/abs/1906.11768 | hierarchical OT 做多模态分布对齐 | 当前未见 det(R) 分支讨论 | 否 | 是 | 可用于 neural | 否 | 直接基础方法 | HiWA |
| HiWA code | hiwa-matlab | siplab-gt | 2019 | GitHub | https://github.com/siplab-gt/hiwa-matlab | 官方实现 | 未显式讨论 ROCA 问题 | 否 | 是 | 可能 | 否 | 复现基础 | HiWA |
| Degeneracy registration | Informed, Constrained, Aligned: Degeneracy-aware Point Cloud Registration in the Wild | Nubert et al. | 2024/2025 | arXiv | https://arxiv.org/html/2408.11809v2 | TSVD/constraints/regularization 处理退化 | 非核心 | 否 | 否 | 否 | 否 | condition warning 思想相近 | Degeneracy |
| Degeneracy ICP | Degeneracy-aware factors with applications to underwater SLAM | Hinduja et al. | 2019 | IROS | https://www.cs.cmu.edu/~kaess/pub/Hinduja19iros.pdf | ICP degeneracy 与 factor graph | 非核心 | 否 | 否 | 否 | 否 | 支持低置信诊断 | Degeneracy |
| NDT condition | Smoothed NDT point cloud registration | 待核验 | 待核验 | PDF | https://elib.dlr.de/196292/1/smoothed_ndt.pdf | covariance condition number regularization | 非核心 | 否 | 否 | 否 | 否 | condition number 使用先例 | Degeneracy |
| Chirality | REFMAC chiral centres | CCP4 | 访问2026 | 官方文档 | https://www.ccp4.ac.uk/html/refmac5/theory/chiral.html | tetrahedral chiral volume sign | reflection changes sign | 是 | 否 | 否 | 否 | signed volume 背景 | Orientation |
| Oriented simplex | Simplicial Complexes introduction | Erleben | 待核验 | teaching PDF | https://ncatlab.org/nlab/files/Erleben_SimplicialComplexes.pdf | simplex orientation by vertex order | 是，orientation 相关 | 是 | 否 | 否 | 否 | 几何基础 | Orientation |
| TACO | TACO soft prototype / group-conditioned OT | 待核验 | 待核验 | 待核验 | 待补准确论文 | soft prototype / soft assignment / group-conditioned OT 思想 | 未确认 | 未确认 | 是 | 未确认 | 是 | 需要精读 | Soft prototype |

## 4. 必答问题

### 4.1 signed volume / oriented simplex 用于判断 orientation 是不是经典事实？

是。oriented simplex 的符号由 determinant 和顶点顺序决定；tetrahedral signed volume 在 chirality 判断中也有成熟用法。ROCA 不能把 signed volume 本身作为新数学。

### 4.2 Kabsch / Procrustes 中 determinant correction 是否已有？

是。Kabsch-Umeyama、Umeyama、Arun 等都属于经典 SVD/Procrustes 对齐路线。MATLAB Procrustes 文档也明确支持或禁止 reflection。

### 4.3 point cloud registration 中 reflection ambiguity 是否已有大量研究？

reflection / orientation-preserving registration 已有大量背景；degeneracy-aware registration 也已有较多工作。当前需继续精读“reflection ambiguity”专门文献。

### 4.4 Wasserstein Procrustes 是否已经处理过类似分支问题？

Wasserstein Procrustes 已经处理 OT + Procrustes 对齐，但目前未发现它将 soft representatives 的 oriented simplex 用于无标签 determinant branch selection。需要继续精读 Grave et al. 与 Two-sided Wasserstein Procrustes。

### 4.5 HiWA 是否讨论过 det(R)=+1 / -1 的分支问题？

当前检索和项目代码阅读中尚未看到 HiWA 明确把 determinant branch instability 作为问题讨论。ROCA 的动机来自复现实验中的实际诊断。

### 4.6 TACO 是否把 soft representatives 用于 determinant branch selection？

尚未确认。当前只把 TACO 作为 soft prototype / soft assignment / group-conditioned OT 思想来源。必须补 TACO 原文准确题名、作者、年份和全文阅读后才能下最终结论。

### 4.7 是否已有 soft representatives + signed volume + HiWA / neural alignment 组合？

当前未发现完全同型方法。仍需系统检索 neural manifold alignment、BMI alignment、OT alignment 中是否有相近组合。

## 5. 当前 novelty 写法

推荐谨慎表述：

> ROCA-HiWA does not introduce determinant-constrained Procrustes or signed volume as new mathematical tools. Its potential contribution is to use TACO-style soft representatives as label-free cross-domain oriented geometric anchors for resolving the $O(3)$ determinant branch instability observed in HiWA/Soft-HiWA neural-movement alignment.

中文：

> ROCA-HiWA 的新意不是 determinant、signed volume 或 Procrustes 本身，而是把 soft representatives 用作跨域有向几何锚点，在不使用方向标签、测试 accuracy 或测试 $R^2$ 的情况下选择 HiWA / Soft-HiWA 的 determinant branch。

## 6. 不能写成最终 novelty claim 的原因

- TACO 原文尚未完整核验；
- neural manifold alignment 与 BMI alignment 文献仍需系统补；
- Wasserstein Procrustes 是否隐含类似分支处理需要精读；
- 当前真实验证仍只有一个 macaque neural-movement 数据集；
- confidence warning 阈值来自当前 synthetic grid，不能普适化。

