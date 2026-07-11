# Graph-informed CC-HiWA方法升级计划-2026-07-10

## 1. 为什么需要升级？

PBMC 300-cell 和 1k top-feature smoke 都给出了同一个信号：

- 数据本身是可对齐的；
- 普通 Orthogonal Procrustes 明显强于随机；
- 未旋转 CC-HiWA 简化版没有超过 Procrustes；
- Procrustes warm-start CC-HiWA 接近 Procrustes baseline，但 paired-cell retrieval 仍未超过；
- 只调 $\beta$ 会改变 transport entropy，但不能带来更正确的匹配。

这说明问题不在于“PBMC 不适合做对齐”，而在于当前版本还没有真正吸收 TACO / GLUE / 单细胞多组学对齐中的结构思想。

当前 CC-HiWA 用到的是：

$$
X=\text{RNA embedding}, \quad Y=\text{ATAC embedding}
$$

以及各自独立 KMeans 得到的 soft assignment：

$$
A^X \in \mathbb{R}^{n\times K}, \quad A^Y \in \mathbb{R}^{m\times K}.
$$

但是这两个 assignment 没有被跨模态 feature graph 约束，因此“RNA 第 3 簇”和“ATAC 第 3 簇”不一定有共同语义。

2026-07-10 进一步修正：PBMC runner 第一版也没有把 Procrustes rotation 传入 `fit_cc_hiwa`，导致 CC-HiWA 在未对齐空间中运行。加入 `--rotation-mode procrustes` 后，CC-HiWA 结果明显提高，接近 Procrustes baseline，但 component-conditioned term 仍没有带来 paired-cell retrieval 增益。

## 2. 这次升级要吸收 TACO/GLUE 的什么？

我们不能只说“用了 soft clustering”。更核心的是：

1. soft prototype：样本不是硬属于某一簇，而是对多个 component 有概率归属；
2. group-conditioned OT：transport cost 不只看点到点，还要看 component 到 component 的兼容性；
3. feature / structure prior：跨模态之间不是凭空对齐，而是由某种已知结构约束；
4. representative / prototype geometry：代表点不是只用于展示，而是参与稳定化、初始化或 cost shaping；
5. no-label alignment：评价标签不能进入训练或模型选择。

对 PBMC 来说，最自然的结构先验是：

> peak-gene graph / gene activity。

也就是用 ATAC peak 与 gene 的基因组邻近关系，把 ATAC 映射到 gene-level activity，再与 RNA gene expression 建立共享 feature space。

## 3. 拟定方法名

建议暂名：

> Graph-informed Component-Conditioned HiWA，简称 GI-CC-HiWA。

它可以作为大论文骨干中的一个通用模块：

> structure-informed hierarchical optimal transport alignment with soft prototypes。

ROCA 则是其中针对低维正交分支的特殊稳定化模块：

> representative-oriented branch selection for low-dimensional orthogonal alignment。

## 4. PBMC 版本数学设计

### 4.1 构造 peak-gene graph

RNA genes：

$$
g_j=(c_j, s_j, e_j)
$$

其中 $c_j$ 是 chromosome，$s_j,e_j$ 是 gene start/end。

ATAC peaks：

$$
p_l=(c_l, u_l, v_l)
$$

如果 peak $p_l$ 与 gene $g_j$ 在同一 chromosome，且距离小于窗口 $w$，则连边：

$$
G_{lj}=1.
$$

一种简单距离定义：

$$
d(p_l,g_j)=
\begin{cases}
0, & [u_l,v_l]\cap [s_j,e_j]\neq \emptyset,\\
\min(|u_l-e_j|,|v_l-s_j|), & \text{otherwise}.
\end{cases}
$$

连边规则：

$$
G_{lj}= \mathbf{1}\{c_l=c_j,\ d(p_l,g_j)\leq w\}.
$$

第一版可以取 $w=50\mathrm{kb}$ 或 $100\mathrm{kb}$，后续做敏感性分析。

### 4.2 ATAC gene activity

令 ATAC peak matrix 为：

$$
Y^{peak}\in \mathbb{R}^{n\times P}.
$$

peak-gene adjacency：

$$
G\in \{0,1\}^{P\times G}.
$$

则 gene activity 可以写成：

$$
Y^{gene}=Y^{peak}G.
$$

为了避免高连接 gene 被放大，需要做归一化：

$$
\tilde{G}_{lj}=
\frac{G_{lj}}{\sum_{l'}G_{l'j}+\epsilon}.
$$

于是：

$$
Y^{gene}=Y^{peak}\tilde{G}.
$$

### 4.3 共享 feature space

RNA 侧选取同一批 genes：

$$
X^{gene}\in \mathbb{R}^{n\times G}.
$$

ATAC 侧得到：

$$
Y^{gene}\in \mathbb{R}^{n\times G}.
$$

然后对二者做相同预处理：

$$
\log(1+x), \quad \text{library normalization}, \quad \text{SVD/PCA}.
$$

得到：

$$
Z^X\in \mathbb{R}^{n\times d}, \quad Z^Y\in \mathbb{R}^{n\times d}.
$$

### 4.4 graph-informed soft assignment

当前 soft assignment 是各模态独立 KMeans：

$$
A^X = \operatorname{softKMeans}(Z^X), \quad A^Y = \operatorname{softKMeans}(Z^Y).
$$

升级版有两种干净路线。

#### 路线 A：共享 gene activity space 后再聚类

把 $Z^X$ 与 $Z^Y$ 拼接后学习共享 prototypes：

$$
C=\operatorname{KMeans}([Z^X;Z^Y]).
$$

再分别计算：

$$
A^X_{ik}
=
\frac{\exp(-\|z_i^X-c_k\|^2/\tau)}
{\sum_{k'}\exp(-\|z_i^X-c_{k'}\|^2/\tau)}
$$

$$
A^Y_{ik}
=
\frac{\exp(-\|z_i^Y-c_k\|^2/\tau)}
{\sum_{k'}\exp(-\|z_i^Y-c_{k'}\|^2/\tau)}.
$$

优点：最简单，最容易复现。

风险：如果 $Z^X$ 与 $Z^Y$ 初始尺度仍有差异，共享 KMeans 会偏向一侧。

#### 路线 B：先 Procrustes warm-start，再共享 prototypes

先用 Procrustes 得到粗对齐：

$$
\hat{R}=\arg\min_{R^\top R=I}\|Z^XR-Z^Y\|_F^2.
$$

再在 $[Z^X\hat{R};Z^Y]$ 上学习 prototypes。

优点：更稳定。

风险：需要说明 Procrustes 是无标签 warm-start，不是测试集选择。

第一版必须使用路线 B，因为 1k smoke 显示 Procrustes warm-start 是 CC-HiWA 在 PBMC 上接近合理性能的必要步骤。它是无标签初始化，不使用 `cell_type` 或 paired-cell 评价结果做选择。

## 5. graph-informed CC-HiWA cost

当前 CC-HiWA 的思路可以抽象成：

$$
C_{ij}^{sample}
=
\|x_i-y_j\|^2
\beta\cdot C_{ij}^{component}.
$$

其中 component compatibility 来自 soft assignment：

$$
C_{ij}^{component}
=
a_i^\top M b_j.
$$

升级版中，$a_i,b_j$ 来自 graph-informed shared prototypes，并且 sample cost 在 Procrustes warm-start 后计算，因此 component 语义更可能跨模态一致。

进一步，还可以加入 gene activity reconstruction consistency：

$$
C_{ij}
=
\|z_i^X-z_j^Y\|^2
\beta a_i^\top M b_j
\lambda \|q_i^X-q_j^Y\|^2,
$$

其中 $q_i^X,q_j^Y$ 是 gene-level low-dimensional activity 表示。

第一版建议先不要加 $\lambda$，避免模型太复杂。先做：

> graph-informed representation + shared prototype + 原 CC-HiWA。

## 6. 干净实验设计

### 6.1 数据

PBMC 1k / 2k / 5k paired cells。

### 6.2 方法对照

| 方法 | 作用 |
|---|---|
| Random | 下界 |
| Raw SVD cosine | 不对齐 baseline |
| Orthogonal Procrustes | 强无标签线性 baseline |
| 当前 CC-HiWA | 无 graph、无 rotation 的 group-conditioned OT |
| Procrustes-warm-start CC-HiWA | 无 graph、但有正交 warm-start 的 group-conditioned OT |
| Gene-activity Procrustes | 检查 gene activity 是否有帮助 |
| GI-CC-HiWA | 我们升级方法 |

### 6.3 指标

| 指标 | 含义 |
|---|---|
| Recall@1 | 是否找回 paired cell |
| Recall@5 | paired cell 是否在前 5 |
| Recall@10 | paired cell 是否在前 10 |
| Cell-type transfer accuracy | 是否转移到同类细胞 |
| Transport entropy | coupling 是否过平滑 |
| Runtime / memory | 是否可扩展 |

### 6.4 标签使用边界

`cell_type` 只用于最终评价，不参与：

- feature graph 构造；
- prototype 学习；
- Procrustes；
- OT；
- $\beta$ 选择；
- 模型选择。

## 7. 如果 GI-CC-HiWA 仍不提升怎么办？

这也是有价值结果。它会说明：

1. PBMC 任务中 cell-level paired retrieval 主要由表示学习决定；
2. 后处理 OT 不足以替代强表示学习；
3. 我们的通用论文主线应从“OT 后处理”升级到“joint representation + OT refinement”；
4. HiWA/ROCA 更适合低维几何稳定化，而 PBMC 更适合展示 structure-informed representation。

这不是失败，而是帮助我们把论文定位得更诚实。

## 8. 第一版实现任务

下一步代码任务：

1. 新增 `scripts/prepare_pbmc_gene_activity.py`；
2. 从 RNA var 中读取 gene chromosome/start/end/name；
3. 从 ATAC var 中读取 peak chromosome/start/end；
4. 构造 peak-gene sparse adjacency；
5. 生成 ATAC gene activity；
6. 选择 shared genes；
7. 输出：
   - `rna_gene_activity_embedding`
   - `atac_gene_activity_embedding`
   - `shared_gene_names`
   - `peak_gene_edge_count`
   - `cell_ids`
   - `cell_type`
8. 新增 `run_gi_cc_hiwa_pbmc.py` 或扩展现有 runner；
9. 先跑 1k；
10. 写实验笔记。

## 9. 当前决定

基于 300-cell 和 1k smoke，当前阶段的决定是：

> 不继续在未旋转 CC-HiWA 上追 $\beta$；PBMC 协议中固定加入 Procrustes warm-start，然后转向 Graph-informed CC-HiWA。

这个决定是受实验支持的，不是凭感觉换方向。

## 10. 2026-07-10 第一版 GI 输入已实现

已新增：

```text
D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/scripts/prepare_pbmc_gene_activity.py
```

第一版结果：

| 方法 | Recall@1 | Recall@5 | Cell-type acc |
|---|---:|---:|---:|
| Gene-activity Procrustes | 0.021 | 0.084 | 0.704 |
| GI-CC-HiWA best | 0.020 | 0.073 | 0.668 |

解释：

- gene activity 明显增强 cell-type-level signal；
- Procrustes 在 gene-activity space 中仍是最强 paired retrieval baseline；
- GI-CC-HiWA 还没有超过 Procrustes；
- 说明下一步应检查 component-conditioned cost 的设计，而不是只改表示或只调 $\beta$。

更新后的第一版干净实验应加入：

1. top-feature Procrustes；
2. top-feature CC-HiWA + Procrustes warm-start；
3. gene-activity Procrustes；
4. gene-activity GI-CC-HiWA + Procrustes warm-start；
5. representative/prototype direct-cost ablation。
