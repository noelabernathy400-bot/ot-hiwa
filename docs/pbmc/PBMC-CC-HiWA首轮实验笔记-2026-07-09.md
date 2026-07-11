# PBMC-CC-HiWA首轮实验笔记-2026-07-09

## 1. 这次为什么换数据？

我们前面在猕猴神经元—运动数据上发现了一个很有意思但偏局部的问题：HiWA / Soft-HiWA 的 $O(3)$ 正交分支选择不稳定，ROCA 可以用代表点有向几何来无标签选择更稳定的分支。

但是如果我们的目标是写一篇更一般的方法论文，就不能只停留在“一个神经数据集上的三维方向解码修补”。所以这次单独开了一个 PBMC RNA-ATAC 子项目，用一个更典型的跨模态对齐任务检查：

- CC-HiWA / ROCA 思路能不能迁移到非神经领域；
- TACO 式 soft prototype、group-conditioned OT 是否真的能帮助跨模态对齐；
- 当前方法的短板到底是“数据问题”，还是“方法还没把结构信息用充分”。

本笔记是 smoke experiment，不是最终论文结果。

## 2. 数据是什么？

数据使用公开的 10x Multiome PBMC 10k 预处理 h5ad 文件：

- RNA：`10x-Multiome-Pbmc10k-RNA.h5ad`
- ATAC：`10x-Multiome-Pbmc10k-ATAC.h5ad`

来源记录：

- scGLUE example datasets 页面列出 `10x-Multiome-Pbmc10k`，协议为 10x Multiome，物种为 human，器官/样本为 PBMC，并提供 RNA / ATAC h5ad 文件名。
- 10x Genomics 页面说明该数据来自 human PBMC multiome 数据。

本地原始数据位置：

- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/data/raw/10x-Multiome-Pbmc10k-RNA.h5ad`
- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/data/raw/10x-Multiome-Pbmc10k-ATAC.h5ad`

读取检查：

| 模态 | shape | 说明 |
|---|---:|---|
| RNA | $9631 \times 29095$ | 9631 个细胞，29095 个基因/特征 |
| ATAC | $9631 \times 107194$ | 同 9631 个细胞，107194 个 ATAC peak/特征 |
| 标签 | `cell_type` | 只用于最终解释性评价，不进入训练或选择 |

这类数据天然适合做对齐，因为 RNA 和 ATAC 来自同一批细胞，理论上第 $i$ 个 RNA 细胞应匹配第 $i$ 个 ATAC 细胞。

## 3. 实验协议

为了先验证代码链条和任务可行性，本轮只抽样 300 个 paired cells：

$$
n = 300.
$$

预处理步骤：

1. RNA 与 ATAC h5ad 使用 backed mode 读取，避免一次性加载完整矩阵；
2. 取共同 cell barcode；
3. 随机种子 `seed=0` 抽样 300 个细胞；
4. 对 RNA / ATAC 稀疏矩阵做 $\log(1+x)$ 与 L1 normalization；
5. 分别用 TruncatedSVD 得到 20 维 embedding；
6. 对 embedding 做标准化；
7. 用 KMeans 得到 $K=8$ 个 soft assignments；
8. 运行 CC-HiWA，扫描：

$$
\beta \in \{0,0.01,0.05,0.1,0.25,0.5\}.
$$

评价指标：

### Recall@k

对每个 RNA 细胞，模型会给出它和所有 ATAC 细胞的匹配分数。如果真实对应的 ATAC 细胞排在前 $k$ 个里，就算命中。

$$
\mathrm{Recall@k}
= \frac{1}{n}\sum_{i=1}^{n}
\mathbf{1}\{i \in \operatorname{TopK}(s_{i,:}, k)\}.
$$

在 $n=300$ 时，随机期望为：

$$
\mathrm{Recall@1}_{random}=\frac{1}{300}=0.0033,
$$

$$
\mathrm{Recall@5}_{random}=\frac{5}{300}=0.0167.
$$

### Cell-type transfer accuracy

这个指标不看是否匹配到同一个细胞，而看 RNA 细胞通过最近 ATAC 邻居转移来的 `cell_type` 是否相同。

它比 Recall@1 宽松，因为同一 cell type 内有很多细胞，匹配到同类但不是同一个细胞也可能算对。

标签只用于最终评价，没有进入 CC-HiWA 的训练、分支选择或 transport 计算。

## 4. 首轮结果

图像已经复制进本笔记文件夹，应该可以在 Obsidian 中直接显示：

![PBMC CC-HiWA smoke summary](_attachments/cc_hiwa_pbmc_summary_smoke_300_v3_2026_07_09.png)

如果上面因为空格无法显示，使用这个备用链接：

![[_attachments/cc_hiwa_pbmc_summary_smoke_300_v3_2026_07_09.png]]

主结果表：

| 方法 | Recall@1 | Recall@5 | Recall@10 | Cell-type transfer acc | 说明 |
|---|---:|---:|---:|---:|---|
| Random expectation | 0.0033 | 0.0167 | 0.0333 | - | 随机猜测期望 |
| Raw embedding cosine | 0.0167 | 0.0600 | 0.1067 | 0.3133 | 不做 OT，只用原始低维表示最近邻 |
| Orthogonal Procrustes cosine | 0.0833 | 0.2400 | 0.3867 | 0.5433 | 普通正交对齐后最近邻 |
| CC-HiWA $\beta=0$ | 0.0033 | 0.0800 | 0.1233 | 0.3600 | 比随机 R@5 高，但 R@1 随机 |
| CC-HiWA $\beta=0.01$ | 0.0033 | 0.0800 | 0.1233 | 0.3600 | 与 $\beta=0$ 基本一致 |
| CC-HiWA $\beta=0.05$ | 0.0033 | 0.0733 | 0.1233 | 0.3633 | R@5 略降 |
| CC-HiWA $\beta=0.1$ | 0.0033 | 0.0733 | 0.1233 | 0.3600 | R@5 略降 |
| CC-HiWA $\beta=0.25$ | 0.0033 | 0.0700 | 0.1233 | 0.3500 | R@5 略降 |
| CC-HiWA $\beta=0.5$ | 0.0033 | 0.0700 | 0.1233 | 0.3467 | R@5 略降 |

## 5. 怎么解释这些数字？

### 5.1 这个数据集是有信号的

Raw embedding cosine 的 Recall@1 = 0.0167，已经高于随机 0.0033。普通 Orthogonal Procrustes 后 Recall@1 = 0.0833，Recall@5 = 0.2400，说明 RNA 和 ATAC 的 20 维 embedding 之间确实存在可对齐结构。

这点很重要：如果普通 Procrustes 都完全失败，我们会怀疑数据预处理或任务定义；但它表现不错，说明 PBMC 是一个合理的新测试场。

### 5.2 当前 CC-HiWA 没有赢过简单 Procrustes

CC-HiWA 的 Recall@1 停在 0.0033，等于随机期望；Recall@5 大约 0.07 到 0.08，高于随机但低于 Procrustes 的 0.24。

这说明当前版本的 group-conditioned OT 没有把 paired-cell 层面的精细结构保留下来。它在较宽松的 Top-5/Top-10 上有一点信号，但不够精准。

### 5.3 增大 $\beta$ 没有带来提升

$\beta$ 越大，sample transport entropy 越低：

$$
H(P)=-\sum_{ij}P_{ij}\log P_{ij}.
$$

这表示 transport coupling 变得更集中。但更集中不等于更正确：Recall@5 反而从 0.0800 降到 0.0700。

所以这里不是“继续调大 group-conditioned 项就能解决”的情况。

### 5.4 这不是坏消息，反而指出了下一步

普通 Procrustes 强，CC-HiWA 弱，说明目前主要问题可能不是“跨模态完全不能对齐”，而是：

1. soft prototype / group-conditioned cost 还没有和表示学习联合起来；
2. 当前 soft assignment 是在两个模态各自 embedding 上独立 KMeans 得到的，组语义未必一致；
3. OT coupling 可能过度平滑，损失了 paired-cell 精细匹配；
4. TACO / GLUE 类方法的关键不只是“软聚类”，还包括用结构图、跨模态先验和表示学习来塑造共同空间。

## 6. 与我们论文主线的关系

这次 PBMC 实验强化了一个判断：

> 我们的论文骨干不应该写成“给 HiWA 加一个 soft clustering 参数就提高精度”，因为证据不支持。

更合理的骨干是：

> 基于 HiWA 的 hierarchical/group-aware OT 框架，吸收 TACO / multi-omics alignment 的 soft prototype、component-conditioned transport 和结构先验思想，构建一个面向多数据集跨模态对齐的通用框架；ROCA 是其中针对低维正交分支不稳定的一个无标签稳定化模块。

PBMC 数据给我们提供了第二类任务：

- 猕猴神经元—运动：低维连续行为对齐，ROCA 很有价值；
- PBMC RNA-ATAC：高维跨组学配对对齐，普通 Procrustes 很强，说明我们需要把表示学习/结构先验做得更完整。

## 7. 本轮失败和工程限制

### 7.1 1000 / 500 cell 预处理暂时失败

1000-cell 版本和 500-cell 版本在当前 Python 环境下因为 anndata / scipy sparse 切片内存分配失败，没有继续强行跑。

失败点不是模型本身，而是读取和稀疏矩阵 materialization 策略。

后续修复路线：

- 写分块读取；
- 或先保存轻量 feature subset；
- 或直接读取 scGLUE / Seurat 已处理 embedding；
- 或在更大内存环境跑 1k / 5k。

### 7.2 当前结果不能作为最终论文结论

本轮是 smoke experiment，只能说明：

- PBMC 数据已经接入；
- 代码链路跑通；
- 当前 CC-HiWA 在 300-cell smoke 上没有超过普通 Procrustes；
- 这提示方法需要更强的表示学习和跨模态结构约束。

不能声称：

- CC-HiWA 在 PBMC 上无效；
- ROCA / CC-HiWA 一般无效；
- Procrustes 一定是最终最强基线；
- 当前数字可直接写入最终论文主表。

## 8. 文件位置

代码与结果：

- 预处理脚本：`D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/scripts/prepare_pbmc_multiome.py`
- 实验脚本：`D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/scripts/run_cc_hiwa_pbmc.py`
- 处理后数据：`D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/data/processed/pbmc_multiome_300_cc_hiwa.npz`
- raw JSON：`D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/results/cc_hiwa_pbmc_raw_smoke_300_v3_2026_07_09.json`
- summary JSON：`D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/results/cc_hiwa_pbmc_summary_smoke_300_v3_2026_07_09.json`
- 图像：`D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/figures/cc_hiwa_pbmc_summary_smoke_300_v3_2026_07_09.png`
- 笔记内嵌图：`D:/ai/数学ai/Projects/最优传输/最优传输/文本笔记/CC-HiWA-PBMC-RNA-ATAC/_attachments/cc_hiwa_pbmc_summary_smoke_300_v3_2026_07_09.png`

## 9. 下一步建议

我建议下一步不要继续盲目调 $\beta$，而是做三件事：

1. 先把 PBMC 预处理改成稳定的 1k / 5k 管线；
2. 加入真正的 multi-omics 结构先验，例如 gene activity、peak-gene graph 或 GLUE 式 feature graph；
3. 设计一个“Procrustes / OT / group-conditioned OT / graph-informed CC-HiWA”的统一对照表。

这样我们才是在吸收 TACO/GLUE 的精髓，而不是只借了“软分组”这个外壳。

---

## 10. 2026-07-10 更新：1k top-feature smoke

上一轮 300-cell smoke 解决了“能不能跑”的问题，但样本太小。今天我把预处理脚本改成了更稳定的稀疏友好版本：

1. 不再用 anndata 同时 fancy-index 行和列，因为这会触发 ATAC 完整矩阵读入；
2. 直接读取 h5ad 底层 CSR 结构：`X/data`、`X/indices`、`X/indptr`；
3. 逐行读取选中的 cell；
4. 同时用 feature map 只保留选中的 RNA/ATAC features；
5. RNA 使用 `highly_variable_rank`，实际可用高变基因为 2000 个；
6. ATAC 使用 `n_counts`，保留 top 20000 peaks。

运行命令：

```powershell
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'
python scripts\prepare_pbmc_multiome.py --rna-h5ad data\raw\10x-Multiome-Pbmc10k-RNA.h5ad --atac-h5ad data\raw\10x-Multiome-Pbmc10k-ATAC.h5ad --sample-cells 1000 --top-rna-features 5000 --top-atac-features 20000 --embedding-dim 20 --components 8 --temperature 1.0 --seed 0 --output data\processed\pbmc_multiome_1k_topfeat_cc_hiwa.npz
```

```powershell
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'
python scripts\run_cc_hiwa_pbmc.py --input data\processed\pbmc_multiome_1k_topfeat_cc_hiwa.npz --betas 0 0.01 0.05 0.1 0.25 0.5 --sinkhorn-maxiter 150 --tag smoke_1k_topfeat_2026_07_10
```

图像：

![PBMC CC-HiWA 1k top-feature smoke](_attachments/cc_hiwa_pbmc_summary_smoke_1k_topfeat_2026_07_10.png)

### 10.1 1k 结果表

| 方法 | Recall@1 | Recall@5 | Recall@10 | Cell-type transfer acc |
|---|---:|---:|---:|---:|
| Random expectation | 0.0010 | 0.0050 | 0.0100 | - |
| Raw embedding cosine | 0.0010 | 0.0080 | 0.0170 | 0.0860 |
| Orthogonal Procrustes cosine | 0.0280 | 0.0750 | 0.1400 | 0.6430 |
| CC-HiWA $\beta=0$ | 0.0020 | 0.0150 | 0.0220 | 0.0910 |
| CC-HiWA $\beta=0.01$ | 0.0010 | 0.0150 | 0.0220 | 0.0890 |
| CC-HiWA $\beta=0.05$ | 0.0010 | 0.0140 | 0.0210 | 0.0770 |
| CC-HiWA $\beta=0.1$ | 0.0010 | 0.0130 | 0.0210 | 0.0730 |
| CC-HiWA $\beta=0.25$ | 0.0010 | 0.0120 | 0.0220 | 0.0640 |
| CC-HiWA $\beta=0.5$ | 0.0010 | 0.0110 | 0.0200 | 0.0570 |

### 10.2 新结论

1k 结果比 300-cell 更可信，结论也更硬：

- PBMC RNA-ATAC 确实有跨模态可对齐信号，因为 Procrustes 仍明显高于随机；
- 当前 CC-HiWA 只比随机略好，远低于 Procrustes；
- $\beta$ 增大会让 transport 更集中，但不是更正确；
- 当前 soft assignment / group-conditioned OT 还没有真正利用单细胞多组学中的 feature-level 生物结构。

### 10.3 对主线的影响

这一步非常关键。它告诉我们：

> 如果要把方法写成一般方法，不能只说“HiWA + soft clustering + group-conditioned OT”。这个版本在 PBMC 上证据不足。

更合理的下一版应叫作类似：

> Structure-informed Component-Conditioned HiWA。

它应该把 TACO / GLUE / 单细胞多组学对齐里的精髓吸收进来：

- 不是只做 soft clusters；
- 而是让 component / prototype / transport cost 受到跨模态 feature graph 约束；
- 在 PBMC 中可以是 gene activity 或 peak-gene graph；
- 在神经数据中可以是方向/时间/局部轨迹结构；
- 在第三个数据集中也应找到相应的跨模态结构先验。

### 10.4 下一步更具体

下一步建议不要再调 $\beta$，而是做一个最小的 graph-informed PBMC 版本：

1. 从 RNA gene 坐标和 ATAC peak 坐标构造 peak-gene 邻接；
2. 把 ATAC peaks 聚合成 gene activity；
3. 让 RNA 表达和 ATAC gene activity 进入共享 gene feature space；
4. 在共享空间上重新构造 soft prototype；
5. 再运行 CC-HiWA；
6. 与 Raw、Procrustes、当前 CC-HiWA 比较。

这才是真正吸收 TACO/GLUE 的结构思想。

---

## 11. 2026-07-10 关键修正：Procrustes warm-start CC-HiWA

进一步检查代码后发现一个重要问题：

`fit_cc_hiwa` 核心函数本来支持传入 `rotation`，但 PBMC runner 第一版没有把 Procrustes rotation 传进去。因此上一节的 “CC-HiWA” 实际是：

> 未旋转的 CC-HiWA。

而 Procrustes baseline 是：

> 已经做过无标签正交对齐后的 nearest-neighbor baseline。

所以如果直接比较二者，会不够公平。

我已经给 `run_cc_hiwa_pbmc.py` 增加：

```powershell
--rotation-mode procrustes
```

这会先用无标签 Procrustes 得到：

$$
\hat{Q}=\arg\min_{Q^\top Q=I}\|XQ-Y\|_F^2,
$$

然后把这个 rotation 传入 CC-HiWA，使 CC-HiWA 在对齐后的空间中计算 representatives、group OT 和 sample OT。

运行命令：

```powershell
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'
python scripts\run_cc_hiwa_pbmc.py --input data\processed\pbmc_multiome_1k_topfeat_cc_hiwa.npz --betas 0 0.01 0.05 0.1 0.25 0.5 --sinkhorn-maxiter 150 --rotation-mode procrustes --tag smoke_1k_topfeat_procrustes_2026_07_10
```

图像：

![PBMC CC-HiWA 1k Procrustes warm-start smoke](_attachments/cc_hiwa_pbmc_summary_smoke_1k_topfeat_procrustes_2026_07_10.png)

### 11.1 Procrustes warm-start 结果

| 方法 | Recall@1 | Recall@5 | Recall@10 | Cell-type transfer acc |
|---|---:|---:|---:|---:|
| Random expectation | 0.0010 | 0.0050 | 0.0100 | - |
| Raw embedding cosine | 0.0010 | 0.0080 | 0.0170 | 0.0860 |
| Orthogonal Procrustes cosine | 0.0280 | 0.0750 | 0.1400 | 0.6430 |
| CC-HiWA + Procrustes $\beta=0$ | 0.0200 | 0.0650 | 0.1170 | 0.6570 |
| CC-HiWA + Procrustes $\beta=0.01$ | 0.0200 | 0.0650 | 0.1170 | 0.6570 |
| CC-HiWA + Procrustes $\beta=0.05$ | 0.0200 | 0.0650 | 0.1150 | 0.6590 |
| CC-HiWA + Procrustes $\beta=0.1$ | 0.0210 | 0.0640 | 0.1160 | 0.6580 |
| CC-HiWA + Procrustes $\beta=0.25$ | 0.0210 | 0.0640 | 0.1170 | 0.6600 |
| CC-HiWA + Procrustes $\beta=0.5$ | 0.0190 | 0.0660 | 0.1160 | 0.6600 |

### 11.2 修正后的解释

更准确的结论是：

1. 未旋转 CC-HiWA 接近随机，说明对齐前直接做 OT 不够；
2. Procrustes warm-start 后，CC-HiWA 显著提高，说明正交初始化是必要步骤；
3. Procrustes warm-start CC-HiWA 接近 Procrustes baseline，但 paired-cell Recall 仍略低；
4. cell-type transfer accuracy 约 0.657 到 0.660，略高于 Procrustes 的 0.643，说明 OT coupling 可能更偏向 cell-type 层面的匹配，而不是精确 paired-cell 匹配；
5. $\beta$ 仍没有明显带来 paired-cell retrieval 增益。

这对论文主线很重要：

> 在 PBMC 这类高维跨模态任务中，Procrustes / rotation warm-start 不是可选细节，而是 CC-HiWA 的必要前置步骤。

但同时：

> 当前 component-conditioned term 还没有证明能提高 paired-cell retrieval；它可能更影响 cell-type-level coupling。

### 11.3 更新后的下一步

下一版 GI-CC-HiWA 应采用：

1. gene activity / peak-gene graph 构造共享结构；
2. Procrustes warm-start；
3. shared prototypes；
4. component-conditioned OT；
5. paired-cell retrieval 与 cell-type transfer 双指标评价。

这样才是公平、完整、可写进论文方法的 PBMC 协议。

---

## 12. 2026-07-10 第一版 Gene-activity / shared-prototype GI-CC-HiWA

为了把 graph-informed 方向从计划推进到可运行入口，我新增了：

```text
D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/scripts/prepare_pbmc_gene_activity.py
```

这个脚本做了四件事：

1. 从 RNA var 读取 gene name 和 gene 坐标；
2. 从 ATAC var 读取 peak 坐标；
3. 用 $\pm 50\mathrm{kb}$ 窗口构造 peak-gene graph；
4. 把 ATAC peak matrix 聚合成 gene activity，再生成 CC-HiWA 输入。

本轮设置：

| 项目 | 值 |
|---|---:|
| cells | 1000 |
| genes | 2000 |
| selected peaks | 23853 |
| peak-gene edges | 26357 |
| window | 50000 bp |
| prototype mode | shared_procrustes |

运行命令：

```powershell
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'
python scripts\prepare_pbmc_gene_activity.py --rna-h5ad data\raw\10x-Multiome-Pbmc10k-RNA.h5ad --atac-h5ad data\raw\10x-Multiome-Pbmc10k-ATAC.h5ad --sample-cells 1000 --top-genes 2000 --window-bp 50000 --embedding-dim 20 --components 8 --temperature 1.0 --seed 0 --prototype-mode shared_procrustes --output data\processed\pbmc_gene_activity_1k_sharedproto_cc_hiwa.npz
```

```powershell
$env:OPENBLAS_NUM_THREADS='1'; $env:OMP_NUM_THREADS='1'
python scripts\run_cc_hiwa_pbmc.py --input data\processed\pbmc_gene_activity_1k_sharedproto_cc_hiwa.npz --betas 0 0.01 0.05 0.1 0.25 0.5 --sinkhorn-maxiter 150 --rotation-mode procrustes --tag gene_activity_1k_sharedproto_procrustes_2026_07_10
```

图像：

![PBMC gene-activity GI-CC-HiWA 1k smoke](_attachments/cc_hiwa_pbmc_summary_gene_activity_1k_sharedproto_procrustes_2026_07_10.png)

### 12.1 Gene-activity 结果

| 方法 | Recall@1 | Recall@5 | Recall@10 | Cell-type transfer acc |
|---|---:|---:|---:|---:|
| Random expectation | 0.0010 | 0.0050 | 0.0100 | - |
| Raw gene-activity cosine | 0.0000 | 0.0160 | 0.0300 | 0.3630 |
| Gene-activity Procrustes cosine | 0.0210 | 0.0840 | 0.1220 | 0.7040 |
| GI-CC-HiWA $\beta=0$ | 0.0200 | 0.0700 | - | 0.6610 |
| GI-CC-HiWA $\beta=0.1$ | 0.0190 | 0.0710 | - | 0.6630 |
| GI-CC-HiWA $\beta=0.5$ | 0.0200 | 0.0730 | - | 0.6680 |

完整 JSON 见：

```text
D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/results/cc_hiwa_pbmc_summary_gene_activity_1k_sharedproto_procrustes_2026_07_10.json
```

### 12.2 怎么理解？

这次结果比“未旋转 CC-HiWA”更有研究价值：

1. gene activity 明显增强了 cell-type 层面的结构信号：
   - raw top-feature cell-type acc：0.086；
   - raw gene-activity cell-type acc：0.363。
2. gene-activity Procrustes 的 cell-type acc 达到 0.704，高于普通 top-feature Procrustes 的 0.643。
3. 但是 GI-CC-HiWA 的 paired retrieval 仍没有超过 gene-activity Procrustes：
   - Procrustes Recall@5：0.084；
   - GI-CC-HiWA best Recall@5：0.073。
4. $\beta$ 增大有轻微 R@5 增益，但幅度很小，不能写成明确提升。

### 12.3 更新后的研究判断

现在的判断比之前更细：

> graph-informed representation 是有帮助的，尤其增强 cell-type-level signal；

但：

> 当前 component-conditioned OT 还没有把这种结构优势转化为 paired-cell retrieval 的提升。

所以下一步不是推翻 graph-informed 路线，而是检查 component-conditioned OT 的设计：

- group transport 是否太粗；
- sample OT 是否过度平滑；
- shared prototypes 是否真的跨模态一致；
- 是否应该让 representative/prototype 参与 cost，而不只是通过 $A P B^\top$ 间接影响；
- 是否要把 Procrustes nearest-neighbor 作为 candidate graph，再在局部做 OT refinement。

这已经比“换数据看看”前进了一大步：我们现在有了一个可运行、可复现、能指出失败原因的第二数据集底座。
