# 同学接手包：CC-HiWA / GI-CC-HiWA PBMC RNA-ATAC

> 日期：2026-07-10  
> 用途：让同学可以直接接手项目的一部分工作，避免两个人重复做同一件事。

## 1. 一句话说明这个项目

我们正在以 HiWA / hierarchical OT 为基础，借鉴 TACO / GLUE / 单细胞多组学对齐中的 soft prototype、group-conditioned OT 和 graph-informed representation 思想，尝试构建一个更一般的跨模态对齐方法。

当前 PBMC RNA-ATAC 是第二个数据集，用来验证方法是否能从“猕猴神经元—运动数据”推广到非神经领域。

## 2. 你接手前先看哪几个文件？

请按顺序看：

1. `D:/ai/数学ai/Projects/最优传输/最优传输/文本笔记/CC-HiWA-PBMC-RNA-ATAC/README.md`
2. `D:/ai/数学ai/Projects/最优传输/最优传输/文本笔记/CC-HiWA-PBMC-RNA-ATAC/PBMC-CC-HiWA首轮实验笔记-2026-07-09.md`
3. `D:/ai/数学ai/Projects/最优传输/最优传输/文本笔记/CC-HiWA-PBMC-RNA-ATAC/Graph-informed-CC-HiWA方法升级计划-2026-07-10.md`
4. `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/README.md`

如果只看一个，先看第 2 个。

## 3. 当前已经完成了什么？

### 3.1 数据

已经下载 PBMC Multiome RNA/ATAC h5ad：

```text
D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/data/raw/10x-Multiome-Pbmc10k-RNA.h5ad
D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/data/raw/10x-Multiome-Pbmc10k-ATAC.h5ad
```

数据规模：

| 模态 | shape |
|---|---:|
| RNA | 9631 × 29095 |
| ATAC | 9631 × 107194 |

### 3.2 已经写好的脚本

```text
D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/scripts/prepare_pbmc_multiome.py
D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/scripts/prepare_pbmc_gene_activity.py
D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/scripts/run_cc_hiwa_pbmc.py
```

### 3.3 当前核心结果

#### Top-feature + Procrustes warm-start

| 方法 | Recall@1 | Recall@5 | Cell-type acc |
|---|---:|---:|---:|
| Procrustes baseline | 0.028 | 0.075 | 0.643 |
| CC-HiWA + Procrustes best | 0.021 | 0.066 | 0.660 |

解释：

- Procrustes 是强 baseline；
- CC-HiWA warm-start 后接近 baseline；
- 但是 paired-cell retrieval 没有超过 baseline；
- cell-type acc 略高，说明 OT 可能更偏向 cell-type 层级。

#### Gene-activity / GI-CC-HiWA

| 方法 | Recall@1 | Recall@5 | Cell-type acc |
|---|---:|---:|---:|
| Gene-activity Procrustes | 0.021 | 0.084 | 0.704 |
| GI-CC-HiWA best | 0.020 | 0.073 | 0.668 |

解释：

- gene activity 明显增强 cell-type-level signal；
- 但第一版 GI-CC-HiWA 仍没有超过 gene-activity Procrustes；
- 下一步要改 component-conditioned OT 的设计，不是继续盲目调 $\beta$。

## 4. 环境和运行方式

建议在 PowerShell 里进入：

```powershell
cd D:\ai\数学ai\Projects\最优传输\Experiments\cc_hiwa_pbmc
```

为了避免 OpenBLAS 抢内存，建议每次运行前设置：

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
```

### 4.1 复现 1k top-feature 输入

```powershell
python scripts\prepare_pbmc_multiome.py --rna-h5ad data\raw\10x-Multiome-Pbmc10k-RNA.h5ad --atac-h5ad data\raw\10x-Multiome-Pbmc10k-ATAC.h5ad --sample-cells 1000 --top-rna-features 5000 --top-atac-features 20000 --embedding-dim 20 --components 8 --temperature 1.0 --seed 0 --output data\processed\pbmc_multiome_1k_topfeat_cc_hiwa.npz
```

### 4.2 复现 1k top-feature + Procrustes warm-start

```powershell
python scripts\run_cc_hiwa_pbmc.py --input data\processed\pbmc_multiome_1k_topfeat_cc_hiwa.npz --betas 0 0.01 0.05 0.1 0.25 0.5 --sinkhorn-maxiter 150 --rotation-mode procrustes --tag smoke_1k_topfeat_procrustes_2026_07_10
```

### 4.3 复现 gene-activity 输入

```powershell
python scripts\prepare_pbmc_gene_activity.py --rna-h5ad data\raw\10x-Multiome-Pbmc10k-RNA.h5ad --atac-h5ad data\raw\10x-Multiome-Pbmc10k-ATAC.h5ad --sample-cells 1000 --top-genes 2000 --window-bp 50000 --embedding-dim 20 --components 8 --temperature 1.0 --seed 0 --prototype-mode shared_procrustes --output data\processed\pbmc_gene_activity_1k_sharedproto_cc_hiwa.npz
```

### 4.4 复现 gene-activity GI-CC-HiWA

```powershell
python scripts\run_cc_hiwa_pbmc.py --input data\processed\pbmc_gene_activity_1k_sharedproto_cc_hiwa.npz --betas 0 0.01 0.05 0.1 0.25 0.5 --sinkhorn-maxiter 150 --rotation-mode procrustes --tag gene_activity_1k_sharedproto_procrustes_2026_07_10
```

## 5. 你不要重复做什么？

不要重复做：

- 不要重新下载数据，除非原文件丢失；
- 不要重新从零写 PBMC 数据读取；
- 不要继续只调 $\beta$；
- 不要把测试标签用于训练或模型选择；
- 不要只报告最好的一次结果；
- 不要覆盖已有 JSON / PNG / Markdown。

如果要重新跑，请换新的 `--tag`。

## 6. 推荐你负责的任务

最推荐你负责下面三个方向之一。

### 任务 A：做 2k / 5k 扩展实验

目的：

验证当前结论在更大样本下是否稳定。

建议命令改动：

- `--sample-cells 2000`
- 如果 2k 成功，再尝试 `--sample-cells 5000`

要记录：

| 项目 | 要写清楚 |
|---|---|
| sample size | 1000 / 2000 / 5000 |
| 输入类型 | top-feature / gene-activity |
| rotation mode | none / procrustes |
| best Recall@1 | 数字 |
| best Recall@5 | 数字 |
| best cell-type acc | 数字 |
| 是否超过 Procrustes | 是/否 |
| 运行时间 | 粗略记录 |
| 是否内存失败 | 是/否 |

### 任务 B：做 window-bp 消融

目的：

检查 gene activity 的 peak-gene graph 是否受窗口大小影响。

建议窗口：

```text
10000, 50000, 100000, 250000
```

要回答：

- window 越大，edge 越多，是否 cell-type acc 更高？
- paired-cell retrieval 有没有改善？
- 有没有出现过度平滑？

### 任务 C：改 component-conditioned OT

目的：

当前最大问题是 component-conditioned term 没有提升 paired-cell retrieval。你可以尝试让 representative 更直接进入 sample cost。

当前 cost 类似：

$$
C_{ij}=\|x_i-y_j\|^2-\beta\log(a_i^\top P b_j).
$$

可以尝试加入 representative direct cost：

$$
C_{ij}
=
\|x_i-y_j\|^2
\beta C_{ij}^{component}
\lambda \sum_{kl} a_{ik}b_{jl}\|r_k^X-r_l^Y\|^2.
$$

注意：这是新方法，不要直接覆盖原脚本。建议新建脚本或在 core 中加可选参数。

## 7. 你的交付格式

请每完成一个实验，写一个 Markdown：

```text
D:/ai/数学ai/Projects/最优传输/最优传输/文本笔记/CC-HiWA-PBMC-RNA-ATAC/同学实验记录-YYYY-MM-DD-任务名.md
```

必须包含：

1. 你做了什么；
2. 运行命令；
3. 参数；
4. 输出 JSON 路径；
5. 输出 PNG 路径；
6. 结果表；
7. 结论；
8. 是否失败；
9. 下一步建议。

## 8. 代码安全边界

可以改：

- `Experiments/cc_hiwa_pbmc/scripts/` 下新增脚本；
- 新增 `results/` 文件；
- 新增 `figures/` 文件；
- 新增 PBMC 笔记。

谨慎改：

- `Experiments/hiwa_python_reproduction/scripts/cc_hiwa.py`

如果要改 core，必须说明：

- 改了什么；
- 为什么要改；
- 是否影响已有神经实验；
- 是否跑了单元测试。

禁止：

- 删除 raw h5ad；
- 覆盖已有 JSON / PNG；
- 修改原作者 PyHiWA 源码；
- 用标签训练；
- 用测试结果选择最优模型然后伪装成无标签。

## 9. 最适合你现在开始做的任务

如果你想马上动手，建议从任务 A 开始：

> 跑 2k top-feature + Procrustes warm-start 和 2k gene-activity + Procrustes warm-start。

原因：

- 不需要改核心代码；
- 最容易验证当前结论是否稳定；
- 对论文最有用；
- 也最不容易和主线工作重叠。

如果 2k 跑通，再做任务 B。

## 10. 当前研究主线一句话

我们现在不是单纯追求分数，而是在判断：

> HiWA/TACO 式 soft prototype 与 group-conditioned OT，是否能在 Procrustes warm-start 和 graph-informed representation 的基础上，提供超过强 baseline 的跨模态对齐改进。

目前答案是：

> 数据和结构方向是有希望的，但第一版 component-conditioned OT 还不够，需要继续改 cost 和 representative 的参与方式。

