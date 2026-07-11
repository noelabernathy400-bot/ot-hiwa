---
title: CC-HiWA PBMC RNA-ATAC 子项目入口
date: 2026-07-09
tags:
  - CC-HiWA
  - PBMC
  - RNA-ATAC
  - single-cell
  - multiome
---

# CC-HiWA PBMC RNA-ATAC 子项目入口

## 1. 子项目定位

这是 CC-HiWA 三数据集计划中的第二个数据集：

> D2：PBMC RNA–ATAC 单细胞多组学跨模态对齐。

它的作用是验证：

> CC-HiWA 是否不仅适用于神经—运动对齐，也能用于非神经领域的多模态表示匹配。

当前神经数据首轮 CC-HiWA 结果显示，后处理式 sample OT 在神经—运动任务上不理想。PBMC RNA–ATAC 更适合作为 CC-HiWA 最小版的试金石，因为它天然是跨模态 label transfer / matching 问题。

## 2. 为什么选择 PBMC RNA–ATAC

PBMC multiome 数据同时测量同一批细胞的：

- RNA gene expression；
- ATAC chromatin accessibility。

因此它天然具有：

- 两个模态；
- 潜在 cell type / cell state components；
- 可以保留但不用于训练的评价标签；
- paired cells 或 cell type transfer 评价；
- 不需要 3D rotation，也不需要 ROCA。

## 3. 当前文件夹结构

文档目录：

```text
最优传输/文本笔记/CC-HiWA-PBMC-RNA-ATAC/
```

实验目录：

```text
Experiments/cc_hiwa_pbmc/
├── scripts/
├── data/
│   ├── raw/
│   └── processed/
├── results/
├── figures/
└── tests/
```

## 4. 当前推荐数据路线

优先路线：

> 使用 scGLUE example datasets 中的 10x-Multiome-Pbmc10k RNA/ATAC h5ad 文件。

原因：

- 已经被单细胞多组学对齐教程使用；
- RNA 和 ATAC 分开成 h5ad，适合 Python；
- 比直接处理 10x 原始 fragments / matrix 更轻；
- 更适合第一版 CC-HiWA。

备选路线：

> 使用 10x Genomics 官方 10k Human PBMCs Multiome v1.0 原始数据。

风险：

- 数据更大；
- ATAC fragments 和 peaks 处理更重；
- 需要更多单细胞预处理代码；
- 不适合作为第一版快速验证。

## 5. 与 CC-HiWA 的接口

目标是把 PBMC 数据转成：

$$
X=\text{RNA embedding}
$$

$$
Y=\text{ATAC embedding}
$$

$$
A=\text{RNA soft component assignment}
$$

$$
B=\text{ATAC soft component assignment}
$$

$$
\mathcal{E}=\text{cell type labels / paired cell ids}
$$

其中 $\mathcal{E}$ 只用于最终评价。

## 6. 第一版评价指标

优先指标：

1. paired-cell retrieval accuracy；
2. cell type transfer accuracy；
3. macro F1；
4. transport entropy；
5. beta sensitivity。

如果没有可靠 cell type labels，第一版先做 paired retrieval。

## 7. 当前不做什么

第一版暂时不做：

- 端到端神经网络；
- peak-to-gene graph；
- scGLUE 完整模型复现；
- 大规模全量 fragments 处理；
- 与所有单细胞 SOTA 比较；
- 用 cell type labels 训练 CC-HiWA。

## 8. 当前完成状态

已完成：

1. 下载 scGLUE / 10x PBMC Multiome RNA 与 ATAC h5ad；
2. 检查数据 shape、共同 cell barcode 和 `cell_type` 字段；
3. 编写 `prepare_pbmc_multiome.py`；
4. 编写 `run_cc_hiwa_pbmc.py`；
5. 完成 300-cell smoke experiment；
6. 输出 raw JSON、summary JSON 和 PNG；
7. 将图片复制到本笔记文件夹 `_attachments`，避免 Obsidian 断链；
8. 写入首轮实验笔记：[[PBMC-CC-HiWA首轮实验笔记-2026-07-09]]。

当前主要结果：

- PBMC 数据有明显可对齐信号；
- 普通 Orthogonal Procrustes 在 300-cell smoke 上强于当前 CC-HiWA；
- 单纯增大 $\beta$ 没有提升 paired-cell retrieval；
- 下一步应加入更强的表示学习、feature graph 或 gene activity 结构，而不是继续盲目调参。

2026-07-10 更新：

- 已修复 backed sparse h5ad 同时行列切片导致的内存问题；
- 新预处理直接读取 h5ad 底层 CSR 结构，支持 top-feature 1k smoke；
- 已完成 1k top-feature 实验；
- 1k 未旋转 CC-HiWA 明显弱于 Procrustes；
- 已新增 `--rotation-mode procrustes` 并完成 Procrustes warm-start CC-HiWA；
- warm-start 后 CC-HiWA 接近 Procrustes baseline，但 paired-cell retrieval 仍未超过；
- 结论进一步转向 “Procrustes warm-start + graph-informed / structure-informed CC-HiWA”，而非继续调 $\beta$。
- 已新增第一版 gene-activity 输入脚本；
- gene activity 明显增强 cell-type signal，但 GI-CC-HiWA 仍未超过 gene-activity Procrustes。

## 9. 下一步

1. 在现有 CSR 读取基础上扩大到 2k / 5k；
2. 加入 gene activity / peak-gene graph / GLUE 式结构先验；
3. 建立 Procrustes、OT、CC-HiWA、graph-informed CC-HiWA 的统一对照表；
4. 将 PBMC 作为第二个正式数据集候选，而不是最终结论数据集。
