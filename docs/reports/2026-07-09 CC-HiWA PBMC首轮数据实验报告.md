# 2026-07-09 CC-HiWA PBMC首轮数据实验报告

## 目标

将 CC-HiWA / ROCA-HiWA 项目从猕猴神经元—运动数据扩展到一个非神经跨模态数据集，并单独建立 PBMC RNA-ATAC 子项目，用于检查方法是否具备一般化研究潜力。

## 本次改动

新增或修改：

- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/scripts/prepare_pbmc_multiome.py`
- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/scripts/run_cc_hiwa_pbmc.py`
- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/data/raw/10x-Multiome-Pbmc10k-RNA.h5ad`
- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/data/raw/10x-Multiome-Pbmc10k-ATAC.h5ad`
- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/data/processed/pbmc_multiome_300_cc_hiwa.npz`
- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/results/cc_hiwa_pbmc_raw_smoke_300_v3_2026_07_09.json`
- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/results/cc_hiwa_pbmc_summary_smoke_300_v3_2026_07_09.json`
- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/figures/cc_hiwa_pbmc_summary_smoke_300_v3_2026_07_09.png`
- `D:/ai/数学ai/Projects/最优传输/最优传输/文本笔记/CC-HiWA-PBMC-RNA-ATAC/_attachments/cc_hiwa_pbmc_summary_smoke_300_v3_2026_07_09.png`
- `D:/ai/数学ai/Projects/最优传输/最优传输/文本笔记/CC-HiWA-PBMC-RNA-ATAC/PBMC-CC-HiWA首轮实验笔记-2026-07-09.md`

依赖安装：

- `anndata`
- `h5py`

## 数据

使用 scGLUE 提供的 10x-Multiome-Pbmc10k processed h5ad 文件：

- RNA shape: $9631 \times 29095$
- ATAC shape: $9631 \times 107194$
- 共同细胞数：9631
- 含 `cell_type` 字段

标签只用于最终评价，没有进入训练、transport 或模型选择。

## 运行命令

```powershell
python scripts\prepare_pbmc_multiome.py --rna-h5ad data\raw\10x-Multiome-Pbmc10k-RNA.h5ad --atac-h5ad data\raw\10x-Multiome-Pbmc10k-ATAC.h5ad --sample-cells 300 --embedding-dim 20 --components 8 --temperature 1.0 --seed 0 --output data\processed\pbmc_multiome_300_cc_hiwa.npz
```

```powershell
python scripts\run_cc_hiwa_pbmc.py --input data\processed\pbmc_multiome_300_cc_hiwa.npz --betas 0 0.01 0.05 0.1 0.25 0.5 --sinkhorn-maxiter 150 --tag smoke_300_v3_2026_07_09
```

## 结果摘要

| 方法 | Recall@1 | Recall@5 | Recall@10 | Cell-type transfer acc |
|---|---:|---:|---:|---:|
| Random expectation | 0.0033 | 0.0167 | 0.0333 | - |
| Raw embedding cosine | 0.0167 | 0.0600 | 0.1067 | 0.3133 |
| Orthogonal Procrustes cosine | 0.0833 | 0.2400 | 0.3867 | 0.5433 |
| CC-HiWA best R@5 beta=0/0.01 | 0.0033 | 0.0800 | 0.1233 | 0.3600 |

主要判断：

- PBMC 数据存在可对齐信号，普通 Procrustes 明显强于随机；
- 当前 CC-HiWA smoke 版本没有超过普通 Procrustes；
- 增大 group-conditioned 权重 $\beta$ 降低 transport entropy，但没有提升 Recall；
- 这支持下一阶段转向更完整的 structure-informed / graph-informed / representation-learning 版本，而不是继续单纯调 $\beta$。

## 验证

已通过：

- `python scripts\prepare_pbmc_multiome.py --help`
- `python scripts\run_cc_hiwa_pbmc.py --help`
- `python -m unittest discover -s tests -v`

说明：

- 第一次并行验证时 OpenBLAS 在当前内存紧张环境下触发 memory allocation failure；
- 设置 `OPENBLAS_NUM_THREADS=1` 和 `OMP_NUM_THREADS=1` 后顺序重跑，35 个 HiWA / ROCA / CC-HiWA 单元测试全部通过。

## 风险与限制

- 本轮只跑了 300-cell smoke，不是最终性能实验；
- 1000-cell 和 500-cell 在当前环境下触发 anndata/scipy sparse 内存分配失败；
- 当前 embedding 只是 SVD baseline，不是 scGLUE/TACO/Seurat 级别的强表示；
- 当前 soft assignment 独立来自两个模态的 KMeans，组语义不一定一致；
- 结论不能扩大为“CC-HiWA 在 PBMC 上无效”。

## 下一步

建议优先：

1. 改造 PBMC 预处理为分块或轻量 feature subset，使 1k/5k 可稳定运行；
2. 加入 gene activity / peak-gene graph / GLUE 式 feature graph；
3. 建立三个数据集统一协议：猕猴神经元—运动、PBMC RNA-ATAC、第三个非生物或视觉/文本类数据；
4. 把论文主线定位为 “component-conditioned / structure-informed hierarchical OT alignment”，ROCA 作为低维正交分支稳定化模块。

## 已知阻塞

Claude/OpenCode 辅助本轮未成功使用：之前尝试本地 Claude worker 时遇到 PowerShell execution policy 和 Claude 预算限制。本轮关键实验由 Codex 直接完成。

## 2026-07-10 补充：1k top-feature smoke

已修复 500/1000-cell 预处理中的关键内存问题：

- 原因：backed AnnData 同时 fancy-index 行和列时，会触发 ATAC 完整稀疏矩阵读入；
- 修复：直接读取 h5ad 底层 CSR 结构 `X/data`、`X/indices`、`X/indptr`，逐行读取选中细胞，并用 feature map 保留 top features；
- 当前 1k 设置：1000 cells，RNA 2000 个高变基因，ATAC 20000 个高计数 peaks。

1k 结果：

| 方法 | Recall@1 | Recall@5 | Recall@10 | Cell-type transfer acc |
|---|---:|---:|---:|---:|
| Random expectation | 0.0010 | 0.0050 | 0.0100 | - |
| Raw embedding cosine | 0.0010 | 0.0080 | 0.0170 | 0.0860 |
| Orthogonal Procrustes cosine | 0.0280 | 0.0750 | 0.1400 | 0.6430 |
| CC-HiWA best R@5 beta=0/0.01 | 0.0020 | 0.0150 | 0.0220 | 0.0910 |

新增结果文件：

- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/data/processed/pbmc_multiome_1k_topfeat_cc_hiwa.npz`
- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/results/cc_hiwa_pbmc_raw_smoke_1k_topfeat_2026_07_10.json`
- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/results/cc_hiwa_pbmc_summary_smoke_1k_topfeat_2026_07_10.json`
- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/figures/cc_hiwa_pbmc_summary_smoke_1k_topfeat_2026_07_10.png`

更新判断：1k 结果进一步确认 PBMC 有可对齐信号，但当前 CC-HiWA 简化版没有超过 Procrustes。下一步应转向 graph-informed / structure-informed CC-HiWA，例如 peak-gene graph、gene activity 或 GLUE 式 feature graph，而不是继续调 $\beta$。

## 2026-07-10 关键修正：Procrustes warm-start

复查发现，PBMC runner 第一版没有把 Procrustes rotation 传入 `fit_cc_hiwa`，因此上一节的 CC-HiWA 是未旋转版本。已新增参数：

- `--rotation-mode none`
- `--rotation-mode procrustes`

Procrustes warm-start 1k 结果：

| 方法 | Recall@1 | Recall@5 | Recall@10 | Cell-type transfer acc |
|---|---:|---:|---:|---:|
| Orthogonal Procrustes cosine | 0.0280 | 0.0750 | 0.1400 | 0.6430 |
| CC-HiWA + Procrustes best R@1 beta=0.1/0.25 | 0.0210 | 0.0640 | 0.1160--0.1170 | 0.6580--0.6600 |
| CC-HiWA + Procrustes best R@5 beta=0.5 | 0.0190 | 0.0660 | 0.1160 | 0.6600 |

新增结果文件：

- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/results/cc_hiwa_pbmc_raw_smoke_1k_topfeat_procrustes_2026_07_10.json`
- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/results/cc_hiwa_pbmc_summary_smoke_1k_topfeat_procrustes_2026_07_10.json`
- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/figures/cc_hiwa_pbmc_summary_smoke_1k_topfeat_procrustes_2026_07_10.png`

修正后判断：

- Procrustes warm-start 是 PBMC CC-HiWA 的必要协议组件；
- warm-start 后 CC-HiWA 接近强 Procrustes baseline，但 paired-cell retrieval 仍未超过；
- cell-type transfer accuracy 略高于 Procrustes，提示 OT coupling 可能更偏 cell-type 层级而非精确 paired-cell；
- 后续应固定 warm-start，再研究 graph-informed/shared prototype 是否提供增益。

## 2026-07-10 补充：第一版 Gene-activity GI-CC-HiWA

已新增：

- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/scripts/prepare_pbmc_gene_activity.py`

该脚本从 RNA gene 坐标与 ATAC peak 坐标构造 peak-gene graph，并把 ATAC peak matrix 聚合为 gene activity。第一版设置：

- cells: 1000
- genes: 2000
- selected peaks: 23853
- peak-gene edges: 26357
- window: 50000 bp
- prototype mode: shared_procrustes

结果：

| 方法 | Recall@1 | Recall@5 | Recall@10 | Cell-type transfer acc |
|---|---:|---:|---:|---:|
| Raw gene-activity cosine | 0.0000 | 0.0160 | 0.0300 | 0.3630 |
| Gene-activity Procrustes cosine | 0.0210 | 0.0840 | 0.1220 | 0.7040 |
| GI-CC-HiWA best beta=0.5 | 0.0200 | 0.0730 | 0.1160 | 0.6680 |

新增结果：

- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/data/processed/pbmc_gene_activity_1k_sharedproto_cc_hiwa.npz`
- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/results/cc_hiwa_pbmc_summary_gene_activity_1k_sharedproto_procrustes_2026_07_10.json`
- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/figures/cc_hiwa_pbmc_summary_gene_activity_1k_sharedproto_procrustes_2026_07_10.png`

当前判断：

- graph-informed representation 有帮助，尤其增强 cell-type-level signal；
- 但第一版 GI-CC-HiWA 仍没有超过 gene-activity Procrustes 的 paired-cell retrieval；
- 下一步应检查 component-conditioned OT 本身：group transport 是否太粗、sample OT 是否过平滑、representative 是否应直接进入 cost。

验证补充：

- `python scripts\prepare_pbmc_multiome.py --help` 退出码 0；
- `python scripts\prepare_pbmc_gene_activity.py --help` 退出码 0；
- `python scripts\run_cc_hiwa_pbmc.py --help` 退出码 0；
- 设置 `OPENBLAS_NUM_THREADS=1` 与 `OMP_NUM_THREADS=1` 后，`python -m unittest discover -s tests -v` 通过，35/35 OK。
