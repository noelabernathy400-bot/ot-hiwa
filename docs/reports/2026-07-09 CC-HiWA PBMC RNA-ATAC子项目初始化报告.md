# 2026-07-09 CC-HiWA PBMC RNA-ATAC子项目初始化报告

## 目标

根据用户要求，将 CC-HiWA 的第二个数据集 PBMC RNA–ATAC 单独开设文件夹，避免与 ROCA/神经数据笔记混在一起，并为后续换数据验证通用方法做准备。

## 新增文件夹

- `D:/ai/数学ai/Projects/最优传输/最优传输/文本笔记/CC-HiWA-PBMC-RNA-ATAC/`
- `D:/ai/数学ai/Projects/最优传输/最优传输/文本笔记/CC-HiWA-PBMC-RNA-ATAC/_attachments/`
- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/`
- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/scripts/`
- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/data/raw/`
- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/data/processed/`
- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/results/`
- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/figures/`
- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/tests/`

## 新增文档

- `D:/ai/数学ai/Projects/最优传输/最优传输/文本笔记/CC-HiWA-PBMC-RNA-ATAC/README.md`
- `D:/ai/数学ai/Projects/最优传输/最优传输/文本笔记/CC-HiWA-PBMC-RNA-ATAC/PBMC数据源与接入路线-2026-07-09.md`
- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/README.md`

## 新增脚本

- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/scripts/prepare_pbmc_multiome.py`
- `D:/ai/数学ai/Projects/最优传输/Experiments/cc_hiwa_pbmc/scripts/run_cc_hiwa_pbmc.py`

## 数据源判断

优先路线：

- scGLUE example datasets 中的 `10x-Multiome-Pbmc10k-RNA.h5ad` 和 `10x-Multiome-Pbmc10k-ATAC.h5ad`。

原因：

- Python 生态友好；
- h5ad 可直接读取；
- 适合第一版 smoke experiment；
- 避免一开始处理大型 10x 原始 fragments。

最终论文仍应追溯官方来源：

- 10x Genomics 10k Human PBMCs Multiome v1.0；
- Signac PBMC multiomic vignette 可作为处理流程参考。

## 验证结果

脚本 help 正常：

```bash
python scripts\prepare_pbmc_multiome.py --help
python scripts\run_cc_hiwa_pbmc.py --help
```

HiWA/CC-HiWA 全量测试：

```bash
python -m unittest discover -s tests -v
```

结果：

```text
Ran 35 tests
OK
```

## 风险

- 当前尚未下载 PBMC h5ad 数据。
- 尚未确认 h5ad 中是否包含 cell type labels。
- 如果无 labels，第一版先做 paired-cell retrieval。
- `anndata`/`scanpy` 依赖可能需要安装或使用现有 Python 环境确认。

## 下一步建议

1. 下载或定位 scGLUE `10x-Multiome-Pbmc10k` h5ad 文件。
2. 运行 `prepare_pbmc_multiome.py` 生成 1000 cells smoke NPZ。
3. 运行 `run_cc_hiwa_pbmc.py` 做 beta 网格。
4. 如果 paired retrieval 有改善，再扩大 cell 数和加入 cell type transfer。

