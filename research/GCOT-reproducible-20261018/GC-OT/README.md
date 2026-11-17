# GCOT：算法验证与 DANDI 000688 跨日神经对齐

工程保留 HiWA/GCOT 的合成与神经算法验证，并包含 DANDI 000688 猕猴 M1/PMd 跨日记录与 CMU Panoptic 跨相机三维骨架实验。方向标签和样本对应均不参与 GCOT 的拟合或停止判断。

## 目录职责

- `gcot/`：GCOT 算法、数值计算、分组和基线实现。
- `experiments/dandi_000688/`：正式数据处理、对齐、评价和绘图流水线。
- `experiments/panoptic/`：`171204_pose2` 三簇跨相机对齐流水线。
- `experiments/synthetic/`、`experiments/neural/`：通用算法验证。
- `visualization/`：只读取已保存结果并绘图。
- `evaluation/`：指标与数据/结果完整性验证。
- `gcot/preprocessing.py`：实验共用的投影与预处理实现。
- `data/processed/`：固定实验输入。
- `results/dandi_000688/`：正式 DANDI 实验结果。
- `tests/`：算法、协议和目录边界测试。
- `docs/`：算法说明、来源和历史审计记录。

根目录只保留项目说明和 Python 环境元数据，不放运行脚本或绘图脚本。

## Python 模块入口

```powershell
# 计算
.venv/Scripts/python.exe -m experiments.synthetic.run --trials 50 --workers 6
.venv/Scripts/python.exe -m experiments.neural.run --stage all
.venv/Scripts/python.exe -m experiments.dandi_000688.run data/dandi_000688/raw results/dandi_000688
.venv/Scripts/python.exe -m experiments.panoptic.run data/panoptic/171204_pose2 results/panoptic

# 仅绘图
.venv/Scripts/python.exe -m visualization.synthetic
.venv/Scripts/python.exe -m visualization.neural

# 评估与完整性检查
.venv/Scripts/python.exe -m evaluation.metrics
.venv/Scripts/python.exe -m evaluation.validation --scope all
```

## 主要结果

- [合成六组配置](results/synthetic/main/figures/figure1_all_six.png)
- [合成 Figure 1e/f](results/synthetic/main/figures/figure1ef.png)
- [神经 Figure 2a（GCOT）](results/neural/main/gcot/figures/figure2a.png)
- [DANDI 000688 跨日对齐](results/dandi_000688/figure2a_dandi.png)
- [DANDI 000688 定量比较](results/dandi_000688/alignment_accuracy.png)
- [Panoptic PCA/t-SNE 三簇](results/panoptic/pca_tsne_three_groups.png)
- [Panoptic HiWA 风格对齐](results/panoptic/figure2a_panoptic.png)

`results/*/author_reference/` 是未经改写的作者参考输入和产物，只用于协议复核。
