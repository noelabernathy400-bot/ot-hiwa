---
name: modeling-programming
description: Use when writing mathematical modeling code, Python scripts, notebooks, optimization models, simulations, machine learning baselines, visualizations, and reproducible modeling programs.
---

# 模型程序开发

## 何时使用

- 为数学建模或研究项目写 Python 代码。
- 实现优化、仿真、预测、统计、机器学习、可视化或敏感性分析。

## 工作流

1. 先明确数学模型、输入、输出和评价指标。
2. 先写最小可运行版本，再扩展。
3. 保留原始数据，只读入不覆盖。
4. 将清洗、建模、评估、画图拆成清晰函数。
5. 固定随机种子，记录环境和依赖。
6. 输出结果时同时保存图表、指标和解释。
7. 把关键运行写入实验记录。

## 推荐结构

- `Data/processed/`
- `Experiments/notebooks/`
- `Experiments/scripts/`
- `Experiments/results/`
- `Experiments/figures/`

## 质量标准

- 能运行。
- 输入输出清楚。
- 参数集中管理。
- 图表有标题、单位和说明。
- 结果可复现。

## 安全边界

- 不静默删除数据。
- 不把调参结果伪装成验证结果。
- 不覆盖原始数据。
