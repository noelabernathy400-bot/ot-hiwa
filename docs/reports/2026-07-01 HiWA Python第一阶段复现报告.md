# HiWA Python 第一阶段复现报告

- 日期：2026-07-01
- 目标：不改动官方仓库，建立可追溯的 Python 复现入口，并检验合成数据与猴神经演示。

## 改动文件

- `Projects/最优传输/Experiments/hiwa_python_reproduction/README.md`
- `Projects/最优传输/Experiments/hiwa_python_reproduction/requirements.txt`
- `Projects/最优传输/Experiments/hiwa_python_reproduction/scripts/common.py`
- `Projects/最优传输/Experiments/hiwa_python_reproduction/scripts/run_synthetic.py`
- `Projects/最优传输/Experiments/hiwa_python_reproduction/scripts/run_neural.py`
- `Projects/最优传输/Experiments/hiwa_python_reproduction/results/`
- `Projects/最优传输/Experiments/hiwa_python_reproduction/figures/`
- `Projects/最优传输/Experiments/hiwa_python_reproduction/复现记录-2026-07-01.md`

## 关键结果

- 猴神经速度 R2：5 种子 `0.6282 +/- 0.0042`，作者保存值为 `0.6307`。
- 猴神经方向准确率：平均 `45.46%`，范围 `41.41%-51.85%`，作者保存单次值为 `52.65%`。
- 合成数据严格档固定种子 2 得到 `26.00% -> 50.00%`，未收敛，尚未复现作者的 `94.75%`。
- 合成数据 10 种子、50 轮诊断的准确率范围为 `25.75%-93.25%`，显示强烈的初始化敏感性；`93.25%` 运行未收敛，不作为正式复现证据。

## 关键决定

- Python 作为后续主实现，MATLAB 仅作原始数值参考。
- 原始 Python/MATLAB 仓库和数据保持未修改。
- 作者单次保存输出与我们的固定多种子结果分开报告。
- 合成数据未达验收标准，明确标记为未完成，不宣称论文已全面复现。

## 风险与下一步

- Python 实现有归一化状态未传递到 `transform()` 的风险。
- 需测试旧版 NumPy/SciPy/scikit-learn 组合，并用 MATLAB 官方演示交叉验证合成数据。
- 后续创新实验应以多种子均值和不确定性为主，不以单次最优值作为基线。
