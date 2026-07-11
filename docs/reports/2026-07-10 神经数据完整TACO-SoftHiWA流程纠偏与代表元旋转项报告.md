# 2026-07-10 神经数据完整 TACO-SoftHiWA 流程纠偏与代表元旋转项报告

## 目标

回到猕猴神经元—运动数据，补齐 TACO 启发的 Soft-HiWA 主流程，尤其检查并实现代表元直接参与正交矩阵更新，而不是继续把 PBMC 诊断结果当成主线。

## 改动文件

- `D:\ai\数学ai\Projects\最优传输\Experiments\hiwa_python_reproduction\scripts\soft_hiwa.py`
- `D:\ai\数学ai\Projects\最优传输\Experiments\hiwa_python_reproduction\scripts\run_soft_neural.py`
- `D:\ai\数学ai\Projects\最优传输\Experiments\hiwa_python_reproduction\scripts\run_rotation_stabilized.py`
- `D:\ai\数学ai\Projects\最优传输\Experiments\hiwa_python_reproduction\scripts\run_rgca_roca.py`
- `D:\ai\数学ai\Projects\最优传输\Experiments\hiwa_python_reproduction\tests\test_soft_hiwa.py`

## 新增笔记

- `D:\ai\数学ai\Projects\最优传输\最优传输\文本笔记\代码方面\神经数据完整TACO-SoftHiWA流程审计与纠偏-2026-07-10.md`
- `D:\ai\数学ai\Projects\最优传输\最优传输\文本笔记\代码方面\代表元直接参与旋转更新实验笔记-2026-07-10.md`

## 新增/复制图片

- `D:\ai\数学ai\Projects\最优传输\最优传输\文本笔记\代码方面\_attachments\rgca_roca_summary_neural_rep_rotation_smoke_2026_07_10.png`
- `D:\ai\数学ai\Projects\最优传输\最优传输\文本笔记\代码方面\_attachments\rgca_roca_summary_neural_rep_rotation_tiny_smoke_2026_07_10.png`

## 关键决定

1. PBMC RNA-ATAC 目前降级为第二数据集迁移前的诊断性预研，不作为当前主方法结果。
2. 神经数据主线先补完整 TACO/Soft-HiWA 流程。
3. 新增 `representative_rotation_weight`，与旧的 `representative_guidance_weight` 分离。
4. 默认新参数为 0，保证旧 ROCA / Soft-HiWA 结果可复现。
5. 当前不把 representative-guided optimization 写成已提升精度，只写成已补齐、可消融、当前证据谨慎。

## 验证

运行：

```bash
python -m py_compile scripts\soft_hiwa.py scripts\run_soft_neural.py scripts\run_rotation_stabilized.py scripts\run_rgca_roca.py
```

通过。

运行：

```bash
python -m unittest discover -s tests -v
```

结果：

```text
Ran 38 tests
OK
```

## 神经 smoke 实验

运行：

```bash
python scripts\run_rgca_roca.py --profile quick --seeds 50 51 --guidance-weights 0.0 0.01 --rotation-weights 0.0 0.01 --tag neural_rep_rotation_smoke_2026_07_10
```

运行：

```bash
python scripts\run_rgca_roca.py --profile quick --seeds 50 51 --guidance-weights 0.0 --rotation-weights 0.0 0.001 0.003 --tag neural_rep_rotation_tiny_smoke_2026_07_10
```

关键结果：

- baseline $\lambda_T=0,\lambda_R=0$：mean accuracy 0.592295，mean $R^2$ 0.601457。
- $\lambda_R=0.001$：mean accuracy 不变，mean $R^2$ 仅下降约 0.000082。
- $\lambda_R=0.003$：mean accuracy 下降约 0.002408。
- $\lambda_R=0.01$：mean accuracy 下降约 0.009631。
- $\lambda_T=0.01$ 在 seed 50 出现灾难性失败，selected $R^2$ 约 -0.804。

## 风险或跳过事项

- 当前只做了 smoke，不是完整 seeds 50-69。
- 尚未诊断 seed 50 中 $\lambda_T=0.01$ 的失败机制。
- 不能声称代表元进入优化提高了精度。
- 目前最强、最稳的贡献仍是 ROCA 的无标签 determinant branch selection。

## 下一步建议

1. 固定 $\lambda_T=0$，在 seeds 50-59 检查 $\lambda_R \in \{0,0.0005,0.001,0.002\}$。
2. 单独诊断 seed 50 的 transport guidance 失败。
3. 论文主线仍以 ROCA 为核心，representative-guided optimization 暂放扩展/消融。

