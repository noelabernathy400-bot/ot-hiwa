# 2026-07-10 TACO式 Component-Conditioned Sample OT 接入报告

## 目标

在猕猴神经元—运动数据的 Soft-HiWA 主线中，接入 TACO 后半部分更核心的 soft component-conditioned sample OT 机制。此次不是新增 sample OT，而是把已有局部样本运输的 cost 改为几何代价加 soft component compatibility penalty。

## 改动文件

- `D:\ai\数学ai\Projects\最优传输\Experiments\hiwa_python_reproduction\scripts\soft_hiwa.py`
- `D:\ai\数学ai\Projects\最优传输\Experiments\hiwa_python_reproduction\scripts\run_soft_neural.py`
- `D:\ai\数学ai\Projects\最优传输\Experiments\hiwa_python_reproduction\scripts\run_rotation_stabilized.py`
- `D:\ai\数学ai\Projects\最优传输\Experiments\hiwa_python_reproduction\scripts\run_rgca_roca.py`
- `D:\ai\数学ai\Projects\最优传输\Experiments\hiwa_python_reproduction\tests\test_soft_hiwa.py`

## 新增笔记

- `D:\ai\数学ai\Projects\最优传输\最优传输\文本笔记\代码方面\TACO式Component-Conditioned-Sample-OT接入实验笔记-2026-07-10.md`

## 新增图片副本

- `D:\ai\数学ai\Projects\最优传输\最优传输\文本笔记\代码方面\_attachments\rgca_roca_summary_neural_component_conditioned_sample_ot_smoke_2026_07_10.png`
- `D:\ai\数学ai\Projects\最优传输\最优传输\文本笔记\代码方面\_attachments\rgca_roca_summary_neural_component_conditioned_sample_ot_tiny_smoke_2026_07_10.png`

## 数学实现

新增参数：

```text
component_conditioning_weight
```

记为 $\beta$。对局部样本 OT，计算：

$$
s_{nm}=a_n^XP(a_m^Y)^\top
$$

并加入：

$$
C_{nm}^{total}=C_{nm}^{geom}+\beta\widetilde{-\log(s_{nm}+\epsilon)}.
$$

默认 $\beta=0$，严格退化为旧算法。

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
Ran 42 tests
OK
```

## smoke 实验

运行：

```bash
python scripts\run_rgca_roca.py --profile quick --seeds 50 51 --guidance-weights 0.0 --rotation-weights 0.0 --component-weights 0.0 0.001 0.003 0.01 --tag neural_component_conditioned_sample_ot_smoke_2026_07_10
```

运行：

```bash
python scripts\run_rgca_roca.py --profile quick --seeds 50 51 --guidance-weights 0.0 --rotation-weights 0.0 --component-weights 0.0 0.0001 0.0003 --tag neural_component_conditioned_sample_ot_tiny_smoke_2026_07_10
```

## 关键结果

- Baseline $\beta=0$：mean accuracy 0.592295，mean $R^2$ 0.601457。
- $\beta=0.001$：mean accuracy 0.577849，mean $R^2$ 0.605532。
- $\beta=0.003$：mean accuracy 0.564205，mean $R^2$ 0.605396。
- $\beta=0.01$：mean accuracy 0.569021，mean $R^2$ 0.602562。
- 极小 $\beta=0.0001,0.0003$ 基本不改变 accuracy，但也不提升 $R^2$。

## 当前判断

Component-conditioned sample OT 已经接入现有 HiWA 交替优化流程，不是外部简化模型。它在当前数据上没有提升 direction accuracy；中等 $\beta$ 可提高连续 movement $R^2$，但会牺牲方向语义。

因此它应作为 TACO 思想完整性和 trade-off 消融保留，不应作为当前主贡献。

## 下一步建议

1. 暂停大规模扫 $\beta$。
2. 保留该模块作为可开关组件。
3. 下一步重点转向自动 $K$、simplex voting、confidence rejection，以支撑一般方法。

