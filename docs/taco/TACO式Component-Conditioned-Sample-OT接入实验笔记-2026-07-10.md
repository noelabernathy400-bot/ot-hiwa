---
title: TACO式 Component-Conditioned Sample OT 接入实验笔记
date: 2026-07-10
tags:
  - HiWA
  - TACO
  - Soft-HiWA
  - Component-Conditioned-OT
  - ROCA-HiWA
---

# TACO式 Component-Conditioned Sample OT 接入实验笔记

## 1. 这次补的不是“新增 sample OT”

需要先澄清一件事：HiWA / Soft-HiWA 本来就有 sample-level OT，也本来有 group-level OT。

所以这次不是凭空加入一个新的 sample OT，而是把已有的局部样本运输：

$$
T^{kl}
=
\operatorname{OT}
\left(
\|R_{kl}x-y\|^2
\right)
$$

改成受 soft component compatibility 调制的样本运输：

$$
T^{kl}
=
\operatorname{OT}
\left(
C_{\mathrm{geom}}^{kl}
+
\beta C_{\mathrm{comp}}^{kl}
\right).
$$

这才是更接近 TACO 后半部分的地方。

## 2. 新增数学项

对每个 source 样本 $x_n$ 和 target 样本 $y_m$，根据 soft assignment 和 group transport 计算 compatibility：

$$
s_{nm}
=
a_n^X P (a_m^Y)^\top.
$$

其中：

- $a_n^X$ 是 source 样本对所有 soft groups 的归属概率；
- $a_m^Y$ 是 target 样本对所有 soft groups 的归属概率；
- $P$ 是 group-level transport。

如果两个样本所属的 soft groups 通过 $P$ 更兼容，则 $s_{nm}$ 更大。

构造 component penalty：

$$
C_{nm}^{\mathrm{comp}}
=
-\log(s_{nm}+\epsilon).
$$

为了避免尺度过大，代码中对该矩阵做了 unit-scale normalization，然后加入局部样本 OT cost：

$$
C_{nm}^{\mathrm{total}}
=
\|R_{kl}x_n-y_m\|^2
+
\beta \widetilde C_{nm}^{\mathrm{comp}}.
$$

## 3. 代码改动

修改文件：

- `Experiments/hiwa_python_reproduction/scripts/soft_hiwa.py`
- `Experiments/hiwa_python_reproduction/scripts/run_soft_neural.py`
- `Experiments/hiwa_python_reproduction/scripts/run_rotation_stabilized.py`
- `Experiments/hiwa_python_reproduction/scripts/run_rgca_roca.py`
- `Experiments/hiwa_python_reproduction/tests/test_soft_hiwa.py`

新增参数：

```text
component_conditioning_weight
```

命令行参数：

```text
--component-weights
```

默认：

```text
component_conditioning_weight = 0.0
```

因此旧算法可以严格退化回来。

## 4. 与之前两个代表元模块的关系

| 模块 | 参数 | 控制对象 | 当前结论 |
|---|---|---|---|
| representative group transport guidance | $\lambda_T$ | group transport $P$ | seed 50 可灾难性失败 |
| representative rotation guidance | $\lambda_R$ | global rotation $R$ | 可控但未提升 |
| component-conditioned sample OT | $\beta$ | 局部样本 OT cost | 本次补上，R² 有时升，accuracy 下降 |
| ROCA selector | 无标签有向体积 | determinant branch | 目前最稳贡献 |

## 5. 验证

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

新增测试检查：

- $\beta<0$ 会报错；
- $\beta=0$ 退化为旧算法；
- $\beta>0$ 时 component conditioning diagnostics 存在；
- 加入 extra cost 后 Sinkhorn 仍满足边际约束。

## 6. 神经数据 smoke 设置

只开 component-conditioned sample OT，不同时开前面两个代表元优化项：

```bash
python scripts\run_rgca_roca.py --profile quick --seeds 50 51 --guidance-weights 0.0 --rotation-weights 0.0 --component-weights 0.0 0.001 0.003 0.01 --tag neural_component_conditioned_sample_ot_smoke_2026_07_10
```

随后做更小权重检查：

```bash
python scripts\run_rgca_roca.py --profile quick --seeds 50 51 --guidance-weights 0.0 --rotation-weights 0.0 --component-weights 0.0 0.0001 0.0003 --tag neural_component_conditioned_sample_ot_tiny_smoke_2026_07_10
```

## 7. 结果

### 7.1 较大 beta smoke

![[rgca_roca_summary_neural_component_conditioned_sample_ot_smoke_2026_07_10.png]]

| $\beta$ | Mean direction accuracy | 相对 baseline | Mean movement $R^2$ | 相对 baseline | ROCA 选中高 accuracy 分支 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.592295 | 0.000000 | 0.601457 | 0.000000 | 2/2 |
| 0.001 | 0.577849 | -0.014446 | 0.605532 | +0.004075 | 2/2 |
| 0.003 | 0.564205 | -0.028090 | 0.605396 | +0.003939 | 2/2 |
| 0.010 | 0.569021 | -0.023274 | 0.602562 | +0.001105 | 2/2 |

解释：

- $\beta=0.001$ 到 $0.003$ 会提高 movement $R^2$；
- 但 direction accuracy 明显下降；
- ROCA 仍然能选中两个候选中 accuracy 更高的分支；
- 这说明分支选择没坏，主要是样本 OT / 连续几何和离散方向语义之间出现 trade-off。

### 7.2 极小 beta smoke

![[rgca_roca_summary_neural_component_conditioned_sample_ot_tiny_smoke_2026_07_10.png]]

| $\beta$ | Mean direction accuracy | 相对 baseline | Mean movement $R^2$ | 相对 baseline |
|---:|---:|---:|---:|---:|
| 0 | 0.592295 | 0.000000 | 0.601457 | 0.000000 |
| 0.0001 | 0.592295 | 0.000000 | 0.601405 | -0.000052 |
| 0.0003 | 0.592295 | 0.000000 | 0.601310 | -0.000147 |

解释：

- 极小 $\beta$ 基本不改变 direction accuracy；
- 但也没有提升 $R^2$；
- 真正能改变 $R^2$ 的 $\beta$ 又会降低 accuracy。

## 8. 当前判断

这次接入是必要的，因为它补上了 TACO 后半部分中最像“component-conditioned sample OT”的机制。

但是结果不支持把它写成当前神经数据上的性能提升来源。

更准确的结论是：

> Component-conditioned sample OT 在当前猕猴神经—运动数据上可以改变连续运动几何对齐，甚至略微提高 $R^2$，但会牺牲方向分类语义。因此它目前更像一个揭示 trade-off 的消融模块，而不是主贡献。

## 9. 对论文主线的影响

当前主线应该收敛为：

1. HiWA 已经有较强的连续几何对齐；
2. TACO 式 soft components / representatives 可以提供结构锚点；
3. 这些结构锚点最稳定的用途是 ROCA 的无标签 determinant branch selection；
4. 把 TACO 后半部分直接放入优化过程并不一定提升当前数据；
5. 这说明我们的方法贡献应强调“稳定性与可解释分支选择”，而不是吹成“所有指标全面提升”。

## 10. 下一步

建议暂时不要继续大规模扫 $\beta$。

更值得做的是：

1. 诊断为什么 $\beta=0.001$ 提高 $R^2$ 但降低 accuracy；
2. 检查是否方向标签语义与连续轨迹几何本来就是分离的；
3. 把 component-conditioned sample OT 保留为论文消融；
4. 下一步转向自动 $K$ / simplex voting / confidence rejection，这些更贴近“一般方法”的构建。

