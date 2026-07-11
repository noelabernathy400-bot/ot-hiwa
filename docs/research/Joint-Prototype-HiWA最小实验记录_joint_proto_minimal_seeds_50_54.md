# Joint-Prototype HiWA 最小实验记录

## 实验定位

本实验只补齐 TACO-inspired HiWA 主线中的最小联合优化环节：固定神经与运动 3D embedding，固定一个由 frozen ROCA 无标签选择得到的 rotation，只优化 soft prototypes 及其诱导的 assignments。

不使用 direction label 训练，不使用 direction accuracy 或 $R^2$ 选择超参数，不扩展 PBMC / 新数据集 / encoder。

## 目标函数

$$
\mathcal L = \mathcal L_{proto}^X + \mathcal L_{proto}^Y + \lambda_{align}\mathcal L_{align} + \lambda_{bal}\mathcal L_{balance} + \lambda_{ent}\mathcal L_{entropy}
$$

其中 assignment 在各自标准化 embedding 空间由 prototypes 产生；alignment representative 使用原始固定 3D embedding 的加权均值。

## 参数

```json
{
  "seeds": [
    50,
    51,
    52,
    53,
    54
  ],
  "profile": "quick",
  "profile_parameters": {
    "maxiter": 12,
    "tol": 0.01,
    "mu": 0.02,
    "shorn_maxiter": 100,
    "sa_maxiter": 12,
    "sa_shorn_maxiter": 30
  },
  "annealing_path": [
    0.25,
    0.35,
    0.5
  ],
  "groups": 4,
  "entropy_weight": 0.05,
  "retain_mass": 0.9,
  "max_support_factor": 1.5,
  "anchor_weight": 0.0,
  "joint_config": {
    "n_groups": 4,
    "temperature": 0.5,
    "lambda_align": 0.1,
    "lambda_balance": 0.05,
    "lambda_entropy": 0.1,
    "group_gamma": 0.1,
    "sinkhorn_maxiter": 200,
    "maxiter": 80,
    "gtol": 1e-05
  }
}
```

## 汇总

| 指标 | Frozen ROCA | Joint prototypes + ROCA | Delta |
|---|---:|---:|---:|
| mean direction accuracy | 0.5743 | 0.5740 | -0.0003 |
| mean movement R2 | 0.6008 | 0.5839 | -0.0169 |
| mean objective delta |  |  | -0.014783 |

## 每个 seed

| seed | frozen det | joint det | frozen acc | joint acc | frozen R2 | joint R2 | objective initial | objective final |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | -1 | -1 | 0.5746 | 0.5682 | 0.6008 | 0.5967 | 0.497721 | 0.482932 |
| 51 | -1 | -1 | 0.5730 | 0.5650 | 0.6052 | 0.5825 | 0.496860 | 0.482122 |
| 52 | -1 | -1 | 0.5698 | 0.5971 | 0.5976 | 0.5796 | 0.497623 | 0.482825 |
| 53 | -1 | -1 | 0.5795 | 0.5843 | 0.5941 | 0.5746 | 0.498471 | 0.483604 |
| 54 | -1 | -1 | 0.5746 | 0.5554 | 0.6063 | 0.5859 | 0.496697 | 0.481976 |

## 初步判断

- 若 joint objective 稳定下降但 final evaluation 不升，说明 alignment-aware prototypes 在当前固定 $R$ 下学到了目标函数，但该目标还未足以改善 sample-level OT。
- 若 assignment entropy 或 group mass 出现塌缩，需要优先调整 balance/entropy 约束，而不是扩大数据集或引入 encoder。
- 本实验是最小机制验证，不应被解释为完整 TACO 迁移完成。
