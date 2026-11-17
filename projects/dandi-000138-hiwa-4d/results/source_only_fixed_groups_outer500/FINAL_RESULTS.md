**实验类型：组级监督诊断（NOT UNSUPERVISED）**。目标训练标签或软隶属度从同步配对的源域标签传播，不能当作严格无监督 Hard HiWA / Soft-GCOT 基线；数值、图像及其他参数保留为历史诊断。见 [专项审计](https://github.com/noelabernathy400-bot/ot-hiwa/blob/main/projects/dandi-000138-hiwa-4d/GROUP_LABEL_LEAKAGE_AUDIT.md)。

# 最终审计结果：同一来源 KMeans、固定组 Hard HiWA，outer≤500

所有组标签严格复用 results/source_only_fixed_groups/fixed_groups_4d.npz；
源域只执行过一次 KMeans，目标依同步神经样本继承源标签；
目标域没有单独 KMeans，也未在变换后重新聚类。

## 测试结果

| 项目 | 未对齐 | HiWA 五次均值 ± 样本 SD |
|---|---:|---:|
| one_nn_shared_group | 0.231429 | 0.377429 ± 0.216217 |
| MAE | 0.531486 | 0.497667 ± 0.055478 |
| paired_relative_MSE | 1.792656 | 1.603956 ± 0.302491 |
| rMSE_rotation_proxy | 1.964698 | 1.492680 ± 0.786639 |
| R2 | -0.796486 | -0.607383 ± 0.303137 |

## 收敛审计

| seed | 最终 global change | 最大 local-global residual |
|---:|---:|---:|
| 42 | 0.000926888 | 0.000385662 |
| 43 | 3.23022e-06 | 1.36379e-06 |
| 44 | 6.77051e-05 | 4.05166e-05 |
| 45 | 1.42206e-05 | 1.31074e-05 |
| 46 | 0.000245451 | 0.000137132 |

全部五次均符合 global<=0.01、primal<=0.01、簇间质量边缘差<1e-5、正交误差<1e-7。
尽管五次均在 500 次外循环时达到审计阈值，这不能证明全局最优或统计显著性。

**科学限制：**目标簇标签通过真实同步配对继承，所以该实验不能称为完全无配对或无监督分组。
作者原版真实旋转 Rgt 未知，所谓旋转 rMSE 为训练配对 Procrustes 代理值。
1NN 比较共同继承标签，并不是独立真实运动类别的准确率。
