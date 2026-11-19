# Hyser PCA / Factor Analysis: five-trial independent dimensionality-reduction experiment

原HiWA论文的NHP实验确实使用FA把神经放电数据降为3D，再与3D运动目标对齐。
此处既在EMG源域，也在真实五根手指力量目标域单独比较PCA/FA。
5折完整试次留出，每折1000训练/250测试；标准化、PCA、FA、白化全部只用4训练试次拟合。
FA用独立噪声模型，PCA最大化几何方差，二者目的不同。
不可给FA编造PCA解释方差指标；应比较同一折的重建质量、heldout高斯似然、物理通道恢复、聚类稳定性。

## Fivefold means: global test standardized reconstruction R²

| Domain | d | PCA | FA | ΔFA−PCA |
|---|---:|---:|---:|---:|
| source_EMG | 3 | 0.87177 | 0.85182 | -0.01995 |
| source_EMG | 5 | 0.90806 | 0.89498 | -0.01308 |
| source_EMG | 10 | 0.96230 | 0.94897 | -0.01333 |
| target_fivefinger | 3 | 0.71447 | 0.58053 | -0.13394 |
| target_fivefinger | 4 | 0.89401 | 0.62780 | -0.26621 |
| target_fivefinger | 5 | 1.00000 | 0.41967 | -0.58033 |

## Heldout paired-supervised source information probe (not OT training)
Five-finger physical force from EMG latent via TRAINING trial-only Ridge (fixed alpha=10).
| EMG dimension | PCA heldout force R² | FA heldout force R² | Δ FA−PCA |
|---:|---:|---:|---:|
| 3 | 0.12287 | 0.13635 | +0.01347 |
| 5 | 0.16813 | 0.23608 | +0.06795 |
| 10 | 0.38224 | 0.48079 | +0.09855 |

## Target five-finger reconstruction
A reduction of Y from five physical fingers to 3D or 4D loses information by definition.
The channel-by-channel heldout R² lives in physical_fivefinger_force_reconstruction_per_fold.csv.
For OT, the 5D target is retained losslessly up to a train-fitted full-rank whitening.

## OT input configurations
- PCA3_PCA3: source=PCA(3D), target=PCA(3D)
- FA3_PCA3: source=FA(3D), target=PCA(3D)
- PCA3_FA3: source=PCA(3D), target=FA(3D)
- FA3_FA3: source=FA(3D), target=FA(3D)
- PCA5_RAW5: source=PCA(5D), target=RAW(5D)
- FA5_RAW5: source=FA(5D), target=RAW(5D)

OT comparison must remain INSIDE each representation because alternative FA/PCA feature spaces have different geometry.
The source Ridge probe is SUPERVISED and MUST NOT be used to tune, train, label, or compare unsupervised OT.
The old 3D experiment used physical thumb,middle,index, whereas new PCA3/FA3 target uses all five fingers: do not mix scores.

