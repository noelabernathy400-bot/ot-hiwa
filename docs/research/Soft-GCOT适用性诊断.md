# Soft-GCOT 适用性诊断

日期：2026-07-11
范围：仅诊断既有 Hard HiWA、TACO-faithful full-support Soft-GCOT、sparse approximation 与 ROCA；没有修改模型、损失、模块或超参数。

## 结论

1. **当前不能用“Soft-GCOT 不如 Hard HiWA”作为收敛后的结论。** 在固定的充分迭代诊断中，Soft-GCOT 的全局旋转变化变小，但 ADMM primal consensus residual 没有收敛；Hard HiWA 在同一预算下更常达到其已有停止准则。因此之前的 accuracy 差异混合了“不同目标/表示”与“Soft-GCOT 未完成 ADMM 共识”，不能归因为 soft grouping 本身无效。
2. **soft assignment 不过软，也没有 prototype collapse。** 两域 assignment entropy 为最大熵的约 38% 与 41%，组质量没有空组，原型和 representative 间距离均远离零。
3. **sparse-support 失败的直接机制是截断改变了 soft empirical measures。** 每组仅保留约 26--36/96 个样本；运动域两组实际只保留 84.6% 与 85.7% 质量（虽请求 90%，但受到支持大小上限），随后重新归一化。因此每一个相关 $Q_{ij}$ 都在不同的边缘测度上求解；对连续软边界样本丰富的任务，这不是无害加速。
4. **ROCA 的问题更符合 A（当前 branch selection / branch-optimization 不相容），而非 B（representative geometry 退化）。** seed 0 的 simplex 体积、条件数均良好；但 ROCA 选中的 det$=-1$ 分支在 80 轮后 ADMM residual 为 2.828，另一分支为 0.643。该选择没有提供一个已达共识的候选变换。严格说，在不使用标签或外部真值的前提下，不能把它称为“语义真值上的错误 branch”；这里的 A 是可观测的优化意义：被选 branch 不满足当前 HiWA consensus 条件。

## 诊断协议

- 数据：MiHiA 神经--运动数据的固定、等距抽取 96-by-96 子问题；这是为了让 full-support $Q_{ij}$ 的 3-seed、80-outer-iteration 检查可运行。
- seeds：0、1、2。
- 共同设置：4 个组，temperature=1.0，entropy weight=0.05，full support，outer maxiter=80，inner Sinkhorn maxiter=80，group Sinkhorn maxiter=300。
- 不读方向标签、不计算 accuracy 或 $R^2$、不以标签选择 branch 或参数。
- Hard 使用同一无监督 prototype 的 argmax 组；Soft 使用连续 assignments。

## 一、收敛与外层目标

### 80 轮终点（多 seed）

| seed | Hard iterations / final global residual | Soft iterations / global residual | Soft ADMM primal residual | Soft max Sinkhorn marginal error | Soft transport objective |
|---:|---:|---:|---:|---:|---:|
| 0 | 55 / 0.00990 | 80 / 0.02471 | 0.66553 | $1.52\times10^{-14}$ | 0.84841 |
| 1 | 77 / 0.00378 | 80 / 0.04924 | 0.52609 | $9.74\times10^{-15}$ | 0.84304 |
| 2 | 80 / 0.68150 | 80 / 0.24792 | 0.58567 | $9.79\times10^{-15}$ | 0.84306 |

解释：Sinkhorn 的数值边缘约束是充分满足的；故问题不是 OT marginal drift。Soft 的 global rotation residual 已比早期小，但其局部 $R_{ij}$ 尚未与全局 $R$ 达成 ADMM 共识。按当前 $10^{-2}$ 阈值，三个 Soft seed 都不应标为收敛。Hard 的实现只有全局旋转停止准则，seed 0、1 达到该阈值，seed 2 未达到；这与 Soft 的“双残差”验收不对称，不能把两者的 `converged` 标记当成同一强度的收敛证据。

### Soft outer-iteration objective checkpoints（seed 0）

下表通过固定随机流、分别运行到不同 outer budget 得到；这是同一确定性迭代轨迹的检查点，而非用标签选择的曲线。

| outer budget | Soft global residual | Soft ADMM primal residual | Soft objective |
|---:|---:|---:|---:|
| 6 | 2.72147 | 2.77947 | 0.95576 |
| 20 | 2.00000 | 2.00005 | 0.80408 |
| 40 | 2.00000 | 2.00001 | 0.85100 |
| 80 | 0.02471 | 0.66553 | 0.84841 |

该目标先降后回升/平台化，不是单调下降量；这与带局部旋转、$P$、$Q_{ij}$ 的交替 ADMM 过程一致。Hard 的现有 diagnostics 没有逐外层保存 `C` 与 objective（非终止时 `C` 未写回 diagnostics），故在“不修改算法”的约束下只能报告其 residual，而不能诚实地补造 Hard objective curve。

## 二、soft grouping 是否有结构

seed 0，最大熵 $\log 4=1.38629$：

| 指标 | Neural | Movement |
|---|---:|---:|
| Mean assignment entropy | 0.53147 | 0.56889 |
| Normalized entropy $H/\log4$ | 0.3834 | 0.4104 |
| Soft group masses $\alpha$ | [0.2766, 0.2672, 0.2195, 0.2366] | [0.2237, 0.2270, 0.3469, 0.2024] |
| Prototype pairwise distance, min / mean | 2.7515 / 2.9100 | 22.4466 / 26.2577 |
| Representative pairwise distance, min / mean | 1.7437 / 2.0003 | 12.6881 / 17.9117 |

判断：softness 是中等而非接近均匀；质量都在 0.20 以上；最小原型和 representative separation 均明显非零。因此没有 prototype collapse，也没有“所有样本对所有组几乎相同”的证据。Soft grouping 已形成可分结构；没有提升 accuracy 的更直接候选原因是该结构进入 HiWA 的全体 $Q_{ij}$ 后，局部旋转之间仍难以形成全局共识。

## 三、sparse approximation 为什么失败

在 seed 0、默认 `retain_mass=0.90, max_support_factor=1.5` 下：

| 域 | 每组 support size | 每组 retained mass |
|---|---|---|
| Neural | [32, 31, 31, 34] | [0.9046, 0.9071, 0.9037, 0.9026] |
| Movement | [35, 26, 36, 36] | [0.9039, 0.9062, 0.8456, 0.8565] |

full-support $Q_{ij}$ 的边缘是全部 96 个样本的 assignment column。sparse 版本删除 9--15% 的归属质量后再重新归一化，尤其 target groups 3、4 因 support cap 不能达到请求的 90%。这会同时改变局部 OT cost、对应的 $R_{ij}$，以及最终 ADMM vote。结论不是“soft method 必然脆弱”，而是该 sparse rule 对本任务不是忠实替代；任何性能解释都应以 full-support 为准。

## 四、ROCA 理论条件与失败定位

seed 0、soft representatives、两条 determinant 分支均运行 80 轮：

| 指标 | 数值 |
|---|---:|
| Source oriented simplex volume | -6.02519 |
| Target matched simplex volume | 4.44633 |
| Absolute volume product margin | 26.78999 |
| Source / target condition number | 2.3335 / 3.3596 |
| ROCA selected determinant | -1 |
| Assignment-entropy warning | 是 |
| Selected branch ADMM residual | 2.82784 |
| Other branch ADMM residual | 0.64251 |
| Selected / other transport objective | 0.81621 / 0.84363 |

体积远离零、条件数接近 1--4，故 representative simplex 本身满足“非退化、可定向”的几何前提，排除 B 为主要解释。warning 的来源是 raw entropy threshold 0.45；本实验的 entropy 约 0.53--0.57，属于中等软分配，但该旧阈值把它列为 warning。这一 warning 应被视作“当前 selector 的适用条件未满足”，而非几何退化证据。

ROCA 选择的分支表面 transport objective 更低，却有远高于另一分支的 ADMM residual。另一分支的 residual 0.64251 也仍高于 $10^{-2}$，因此它不是“已经收敛的正确分支”；它只能说明被选分支的 consensus 更不稳定。由于 objective 中未把“尚未共识”作为独立验收条件，不能把这个较低数值解释为更好的有效对齐。因此当前 ROCA 的实用失败是：orientation selector 给出的 O(3) 分支没有同时通过 Soft-GCOT/HiWA 的 consensus 可行性检查。它不是 simplex volume 或 condition number 的失败。

## 建议：改变使用条件，不先改算法

在不改算法树的前提下，下一步应先改变**验收与使用条件**：

1. 对 Soft-GCOT 报告方向 accuracy 前，要求 global residual 与 ADMM primal residual 均达到预注册阈值；未收敛 seed 单列为 optimizer failure，不与 Hard 直接平均。
2. full-support 是当前唯一的 Soft-GCOT 基准；sparse 只能作为近似误差/运行时对照。
3. ROCA 只在其选定 branch 也达到 Soft-GCOT consensus、且 warning 未触发时才可进入最终比较；否则报告为“不适用/低置信”，而不是强行输出对齐结果。
4. 在这些使用条件成立之前，不应依据现有 accuracy 扩展或重写算法；先完成多 seed 的充分收敛诊断。
