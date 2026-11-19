# HiWA 原作者源码审计、代码修复与完整 P/Q/R 复算结论

**日期：2026-11-19。所有下列实验已实际运行并上传代码和数字。原始已归档数据及随机初值求解结果未被覆盖。**

## 最终结论

1. **核心算法没有发现矩阵公式错误。** 按作者MATLAB原文独立逐步实现一个外层迭代，R、P、局部R、代价及ADMM乘子与我们的ZIP求解器最大数值差异只有约1e-15。再从Git历史提取修改前的Python求解器，用相同输入与随机种子检查所有P/R/16个局部Q/乘子/迭代历史：修复前后默认模式**完全相同，差异0**。
2. **确定存在代码缺失，并已修复。** 原GCOT/TPAMIGCOT设置projections参数时导入不存在的zip_gcot.preprocessing，会触发ModuleNotFoundError；过去未开启投影的正式实验不受这处错误影响，但无法跑原论文式几何结构。已新增完整独立投影模块，分别支持源/目标训练集拟合的簇子空间投影。
3. **之前的四维四簇合成对照过于不同于作者的真实设置。** 作者Fig1cd在高维情形使用S=5、子空间内在维数d=2、环境维数D=6；我们以前在D=4用4个满维簇，并且没有子空间结构。这样的几何可能有多个匹配解，不能凭失败推论矩阵公式写错。
4. **新增训练域边际的几何初始化，修复了原四维合成实验的优化不稳定问题。** 不使用真实组或配对，仅从两个训练域各自的协方差主轴生成16个符号旋转候选，用无配对训练OT选择初值，再完全优化P、Q、R。过去失败的清晰四簇实验，现在3/3初始化下真实组1NN正确率100%，P正确组质量均值87.5%；重叠四簇对应约99.1%正确率、P正确质量80.7%。该改进是新优化策略，不能宣称原公式本身有bug。
5. **新求解策略没有让真实DANDI突然成功。** 使用原样本层级、原P/Q/R及OT预算，新增rank2或rank3独立组投影与warm start，对3500点做真实跨试次运动位置和速度检索；其物理准确率仍比恒等R差。此时不应把合成成功泛化成真实神经—运动解码的成功。

## 数学源码的实际对照

| 更新/模块 | 原作者 MATLAB HiWA.m | 现有ZIP Python实现 | 审计 |
|---|---|---|---|
| 局部旋转Rij | polar(2Pij Yj Qij^T Xi^T+mu/D(Rg-Lij)) | 行向量转置等价式 | 一致 |
| 局部运输Qij | Sinkhorn，温度epsilon_sample/Pij | 相同温度及局部边际 | 一致 |
| 簇间运输P | 在局部cost矩阵上Sinkhorn | 相同均匀组边际 | 一致 |
| 全局R及ADMM乘子 | mean(Rij+Lij)极分解；Lij+=Rij-Rg | 相同 | 一致 |
| 各簇低维子空间 | AiAi^T Xi，BjBj^T Yj | 原4D为完整原坐标，没有投影 | 实际差异，已新增可选实现 |
| 参数 | Fig1cd outer300、inner100、sample Sinkhorn150、group Sinkhorn1000、mu=.005、停止tol=.1 | 原DANDI outer500、inner20、sample45、group200、mu=.05、停止tol=1e-9 | 实验预算不同，不是公式BUG |

[作者 MATLAB HiWA.m 原始来源](https://github.com/siplab-gt/hiwa-matlab/blob/master/code/toolbox/HiWA.m)；
[作者 Fig1cd 原实验参数](https://github.com/siplab-gt/hiwa-matlab/blob/master/code/figure1cd.m)；
[本项目对照及全部12次实验](results/author_matlab_code_audit/HIWA_AUTHOR_SOURCE_AUDIT_REPORT.md)。

另一个独立诊断问题：原作者Python的hiwa.py内部_eval_R2使用未平方的平均残差，不是标准R²公式；原作者Notebook最后使用的是另一个utils.eval_R2（平方残差，但会分别协方差白化），不等于我们原始物理目标空间的R²。原严格研究本来就使用标准平方同步误差，因此这处作者内部诊断问题没有导致我们当前训练失败。

## 原作者式正向对照，确实跑通

生成三组独立实例，每组五个二维内在子空间嵌入六维，打乱目标组编号，目标与源独立抽样，无样本对应。采用**已知域内簇但未知组间对应P**的作者半监督分组正向对照（这与真实DANDI严格独立KMeans分组不是同一种难度）。

比较原DANDI预算/原作者Fig1cd预算，以及无投影/训练簇PCA2投影，得到3×2×2=12个完整P/Q/R结果：

- 全部12次真实对应组P质量均高于99.8%；
- 全部12次R相对于真实正交R*的误差均很低（相对均方误差约0.00038—0.0100）；
- 有无投影都成功，这说明**不能简单说以前失败仅仅因为没有子空间投影**；数据结构更关键。

[完整报告和逐种子数据](results/author_matlab_code_audit/HIWA_AUTHOR_SOURCE_AUDIT_REPORT.md)。

## 旧四维合成实验：通过新初始化得到实际改善

| 条件 | 原随机Hard正确类别1NN | 新train-only初始化+完整Hard P/Q/R |
|---|---:|---:|
| 4D四个清晰簇 | 25.0%（三种子平均） | 100.0%（3/3） |
| 4D四个重叠簇 | 25.2%（三种子平均） | 约99.1%（3/3） |

新Hard的P正确组质量清晰约0.875、重叠约0.807；3/3通过ADMM。核心计算仍然是原HiWA P/Q/R，只改变初始R。

[修复前后默认输出逐位一致及暖启动完整正向对照](results/hiwa_source_code_repair/HIWA_CORRECTED_INITIALIZATION_REPORT.md)。

## 真实DANDI新旧对照：物理匹配仍未解决

训练/测试为原400/100个trial，14000/3500点，每次160点/域，K=4；所有方法真实Hard/Soft分组各域独立；新模型完整学习P/Q/R，使用原500/20/45/200预算。用原相同种子42、43、44作公平比较；从测试目标候选点云排除同一试次候选，以原先训练集确定的实际位置行程10%和速度方向30°评估。

| 方法，种子42—44 | 位置容差命中率 | 速度方向命中率 |
|---|---:|---:|
| 恒等R + 最近邻 | 9.69% | 20.29% |
| 旧Hard随机初始化 | 7.25% | 16.35% |
| 旧Soft随机初始化 | 2.22% | 13.95% |
| 新Hard谱初始化，完整4D | 3.88% | 12.13% |
| 新Hard谱初始化，组rank2投影 | 3.72% | 12.35% |
| 新Hard谱初始化，组rank3投影 | 3.78% | 12.13% |
| 新Soft谱初始化，完整4D | 6.76% | 12.49% |

Hard三种新变体严格ADMM均3/3通过；Soft新变体0/3通过。新的无配对W2代价在一些方案下降，却没有换来真实物理运动匹配改善，表明**优化所用的分布几何与真实同步运动含义没有自动一致**。

[真实DANDI完整P/Q/R新旧对照](results/hiwa_repaired_full_pqr_dandi/HIWA_DANDI_REAL_SOURCE_CODE_CORRECTION_REPORT.md)；
[逐种子真实评分及所有P、R](results/hiwa_repaired_full_pqr_dandi/HIWA_DANDI_REAL_SOURCE_CODE_CORRECTION_RESULTS.json)；
[实际16组Q和采样索引](results/hiwa_repaired_full_pqr_dandi/hard_warm_full_seed42_COMPLETE_P_Q_R.npz)。

注意旧的100测试trial此前已反复检查，所以这次属于探索性验证，不能利用这些分数继续选择模型参数后又宣称它们是全新独立测试。

## 代码改动与向后兼容

- 新增 [zip_gcot/preprocessing.py](code/zip_gcot/preprocessing.py)：之前缺失的两域独立投影、训练簇各自子空间投影和无配对协方差谱候选。
- 更新 [zip_gcot/alignment.py](code/zip_gcot/alignment.py)：增加可选initial_R及local_start_from_global。两者默认关闭，**所有原行为不变**。
- 新增 [回归与四维正向对照测试](code/test_and_evaluate_corrected_hiwa.py)；使用Git历史里的旧核心求解器校验所有P/Q/R逐位相同。
- 新增 [作者MATLAB数学一致性与原结构测试](code/audit_author_hiwa_parity_and_subspaces.py)。
- 新增 [真实DANDI原参数完整P/Q/R控制实验](code/compare_corrected_original_hiwa_on_dandi.py)。

## 研究决策

**当前可以排除“核心矩阵公式简单写反”这一解释，也修复了一个确实存在的缺失模块。** 在合成数据上，现在能稳定恢复以前失败的几何对应，并能复现原作者式子空间成功现象。但真实神经域PCA4与手部pos/vel4的当前表示，即使有谱初始化和作者式子空间投影，仍然不能实现可靠物理对应。

**优先下一步：**在训练trial内部划分验证集，检查FA4/PCA4、运动目标和瞬时方向意义、低维神经/运动结构与时滞；并在使用完整P/Q/R的情况下评估组对应而非只看全域W2。Soft数值未收敛前不可断言软分组机制失败。不能反复拿已经分析过的测试trial挑方案。
