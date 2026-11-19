# 原 HiWA / 当前 SoftHiWA 代码审查及本轮实验口径（2026-11-19）

## 原始来源

- Lee, Dabagia, Dyer, Rozell, *Hierarchical Optimal Transport for Multimodal Distribution Alignment*，NeurIPS 2019，https://proceedings.neurips.cc/paper_files/paper/2019/file/e41990b122b864f164d2ab96e8322690-Paper.pdf
- 原作者 Python 实现 https://github.com/nerdslab/PyHiWA ，核心 `src/hiwa.py`
- 本仓库的 Hard 移植 `src/hiwa/hiwa.py` 与适配的软分组局部ADMM求解器 `src/cc_hiwa/soft_hiwa.py`。
- 原论文第5.2节把神经放电先经过 Factor Analysis 降为3维，目标3维为手臂运动，计算运动方向**类别准确率**。当前 Hyser Random Task **没有逐时窗动作方向真标签**，故新设最近邻正确邻域不是原作者完全相同的基准。

## 实际逐项核对（本轮不是简化全局R）

| 数学对象/更新 | 原作者HiWA / 本仓库原版 | 本轮同求解器Hard/Soft |
|---|---|---|
| 源/目标独立分组 | `X_labels`,`Y_labels`，不继承另一边标签 | 两域分别KMeans，Hard为独立one-hot，Soft为独立softmax距离/τ，τ=0.5 |
| 每组对局部旋转 `R_kl` | 逐组对 Stiefel/正交群做polar更新 | 完整保留本仓库 `SoftHiWA` 的每组对局部旋转 |
| 条件点级 `Q_kl` | 局部 Sinkhorn；两组点各自均匀边缘 | Hard每组one-hot条件点测度，Soft归一化隶属度；每个 `Q_kl` 真求解、真实矩阵保留指定示例 |
| 簇级 `P` | 组边缘各均匀1/K，γ1=0.2 | 同一K=4，行/列边缘1/4；γ1=0.2；**数值上替换为稳定的同一熵正则OT对偶解**以修复原核缩放在此数据上的边缘约束失败 |
| 点级温度 | γ2/`P_kl`，γ2=0.1 | 完全一致，`inner_entropy_mode=legacy_inverse_group_mass` |
| ADMM共识 | `Rg=polar(mean(R_kl + L_kl))`，`L+=Rloc-Rg` | 同一个逐对共识和对偶更新；`mu=0.005` |
| 额外目标 | 原版没有物理时间签名/原型引导 | `rotation_anchor=0, representative_guidance=0, representative_rotation=0, component_conditioning=0, temporal_signature=0`，不引入新目标 |
| 局部/全局对齐域 | 原作者对列向量旋转，先除sqrt(d) | 当前X/Y都是五维训练内部标准化/白化；本求解器依原版再除sqrt(5)；留出评价统一除sqrt(5) |
| 源目标样本配对 | 原论文无监督训练不使用真配对 | 训练源/目标分别独立抽点；真实同步标签严格后处理评价 |
| 训练量级 | 理想最终实验应在每折完整1000点训练Q | 本轮**验证代码的算例**明确每折独立抽取250个真实训练点（4试次各63/63/63/61），对这250点每组局部Q完整求解；这不是全1000点Q，也不是96锚点量化近似 |

**注意**：`SoftHiWA` 与原版 `HiWA` 具有同一局部旋转/层级OT/ADMM数学骨架，但它为软分组做了必要的**加权条件边缘推广**，且两份代码的初始化与数值稳定性维护细节不完全逐行相同。因此本轮称为“原版结构的本地SoftHiWA全Q求解”，而不是“直接运行未修改的原作者代码”。

当前第一次跑原样 P 核缩放时出现组边缘违反约束，严格核验失败，已改为使用等价的熵正则 OT 对偶求解计算同一个 `P` 子问题，严格检验每行列均为1/4；不改变组级目标或正则权重。

## 训练/绘图/新成功率三者隔离

- **主模型**：K=4，源PCA5/FA5，完整五根手指的目标5维，原样五折留一试次划分；固定所有其余参数。
- **展示**：五维每域独立拟合Kmeans训练标签，在全部10个二维坐标平面扫描训练轮廓系数；另选训练内Fisher二维线性子空间，最大化簇间/簇内散度比；两者都只用于**展示**，不改变训练、P/Q/R和五维指标。
- **邻域检索**：`j*(i)=argmin_j ||R x_i - y_j||`，必须计算全部d维距离而非挑出的2维。仅`||Rx_i-y_j*||<ε`的判据叫**覆盖率**；作为“预测正确”需再用训练后实际同步真值检查`||y_j*-y_i||<ε`，也给出严格时间索引命中率、Top5索引、随机目标选择基线。
- **阈值**：对目标域四个**训练试次间**随机无关的目标点距离取**5%分位**，一折一数，训练后锁定，同维同表示Hard和Soft共享。不能利用测试集选择使成功率看起来更高的ε。
- **3D控制**：目标三维特征用五指力量共同PCA3/FA3拟合，不是从五根手指删除两根；阈值在三维中另校准。跨3D和5D的命中百分比不能无条件解释为维度的因果效应。

## 已核实的先导模型固定变换结果（原96锚点全局R，**非本轮局部ADMM**）

五维（5折×2种子×2方法）：

| 表示 | Hard覆盖率 | Soft覆盖率 | Hard真实力量邻域 | Soft真实力量邻域 | 随机邻域 |
|---|---:|---:|---:|---:|---:|
| PCA5 + full5 |39.88%|40.76%|9.96%|12.00%|9.34%|
| FA5 + full5 |45.60%|49.00%|12.08%|13.60%|9.34%|

这些结果说明：Soft可在空间几何覆盖与近邻力量状态检索上小幅改善，但尚未达到可靠的运动预测准确度；原版神经解码真实动作分类准确率在当前无动作标签Hyser Random Task上**不可直接复制**。

## 结果所在

- [五维最近邻真实命中率与完整十对投影](results_best2D_NN/)
- [原生最佳Fisher二维子空间](results_best2D_NN/fisher_subspace/)
- [新三维PCA/FA近邻对照](results_3D_best2D_NN/)
- [训练抽样全Q局部ADMM代码](local_PQR_ADMM_hyser.py)及独立试次结果文件夹`results_local_ADMM_sanity/`（以GitHub Actions最终成功的任务为准）
- [PCA/FA主实验先导结论](PCA_FA_BOTH_DOMAINS_FINAL_ZH.md)

对任何未通过残差或边缘约束的运行，**必须标明数值未收敛，不能暗中剔除或写为成功。**
