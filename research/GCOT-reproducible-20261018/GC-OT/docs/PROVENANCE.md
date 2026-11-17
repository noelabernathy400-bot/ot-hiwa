# 来源与复现边界

优先级为作者代码、作者历史、论文、明确记录的实现选择。

## 当前保留实验

Figure 1 使用作者 MATLAB 生成的100个合成数据集及保存的五种基线点云；
GCOT 在相同数据上运行。Figure 2a 使用 `PyHiWA/data/mihi_demo.mat`，
源域为FA后的神经点云，目标域为三维运动特征。正式结果位于
`results/synthetic/main/` 与 `results/neural/main/gcot/`；`author_reference/`
只保存作者输入和参照产物。

GCOT 和 HiWA 对齐阶段均使用 X/√D、mu/D、`mu=.005`、组熵`.2`、
样本熵`.1`、外层最多300次/阈值`.1`、内层最多100次/阈值`.01`，以及
固定150/1000次 Sinkhorn。神经实验两者共享同一份 Isomap(2, 12近邻)
辅助投影；分组在投影前完成，所得旋转应用于原白化坐标。

GCOT 的软原型学习、全支持软成员权重与 HiWA-SSC/GMM 的硬分组不同，
这是保留的模型差异。原型优化器参数在 TPAMI 文本中未公开，维持
`lambda_h=1`、单次 KMeans++ 初始化和L-BFGS-B最多300次，未用标签调参。

## 指标

Figure 1 的 rMSE 为 `||aligned - Rgt @ source||² / ||Rgt @ source||²`。
Figure 2 的1NN是目标点查询对齐源点；回归真值为与神经源点逐行配对的
`Xte`，统一到目标白化二维坐标。标准R²、作者白化版R²、rMSE、RMSE和MAE
的精确定义见 `single_run_metrics/REPORT.md`。

## 限制

MATLAB 与 Python 随机序列、线性代数细节不能保证逐位一致。共同停止规则
只检查全局旋转变化，不能等同于 ADMM 严格收敛；审计保留局部一致性残差。
神经 Isomap 步骤用于与 HiWA 配置对齐，不是 TPAMI 原文声称的步骤。
Figure 2b 原始采样与阴影协议未找到，因此未交付。
