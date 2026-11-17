# GCOT 神经实验修订

计算：`.venv\Scripts\python.exe -m experiments.neural.run`；仅绘图：`.venv\Scripts\python.exe -m visualization.neural`。
新产物写入 `results/neural/main/gcot/`，图内方法名仍为 **GCOT**。
旧 TPAMI 拟合数值位于 `results/neural/author_reference/tpami/`，不会覆盖。Figure 1e/f 按 MATLAB 脚本名称重画，
Figure 2a 按 Python 神经数据 demo 重画。

## 恢复辅助投影

使用相同的作者数据、FA seed=0、GCOT seed=1 和四个软原型。
两域首先使用作者 `normal()` 白化。源域白化数据上执行
`Isomap(n_components=2, n_neighbors=12)`，计算
`Tx = pinv(X) @ Isomap(X)`；目标域取同一白化坐标的前两维，
`Ty = pinv(Y) @ Y[:, :2]`。

与 HiWA 相同，用 `X @ Tx @ Tx.T`、`Y @ Ty @ Ty.T` 构造辅助对齐目标。
这不是把原型替换成 Isomap 聚类：软分组仍在未做辅助投影的白化输入上学习，
所有样本仍保留在每个软组中。未额外引入 HiWA 的 `1/sqrt(d)` 缩放。
局部及全局矩阵仍通过 SVD 极分解满足 `R.T @ R = I`。

辅助投影是本次用户要求恢复的实验预处理，不是 TPAMI 原稿公式 (2)-(12)
本身规定的步骤。该差异记录在本文和 manifest 中，不写入图内方法名。
`TPAMIGCOT` 不传 `projections` 时仍可运行无辅助投影版本。

## 收敛条件

主运行器现在与 HiWA 作者实验保持一致：外层最多 300 轮，至少完成
6 轮，当 `||R_global(k)-R_global(k-1)||_F <= 0.1` 时停止。
这里的两个 R 都是模型相邻迭代的估计，不是真实 `Rgt`。真实 R
不进入求解器接口、初始化或停止判断。

局部交替最多 100 轮，谱范数变化阈值为 `0.01`；样本 Sinkhorn
固定 150 轮，组 Sinkhorn 固定 1000 轮，不使用边际误差提前停止。
`P` 非有限时也依 HiWA 的时机停止，但记为 `numerical_failure`，不记为收敛。

代码仍保留 ADMM 残差、边际误差和组耦合变化作为事后诊断，但它们
不决定主运行的停止。HiWA 的规则只说明全局旋转在该阈值下稳定，
不保证局部共识、传输边际精度或非凸全局最优性。

`progress.jsonl` 每轮保存残差，`GCOT.npz` 包含完整最终 Q/P/R 与诊断，
`audit.json` 保存最终检查。标签只在拟合结束后用于颜色和簇中心对应核验。

## 作者展示坐标

MATLAB 仓库没有神经数据 Figure 2 脚本，因此该图按作者 Python notebook。
Source 使用原始三维 FA，Target 使用原始二维 `X_tr`；GCOT 和 WA
将在白化空间拟合的 R 直接作用于原始 FA，然后绘制前两列。
各 subplot 使用 Matplotlib 默认独立坐标范围，不共享坐标轴，不自行增加
equal aspect、白化显示或标签重排。DAD 保留其 MATLAB 作者输出。

这是对作者 demo 的忠实复刻，也保留了其“白化空间拟合、原始空间 transform”
的坐标约定。因为面板尺度不同，不应从像素位置直接比较跨面板距离。
五列组合是用户要求的扩展，不是 HiWA 仓库中的现成 Figure 2a 脚本。
