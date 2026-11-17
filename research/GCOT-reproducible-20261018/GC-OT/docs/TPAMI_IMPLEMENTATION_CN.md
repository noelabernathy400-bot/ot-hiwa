# TPAMI GCOT 实现与 HiWA 实验复刻

## 交付入口

算法类：`gcot.alignment.TPAMIGCOT`；合成与神经实验入口分别为 `experiments.synthetic.run` 和 `experiments.neural.run`。
结果按实验类型归档在 `results/synthetic/author_reference/tpami/` 与
`results/neural/author_reference/tpami/`，与主实验结果分开。
本次实现对应所提供 TPAMI 稿件第 4-6 页公式 (2)-(12) 的特征对齐部分。
它不包含 EquiformerV2、RoBERTa、能量回归训练，也不意味着验证了稿件的理论结论。

## 算法流程与公式对应

以下使用样本按行的代码约定：源域 X 为 N×d，目标域 Y 为 M×d。
论文使用样本按列，故论文 Ru 在代码中对应 `X @ R.T`。
支持 N≠M；当 N=M 时即原稿记号。实验中神经源域 803 点、目标域 623 点。

1. **独立学习两域软原型（公式 2）**。
   对每个域学习 S×d 原型 C，计算 A=softmax(X Cᵀ)，softmax 沿组维归一化。
   最小化 `||A C-X||_F² + lambda_h/N * sum(A log A)`。
   梯度同时通过 A 和重建项中的 C 传播，没有将 A 当常数。
   两域分别优化，不使用类别标签、真值旋转或测试误差。
2. **构造全支持加权分布（公式 3-5）**。
   第 i 组权重 `a_i=A[:,i]/sum(A[:,i])`；目标域类似。
   每个组仍包含全部样本，不取 argmax、不裁剪低权重点。
   仅对浮点下溢使用机器最小正数，然后重新归一化。
3. **定义双层目标（公式 8-10）**。
   `C_ij(R,Q)=sum_kl Q_kl * ||R x_k-y_l||²`。
   优化 `sum_ij [P_ij C_ij + epsilon_sample sum(Q_ij log Q_ij)]
   + epsilon_group sum(P log P)`，约束 RᵀR=I。
   P 的行列边际均为 1/S，Q_ij 的边际分别为 a_i、b_j。
   **样本熵项不乘 P_ij**，这一点决定局部 Sinkhorn 温度必须是
   `epsilon_sample/P_ij`，不能简单写成 epsilon_sample。
4. **局部旋转与传输交替更新（公式 11-12）**。
   给每个组对设置局部 R_ij、缩放对偶 Λ_ij，与全局 Rg 做共识。
   局部旋转：
   `R_ij = polar(2 P_ij Yᵀ Q_ijᵀ X + mu (Rg-Lambda_ij))`。
   polar(M)=U Vᵀ，其中 M=U Sigma Vᵀ，允许反射，符合 V_d,d=O(d)。
   接着用局部距离矩阵及 `epsilon_sample/P_ij` 更新 Q_ij。
5. **组耦合与全局共识更新**。
   `P = Sinkhorn(1/S,1/S,C,epsilon_group)`；
   `Rg = polar(mean_ij(R_ij+Lambda_ij))`；
   `Lambda_ij += R_ij-Rg`。
   重复直到指定停止条件或预算耗尽。
6. **推理（第 6 页 E-F 节）**。
   对输入特征应用 `X @ Rg.T`。软耦合用于训练对齐矩阵，推理不做硬匹配。

`TPAMIGCOT` 不执行 HiWA 的 X/sqrt(d)、mu/d 缩放，也不插入 Isomap。
旧 `GCOT` 类和旧运行器仍保留作历史兼容；本次运行器明确调用新类。

## 原稿未公开的细节及本次固定选择

所提供 PDF 没有附带其声称的 ADMM 补充材料或算法源代码。
因此“逐式实现”不等于“与作者训练程序逐位一致”。以下均显式固定且未调参：

| 项目 | 设置 | 来源/边界 |
|---|---|---|
| 原型初始化 | 单次 KMeans++，n_init=1 | 原稿未规定 |
| 原型优化 | L-BFGS-B，最多300步，解析梯度 | 原稿未规定 |
| lambda_h | 1 | 原稿未规定具体值 |
| mu | 0.005 | 借用 HiWA 数值设置，作用在未缩放 TPAMI 目标 |
| 组/样本熵 | 0.2 / 0.1 | 借用 HiWA 数值设置 |
| 外层 | 最多300次，至少6次，旋转变化≤0.1 | HiWA 停止规则 |
| 局部交替 | 最多100次，谱范数变化≤0.01 | HiWA 设置 |
| Sinkhorn | 样本150次、组1000次 | HiWA 固定迭代预算 |
| 初始 P | 1/S² | HiWA 初始化 |
| 初始全局 R | 随机方阵的极分解 | NumPy 固定种子 |
| 原型随机种子 | 源 seed，目标 seed+1 | 显式实现选择 |

原稿公式 (2) 的 H 指 Shannon 熵，而公式 (10) 将 H 写成 sum(p log p)，
两处符号用法不一致。实现按两处实际目标含义，均鼓励软分配：公式 (2)
是减 Shannon 熵，公式 (10) 是加负熵。

外层仅靠旋转变化停止，**不保证共识 ADMM 已收敛**。输出保留共识残差；
达到300次的试验照常报告，不换种子、不丢弃失败试验、不调整结果。
全支持软分布也不保证硬的一对一匹配。

## HiWA Figure 1e/f

直接调用 `D:/Desktop/最优传输/HiWA/hiwa-matlab/code/toolbox/GenerateSyntheticSubspaceData.m`，
以 MATLAB `rng(1)` 至 `rng(50)` 重新生成每种配置的数据。
每组50点，固有维度2，噪声0、组大小变化0。

WA、SA、ICP、CORAL 使用同一作者仓库 `code/results/figure1{e,f}_METHOD.mat`
中的原始50次输出点云；在重新生成的对应真值旋转上重算误差。
没有调用本工程修改过的 Python ICP，也没有替换作者 SA/CORAL 公式。
这保留了作者原始完整实验中的随机数消耗顺序。
GCOT 在相同新导出的 XX/YY 数据上运行，未使用 Rgt 拟合。

指标严格为 `||aligned-Rgt*XX||_F² / ||Rgt*XX||_F²`，是相对均方误差，
没有开平方。CDF 使用50个误差排序，纵坐标为 `(1:50)/50`。
线宽2，标记间隔3，标记大小10，横轴[0,2]、纵轴[0,1]，MATLAB默认颜色。
所有超出横轴范围的点仍保留在 CSV 和统计中。

注意原论文图注与作者脚本命名不一致：根据用户确认的作者代码优先原则，
现在严格保留 **MATLAB 脚本名称**：

| 输出面板 | 参数 | 作者脚本/压缩文件前缀 |
|---|---|---|
| Figure 1e | S=2,D=2 | figure1e |
| Figure 1f | S=5,D=6 | figure1f |

曲线仅 GCOT、WA、SA、ICP、CORAL；没有 HiWA 或 HiWA-SSC。

## HiWA Figure 2a

检查了两个 HiWA 仓库的现有代码与本地完整历史，包括 PyHiWA 初期
`a5fefc1` 和 `3c1e306`。未找到同时生成原论文 Figure 2a 和 WA/DAD
对比的完整脚本。原论文 2a 本身是分布映射示意图，不是五方法并列图。
本次按用户要求输出 Source、Target、GCOT、WA、DAD 五列点云，使用作者 demo
提供的数据、嵌入和 `plot_2d_clusters` 绘图函数。该组合应称为基于作者代码的
Figure 2a 扩展复刻，不能声称原图逐像素重现。

- 数据直接读取 `HiWA/PyHiWA/data/mihi_demo.mat`。
- 神经源域：Yte去掉常数列，作者demo的FactorAnalysis(n_components=3)，固定seed=0。
- 目标域：作者map_X_3D，将Xtr二维运动向量补上模长，得到3维。
- GCOT和WA拟合采用作者normal()白化后的共同数据；GCOT保持TPAMI目标，不用Isomap。
- 展示遵循PyHiWA notebook原有调用：源域原始FA，目标原始Xtr二维，GCOT/WA
  的R作用于原始FA再取前两维；另存 `aligned_objective` 为白化拟合坐标下的结果。
- DAD对共同白化FA执行作者 MATLAB DAD_3Dsearch：gridsz=8、numT=0、check2D=0，
  输出其原生二维结果；gridsz来自DAD作者demo，另外两项来自runDAD3d默认值。
- DAD沿用作者3D KL网格搜索、最终1000个平移候选和90角度/两种反射的2D搜索。
- 绘图不做标签重排、额外旋转、挑选种子或裁剪点；标签只用于颜色与核查。

PyHiWA notebook存在“fit内白化、transform作用于原始FA”的坐标约定。
本次保留该调用方式以遵守作者demo；所以该五列图不宜直接比较跨列坐标数值，
也不能把点云外观当作严格收敛或几何重建成功的证明。

## 源码来源与兼容处理

- HiWA MATLAB：siplab-gt/hiwa-matlab，commit `76f0c178ebb9d8a02d9b4bf6cf3c9852d27d9dbe`。
- PyHiWA：nerdslab/PyHiWA，commit `6b60a137c3ec696d13681827504178e2528d379f`。
- DAD：从作者项目页 https://nerdslab.github.io/neuralign/ 链接取得
  https://github.com/KordingLab/DAD ，commit `2f2059798cf599df5839a704682a18958811e13b`。
  下载源码保存在 `vendor/dad_matlab/DAD-master`。

## 运行与输出

在本项目目录运行（已建立 `.venv`）：

```powershell
.\.venv\Scripts\python.exe -m experiments.synthetic.run --trials 50 --workers 4
.\.venv\Scripts\python.exe -m experiments.neural.run --stage all
.\.venv\Scripts\python.exe -m visualization.synthetic
.\.venv\Scripts\python.exe -m visualization.neural
.\.venv\Scripts\python.exe -m evaluation.validation --scope all
.\.venv\Scripts\python.exe -m pytest -q
```

当前固定输入已保存在 `data/processed/`；实验计算、绘图和验证彼此独立。

输出包括 PNG、可编辑SVG、每个试验误差CSV、每列点云CSV、MAT/NPZ原始数据、
软分组/原型、旋转、P、迭代记录。每个合成配置seed=1及神经实验额外保存全部Q、
局部R、对偶变量、最终目标值与共识残差。

测试覆盖原型解析梯度、epsilon/P的目标最优性、全支持边际、未缩放目标重算和
正交性；另对100次数据和原始基线逐项核查，并检查最终图像。
