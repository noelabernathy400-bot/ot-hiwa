# 新数据首选：Mihi–Chewie 中心向外伸手实验

**2026-11-19：真实作者数据仅做先验适用性检查，尚未拟合新 Hard/Soft OT，也尚未证明 Soft 更优。** 本文件只存于独立候选筛选分支，与旧 DANDI 和 Hyser 项目隔离。

## 真实数据来源与规模

1. 官方 HiWA [小型示例及 notebook](https://github.com/nerdslab/PyHiWA)：data/mihi_demo.mat，105,990 bytes，SHA256 907bc78863d9488ef352eaad8af8296bc28484d5a49591309b1c8576c6009f8c。示例源为 803 组神经观测×187 原始通道，目标为 623 组二维手部运动观测，四个真实方向 ID=3、4、5、7。
2. 更适合正式实验的 [完整 Mihi–Chewie 两猕猴四会话记录](https://github.com/nerdslab/myow-neuro/tree/main/data/mihi-chewie)：Mihi day1 209 trials/187 neurons, day2 215/172；Chewie day1 159/174, day2 180/155。原始 MAT 以及部分已处理 100ms trial 数据均公开，原始预处理程序 myow/data/monkey_neural_dataset.py 提供按 trial 的神经放电、手部位置、速度、加速度、力和八方向标签。各会话真正有效试次与类别计数需下载核实。

## 已完成的真实几何预检查

[实测 JSON](preflight_results/REAL_MIHIDEMO_PREFLIGHT.json) 与 [3D 真实散点预检查](preflight_results/MIHI_REAL_DATA_PREFLIGHT_K4_SOURCE_TARGET_3D.png)。

- 源神经原187通道中剔除常数后154通道，先 FA3；目标手部 XY 映射至物理含义明确但非独立测量的三个量 (x,y, sqrt(x²+y²))；两域各自 RMS 标准化。
- 无标签拟合的 KMeans(K4) silhouette：神经0.4251，手部0.5427；训练完成后才计算真实方向标签 NMI：神经0.4131，手部0.3719。
- 四个真实方向类别中心之间**仅6个距离**的跨域 Pearson 相关0.9237（Spearman0.8286）。这只是小样本中心几何诊断，不能直接当作成功率。
- 作者准备的同会话同步神经 Yte 与运动 Xte 上抽约15k组配对样本距离，Pearson相关0.6300；**这个统计不是跨会话 Yte→Xtr 的逐样本对应**。源/目标的独立真实标签**只在后验诊断使用**。
- [真实执行通过的 GitHub Actions](https://github.com/noelabernathy400-bot/ot-hiwa/actions/runs/44525636151)。

**最重要的严谨性限定**：原作者 notebook 训练时向 HiWA 直接传入了两域真实方向标签；原作者报告的方向准确率不能解释为独立双域无配对自动分组的结果。此处预检查用真实标签来选择候选数据，存在数据筛选偏差；新的完整会话数据必须另外冻结 trial/会话验证，不能拿这一子集当最终盲测。

## 正式研究问题与实验门槛

> 在相同伸手动作下，跨日期神经表征或跨动物神经—手部运动分布能否利用独立无配对 P/Q/R 建立有语义的组匹配？当邻近伸手方向的神经簇发生重叠时，Soft-GCOT 是否比 Hard-HiWA 更稳健？

推荐按以下先后顺序进行，**目前全属计划，不是已运行实验**：

1. 对 Mihi/Chewie 四会话全部原始 trial 级数据做下载、SHA/元数据、真实方向计数和位置/速度单位审核；按真正独立 trial 拆分，冻结最终测试集。
2. 首先试同动物 Chewie day1→day2 的神经 FA3↔FA3，检验跨日神经域的自然共同结构；其次 Mihi→Chewie 神经↔神经；最后原作者式神经↔手部不同动物运动分布。
3. 数据预检：两域各自独立 FA/PCA、KMeans/Soft 原型、silhouette、真实方向标签 postfit NMI、8类中心28个距离的结构相关、置信区间/置换零假设、全维散点，**没有可检验共同结构就停止大规模OT**。
4. 固定同样真实训练/验证 trials、样本池和 P/Q/R 内核，比较原版 HiWA、同核 Hard、Soft、SA、WA、DAD（官方/改编要标明），统一看组方向对应的外部标签传输正确率、实际物理运动状态匹配、共享层级 T/J、自由 W2²、数值残差、训练时间与对齐稳定性。
5. 未来预注册 K=4或8、Soft 温度、ADMM μ、样本与组熵温度，以及组重叠/准确率曲线；决不能靠放大匹配阈值制造模型提升。历史 DANDI/HYser 实验分数不能搬用。
6. 散点与 Fisher2/3 均从真实全数据与独立训练组开始，跨域真运动方向标签只能在训练后用于展示或独立评价；Soft 颜色 argmax 只为作图，内部训练保留完整概率权重。

## 备选与为什么不是首选

- [3DMatch 真实RGB-D三维点云](https://3dmatch.cs.princeton.edu/)有真实刚体对齐/姿态参考，最容易展示清晰几何成功；但不同扫描片段只部分重合，传统平衡HiWA可能因非重叠点产生偏差；它更接近几何配准而非神经分布对齐，需要与 ICP 等强基线竞争。
- [DANDI 000688 中心向外伸手](https://dandiarchive.org/dandiset/000688)有多次真实神经和运动记录，但下载约13GB，且仍需证明跨模态潜变量几何成立。

**决策**：优先采用完整 Mihi–Chewie 作为新主数据候选，作者 demo 仅做有利先验和代码正向控制；实际八方向多会话数据通过独立几何预检之后才正式开始 Hard/Soft/SA/WA/DAD 对比。
