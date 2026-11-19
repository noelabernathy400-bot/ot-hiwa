# HySeR：SA、WA、DAD-like 与 Hard/Soft HiWA 真实1000点实验

## 已实际运行的对照及限制
正式对照以单一受试者 HySeR S01/S1 Random Task 五个试次做留一训练验证，四次训练试次各域1000点，剩余一次各域250测试点；源和目标无配对拟合，单点检索只访问1000目标训练点，原始五指真值只用于事后评分。

- SA 是 DANDI 实现的全基 PCA Subspace Alignment 转写，每折确定性运行一次。
- WA 为 DANDI 的退火 Sinkhorn/正交 Procrustes（200外迭代、温度100至0.1），在1000训练池采用250/轮随机小批量：每4轮完整覆盖1000点、重复50轮。这是1000池上的随机 WA 变体，并非原DANDI 160点全批次的逐步重现。
- DAD-like 仅对3D实现 DANDI 的 k=5 对称密度KL代理、128网格加Powell局部优化，明确不是KordingLab原版DAD；不编造5D DAD。
- Hard/Soft HiWA使用先前冻结的每折全1000训练点局部P/Q/Rkl模型。
- 匹配成功是五指原始物理力量距离低于各外折训练内非对应力距离的5%或20%分位。Fisher二维只用于可视化，不影响评价。
- 每个方法各有独立的目标域和变换后源域散点图，左右统一轴范围，每张图包含真正1000点；PCA3/FA3另绘制完整三维（不是Fisher二维）散点，不进行稀疏采样。

## 全五折物理评价（越高越好；L2越低越好）

|特征表示|方法|合格/运行|Hit@5%|Hit@20%|平均五指力量L2|
|---|---|---:|---:|---:|---:|
|PCA3_PCA3|Hard HiWA|10/10|11.72%|34.28%|0.25396|
|PCA3_PCA3|Soft HiWA|10/10|10.84%|33.88%|0.26199|
|PCA3_PCA3|SA|5/5|0.40%|25.92%|0.29848|
|PCA3_PCA3|WA|10/10|7.12%|29.88%|0.27036|
|PCA3_PCA3|DAD_like_3D|5/5|14.16%|34.40%|0.25812|
|FA3_FA3|Hard HiWA|10/10|3.96%|35.16%|0.28545|
|FA3_FA3|Soft HiWA|10/10|3.84%|33.72%|0.27095|
|FA3_FA3|SA|5/5|1.44%|16.88%|0.31795|
|FA3_FA3|WA|10/10|4.00%|32.80%|0.27820|
|FA3_FA3|DAD_like_3D|5/5|5.92%|22.88%|0.29924|
|PCA5_RAW5|Hard HiWA|10/10|12.52%|34.84%|0.26070|
|PCA5_RAW5|Soft HiWA|10/10|12.36%|34.12%|0.25767|
|PCA5_RAW5|SA|5/5|1.04%|19.04%|0.28405|
|PCA5_RAW5|WA|10/10|10.40%|30.68%|0.27437|
|FA5_RAW5|Hard HiWA|10/10|8.84%|36.40%|0.26823|
|FA5_RAW5|Soft HiWA|10/10|6.88%|37.00%|0.26485|
|FA5_RAW5|SA|5/5|0.56%|18.08%|0.29234|
|FA5_RAW5|WA|10/10|10.32%|30.76%|0.28194|

## 五折1000点分离散点图（二维 Fisher；三维完整坐标）
源域和目标域分别由训练数据拟合独立KMeans与Fisher；对齐后源域统一投到目标训练Fisher平面。**绝不重叠绘制两个域**：每个方法左图仅目标1000点，右图仅变换后的源1000点，二者和其它方法统一使用相同xlim/ylim及等轴尺度。目标与源各自独立的四簇着色只表达本域伪簇，不隐含真实簇对应关系。对于PCA3/FA3，另有原始全三维1000点左右并排视图，使用同一XYZ范围和相同相机视角。所有数值评价仍在完整d维和原始五指力量中。

### PCA3_PCA3：五折每个方法全部1000点
- [留出试次 1：全部方法 Fisher 左右分离、同尺度1000+1000对照](figures/PCA3_PCA3_FOLD1_ALL1000_Fisher_METHODS.png)
- [留出试次 2：全部方法 Fisher 左右分离、同尺度1000+1000对照](figures/PCA3_PCA3_FOLD2_ALL1000_Fisher_METHODS.png)
- [留出试次 3：全部方法 Fisher 左右分离、同尺度1000+1000对照](figures/PCA3_PCA3_FOLD3_ALL1000_Fisher_METHODS.png)
- [留出试次 4：全部方法 Fisher 左右分离、同尺度1000+1000对照](figures/PCA3_PCA3_FOLD4_ALL1000_Fisher_METHODS.png)
- [留出试次 5：全部方法 Fisher 左右分离、同尺度1000+1000对照](figures/PCA3_PCA3_FOLD5_ALL1000_Fisher_METHODS.png)
- [留出试次 1：完整三维1000点，所有方法左右分离同XYZ尺度](figures/native3D_allmethods/PCA3_PCA3_FOLD1_ALL1000_NATIVE3D_SEPARATE_METHODS.png)
- [留出试次 2：完整三维1000点，所有方法左右分离同XYZ尺度](figures/native3D_allmethods/PCA3_PCA3_FOLD2_ALL1000_NATIVE3D_SEPARATE_METHODS.png)
- [留出试次 3：完整三维1000点，所有方法左右分离同XYZ尺度](figures/native3D_allmethods/PCA3_PCA3_FOLD3_ALL1000_NATIVE3D_SEPARATE_METHODS.png)
- [留出试次 4：完整三维1000点，所有方法左右分离同XYZ尺度](figures/native3D_allmethods/PCA3_PCA3_FOLD4_ALL1000_NATIVE3D_SEPARATE_METHODS.png)
- [留出试次 5：完整三维1000点，所有方法左右分离同XYZ尺度](figures/native3D_allmethods/PCA3_PCA3_FOLD5_ALL1000_NATIVE3D_SEPARATE_METHODS.png)
- [该表示多方法严格Hit@5%图](figures/PCA3_PCA3_Hit5_all_methods.png)
- [该表示真实距离阈值曲线](figures/PCA3_PCA3_eps_curves_all_methods.png)

### FA3_FA3：五折每个方法全部1000点
- [留出试次 1：全部方法 Fisher 左右分离、同尺度1000+1000对照](figures/FA3_FA3_FOLD1_ALL1000_Fisher_METHODS.png)
- [留出试次 2：全部方法 Fisher 左右分离、同尺度1000+1000对照](figures/FA3_FA3_FOLD2_ALL1000_Fisher_METHODS.png)
- [留出试次 3：全部方法 Fisher 左右分离、同尺度1000+1000对照](figures/FA3_FA3_FOLD3_ALL1000_Fisher_METHODS.png)
- [留出试次 4：全部方法 Fisher 左右分离、同尺度1000+1000对照](figures/FA3_FA3_FOLD4_ALL1000_Fisher_METHODS.png)
- [留出试次 5：全部方法 Fisher 左右分离、同尺度1000+1000对照](figures/FA3_FA3_FOLD5_ALL1000_Fisher_METHODS.png)
- [留出试次 1：完整三维1000点，所有方法左右分离同XYZ尺度](figures/native3D_allmethods/FA3_FA3_FOLD1_ALL1000_NATIVE3D_SEPARATE_METHODS.png)
- [留出试次 2：完整三维1000点，所有方法左右分离同XYZ尺度](figures/native3D_allmethods/FA3_FA3_FOLD2_ALL1000_NATIVE3D_SEPARATE_METHODS.png)
- [留出试次 3：完整三维1000点，所有方法左右分离同XYZ尺度](figures/native3D_allmethods/FA3_FA3_FOLD3_ALL1000_NATIVE3D_SEPARATE_METHODS.png)
- [留出试次 4：完整三维1000点，所有方法左右分离同XYZ尺度](figures/native3D_allmethods/FA3_FA3_FOLD4_ALL1000_NATIVE3D_SEPARATE_METHODS.png)
- [留出试次 5：完整三维1000点，所有方法左右分离同XYZ尺度](figures/native3D_allmethods/FA3_FA3_FOLD5_ALL1000_NATIVE3D_SEPARATE_METHODS.png)
- [该表示多方法严格Hit@5%图](figures/FA3_FA3_Hit5_all_methods.png)
- [该表示真实距离阈值曲线](figures/FA3_FA3_eps_curves_all_methods.png)

### PCA5_RAW5：五折每个方法全部1000点
- [留出试次 1：全部方法 Fisher 左右分离、同尺度1000+1000对照](figures/PCA5_RAW5_FOLD1_ALL1000_Fisher_METHODS.png)
- [留出试次 2：全部方法 Fisher 左右分离、同尺度1000+1000对照](figures/PCA5_RAW5_FOLD2_ALL1000_Fisher_METHODS.png)
- [留出试次 3：全部方法 Fisher 左右分离、同尺度1000+1000对照](figures/PCA5_RAW5_FOLD3_ALL1000_Fisher_METHODS.png)
- [留出试次 4：全部方法 Fisher 左右分离、同尺度1000+1000对照](figures/PCA5_RAW5_FOLD4_ALL1000_Fisher_METHODS.png)
- [留出试次 5：全部方法 Fisher 左右分离、同尺度1000+1000对照](figures/PCA5_RAW5_FOLD5_ALL1000_Fisher_METHODS.png)
- [该表示多方法严格Hit@5%图](figures/PCA5_RAW5_Hit5_all_methods.png)
- [该表示真实距离阈值曲线](figures/PCA5_RAW5_eps_curves_all_methods.png)

### FA5_RAW5：五折每个方法全部1000点
- [留出试次 1：全部方法 Fisher 左右分离、同尺度1000+1000对照](figures/FA5_RAW5_FOLD1_ALL1000_Fisher_METHODS.png)
- [留出试次 2：全部方法 Fisher 左右分离、同尺度1000+1000对照](figures/FA5_RAW5_FOLD2_ALL1000_Fisher_METHODS.png)
- [留出试次 3：全部方法 Fisher 左右分离、同尺度1000+1000对照](figures/FA5_RAW5_FOLD3_ALL1000_Fisher_METHODS.png)
- [留出试次 4：全部方法 Fisher 左右分离、同尺度1000+1000对照](figures/FA5_RAW5_FOLD4_ALL1000_Fisher_METHODS.png)
- [留出试次 5：全部方法 Fisher 左右分离、同尺度1000+1000对照](figures/FA5_RAW5_FOLD5_ALL1000_Fisher_METHODS.png)
- [该表示多方法严格Hit@5%图](figures/FA5_RAW5_Hit5_all_methods.png)
- [该表示真实距离阈值曲线](figures/FA5_RAW5_eps_curves_all_methods.png)

## 二维单独双面板散点图（Fold1，每个域各1000点，绝不重叠）

- [PCA3_PCA3 / Identity / 1000源+1000目标](figures/individual_fold1/PCA3_PCA3_FOLD1_Identity_ALL1000_Fisher.png)
- [PCA3_PCA3 / ICP / 1000源+1000目标](figures/individual_fold1/PCA3_PCA3_FOLD1_ICP_ALL1000_Fisher.png)
- [PCA3_PCA3 / SingleOT / 1000源+1000目标](figures/individual_fold1/PCA3_PCA3_FOLD1_SingleOT_ALL1000_Fisher.png)
- [PCA3_PCA3 / SA / 1000源+1000目标](figures/individual_fold1/PCA3_PCA3_FOLD1_SA_ALL1000_Fisher.png)
- [PCA3_PCA3 / WA / 1000源+1000目标](figures/individual_fold1/PCA3_PCA3_FOLD1_WA_ALL1000_Fisher.png)
- [PCA3_PCA3 / DAD_like_3D / 1000源+1000目标](figures/individual_fold1/PCA3_PCA3_FOLD1_DAD_like_3D_ALL1000_Fisher.png)
- [PCA3_PCA3 / Hard HiWA / 1000源+1000目标](figures/individual_fold1/PCA3_PCA3_FOLD1_Hard_HiWA_ALL1000_Fisher.png)
- [PCA3_PCA3 / Soft HiWA / 1000源+1000目标](figures/individual_fold1/PCA3_PCA3_FOLD1_Soft_HiWA_ALL1000_Fisher.png)
- [FA3_FA3 / Identity / 1000源+1000目标](figures/individual_fold1/FA3_FA3_FOLD1_Identity_ALL1000_Fisher.png)
- [FA3_FA3 / ICP / 1000源+1000目标](figures/individual_fold1/FA3_FA3_FOLD1_ICP_ALL1000_Fisher.png)
- [FA3_FA3 / SingleOT / 1000源+1000目标](figures/individual_fold1/FA3_FA3_FOLD1_SingleOT_ALL1000_Fisher.png)
- [FA3_FA3 / SA / 1000源+1000目标](figures/individual_fold1/FA3_FA3_FOLD1_SA_ALL1000_Fisher.png)
- [FA3_FA3 / WA / 1000源+1000目标](figures/individual_fold1/FA3_FA3_FOLD1_WA_ALL1000_Fisher.png)
- [FA3_FA3 / DAD_like_3D / 1000源+1000目标](figures/individual_fold1/FA3_FA3_FOLD1_DAD_like_3D_ALL1000_Fisher.png)
- [FA3_FA3 / Hard HiWA / 1000源+1000目标](figures/individual_fold1/FA3_FA3_FOLD1_Hard_HiWA_ALL1000_Fisher.png)
- [FA3_FA3 / Soft HiWA / 1000源+1000目标](figures/individual_fold1/FA3_FA3_FOLD1_Soft_HiWA_ALL1000_Fisher.png)
- [PCA5_RAW5 / Identity / 1000源+1000目标](figures/individual_fold1/PCA5_RAW5_FOLD1_Identity_ALL1000_Fisher.png)
- [PCA5_RAW5 / ICP / 1000源+1000目标](figures/individual_fold1/PCA5_RAW5_FOLD1_ICP_ALL1000_Fisher.png)
- [PCA5_RAW5 / SingleOT / 1000源+1000目标](figures/individual_fold1/PCA5_RAW5_FOLD1_SingleOT_ALL1000_Fisher.png)
- [PCA5_RAW5 / SA / 1000源+1000目标](figures/individual_fold1/PCA5_RAW5_FOLD1_SA_ALL1000_Fisher.png)
- [PCA5_RAW5 / WA / 1000源+1000目标](figures/individual_fold1/PCA5_RAW5_FOLD1_WA_ALL1000_Fisher.png)
- [PCA5_RAW5 / Hard HiWA / 1000源+1000目标](figures/individual_fold1/PCA5_RAW5_FOLD1_Hard_HiWA_ALL1000_Fisher.png)
- [PCA5_RAW5 / Soft HiWA / 1000源+1000目标](figures/individual_fold1/PCA5_RAW5_FOLD1_Soft_HiWA_ALL1000_Fisher.png)
- [FA5_RAW5 / Identity / 1000源+1000目标](figures/individual_fold1/FA5_RAW5_FOLD1_Identity_ALL1000_Fisher.png)
- [FA5_RAW5 / ICP / 1000源+1000目标](figures/individual_fold1/FA5_RAW5_FOLD1_ICP_ALL1000_Fisher.png)
- [FA5_RAW5 / SingleOT / 1000源+1000目标](figures/individual_fold1/FA5_RAW5_FOLD1_SingleOT_ALL1000_Fisher.png)
- [FA5_RAW5 / SA / 1000源+1000目标](figures/individual_fold1/FA5_RAW5_FOLD1_SA_ALL1000_Fisher.png)
- [FA5_RAW5 / WA / 1000源+1000目标](figures/individual_fold1/FA5_RAW5_FOLD1_WA_ALL1000_Fisher.png)
- [FA5_RAW5 / Hard HiWA / 1000源+1000目标](figures/individual_fold1/FA5_RAW5_FOLD1_Hard_HiWA_ALL1000_Fisher.png)
- [FA5_RAW5 / Soft HiWA / 1000源+1000目标](figures/individual_fold1/FA5_RAW5_FOLD1_Soft_HiWA_ALL1000_Fisher.png)

## 额外完整三维分离图：PCA3/FA3 每个方法、每一折

- [PCA3_PCA3 / trial 1 / Identity：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD1_Identity_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 1 / ICP：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD1_ICP_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 1 / SingleOT：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD1_SingleOT_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 1 / SA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD1_SA_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 1 / WA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD1_WA_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 1 / DAD_like_3D：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD1_DAD_like_3D_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 1 / Hard HiWA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD1_Hard_HiWA_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 1 / Soft HiWA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD1_Soft_HiWA_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 2 / Identity：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD2_Identity_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 2 / ICP：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD2_ICP_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 2 / SingleOT：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD2_SingleOT_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 2 / SA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD2_SA_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 2 / WA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD2_WA_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 2 / DAD_like_3D：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD2_DAD_like_3D_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 2 / Hard HiWA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD2_Hard_HiWA_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 2 / Soft HiWA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD2_Soft_HiWA_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 3 / Identity：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD3_Identity_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 3 / ICP：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD3_ICP_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 3 / SingleOT：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD3_SingleOT_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 3 / SA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD3_SA_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 3 / WA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD3_WA_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 3 / DAD_like_3D：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD3_DAD_like_3D_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 3 / Hard HiWA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD3_Hard_HiWA_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 3 / Soft HiWA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD3_Soft_HiWA_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 4 / Identity：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD4_Identity_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 4 / ICP：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD4_ICP_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 4 / SingleOT：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD4_SingleOT_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 4 / SA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD4_SA_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 4 / WA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD4_WA_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 4 / DAD_like_3D：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD4_DAD_like_3D_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 4 / Hard HiWA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD4_Hard_HiWA_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 4 / Soft HiWA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD4_Soft_HiWA_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 5 / Identity：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD5_Identity_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 5 / ICP：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD5_ICP_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 5 / SingleOT：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD5_SingleOT_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 5 / SA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD5_SA_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 5 / WA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD5_WA_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 5 / DAD_like_3D：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD5_DAD_like_3D_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 5 / Hard HiWA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD5_Hard_HiWA_NATIVE3D_SEPARATE1000.png)
- [PCA3_PCA3 / trial 5 / Soft HiWA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/PCA3_PCA3_FOLD5_Soft_HiWA_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 1 / Identity：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD1_Identity_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 1 / ICP：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD1_ICP_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 1 / SingleOT：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD1_SingleOT_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 1 / SA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD1_SA_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 1 / WA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD1_WA_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 1 / DAD_like_3D：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD1_DAD_like_3D_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 1 / Hard HiWA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD1_Hard_HiWA_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 1 / Soft HiWA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD1_Soft_HiWA_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 2 / Identity：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD2_Identity_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 2 / ICP：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD2_ICP_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 2 / SingleOT：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD2_SingleOT_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 2 / SA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD2_SA_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 2 / WA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD2_WA_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 2 / DAD_like_3D：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD2_DAD_like_3D_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 2 / Hard HiWA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD2_Hard_HiWA_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 2 / Soft HiWA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD2_Soft_HiWA_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 3 / Identity：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD3_Identity_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 3 / ICP：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD3_ICP_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 3 / SingleOT：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD3_SingleOT_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 3 / SA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD3_SA_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 3 / WA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD3_WA_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 3 / DAD_like_3D：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD3_DAD_like_3D_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 3 / Hard HiWA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD3_Hard_HiWA_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 3 / Soft HiWA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD3_Soft_HiWA_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 4 / Identity：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD4_Identity_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 4 / ICP：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD4_ICP_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 4 / SingleOT：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD4_SingleOT_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 4 / SA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD4_SA_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 4 / WA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD4_WA_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 4 / DAD_like_3D：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD4_DAD_like_3D_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 4 / Hard HiWA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD4_Hard_HiWA_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 4 / Soft HiWA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD4_Soft_HiWA_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 5 / Identity：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD5_Identity_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 5 / ICP：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD5_ICP_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 5 / SingleOT：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD5_SingleOT_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 5 / SA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD5_SA_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 5 / WA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD5_WA_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 5 / DAD_like_3D：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD5_DAD_like_3D_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 5 / Hard HiWA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD5_Hard_HiWA_NATIVE3D_SEPARATE1000.png)
- [FA3_FA3 / trial 5 / Soft HiWA：3D目标1000与源1000独立同尺度](figures/native3D_individual_allfolds/FA3_FA3_FOLD5_Soft_HiWA_NATIVE3D_SEPARATE1000.png)

## 数值存档和严谨性
全部新模型的正交R、每条测试样本真实五指误差、训练配置和数值状态都在下列JSON。上表按相同试次折比较，不能把同一被试的两种子当独立受试者；也不能将事后阈值70%作为预注册泛化准确率。

- [所有SA/WA/DAD-like逐折逐种子原始数据](all_SA_WA_DAD_run_records.json)
- [逐试次完整对照CSV](all_fivefold_methods.csv)
- [汇总表CSV](summary_SA_WA_DAD_HiWA.csv)
- [实际物理误差阈值曲线CSV](dense_physical_tolerance_all_methods.csv)
- [1000点图像数目、XYZ轴一致性及独立分簇审计](scatter_count_audit.json)
- [此前额外Identity/ICP/单层OT](../additional_baselines/ADDITIONAL_UNSUPERVISED_BASELINES_ZH.md)
- [此前80次HiWA共享P/Q层级目标](../shared_hierarchical_full250/SHARED_HIERARCHICAL_EVALUATION_ZH.md)
- [DANDI原迁移代码源分支](https://github.com/noelabernathy400-bot/ot-hiwa/blob/experiment/dandi-pca3-fa3-vs-pca4-fa4-20261119/projects/dandi-000138-hiwa-4d/code/closeout_dandi_sa_wa_dad_baselines.py)

这是一组特定参数的实际基线复核，不是作者原SA/WA/DAD所有超参的最优性或统计跨人群显著性证明。
