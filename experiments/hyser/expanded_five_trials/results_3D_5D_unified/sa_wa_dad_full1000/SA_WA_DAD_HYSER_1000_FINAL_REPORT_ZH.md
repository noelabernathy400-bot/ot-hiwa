# HySeR：SA、WA、DAD-like 与 Hard/Soft HiWA 真实1000点实验

## 已实际运行的对照及限制
正式对照以单一受试者 HySeR S01/S1 Random Task 五个试次做留一训练验证，四次训练试次各域1000点，剩余一次各域250测试点；源和目标无配对拟合，单点检索只访问1000目标训练点，原始五指真值只用于事后评分。

- SA 是 DANDI 实现的全基 PCA Subspace Alignment 转写，每折确定性运行一次。
- WA 为 DANDI 的退火 Sinkhorn/正交 Procrustes（200外迭代、温度100至0.1），在1000训练池采用250/轮随机小批量：每4轮完整覆盖1000点、重复50轮。这是1000池上的随机 WA 变体，并非原DANDI 160点全批次的逐步重现。
- DAD-like 仅对3D实现 DANDI 的 k=5 对称密度KL代理、128网格加Powell局部优化，明确不是KordingLab原版DAD；不编造5D DAD。
- Hard/Soft HiWA使用先前冻结的每折全1000训练点局部P/Q/Rkl模型。
- 匹配成功是五指原始物理力量距离低于各外折训练内非对应力距离的5%或20%分位。Fisher二维只用于可视化，不影响评价。
- 1000个点是每个域的真正训练点数；每张叠加图展示1000源点和1000目标点，不进行稀疏采样。

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

## 所有真实1000点Fisher散点图
原始两域各自用训练组独立训练Fisher子空间；对齐后都投到目标域训练Fisher子空间。四种颜色代表源域独立训练的四个伪簇，灰色代表目标域背景点。颜色不意味着真实动作类别/跨域真配对。评分使用完整d维与五指原始力量。

### PCA3_PCA3：五折每个方法全部1000点
- [留出试次 1：全部方法 Fisher 1000+1000 对照](figures/PCA3_PCA3_FOLD1_ALL1000_Fisher_METHODS.png)
- [留出试次 2：全部方法 Fisher 1000+1000 对照](figures/PCA3_PCA3_FOLD2_ALL1000_Fisher_METHODS.png)
- [留出试次 3：全部方法 Fisher 1000+1000 对照](figures/PCA3_PCA3_FOLD3_ALL1000_Fisher_METHODS.png)
- [留出试次 4：全部方法 Fisher 1000+1000 对照](figures/PCA3_PCA3_FOLD4_ALL1000_Fisher_METHODS.png)
- [留出试次 5：全部方法 Fisher 1000+1000 对照](figures/PCA3_PCA3_FOLD5_ALL1000_Fisher_METHODS.png)
- [该表示多方法严格Hit@5%图](figures/PCA3_PCA3_Hit5_all_methods.png)
- [该表示真实距离阈值曲线](figures/PCA3_PCA3_eps_curves_all_methods.png)

### FA3_FA3：五折每个方法全部1000点
- [留出试次 1：全部方法 Fisher 1000+1000 对照](figures/FA3_FA3_FOLD1_ALL1000_Fisher_METHODS.png)
- [留出试次 2：全部方法 Fisher 1000+1000 对照](figures/FA3_FA3_FOLD2_ALL1000_Fisher_METHODS.png)
- [留出试次 3：全部方法 Fisher 1000+1000 对照](figures/FA3_FA3_FOLD3_ALL1000_Fisher_METHODS.png)
- [留出试次 4：全部方法 Fisher 1000+1000 对照](figures/FA3_FA3_FOLD4_ALL1000_Fisher_METHODS.png)
- [留出试次 5：全部方法 Fisher 1000+1000 对照](figures/FA3_FA3_FOLD5_ALL1000_Fisher_METHODS.png)
- [该表示多方法严格Hit@5%图](figures/FA3_FA3_Hit5_all_methods.png)
- [该表示真实距离阈值曲线](figures/FA3_FA3_eps_curves_all_methods.png)

### PCA5_RAW5：五折每个方法全部1000点
- [留出试次 1：全部方法 Fisher 1000+1000 对照](figures/PCA5_RAW5_FOLD1_ALL1000_Fisher_METHODS.png)
- [留出试次 2：全部方法 Fisher 1000+1000 对照](figures/PCA5_RAW5_FOLD2_ALL1000_Fisher_METHODS.png)
- [留出试次 3：全部方法 Fisher 1000+1000 对照](figures/PCA5_RAW5_FOLD3_ALL1000_Fisher_METHODS.png)
- [留出试次 4：全部方法 Fisher 1000+1000 对照](figures/PCA5_RAW5_FOLD4_ALL1000_Fisher_METHODS.png)
- [留出试次 5：全部方法 Fisher 1000+1000 对照](figures/PCA5_RAW5_FOLD5_ALL1000_Fisher_METHODS.png)
- [该表示多方法严格Hit@5%图](figures/PCA5_RAW5_Hit5_all_methods.png)
- [该表示真实距离阈值曲线](figures/PCA5_RAW5_eps_curves_all_methods.png)

### FA5_RAW5：五折每个方法全部1000点
- [留出试次 1：全部方法 Fisher 1000+1000 对照](figures/FA5_RAW5_FOLD1_ALL1000_Fisher_METHODS.png)
- [留出试次 2：全部方法 Fisher 1000+1000 对照](figures/FA5_RAW5_FOLD2_ALL1000_Fisher_METHODS.png)
- [留出试次 3：全部方法 Fisher 1000+1000 对照](figures/FA5_RAW5_FOLD3_ALL1000_Fisher_METHODS.png)
- [留出试次 4：全部方法 Fisher 1000+1000 对照](figures/FA5_RAW5_FOLD4_ALL1000_Fisher_METHODS.png)
- [留出试次 5：全部方法 Fisher 1000+1000 对照](figures/FA5_RAW5_FOLD5_ALL1000_Fisher_METHODS.png)
- [该表示多方法严格Hit@5%图](figures/FA5_RAW5_Hit5_all_methods.png)
- [该表示真实距离阈值曲线](figures/FA5_RAW5_eps_curves_all_methods.png)

## 单独散点图（Fold1，显示全部1000个训练样本/域）

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

## 数值存档和严谨性
全部新模型的正交R、每条测试样本真实五指误差、训练配置和数值状态都在下列JSON。上表按相同试次折比较，不能把同一被试的两种子当独立受试者；也不能将事后阈值70%作为预注册泛化准确率。

- [所有SA/WA/DAD-like逐折逐种子原始数据](all_SA_WA_DAD_run_records.json)
- [逐试次完整对照CSV](all_fivefold_methods.csv)
- [汇总表CSV](summary_SA_WA_DAD_HiWA.csv)
- [实际物理误差阈值曲线CSV](dense_physical_tolerance_all_methods.csv)
- [1000点图像数目及K4颜色审计](scatter_count_audit.json)
- [此前额外Identity/ICP/单层OT](../additional_baselines/ADDITIONAL_UNSUPERVISED_BASELINES_ZH.md)
- [此前80次HiWA共享P/Q层级目标](../shared_hierarchical_full250/SHARED_HIERARCHICAL_EVALUATION_ZH.md)
- [DANDI原迁移代码源分支](https://github.com/noelabernathy400-bot/ot-hiwa/blob/experiment/dandi-pca3-fa3-vs-pca4-fa4-20261119/projects/dandi-000138-hiwa-4d/code/closeout_dandi_sa_wa_dad_baselines.py)

这是一组特定参数的实际基线复核，不是作者原SA/WA/DAD所有超参的最优性或统计跨人群显著性证明。
