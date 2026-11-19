# HySeR：全部方法的1000点实证综合对照（九种方法，含随机基线）

**全部数据均来自已实际运行的实验**。四个表示、每折1000源点+1000目标点训练、250测试点、五折，物理命中按照折内训练非对应五指力量距离第5%/20%分位阈值。模型均没有使用同步真值配对训练。

**散点图：** [查看20张五折全方法1000点拼图，以及30张Fold1各方法单独1000+1000点图](SA_WA_DAD_HYSER_1000_FINAL_REPORT_ZH.md)。四种颜色只代表源域独立伪簇；对齐结果投到目标训练集拟合的Fisher二维平面，仅供展示，误差在原始五指力量空间计算。

## PCA3_PCA3

| 方法 | 实际合格运行 | Hit@5% ↑ | Hit@20% ↑ | 平均五指L2 ↓ |
|---|---:|---:|---:|---:|
| Random | 5/5 | 5.25% | 20.20% | 0.31632 |
| Identity | 5/5 | 0.56% | 14.00% | 0.31506 |
| ICP | 10/10 | 0.92% | 16.96% | 0.31075 |
| SingleOT | 10/10 | 1.48% | 21.92% | 0.29062 |
| SA | 5/5 | 0.40% | 25.92% | 0.29848 |
| WA | 10/10 | 7.12% | 29.88% | 0.27036 |
| DAD_like_3D | 5/5 | **14.16%** | 34.40% | 0.25812 |
| Hard HiWA | 10/10 | 11.72% | 34.28% | **0.25396** |
| Soft HiWA | 10/10 | 10.84% | 33.88% | 0.26199 |

[PCA3_PCA3 完整五折1000点散点图](SA_WA_DAD_HYSER_1000_FINAL_REPORT_ZH.md)

## FA3_FA3

| 方法 | 实际合格运行 | Hit@5% ↑ | Hit@20% ↑ | 平均五指L2 ↓ |
|---|---:|---:|---:|---:|
| Random | 5/5 | 5.25% | 20.20% | 0.31632 |
| Identity | 5/5 | 0.80% | 18.80% | 0.30216 |
| ICP | 10/10 | 1.84% | 23.24% | 0.28812 |
| SingleOT | 10/10 | 2.72% | 25.36% | 0.29243 |
| SA | 5/5 | 1.44% | 16.88% | 0.31795 |
| WA | 10/10 | 4.00% | 32.80% | 0.27820 |
| DAD_like_3D | 5/5 | **5.92%** | 22.88% | 0.29924 |
| Hard HiWA | 10/10 | 3.96% | 35.16% | 0.28545 |
| Soft HiWA | 10/10 | 3.84% | 33.72% | **0.27095** |

[FA3_FA3 完整五折1000点散点图](SA_WA_DAD_HYSER_1000_FINAL_REPORT_ZH.md)

## PCA5_RAW5

| 方法 | 实际合格运行 | Hit@5% ↑ | Hit@20% ↑ | 平均五指L2 ↓ |
|---|---:|---:|---:|---:|
| Random | 5/5 | 5.25% | 20.20% | 0.31632 |
| Identity | 5/5 | 0.56% | 13.68% | 0.30002 |
| ICP | 10/10 | 0.96% | 28.32% | 0.28689 |
| SingleOT | 10/10 | 4.80% | 29.96% | 0.27878 |
| SA | 5/5 | 1.04% | 19.04% | 0.28405 |
| WA | 10/10 | 10.40% | 30.68% | 0.27437 |
| Hard HiWA | 10/10 | **12.52%** | 34.84% | 0.26070 |
| Soft HiWA | 10/10 | 12.36% | 34.12% | **0.25767** |

[PCA5_RAW5 完整五折1000点散点图](SA_WA_DAD_HYSER_1000_FINAL_REPORT_ZH.md)

## FA5_RAW5

| 方法 | 实际合格运行 | Hit@5% ↑ | Hit@20% ↑ | 平均五指L2 ↓ |
|---|---:|---:|---:|---:|
| Random | 5/5 | 5.25% | 20.20% | 0.31632 |
| Identity | 5/5 | 0.32% | 17.60% | 0.28564 |
| ICP | 10/10 | 0.68% | 18.16% | 0.29323 |
| SingleOT | 10/10 | 7.40% | 31.56% | 0.27060 |
| SA | 5/5 | 0.56% | 18.08% | 0.29234 |
| WA | 10/10 | **10.32%** | 30.76% | 0.28194 |
| Hard HiWA | 10/10 | 8.84% | 36.40% | 0.26823 |
| Soft HiWA | 10/10 | 6.88% | 37.00% | **0.26485** |

[FA5_RAW5 完整五折1000点散点图](SA_WA_DAD_HYSER_1000_FINAL_REPORT_ZH.md)

## 关键结论

- **PCA3：**改编DAD-like的严格Hit@5%为14.16%，超过Hard HiWA的11.72%，在4/5留出折表现更高；但DAD-like平均五指L2=0.25812，略差于Hard的0.25396。
- **FA3：**DAD-like的Hit@5%=5.92%，高于Hard的3.96%，但DAD-like平均L2=0.29924，差于Soft的0.27095；仅2/5折DAD-like优于Hard，优势不稳定。
- **PCA5：**Hard HiWA的Hit@5%=12.52%，高于WA的10.40%；Soft HiWA的平均L2=0.25767略优于Hard的0.26070。
- **FA5：**WA的Hit@5%=10.32%，高于Hard HiWA的8.84%，但WA平均L2=0.28194，差于Hard的0.26823和Soft的0.26485；WA在3/5折上优于Hard。

因此**HiWA并非在所有真实物理指标或表示上战胜其他方法**；Soft也尚未显示稳定优于Hard。需要区分严格阈值Hit、宽松Hit和平均L2三种数学上不同的质量评价。此数据仅一个受试者一次会话，不能得出跨人群显著性结论。

## 方法来源和边界

SA与WA直接以DANDI归档的研究实现为依据。SA采用标准化全基PCA的确定性旋转；WA采用200轮退火和正交Procrustes，但本Hyser版本为**每轮250点小批量、每四轮覆盖完整1000点、重复50个遍历周期**，不能称作原DANDI 160点整批算法的逐步复现。DAD-like是DANDI已明确标注的三维密度KL代理+128网格+Powell改编版，**不是官方DAD**，没有生成假造的5D DAD结果。

[DANDI实验分支真实方法源码](https://github.com/noelabernathy400-bot/ot-hiwa/blob/experiment/dandi-pca3-fa3-vs-pca4-fa4-20261119/projects/dandi-000138-hiwa-4d/code/closeout_dandi_sa_wa_dad_baselines.py)

## 完整证据

- [九种方法完整同口径CSV](ALL_NINE_METHODS_COMPLETE_1000_POINT_COMPARISON.csv)
- [全部1000点散点图库、各方法图及原始数据](SA_WA_DAD_HYSER_1000_FINAL_REPORT_ZH.md)
- [新增70个SA/WA/DAD-like实际运行的R与逐点原始五指误差](all_SA_WA_DAD_run_records.json)
- [SA/WA/DAD-like与Hard/Soft五折评分CSV](all_fivefold_methods.csv)
- [原Identity/ICP/SingleOT对照报告](../additional_baselines/ADDITIONAL_UNSUPERVISED_BASELINES_ZH.md)
- [此前全部1000训练点HiWA层级运输评价](../shared_hierarchical_full250/SHARED_HIERARCHICAL_EVALUATION_ZH.md)
