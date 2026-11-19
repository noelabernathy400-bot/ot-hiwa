# DANDI:000138 项目完整归档索引（与 Hyser 完全隔离）

**归档策略**：此处只包含 DANDI 项目的文档、源代码引用、真实实验结果与图。**不复制 Hyser 实验、不覆盖旧 DANDI 原始/失败记录、不把历史泄漏实验伪装为无监督实验、不把原始148MB NWB直接塞进 Git。** GitHub Actions 必须识别根目录 `.github/workflows/` 下的工作流，因此其**可执行原件仍位于 GitHub 约定目录**，但本归档附有工作流快照，便于完全在 DANDI 子目录中查阅脚本。

## 一、阅读顺序

1. [DANDI 最终实验总结](FINAL_EXPERIMENT_SUMMARY_ZH.md)——从先验数据检查、算法实现、实验尝试、物理匹配、失败/成功到项目决策。
2. [下一次实验标准工作流](NEXT_EXPERIMENT_WORKFLOW_ZH.md)——先判断 OT 的物理/几何合理性，确定适用门槛、严格数据划分、图像边界，再进行 Hard/Soft/基线/参数曲线。
3. [数学定义、全部模型与参数表](METHODS_AND_PARAMETERS_ZH.md)——Hard/Soft/P/Q/R，SA/WA/改编 DAD；原始和内验配置分开；有真实分数和未跑模型分开。
4. [图表清单与真实源输入](FIGURES_EVIDENCE_MANIFEST.json)、[曲线源CSV](tables/DANDI_3D4D_REAL_PHYSICAL_TOLERANCE_CURVES.csv)——图表由真实 JSON 重绘，不是人为描点。

## 二、DANDI 原项目文件结构与证据

### 根目录研究协议

- `../README.md`：历史完整项目 README（含早期结论、原始测试的探索性结果）；**本归档的结论优先用于当前收尾判断**。
- `../PROTOCOL.md`：原始实验冻结参数和后来纠正的指标；阅读时注意不同日期协议不是同一个同时执行的方案。
- `../EXPERIMENT_LOG.md`：历史流水账；不能将旧记录中“下一步计划”误写为已运行。
- `../GROUP_LABEL_LEAKAGE_AUDIT.md` + `../HISTORICAL_GROUP_SUPERVISED_README.md`：关键失败审计与所有监督泄漏诊断的历史说明。
- `../CODE_AUDIT_AND_FIX_2026-11-19.md`：原作者公式/源码差异、漏实现接口、正向对照与修复边界。
- `../NEXT_STAGE_EXPERIMENT_PROTOCOL.md`：**历史计划**，后来部分步骤已在本分支完成，需以当前总结判定实际完成状态。

### 已保存的 DANDI 实验与源码

| 内容类型 | 原始路径（均在DANDI项目内） | 证据 |
|---|---|---|
| 全量输入缓存、聚类、标签独立审计 | `../results/clustering/`、`../code/run_hard_clustering.py` | 500×35点、训练/测试trial及原始4D |
| 早期原版HiWA | `../results/hard_hiwa/`、`../code/hiwa_original.py` | 5 seed 原始运行、P/R/收敛 |
| 严格独立 Hard/Soft及原版3方对照 | `../results/strict_unsupervised_hard_soft/`、`../code/run_strict_unsupervised_hard_soft.py` | 真无配对、不同类层级成本、Soft ADMM失败 |
| 训练软组被真实同步标签泄漏的历史诊断 | `../results/controlled_hard_vs_soft_PQ/`、`../results/soft_fixedgroup_vs_hard/`、`../results/source_only_fixed_groups*/`、`../results/boundary_adaptive_trial_cv/` | **禁止用于正式无监督胜负判断** |
| 实际原目标转换 | `../results/target_only_geometry_full/`、`../code/run_target_only_geometry_ablation.py` | 四种目标域处理和共同比较空间 |
| 正向控制/谱初始化/源码适配 | `../results/phase1_*/`、`../results/hiwa_repaired_full_pqr_dandi/`、`../results/hiwa_source_code_repair/`、`../results/author_matlab_code_audit/` | **求解器检验**，不是现实对应已恢复证据 |
| PCA4/FA4内部320/80严格验证 | `../results/PCA4_FA4_inner_validation/`、`../code/compare_pca4_fa4_inner_train_ot.py` | 每seed完整16块Q及P/R，原NWB重新拟合 |
| PCA3/FA3 vs 四维，原四维物理评分 | `../results/PCA3_FA3_vs_PCA4_FA4_inner_validation/`、`../code/compare_pca3_fa3_to_archived_4d.py` | 原NWB重做3D，实际匹配与残差 |
| 原始全点图库 | `../results/full_scatter_gallery/` | 17,500点、14,000点、3,500点分开陈述 |
| 3D/4D Fisher 训练内投影、物理载荷 | `../results/real3d4d_fisher_gallery/` | 最初**两色模态叠加**历史版本，保留为可视化修复前审计 |
| **正确的4组上色** Fisher2/3和真实3D | `../results/real3d4d_fisher_gallery_fourgroups/`、`../code/plot_real_3d_4d_fisher_gallery.py` | Hard/Soft各自四簇，源色保留、灰色目标背景；仅作图 |
| **真实 DANDI SA/WA/DAD-like** | `../results/closeout_baselines_real_DANDI/`、`../code/closeout_dandi_sa_wa_dad_baselines.py` | FA3 SA/WA/改编DAD及FA4 SA/WA，逐seed物理指标 |
| **归档图与CSV** | `figures/`、`tables/`、`../code/build_dandi_closeout_figures.py` | 全部由真实仓库JSON生成，历史tau泄漏图标红 |

### 从另一个 DANDI 专用实验分支补入的审计快照

源分支：`experiment/dandi-phase1-multiinstance-pilot-20261119`（2026-11-19 的 Phase1 多实例/合成几何审计），其对应的**11个新增 DANDI 文件**（3份结论Markdown、3份代码、5份workflow）原来没有并入3D/4D收尾分支。本归档在 `phase1c_readonly_snapshot/` 以**只读快照**保存这些代码与报告，并标出源分支，避免 Phase1C 证据遗失或被误认为当下真实运动匹配主结果。**Phase1C以合成/数值审计为主，真实数据结论仍以本报告中的 DANDI 主线为准。**

### 工作流文件说明

`workflows_snapshot/` 保存 GitHub 根目录 `.github/workflows/dandi138*.yml` 的**全部 DANDI 专用工作流源码快照**。执行本身仍需位于 GitHub 标准路径的原件；快照用于报告可追溯，不能作为可直接调度的工作流。这是 GitHub 固定运行目录约束，**不是把 Hyser 数据放进DANDI**。

## 三、原始文件不入仓库，如何复现？

官方原始版本：`https://api.dandiarchive.org/api/dandisets/000138/versions/0.220113.0407/assets/`；所需 NWB 为 `sub-Jenkins_ses-large_desc-train_behavior+ecephys.nwb`，下载字节148,590,536，SHA256 `1188ddf5b822dd9ac49afb8916f4b42a85867613cf7f169d0b04a1d1aeecd700`。最近3D拟合、Fisher和新增基线工作流均在下载后严格校验此SHA，防止混入别的数据或对旧PCA缓存二次PCA。

- [已成功执行的原162神经 PCA3/FA3 3D-vs-4D 实验](https://github.com/noelabernathy400-bot/ot-hiwa/actions/runs/44511150437)。
- [已成功执行的四簇修正 Fisher 图](https://github.com/noelabernathy400-bot/ot-hiwa/actions/runs/44516265748)。
- [已成功执行的真实 SA/WA/改编DAD基线](https://github.com/noelabernathy400-bot/ot-hiwa/actions/runs/44521819255)。

**使用限制**：本研究不是“Soft 在 DANDI 上赢得了真实运动对齐”；实验结论与前提限制见总报告。历史数据、超参曲线、模型与求解失败均保留，结果不按漂亮程度删选。
