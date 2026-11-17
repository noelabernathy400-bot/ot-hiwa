# 分组信息泄漏专项审计（2026-11-17）

## 核心判断

**已确认：最初双域独立 K-means 的 Hard HiWA 输入无此组标签复制；后续源域唯一 K-means 生成的训练文件则把配对可视化标签复制到目标域，属于组级监督诊断，不是严格无监督。**

### 逐项代码定位

| 文件 | 关键证据 | 判断 |
|---|---|---|
| code/run_hard_clustering.py | 对源域 x 和目标域 y 分别调用 _cluster；results/clustering/clustered_4d.npz 保留两套独立 neural_groups / hand_groups | 独立分组，符合要求 |
| code/run_hard_hiwa.py | 分别读取 neural_groups 和 hand_groups，分开抽样然后传入原版 HiWA | 入口本身未复制；必须检查输入文件 |
| code/source_only_fixed_group_pipeline.py:76–85 | target_labels = labels.copy()，hand_groups = target_labels，shared_labels = labels | **确证泄漏源** |
| .github/workflows/dandi138-source-only-fixed-groups.yml | 训练输入为结果目录下的 fixed_groups_4d.npz | **确证使用了上述泄漏文件** |
| code/run_soft_fixedgroup.py:140–142,174–178 | target_soft = soft.copy()；B = target_soft[iy] | **确证 Soft 的 B 从源域复制** |
| code/run_controlled_soft_hard_ablation.py:163–166,224–228 | 同一份 Aall 赋给两域训练点 | **组级监督诊断** |
| code/run_boundary_adaptive_ablation.py:236–238,101–103 | 同源 Aall 供源/目标使用 | **组级监督诊断** |
| code/joint_ot_guided_membership.py | 目标分组权重由源域标签和源域原型计算 | **有同一泄漏，尚未构成有效无监督结果** |
| code/evaluate_hierarchical_ot.py | 测试同编号组的 OT 使用继承目标标签 | 只能称已知同组对应的诊断指标 |
| code/hiwa_original.py:165,202 | P 初始均匀 1/16，每轮调用 _sinkhorn_clusters 自主更新 | **未发现固定 P=I4/4** |
| code/zip_gcot/grouping.py | soft_groups 可分别对 X 和 Y 拟合独立原型 | 原始模块无需 B=A |

## 结果归档标识

**组级监督诊断（不作无监督性能结论）**：results/source_only_fixed_groups/、results/source_only_fixed_groups_outer500/、results/soft_fixedgroup_vs_hard/、results/controlled_hard_vs_soft_PQ/、results/boundary_adaptive_trial_cv/，以及它们的 OT 图和比较报告。保留原始数据、图表及指标，不覆写。

**可继续作为原始独立分组流程审计的输入及结果**：results/clustering/clustered_4d.npz、results/hard_hiwa/。数值残差问题仍需独立审计；当前这里只评价标签泄漏。

## 最小纠正与要求

训练通道：Hard 由本域 X、Y 的训练特征分别独立获得簇标签；Soft 必须分别在 X、Y 上学习 A、B；P 由 OT 优化，不能利用配对确定簇排列。评估和绘图通道：只有拟合结束后才可用同步索引统一颜色以及计算配对误差。两条通道不可混用。

为阻止误重跑，code/run_hard_hiwa.py 默认拒绝带有 shared_labels 且 neural_groups==hand_groups==shared_labels 的泄漏 NPZ。历史组级监督诊断只有显式添加 --allow-group-supervised-diagnostic 才可重现，并将该状态记录进输出；两个旧相关工作流的标题改为 DIAGNOSTIC ONLY。原版 HiWA/GCOT 求解器本身没有改动。

**数据与其他设置不变**：原 DANDI 000138 四维缓存仍是500 trials/17500 windows，400/100 trials 分割；先前误引入的250对样本额外检查工作流已经删除。本次不实施新的250对数据试验，也不重训或修改求解器参数。
