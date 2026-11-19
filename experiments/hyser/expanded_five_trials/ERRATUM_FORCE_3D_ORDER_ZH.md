# Hyser 三维目标域力量通道勘误（2026-11-19）

**旧的五试次三维数据** `hyser_s01_s1_random_5trials_features_cv.zip` 由 `expanded_five_trials/fetch_prepare.py` 中
`FORCE_INDICES=(0,2,1)` 提取。根据五次原始 WFDB 文件头，实际力通道顺序为
`thumb,index,middle,ring,little`，因此旧三维目标数组 `Y_force_thumb_middle_index_3` 的物理次序**确实是**
`thumb,middle,index`。

旧的少数文本报告误写成「拇指、食指、中指」或 `thumb/index/middle`。
这里显式勘误并同步更正中文报告与物理标签结果中的列名。**只更正名称，旧模型的三维数组、正交变换、W2、P/Q及全部数值结果一律没有改变**。

完整五指新数据 `hyser_s01_s1_5fingers_5D_folds_v1.zip` 来自官方 `.hea` 文件明确指定的五指名字，
顺序为 `thumb,index,middle,ring,little`，已在发布数据的
`FIVE_FINGER_SOURCE_CHECK.json` 中逐试次核实。因此三维实验、五维实验不可直接混用力量维度的标签。

**没有真实五动作标签**：Hyser Random Task 的不同手指可以同时收缩，K=5只是统计簇数而不是五个单指运动真值。
强行按簇编号映射至手指违反实验可解释性要求。
