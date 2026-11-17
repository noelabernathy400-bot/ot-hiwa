# 历史 Hyser Hard HiWA 探索脚本（非正式基线）

这里是为了让实验代码集中管理而整理的先前临时重实现代码：

- `hard_hiwa_adapted.py`：稳定化 Sinkhorn + K-means 簇 + 局部 OT/旋转 + ADMM。
- `scan_hiwa_mu.py`：探索 μ=0.05/0.5/2，22 轮。
- `run_mu2_extended.py`：μ=2、110 轮的未收敛扩展诊断。

脚本已将原临时机路径转换为本实验目录的 `output/`，并精简了重复绘图与输出代码。**并非此前本地脚本的字节级快照**，不得把重新运行的细微差异误当成原结果。请以 `../results/trial1_results.json` 和 `../EXPERIMENT_LOG.md` 为已记录数值。

下一阶段正式基线应使用 `../scripts/run_hard_original.py`。
