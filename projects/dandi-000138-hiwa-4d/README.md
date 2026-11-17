# DANDI 000138 × HiWA：四维神经—手部运动对齐

**项目唯一入口：** [实验日志](EXPERIMENT_LOG.md) · [冻结协议](PROTOCOL.md) · [实验代码](code/) · [真实数据结果](results/)

## 四维 Hard HiWA 正式技术报告（XeLaTeX 编译）

- **[阅读 13 页 PDF：数据来源、预处理、两域原始簇图、算法与失败诊断](四维HardHiWA实验与问题诊断.pdf)**
- [LaTeX 源码](四维HardHiWA实验与问题诊断.tex)（XeLaTeX / ctexart）
- [新增问题诊断结果（JSON + PNG）](results/diagnostics/) · [诊断代码](code/) · [更新后的实验日志](EXPERIMENT_LOG.md)
- [GitHub Actions 自动编译与可复现实验日志](https://github.com/noelabernathy400-bot/ot-hiwa/actions/workflows/dandi138-generate-report.yml)

**新诊断摘要：** 神经四维 PCA 表示并非没有运动信息——同一留出测试集的监督全线性 Ridge 映射得到标准 `R²=0.2868`，但监督**正交** Procrustes 只有 `R²=-0.0604`，原版无监督 Hard HiWA 五次均值 `R²=-0.8947`。此外，在固定目标评价窗口条件下，由训练验证选取的神经领先行为 140ms 给出监督 Ridge 测试 `R²=0.3902`；目标域实际簇质量与原版 HiWA 强制的均匀 25% 组质量不一致。**以上监督模型和时间分析仅是诊断，不代表改进了 HiWA，不能据此断言存在已确定的生理时间延迟。**

## 核心研究问题
神经放电的 4D PCA 表示和原生四维手部运动 \((x,y,v_x,v_y)\) 是否具有可由无监督 Hard HiWA 恢复的共通几何？通过原版 rMSE（代理参照）与标准 held-out R² 评价，是否真实改善运动预测？

## 当前结果（2026-11-17）
- **真实数据**：DANDI:000138，MC_Maze_Large，NWB 训练集，500 个试次，400 train / 100 test，每试次 35 个窗口。
- **硬聚类**：两域独立 KMeans(K=4)；训练 silhouette：神经 0.33961，运动 0.37032。
- **Hard HiWA（原版上传压缩包源码）**：5 个随机种子均通过 ADMM 残差审计（不是仅凭原版停止条件）。
- **HiWA rMSE（代理值，非真值）**：identity 1.96470 → Hard HiWA **1.91621 ± 0.37054**（5 seeds）；原始真值旋转不存在，评估参照是仅使用训练同步配对得到的正交 Procrustes。
- **标准测试 R²**：identity -0.79649 → Hard HiWA **-0.89472 ± 0.22508**（5 seeds）：平均更差，五个 seed 均为负数。
- 原版 `_eval_R2` 未采用平方残差，不能把它返回值当成标准决定系数；评估脚本计算修正的标准 R²，**不修改原版 HiWA 求解器**。
- **不能宣称**：Hard HiWA 已恢复正确逐时刻神经—运动映射，或 Soft-GCOT 优于 Hard。
详见 [实验日志](EXPERIMENT_LOG.md)。

## 源码与来源
- [code/hiwa_original.py](code/hiwa_original.py)：**逐字复制**自用户上传的 GCOT-reproducible-20261018.zip 内 `reference/HiWA/PyHiWA/src/hiwa.py`，SHA-256 `4419e19004b06c7840d187ddb12a19b2ad97974dd28ad1dc9510d8ead741da6a`；没有修改求解器。**不使用旧仓库 src/hiwa/**。
- [code/run_hard_clustering.py](code/run_hard_clustering.py)：数据预处理和四维 KMeans；图片仅为二维投影。
- [code/run_hard_hiwa.py](code/run_hard_hiwa.py)：调用原版 HiWA，历史 NRMSE 仅作为归档；额外审计最终局部旋转、全局旋转和 OT 边缘误差。
- [code/evaluate_hiwa_two_metrics.py](code/evaluate_hiwa_two_metrics.py)：**当前主评价入口**；用冻结的 HiWA 旋转和测试集计算 rMSE 代理值及标准 R²。
- [code/run_hard_hiwa_2metric.py](code/run_hard_hiwa_2metric.py)：以后完整实验统一使用此包装入口。
- [code/summarize_results.py](code/summarize_results.py)：**历史 NRMSE 绘图**，保留用于复核，不再作为主结论。
- 全压缩包的其他 GCOT 项目源码仍保存在 [原始快照（实验分支）](https://github.com/noelabernathy400-bot/ot-hiwa/tree/experiments/dandi-000138-4d-hard-clustering-20261117/research/GCOT-reproducible-20261018/)；该目录不是本实验的算法入口。

**数据协议**：[DANDI 官方数据版本 0.220113.0407](https://dandiarchive.org/dandiset/000138/0.220113.0407)。原始 NWB 约 149 MB，不放入 Git；完整处理输出及实验图存入 `results/`。

## 本地复现
```bash
pip install numpy scipy scikit-learn h5py matplotlib
python code/run_hard_clustering.py --nwb /path/to/sub-Jenkins_ses-large_desc-train_behavior+ecephys.nwb --output results/clustering --seed 42 --k 4
OPENBLAS_NUM_THREADS=1 python code/run_hard_hiwa.py --clusters results/clustering/clustered_4d.npz --output results/hard_hiwa --seeds 42 43 44 45 46
python code/evaluate_hiwa_two_metrics.py --clusters results/clustering/clustered_4d.npz --alignment results/hard_hiwa/hard_hiwa_results.json --output results/hard_hiwa/hiwa_rmse_r2_evaluation.json
```
