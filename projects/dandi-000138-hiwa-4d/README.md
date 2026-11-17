# DANDI 000138 × HiWA：四维神经—手部运动对齐

**项目唯一入口：** [实验日志](EXPERIMENT_LOG.md) · [冻结协议](PROTOCOL.md) · [实验代码](code/) · [真实数据结果](results/)

## 核心研究问题
神经放电的 4D PCA 表示和原生四维手部运动 \((x,y,v_x,v_y)\) 是否具有可由无监督 Hard HiWA 恢复的共通几何？分布 OT 误差的下降是否意味着同步神经—行为误差下降？

## 当前结果（2026-11-17）
- **真实数据**：DANDI:000138，MC_Maze_Large，NWB 训练集，500 个试次，400 train / 100 test，每试次 35 个窗口。
- **硬聚类**：两域独立 KMeans(K=4)；训练 silhouette：神经 0.33961，运动 0.37032。
- **Hard HiWA（原版上传压缩包源码）**：5 个随机种子均通过 ADMM 残差审计（不是仅凭原版停止条件）。
- **无配对 OT NRMSE**：identity 0.66162 → Hard HiWA 0.63127 ± 0.01717（5 seeds）。
- **同步配对 NRMSE**：identity 1.33890 → Hard HiWA 1.37299 ± 0.08347（5 seeds）：平均没有改善。
- **不能宣称**：Hard HiWA 已恢复正确逐时刻神经—运动映射，或 Soft-GCOT 优于 Hard。
详见 [实验日志](EXPERIMENT_LOG.md)。

## 源码与来源
- [code/hiwa_original.py](code/hiwa_original.py)：**逐字复制**自用户上传的 GCOT-reproducible-20261018.zip 内 `reference/HiWA/PyHiWA/src/hiwa.py`，SHA-256 `4419e19004b06c7840d187ddb12a19b2ad97974dd28ad1dc9510d8ead741da6a`；没有修改求解器。**不使用旧仓库 src/hiwa/**。
- [code/run_hard_clustering.py](code/run_hard_clustering.py)：数据预处理和四维 KMeans；图片仅为二维投影。
- [code/run_hard_hiwa.py](code/run_hard_hiwa.py)：调用原版 HiWA；测试前后同尺度测量两类 NRMSE；额外审计最终局部旋转、全局旋转和 OT 边缘误差。
- [code/summarize_results.py](code/summarize_results.py)：绘图和均值/标准差。
- 全压缩包的其他 GCOT 项目源码仍保存在 [原始快照（实验分支）](https://github.com/noelabernathy400-bot/ot-hiwa/tree/experiments/dandi-000138-4d-hard-clustering-20261117/research/GCOT-reproducible-20261018/)；该目录不是本实验的算法入口。

**数据协议**：[DANDI 官方数据版本 0.220113.0407](https://dandiarchive.org/dandiset/000138/0.220113.0407)。原始 NWB 约 149 MB，不放入 Git；完整处理输出及实验图存入 `results/`。

## 本地复现
```bash
pip install numpy scipy scikit-learn h5py matplotlib
python code/run_hard_clustering.py --nwb /path/to/sub-Jenkins_ses-large_desc-train_behavior+ecephys.nwb --output results/clustering --seed 42 --k 4
OPENBLAS_NUM_THREADS=1 python code/run_hard_hiwa.py --clusters results/clustering/clustered_4d.npz --output results/hard_hiwa --seeds 42 43 44 45 46
python code/summarize_results.py
```
