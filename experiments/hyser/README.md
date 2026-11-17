# Hyser：高密度肌电与手指力量的跨模态 HiWA 对齐实验

> 状态：真实数据已预处理并画图；Hard HiWA 参考风格复测的 ADMM 达到 0.01 双残差标准，但跨模态对齐未成功；**原项目 `src/hiwa/hiwa.py` 的直接复跑待完成**。软分组尚未实验。

## 研究问题

在同步采集的高密度肌电（HD-sEMG）和手指力量数据中，是否存在足以支持**无监督分组最优传输**的共同低维几何结构？Hard HiWA 的失败原因是数值优化、样本聚类不一致，还是跨模态非等距？在控制其他条件后，Soft-GCOT 能否利用真实过渡/混合状态改善对应？

## 目录

- [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)：从数据来源、处理、分组、绘图、参数排查到全部阶段性结果的连续笔记。
- [DATA.md](DATA.md)：数据目录、版权/来源、下载与预处理说明。
- [figures/](figures/)：**由本次真实数据的 200 个训练点生成**的 SVG 二维 PCA 投影和三维视角投影（同域颜色代表 K-means 簇，不代表跨域配对）。
- [results/trial1_results.json](results/trial1_results.json)：真实试次的聚类、旧版本运行、修正风格复测、oracle 诊断结果，包含出处和限制。
- [scripts/prepare.py](scripts/prepare.py)：WFDB 文件 → 窗口特征 → 训练/留出数据 → K-means → 本地 PNG 绘图。
- [scripts/run_hard_original.py](scripts/run_hard_original.py)：**直接调用本仓库的** `src.hiwa.hiwa.HiWA` 复跑入口（提供但尚未产生可核验的新结果）。
- [requirements.txt](requirements.txt)：环境依赖。
- [NEXT_STEPS.md](NEXT_STEPS.md)：正式对照与软分组计划。

## 快速复现

从仓库根目录执行：

```bash
pip install -r experiments/hyser/requirements.txt
mkdir -p experiments/hyser/data
# 放入 subject01/session1/random 的 preprocess 与 force 的 .hea/.dat
python experiments/hyser/scripts/prepare.py --data-dir experiments/hyser/data --out-dir experiments/hyser/output --trials 1
python experiments/hyser/scripts/run_hard_original.py --out-dir experiments/hyser/output --seed 42
```

注意：上传到对话的力量数据文件曾命名 `random_force_sample1(1).dat`，请将其重命名为 `random_force_sample1.dat`。原始 `.dat` 未上传仓库，避免大型二进制记录进入 Git 历史。

**所有当前已记录的数值结果均来自 2026-11-17 的真实 Hyser trial1，本目录中的直接调用原类脚本为后续标准化复跑入口，不能把它的存在当作已经完成的结果。**

主要源码：[本仓库 Hard HiWA](../../src/hiwa/hiwa.py) · [SoftHiWA](../../src/cc_hiwa/soft_hiwa.py) · [原基线](../hiwa/run_taco_faithful_baseline.py)。
