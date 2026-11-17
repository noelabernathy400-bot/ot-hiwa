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


## 2026-11-17 PCA 后标准化审计与正式报告

**本目录根部已经保存 XeLaTeX 编译的四页实验报告：**

- [Hyser_Hard_HiWA_实验报告.pdf](Hyser_Hard_HiWA_实验报告.pdf)：真实数据介绍、完整预处理、全 250 点 K-means / GMM 三维簇状图、HiWA 参数与 R²/相对平方误差
- [LaTeX 源码](Hyser_Hard_HiWA_实验报告.tex)：可在仓库 GitHub Actions 中重新编译
- [标准化与对齐详细结果](results/POST_PCA_NORMALIZATION_AUDIT.md)
- [完整数值 JSON](results/POST_PCA_NORMALIZATION_AUDIT.json)
- [直接调用用户新 GCOT 分支的复现脚本](scripts/run_hiwa_postpca_normalized.py)

关键结论：之前 PCA 后的全量点云已经进行协方差白化；进一步检查逐维 Z-score 并按新 GCOT 源码结构复算后，当前零时延三维表示上的 Hard HiWA 配对 R² 仍为负。当前运行仅满足作者旋转变化停止条件，尚未满足严格 ADMM 共识。请参阅本次详细报告，不要将源码移植版的数字标成远程原版字节级执行结果。

## 2026-11-17：硬分组后点云与 PDF 排版修订

已重新编译仓库根部 [Hyser_Hard_HiWA_实验报告.pdf](Hyser_Hard_HiWA_实验报告.pdf)。
实验报告图像现为：早期200点真实硬聚类四联图（基于记录的200点重建原图布局）、正式250点完整白化独立硬簇、
[Hard HiWA 对齐后三维叠加](figures/all250_after_hard_hiwa_3D_overlay.png) 与
[二维叠加](figures/all250_after_hard_hiwa_2D_overlay.png)。

图片由 [render_report_figures.py](scripts/render_report_figures.py) 从已保存的真实实验绘图点重建，属于**仅用于展示的坐标量化版**；
评估误差不在这些量化数据上计算，仍以原始全精度数值为准。
PDF 中的早期图与当前250点实验在数据量及缩放上不同。用户上传的原始PNG直接用于对话中交付的 PDF，
仓库自动构建则采用相同原始200点的可复现绘图数据重建四联布局。
