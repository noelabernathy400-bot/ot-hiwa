# 代码来源

- 首轮试验曾运行本地独立重实现：`hyser_hiwa_experiment.py`、`hard_hiwa_adapted.py`、`scan_hiwa_mu.py`、`run_mu2_extended.py`，并非原项目中的 `HiWA` 类。
- 后续数值审计也由本地 `hyser_hiwa_audited_solver.py` 等独立脚本完成。旧实验数值及其不确定性已保存在笔记与 `results/trial1_results.json`。
- 此文件夹新加入 `prepare.py` 和 `run_hard_original.py`，后者**直接调用本仓库原类**。这两份是整理后的复现入口，不能宣称是历史脚本的逐字复制或已成功复跑。
- 历史脚本仍可从此前对话提供的归档获取；后续如需逐行复现旧实现，应把原文件作为 `legacy/` 另行保存并注明版本。
