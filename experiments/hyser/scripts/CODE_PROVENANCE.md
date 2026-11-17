# 代码来源与验证等级

- **正式入口（新增）：** `scripts/prepare.py` 处理真实 WFDB 数据，`scripts/run_hard_original.py` 直接导入仓库内 `src.hiwa.hiwa.HiWA`，但**截至本提交尚未执行并核验该入口的新数值结果**。
- **早期 Hyser 历史代码：** `legacy/hard_hiwa_adapted.py`、`legacy/scan_hiwa_mu.py`、`legacy/run_mu2_extended.py` 是为仓库存档而整理的**可读移植版**，算法结构与原先本地同名脚本一致，但并非逐字复制（例如本地绝对路径、绘图输出、冗余字段已整理）。旧结果仍以 `results/trial1_results.json` 和 `EXPERIMENT_LOG.md` 为准。
- **参考风格 ADMM 审计：** 数值来源于此前单独执行的本地 `hyser_hiwa_audited_solver.py` 等脚本，结果已记录；该审计脚本尚未完整逐行迁移到仓库，不能据此认为仓库已直接复现了审计曲线。
- 为避免误用，**任何论文级 Hard-vs-Soft 比较，都以之后直接运行同仓库类、统一预处理和独立试次为前提**。
