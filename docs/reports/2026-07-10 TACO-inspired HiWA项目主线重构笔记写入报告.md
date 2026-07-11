# 2026-07-10 TACO-inspired HiWA 项目主线重构笔记写入报告

## 日期

2026-07-10

## 目标

将本次对 HiWA / TACO / ROCA / CC-HiWA / PBMC 项目主线的重新梳理写入 Obsidian vault，形成后续科研路线和下一步 Codex 执行任务的正式笔记。

## 改动文件

- `Projects/最优传输/最优传输/文本笔记/代码方面/TACO-inspired HiWA 项目主线重构与下一步计划.md`

## 关键决定

- 当前项目主线收束为：HiWA baseline → frozen Soft-Prototype HiWA → naive soft assignment 诊断 → $O(3)$ determinant branch instability → ROCA → TACO-style joint prototype optimization。
- ROCA 定位为 TACO-inspired Soft-HiWA 中的 orientation stabilization module，而不是整个项目的全部。
- PBMC / CC-HiWA / GI-CC-HiWA 定位为外部迁移预研和负结果诊断，暂不作为当前论文主线。
- 下一步优先做固定 $X,Y$ 的最小 prototype / assignment / alignment 联合优化，不做 encoder 端到端训练、不扩展新数据集、不继续大量 beta 扫描。

## 风险或跳过事项

- 本次只写入主线整理笔记，没有修改代码、实验结果或原始资料。
- 文中涉及 TACO 原论文的理解基于现有项目笔记，后续写论文时仍需回到原文逐条核验。
- 本次没有更新 `System/Handoff/AI交接说明.md`，因为当前是一次中等规模笔记写入，不是长任务结束交接。

## 下一步建议

按照笔记第 10 节提示词，让 Codex 实现 `run_joint_prototype_hiwa.py` 和最小 TACO-style joint prototype optimization 实验，先在 seeds 50-54 上验证机制。
