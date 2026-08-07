---
title: "GC-HiWA v2 仓库说明与复现指南"
aliases:
  - "OT-HiWA repository guide"
  - "GC-HiWA v2 reproducibility guide"
tags:
  - "research/optimal-transport"
  - "reproducibility"
created: 2026-08-07
status: "current"
---

# GC-HiWA v2 仓库说明与复现指南

_本文件说明当前最优传输研究的目标、已验证结果、目录边界与复现入口。_

---

## 📋 项目定位

GC-HiWA v2 是一个带适用性诊断的无监督坐标对齐方法。它不假设最优传输能在任意两域间创造语义映射；只有当两域共享群组几何、且坐标间存在预先声明的结构化正交关系时，才恢复旋转。条件不满足时，方法输出弃权及其原因。

对三维姿态，允许的变换不是任意 $O(3J)$，而是一个对所有关节共享的相机旋转：

$$
\mathcal{R}_{\mathrm{pose}}
=\{I_J\otimes R_3:R_3\in SO(3)\}.
$$

完整模型、推导、ROCA 分支诊断与限制见 [[research/GC-HiWA_v2_完整研究笔记_Obsidian]]。

## 🎯 当前可信结论

| 证据层级 | 已确认结果 | 不能推出的结论 |
|---|---|---|
| 合成边界 | 15/15 合同满足运行接受；25/25 预声明违例弃权；谱伪装非正交 5/5 弃权 | 不能证明任意真实跨域任务可对齐 |
| Panoptic Studio | 三组受控相机配置共 11 次均收敛；最大旋转误差 0.0308；Recall@1=1 | 不能等同于跨设备动作识别泛化 |
| 冻结下游代理 | 一个受控窗口由 0.333 恢复至 0.864 | 状态来自源姿态而非人工动作标签 |
| 负结果审计 | Indy-Loco、PBMC、PAMAP2、Paderborn 等不满足当前几何合同或独立验证要求 | 不能把负结果写成方法在这些任务上优于基线 |

## 📚 目录与保留规则

```text
src/                    方法实现
tests/                  回归测试与资格检查测试
experiments/
  synthetic/            合成边界生成、运行与证据审计
  panoptic/             受控真实相机坐标实验
  meta_analysis/        独立的鼻咽癌试验层 Meta 复算脚本
  results/              结构化、可追溯的实验 JSON 与审计记录
docs/
  research/             数学合同、实验解释、数据审计与研究笔记
Writing/                论文源文件和参考文献；不包含编译缓存
data/raw/               本地原始数据；被 Git 忽略
```

`experiments/results/` 是结果的权威记录；可视化可由其再生成。`Writing/build/`、`deliverables/` 和 `experiments/figures/` 都是可再生成或过时的导出副本，因此不进入版本库。原始数据也不上传，以避免大文件、许可和可追溯性问题。

## ⚙️ 复现主要证据

在仓库根目录执行：

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -m pytest tests -q
python experiments/synthetic/verify_gc_hiwa_v2_evidence.py
```

第一条命令运行回归测试；第二条命令审计已冻结的 GC-HiWA v2 证据。单线程环境变量仅规避 Windows 下的运行时不稳定性，不改变方法超参数。

若本地已有 Panoptic 原始数据，可运行：

```powershell
python experiments/panoptic/run_camera_coordinate_alignment.py
```

运行器不会把相机外参、同步帧、ID 或标签交给拟合器；它们只用于最终评分。

## 🔍 核心实现入口

| 组件 | 位置 | 作用 |
|---|---|---|
| 软群组 | `src/cc_hiwa/soft_groups.py` | 全局标量标准化、原型与软归属 |
| 层级 OT | `src/cc_hiwa/soft_hiwa.py` | 组内 Sinkhorn、组间 OT、ADMM 与结构化旋转 |
| 资格检查 | `src/cc_hiwa/alignment_qualification.py` | 边际误差、正交误差、拟合比、谱与多启动门槛 |
| ROCA | `src/cc_hiwa/roca_qualification.py` | $O(3)$ 分支候选、定向单纯形与模糊弃权 |
| 合成边界 | `src/cc_hiwa/synthetic_boundary.py` | 可恢复与不可恢复情形生成 |
| 证据审计 | `experiments/synthetic/verify_gc_hiwa_v2_evidence.py` | 一键复核主要结论 |

## ⚠️ 研究边界

- 可恢复性定理假设正确耦合，不是 OT--ADMM 全局收敛证明
- 固定阈值是工程资格合同，不是普适统计显著性阈值
- Panoptic 证明受控几何恢复，不证明真实跨设备语义泛化
- UWA3DII 完整数据尚未接入，不能声称真实跨设备动作识别提升
- 自动群组数、联合优化和编码器模块没有展示稳定、独立增益，故未进入主结果

## 🔗 关键文档

| 用途 | 文件 |
|---|---|
| 完整数学与结果笔记 | `docs/research/GC-HiWA_v2_完整研究笔记_Obsidian.md` |
| 数学合同与 Panoptic 验证 | `docs/research/GC-HiWA_v2_数学合同、适用性诊断与Panoptic验证_2026-07-23.md` |
| 论文主稿 | `Writing/GC-HiWA_v2_论文草稿.tex` |
| 最终 PDF | `Writing/GC-HiWA_v2_final.pdf` |
| 论文质量边界 | `Writing/终稿审查清单.md` |
| 鼻咽癌 Meta 分析 | `docs/research/鼻咽癌_辅助卡培他滨_随机证据更新Meta_2026-07-27.md` |
