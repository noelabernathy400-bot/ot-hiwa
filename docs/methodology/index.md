---
date: 2026-06-30
type: index
tags: [index, catalog]
ai-first: true
---

# Vault Index — 最优传输研究

> 自动生成于 2026-06-30。每个条目包含一句话描述。

## HiWA 笔记（13章 + 总览 + deep-research）

- [[HiWA精读笔记_Obsidian兼容版]] — HiWA 论文完整精读总览：问题动机、层级 OT 框架、数学设定
- [[第一章 论文主要公式解释]] — 分布对齐 → P/Q_ij/R 三变量优化的公式推导
- [[第二章 正交procrustes问题的求解]] — Orthogonal Procrustes 闭式解（SVD 方法）
- [[第三章 论文中的ADMM项解释]] — ADMM 共识优化在 HiWA 中的具体形式
- [[第四章 拉格朗日乘子法以及在论文中的应用]] — 拉格朗日乘子法推导与 ADMM 对偶变量
- [[第五章 ADMM以及加熵]] — 熵正则化 ADMM：为什么加熵、如何影响优化
- [[第六章 sinkhorn处理更新Qij和Pij]] — Sinkhorn 算法更新点级传输 Q_ij 和簇对应 P
- [[第七章 Algorithm 1 理解]] — HiWA 主算法伪代码逐行解读
- [[第八章 Theorem 4.1 对应消歧准则]] — 簇对应可识别性的理论条件
- [[第十章 Theorem 4.2 对齐误差界]] — 全局对齐误差的 perturbation bound
- [[第十一章 Lemma 4.3 最坏几何结构]] — 最坏情形子空间几何结构分析
- [[第十二章 数值实验总览与合成实验]] — 合成低秩高斯混合数据实验
- [[第十三章 神经解码实验 Figure 2]] — 运动皮层神经解码实验
- [[deep-research-report_obsidian_fixed]] — HiWA 深度研究报告（执行摘要 + 数学设定 + 关键公式推导）

## Taco 笔记（5章）

- [[Taco精读笔记_00_总览与问题背景]] — Taco 论文总览：Geometry-Free 吸附构型筛选
- [[Taco精读笔记_01_化学任务背景]] — 化学背景：吸附构型、DFT、催化剂筛选
- [[Taco精读笔记_02_问题定义]] — 形式化问题定义：文本-only 推理 + 几何知识蒸馏
- [[Taco精读笔记_03_几何模型与文本模型]] — 几何编码器（EquiformerV2）与文本编码器架构
- [[Taco精读笔记_04_整体流程图与训练推理]] — Figure 1 完整 pipeline：训练/推理双路径

## 方法论与工具

- [[读论文的方法论笔记]] — 9 步科研论文阅读框架（问题→结构→公式→实验→延伸）
- [[snippets配置]] — Obsidian LaTeX Suite 自定义片段（`@a`→`\alpha` 等）
- [[deepseek接入方案研究_修复版]] — DeepSeek API 接入 Obsidian 笔记清洗工作流方案

## 代码

- `hiwa-matlab/` — HiWA MATLAB 官方实现（`HiWA.m`, `HiWASSC.m`, `WA.m`, demo 脚本）
- `PyHiWA/` — HiWA Python 移植版（`hiwa.py`: Sinkhorn, Procrustes, HiWA 类）

## Excalidraw 图

- [[Excalidraw/Drawing 2026-06-28 17.43.50.excalidraw]]
- [[Excalidraw/Drawing 2026-06-28 18.29.50.excalidraw]]
- [[Excalidraw/Drawing 2026-06-28 18.39.42.excalidraw]]
