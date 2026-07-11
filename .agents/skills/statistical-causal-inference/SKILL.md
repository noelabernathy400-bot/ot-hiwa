---
name: statistical-causal-inference
description: Use for serious applied statistics and causal inference projects, especially observational panel/time-series data, industry linkage analysis, causal graphs, Granger-style temporal evidence, DiD, synthetic control, IV, DML, mediation, sensitivity analysis, and counterfactual policy interpretation.
---

# 统计因果推断

## 何时使用

- 用户目标不是只预测，而是解释“为什么变化”和“如果某因素改变会怎样”。
- 数据是观察性数据、面板数据、时间序列、行业/地区/企业分组数据。
- 需要处理因果图、上下游传导、滞后效应、冲击响应、政策评估、反事实预测。
- 需要区分相关、预测、Granger 时序领先、结构因果和政策可干预因果。

## 核心边界

- 预测准确不等于因果识别成功。
- Granger 因果只能作为时间领先证据，不自动等于结构因果。
- 任何因果结论必须写清识别假设、混杂变量、滞后结构、反事实定义和稳健性检查。
- 没有可信外生冲击、准实验、工具变量、明确 DAG 或强领域约束时，不要把相关关系写成“导致”。
- 机器学习模型可以辅助预测和异质性发现，但不能替代识别策略。

## 项目标准流程

1. 定义因果问题
   - 处理变量：哪个行业或因素的变化？
   - 结果变量：总用电量、行业用电量、增长率、结构占比还是波动风险？
   - 反事实：如果某行业不变化，未来用电量会怎样？
   - 时间尺度：日、周、月、季度、年。

2. 建立变量字典和 DAG
   - 区分处理、结果、混杂、中介、碰撞点、滞后变量、控制变量。
   - 行业上下游关系先作为“待检验结构假设”，不要直接当成事实。
   - 把政策、价格、气温、节假日、宏观经济、产业季节性作为潜在混杂或外生冲击。

3. 数据审计
   - 检查频率、缺失、口径变化、行政区划变化、行业分类变化。
   - 保留原始数据，只在 `Data/processed/` 或 `Experiments/` 中生成派生数据。
   - 对异常点做事件解释，不要静默删除。

4. 预测基线
   - 朴素季节模型、SARIMAX/VAR、机器学习、foundation model 预测作为基线。
   - 用滚动窗口回测，避免未来信息泄漏。
   - 输出 MAE、RMSE、MAPE/SMAPE、区间覆盖率和分行业误差。

5. 因果识别路线
   - 面板固定效应：行业/地区/时间固定效应，适合控制不变异质性和共同冲击。
   - DiD / event study：适合有明确政策、冲击或行业事件。
   - Synthetic control / CausalImpact：适合少数处理对象和明确干预时点。
   - VAR / Granger / impulse response：适合探索滞后传导，但只能写成时序领先证据。
   - DoWhy / EconML / DML：适合有明确处理变量、协变量和目标效应时。
   - Tigramite / PCMCI：适合多变量时间序列因果发现，但需要谨慎解释假设。
   - Bayesian hierarchical model：适合多行业、多地区、小样本和不确定性传播。

6. 稳健性和反证
   - 替换滞后阶数。
   - 替换行业分组口径。
   - 替换控制变量集。
   - placebo time / placebo industry。
   - 留出若干年份或地区做外推检查。
   - 对未观测混杂做敏感性分析。

7. 结论分级
   - Level 0：描述性相关。
   - Level 1：时间领先或预测贡献。
   - Level 2：控制混杂后的稳健关联。
   - Level 3：准实验或强识别假设下的因果效应。
   - Level 4：可支持政策/管理反事实模拟的因果估计。

## 推荐技能组合

- `data-analysis`：数据清洗和 EDA。
- `modeling-programming`：Python 建模和回测。
- `statistical-analysis`：统计检验和报告。
- `statsmodels`：SARIMAX、VAR、面板/回归基础。
- `aeon`：时间序列机器学习。
- `timesfm-forecasting`：零样本/批量时间序列预测基线。
- `pymc`：贝叶斯层级模型和不确定性。
- `scientific-critical-thinking`：偏倚、混杂和证据质量审查。
- `statistical-reviewer`：阶段性统计审查。
- `reproducibility-audit`：复现性审计。

## Agent 分工建议

- OpenCode：扫描文件、生成字段清单、检查缺失、定位脚本、跑小范围验证。
- Codex：定义识别策略、审查因果假设、整合报告、判断结论等级。
- 统计审查 agent：专门审查模型假设、效应量、置信区间、泄漏和混杂。
- 复现审计 agent：专门审查数据、代码、图表、论文结论是否可追溯。

## 输出要求

任何因果分析输出都必须包含：

- 研究问题。
- 处理变量、结果变量、控制变量。
- 识别假设。
- 数据来源和时间范围。
- 模型公式或算法。
- 估计结果和不确定性。
- 稳健性检查。
- 不支持的结论边界。
- 下一步验证计划。
