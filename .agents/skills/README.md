# Skills 说明

更新时间：2026-05-04

## 一句话结论

这里确实已经加了很多 skills，但它们不全是“全局”的。

- `.agents/skills/`：本 vault 内的项目级 skills，主要服务 `D:\ai\数学ai` 这个 Obsidian vault。
- `C:\Users\A2826\.codex\skills\`：用户级 Codex skills，通常对这个用户的 Codex 会话全局可用。
- `C:\Users\A2826\.codex\plugins\cache\...`：插件或系统自带 skills，随插件和当前会话环境出现，不属于本 vault 文件。

当前检查结果：

| 层级 | 位置 | 数量 | 是否全局 |
| --- | --- | ---: | --- |
| Vault 本地 skills | `.agents/skills/` | 22 | 不是严格全局，主要随本 vault 使用 |
| 用户级 skills | `C:\Users\A2826\.codex\skills\` | 41 | 是用户级全局 |
| 系统/插件 skills | Codex 系统目录、插件缓存目录 | 会随环境变化 | 不是本 vault 管理 |

## 本 vault 的本地 skills

这些 skills 是本 vault 的工作规则和研究流程能力。它们应该优先服从 `AGENTS.md` 的安全规则，不要把某个项目类型的规则污染到整个 vault。

| Skill | 用途 | 作用范围 |
| --- | --- | --- |
| `obsidian-knowledge-base` | 安全操作 Obsidian vault、Markdown、模板、MOC、双链、报告和目录结构 | 通用 |
| `research-project` | 管理通用研究项目，包括数学建模、大创、课程论文、科研训练等 | 通用 |
| `research-source-collection` | 搜集网页、PDF、数据集、官方文档、竞赛题目、代码仓库等资料 | 通用 |
| `literature-review` | 整理论文、PDF、网页、书籍，生成文献笔记和综述矩阵 | 通用 |
| `citation-verification` | 核验引用、DOI、BibTeX、网页链接和主张证据链 | 通用 |
| `method-note` | 记录研究方法、数学模型、统计方法、算法、适用条件和局限 | 通用 |
| `experiment-log` | 记录实验、数据处理、模型运行、参数、结果和复现步骤 | 通用 |
| `data-analysis` | 数据清洗、探索分析、统计分析、可视化和模型评价 | 通用 |
| `modeling-programming` | Python 建模代码、优化、仿真、机器学习、可视化和可复现程序 | 通用偏代码 |
| `reproducibility-audit` | 审计数据、代码、图表、结论是否可复现、可追溯 | 通用 |
| `figure-table-production` | 制作论文图、结果表、流程图、Mermaid 图和展示图 | 通用 |
| `academic-writing` | 论文、报告、摘要、引言、方法、结果、讨论、LaTeX 和展示稿 | 通用 |
| `latex-deliverable` | LaTeX 论文、公式、图表、BibTeX、附录和编译检查 | 通用 |
| `research-quality-gate` | 论文、报告、PPT、代码提交前的最终质量门 | 通用 |
| `handoff-and-status` | 维护状态、任务、决策、操作报告、复盘和 AI 交接说明 | 通用 |
| `self-evolution-manager` | 判断是否把可复用流程固化为 workflow、skill、template 或 rule | 通用但需谨慎 |
| `mathematical-modeling` | 数学建模专项：假设、变量、目标、约束、求解、验证、论文 | 只限数学建模项目 |
| `innovation-project` | 大创项目专项：选题、申报、中期、实验、结题和展示材料 | 只限大创项目 |
| `course-paper` | 课程论文专项：选题、综述、论点、提纲、引用、写作和格式 | 只限课程论文 |
| `opencode-collaboration` | Codex + OpenCode 协作调度，简单本地任务给 OpenCode，高级判断留给 Codex | 本 vault 协作 |
| `statistical-causal-inference` | 严肃统计因果推断：DAG、面板/时间序列、DiD、合成控制、DML、稳健性和反事实解释 | 通用偏严肃统计 |
| `statistical-reviewer` | 统计审查质量门：审查模型假设、因果语言、效应量、不确定性、泄漏和复现风险 | 通用偏审查 |

## 用户级全局 skills

这些位于 `C:\Users\A2826\.codex\skills\`，通常会被当前用户的 Codex 会话加载。它们是全局辅助能力，不应替代本 vault 的 `AGENTS.md` 和本地项目规则。

| Skill | 主要用途 |
| --- | --- |
| `academic-slides` | 学术汇报、答辩、论文转 slides |
| `citation-management` | 引用管理、文献元数据、BibTeX |
| `evo-memory` | 研究记忆和实验经验沉淀 |
| `experiment-craft` | 实验诊断、调试和迭代记录 |
| `experiment-iterative-coder` | 高质量实验代码迭代开发 |
| `experiment-pipeline` | 基线复现、调参、方法验证、消融实验流程 |
| `exploratory-data-analysis` | 科学数据探索性分析 |
| `jupyter-notebook` | 创建和编辑 Jupyter Notebook |
| `markdown-mermaid-writing` | Markdown 文档和 Mermaid 图 |
| `matplotlib` | Python 低层图表控制 |
| `networkx` | 网络和图算法 |
| `opencode-collaboration` | 用户级 OpenCode 协作规则 |
| `paper-lookup` | 多数据库论文检索 |
| `paper-navigator` | 找论文、读论文、查相关工作和趋势 |
| `paper-planning` | 论文写作前的故事线、实验和图表规划 |
| `paper-review` | 自己论文提交前的压力测试和审阅 |
| `paper-writing` | 学术论文分节写作 |
| `pdf` | PDF 读取、生成、渲染和版面检查 |
| `peer-review` | 正式审稿式检查 |
| `polars` | 高性能表格数据处理 |
| `pymoo` | 多目标优化 |
| `research-lookup` | 研究资料查询 |
| `research-survey` | 文献综述和领域调研报告 |
| `scientific-visualization` | 期刊级科研图表 |
| `scientific-writing` | IMRAD 科研写作 |
| `scikit-learn` | 机器学习建模 |
| `seaborn` | 统计可视化 |
| `shap` | 模型解释性分析 |
| `simpy` | 离散事件仿真 |
| `statistical-analysis` | 统计检验选择和报告 |
| `statsmodels` | 统计建模和推断 |
| `sympy` | 符号数学 |
| `transcribe` | 音频或视频转写 |
| `vibe-paper-writing` | 将用户材料整合进论文写作 |
| `xlsx` | Excel、CSV、TSV 表格文件处理 |
| `pymc` | 贝叶斯建模、层级模型、MCMC、后验诊断和不确定性 |
| `aeon` | 时间序列机器学习、预测、分类、聚类、异常检测 |
| `umap-learn` | 高维数据降维、可视化和聚类前处理 |
| `scientific-critical-thinking` | 科学证据质量、偏倚、混杂和实验设计有效性审查 |
| `dask` | 超内存或分布式 pandas / NumPy 工作流 |
| `timesfm-forecasting` | TimesFM 零样本时间序列预测和预测区间 |

## 系统和插件 skills

当前 Codex 环境还可能显示以下来源的 skills：

- Codex 系统 skills，例如 `skill-creator`、`skill-installer`、`openai-docs`、`imagegen`。
- 已启用插件贡献的 skills，例如 GitHub、Gmail、Google Calendar、Google Drive、Documents、Spreadsheets、Presentations、Browser Use 等。

这些 skills 不在本 vault 目录中，不建议在本目录维护它们的源文件。需要知道当前会话可用能力时，以 Codex 当前显示的 skills 列表为准。

## 使用优先级

1. 先遵守 `AGENTS.md`。
2. 处理本 vault 文件时，优先看 `.agents/skills/` 中的本地 skills。
3. 遇到专门格式或工具任务时，再调用用户级全局 skills，例如 `pdf`、`xlsx`、`jupyter-notebook`、`paper-lookup`。
4. 涉及 Google Drive、GitHub、日历、邮件、PPT、Word、浏览器测试等连接器或插件任务时，再使用插件 skills 和对应工具。
5. 新增、修改、迁移、删除 skills 前，先说明计划；涉及全局规则或高风险批量操作时，等待用户确认。

## Skill 路由与降噪

为了避免 skills 太多导致 agent 犹豫，按下面的路由优先选择：

| 任务 | 优先 skill | 辅助 skill / 工具 | 说明 |
| --- | --- | --- | --- |
| vault 笔记、模板、MOC、目录和报告 | `obsidian-knowledge-base` | `handoff-and-status` | 操作 vault 文件时先服从本地规则 |
| 项目创建、状态、任务、决策 | `research-project` | `handoff-and-status` | 具体项目再叠加专项 skill |
| 文献搜集 | `research-source-collection` | `paper-lookup`、`paper_search` MCP、`research-lookup` | 搜索结果先记录来源，不等于已核验 |
| 文献阅读和综述矩阵 | `literature-review` | `paper-navigator`、`research-survey` | Zotero 管引用源，Obsidian 管理解 |
| 引用、DOI、BibTeX 核验 | `citation-verification` | `refcheck` MCP、只读 `zotero` MCP、`citation-management` | 未核验必须标“待核验” |
| 方法、模型、算法节点 | `method-note` | `statistical-causal-inference`、`mathematical-modeling` | 方法节点写适用条件和局限 |
| 数据清洗、EDA、统计图 | `data-analysis` | `exploratory-data-analysis`、`polars`、`xlsx` | 原始数据只读，不静默删行列 |
| 建模代码和实验脚本 | `modeling-programming` | `jupyter-notebook`、`scikit-learn`、`statsmodels`、`pymoo` | 代码和输出要能追溯 |
| 实验记录 | `experiment-log` | `experiment-pipeline`、`papermill` | 记录参数、结果、随机种子和输出路径 |
| 复现审计 | `reproducibility-audit` | DVC、papermill、`research-quality-gate` | DVC 初始化前必须说明计划 |
| 论文草稿 | `academic-writing` | `paper-writing`、`scientific-writing`、`latex-deliverable` | 本地 skill 管边界，外部 skill 管写作技巧 |
| LaTeX / 可交付报告 | `latex-deliverable` | Quarto、`vibe-paper-writing` | 正式交付前做编译和引用检查 |
| 图表、流程图、展示图 | `figure-table-production` | `scientific-visualization`、`matplotlib`、`seaborn`、`markdown-mermaid-writing` | 图表必须服务证据链 |
| 统计因果项目 | `statistical-causal-inference` | `statistical-reviewer`、`statsmodels`、`pymc` | 预测准确不等于因果识别 |
| 终稿硬伤检查 | `research-quality-gate` | `paper-review`、`peer-review`、`statistical-reviewer` | 先列硬伤，再给摘要 |
| 多 agent / OpenCode 协作 | vault 本地 `opencode-collaboration` | 用户级同名 skill 只作参考 | 当前 vault 优先使用本地中文版规则 |
| 固化新经验 | `self-evolution-manager` | `System/Workflows/受控自我进化流程.md` | 修改全局规则或 skill 前先确认 |

降噪原则：

- 同一任务优先选 1 个主 skill，最多再选 1-2 个辅助 skill。
- 本 vault 文件和研究规则优先使用本地 skills；外部 skills 只补专门能力。
- 出现同名 skill 时，本 vault 内优先使用 `.agents/skills/` 版本。
- 不因外部 skill 写了“always”就绕过 `AGENTS.md`、原始资料保护、引用核验和用户确认规则。
- 用户没有明确要求高强度实验或长流程时，不主动触发大型 experiment/evo-memory 类工作流。

## 重复和重叠处理

当前已知重叠：

| 重叠项 | 处理 |
| --- | --- |
| 本地 `opencode-collaboration` 与用户级 `opencode-collaboration` 同名 | 在本 vault 中优先使用本地版本；用户级版本作为外部参考 |
| `literature-review`、`paper-navigator`、`research-survey` | 本地 skill 管文献进入 vault 和笔记结构；外部 skills 用于找论文或生成综述草稿 |
| `citation-verification`、`citation-management`、`refcheck` MCP | 本地 skill 管核验规则和安全边界；`refcheck` 做真实数据库核验；citation-management 可辅助格式和 BibTeX |
| `academic-writing`、`paper-writing`、`scientific-writing` | 本地 skill 管项目写作边界；外部 skills 辅助具体段落、IMRAD 或论文技巧 |
| `data-analysis`、`exploratory-data-analysis`、`polars`、`dask` | 本地 skill 管 vault 数据规则；外部 skills 按数据规模和工具选择调用 |

## 新增 skills 放哪里

| 需求 | 建议位置 |
| --- | --- |
| 只服务这个 vault 的长期工作流 | `.agents/skills/<skill-name>/SKILL.md` |
| 希望所有 Codex 项目都能用 | `C:\Users\A2826\.codex\skills\<skill-name>\SKILL.md` |
| 只是项目经验或复盘 | 对应项目 `Notes/`、`Decision Log.md` 或 `Handoff Note.md` |
| 可复用研究方法 | `Research/Methods/` 或 `Domains/` |
| 可复制笔记结构 | `Templates/` |
| 系统流程 | `System/Workflows/` |

## 维护注意

- 不要把数学建模专项规则写成全局规则。
- 不要让外部全局 skills 绕过本 vault 的安全边界。
- 不要覆盖已有 `SKILL.md`；修改前先检查差异。
- 安装新的用户级全局 skills 后，通常需要重启或刷新 Codex 才能在新会话中自动识别。
- 每次重要修改 skills 后，都应在 `System/Reports/` 写操作报告。
