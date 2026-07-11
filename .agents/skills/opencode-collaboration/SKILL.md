---
name: opencode-collaboration
description: Use inside this vault when Codex should collaborate with local OpenCode: simple/local/file-heavy work goes to OpenCode, while high-level research judgment, experiment design, writing, integration, and final decisions stay with Codex.
---

# Codex + OpenCode 协作调度

## 核心规则

- 当前 vault 中如同时存在用户级同名 `opencode-collaboration` skill，优先使用本地中文版规则；用户级 skill 只作为参考。
- 简单、本地、机械、文件很多的任务：优先交给 OpenCode。
- 高级判断、研究路线、实验设计、论文写作、材料整合、最终决策：由 Codex 负责。
- OpenCode 的输出视为“第一遍证据”，不是最终结论；Codex 需要复核、取舍和整合。

## 适合交给 OpenCode

- 扫描项目结构、找入口文件、找测试命令。
- 搜索关键词、符号、报错、TODO、配置文件。
- 总结某个局部模块、脚本、数据处理流程。
- 根据日志初步定位 bug。
- 审查局部 diff 的明显风险。
- 对明确范围做小步实现或整理。

默认要求 OpenCode 输出短结果：关键文件、发现、证据、建议下一步、验证命令。探索阶段必须要求“不要改代码”。

## Codex 保留职责

- 国家级大创的选题意义、创新点、研究路线、阶段计划。
- 数学/AI/材料方向的方法选择、实验设计、评价指标和风险判断。
- 论文、申报书、中期报告、结题报告、答辩材料。
- 对 OpenCode 结果做可信度判断、整合、修订和最终解释。
- 高风险操作前列计划并等用户确认。

## 调用模板

当前 vault 的 OpenCode 工作目录：

```powershell
wsl /home/a2826/.opencode/bin/opencode run --dir "/mnt/d/ai/数学ai" "<任务>"
```

项目扫描：

```text
使用 context-engineering 扫描当前 vault/项目。只输出：关键目录、相关项目文件、状态文件、可能的运行或验证命令、建议 Codex 下一步读取的内容。不要改代码。
```

定位问题：

```text
使用 debugging-and-error-recovery 根据下面的问题或日志定位根因。先不要改代码。只输出：最可能原因、相关文件、证据、最小修复建议。
```

小步实现：

```text
使用 incremental-implementation 只处理这个明确范围：<scope>。保持修改最小。最后输出改动文件、diff 摘要、验证命令。
```

局部审查：

```text
使用 code-review-and-quality 审查当前 diff。只列真实风险、回归点和测试缺口；没有问题就明确说没有发现。
```

## 工作循环

1. 先判断任务是否适合 OpenCode 低 token 探索。
2. 能交给 OpenCode 的，给它窄任务和短输出要求。
3. Codex 阅读结果后决定：继续探索、直接实现、写材料、更新报告，或询问用户。
4. 重要写入后遵守 `AGENTS.md`：写入 `System/Reports/` 报告，不覆盖用户已有改动。
