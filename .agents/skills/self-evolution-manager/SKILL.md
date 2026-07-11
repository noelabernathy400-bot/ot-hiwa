---
name: self-evolution-manager
description: Use when deciding whether a useful conversation pattern, workflow, checklist, template, prompt, or lesson should be turned into a persistent workflow, skill, template, rule, or capability map entry.
---

# 自我进化管理

## 何时使用

- 聊天中出现可复用流程、提示词、模板、检查清单、项目经验或安全规则。
- 用户希望系统长期变聪明。
- 某次任务暴露出重复问题或可沉淀经验。

## 工作流

1. 判断内容是否可复用。
2. 建议固化形式：workflow、skill、template、rule、MOC、project note。
3. 说明将创建或修改哪些文件。
4. 涉及全局规则、skills、权限、安全边界时先等用户确认。
5. 写入对应位置。
6. 更新 `System/Capabilities/系统能力地图.md`。
7. 写入 `System/Reports/`。
8. 必要时提交 Git。

## 安全边界

- 不自动改全局规则。
- 不把一次性偏好写成长期规则。
- 不绕过 `AGENTS.md`。
- 不安装来源不明的 skill。
