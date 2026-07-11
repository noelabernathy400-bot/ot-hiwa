---
name: citation-verification
description: Use when verifying citations, references, bibliographic metadata, DOI, URLs, quoted claims, and whether paper claims are traceable to reliable sources.
---

# 引用核验

## 何时使用

- 写论文、综述、报告、数学建模参考文献时。
- 检查文献笔记、引用格式、DOI、网页链接、BibTeX 和直接引用。

## 工作流

1. 列出所有引用和关键事实主张。
2. 对每条引用核验标题、作者、年份、来源、DOI/URL。
3. 对关键事实追溯原文位置或数据来源。
4. 标注无法核验、信息不完整或疑似错误的引用。
5. 输出可用引用、待补充引用、应删除引用。
6. 需要时生成 BibTeX 或参考文献列表。

## 工具调用规则

- 遇到 DOI、BibTeX、参考文献真假、题名/作者/年份/期刊核验时，优先使用 `refcheck` MCP。
- 遇到“我的文献库”“Zotero”“已收藏论文”“文献集合”“注释”“笔记”“citation key”时，优先使用只读 `zotero` MCP。
- 使用 `zotero` MCP 时，默认只读：可以检索条目、集合、标签、注释、笔记、全文、PDF 目录和重复项；不得默认创建、删除、更新、添加、合并或批量改标签。
- 写入 Zotero 时简要说明目标和影响范围；用户当前请求已明确授权时可直接执行。
- `refcheck` 和 `zotero` 给出的 BibTeX 仍需在正式写入论文前检查字段、大小写、页码和特殊字符。

## 检查项

- 是否真实存在。
- 是否引用了正确版本。
- 是否支持正文中的主张。
- 是否需要页码或章节位置。
- 格式是否统一。

## 安全边界

- 不编造文献。
- 不为不存在的文献生成 DOI。
- 无法核验时明确标注“待核验”。
