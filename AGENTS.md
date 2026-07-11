# Research workspace guidance

This repository is a research workspace for notes, code, data, experiments, and reusable methods.

## Operating principle

Move research forward directly. A user request authorizes the normal edits, experiments, documents, and Git actions required to complete that request. Do not require an additional planning/confirmation round for ordinary scoped work.

Use judgment rather than rigid checklists. Read only the files relevant to the task; a missing handoff, report, or status file is not a blocker and must not be fabricated merely to satisfy process.

## Preserve evidence

- Do not knowingly fabricate sources, citations, experimental results, data provenance, or claims.
- Do not silently destroy original data, reference PDFs, or user-authored diary content.
- Preserve existing results unless the user explicitly asks to remove them.
- Treat pre-existing working-tree changes as user work: do not revert them.

## Research execution

- Keep code, parameter choices, seeds, outputs, and conclusions traceable when they matter to a result.
- Clearly separate confirmed findings, numerical diagnostics, exploratory results, and open questions.
- Use labels only where the experiment protocol permits them; do not hide selection leakage.
- Create reports or handoff notes when they improve continuity, not as mandatory ceremony.

## Collaboration

- Codex owns research design, mathematical judgment, integration, and final conclusions.
- Claude should handle low-level experiments, scans, and mechanical checks when available; if it is unavailable or stalls, state this and continue with a bounded local fallback.
- Skills and workflow notes are advisory aids. The user’s current request and this file take priority.

## File organization

- Projects: `Projects/`; reusable knowledge: `Domains/`; research material: `Research/`.
- Templates: `Templates/`; indexes: `MOCs/`; entities: `Entities/`; system notes: `System/`.
- Repository-specific skills may live in `.agents/skills/`, but no skill may block a user-authorized research task.
