# AI_CONTEXT — OT-HiWA Research Project

> For Claude Code, ChatGPT, Codex. Read this first.

## Project

Optimal Transport research: **HiWA** (hierarchical Wasserstein alignment) → **CC-HiWA** → **ROCA-HiWA** → **GI-CC-HiWA**, inspired by **TACO**.

- Python 3.13 | numpy 2.4 | scipy 1.17 | scikit-learn 1.9
- Notes in Chinese, code/math in English

## Code Map

```
src/hiwa/               ← Base HiWA (stable reference, 2 files)
src/cc_hiwa/            ← Our extensions (active dev, 5 files)

experiments/hiwa/       ← HiWA reproduction + Soft-HiWA (7 scripts)
experiments/roca/       ← ROCA-HiWA experiments (16 scripts)
experiments/pbmc/       ← PBMC CC-HiWA (3 scripts)
tests/                  ← Unit tests (4 files)

docs/hiwa/              ← HiWA paper 13-chapter reading notes
docs/taco/              ← TACO paper reading notes
docs/research/          ← CC-HiWA, ROCA, Soft research reports
docs/pbmc/              ← PBMC experiment documentation
docs/methodology/       ← Reading methods, literature survey
papers/                 ← Reference PDFs
data/                   ← Demo data
```

## Key Docs

| File | Content |
|------|---------|
| `docs/hiwa/第一章 论文主要公式解释.md` | HiWA math foundations |
| `docs/research/从HiWA到ROCA-HiWA学习总览-*.md` | Complete learning path |
| `docs/research/CC-HiWA研究骨干与TACO迁移总纲-*.md` | Research roadmap |
| `docs/research/ROCA-HiWA阶段性研究报告-*.md` | ROCA phase report |

## Rules

1. **Never delete/overwrite** data, reference PDFs, or existing results
2. **Do not modify `src/hiwa.py`** without explicit instruction — stable reference
3. **Always ≥5 random seeds** for results; single-seed = invalid
4. **Run tests** before and after modifying `src/`: `python -m pytest tests/`
5. **Log all runs** including failures; don't cherry-pick
6. **Don't fabricate** paper details — mark unknowns `TBD`

## Current Priority

1. Improve CC-HiWA on PBMC — add graph/structure information (Procrustes baseline still stronger)
2. TACO-style component-conditioned sample OT for neural data
3. GI-CC-HiWA method iteration

## Environment

```bash
pip install -r requirements.txt
```
