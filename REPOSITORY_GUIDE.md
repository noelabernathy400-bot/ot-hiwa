# OT-HiWA Research Repository Guide

## What this repository contains

This is the implementation and evidence repository for **GC-HiWA v2**, an
applicability-aware method for unsupervised coordinate alignment.  The method
is designed to recover a declared structured orthogonal map only when the
geometric assumptions are supported by diagnostics; otherwise it abstains.

The central contribution is therefore not a claim that optimal transport
always improves transfer.  It is a traceable workflow that pairs alignment
with qualification checks, held-out evaluation, and explicit negative
evidence.

## Start here

| Purpose | Entry point |
| --- | --- |
| First reading and project scope | `README.md` |
| Full project progression and claim boundaries | `docs/research/OT-HiWA项目进度总览_从零理解_2026-08-07.md` |
| Mathematical contract and Panoptic protocol | `docs/research/GC-HiWA_v2_数学合同、适用性诊断与Panoptic验证_2026-07-23.md` |
| Repository layout and reproducibility commands | `docs/GC-HiWA_v2_仓库说明与复现指南.md` |
| Final manuscript sources and PDFs | `Writing/GC-HiWA_v2_最终论文/` |
| Recorded numerical outputs | `experiments/results/` |

## Reproduce the checked software path

```powershell
pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src;$PWD\src\cc_hiwa;$PWD\experiments\hiwa;$PWD\experiments\roca"
python -m pytest tests -q
python experiments/synthetic/verify_gc_hiwa_v2_evidence.py
```

The synthetic qualification benchmark is the lightweight public reproduction
path.  Commands and expected interpretation for the Panoptic protocol are
recorded in the mathematical contract because the licensed source data are not
distributed here.

## Evidence boundary

Supported evidence includes synthetic known-truth recovery/abstention and a
controlled Panoptic coordinate-alignment protocol.  Historical neural and
external-data pilots remain in the repository as code and negative
applicability evidence; they are not independent support for the GC-HiWA v2
claim.  Please use the project overview and manuscript for the exact
qualification criteria before citing results.

## Data policy

Raw and license-restricted datasets are intentionally excluded.  Small demo
inputs, scripts, result JSON files, figures, protocols, and data-preparation
code are retained so a reader can see what was run and how each result is
traced.  See `DATA_AVAILABILITY.md` for details.
