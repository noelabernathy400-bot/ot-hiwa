# OT-HiWA

## Paper

- [GC-HiWA English PDF](papers/GC-HiWA_EN.pdf)
- [GC-HiWA 中文 PDF](papers/GC-HiWA_CN.pdf)

The manuscript PDFs above are the project-specific copies previously published in the separate `papers` repository. They are included here so the code, evidence record, and paper can be reviewed from one repository.

## Current research line

The active, evidence-backed line is **GC-HiWA v2: applicability-aware unsupervised coordinate alignment**. It attempts to recover a declared structured orthogonal coordinate map only when two unpaired domains share geometric group structure; otherwise it abstains with explicit numerical and geometric reasons. Its primary application is uncalibrated cross-device / cross-view 3-D pose coordinates, using the structured family \(I_J\otimes SO(3)\), rather than an unrestricted rotation in \(O(3J)\).

What is currently supported is deliberately narrow:

- known-truth synthetic recovery and explicit abstention boundaries;
- controlled Panoptic multi-view pose recovery with withheld camera calibration and shuffled target frames;
- a label-free ROCA branch-selection rule that abstains when its determinant branches are not sufficiently qualified.

This repository does **not** currently support the claim that GC-HiWA universally improves cross-domain transfer or real cross-device semantic action recognition. The pending semantic-action validation is documented and frozen in `docs/research/真实语义动作验证的数据接入与冻结协议_2026-07-25.md`.

The canonical mathematical contract and evidence record are:

- `docs/research/OT-HiWA项目进度总览_从零理解_2026-08-07.md`（首次阅读建议从这里开始）
- `docs/research/GC-HiWA_v2_数学合同、适用性诊断与Panoptic验证_2026-07-23.md`
- `Writing/GC-HiWA_v2_论文草稿.tex`（公开论文 PDF：[`papers/GC-HiWA_CN.pdf`](papers/GC-HiWA_CN.pdf)）
- `docs/research/GC-HiWA_v2_完整研究笔记_Obsidian.md`
- `Writing/终稿审查清单.md`
- `docs/GC-HiWA_v2_仓库说明与复现指南.md`（仓库结构、运行入口与证据边界）

## Reproducing the current verified checks

```powershell
pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src;$PWD\src\cc_hiwa;$PWD\experiments\hiwa;$PWD\experiments\roca"
python -m pytest tests -q
python experiments/synthetic/verify_gc_hiwa_v2_evidence.py
python experiments/synthetic/run_gc_hiwa_boundary_benchmark.py --seeds 2201 2202 2203 2204 2205 --scenarios isospectral_nonorthogonal --methods soft_transport_oracle_component soft_roca --tag local_isospectral_audit
```

The synthetic runner uses no labels, true rotation, pairing, or deformation during fitting; these quantities are evaluation-only. In the isospectral nonorthogonal condition, the source and target covariance spectra are deliberately identical, yet the method must abstain through the combined fit, convergence, and ROCA checks. The Panoptic protocol and its exact commands are recorded in the canonical mathematical contract above because the underlying data are not bundled.

## Historical TACO / MiHiA line

The remaining runner description and method table describe an earlier neural--movement development line. They remain useful as historical code and negative applicability evidence, but they are not the current primary claim and must not be cited as independent validation of GC-HiWA v2.

Successful mainline runs automatically commit and push only their generated JSON and figures. Use `--no-sync-results` only for a local dry run; automatic sync refuses to proceed if unrelated changes are already staged.

| Method | Contract |
|---|---|
| Hard HiWA | Original HiWA using hard labels from unlabeled learned prototypes. |
| Soft-GCOT HiWA | Weighted full-support soft groups, $P$, all $Q_{ij}$, and HiWA ADMM. Representative guidance, anchors, component conditioning, Joint-Prototype, and ROCA are off. |
| Sparse approximation | Computational top-membership approximation to Soft-GCOT HiWA; never a silent replacement. |
| Soft-GCOT HiWA + ROCA | Soft-GCOT HiWA over both determinant components; unlabeled representative orientation selects the branch and emits a warning. |

Key paths: `src/hiwa/` (reference HiWA), `src/cc_hiwa/` (soft and exploratory extensions), `experiments/hiwa/` (runners), `docs/taco/` and `docs/hiwa/` (paper notes). See `docs/research/TACO-faithful_Soft-GCOT_HiWA_audit_2026-07-11.md` and `experiments/results/taco_faithful_baseline_20260711_minimal.md`.
