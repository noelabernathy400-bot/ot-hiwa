# OT-HiWA

The current primary line is **TACO-faithful Soft-GCOT HiWA** for MiHiA macaque neural--movement alignment: learned soft prototypes, weighted group empirical measures, group transport $P$, one weighted sample transport $Q_{ij}$ per group pair, and the original HiWA local/global orthogonal ADMM consensus.

## Priority

1. Reproduce and audit TACO-faithful Soft-GCOT HiWA.
2. Test ROCA only on that unchanged baseline.
3. Retain Joint-Prototype, CC-HiWA, PBMC, and GI-CC-HiWA as exploratory extensions.

## Quick start

```powershell
pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src;$PWD\src\cc_hiwa;$PWD\experiments\hiwa;$PWD\experiments\roca"
python -m pytest tests -q
python experiments/hiwa/run_taco_faithful_baseline.py --seeds 0
```

The runner compares Hard HiWA, full-support Soft-GCOT, sparse-support Soft-GCOT, and full-support Soft-GCOT + ROCA. The default deterministic 96-by-96 neural subset is a bounded smoke experiment; use `--max-samples 0` for the full data at substantially greater cost.

| Method | Contract |
|---|---|
| Hard HiWA | Original HiWA using hard labels from unlabeled learned prototypes. |
| Soft-GCOT (full) | Weighted full-support soft groups, $P$, all $Q_{ij}$, and HiWA ADMM. Representative guidance, anchors, component conditioning, Joint-Prototype, and ROCA are off. |
| Soft-GCOT (sparse) | Computational top-membership approximation to full support; never a silent replacement. |
| Soft-GCOT + ROCA | Full baseline over both determinant components; unlabeled representative orientation selects the branch and emits a warning. |

Key paths: `src/hiwa/` (reference HiWA), `src/cc_hiwa/` (soft and exploratory extensions), `experiments/hiwa/` (runners), `docs/taco/` and `docs/hiwa/` (paper notes). See `docs/research/TACO-faithful_Soft-GCOT_HiWA_audit_2026-07-11.md` and `experiments/results/taco_faithful_baseline_20260711_minimal.md`.
