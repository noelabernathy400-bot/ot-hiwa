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
python experiments/hiwa/run_taco_faithful_baseline.py --seeds 50 51 52 53 54 --tag audit
```

The runner compares Hard HiWA, Soft-GCOT HiWA, its declared sparse approximation, and Soft-GCOT HiWA + ROCA. The default `audit` profile uses a fixed numerical budget and a common global-plus-primal ADMM stopping rule. It writes JSON plus accuracy and convergence figures. The deterministic 96-by-96 neural subset is a bounded audit subset; use `--max-samples 0` for the full data at substantially greater cost.

| Method | Contract |
|---|---|
| Hard HiWA | Original HiWA using hard labels from unlabeled learned prototypes. |
| Soft-GCOT HiWA | Weighted full-support soft groups, $P$, all $Q_{ij}$, and HiWA ADMM. Representative guidance, anchors, component conditioning, Joint-Prototype, and ROCA are off. |
| Sparse approximation | Computational top-membership approximation to Soft-GCOT HiWA; never a silent replacement. |
| Soft-GCOT HiWA + ROCA | Soft-GCOT HiWA over both determinant components; unlabeled representative orientation selects the branch and emits a warning. |

Key paths: `src/hiwa/` (reference HiWA), `src/cc_hiwa/` (soft and exploratory extensions), `experiments/hiwa/` (runners), `docs/taco/` and `docs/hiwa/` (paper notes). See `docs/research/TACO-faithful_Soft-GCOT_HiWA_audit_2026-07-11.md` and `experiments/results/taco_faithful_baseline_20260711_minimal.md`.
