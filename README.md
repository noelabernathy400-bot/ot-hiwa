# OT-HiWA — Optimal Transport for Multi-Modal Distribution Alignment

Hierarchical Wasserstein Alignment (**HiWA**, NeurIPS 2019) and extensions.

- **HiWA**: base algorithm — cluster-guided OT + Procrustes rotation
- **CC-HiWA**: component-conditioned HiWA with soft assignments
- **Soft-HiWA**: temperature-annealed prototype assignment
- **ROCA-HiWA**: rotation-oriented component-aware with determinant validation
- **GI-CC-HiWA**: graph-informed CC-HiWA for single-cell multi-omics (PBMC)

## Quick Start

```bash
pip install -r requirements.txt

# PowerShell: expose the migrated source and experiment modules.
$env:PYTHONPATH = "$PWD\src;$PWD\src\cc_hiwa;$PWD\experiments\hiwa;$PWD\experiments\roca"

# Unit tests
python -m pytest tests -q

# HiWA on synthetic data
python experiments/hiwa/run_synthetic.py --profile quick --seeds 0 1 2

# HiWA on neural data
python experiments/hiwa/run_neural.py --profile quick --seeds 0
```

## Structure

```
├── src/             ← Core algorithms
├── experiments/     ← Run & analyze scripts
├── tests/           ← Unit tests
├── docs/            ← Reading notes & reports
├── papers/          ← Reference PDFs
├── data/            ← Demo data
├── README.md        ← You are here
└── AI_CONTEXT.md    ← AI agent guide
```

## References

- Lee et al. (2019). *Hierarchical Optimal Transport for Multimodal Distribution Alignment.* NeurIPS. [arXiv:1906.11768](https://arxiv.org/abs/1906.11768)
- *TACO: Learning Geometric Knowledge from Text for Geometry-Free Adsorption Screening.* TPAMI.

## Status

| Method | Status |
|--------|--------|
| HiWA reproduction (synthetic + neural) | ✅ matches paper |
| Soft-Prototype HiWA | ✅ pilot done |
| ROCA-HiWA (synthetic validation) | ✅ |
| CC-HiWA PBMC smoke tests | ✅ |
| GI-CC-HiWA (graph-informed) | 🔧 in progress |

> **Note:** Report results across ≥5 random seeds. HiWA is non-convex.
