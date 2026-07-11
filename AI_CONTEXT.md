# AI context: OT-HiWA

## Frozen primary objective

Establish whether the MiHiA implementation faithfully transfers TACO soft prototypes and GCOT into HiWA. Priority: (1) Soft-GCOT HiWA, (2) ROCA on that fixed baseline, (3) exploratory Joint-Prototype, CC-HiWA, PBMC, and GI-CC-HiWA. Do not delete prior work or present an extension as the baseline.

## Baseline contract

`SoftHiWA(..., support_mode="full")` is the reference.

- `soft_groups.py`: independently learned prototypes with reconstruction plus assignment-entropy regularization.
- Normalized assignment columns define the soft empirical measures; every $(i,j)$ has a weighted Sinkhorn $Q_{ij}$.
- $P$ uses the TACO Birkhoff (uniform-group) marginal. “Soft group marginals” means the $Q_{ij}$ marginals, not non-uniform $P$ marginals.
- HiWA provides $R_{ij}$, global orthogonal $R$, and ADMM. Hard and Soft both record global, primal, and dual residuals, transport objective, and group/Sinkhorn marginal error; comparable stopping requires global and primal residuals.
- `support_mode="sparse"` is a declared approximation; `full` is the comparison reference.

Pure baseline settings: representative guidance/rotation, component conditioning, and rotation anchor all zero; Joint-Prototype and ROCA off. ROCA may only choose `det(R)=-1/+1` without labels; labels are final evaluation only.

## Main paths and checks

```text
src/hiwa/hiwa.py                         hard HiWA reference
src/cc_hiwa/soft_groups.py               prototype learning
src/cc_hiwa/soft_hiwa.py                 full/sparse Soft-GCOT HiWA
src/cc_hiwa/cc_hiwa.py                   exploratory S = A P B^T extension
src/cc_hiwa/joint_prototypes.py          exploratory module
experiments/hiwa/run_taco_faithful_baseline.py  unified runner
```

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD\src\cc_hiwa;$PWD\experiments\hiwa;$PWD\experiments\roca"
python -m pytest tests -q
python experiments/hiwa/run_taco_faithful_baseline.py --seeds 0
```

Use only these visible names: `Hard HiWA`, `Soft-GCOT HiWA`, and `Soft-GCOT HiWA + ROCA`; call sparse support an approximation, not a separate method. Record accuracy, $R^2$, residuals, Sinkhorn errors, $P$, group costs, determinant, orthogonality, support mode, ROCA volume/condition/warning, and runtime. Never use labels for a branch or hyperparameter choice.
