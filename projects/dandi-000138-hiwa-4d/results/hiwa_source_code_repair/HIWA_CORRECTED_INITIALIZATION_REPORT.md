# HiWA original MATLAB parity and genuinely repaired optional code

**Verified concrete code issue:** the GCOT(projections=...) branch imported zip_gcot.preprocessing, which did not exist and would crash. The new module supplies independently computed source/target projections and original HiWA per-cluster subspace projectors; no physical labels are used for cross-domain fitting.

**Verified absence of core update bug:** an independently implemented original MATLAB one-iteration port matched P,Q,R at floating point precision. Additionally this test imported the EXACT old ZIP solver from the pre-edit Git commit and compared P,R, all local Q, history, dual for identical K4 synthetic input. Every tested value is BITWISE IDENTICAL in the default (no-arg) case.

Old source commit = fc217c23217c7c25c5e8322fe7b2e8f98ba29eb9, max original-vs-new default elementwise difference = 0.0.

## Experimental solver change (NOT secretly substituted for original method)

Added optional initial_R and local_start_from_global to hierarchical_alignment(). Both default to original behavior. The seed R is chosen only from the TWO SEPARATE TRAIN DOMAIN COVARIANCES by enumerating 16 sign choices and selecting by UNPAIRED TRAIN W2; true R* and physical labels are used POSTFIT ONLY.

| Geometry | Solver variant | True group P correspondence mass (mean) | Heldout true-group nearest-neighbor accuracy (mean) | Rotation Rgt Frobenius error (mean) | ADMM passed |
|---|---|---:|---:|---:|---:|
| well_separated | global_only | 0.8753 | 1.0000 | 0.0383 | 3/3 |
| well_separated | global_and_local | 0.8753 | 1.0000 | 0.0383 | 3/3 |
| overlapping | global_only | 0.8071 | 0.9907 | 0.1566 | 3/3 |
| overlapping | global_and_local | 0.8071 | 0.9898 | 0.1578 | 3/3 |

All numbers are genuine P/Q/R full solver runs on the same known-rotatable synthetic distribution, not just 1NN from the initial R. This does NOT establish Soft improvements or real DANDI physical validity.

## Carefully separate causal findings

- The default core numerical updates match original HiWA MATLAB; only the optional projection import was a definite executable code omission.
- If initialization improves synthetic recovery, it is an algorithmic/optimizer extension, not a repair of a transposition bug.
- Original paper's known-cluster d2/D6 data were closer to authors' geometric assumptions than our old K4 in D4 full-dim toy problem. See author parity audit.

[Raw results and exact audit assertions](HIWA_CORRECTED_INITIALIZATION_CONTROL.json) | [Optional correction implementation](../../code/zip_gcot/alignment.py) | [Independent projection code](../../code/zip_gcot/preprocessing.py) | [Regression/recovery test script](../../code/test_and_evaluate_corrected_hiwa.py)

