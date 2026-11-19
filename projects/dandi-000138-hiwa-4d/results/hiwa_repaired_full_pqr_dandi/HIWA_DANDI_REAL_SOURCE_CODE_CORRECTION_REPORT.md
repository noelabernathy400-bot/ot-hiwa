# DANDI000138 4D: author-compatible group subspaces and train-only warm starts with FULL P/Q/R fit

**New controlled exploratory models, not an overwrite of any archived results.** All use original 17500/14000/3500/160 layers, K4, same 40/Hard-group training sample rows for all methods and each seed, original mu/epsilon/local/global iteration budgets, independent domains. All learn P,Q,R with original HiWA update equations; only optional R start and independently fitted local group projections differ.

Initial R selected BEFORE test on unpaired 400 source and 400 target training coordinates using covariance eigenspace 16-sign candidates and train Hungarian W2; within-domain projections fitted on 14000 training points separately for source and target without pairs.

| Method | Heldout cross-trial position hit10% | Velocity direction hit30° | Original full-domain test W2² | Projected P-Q heldout nested OT | Strict ADMM pass |
|---|---:|---:|---:|---:|---:|
| hard_warm_full | 0.03876 ± 0.01429 | 0.12132 ± 0.05255 | 0.36426 ± 0.02346 | 1.37352 ± 0.02798 | 3/3 |
| hard_warm_subspace2 | 0.03724 ± 0.01303 | 0.12348 ± 0.04837 | 0.35919 ± 0.02075 | 1.08015 ± 0.03376 | 3/3 |
| hard_warm_subspace3 | 0.03781 ± 0.01114 | 0.12132 ± 0.04790 | 0.36298 ± 0.02216 | 1.29517 ± 0.02545 | 3/3 |
| soft_warm_full | 0.06762 ± 0.01684 | 0.12492 ± 0.02425 | 0.36477 ± 0.02656 | 1.77104 ± 0.00941 | 0/3 |

**Original archived original Hard/Soft**: comparison values and their R/P remain accessible under strict_unsupervised_hard_soft; do not mix the projected nested OT geometry with raw nested OT when ranking methods because the metric differs.

Any source-target physical success claim is exploratory; original heldout 100 trials have been examined previously. This trial did not tune using true test outcomes. The source groups remain neural Hard Kmeans, the target groups remain target-only Hard Kmeans; the Soft version keeps full-support independent prototypes and does not secretly inherit Hard labels.

A correct implementation of the original optional projector path now exists in zip_gcot.preprocessing.py. Historical source code default behavior was regression tested bit-for-bit.

[All rotations, 4x4 group P and per-seed tests](HIWA_DANDI_REAL_SOURCE_CODE_CORRECTION_RESULTS.json) | [Exact full Q plans and original row IDs](hard_warm_full_seed42_COMPLETE_P_Q_R.npz) | [Full archived original](../strict_unsupervised_hard_soft/STRICT_UNSUPERVISED_REPORT.md) | [This controlled script](../../code/compare_corrected_original_hiwa_on_dandi.py)
