# Author source comparison and first fully reproducible correction audit

**Finding:** We independently transcribed the MATLAB update rules and ran one identical random-number-stream outer iteration against the current Python solver. No incorrect matrix multiplication or q/P rotation temperature was found in the audited step.

## One-outer-step literal MATLAB parity

| Array | Maximum elementwise disagreement |
|---|---:|
| R | 5.551e-16 |
| P | 2.776e-17 |
| local_R | 5.551e-16 |
| cost | 6.939e-17 |
| dual | 1.11e-15 |

## ORIGINAL-LIKE synthetic experiment: low-dimensional local subspaces

The author's Fig1cd used **five clusters, intrinsic dimension2 and ambient dimension6**. Our earlier 4-cluster full-dimensional 4D synthetic data was structurally very different. In this experiment we generate five 2D low-rank clusters in 6D, apply unknown true R*, independently draw target and source, and shuffle target cluster IDs (true correspondence only evaluated postfit). We disclose the **known within-domain group partitions** like original semi-supervised HiWA; P remains freely solved. Uses 35 samples/cluster/domain. Ambient noise 0.08 means learned rank2 subspace projection is an actual change.

| Instance/solver seed | Parameters | Local projection | Correct P group mass↑ | True Rgt alignment rMSE↓ | Stopped after | Strict ADMM |
|---|---|---|---:|---:|---:|---|
| 91 | original DANDI | none | 1.0000 | 0.0044 | 500 | False |
| 91 | original DANDI | local PCA2 projection | 1.0000 | 0.0044 | 500 | False |
| 91 | author Fig1cd | none | 1.0000 | 0.0069 | 43 | False |
| 91 | author Fig1cd | local PCA2 projection | 1.0000 | 0.0052 | 51 | False |
| 92 | original DANDI | none | 1.0000 | 0.0004 | 500 | False |
| 92 | original DANDI | local PCA2 projection | 1.0000 | 0.0004 | 500 | False |
| 92 | author Fig1cd | none | 1.0000 | 0.0009 | 17 | False |
| 92 | author Fig1cd | local PCA2 projection | 1.0000 | 0.0007 | 25 | False |
| 93 | original DANDI | none | 0.9983 | 0.0078 | 474 | False |
| 93 | original DANDI | local PCA2 projection | 0.9983 | 0.0078 | 472 | False |
| 93 | author Fig1cd | none | 0.9984 | 0.0100 | 30 | False |
| 93 | author Fig1cd | local PCA2 projection | 0.9984 | 0.0071 | 36 | False |

## Identified differences from MATLAB

1. **Local projection:** Author HiWA.m lines81-93 computes per-group A_i A_i^T X_i and B_j B_j^T Y_j before local OT. Current DANDI strict4D solver uses identity/full 4D (closer to HiWASSC.m raw-data path); this is a method difference, not a transpose bug. Explicit independent rank2 PCA-per-group is now tested on a low-rank reference.
2. **Synthetic identifiability:** prior DANDI-positive control used K=4 clusters in ambient D=4, each with full-dimensional noise. Authors' Fig1cd varies intrinsic d2 and D6 with S5 group partitions. Four 4D centroids do not generically identify every O(4) degree of freedom.
3. **Parameter budgets:** Original Fig1cd max outer300, tol0.1, mu0.005, local100, sample Sinkhorn150, group Sinkhorn1000; our DANDI4D max outer500, tol1e-9, mu0.05, local20, sample45, group200. All differences are independently labeled, and epsilon_group0.2/sample0.1 remain equal.
4. **Stopping**: original only tests global rotation change; stringent final ADMM tolerances used in our archive are additional diagnostics, not a proof of a globally correct solution.
5. **Preprocessing**: the author's Python neural demo whitened each domain and used Isomap-based low-d projections, not PCA4+full 4D unprojected data.

## Next stage: fix/verify without rewriting old evidence

- Do not overwrite archived original Hard/Soft results.
- Add compatible optional initial_R and per-cluster projection interfaces to NEW experimental variant, preserving legacy defaults and regression tests.
- On synthetic known-group/subspace benchmarks, increase independent generated instances and optimizer seeds. Only add proposed initialization changes after a train-only objective, not true Rgt, selects candidates.
- For DANDI choose PCA/FA/subspace variants using train-internal validation. Keep P/Q/R, actual full hierarchical transport, physical position and velocity retrieval and convergence separate.

[Original MATLAB HiWA](https://github.com/siplab-gt/hiwa-matlab/blob/master/code/toolbox/HiWA.m) | [Original MATLAB Figure1cd configuration](https://github.com/siplab-gt/hiwa-matlab/blob/master/code/figure1cd.m) | [Repo original numerics](../../code/zip_gcot/alignment.py) | [Audit source](../../code/audit_author_hiwa_parity_and_subspaces.py)

