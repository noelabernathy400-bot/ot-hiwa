# Hyser full 1000-point per-domain Hard/Soft K=4 P/Q/Rkl ADMM formal benchmark

Study: Hyser S01 S1 Random Task, 5 actual trials x 250 time windows, 5-fold leave-one-trial-out.
X=256-channel EMG mapped to 5D PCA or FA (train-only); Y=all 5 actual finger forces whitened train-only, with physical channels retained.
Train 1000 source + 1000 target windows, separately shuffled, no synchronized row pair used to fit R/P/Q. Holdout intact 250 windows.
K4 independently in both domains; Hard one-hot vs Soft τ=0.5; local Q uses the EXACT positive-mass group support (no probability mass dropped). For Soft normally all 1000 per group; for Hard only actual member points; local Rkl, P, ADMM; NO anchors.
Original author default μ=.005 did NOT numerically converge on Hyser after 300 iterations. Training-subset solver checks predeclared μ=.05 and the SAME local warm-start for BOTH groups using numerical convergence only, not heldout test scores. K4, γ1=.2, γ2=.1/P, outer max300, inner max40, both domains independently grouped. Stable log-domain Q is used only when the identical entropic objective's legacy kernel scaling violates mass marginals.
Fisher 2D subspaces are presentation only, selected by TRAIN pseudolabels; no 2D coordinate plane or 2D transport metric is used.

Number of full runs found: 40/40. Eligible matched Hard/Soft seed-fold pairs: 0/20.
Status: PARTIAL_OR_UNCONVERGED

## All run observations (UNQUALIFIED rows are not valid model-performance claims)

| Representation | Method | Observed/expected | Numerically converged | Heldout W2² (all descriptive) | Inductive target-train-bank physical NN (all descriptive) | Transductive test-bank NN (all descriptive) |
|---|---|---:|---:|---:|---:|---:|
|PCA5_RAW5|hard|10/10|0|0.90553|0.13600|0.17760|
|PCA5_RAW5|soft05|10/10|10|0.96746|0.13920|0.17400|
|FA5_RAW5|hard|10/10|0|0.77235|0.11040|0.15160|
|FA5_RAW5|soft05|10/10|10|0.83204|0.10320|0.14880|

## Matched paired effects on BOTH-converged runs (Soft - Hard, negative is better for errors)
| Representation | eligible fold-seed pairs | Δ heldout W2² | Δ inductive physical NN correct fraction | Δ transductive physical NN correct fraction | Δ physical RMSE |
|---|---:|---:|---:|---:|---:|
|PCA5_RAW5|0|NA|NA|NA|NA|
|FA5_RAW5|0|NA|NA|NA|NA|

## Scientific inference restrictions
- If any paired runs do not meet both solver residuals, full-sample formal comparison is numerically INCONCLUSIVE. Never report cherry-picked converged subsets as if they represented five folds.
- Training target-bank retrieval uses only Y training reference cloud; transductive retrieval uses the heldout Y cloud but not its row-to-row pairing in training; NN correctness uses true synchronized Y only after training.
- The NN radius is fixed by the 5th percentile of unrelated target TRAIN inter-trial distances; coverage under this radius is not successful physical prediction.
- Full sample local Rkl/Qkl solver is a numerically stabilized, train-only feature-initialized version of the repository SoftHiWA, not a byte-for-byte rerun of the original NeurIPS PyHiWA implementation.
- Some data may not support common orthogonal alignment even if soft grouping helps distribution W2. A lack of convergence can be a model-mismatch signal, not just compute precision.
- Only compare Soft-vs-Hard WITHIN the same FA/PCA embedding. Comparing absolute cross-feature latent-space W2 is invalid.
- Five folds from one person/session, not five independent people: no claim of population significance.
- Individual time-window force regression, retrieval and distribution alignment remain distinct scientific tasks.
