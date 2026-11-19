# Hyser full 1000-point per-domain Hard/Soft K=4 P/Q/Rkl ADMM formal benchmark

Study: Hyser S01 S1 Random Task, 5 actual trials x 250 time windows, 5-fold leave-one-trial-out.
X=256-channel EMG mapped to 5D PCA or FA (train-only); Y=all 5 actual finger forces whitened train-only, with physical channels retained.
Train 1000 source + 1000 target windows, separately shuffled, no synchronized row pair used to fit R/P/Q. Holdout intact 250 windows.
K4 both domains independently; Hard one-hot vs Soft τ=0.5; dense 1000x1000 local Q for all 16 block pairs, local Rkl, P, ADMM; no anchors.
Original author default μ=.005 did NOT numerically converge on Hyser after 300 iterations. The predeclared numeric repair here uses *the same μ=.05 and local warm-start for BOTH methods*, selected using inner training-only ADMM primal/rotation convergence, not test success. The actual method remains the same local PQR/entropic constraints and γ1=.2, γ2=.1/P, max outer=300, inner=40, group Sinkhorn same mathematical objective computed stably.
Fisher 2D subspaces are presentation only, selected by TRAIN pseudolabels; no 2D coordinate plane or 2D transport metric is used.

Number of full runs found: 40/40. Eligible matched Hard/Soft seed-fold pairs: 13/20.
Status: PARTIAL_OR_UNCONVERGED

## All run observations (UNQUALIFIED rows are not valid model-performance claims)

| Representation | Method | Observed/expected | Numerically converged | Heldout W2² (all descriptive) | Inductive target-train-bank physical NN (all descriptive) | Transductive test-bank NN (all descriptive) |
|---|---|---:|---:|---:|---:|---:|
|PCA5_RAW5|hard|10/10|6|0.90741|0.13640|0.18320|
|PCA5_RAW5|soft05|10/10|7|0.96802|0.13920|0.17400|
|FA5_RAW5|hard|10/10|9|0.77923|0.12040|0.16280|
|FA5_RAW5|soft05|10/10|8|0.83196|0.10360|0.14880|

## Matched paired effects on BOTH-converged runs (Soft - Hard, negative is better for errors)
| Representation | eligible fold-seed pairs | Δ heldout W2² | Δ inductive physical NN correct fraction | Δ transductive physical NN correct fraction | Δ physical RMSE |
|---|---:|---:|---:|---:|---:|
|PCA5_RAW5|5|+0.007731|-0.008000|-0.006400|-0.000127|
|FA5_RAW5|8|+0.061140|-0.022000|-0.021500|-0.002556|

## Scientific inference restrictions
- If any paired runs do not meet both solver residuals, full-sample formal comparison is numerically INCONCLUSIVE. Never report cherry-picked converged subsets as if they represented five folds.
- Training target-bank retrieval uses only Y training reference cloud; transductive retrieval uses the heldout Y cloud but not its row-to-row pairing in training; NN correctness uses true synchronized Y only after training.
- The NN radius is fixed by the 5th percentile of unrelated target TRAIN inter-trial distances; coverage under this radius is not successful physical prediction.
- Full sample local Rkl/Qkl solver is a numerically stabilized, train-only feature-initialized version of the repository SoftHiWA, not a byte-for-byte rerun of the original NeurIPS PyHiWA implementation.
- Some data may not support common orthogonal alignment even if soft grouping helps distribution W2. A lack of convergence can be a model-mismatch signal, not just compute precision.
- Only compare Soft-vs-Hard WITHIN the same FA/PCA embedding. Comparing absolute cross-feature latent-space W2 is invalid.
- Five folds from one person/session, not five independent people: no claim of population significance.
- Individual time-window force regression, retrieval and distribution alignment remain distinct scientific tasks.
