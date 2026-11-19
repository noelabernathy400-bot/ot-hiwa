# Hyser local P/Q/Rkl ADMM vs 96-anchor global-R pilot

This code runs the existing repo src.cc_hiwa.soft_hiwa.SoftHiWA with full Q for each of 16 local group pairs, local orthogonal R_kl, group P and ADMM consensus global Rg.
HARD group assignment is one-hot, SOFT group assignment is independent temperature=.5 soft Kmeans; all P/Q/entropy/consensus parameters are otherwise identical.
**FULL TRAINING**: 1000 training points/domain per fold, independent domain permutations, 16 full local Q matrices each 1000x1000, no anchors, no test pairs in model fit.
Holdout is the entire remaining 250 windows, never used to train PCA/FA/ADMM.
Nearest target accuracy includes BOTH transductive 250-target heldout pool retrieval and INDUCTIVE 1000-point training-target bank retrieval; labels are used only AFTER nearest selection to judge physical correctness. Train-only 5% inter-trial quantile radius; geometry coverage is not physical correctness.

|Feature|Method|n|TEST-cloud coverage|TEST-cloud physical hit|TEST-cloud random hit|TRAIN-bank physical hit|TRAIN-bank random hit|W2²|physical RMSE|exact time index|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|PCA5_RAW5|hard|1|39.200%|13.600%|6.280%|8.400%|3.651%|1.90697|0.23545|0.000%|
|PCA5_RAW5|soft05|1|37.600%|2.000%|6.280%|1.200%|3.651%|2.43073|0.21881|0.000%|

Solver initialization: warm_start_local=True; ADMM mu=0.05; outer limit=300; per-pair inner limit=40.
**Do not interpret a local-ADMM run with nonconverged primal residual as a successfully converged full HiWA model.** The numerical history, P/Q block statistics and group marginal errors are saved for review.
No inherited source cluster labels have been given to target. No synchronized row pair enters OT fitting.
