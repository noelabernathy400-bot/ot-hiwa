# Hyser local P/Q/Rkl ADMM vs 96-anchor global-R pilot

This code runs the existing repo src.cc_hiwa.soft_hiwa.SoftHiWA with full Q for each of 16 local group pairs, local orthogonal R_kl, group P and ADMM consensus global Rg.
HARD group assignment is one-hot, SOFT group assignment is independent temperature=.5 soft Kmeans; all P/Q/entropy/consensus parameters are otherwise identical.
**FULL TRAINING**: 1000 training points/domain per fold, independent domain permutations, 16 full local Q matrices each 1000x1000, no anchors, no test pairs in model fit.
Holdout is the entire remaining 250 windows, never used to train PCA/FA/ADMM.
Nearest target accuracy includes BOTH transductive 250-target heldout pool retrieval and INDUCTIVE 1000-point training-target bank retrieval; labels are used only AFTER nearest selection to judge physical correctness. Train-only 5% inter-trial quantile radius; geometry coverage is not physical correctness.

|Feature|Method|n|TEST-cloud coverage|TEST-cloud physical hit|TEST-cloud random hit|TRAIN-bank physical hit|TRAIN-bank random hit|W2²|physical RMSE|exact time index|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|FA5_RAW5|hard|1|64.000%|2.000%|5.790%|2.400%|4.410%|0.34932|0.12439|0.000%|
|FA5_RAW5|soft05|1|64.400%|2.400%|5.790%|1.200%|4.410%|0.38089|0.12527|0.800%|

Solver initialization: warm_start_local=True; ADMM mu=0.05; outer limit=300; per-pair inner limit=40; exact positive mass support=True; stable conditional log-Sinkhorn fallback=True.
**Do not interpret a local-ADMM run with nonconverged primal residual as a successfully converged full HiWA model.** The numerical history, P/Q block statistics and group marginal errors are saved for review.
No inherited source cluster labels have been given to target. No synchronized row pair enters OT fitting.
