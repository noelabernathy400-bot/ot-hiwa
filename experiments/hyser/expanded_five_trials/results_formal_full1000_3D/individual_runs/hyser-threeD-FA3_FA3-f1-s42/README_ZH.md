# Hyser local P/Q/Rkl ADMM vs 96-anchor global-R pilot

This code runs the existing repo src.cc_hiwa.soft_hiwa.SoftHiWA with full Q for each of 16 local group pairs, local orthogonal R_kl, group P and ADMM consensus global Rg.
HARD group assignment is one-hot, SOFT group assignment is independent temperature=.5 soft Kmeans; all P/Q/entropy/consensus parameters are otherwise identical.
**FULL TRAINING**: 1000 training points/domain per fold, independent domain permutations, 16 full local Q matrices each 1000x1000, no anchors, no test pairs in model fit.
Holdout is the entire remaining 250 windows, never used to train PCA/FA/ADMM.
Nearest target accuracy includes BOTH transductive 250-target heldout pool retrieval and INDUCTIVE 1000-point training-target bank retrieval; labels are used only AFTER nearest selection to judge physical correctness. Train-only 5% inter-trial quantile radius; geometry coverage is not physical correctness.

|Feature|Method|n|TEST-cloud coverage|TEST-cloud physical hit|TEST-cloud random hit|TRAIN-bank physical hit|TRAIN-bank random hit|W2²|physical RMSE|exact time index|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|FA3_FA3|hard|1|72.000%|6.800%|4.082%|5.600%|3.589%|0.19950|0.10913|0.400%|
|FA3_FA3|soft05|1|64.800%|2.400%|4.082%|10.400%|3.589%|0.21315|0.12629|0.000%|

Solver initialization: warm_start_local=True; ADMM mu=0.05; outer limit=300; per-pair inner limit=40; exact positive mass support=True; stable conditional log-Sinkhorn fallback=True.
**Do not interpret a local-ADMM run with nonconverged primal residual as a successfully converged full HiWA model.** The numerical history, P/Q block statistics and group marginal errors are saved for review.
No inherited source cluster labels have been given to target. No synchronized row pair enters OT fitting.
