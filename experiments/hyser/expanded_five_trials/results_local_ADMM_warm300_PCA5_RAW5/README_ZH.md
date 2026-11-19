# Hyser local P/Q/Rkl ADMM vs 96-anchor global-R pilot

This code runs the existing repo src.cc_hiwa.soft_hiwa.SoftHiWA with full Q for each of 16 local group pairs, local orthogonal R_kl, group P and ADMM consensus global Rg.
HARD group assignment is one-hot, SOFT group assignment is independent temperature=.5 soft Kmeans; all P/Q/entropy/consensus parameters are otherwise identical.
**SUBSAMPLE**: only 250 training points/domain per fold, full Q on subsample, NOT the 1000-point formal fit.
Holdout is the entire remaining 250 windows, never used to train PCA/FA/ADMM.
Nearest target accuracy is based on true synchronized heldout Y, and train-only 5% inter-trial quantile radius. Coverage is not physical correctness.

|Feature|Method|n|Nearest coverage|physical-neighbor hit|random neighbor hit|exact time index|
|---|---|---:|---:|---:|---:|---:|
|PCA5_RAW5|hard|1|46.800%|2.400%|5.790%|0.000%|
|PCA5_RAW5|soft05|1|43.200%|2.000%|5.790%|0.400%|

Solver initialization: warm_start_local=True; ADMM mu=0.005; outer limit=300; per-pair inner limit=40.
**Do not interpret a local-ADMM run with nonconverged primal residual as a successfully converged full HiWA model.** The numerical history, P/Q block statistics and group marginal errors are saved for review.
No inherited source cluster labels have been given to target. No synchronized row pair enters OT fitting.
