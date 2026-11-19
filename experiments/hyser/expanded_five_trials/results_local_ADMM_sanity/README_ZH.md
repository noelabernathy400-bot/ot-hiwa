# Hyser local P/Q/Rkl ADMM vs 96-anchor global-R pilot

This code runs the existing repo src.cc_hiwa.soft_hiwa.SoftHiWA with full Q for each of 16 local group pairs, local orthogonal R_kl, group P and ADMM consensus global Rg.
HARD group assignment is one-hot, SOFT group assignment is independent temperature=.5 soft Kmeans; all P/Q/entropy/consensus parameters are otherwise identical.
**Scope limit**: each fold's complete 1000 training points build PCA5/FA5 representations and 4 group K-means is fitted on **250 real stratified training points per domain**, selected independently from each of four training trials (63+63+63+61). The fit fully solves all local Q blocks on those 250 points, not on 1000 original training points; no 96-anchor approximation is used. This is a scientifically honest small-data full-PQR-ADMM CONTROL, NOT claimed as full 1000-point original HiWA run.
Holdout is the entire remaining 250 windows, never used to train PCA/FA/ADMM.
Nearest target accuracy is based on true synchronized heldout Y, and train-only 5% inter-trial quantile radius. Coverage is not physical correctness.

|Feature|Method|n|Nearest coverage|physical-neighbor hit|random neighbor hit|exact time index|
|---|---|---:|---:|---:|---:|---:|
|PCA5_RAW5|hard|1|42.400%|2.400%|5.790%|0.400%|
|PCA5_RAW5|soft05|1|43.200%|2.400%|5.790%|0.400%|
|FA5_RAW5|hard|1|52.400%|3.600%|5.790%|0.800%|
|FA5_RAW5|soft05|1|50.000%|4.400%|5.790%|0.400%|

**Do not interpret a local-ADMM run with nonconverged primal residual as a successfully converged full HiWA model.** The numerical history, P/Q block statistics and group marginal errors are saved for review.
No inherited source cluster labels have been given to target. No synchronized row pair enters OT fitting.
