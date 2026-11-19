# Hyser PCA / FA factor representation on K4 Hard vs Soft-GCOT

**This file contains a specified subset of the predeclared six representations, evaluated on the indicated heldout trials and fixed seeds.**
Both source and target factor methods are fitted independently on the four train trials.
Two-level P/Q OT + global orthogonal R with 96 weighted anchors, not original full local R_kl HiWA ADMM.
FA and PCA output coordinates are train-whitened before training both hard and soft.
No synchronized pairings/force labels in R,P,Q fitting, at any stage.

For 3D the four combos implement X PCA/FA and Y PCA/FA from ALL FIVE original fingers.
For 5D X is PCA5 or FA5; the target is FULL FIVE FORCE CHANNELS with train whitening, not reduced.
This is not comparable to the historical Y3 experiment that discarded ring/little.

| Setup | Hard W2² | Soft W2² | Hard shared T | Soft shared T | Hard physical RMSE | Soft physical RMSE |
|---|---:|---:|---:|---:|---:|---:|
| PCA5_RAW5 | 1.09919 | 1.08608 | 1.3519 | 1.27297 | 0.1616 | 0.15439 |
| FA5_RAW5 | 0.86018 | 0.83183 | 1.35448 | 1.25917 | 0.15464 | 0.14951 |

## Correct inference

- Absolute W2 or T between different FA/PCA embeddings is NOT a common metric because the learned geometries differ.
- Compare Soft−Hard differences WITHIN each representation; the physical five-finger RMSE is also in common physical units.
- Heldout shared T is evaluated after re-solving P/Q on a SAME Hard Kmeans cluster reference within each representation; it is NOT the cost of freezing the training P/Q.
- The physical test RMSE uses paired timestamps ONLY AFTER unpaired R has been learned and target-only force decoder has been fitted on train data.
- A target-only decoder is not a supervised EMG-to-force model: it does not learn paired Xtrain/Ytrain relationships.
- Tiny differences based on five trials of one person cannot establish a universal effect or statistical significance.
- Check convergence and any undefined T per run in OT_results_all.json; no missing values are imputed.
- Baseline trial splits share a subject and session; external replication must include other sessions or subjects.
