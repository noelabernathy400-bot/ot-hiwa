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
| PCA3_PCA3 | 0.67107 | 0.67345 | 0.89875 | 0.90585 | 0.14266 | 0.14262 |
| FA3_PCA3 | 0.8723 | 0.86958 | 1.34169 | 1.3498 | 0.14737 | 0.14749 |
| PCA3_FA3 | 0.74238 | 0.7451 | 0.82101 | 0.82136 | 0.15085 | 0.15145 |
| FA3_FA3 | 0.71264 | 0.71589 | 1.05483 | 1.0601 | 0.14466 | 0.14513 |
| PCA5_RAW5 | 1.09919 | 1.06716 | 1.3519 | 1.26003 | 0.1616 | 0.15518 |
| FA5_RAW5 | 0.86093 | 0.83287 | 1.35457 | 1.26164 | 0.15459 | 0.1495 |

## Correct inference

- Absolute W2 or T between different FA/PCA embeddings is NOT a common metric because the learned geometries differ.
- Compare Soft−Hard differences WITHIN each representation; the physical five-finger RMSE is also in common physical units.
- Heldout shared T is evaluated after re-solving P/Q on a SAME Hard Kmeans cluster reference within each representation; it is NOT the cost of freezing the training P/Q.
- The physical test RMSE uses paired timestamps ONLY AFTER unpaired R has been learned and target-only force decoder has been fitted on train data.
- A target-only decoder is not a supervised EMG-to-force model: it does not learn paired Xtrain/Ytrain relationships.
- Tiny differences based on five trials of one person cannot establish a universal effect or statistical significance.
- Check convergence and any undefined T per run in OT_results_all.json; no missing values are imputed.
- Baseline trial splits share a subject and session; external replication must include other sessions or subjects.
