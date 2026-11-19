# Hyser 5D: best train-only 2D views and correct nearest-neighbor evaluation

Nearest target neighbor retrieved among 250 heldout Y windows: j*(i)=argmin_j||R x_i-y_j||. Radius eps is calibrated on TRAIN-only unrelated inter-trial target distances, 5th percentile; same for Hard and Soft. Full-dimensional Euclidean distances, never 2D distances, are used in evaluation.

**WARNING**: d(Rx,Y_nn)<eps is NOT evidence of correct sample prediction; this is point-cloud coverage only. The real external check is ||Y_nn - Y_true(i)||<eps, using synchrony only after model fitting. Report exact synchronized time index retrieval and random-baseline hit as independent measures.

**Not original paper's identical metric**: the paper's real neural decoding accuracy uses reach-direction cluster labels, which Hyser Random Task does not contain.

## Fivefold fixed-seed means

| Embedding | Method | Train-only eps | NN cloud coverage | Correct physical neighborhood | Exact time index | Random target hit |
|---|---|---:|---:|---:|---:|---:|
|PCA5_RAW5|hard|0.3788|39.880%|9.960%|0.400%|9.341%|
|PCA5_RAW5|soft05|0.3788|40.760%|12.000%|0.360%|9.341%|
|FA5_RAW5|hard|0.3788|45.600%|12.080%|0.760%|9.341%|
|FA5_RAW5|soft05|0.3788|49.000%|13.600%|0.760%|9.341%|

## Figure details
The best two-coordinate view is chosen among the 10 coordinate pairs using TRAINING Kmeans 4 labels and TRAINING silhouette, for each independent domain. The figure is fixed to FA5, heldout trial1, seed42 Soft (not chosen by performance).
All 10 coordinate pairs and the best 3-coordinate projection are saved; separate source and target figures are never overlaid. The five-dimensional geometry CANNOT be faithfully represented by one two-dimensional figure; the 5x5 matrix is a view of all coordinate pairs.
Using test labels to choose the 'nicest' view would be visualization selection bias.
No additional R/P/Q optimization is performed in this stage, only frozen 120step global-R models.
## Limitation
This does not substitute for a full original-local-R-ADMM HiWA rerun. The physical labels here are actual synchronized force measurements used only in heldout evaluation.
