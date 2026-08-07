# ROCA geometry--objective disagreement audit

This is a blind evaluation of a diagnostic frozen before these seeds were inspected. The disagreement marker is computed from unlabeled outputs only; direction accuracy and $R^2$ are shown only after the marker is fixed.

## Frozen diagnostic

- Geometric selector: representative signed-volume sign.
- Objective selector: determinant candidate with lower final transport objective.
- Abstention marker: the two signs disagree and the relative objective gap is at least 0.010.
- This marker is not a replacement selector and was not tuned with labels.

## Aggregate result

- Runs: 5
- Geometry selector accuracy (evaluation only): 0.800
- Material-conflict rejects: 0/5 (coverage 1.000)
- Geometry accuracy among non-rejected seeds (evaluation only): 0.800
- Wrong geometric selections rejected (evaluation only): 0/1

## Per-run diagnostics

| case | solver seed | subset seed | geometry sign | objective sign | material conflict? | relative objective gap | geometry correct (evaluation only) | selected accuracy | alternative accuracy |
| --- | ---: | ---: | ---: | ---: | :---: | ---: | :---: | ---: | ---: |
| subset_1001_solver_0 | 0 | 1001 | -1 | -1 | no | 0.0000 | no | 0.0833 | 0.1042 |
| subset_1002_solver_0 | 0 | 1002 | -1 | -1 | no | 0.0000 | yes | 0.1667 | 0.0417 |
| subset_1003_solver_0 | 0 | 1003 | +1 | -1 | no | 0.0000 | yes | 0.3333 | 0.2500 |
| subset_1004_solver_0 | 0 | 1004 | +1 | -1 | no | 0.0000 | yes | 0.2083 | 0.1250 |
| subset_1005_solver_0 | 0 | 1005 | +1 | -1 | no | 0.0000 | yes | 0.3646 | 0.1667 |

## Interpretation

The frozen material-conflict marker failed to reject at least one wrong geometric selection. This result falsifies its use as a general-purpose ROCA confidence mechanism on this protocol.
Lower transport objective is not a justified ROCA replacement: any disagreement can contain either a geometric error or an objective-preference error.

## Inputs

- `experiments/results/taco_faithful_baseline_roca_cross_subsample_1001_solver0.json`
- `experiments/results/taco_faithful_baseline_roca_cross_subsample_1002_solver0.json`
- `experiments/results/taco_faithful_baseline_roca_cross_subsample_1003_solver0.json`
- `experiments/results/taco_faithful_baseline_roca_cross_subsample_1004_solver0.json`
- `experiments/results/taco_faithful_baseline_roca_cross_subsample_1005_solver0.json`
