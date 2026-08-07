# ROCA geometry--objective disagreement audit

This is a blind evaluation of a diagnostic frozen before these seeds were inspected. The disagreement marker is computed from unlabeled outputs only; direction accuracy and $R^2$ are shown only after the marker is fixed.

## Frozen diagnostic

- Geometric selector: representative signed-volume sign.
- Objective selector: determinant candidate with lower final transport objective.
- Abstention marker: the two signs disagree and the relative objective gap is at least 0.010.
- This marker is not a replacement selector and was not tuned with labels.

## Aggregate result

- Runs: 3
- Geometry selector accuracy (evaluation only): 0.667
- Material-conflict rejects: 0/3 (coverage 1.000)
- Geometry accuracy among non-rejected seeds (evaluation only): 0.667
- Wrong geometric selections rejected (evaluation only): 0/1

## Per-run diagnostics

| case | solver seed | subset seed | geometry sign | objective sign | material conflict? | relative objective gap | geometry correct (evaluation only) | selected accuracy | alternative accuracy |
| --- | ---: | ---: | ---: | ---: | :---: | ---: | :---: | ---: | ---: |
| subset_1001_solver_0 | 0 | 1001 | -1 | -1 | no | 0.0000 | no | 0.0833 | 0.1042 |
| subset_1002_solver_0 | 0 | 1002 | -1 | -1 | no | 0.0000 | yes | 0.1667 | 0.0417 |
| subset_1003_solver_0 | 0 | 1003 | +1 | -1 | no | 0.0000 | yes | 0.3333 | 0.2500 |

## Interpretation

The result is evidence that geometry--objective disagreement may be useful as an abstention signal. It is not evidence that lower transport objective should replace ROCA: any disagreement can contain either a geometric error or an objective-preference error.

## Inputs

- `experiments/results/taco_faithful_baseline_roca_cross_subsample_1001_solver0.json`
- `experiments/results/taco_faithful_baseline_roca_cross_subsample_1002_solver0.json`
- `experiments/results/taco_faithful_baseline_roca_cross_subsample_1003_solver0.json`
