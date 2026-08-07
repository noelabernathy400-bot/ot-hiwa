# ROCA geometry--objective disagreement audit

This is a post-hoc diagnostic of existing fair full-support results. The disagreement marker is computed from unlabeled outputs only; direction accuracy and $R^2$ are shown only after the marker is fixed.

## Frozen diagnostic

- Geometric selector: representative signed-volume sign.
- Objective selector: determinant candidate with lower final transport objective.
- Abstention marker: the two signs disagree.
- This marker is not a replacement selector and was not tuned with labels.

## Aggregate result

- Seeds: 10
- Geometry selector accuracy (evaluation only): 0.800
- Disagreement rejects: 8/10 (coverage 0.200)
- Geometry accuracy among non-disagreements (evaluation only): 1.000
- Wrong geometric selections rejected (evaluation only): 2/2

## Per-seed diagnostics

| seed | geometry sign | objective sign | disagree? | relative objective gap | geometry correct (evaluation only) | selected accuracy | alternative accuracy |
| ---: | ---: | ---: | :---: | ---: | :---: | ---: | ---: |
| 80 | -1 | +1 | yes | 0.0000 | yes | 0.5417 | 0.4271 |
| 81 | -1 | +1 | yes | 0.0000 | yes | 0.5417 | 0.4271 |
| 82 | -1 | +1 | yes | 0.0000 | yes | 0.5417 | 0.4271 |
| 83 | -1 | -1 | no | 0.0649 | yes | 0.5417 | 0.2500 |
| 84 | -1 | -1 | no | 0.0000 | yes | 0.5417 | 0.4271 |
| 85 | -1 | +1 | yes | 0.0649 | no | 0.2292 | 0.4271 |
| 86 | -1 | +1 | yes | 0.0652 | no | 0.2292 | 0.4271 |
| 87 | -1 | +1 | yes | 0.0000 | yes | 0.5417 | 0.4271 |
| 88 | -1 | +1 | yes | 0.0000 | yes | 0.5417 | 0.4271 |
| 89 | -1 | +1 | yes | 0.0000 | yes | 0.5417 | 0.4271 |

## Interpretation

The result is evidence that geometry--objective disagreement may be useful as an abstention signal. It is not evidence that lower transport objective should replace ROCA: any disagreement can contain either a geometric error or an objective-preference error. The next experiment must calibrate this marker on known synthetic truth and test it on independent held-out seeds.

## Inputs

- `experiments/results/taco_faithful_baseline_full_support_roca_seeds_80_84.json`
- `experiments/results/taco_faithful_baseline_full_support_roca_seeds_85_89.json`
