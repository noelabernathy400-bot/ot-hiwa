# ROCA geometry--objective disagreement audit

This is a post-hoc diagnostic of existing fair full-support results. The disagreement marker is computed from unlabeled outputs only; direction accuracy and $R^2$ are shown only after the marker is fixed.

## Frozen diagnostic

- Geometric selector: representative signed-volume sign.
- Objective selector: determinant candidate with lower final transport objective.
- Abstention marker: the two signs disagree and the relative objective gap is at least 0.010.
- This marker is not a replacement selector and was not tuned with labels.

## Aggregate result

- Seeds: 1
- Geometry selector accuracy (evaluation only): 1.000
- Material-conflict rejects: 0/1 (coverage 1.000)
- Geometry accuracy among non-rejected seeds (evaluation only): 1.000
- Wrong geometric selections rejected (evaluation only): 0/0

## Per-seed diagnostics

| seed | geometry sign | objective sign | material conflict? | relative objective gap | geometry correct (evaluation only) | selected accuracy | alternative accuracy |
| ---: | ---: | ---: | :---: | ---: | :---: | ---: | ---: |
| 90 | -1 | -1 | no | 0.0000 | yes | 0.5417 | 0.4271 |

## Interpretation

The result is evidence that geometry--objective disagreement may be useful as an abstention signal. It is not evidence that lower transport objective should replace ROCA: any disagreement can contain either a geometric error or an objective-preference error. The next experiment must calibrate this marker on known synthetic truth and test it on independent held-out seeds.

## Inputs

- `experiments/results/taco_faithful_baseline_roca_material_conflict_blind_seed_90.json`
