# ROCA geometry--objective disagreement audit

This is a blind evaluation of a diagnostic frozen before these seeds were inspected. The disagreement marker is computed from unlabeled outputs only; direction accuracy and $R^2$ are shown only after the marker is fixed.

## Frozen diagnostic

- Geometric selector: representative signed-volume sign.
- Objective selector: determinant candidate with lower final transport objective.
- Abstention marker: the two signs disagree and the relative objective gap is at least 0.010.
- This marker is not a replacement selector and was not tuned with labels.

## Aggregate result

- Seeds: 6
- Geometry selector accuracy (evaluation only): 1.000
- Material-conflict rejects: 0/6 (coverage 1.000)
- Geometry accuracy among non-rejected seeds (evaluation only): 1.000
- Wrong geometric selections rejected (evaluation only): 0/0

## Per-seed diagnostics

| seed | geometry sign | objective sign | material conflict? | relative objective gap | geometry correct (evaluation only) | selected accuracy | alternative accuracy |
| ---: | ---: | ---: | :---: | ---: | :---: | ---: | ---: |
| 90 | -1 | -1 | no | 0.0000 | yes | 0.5417 | 0.4271 |
| 91 | -1 | -1 | no | 0.0000 | yes | 0.5417 | 0.4271 |
| 92 | -1 | -1 | no | 0.0000 | yes | 0.5417 | 0.4271 |
| 93 | -1 | +1 | no | 0.0000 | yes | 0.5417 | 0.4271 |
| 94 | -1 | -1 | no | 0.0000 | yes | 0.5417 | 0.4271 |
| 95 | -1 | -1 | no | 0.0000 | yes | 0.5417 | 0.4271 |

## Interpretation

The result is evidence that geometry--objective disagreement may be useful as an abstention signal. It is not evidence that lower transport objective should replace ROCA: any disagreement can contain either a geometric error or an objective-preference error.

## Inputs

- `experiments/results/taco_faithful_baseline_roca_material_conflict_blind_seed_90.json`
- `experiments/results/taco_faithful_baseline_roca_material_conflict_blind_seed_91.json`
- `experiments/results/taco_faithful_baseline_roca_material_conflict_blind_seed_92.json`
- `experiments/results/taco_faithful_baseline_roca_material_conflict_blind_seed_93.json`
- `experiments/results/taco_faithful_baseline_roca_material_conflict_blind_seed_94.json`
- `experiments/results/taco_faithful_baseline_roca_material_conflict_blind_seed_95.json`
