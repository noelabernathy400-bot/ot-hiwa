# DANDI 000688 cross-day experiment

This experiment uses two center-out sessions from monkey M:

- source: 2015-06-10
- target: 2015-06-12

The raw files are from DANDI dandiset `000688`, published version
`0.250122.1735`.  Cursor speed determines movement onset.  Spike counts are
binned in the interval `[-50, 400] ms`, Gaussian-smoothed, square-rooted,
soft-normalized per neuron, and independently reduced to four PCA dimensions
per day.  Each trial is represented by its mean latent state in the window.

Reach-direction labels never enter PCA, grouping, subsampling, or alignment.
They are retained solely for the final direction-transfer and cluster-matching
metrics.

## Reproduce the accepted experiment

Place the two NWB files in `data/dandi_000688/raw`, then run this single
command from the repository root:

```powershell
python -m experiments.dandi_000688.run data/dandi_000688/raw results/dandi_000688
```

The reported experiment is one direct reproducible run. It is not swept,
averaged, or selected using direction labels.

## Outputs

- `latents.json`: provenance, trial counts, unit counts, and preprocessing.
- `latents.npz`: 4D trial summaries and full latent trajectories.
- `aligned.npz`: all aligned domains and evaluation labels.
- `metrics.csv`: final comparison table.
- `figure2a_dandi.{png,svg,pdf}`: HiWA neural Figure 2a-style first-two-latent-dimension panels.
- `alignment_accuracy.{png,pdf}`: quantitative comparison.
- `run_summary.json`: machine-readable completion record.

The public comparison set is GCOT, WA, SA, ICP, and CORAL. DAD is not part of
this experiment.
