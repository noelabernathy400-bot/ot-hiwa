# Data availability

## Included

- Small demonstration inputs under `data/` that are required by checked unit
  and smoke-test paths.
- Experiment scripts under `experiments/` and compact numerical records under
  `experiments/results/`.
- Figures, manuscripts, mathematical contracts, and protocol notes needed to
  interpret the recorded results.

## Not included

The following raw sources are deliberately excluded from Git:

| Source | Reason for exclusion | Reproduction route |
| --- | --- | --- |
| PAMAP2 wearable-sensor archive | Large third-party archive | Obtain from the original PAMAP2 distribution and place it under `data/raw/pamap2/`. |
| Indy-Loco neural recordings | Large research data with source-specific terms | Follow the documented Indy-Loco protocol; recorded diagnostics remain in `experiments/results/`. |
| Paderborn bearing archives | Large third-party files | Obtain from the Paderborn bearing dataset source and place under `data/bearing/raw/`. |
| PBMC multiome `.h5ad` files | Large external single-cell files | Use the preparation scripts in `experiments/pbmc/`; raw files belong under `data/pbmc/raw/`. |
| CMU Panoptic recordings | Large third-party video/pose archive | Follow the acquisition and protocol notes in `docs/research/`. |

No raw dataset is needed to inspect the code, read the paper, run the test
suite, or examine the frozen numerical result records.  External-data results
must be read with their stated convergence and applicability boundaries.
