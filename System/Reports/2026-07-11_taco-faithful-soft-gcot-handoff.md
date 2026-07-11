# 2026-07-11: TACO-faithful Soft-GCOT HiWA baseline

## Goal

Freeze and verify the TACO-faithful Soft-GCOT HiWA main line before further extensions.

## Changes

- Added explicit `full` versus `sparse` support modes to `SoftHiWA`.
- Recorded soft group masses, group-marginal errors, ADMM primal residual, and support mode; early stopping now checks global and ADMM residuals.
- Fixed the migrated neural-data path so runners work from this repository alone.
- Added the unified four-method neural baseline runner and a full-support test.
- Updated README/AI context and added audit/result documents.

## Evidence and risk

The minimal 96-by-96 one-seed smoke run completed and is recorded in `experiments/results/taco_faithful_baseline_20260711_minimal.json`. It is deliberately under-iterated and non-converged, so it is not a performance conclusion. ROCA had a warning and did not improve this run.

## Next step

Run a fixed multi-seed, converged full-support comparison before reconsidering ROCA or any CC/PBMC extension.
