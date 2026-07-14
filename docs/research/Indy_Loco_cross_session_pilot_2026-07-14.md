# Indy--Loco cross-session transfer pilot

_Data-adapter and applicability audit, 2026-07-14_

## Decision

The public Indy--Loco dataset is a scientifically meaningful external test for
this project: it asks whether a velocity decoder trained on one neural
recording day can be reused on a later day after unlabelled neural alignment.
It is not a second copy of the original HiWA neural--movement point-cloud
task.

The first cross-day pilot **does not validate the current Soft-GCOT solver as
a reliable transfer method**.  In three fixed initialisations it improved the
source-only decoder numerically, but none of the three ADMM solves converged.
Those numbers are exploratory diagnostics, not performance claims.

## Verified source and local integrity

- Official source: O'Doherty, Cardoso, Makin, and Sabes, *Nonhuman Primate
  Reaching with Multichannel Sensorimotor Cortex Electrophysiology*, Zenodo
  record 3854034, DOI `10.5281/zenodo.788569`.
- Accessed 2026-07-14.  The record provides sorted spike timestamps and
  synchronised fingertip, cursor, and target positions for repeated sessions.
- Downloaded pilot sessions are ignored by Git and passed the official MD5
  checks:
  - `indy_20160915_01.mat`: `ef6a95c5a1a8d2126b90f5be0505e398`;
  - `indy_20160921_01.mat`: `788a0fc779bdfffa8388b1108e1d1ce5`.

The new `datasets.indy_loco` adapter reads the MATLAB 7.3/HDF5 sessions,
retains timestamped spike trains and behaviour separately, bins spikes at 50
ms, smooths rates, excludes units below 0.5 Hz, and computes cursor velocity.
The two downloaded sessions contain respectively 7,621 and 7,203 bins, with
223 and 220 retained units.  Behaviour is never an implicit input to the
alignment solver.

## Fixed pilot protocol

| Item | Fixed choice |
| --- | --- |
| Source session | `indy_20160915_01` |
| Target session | `indy_20160921_01` |
| Neural representation | separate source/target standardisation and 8D PCA |
| Source decoder | ridge regression from source neural latent state to 2D cursor velocity |
| Target adaptation | earliest 60% of target neural bins only; no cursor position or velocity supplied to OT |
| Target test | last 40% of target bins; velocity opened only after alignment for metrics |
| Solver subset | 192 evenly spaced bins per fitting/evaluation partition |
| Groups | four learned prototypes, temperature 0.5, full support only |
| Selection | no target behaviour, target score, or target trajectory used for model selection |

The target-supervised ridge model below is an explicitly labelled upper-bound
diagnostic.  It is not an unsupervised baseline and is not used to select any
parameter.

## Results

### Fixed 30-iteration pilot

| Seed | Source-only $R^2$ | Hard-group HiWA $R^2$ | Soft-GCOT $R^2$ | Target-supervised oracle $R^2$ | Soft-GCOT converged? |
| ---: | ---: | ---: | ---: | ---: | --- |
| 701 | -0.1110 | -0.0374 | 0.0176 | 0.1277 | no |
| 702 | -0.1110 | -0.1607 | -0.0723 | 0.1277 | no |
| 703 | -0.1110 | -0.2848 | 0.1174 | 0.1277 | no |
| Mean | -0.1110 | -0.1610 | 0.0209 | 0.1277 | 0/3 |

Soft-GCOT exceeded source-only in all three exploratory runs, and its mean
eight-sector velocity-direction accuracy was 14.2% versus 9.4% for
source-only.  This cannot be interpreted as evidence of a method improvement:
the final ADMM primal residuals were 4.066, 3.442, and 3.741, whereas a
converged solution requires the configured tolerance of 0.1.

### Convergence probes on seed 703

| Variant | Iterations | Final primal residual | $R^2$ | Result |
| --- | ---: | ---: | ---: | --- |
| Frozen uniform consensus | 100 | 3.1246 | 0.0486 | not converged |
| Experimental transport-weighted consensus | 100 | 3.0837 | 0.0579 | not converged |

The experimental variant changes only the global consensus aggregation: it
weights local rotations by the current group transport instead of weighting
all group pairs equally.  It is retained as a tested negative diagnostic; the
default `uniform` solver and existing baselines are unchanged.

## What the new data revealed

This is a useful failure mode, not a reason to keep tuning blindly:

1. There is a real cross-session decoding problem: source-only transfer is
   below zero $R^2$, while a labelled target-session decoder reaches positive
   $R^2$.
2. Current unpaired hard and soft hierarchical OT do not reach ADMM consensus
   on this task.
3. Increasing iterations and reweighting the consensus do not resolve that
   failure.
4. Therefore the present solver is not yet a transferable cross-session neural
   alignment method, and no performance conclusion may be drawn from the
   provisional Soft-GCOT scores.

## Next boundary

Do not add more sessions or tune temperatures, group counts, or ADMM weights
against this session pair.  A future method revision must state a new,
falsifiable mechanism for cross-session correspondence--for example a
behaviourally meaningful *source-only* state representation coupled to an
unlabelled target-domain consistency objective--and must first demonstrate
ADMM convergence on this fixed pair before any cross-session performance
claim.  If it cannot beat the source-only decoder with converged solutions,
the appropriate conclusion is that the current hierarchical OT assumption is
too weak for unconstrained cross-session neural drift.
