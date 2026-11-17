"""Single source of truth for the accepted DANDI 000688 experiment."""

from __future__ import annotations

from dataclasses import dataclass

from gcot.alignment import SolverConfig


@dataclass(frozen=True)
class PreprocessingConfig:
    latent_dimension: int = 4
    window_ms: tuple[float, float] = (-50.0, 400.0)
    bin_ms: float = 10.0
    smoothing_ms: float = 50.0
    onset_threshold_fraction: float = 0.10
    onset_sustain_ms: float = 30.0


PREPROCESSING = PreprocessingConfig()
N_GROUPS = 8

# The experiment is a single reproducible run. This value fixes stochastic
# optimizer initialization; it is not exposed as an experimental variable.
FIXED_INITIALIZATION = 7

SOLVER = SolverConfig(
    stopping="legacy",
    maxiter=40,
    tol=0.05,
    inner_maxiter=30,
    inner_tol=0.01,
    sinkhorn_maxiter=250,
    group_sinkhorn_maxiter=600,
    sinkhorn_tol=1e-8,
)

SOURCE_FILENAME = "sub-M_ses-CO-20150610_behavior+ecephys.nwb"
TARGET_FILENAME = "sub-M_ses-CO-20150612_behavior+ecephys.nwb"
DATASET_ID = "DANDI:000688"
DATASET_VERSION = "0.250122.1735"
