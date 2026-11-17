"""Frozen configuration for the Panoptic 171204_pose2 experiment."""

from gcot.alignment import SolverConfig


SEQUENCE = "171204_pose2"
SOURCE_CAMERA = "00_00"
TARGET_CAMERA = "00_01"
PERSON_ID = 0
FIRST_FRAME = 141
LAST_FRAME = 339
N_GROUPS = 3
FIXED_INITIALIZATION = 7
PCA_DIMENSION = 2
TSNE_INPUT_DIMENSION = 10

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
