"""First controlled OT-guided JOINT membership/alignment pilot on DANDI 000138.

We optimize four source-group-specific softness gates theta_k in [0,1].
Every membership is determined by the source feature and fixed KMeans label,
and is copied to its simultaneously observed target window. No test
labels or aligned target data are used for predicting memberships.

Block coordinate scheme: (R,P,Q) <- original ZIP GCOT solver with fixed A(theta)
then theta <- minimize fixed-(R,P) *full hierarchical* entropic OT with a
KL(Q || a x b) correction and quadratic anchor, Q reoptimized at each theta.
This is a PARAMETERIZED block-coordinate optimization, NOT a proof that
arbitrary memberships are globally optimized. Updated theta is fitted
only on TRAIN samples. No full-trial paired regression loss is used.

Geometry uses four-dimensional original-normalized squared distances, with
ZIP solver's 1/sqrt(4) scaling in optimizer. The baseline Hard/Soft/Boundary
results use the same frozen feature representation and test OT samples.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import softmax,xlogy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from zip_gcot.alignment import hierarchical_alignment,SolverConfig
from zip_gcot.numerics import sinkhorn
from run_controlled_soft_hard_ablation import build_measures
from run_hard_hiwa import sample_groups
from run_boundary_adaptive_ablation import old_test_eval if False else sample_eval
