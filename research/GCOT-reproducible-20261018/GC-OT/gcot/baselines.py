"""Ports of the authors' baselines; see docs/PROVENANCE.md for deviations."""
import numpy as np
from scipy.linalg import svd
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.manifold import Isomap
from .numerics import polar, squared_cost, sinkhorn, covariance_power
from .grouping import ssc
from .alignment import hierarchical_alignment, hard_measures, SolverConfig


def wa(x, y, seed=0, maxiter=200, sinkhorn_maxiter=1000):
    # hiwa-matlab/code/toolbox/WA.m: annealing, no label/true R dependence.
    rng = np.random.default_rng(seed)
    r = svd(rng.random((x.shape[1], x.shape[1])))[0]
    a, b = np.full(len(x), 1 / len(x)), np.full(len(y), 1 / len(y))
    gamma = 100.0
    for _ in range(maxiter):
        if gamma > 0.1:
            gamma *= 0.95
        q = sinkhorn(a, b, squared_cost(x @ r.T, y), gamma, sinkhorn_maxiter)
        r = polar(y.T @ q.T @ x)
    return {"aligned": x @ r.T, "R": r}


def coral(x, y):
    # Direct row-convention port of author CORAL.m, including +I regularizer.
    a = covariance_power(np.cov(x, rowvar=False) + np.eye(x.shape[1]), -0.5)
    a = a @ covariance_power(np.cov(y, rowvar=False) + np.eye(y.shape[1]), 0.5)
    return {"aligned": x @ a, "R": a.T}


def sa(x, y):
    # Keep the actual author's full-basis formula, not a different SA variant.
    def basis(z):
        sd = z.std(axis=0, ddof=1)
        z = (z - z.mean(axis=0)) / np.where(sd > 0, sd, 1)
        return PCA(n_components=z.shape[1], svd_solver="full").fit(z).components_.T
    a = basis(x).T @ basis(y)
    return {"aligned": x @ a, "R": a.T}


def icp(x, y, maxiter=100):
    """Source->target point-to-point ICP, allowed reflections.

    Author figure scripts call ICP(model=source,data=target), then apply
    the returned reverse map to source, ignoring translation. We correct that
    call-site bug and implement the paper's explicitly stated reflection mode.
    """
    r, translation = np.eye(x.shape[1]), np.zeros(x.shape[1])
    points = x.copy()
    tree = cKDTree(y)
    prev = np.inf
    for step in range(maxiter):
        distances, idx = tree.query(points)
        error = float(np.mean(distances ** 2))
        source_mean, target_mean = points.mean(axis=0), y[idx].mean(axis=0)
        rotation = polar((y[idx] - target_mean).T @ (points - source_mean))
        shift = target_mean - source_mean @ rotation.T
        r, translation = rotation @ r, translation @ rotation.T + shift
        points = points @ rotation.T + shift
        if step >= 4 and abs(prev - error) < 1e-5:
            break
        prev = error
    return {"aligned": points, "R": r, "translation": translation}


def hiwa_ssc(x, y, groups, seed=0, config=None):
    lx, ly = ssc(x, groups, seed), ssc(y, groups, seed + 1)
    out = hierarchical_alignment(hard_measures(x, lx), hard_measures(y, ly), seed, config)
    out.update(aligned=x @ out["R"].T, source_clusters=lx, target_clusters=ly)
    return out


def hiwa_gmm(x, y, groups=4, seed=0, config=None, projections=None):
    """Historical GMM recipe + current neural demo's Isomap projection.

    Inputs already normalized, consistently with the public fit() method.
    Fixes its fit/transform normalization mismatch at the caller level.
    Labels from GMM remain anonymous, independently fitted in both domains.
    """
    lx = GaussianMixture(groups, random_state=seed, n_init=1).fit_predict(x)
    ly = GaussianMixture(groups, random_state=seed + 1, n_init=1).fit_predict(y)
    from .preprocessing import neural_projections, project_for_alignment
    tx, ty = neural_projections(x, y) if projections is None else projections
    xp, yp = project_for_alignment(x, y, (tx, ty))
    out = hierarchical_alignment(hard_measures(xp, lx), hard_measures(yp, ly), seed, config)
    out.update(aligned=x @ out["R"].T, source_clusters=lx, target_clusters=ly,
               source_projection=tx, target_projection=ty)
    return out
