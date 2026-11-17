"""Shared numerical primitives. Transforms act as X @ R.T."""
import numpy as np
from scipy.special import logsumexp
from scipy.spatial.distance import cdist


def polar(m):
    u, _, vt = np.linalg.svd(m, full_matrices=False)
    return u @ vt  # O(d), including reflections as in HiWA and TPAMI Eq. 7.


def squared_cost(x, y):
    return cdist(x, y, "sqeuclidean")


def sinkhorn(a, b, cost, epsilon, maxiter=1000, tol=1e-8):
    """Entropy OT; kernel scaling with a log-domain underflow fallback.

    No cost/epsilon rescaling or arbitrary kernel floors. Early stopping tests
    BOTH marginals; tol=0 reproduces the author's fixed iteration budget.
    """
    a, b, cost = np.asarray(a), np.asarray(b), np.asarray(cost)
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    if cost.shape != (len(a), len(b)) or np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("Positive marginals and matching cost shape required")
    if not (np.isclose(a.sum(), 1) and np.isclose(b.sum(), 1)):
        raise ValueError("Marginals must sum to one")
    # Row/column additive cost shifts preserve the entropic OT minimizer.
    # They avoid needless log-domain work when only irrelevant far entries
    # underflow, while preserving at least one nonzero entry per row/column.
    shifted = cost - cost.min(axis=1, keepdims=True)
    shifted -= shifted.min(axis=0, keepdims=True)
    z = -shifted / epsilon
    if True:
        kernel = np.exp(z)
        v = np.ones_like(b)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for step in range(maxiter):
                u = a / (kernel @ v)
                v = b / (kernel.T @ u)
                if not np.isfinite(v).all():
                    break
                if (step + 1) % 10 == 0 and tol:
                    if np.max(np.abs(u * (kernel @ v) - a)) <= tol:
                        break
            plan = (u[:, None] * kernel) * v[None, :]
        if np.isfinite(plan).all():
            return plan
    la, lb = np.log(a), np.log(b)
    v = np.zeros_like(b)
    for step in range(maxiter):
        u = la - logsumexp(z + v[None, :], axis=1)
        v = lb - logsumexp(z + u[:, None], axis=0)
        if (step + 1) % 10 == 0 and tol:
            rows = np.exp(u + logsumexp(z + v[None, :], axis=1))
            if np.max(np.abs(rows - a)) <= tol:
                break
    return np.exp(z + u[:, None] + v[None, :])


def covariance_power(cov, power):
    eig, vec = np.linalg.eigh(cov)
    floor = max(float(eig.max()), 1.0) * 1e-12
    return (vec * np.maximum(eig, floor) ** power) @ vec.T


def whiten(x):
    mean = x.mean(axis=0)
    matrix = covariance_power(np.cov(x, rowvar=False), -0.5)
    return (x - mean) @ matrix, mean, matrix


def alignment_error(aligned, true_aligned):
    return float(np.sum((aligned - true_aligned) ** 2) / np.sum(true_aligned ** 2))
