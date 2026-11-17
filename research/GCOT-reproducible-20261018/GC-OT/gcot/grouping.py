"""Author SSC port and TPAMI Eq. (2) soft prototypes."""
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.special import softmax, xlogy
from sklearn.cluster import KMeans


def prototype_loss_grad(flat, x, groups, entropy_weight):
    """Exact Eq. (2): ||A C - X||_F² + lambda_h/N sum A log A.

    A = row_softmax(X C.T); C contains row prototypes. No temperature,
    hardening, sample deletion, or class labels are introduced.
    """
    c = flat.reshape(groups, x.shape[1])
    a = softmax(x @ c.T, axis=1)
    e = a @ c - x
    loss = np.sum(e * e) + entropy_weight * np.sum(xlogy(a, a)) / len(x)
    da = 2 * e @ c.T + entropy_weight / len(x) * (np.log(np.maximum(a, 1e-300)) + 1)
    dz = a * (da - np.sum(da * a, axis=1, keepdims=True))
    grad = 2 * a.T @ e + dz.T @ x
    return float(loss), grad.ravel()


def soft_groups(x, groups, seed=0, entropy_weight=1.0, maxiter=300):
    # Initialization unspecified in TPAMI: one unsupervised k-means++ run.
    c0 = KMeans(groups, n_init=1, random_state=seed).fit(x).cluster_centers_
    fit = minimize(prototype_loss_grad, c0.ravel(), args=(x, groups, entropy_weight),
                   method="L-BFGS-B", jac=True,
                   options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-7})
    c = fit.x.reshape(groups, x.shape[1])
    a = softmax(x @ c.T, axis=1)
    # Machine-level underflow only; every group retains the full support.
    a = np.maximum(a, np.finfo(float).tiny)
    a /= a.sum(axis=1, keepdims=True)
    return a, c, {"success": bool(fit.success), "iterations": int(fit.nit),
                  "loss": float(fit.fun), "message": str(fit.message)}


def ssc(x, groups, seed=0, alpha=3000.0, rho=0.8, maxiter=200):
    """Port JHUVisionLab admmLasso_mat_func (affine=false), thrC,
    BuildAdjacency and SpectralClustering with HiWASSC.m settings.
    """
    if groups == 1:
        return np.zeros(len(x), dtype=int)
    n = len(x)
    gram = x @ x.T
    off = gram.copy()
    np.fill_diagonal(off, 0)
    lam = np.max(np.abs(off), axis=0).min()
    if lam <= 0:
        raise ValueError("SSC degenerate input: computeLambda_mat is zero")
    mu1, mu2 = alpha / lam, alpha
    inverse = np.linalg.inv(mu1 * gram + mu2 * np.eye(n))
    c, dual = np.zeros((n, n)), np.zeros((n, n))
    for _ in range(maxiter - 1):
        z = inverse @ (mu1 * gram + mu2 * c - dual)
        np.fill_diagonal(z, 0)
        v = z + dual / mu2
        c = np.sign(v) * np.maximum(np.abs(v) - 1 / mu2, 0)
        np.fill_diagonal(c, 0)
        dual += mu2 * (z - c)
        if np.max(np.abs(z - c)) <= 2e-4:
            break
    trimmed = np.zeros_like(c)
    for j in range(n):
        idx = np.argsort(-np.abs(c[:, j]))
        count = np.searchsorted(np.cumsum(np.abs(c[idx, j])), rho * np.abs(c[:, j]).sum()) + 1
        trimmed[idx[:count], j] = np.abs(c[idx[:count], j])
    trimmed /= trimmed.max(axis=0, keepdims=True) + np.finfo(float).eps
    affinity = trimmed + trimmed.T
    invdegree = 1 / np.sqrt(affinity.sum(axis=0) + np.finfo(float).eps)
    lap = np.eye(n) - invdegree[:, None] * affinity * invdegree[None, :]
    _, vectors = eigh(lap, subset_by_index=(0, groups - 1))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + np.finfo(float).eps
    return KMeans(groups, n_init=20, max_iter=1000, random_state=seed).fit_predict(vectors)
