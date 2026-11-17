"""Author synthetic generator port and original neural data loading."""
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.linalg import orth, svd
from sklearn.decomposition import FactorAnalysis
from .numerics import whiten

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


def synthetic(panel, seed):
    # Author-code precedence: e=2 clusters / 2D, f=5 clusters / 6D.
    groups, dim, intrinsic, n = (2, 2, 2, 50) if panel == "e" else (5, 6, 2, 50)
    rng = np.random.default_rng(seed)
    bases = [orth(rng.standard_normal((dim, intrinsic))) for _ in range(groups)]
    r0 = orth(rng.random((dim, dim)))
    v, _, ut = svd(r0)
    signs = np.eye(dim)
    signs[-1, -1] = np.linalg.det(ut.T @ v.T)
    true_r = ut.T @ signs @ v.T
    means, covariances = [], []
    for _ in range(groups):
        means.append(rng.standard_normal(intrinsic))
        matrix = rng.random((intrinsic, intrinsic))
        eigenvalues, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
        covariances.append((vectors * np.maximum(eigenvalues, 0)) @ vectors.T
                           + intrinsic * rng.random() * np.eye(intrinsic))
    x, y = [], []
    for basis, mean, cov in zip(bases, means, covariances):
        x.append(rng.multivariate_normal(mean, cov, n) @ basis.T)
        y.append(rng.multivariate_normal(mean, cov, n) @ basis.T @ true_r.T)
    return {"source": np.vstack(x), "target": np.vstack(y), "true_R": true_r,
            "source_labels": np.repeat(np.arange(groups), n),
            "target_labels": np.repeat(np.arange(groups), n), "groups": groups}


def neural(seed=0):
    mat = loadmat(WORKSPACE / "PyHiWA/data/mihi_demo.mat")
    counts = mat["Yte"]
    counts = counts[:, np.any(counts[1:] != counts[:-1], axis=0)]
    fa = FactorAnalysis(n_components=3, random_state=seed)
    source_raw = fa.fit_transform(counts)
    movement = mat["Xtr"]
    target_raw = np.column_stack((movement, np.linalg.norm(movement, axis=1)))
    source, sm, sw = whiten(source_raw)
    target, tm, tw = whiten(target_raw)
    return {"source": source, "target": target,
            "source_raw": source_raw, "target_raw": target_raw,
            "source_mean": sm, "source_whitening": sw,
            "target_mean": tm, "target_whitening": tw,
            "source_labels": mat["Tte"].ravel(), "target_labels": mat["Ttr"].ravel(),
            "fa_components": fa.components_, "fa_mean": fa.mean_,
            "groups": 4}


def subset_indices(labels, density, seed):
    """Nested stratified subsampling; labels control experimental density only."""
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.permutation(np.flatnonzero(labels == k))[
        :max(1, int(np.floor(density * np.sum(labels == k))))] for k in np.unique(labels)])
