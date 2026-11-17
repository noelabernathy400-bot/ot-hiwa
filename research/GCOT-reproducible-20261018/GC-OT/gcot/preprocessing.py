"""Shared, label-free neural projection for HiWA-GMM and GCOT."""
import numpy as np
from sklearn.manifold import Isomap


def neural_projections(x, y):
    embedded = Isomap(n_components=2, n_neighbors=min(12, len(x)-1)).fit_transform(x)
    return np.linalg.pinv(x) @ embedded, np.linalg.pinv(y) @ y[:, :2]


def project_for_alignment(x, y, projections):
    tx, ty = projections
    return x @ tx @ tx.T, y @ ty @ ty.T
