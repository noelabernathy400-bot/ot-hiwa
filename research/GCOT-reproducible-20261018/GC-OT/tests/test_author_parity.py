"""Numerical comparisons to the unmodified PyHiWA author code."""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
from gcot.alignment import hierarchical_alignment, hard_measures, SolverConfig
from gcot.numerics import sinkhorn, squared_cost, whiten
from gcot.dad import density


def author_module():
    path = Path(__file__).resolve().parents[2] / "PyHiWA/src/hiwa.py"
    spec = importlib.util.spec_from_file_location("author_hiwa", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sinkhorn_and_whitening_match_unmodified_author():
    author = author_module()
    rng = np.random.default_rng(14)
    x, y = rng.normal(size=(17, 3)), rng.normal(size=(19, 3))
    a, b = np.full(17, 1/17), np.full(19, 1/19)
    reference, _ = author._sinkhorn(a[:, None], b[:, None], x.T, y.T, 2., 150)
    actual = sinkhorn(a, b, squared_cost(x, y), 2., 150, tol=0)
    np.testing.assert_allclose(actual, reference, atol=1e-12)
    np.testing.assert_allclose(whiten(x)[0], author._normal(x), atol=1e-12)


def test_hiwa_admm_matches_author_with_identical_random_stream():
    author = author_module()
    rng = np.random.default_rng(19)
    x, y = rng.normal(size=(16, 2)), rng.normal(size=(16, 2))
    labels = np.repeat([0, 1], 8)

    class Identity:
        def fit_transform(self, z):
            return z

    fit = author.HiWA(dim_red_method=Identity(), normalize=False, maxiter=10,
                      sa_maxiter=20, shorn_maxiter=1000, sa_shorn_maxiter=150)
    stream = np.random.default_rng(23)
    with patch.object(np.random, "random", stream.random):
        fit.fit(x, labels, y, labels, X_transform=np.eye(2), Y_transform=np.eye(2))
    actual = hierarchical_alignment(hard_measures(x, labels), hard_measures(y, labels), 23,
                                    SolverConfig(maxiter=10, inner_maxiter=20, sinkhorn_tol=0))
    np.testing.assert_allclose(actual["R"], fit.Rg, atol=1e-8)
    np.testing.assert_allclose(actual["P"], fit.P, atol=1e-8)


def test_dad_knn_density_matches_unmodified_author():
    root = Path(__file__).resolve().parents[1] / "vendor/DAD"
    spec = importlib.util.spec_from_file_location("author_dad", root/"decoder/main.py")
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(root), *sys.path]):
        spec.loader.exec_module(module)
    rng = np.random.default_rng(32)
    for dimension in [2, 3]:
        points, queries = rng.normal(size=(21, dimension)), rng.normal(size=(40, dimension))
        np.testing.assert_allclose(density(queries, points, 5), module.prob1(queries, points, 5), atol=1e-14)
