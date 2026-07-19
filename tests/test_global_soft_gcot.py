from __future__ import annotations

import numpy as np

from global_soft_gcot import GlobalSoftGCOT


def test_global_solver_recovers_a_shared_rotation_and_converges() -> None:
    rng = np.random.default_rng(5)
    source = rng.normal(size=(24, 2))
    rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]])
    target = source @ rotation.T
    assignments = np.zeros((24, 2))
    assignments[:12, 0] = 1.0
    assignments[12:, 1] = 1.0

    model = GlobalSoftGCOT(
        maxiter=80,
        tol=1e-5,
        group_gamma=0.05,
        sample_gamma=0.5,
        sinkhorn_maxiter=500,
    ).fit(source, assignments, target, assignments)

    assert model.diagnostics["converged"]
    assert model.diagnostics["rotation_orthogonality_error"] < 1e-10
    assert model.diagnostics["max_sinkhorn_marginal_error"].max() < 1e-6
    assert np.linalg.norm(model.transform(source) - target) < 0.1


def test_global_solver_respects_reflection_component() -> None:
    rng = np.random.default_rng(7)
    source = rng.normal(size=(18, 3))
    reflection = np.diag([1.0, 1.0, -1.0])
    target = source @ reflection.T
    assignments = np.ones((18, 1))
    model = GlobalSoftGCOT(
        maxiter=80,
        tol=1e-5,
        group_gamma=0.05,
        sample_gamma=0.5,
        sinkhorn_maxiter=500,
        determinant_sign=-1,
    ).fit(source, assignments, target, assignments)
    assert model.diagnostics["converged"]
    assert np.linalg.det(model.Rg) < 0
    assert model.diagnostics["rotation_orthogonality_error"] < 1e-10
    assert model.diagnostics["max_sinkhorn_marginal_error"].max() < 1e-6
