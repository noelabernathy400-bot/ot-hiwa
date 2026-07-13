from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "cc_hiwa"))

from soft_hiwa import SoftHiWA
from solver_adapter import solve_soft_gcot_detached
from transport_consistent_joint import global_coupling


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    rng = np.random.default_rng(123)
    source = np.vstack((rng.normal(-1.0, 0.15, (6, 2)), rng.normal(1.0, 0.15, (6, 2))))
    rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]])
    target = source @ rotation.T
    source_assignments = np.zeros((12, 2))
    source_assignments[:6, 0] = 1.0
    source_assignments[6:, 1] = 1.0
    target_assignments = source_assignments.copy()
    settings: dict[str, object] = {
        "normalize": False,
        "maxiter": 6,
        "tol": 10.0,
        "sa_maxiter": 3,
        "shorn_maxiter": 60,
        "sa_shorn_maxiter": 60,
        "support_mode": "full",
        "random_state": 17,
        "warm_start_local": True,
    }
    return source, source_assignments, target, target_assignments, settings


def test_detached_adapter_exactly_matches_direct_full_support_solver() -> None:
    source, source_assignments, target, target_assignments, settings = _fixture()
    direct = SoftHiWA(**settings).fit(
        source,
        source_assignments,
        target,
        target_assignments,
        initial_rotation=np.eye(2),
    )
    solution = solve_soft_gcot_detached(
        source,
        source_assignments,
        target,
        target_assignments,
        solver_kwargs=settings,
        fit_kwargs={"initial_rotation": np.eye(2)},
    )

    np.testing.assert_allclose(solution.group_transport, direct.P, atol=1e-12)
    np.testing.assert_allclose(solution.rotation, direct.Rg, atol=1e-12)
    np.testing.assert_allclose(solution.aligned_source, direct.transform(source), atol=1e-12)
    np.testing.assert_allclose(
        solution.sample_coupling,
        global_coupling(direct.local_couplings, direct.P),
        atol=1e-12,
    )
    assert np.isclose(solution.sample_coupling.sum(), 1.0)


def test_detached_adapter_rejects_sparse_support() -> None:
    source, source_assignments, target, target_assignments, settings = _fixture()
    settings["support_mode"] = "sparse"
    try:
        solve_soft_gcot_detached(
            source,
            source_assignments,
            target,
            target_assignments,
            solver_kwargs=settings,
        )
    except ValueError as error:
        assert "support_mode='full'" in str(error)
    else:
        raise AssertionError("sparse support must be rejected by the detached adapter")
