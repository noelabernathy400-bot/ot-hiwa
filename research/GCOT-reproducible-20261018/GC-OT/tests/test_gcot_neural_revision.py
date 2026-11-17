import numpy as np
from gcot.alignment import (SolverConfig, TPAMIGCOT, convergence_metrics,
                            hierarchical_alignment, weighted_measures)
from gcot.preprocessing import project_for_alignment
from experiments.neural.core import display_clouds, CFG


def test_neural_runner_uses_exact_hiwa_iteration_and_stop_settings():
    assert (CFG.stopping, CFG.maxiter, CFG.tol) == ('legacy', 300, .1)
    assert (CFG.inner_maxiter, CFG.inner_tol) == (100, .01)
    assert (CFG.sinkhorn_maxiter, CFG.group_sinkhorn_maxiter, CFG.sinkhorn_tol) == (150, 1000, 0)


def test_stagnant_rotation_is_not_consensus_convergence():
    eye = np.eye(2)
    local = np.tile(-eye, (2, 2, 1, 1))
    p = np.ones((2, 2)) / 4
    m = convergence_metrics(local, eye, eye, np.zeros_like(local), p, p, 0, CFG, .005)
    assert m['rotation_delta'] == 0
    assert not m['criteria_satisfied']
    local[:] = eye
    assert convergence_metrics(local, eye, eye, np.zeros_like(local), p, p, 0, CFG, .005)['criteria_satisfied']
    assert not convergence_metrics(local, eye, eye, np.zeros_like(local), p, p, 1e-3, CFG, .005)['criteria_satisfied']


def test_dual_and_coupling_residuals_are_required():
    eye = np.eye(2)
    local = np.tile(eye, (2, 2, 1, 1))
    p = np.ones((2, 2)) / 4
    cfg = SolverConfig(tol=10, stopping='admm')
    assert not convergence_metrics(local, eye, -eye, np.zeros_like(local), p, p, 0, cfg, 1)['criteria_satisfied']
    assert not convergence_metrics(local, eye, eye, np.zeros_like(local), p, np.eye(2)/2, 0, cfg, 1)['criteria_satisfied']


def test_solver_requires_consecutive_passes_before_reporting_convergence():
    x = np.zeros((4, 2))
    measures = [(x, np.full(4, .25))]
    cfg = SolverConfig(maxiter=12, stopping='admm', stable_iterations=8, tol=1e-4)
    result = hierarchical_alignment(measures, measures, 7, cfg, scale=False)
    assert len(result['history']) == 8
    assert result['converged']
    assert result['stop_reason'] == 'admm_tolerance'
    assert result['marginal_error'] <= cfg.marginal_tol


def test_hiwa_stop_uses_previous_estimate_not_ground_truth():
    x = np.zeros((4, 2))
    measures = [(x, np.full(4, .25))]
    cfg = SolverConfig(maxiter=20, stopping='legacy', tol=.1)
    result = hierarchical_alignment(measures, measures, 7, cfg, scale=False)
    assert len(result['history']) == 6
    assert result['converged']
    assert result['stop_reason'] == 'rotation_tolerance'


def test_projection_matches_direct_core_and_preserves_orthogonality():
    rng = np.random.default_rng(12)
    x, y = rng.normal(size=(10, 3)), rng.normal(size=(11, 3))
    transforms = (rng.normal(size=(3, 2)), np.eye(3)[:, :2])
    cfg = SolverConfig(maxiter=3, inner_maxiter=3, stopping='admm')
    model = TPAMIGCOT(2, 3, config=cfg, projections=transforms).fit(x, y)
    xp, yp = project_for_alignment(x, y, transforms)
    ref = hierarchical_alignment(weighted_measures(xp, model.A_), weighted_measures(yp, model.B_),
                                  3, cfg, scale=False)
    np.testing.assert_allclose(model.R_, ref['R'])
    np.testing.assert_allclose(model.R_.T @ model.R_, np.eye(3), atol=1e-12)
    assert not model.result_['converged']
    assert model.result_['stop_reason'] == 'iteration_limit'


def test_author_python_display_uses_raw_source_target_and_fit_rotation():
    rng = np.random.default_rng(9)
    d = dict(source=rng.normal(size=(8, 3)), target=rng.normal(size=(7, 3)),
             source_raw=rng.normal(size=(8, 3)), Xtr=rng.normal(size=(7, 2)))
    r = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    dad = rng.normal(size=(8, 2))
    clouds = display_clouds(d, r, r, dad)
    np.testing.assert_array_equal(clouds['Source'], d['source_raw'])
    np.testing.assert_array_equal(clouds['Target'], d['Xtr'])
    np.testing.assert_array_equal(clouds['GCOT'], d['source_raw'] @ r.T)
    np.testing.assert_array_equal(clouds['WA'], d['source_raw'] @ r.T)
    np.testing.assert_array_equal(clouds['DAD'], dad)


def test_figure1_panel_names_follow_matlab_scripts():
    from experiments.reference import AUTHOR_PANEL_MAPPING
    assert AUTHOR_PANEL_MAPPING == {'e': 'e', 'f': 'f'}
