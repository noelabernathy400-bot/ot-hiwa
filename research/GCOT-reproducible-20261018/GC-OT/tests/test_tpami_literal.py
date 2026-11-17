import numpy as np
from scipy.optimize import check_grad
from scipy.special import xlogy
from gcot.alignment import TPAMIGCOT, SolverConfig, weighted_measures, hierarchical_alignment
from gcot.grouping import prototype_loss_grad
from gcot.numerics import sinkhorn, squared_cost


def test_eq2_gradient_includes_assignment_dependence():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(9, 3))
    c = rng.normal(size=6)
    fun = lambda v: prototype_loss_grad(v, x, 2, .8)[0]
    jac = lambda v: prototype_loss_grad(v, x, 2, .8)[1]
    assert check_grad(fun, jac, c) < 2e-5


def test_eq10_sample_entropy_not_multiplied_by_group_mass():
    a = np.array([.2, .8]); b = np.array([.7, .3])
    cost = np.array([[0., 4.], [2., .1]])
    pij, epsilon = .07, .13
    q = sinkhorn(a, b, cost, epsilon/pij, 2000, 1e-12)
    wrong = sinkhorn(a, b, cost, epsilon, 2000, 1e-12)
    objective = lambda z: pij*np.sum(z*cost) + epsilon*np.sum(xlogy(z,z))
    assert objective(q) < objective(wrong) - 1e-5
    np.testing.assert_allclose(q.sum(axis=1), a, atol=1e-10)
    np.testing.assert_allclose(q.sum(axis=0), b, atol=1e-10)


def test_literal_solver_full_support_and_unscaled_objective():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(12, 2)); y = rng.normal(size=(15, 2))
    cfg = SolverConfig(maxiter=8, inner_maxiter=10, sinkhorn_maxiter=200,
                       group_sinkhorn_maxiter=200, sinkhorn_tol=1e-10)
    fit = TPAMIGCOT(2, 3, config=cfg).fit(x, y)
    a, b = weighted_measures(x, fit.A_), weighted_measures(y, fit.B_)
    reference = hierarchical_alignment(a,b,3,cfg,scale=False,retain_plans=True)
    np.testing.assert_allclose(fit.R_, reference['R'], atol=1e-12)
    np.testing.assert_allclose(fit.R_.T @ fit.R_, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(fit.P_.sum(axis=0), [.5,.5], atol=1e-9)
    objective = cfg.epsilon_group*np.sum(xlogy(fit.P_, fit.P_))
    for i in range(2):
        for j in range(2):
            q = fit.result_[f'Q_{i}_{j}']
            assert q.shape == (12,15)
            np.testing.assert_allclose(q.sum(axis=1), a[i][1], atol=1e-8)
            np.testing.assert_allclose(q.sum(axis=0), b[j][1], atol=1e-8)
            objective += fit.P_[i,j]*np.sum(q*squared_cost(fit.transform(x),y))
            objective += cfg.epsilon_sample*np.sum(xlogy(q,q))
    np.testing.assert_allclose(objective,fit.result_['objective'],atol=1e-10)
