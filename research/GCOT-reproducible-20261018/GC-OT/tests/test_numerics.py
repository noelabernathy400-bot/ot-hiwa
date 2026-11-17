import numpy as np
from scipy.optimize import check_grad, linprog
from gcot.numerics import sinkhorn, squared_cost, polar, alignment_error
from gcot.grouping import prototype_loss_grad, soft_groups, ssc
from gcot.alignment import hierarchical_alignment, hard_measures, weighted_measures, SolverConfig
from gcot.data import synthetic, subset_indices
from gcot.dad import direction_rotation
def summaries(values):
    values = np.asarray(values)
    if len(values) != 3 or not np.isfinite(values).all():
        raise ValueError("requires three finite values")
    return float(values.min()), float(np.median(values)), float(values.max())


def test_nonuniform_ot_marginals_and_exact_limit():
    a, b = np.array([.2, .3, .5]), np.array([.4, .6])
    cost = np.array([[0., 1.], [.4, .5], [.8, 0.]])
    plan = sinkhorn(a, b, cost, .002, maxiter=20000, tol=1e-11)
    np.testing.assert_allclose(plan.sum(1), a, atol=1e-9)
    np.testing.assert_allclose(plan.sum(0), b, atol=1e-9)
    constraints = np.vstack([np.kron(np.eye(3), np.ones((1, 2))), np.kron(np.ones((1, 3)), np.eye(2))])
    exact = linprog(cost.ravel(), A_eq=constraints, b_eq=np.r_[a, b], bounds=(0, None), method="highs")
    assert exact.success
    assert abs(np.sum(cost * plan) - exact.fun) < .005


def test_sinkhorn_log_fallback_extreme_cost():
    a = np.array([.5, .5])
    plan = sinkhorn(a, a, np.array([[0., 1000.], [1000., 0.]]), .01)
    np.testing.assert_allclose(plan, np.eye(2)/2, atol=1e-12)


def test_prototype_gradient_is_derivative_of_paper_objective():
    rng = np.random.default_rng(9)
    x, c = rng.normal(size=(12, 3)), rng.normal(size=(2, 3)).ravel()
    fn = lambda c: prototype_loss_grad(c, x, 2, .7)
    assert check_grad(lambda c: fn(c)[0], lambda c: fn(c)[1], c) < 1e-4


def test_soft_groups_use_full_support_and_hard_limit():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(20, 3))
    a, _, _ = soft_groups(x, 3, seed=5)
    np.testing.assert_allclose(a.sum(1), 1)
    measures = weighted_measures(x, a)
    assert all(len(z) == len(x) and np.all(w > 0) for z, w in measures)
    assert all(np.isclose(w.sum(), 1) for _, w in measures)


def test_known_correspondence_rotation_recovery_and_metric():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(100, 3))
    true = polar(rng.normal(size=(3, 3)))
    y = x @ true.T
    recovered = polar(y.T @ x)
    np.testing.assert_allclose(recovered, true, atol=1e-12)
    assert alignment_error(x @ recovered.T, y) < 1e-24


def test_weighted_measures_match_single_partition_limit():
    rng = np.random.default_rng(6)
    x, y = rng.normal(size=(12, 2)), rng.normal(size=(10, 2))
    cfg = SolverConfig(maxiter=7, inner_maxiter=10, epsilon_sample=1.0)
    first = hierarchical_alignment(hard_measures(x, np.zeros(12)), hard_measures(y, np.zeros(10)), 4, cfg, False)
    second = hierarchical_alignment(weighted_measures(x, np.ones((12, 1))), weighted_measures(y, np.ones((10, 1))), 4, cfg, False)
    np.testing.assert_allclose(first["R"], second["R"])
    assert first["orthogonality_error"] < 1e-12
    assert first["max_marginal_error"] < 1e-6


def test_synthetic_and_sampling_are_paired_deterministic():
    for panel, shape in [("e", (100, 2)), ("f", (250, 6))]:
        data = synthetic(panel, 1)
        assert data["source"].shape == shape
        np.testing.assert_array_equal(data["source"], synthetic(panel, 1)["source"])
        assert not np.allclose(data["target"], data["source"] @ data["true_R"].T)
        assert np.linalg.det(data["true_R"]) > .999999
    labels = np.repeat(np.arange(4), 20)
    small = subset_indices(labels, .05, 1)
    large = subset_indices(labels, .5, 1)
    assert set(small) <= set(large)


def test_dad_rotation_and_three_run_summary():
    for direction in [[0, 0, -1], [1, 2, 3], [0, 0, 1]]:
        r = direction_rotation(np.array(direction, dtype=float))
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(r @ [0, 0, 1], np.array(direction)/np.linalg.norm(direction), atol=1e-12)
    assert summaries([.9, .2, .4]) == (.2, .4, .9)


def test_ssc_returns_requested_number_of_clusters():
    data = synthetic("f", 3)
    labels = ssc(data["source"], 5, seed=3)
    assert len(np.unique(labels)) == 5
