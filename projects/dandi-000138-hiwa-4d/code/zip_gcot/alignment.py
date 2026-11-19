"""Nested weighted OT with local orthogonal blocks and consensus ADMM.

HiWA: MATLAB HiWASSC.m / PyHiWA hiwa.py. GCOT: TPAMI Eqs. 2-12.
All sample arrays have samples in rows. A transform is X @ R.T.
"""
from dataclasses import dataclass
import numpy as np
from .numerics import polar, sinkhorn, squared_cost
from .grouping import soft_groups


@dataclass
class SolverConfig:
    maxiter: int = 300
    tol: float = 0.1
    mu: float = 0.005
    epsilon_group: float = 0.2
    epsilon_sample: float = 0.1
    inner_maxiter: int = 100
    inner_tol: float = 0.01
    sinkhorn_maxiter: int = 150
    group_sinkhorn_maxiter: int = 1000
    sinkhorn_tol: float = 1e-8
    # Legacy is retained only for reproducibility of archived author runs.
    stopping: str = "legacy"
    admm_atol: float = 1e-4
    admm_rtol: float = 1e-3
    marginal_tol: float = 1e-8
    coupling_tol: float = 1e-5
    stable_iterations: int = 3


def convergence_metrics(local, global_r, previous_r, dual, p, previous_p,
                        marginal, cfg, penalty):
    """Consensus ADMM primal/dual residuals in the actual scaled objective."""
    blocks = local.shape[0] * local.shape[1]
    delta = float(np.linalg.norm(global_r - previous_r))
    primal = float(np.linalg.norm(local - global_r))
    dual_residual = float(penalty * np.sqrt(blocks) * delta)
    eps_primal = float(np.sqrt(local.size) * cfg.admm_atol + cfg.admm_rtol * max(
        np.linalg.norm(local), np.sqrt(blocks) * np.linalg.norm(global_r)))
    eps_dual = float(np.sqrt(local.size) * cfg.admm_atol + cfg.admm_rtol *
                     np.linalg.norm(penalty * dual))
    coupling_delta = float(np.linalg.norm(p - previous_p))
    passed = (delta <= cfg.tol and primal <= eps_primal and
              dual_residual <= eps_dual and marginal <= cfg.marginal_tol and
              coupling_delta <= cfg.coupling_tol)
    return dict(rotation_delta=delta, consensus_residual=primal,
                dual_residual=dual_residual, primal_tolerance=eps_primal,
                dual_tolerance=eps_dual, marginal_error=float(marginal),
                coupling_delta=coupling_delta, criteria_satisfied=bool(passed))


def hard_measures(x, labels):
    return [(x[labels == k], np.full(np.sum(labels == k), 1 / np.sum(labels == k)))
            for k in np.unique(labels)]


def weighted_measures(x, memberships):
    return [(x, memberships[:, k] / memberships[:, k].sum())
            for k in range(memberships.shape[1])]


def hierarchical_alignment(source_measures, target_measures, seed=0, config=None, scale=True,
                           retain_plans=False, progress=None, initial_R=None,
                           local_start_from_global=False):
    """Source-grounded local alternating OT/Procrustes + ADMM.

    Local Q minimizes P_ij <Q,C> + epsilon_sample H(Q), hence the
    Sinkhorn temperature is epsilon_sample/P_ij, NOT epsilon_sample.
    For HiWA scale=True reproduces X/sqrt(D), T=mu/D (Rg-L).
    GCOT scale=False uses the literal unscaled TPAMI Eq. (12).
    """
    cfg = config or SolverConfig()
    if cfg.stopping not in ("legacy", "admm"):
        raise ValueError("stopping must be legacy or admm")
    if cfg.maxiter < 1 or cfg.stable_iterations < 1:
        raise ValueError("Positive iteration budgets required")
    d = source_measures[0][0].shape[1]
    nx, ny = len(source_measures), len(target_measures)
    rng = np.random.default_rng(seed)
    # Optional train-only external geometric initialization.  The historical
    # default remains bitwise identical: same RNG stream, polar(rand).
    if initial_R is None:
        rglobal = polar(rng.random((d, d)))
    else:
        init = np.asarray(initial_R, dtype=float)
        if init.shape != (d, d) or not np.isfinite(init).all():
            raise ValueError("initial_R must be a finite D x D orthogonal matrix")
        if np.linalg.norm(init.T @ init - np.eye(d), ord="fro") > 1e-7:
            raise ValueError("initial_R must already be orthogonal")
        rglobal = init.copy()
    dual = np.zeros((nx, ny, d, d))
    local = np.tile(np.eye(d), (nx, ny, 1, 1))
    p = np.full((nx, ny), 1 / (nx * ny))
    a_group, b_group = np.full(nx, 1 / nx), np.full(ny, 1 / ny)
    divisor = np.sqrt(d) if scale else 1.0
    xs = [(np.ascontiguousarray(x / divisor), a) for x, a in source_measures]
    ys = [(np.ascontiguousarray(y / divisor), b) for y, b in target_measures]
    history = []
    residual_history = []
    stable = 0
    converged = False
    max_marginal = 0.0
    plans = {} if retain_plans else None
    for step in range(cfg.maxiter):
        costs = np.zeros((nx, ny))
        current_marginal = 0.0
        previous_p = p.copy()
        for i, (x, a) in enumerate(xs):
            for j, (y, b) in enumerate(ys):
                pij = max(p[i, j], np.finfo(float).tiny)
                penalty = cfg.mu / divisor ** 2 * (rglobal - dual[i, j])
                # Author resets Q to independent marginals for each block call.
                q = np.outer(a, b)
                # Legacy: reinitialize each local R randomly, as in MATLAB HiWA.
                # Optional warm start is a distinct *experimental* solver variant.
                r = rglobal.copy() if local_start_from_global else polar(rng.random((d, d)))
                for _ in range(cfg.inner_maxiter):
                    prev = r
                    r = polar(2 * pij * (y.T @ q.T @ x) + penalty)
                    cost = squared_cost(x @ r.T, y)
                    q = sinkhorn(a, b, cost, cfg.epsilon_sample / pij,
                                 cfg.sinkhorn_maxiter, cfg.sinkhorn_tol)
                    if np.linalg.norm(r - prev, 2) <= cfg.inner_tol:
                        break
                local[i, j] = r
                if retain_plans:
                    plans[f"Q_{i}_{j}"] = q
                costs[i, j] = np.sum(q * cost)
                current_marginal = max(current_marginal, float(np.max(np.abs(q.sum(axis=1)-a))),
                                   float(np.max(np.abs(q.sum(axis=0)-b))))
        p = sinkhorn(a_group, b_group, costs, cfg.epsilon_group,
                     cfg.group_sinkhorn_maxiter, cfg.sinkhorn_tol)
        prev_global = rglobal
        rglobal = polar((local + dual).mean(axis=(0, 1)))
        dual += local - rglobal
        current_marginal = max(current_marginal,
                               float(np.max(np.abs(p.sum(axis=1) - a_group))),
                               float(np.max(np.abs(p.sum(axis=0) - b_group))))
        max_marginal = max(max_marginal, current_marginal)
        metrics = convergence_metrics(local, rglobal, prev_global, dual, p,
                                      previous_p, current_marginal, cfg,
                                      cfg.mu / divisor ** 2)
        delta = float(np.linalg.norm(rglobal - prev_global))
        consensus = float(np.linalg.norm(local - rglobal))
        history.append([delta, consensus, float(np.sum(p * costs))])
        residual_history.append([metrics[k] for k in (
            "rotation_delta", "consensus_residual", "dual_residual", "primal_tolerance",
            "dual_tolerance", "marginal_error", "coupling_delta")])
        stable = stable + 1 if metrics["criteria_satisfied"] else 0
        numerical_failure = not np.isfinite(p).all()
        legacy_converged = step >= 5 and delta <= cfg.tol and not numerical_failure
        admm_converged = step >= 5 and stable >= cfg.stable_iterations
        converged = legacy_converged if cfg.stopping == "legacy" else admm_converged
        if progress is not None:
            progress(dict(iteration=step + 1, converged=converged, **metrics))
        # Match HiWA's stop event: after five completed iterations, stop on
        # global-rotation stagnation or a non-finite group coupling.  Keep the
        # latter classified as failure rather than calling it convergence.
        if step >= 5 and (converged or numerical_failure):
            break
    result = {"R": rglobal, "P": p, "cost": costs, "history": np.asarray(history),
            "max_marginal_error": max_marginal,
            "orthogonality_error": float(np.linalg.norm(rglobal.T @ rglobal - np.eye(d)))}
    result.update(metrics, converged=converged,
                  residual_history=np.asarray(residual_history),
                  residual_columns=np.array(["rotation_delta", "consensus_residual",
                      "dual_residual", "primal_tolerance", "dual_tolerance",
                      "marginal_error", "coupling_delta"]),
                  stop_reason=("numerical_failure" if numerical_failure else
                               "admm_tolerance" if cfg.stopping == "admm" and converged else
                               "rotation_tolerance" if cfg.stopping == "legacy" and converged
                               else "iteration_limit"))
    if retain_plans:
        from scipy.special import xlogy
        global_cost = np.zeros_like(costs)
        entropy = 0.0
        for i, (x, a) in enumerate(xs):
            for j, (y, b) in enumerate(ys):
                q = plans[f"Q_{i}_{j}"]
                global_cost[i, j] = np.sum(q * squared_cost(x @ rglobal.T, y))
                entropy += float(np.sum(xlogy(q, q)))
        result.update(plans)
        result.update(local_R=local, dual=dual, global_cost=global_cost,
                      objective=float(np.sum(p * global_cost) + cfg.epsilon_sample * entropy
                                      + cfg.epsilon_group * np.sum(xlogy(p, p))),
                      consensus_residual=consensus, rotation_delta=delta)
    return result


class GCOT:
    """TPAMI soft prototype grouping and group-conditioned alignment.

    There is deliberately no label argument. The authors' unspecified
    prototype optimizer and regularizers are explicit constructor parameters.
    """
    def __init__(self, groups, seed=0, entropy_weight=1.0, prototype_maxiter=300, config=None,
                 hiwa_scaling=False, projections=None):
        self.groups, self.seed = groups, seed
        self.entropy_weight, self.prototype_maxiter = entropy_weight, prototype_maxiter
        self.config = config or SolverConfig()
        self.hiwa_scaling, self.projections = hiwa_scaling, projections

    def fit(self, x, y):
        self.A_, self.source_prototypes_, self.source_group_fit_ = soft_groups(
            x, self.groups, self.seed, self.entropy_weight, self.prototype_maxiter)
        self.B_, self.target_prototypes_, self.target_group_fit_ = soft_groups(
            y, self.groups, self.seed + 1, self.entropy_weight, self.prototype_maxiter)
        # Group on the same unprojected inputs as HiWA-GMM; project only the
        # alignment objective, and apply the learned R to original coordinates.
        if self.projections is not None:
            from .preprocessing import project_for_alignment
            xp, yp = project_for_alignment(x, y, self.projections)
        else:
            xp, yp = x, y
        self.result_ = hierarchical_alignment(weighted_measures(xp, self.A_),
                                              weighted_measures(yp, self.B_),
                                              self.seed, self.config, scale=self.hiwa_scaling)
        self.R_, self.P_ = self.result_["R"], self.result_["P"]
        return self

    def transform(self, x):
        return x @ self.R_.T


class TPAMIGCOT(GCOT):
    """Eqs. (2)-(12), with an optional explicit auxiliary preprocessing step.

    Equal group priors, full-support soft measures, positive negative-entropy
    regularizers. Numerical budgets are explicit implementation choices because
    the supplied manuscript does not specify its supplementary ADMM procedure.
    """

    def __init__(self, groups, seed=0, entropy_weight=1.0, prototype_maxiter=300,
                 config=None, retain_plans=True, projections=None, progress=None):
        super().__init__(groups, seed, entropy_weight, prototype_maxiter,
                         config or SolverConfig(stopping="admm", tol=1e-4),
                         hiwa_scaling=False, projections=projections)
        self.retain_plans = retain_plans
        self.progress = progress

    def fit(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
            raise ValueError("Source and target must have the same feature dimension")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Input features must be finite")
        self.A_, self.source_prototypes_, self.source_group_fit_ = soft_groups(
            x, self.groups, self.seed, self.entropy_weight, self.prototype_maxiter)
        self.B_, self.target_prototypes_, self.target_group_fit_ = soft_groups(
            y, self.groups, self.seed + 1, self.entropy_weight, self.prototype_maxiter)
        if self.projections is None:
            xp, yp = x, y
        else:
            from .preprocessing import project_for_alignment
            xp, yp = project_for_alignment(x, y, self.projections)
        self.result_ = hierarchical_alignment(weighted_measures(xp, self.A_),
                                              weighted_measures(yp, self.B_), self.seed,
                                              self.config, scale=False,
                                              retain_plans=self.retain_plans,
                                              progress=self.progress)
        self.R_, self.P_ = self.result_["R"], self.result_["P"]
        return self
