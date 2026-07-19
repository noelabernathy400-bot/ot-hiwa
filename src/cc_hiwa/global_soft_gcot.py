"""Globally constrained hierarchical soft-group optimal transport.

This experimental solver keeps the TACO-style soft groups, group transport and
within-group couplings, but uses one orthogonal map for every group pair.  It
is deliberately separate from :class:`SoftHiWA`: it tests whether the local
rotation ADMM consensus, rather than soft grouping itself, is the obstacle to
cross-session transfer.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

try:
    from .soft_hiwa import EPS, _closed_form_rotation, sinkhorn_groups
except ImportError:  # Supports the repository's legacy direct-module test path.
    from soft_hiwa import EPS, _closed_form_rotation, sinkhorn_groups


def _log_sinkhorn_weighted(
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    gamma: float,
    maxiter: int,
) -> tuple[np.ndarray, float, float]:
    """Numerically stable entropic OT for the global solver's local plans."""
    p = np.asarray(source_mass, dtype=float).ravel()
    q = np.asarray(target_mass, dtype=float).ravel()
    p = p / p.sum()
    q = q / q.sum()
    x2 = np.sum(source**2, axis=0)
    y2 = np.sum(target**2, axis=0)
    cost = np.maximum(y2[None, :] + x2[:, None] - 2.0 * (source.T @ target), 0.0)
    log_kernel = -cost / gamma
    log_u = np.zeros_like(p)
    log_v = np.zeros_like(q)
    log_p = np.log(np.maximum(p, EPS))
    log_q = np.log(np.maximum(q, EPS))
    for _ in range(maxiter):
        log_u = log_p - logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = log_q - logsumexp(log_kernel + log_u[:, None], axis=0)
    log_coupling = log_u[:, None] + log_kernel + log_v[None, :]
    coupling = np.exp(log_coupling)
    marginal_error = max(
        float(np.max(np.abs(coupling.sum(axis=1) - p))),
        float(np.max(np.abs(coupling.sum(axis=0) - q))),
    )
    return coupling, float(np.sum(cost * coupling)), marginal_error


class GlobalSoftGCOT:
    """Two-level OT with a single globally shared orthogonal alignment.

    Alternating updates are exact for the rotation block: after the local
    couplings and group transport are fixed, the global map is the weighted
    orthogonal Procrustes solution.  This model has no local rotations and no
    ADMM consensus variable.
    """

    def __init__(
        self,
        *,
        maxiter: int = 100,
        tol: float = 1e-4,
        group_gamma: float = 0.2,
        sample_gamma: float = 0.5,
        sinkhorn_maxiter: int = 300,
        determinant_sign: int | None = None,
    ) -> None:
        if maxiter < 1 or sinkhorn_maxiter < 1:
            raise ValueError("iteration limits must be positive")
        if tol <= 0 or group_gamma <= 0 or sample_gamma <= 0:
            raise ValueError("tol and Sinkhorn regularisation values must be positive")
        if determinant_sign not in (None, -1, 1):
            raise ValueError("determinant_sign must be -1, +1, or None")
        self.maxiter = int(maxiter)
        self.tol = float(tol)
        self.group_gamma = float(group_gamma)
        self.sample_gamma = float(sample_gamma)
        self.sinkhorn_maxiter = int(sinkhorn_maxiter)
        self.determinant_sign = determinant_sign

    @staticmethod
    def _validate(values: np.ndarray, assignments: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(values, dtype=float)
        a = np.asarray(assignments, dtype=float)
        if x.ndim != 2 or a.ndim != 2 or x.shape[0] != a.shape[0]:
            raise ValueError(f"{name} values and assignments must be rank-2 with matching rows")
        if np.any(a < 0) or np.max(np.abs(a.sum(axis=1) - 1.0)) > 1e-6:
            raise ValueError(f"{name} assignment rows must be non-negative and sum to one")
        return x, a

    def fit(
        self,
        source: np.ndarray,
        source_assignments: np.ndarray,
        target: np.ndarray,
        target_assignments: np.ndarray,
        *,
        initial_rotation: np.ndarray | None = None,
    ) -> "GlobalSoftGCOT":
        x, a = self._validate(source, source_assignments, "source")
        y, b = self._validate(target, target_assignments, "target")
        if x.shape[1] != y.shape[1]:
            raise ValueError("source and target must have the same feature dimension")
        d = x.shape[1]
        if initial_rotation is None:
            rotation = np.eye(d)
        else:
            rotation = np.asarray(initial_rotation, dtype=float)
            if rotation.shape != (d, d):
                raise ValueError("initial_rotation has the wrong shape")
            rotation = _closed_form_rotation(rotation, self.determinant_sign)

        source_weights = [a[:, k] / max(float(a[:, k].sum()), EPS) for k in range(a.shape[1])]
        target_weights = [b[:, l] / max(float(b[:, l].sum()), EPS) for l in range(b.shape[1])]
        group_transport = np.full((a.shape[1], b.shape[1]), 1.0 / (a.shape[1] * b.shape[1]))
        residuals: list[float] = []
        objective_history: list[float] = []
        marginal_errors: list[float] = []
        local_couplings: list[list[np.ndarray]] = []

        for _ in range(self.maxiter):
            group_cost = np.zeros_like(group_transport)
            iteration_couplings: list[list[np.ndarray]] = []
            iteration_errors: list[float] = []
            for k, source_mass in enumerate(source_weights):
                row: list[np.ndarray] = []
                for l, target_mass in enumerate(target_weights):
                    coupling, cost, error = _log_sinkhorn_weighted(
                        source_mass,
                        target_mass,
                        rotation @ x.T,
                        y.T,
                        self.sample_gamma,
                        self.sinkhorn_maxiter,
                    )
                    row.append(coupling)
                    group_cost[k, l] = cost
                    iteration_errors.append(error)
                iteration_couplings.append(row)

            group_transport = sinkhorn_groups(group_cost, self.group_gamma, self.sinkhorn_maxiter)
            cross = np.zeros((d, d))
            for k in range(a.shape[1]):
                for l in range(b.shape[1]):
                    cross += group_transport[k, l] * (y.T @ iteration_couplings[k][l].T @ x)
            updated_rotation = _closed_form_rotation(2.0 * cross, self.determinant_sign)
            residuals.append(float(np.linalg.norm(updated_rotation - rotation, ord="fro")))
            objective_history.append(float(np.sum(group_transport * group_cost)))
            marginal_errors.append(float(max(iteration_errors)))
            rotation = updated_rotation
            local_couplings = iteration_couplings
            if residuals[-1] <= self.tol and len(residuals) >= 2:
                break

        self.Rg = rotation
        self.P = group_transport
        self.local_couplings = tuple(tuple(row) for row in local_couplings)
        self.diagnostics = {
            "solver": "global_soft_gcot",
            "rotation_residual": np.asarray(residuals),
            "transport_objective_history": np.asarray(objective_history),
            "max_sinkhorn_marginal_error": np.asarray(marginal_errors),
            "group_transport_row_marginal_error": float(
                np.max(np.abs(group_transport.sum(axis=1) - 1.0 / a.shape[1]))
            ),
            "group_transport_column_marginal_error": float(
                np.max(np.abs(group_transport.sum(axis=0) - 1.0 / b.shape[1]))
            ),
            "rotation_determinant": float(np.linalg.det(rotation)),
            "rotation_orthogonality_error": float(np.linalg.norm(rotation.T @ rotation - np.eye(d), ord="fro")),
            "converged": bool(residuals and residuals[-1] <= self.tol),
        }
        return self

    def transform(self, source: np.ndarray) -> np.ndarray:
        return (self.Rg @ np.asarray(source, dtype=float).T).T
