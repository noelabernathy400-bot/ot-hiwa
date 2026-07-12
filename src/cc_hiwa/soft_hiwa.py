from __future__ import annotations

import numpy as np
from scipy.linalg import orth, sqrtm
from sklearn.decomposition import PCA

try:
    from .soft_groups import build_sparse_group_supports
except ImportError:  # Supports direct execution of legacy experiment scripts.
    from soft_groups import build_sparse_group_supports


EPS = 1e-12


def _normal(values: np.ndarray) -> np.ndarray:
    centered = values - np.mean(values, axis=0, keepdims=True)
    covariance_root = sqrtm(np.cov(centered, rowvar=False))
    covariance_root = np.real_if_close(covariance_root).astype(float)
    return centered @ np.linalg.pinv(covariance_root)


def _closed_form_rotation(
    matrix: np.ndarray,
    determinant_sign: int | None = None,
) -> np.ndarray:
    """Return the closest orthogonal matrix, optionally in a fixed O(d) component."""
    u, _, vh = np.linalg.svd(matrix)
    unconstrained = u @ vh
    if determinant_sign is None:
        return unconstrained
    if determinant_sign not in (-1, 1):
        raise ValueError("determinant_sign must be -1, +1, or None")
    correction = np.eye(unconstrained.shape[0])
    current_sign = 1 if np.linalg.det(unconstrained) >= 0 else -1
    correction[-1, -1] = determinant_sign * current_sign
    return u @ correction @ vh


def sinkhorn_weighted(
    source_mass: np.ndarray,
    target_mass: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    gamma: float,
    maxiter: int,
    extra_cost: np.ndarray | None = None,
    extra_cost_weight: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    p = np.asarray(source_mass, dtype=float).ravel()
    q = np.asarray(target_mass, dtype=float).ravel()
    p = p / p.sum()
    q = q / q.sum()
    x2 = np.sum(source**2, axis=0)
    y2 = np.sum(target**2, axis=0)
    cost = y2[None, :] + x2[:, None] - 2.0 * (source.T @ target)
    cost = np.maximum(cost, 0.0)
    if extra_cost is not None and extra_cost_weight > 0:
        extra = np.asarray(extra_cost, dtype=float)
        if extra.shape != cost.shape:
            raise ValueError("extra_cost shape must match the sample transport cost")
        cost = cost + float(extra_cost_weight) * extra
    safe_gamma = max(float(gamma), 1e-8)
    kernel = np.exp(-(cost - cost.min()) / safe_gamma)
    kernel = np.maximum(kernel, 1e-300)
    v = np.ones_like(q)
    for _ in range(maxiter):
        u = p / np.maximum(kernel @ v, EPS)
        v = q / np.maximum(kernel.T @ u, EPS)
    coupling = (u[:, None] * kernel) * v[None, :]
    marginal_error = max(
        float(np.max(np.abs(coupling.sum(axis=1) - p))),
        float(np.max(np.abs(coupling.sum(axis=0) - q))),
    )
    return coupling, float(np.sum(cost * coupling)), marginal_error


def sinkhorn_groups(cost: np.ndarray, gamma: float, maxiter: int) -> np.ndarray:
    n_source, n_target = cost.shape
    p = np.full(n_source, 1.0 / n_source)
    q = np.full(n_target, 1.0 / n_target)
    kernel = np.exp(-(cost - cost.min()) / max(float(gamma), 1e-8))
    kernel = np.maximum(kernel, 1e-300)
    v = np.ones(n_target)
    for _ in range(maxiter):
        u = p / np.maximum(kernel @ v, EPS)
        v = q / np.maximum(kernel.T @ u, EPS)
    return (u[:, None] * kernel) * v[None, :]


def _soft_representatives(values: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    weights = np.asarray(assignments, dtype=float)
    masses = np.maximum(weights.sum(axis=0), EPS)
    return (weights.T @ np.asarray(values, dtype=float)) / masses[:, None]


def _unit_scale_cost(cost: np.ndarray) -> np.ndarray:
    values = np.asarray(cost, dtype=float)
    spread = float(np.max(values) - np.min(values))
    if spread <= 1e-12:
        return np.zeros_like(values)
    return (values - np.min(values)) / spread


class SoftHiWA:
    """TACO-style soft-group HiWA with an explicit full-support reference mode.

    ``support_mode='full'`` is the faithful GCOT reference: each Q_ij uses all
    samples with the normalized soft assignment column as its marginal.
    ``'sparse'`` preserves the previous top-membership approximation.
    """

    def __init__(
        self,
        dim_red_method=None,
        normalize: bool = True,
        maxiter: int = 60,
        tol: float = 1e-1,
        mu: float = 5e-3,
        shorn_maxiter: int = 300,
        shorn_gamma: float = 2e-1,
        sa_maxiter: int = 40,
        sa_tol: float = 1e-2,
        sa_shorn_maxiter: int = 80,
        sa_shorn_gamma: float = 1e-1,
        retain_mass: float = 0.90,
        max_support_factor: float = 1.5,
        support_mode: str = "full",
        random_state: int = 0,
        warm_start_local: bool = False,
        rotation_anchor_weight: float = 0.0,
        determinant_sign: int | None = None,
        representative_guidance_weight: float = 0.0,
        representative_rotation_weight: float = 0.0,
        component_conditioning_weight: float = 0.0,
        representative_cost_normalization: str = "minmax",
    ) -> None:
        self.dim_red_method = dim_red_method or PCA(n_components=2)
        self.normalize = normalize
        self.maxiter = maxiter
        self.tol = tol
        self.mu = mu
        self.shorn_maxiter = shorn_maxiter
        self.shorn_gamma = shorn_gamma
        self.sa_maxiter = sa_maxiter
        self.sa_tol = sa_tol
        self.sa_shorn_maxiter = sa_shorn_maxiter
        self.sa_shorn_gamma = sa_shorn_gamma
        self.retain_mass = retain_mass
        self.max_support_factor = max_support_factor
        if support_mode not in {"full", "sparse"}:
            raise ValueError("support_mode must be 'full' or 'sparse'")
        self.support_mode = support_mode
        self.random_state = random_state
        self.warm_start_local = warm_start_local
        if rotation_anchor_weight < 0:
            raise ValueError("rotation_anchor_weight must be non-negative")
        self.rotation_anchor_weight = float(rotation_anchor_weight)
        if determinant_sign not in (None, -1, 1):
            raise ValueError("determinant_sign must be -1, +1, or None")
        self.determinant_sign = determinant_sign
        if representative_guidance_weight < 0 or representative_guidance_weight > 1:
            raise ValueError("representative_guidance_weight must be in [0, 1]")
        self.representative_guidance_weight = float(representative_guidance_weight)
        if representative_rotation_weight < 0:
            raise ValueError("representative_rotation_weight must be non-negative")
        self.representative_rotation_weight = float(representative_rotation_weight)
        if component_conditioning_weight < 0:
            raise ValueError("component_conditioning_weight must be non-negative")
        self.component_conditioning_weight = float(component_conditioning_weight)
        if representative_cost_normalization != "minmax":
            raise ValueError("representative_cost_normalization currently supports only 'minmax'")
        self.representative_cost_normalization = representative_cost_normalization

    def fit(
        self,
        source: np.ndarray,
        source_assignments: np.ndarray,
        target: np.ndarray,
        target_assignments: np.ndarray,
        **kwargs,
    ) -> "SoftHiWA":
        x_original = np.asarray(source, dtype=float)
        y_original = np.asarray(target, dtype=float)
        a = np.asarray(source_assignments, dtype=float)
        b = np.asarray(target_assignments, dtype=float)
        if a.shape[0] != x_original.shape[0] or b.shape[0] != y_original.shape[0]:
            raise ValueError("assignment row counts must match sample counts")
        if np.max(np.abs(a.sum(axis=1) - 1.0)) > 1e-6:
            raise ValueError("source assignment rows must sum to one")
        if np.max(np.abs(b.sum(axis=1) - 1.0)) > 1e-6:
            raise ValueError("target assignment rows must sum to one")

        x_fit = _normal(x_original) if self.normalize else x_original.copy()
        y_fit = _normal(y_original) if self.normalize else y_original.copy()
        x_transform = kwargs.get("X_transform")
        if x_transform is None:
            x_transform = np.linalg.pinv(x_fit) @ self.dim_red_method.fit_transform(x_fit)
        y_transform = kwargs.get("Y_transform")
        if y_transform is None:
            y_transform = np.linalg.pinv(y_fit) @ self.dim_red_method.fit_transform(y_fit)
        self.Rgt = kwargs.get("Rgt", np.identity(x_fit.shape[1]))

        high_dim = x_fit.shape[1]
        n_groups_x = a.shape[1]
        n_groups_y = b.shape[1]
        x_mbed = (x_transform @ x_transform.T @ x_fit.T).T / np.sqrt(high_dim)
        y_mbed = (y_transform @ y_transform.T @ y_fit.T).T / np.sqrt(high_dim)
        if self.support_mode == "full":
            x_indices = [np.arange(x_mbed.shape[0]) for _ in range(n_groups_x)]
            y_indices = [np.arange(y_mbed.shape[0]) for _ in range(n_groups_y)]
            x_weights = [a[:, group] / np.maximum(a[:, group].sum(), EPS) for group in range(n_groups_x)]
            y_weights = [b[:, group] / np.maximum(b[:, group].sum(), EPS) for group in range(n_groups_y)]
            x_retained = [1.0] * n_groups_x
            y_retained = [1.0] * n_groups_y
        else:
            x_indices, x_weights, x_retained = build_sparse_group_supports(
                a,
                retain_mass=self.retain_mass,
                max_support_factor=self.max_support_factor,
            )
            y_indices, y_weights, y_retained = build_sparse_group_supports(
                b,
                retain_mass=self.retain_mass,
                max_support_factor=self.max_support_factor,
            )

        # Match the legacy HiWA RandomState stream so paired hard/soft runs
        # start from the same global and local rotation draws.
        rng = np.random.RandomState(self.random_state)
        initial_rotation = kwargs.get("initial_rotation")
        if initial_rotation is None:
            global_rotation = _closed_form_rotation(
                rng.random((high_dim, high_dim)),
                self.determinant_sign,
            )
        else:
            global_rotation = np.asarray(initial_rotation, dtype=float).copy()
            if self.determinant_sign is not None:
                global_rotation = _closed_form_rotation(
                    global_rotation,
                    self.determinant_sign,
                )
        rotation_anchor = kwargs.get("rotation_anchor")
        if rotation_anchor is None:
            if self.rotation_anchor_weight > 0:
                raise ValueError(
                    "rotation_anchor is required when rotation_anchor_weight is positive"
                )
        else:
            rotation_anchor = np.asarray(rotation_anchor, dtype=float).copy()
            if rotation_anchor.shape != (high_dim, high_dim):
                raise ValueError("rotation_anchor shape must match the rotation dimension")
        initial_transport = kwargs.get("initial_transport")
        if initial_transport is None:
            group_transport = np.full(
                (n_groups_x, n_groups_y),
                1.0 / (n_groups_x * n_groups_y),
            )
        else:
            group_transport = np.asarray(initial_transport, dtype=float).copy()
            if group_transport.shape != (n_groups_x, n_groups_y):
                raise ValueError("initial_transport shape does not match soft group counts")
        multipliers = np.zeros((high_dim, high_dim, n_groups_x, n_groups_y))
        local_rotations = np.zeros_like(multipliers)
        if self.warm_start_local:
            local_rotations[:] = global_rotation[:, :, None, None]
        else:
            local_rotations[:] = np.identity(high_dim)[:, :, None, None]
        group_cost = np.zeros((n_groups_x, n_groups_y))
        representative_source = _soft_representatives(x_mbed, a)
        representative_target = _soft_representatives(y_mbed, b)
        representative_cost = np.zeros_like(group_cost)
        mixed_group_cost = np.zeros_like(group_cost)
        residuals: list[float] = []
        max_marginal_errors: list[float] = []
        admm_primal_residuals: list[float] = []
        admm_dual_residuals: list[float] = []
        transport_objectives: list[float] = []
        component_cost_means: list[float] = []
        component_cost_maxima: list[float] = []

        for _ in range(self.maxiter):
            iteration_error = 0.0
            iteration_component_cost_means: list[float] = []
            iteration_component_cost_maxima: list[float] = []
            for i in range(n_groups_x):
                x_i = x_mbed[x_indices[i], :]
                for j in range(n_groups_y):
                    y_j = y_mbed[y_indices[j], :]
                    component_cost = None
                    if self.component_conditioning_weight > 0:
                        compatibility = (
                            a[x_indices[i], :]
                            @ group_transport
                            @ b[y_indices[j], :].T
                        )
                        component_cost = _unit_scale_cost(
                            -np.log(np.maximum(compatibility, EPS))
                        )
                        iteration_component_cost_means.append(float(np.mean(component_cost)))
                        iteration_component_cost_maxima.append(float(np.max(component_cost)))
                    consensus = (self.mu / high_dim) * (
                        global_rotation - multipliers[:, :, i, j]
                    )
                    local_rotations[:, :, i, j], group_cost[i, j], marginal_error = (
                        self._weighted_subspace_alignment(
                            x_i,
                            y_j,
                            x_weights[i],
                            y_weights[j],
                            group_transport[i, j],
                            consensus,
                            rng,
                            local_rotations[:, :, i, j] if self.warm_start_local else None,
                            component_cost,
                        )
                    )
                    iteration_error = max(iteration_error, marginal_error)

            if self.representative_guidance_weight > 0:
                rotated_representatives = (global_rotation @ representative_source.T).T
                diff = (
                    rotated_representatives[:, None, :]
                    - representative_target[None, :, :]
                )
                representative_cost = np.sum(diff**2, axis=2)
                local_scale = float(np.max(group_cost) - np.min(group_cost))
                if local_scale <= 1e-12:
                    local_scale = float(np.mean(np.abs(group_cost)))
                if local_scale <= 1e-12:
                    local_scale = 1.0
                mixed_group_cost = (
                    group_cost
                    + self.representative_guidance_weight
                    * local_scale
                    * _unit_scale_cost(representative_cost)
                )
            else:
                representative_cost = np.zeros_like(group_cost)
                mixed_group_cost = group_cost

            group_transport = sinkhorn_groups(
                mixed_group_cost,
                self.shorn_gamma,
                self.shorn_maxiter,
            )
            previous = global_rotation.copy()
            consensus_mean = np.mean(
                np.reshape(
                    local_rotations + multipliers,
                    (high_dim, high_dim, n_groups_x * n_groups_y),
                    order="F",
                ),
                axis=2,
            )
            if self.rotation_anchor_weight > 0:
                consensus_mean = (
                    consensus_mean
                    + self.rotation_anchor_weight * rotation_anchor
                )
            representative_rotation_cross = (
                representative_target.T @ group_transport.T @ representative_source
            )
            if self.representative_rotation_weight > 0:
                consensus_mean = (
                    consensus_mean
                    + self.representative_rotation_weight * representative_rotation_cross
                )
            global_rotation = _closed_form_rotation(
                consensus_mean,
                self.determinant_sign,
            )
            multipliers = (
                multipliers
                + local_rotations
                - global_rotation[:, :, None, None]
            )
            residual = float(np.linalg.norm(previous - global_rotation, "fro"))
            primal_residual = float(
                np.max(
                    np.linalg.norm(
                        local_rotations - global_rotation[:, :, None, None],
                        axis=(0, 1),
                    )
                )
            )
            residuals.append(residual)
            max_marginal_errors.append(iteration_error)
            admm_primal_residuals.append(primal_residual)
            admm_dual_residuals.append(
                (self.mu / np.sqrt(high_dim))
                * np.sqrt(n_groups_x * n_groups_y)
                * residual
            )
            transport_objectives.append(float(np.sum(group_transport * mixed_group_cost)))
            if iteration_component_cost_means:
                component_cost_means.append(float(np.mean(iteration_component_cost_means)))
                component_cost_maxima.append(float(np.max(iteration_component_cost_maxima)))
            if residual <= self.tol and primal_residual <= self.tol and len(residuals) >= 6:
                break

        local_global_distances = np.linalg.norm(
            local_rotations - global_rotation[:, :, None, None],
            axis=(0, 1),
        )
        self.Rg = global_rotation
        self.P = group_transport
        self.diagnostics = {
            "Rg_norm": np.asarray(residuals),
            "admm_primal_residual": np.asarray(admm_primal_residuals),
            "admm_dual_residual": np.asarray(admm_dual_residuals),
            "transport_objective_history": np.asarray(transport_objectives),
            "C": group_cost,
            "representative_cost": representative_cost,
            "mixed_group_cost": mixed_group_cost,
            "max_sinkhorn_marginal_error": np.asarray(max_marginal_errors),
            "source_support_sizes": [int(len(item)) for item in x_indices],
            "target_support_sizes": [int(len(item)) for item in y_indices],
            "source_retained_mass": x_retained,
            "target_retained_mass": y_retained,
            "support_mode": self.support_mode,
            "sparse_approximation": self.support_mode == "sparse",
            "source_soft_group_mass": (a.sum(axis=0) / a.sum()).tolist(),
            "target_soft_group_mass": (b.sum(axis=0) / b.sum()).tolist(),
            "group_transport_row_marginal_error": float(
                np.max(np.abs(group_transport.sum(axis=1) - 1.0 / n_groups_x))
            ),
            "group_transport_column_marginal_error": float(
                np.max(np.abs(group_transport.sum(axis=0) - 1.0 / n_groups_y))
            ),
            "rotation_anchor_weight": self.rotation_anchor_weight,
            "rotation_anchor_distance": (
                float(np.linalg.norm(global_rotation - rotation_anchor, "fro"))
                if rotation_anchor is not None
                else None
            ),
            "determinant_sign_constraint": self.determinant_sign,
            "representative_guidance_weight": self.representative_guidance_weight,
            "representative_rotation_weight": self.representative_rotation_weight,
            "representative_rotation_cross_norm": float(
                np.linalg.norm(representative_rotation_cross, "fro")
            ),
            "component_conditioning_weight": self.component_conditioning_weight,
            "component_conditioning_cost_mean": (
                float(np.mean(component_cost_means)) if component_cost_means else 0.0
            ),
            "component_conditioning_cost_max": (
                float(np.max(component_cost_maxima)) if component_cost_maxima else 0.0
            ),
            "representative_cost_normalization": self.representative_cost_normalization,
            "representative_cost_mean": float(np.mean(representative_cost)),
            "representative_cost_max": float(np.max(representative_cost)),
            "rotation_determinant": float(np.linalg.det(global_rotation)),
            "rotation_orthogonality_error": float(
                np.linalg.norm(global_rotation.T @ global_rotation - np.eye(high_dim), "fro")
            ),
            "transport_objective": float(np.sum(group_transport * group_cost)),
            "guided_transport_objective": float(np.sum(group_transport * mixed_group_cost)),
            "admm_converged": bool(
                residuals
                and residuals[-1] <= self.tol
                and admm_primal_residuals[-1] <= self.tol
            ),
            "local_global_consensus_mean": float(np.mean(local_global_distances)),
            "local_global_consensus_max": float(np.max(local_global_distances)),
            "local_global_consensus_weighted_rms": float(
                np.sqrt(np.sum(group_transport * local_global_distances**2))
            ),
        }
        return self

    def _weighted_subspace_alignment(
        self,
        source: np.ndarray,
        target: np.ndarray,
        source_mass: np.ndarray,
        target_mass: np.ndarray,
        group_weight: float,
        consensus: np.ndarray,
        rng: np.random.RandomState,
        initial_rotation: np.ndarray | None,
        component_cost: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, float]:
        high_dim = source.shape[1]
        if initial_rotation is None:
            if self.determinant_sign is None:
                rotation = orth(rng.random((high_dim, high_dim)))
            else:
                rotation = _closed_form_rotation(
                    rng.random((high_dim, high_dim)),
                    self.determinant_sign,
                )
        else:
            rotation = np.asarray(initial_rotation, dtype=float).copy()
            if self.determinant_sign is not None:
                rotation = _closed_form_rotation(rotation, self.determinant_sign)
        coupling = np.outer(source_mass, target_mass)
        distance = np.inf
        marginal_error = np.inf
        for _ in range(self.sa_maxiter):
            previous = rotation.copy()
            rotation = _closed_form_rotation(
                2.0 * group_weight * (target.T @ coupling.T @ source) + consensus,
                self.determinant_sign,
            )
            coupling, distance, marginal_error = sinkhorn_weighted(
                source_mass,
                target_mass,
                rotation @ source.T,
                target.T,
                self.sa_shorn_gamma / max(group_weight, 1e-8),
                self.sa_shorn_maxiter,
                extra_cost=component_cost,
                extra_cost_weight=self.component_conditioning_weight,
            )
            if np.linalg.norm(previous - rotation, 2) <= self.sa_tol:
                break
        return rotation, float(distance), float(marginal_error)

    def transform(self, source: np.ndarray) -> np.ndarray:
        # Deliberately matches the legacy HiWA demo for a controlled comparison.
        return (self.Rg @ np.asarray(source).T).T

    def fit_transform(
        self,
        source: np.ndarray,
        source_assignments: np.ndarray,
        target: np.ndarray,
        target_assignments: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        self.fit(source, source_assignments, target, target_assignments, **kwargs)
        return self.transform(source)
