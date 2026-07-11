from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans


EPS = 1e-12


@dataclass
class SoftGroupResult:
    prototypes: np.ndarray
    prototypes_standardized: np.ndarray
    assignments: np.ndarray
    group_weights: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    diagnostics: dict[str, float | int | bool | list[float]]


def _standardize(features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=float)
    mean = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    scale = np.where(scale < EPS, 1.0, scale)
    return (values - mean) / scale, mean, scale


def row_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.maximum(exponentiated.sum(axis=1, keepdims=True), EPS)


def assignments_from_prototypes(
    standardized_features: np.ndarray,
    prototypes_standardized: np.ndarray,
    temperature: float,
) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = standardized_features @ prototypes_standardized.T / temperature
    return row_softmax(logits)


def normalized_group_weights(assignments: np.ndarray) -> np.ndarray:
    values = np.asarray(assignments, dtype=float)
    column_mass = values.sum(axis=0, keepdims=True)
    if np.any(column_mass <= EPS):
        raise ValueError("at least one soft group has zero mass")
    return values / column_mass


def assignment_entropy(assignments: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(assignments, dtype=float), EPS, 1.0)
    return -np.sum(values * np.log(values), axis=1)


def _prototype_objective(
    flat_prototypes: np.ndarray,
    standardized_features: np.ndarray,
    n_groups: int,
    temperature: float,
    entropy_weight: float,
    l2_weight: float,
) -> float:
    prototypes = flat_prototypes.reshape(n_groups, standardized_features.shape[1])
    assignments = assignments_from_prototypes(
        standardized_features,
        prototypes,
        temperature,
    )
    reconstruction = assignments @ prototypes
    reconstruction_mse = float(np.mean((reconstruction - standardized_features) ** 2))
    mean_entropy = float(np.mean(assignment_entropy(assignments)))
    l2_penalty = float(np.mean(prototypes**2))
    return reconstruction_mse - entropy_weight * mean_entropy + l2_weight * l2_penalty


def learn_soft_groups(
    features: np.ndarray,
    n_groups: int = 4,
    temperature: float = 1.0,
    entropy_weight: float = 0.05,
    l2_weight: float = 1e-4,
    seed: int = 0,
    maxiter: int = 200,
) -> SoftGroupResult:
    values = np.asarray(features, dtype=float)
    if values.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if not 1 < n_groups <= values.shape[0]:
        raise ValueError("n_groups must be between 2 and the sample count")

    standardized, mean, scale = _standardize(values)
    kmeans = KMeans(n_clusters=n_groups, init="k-means++", n_init=10, random_state=seed)
    kmeans.fit(standardized)
    initial = kmeans.cluster_centers_.astype(float)
    initial_loss = _prototype_objective(
        initial.ravel(),
        standardized,
        n_groups,
        temperature,
        entropy_weight,
        l2_weight,
    )

    optimization = minimize(
        _prototype_objective,
        initial.ravel(),
        args=(standardized, n_groups, temperature, entropy_weight, l2_weight),
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-10, "maxls": 40},
    )
    prototypes_standardized = optimization.x.reshape(n_groups, values.shape[1])
    assignments = assignments_from_prototypes(
        standardized,
        prototypes_standardized,
        temperature,
    )
    group_weights = normalized_group_weights(assignments)
    prototypes = prototypes_standardized * scale + mean
    entropies = assignment_entropy(assignments)
    group_mass = assignments.mean(axis=0)
    pairwise = np.linalg.norm(
        prototypes_standardized[:, None, :] - prototypes_standardized[None, :, :],
        axis=2,
    )
    nonzero_pairwise = pairwise[np.triu_indices(n_groups, k=1)]

    diagnostics: dict[str, float | int | bool | list[float]] = {
        "success": bool(optimization.success),
        "status": int(optimization.status),
        "iterations": int(optimization.nit),
        "initial_loss": float(initial_loss),
        "final_loss": float(optimization.fun),
        "mean_entropy": float(entropies.mean()),
        "normalized_mean_entropy": float(entropies.mean() / np.log(n_groups)),
        "min_group_mass": float(group_mass.min()),
        "max_group_mass": float(group_mass.max()),
        "group_mass": group_mass.tolist(),
        "min_prototype_distance": float(nonzero_pairwise.min()),
        "max_row_sum_error": float(np.max(np.abs(assignments.sum(axis=1) - 1.0))),
    }
    return SoftGroupResult(
        prototypes=prototypes,
        prototypes_standardized=prototypes_standardized,
        assignments=assignments,
        group_weights=group_weights,
        feature_mean=mean.ravel(),
        feature_scale=scale.ravel(),
        diagnostics=diagnostics,
    )


def build_sparse_group_supports(
    assignments: np.ndarray,
    retain_mass: float = 0.90,
    max_support_factor: float = 1.5,
    min_support: int = 4,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """Keep the highest-membership samples for each group and renormalize.

    The support is the smallest prefix reaching ``retain_mass``, capped at
    ``max_support_factor * n_samples / n_groups`` for tractability.
    """
    values = np.asarray(assignments, dtype=float)
    if not 0 < retain_mass <= 1:
        raise ValueError("retain_mass must be in (0, 1]")
    n_samples, n_groups = values.shape
    max_support = max(min_support, int(np.ceil(max_support_factor * n_samples / n_groups)))
    indices_by_group: list[np.ndarray] = []
    weights_by_group: list[np.ndarray] = []
    retained_by_group: list[float] = []

    for group in range(n_groups):
        full_weights = values[:, group] / np.maximum(values[:, group].sum(), EPS)
        order = np.argsort(full_weights)[::-1]
        cumulative = np.cumsum(full_weights[order])
        required = int(np.searchsorted(cumulative, retain_mass, side="left") + 1)
        support_size = min(max(required, min_support), max_support, n_samples)
        selected = order[:support_size]
        selected_weights = full_weights[selected]
        retained = float(selected_weights.sum())
        selected_weights = selected_weights / np.maximum(retained, EPS)
        indices_by_group.append(selected)
        weights_by_group.append(selected_weights)
        retained_by_group.append(retained)

    return indices_by_group, weights_by_group, retained_by_group
