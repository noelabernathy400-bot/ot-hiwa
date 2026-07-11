"""Dataset-agnostic Component-Conditioned HiWA core utilities.

The functions in this module implement the method backbone shared by the
planned three-dataset study:

1. soft component representatives;
2. group-level entropic OT;
3. sample compatibility S = A P B^T;
4. component-conditioned sample cost;
5. sample-level entropic OT.

This module intentionally does not know about neural, single-cell, or
image-text data. Dataset adapters should convert their data into X, Y, A, B and
then call this core.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class CCHiWAResult:
    """Container for one CC-HiWA forward pass."""

    source_representatives: np.ndarray
    target_representatives: np.ndarray
    group_cost: np.ndarray
    group_transport: np.ndarray
    compatibility: np.ndarray
    base_sample_cost: np.ndarray
    conditioned_sample_cost: np.ndarray
    sample_transport: np.ndarray
    objective: float


def validate_inputs(
    source: np.ndarray,
    target: np.ndarray,
    source_assignments: np.ndarray,
    target_assignments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and return float arrays for CC-HiWA inputs."""
    x = np.asarray(source, dtype=float)
    y = np.asarray(target, dtype=float)
    a = np.asarray(source_assignments, dtype=float)
    b = np.asarray(target_assignments, dtype=float)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("source and target must be 2D arrays")
    if x.shape[1] != y.shape[1]:
        raise ValueError("source and target must have the same embedding dimension")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("assignment matrices must be 2D arrays")
    if a.shape[0] != x.shape[0] or b.shape[0] != y.shape[0]:
        raise ValueError("assignment row counts must match sample counts")
    if np.any(a < -1e-12) or np.any(b < -1e-12):
        raise ValueError("assignments must be non-negative")
    if np.max(np.abs(a.sum(axis=1) - 1.0)) > 1e-6:
        raise ValueError("source assignment rows must sum to one")
    if np.max(np.abs(b.sum(axis=1) - 1.0)) > 1e-6:
        raise ValueError("target assignment rows must sum to one")
    return x, y, a, b


def soft_representatives(values: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    """Return one weighted representative per soft component."""
    v = np.asarray(values, dtype=float)
    a = np.asarray(assignments, dtype=float)
    masses = np.maximum(a.sum(axis=0), EPS)
    return (a.T @ v) / masses[:, None]


def squared_euclidean_cost(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean cost matrix."""
    x = np.asarray(source, dtype=float)
    y = np.asarray(target, dtype=float)
    x2 = np.sum(x**2, axis=1)
    y2 = np.sum(y**2, axis=1)
    cost = x2[:, None] + y2[None, :] - 2.0 * (x @ y.T)
    return np.maximum(cost, 0.0)


def sinkhorn(
    cost: np.ndarray,
    source_mass: np.ndarray | None = None,
    target_mass: np.ndarray | None = None,
    gamma: float = 0.1,
    maxiter: int = 300,
) -> np.ndarray:
    """Entropic OT coupling for an arbitrary rectangular cost matrix."""
    c = np.asarray(cost, dtype=float)
    if c.ndim != 2:
        raise ValueError("cost must be a 2D matrix")
    n_source, n_target = c.shape
    if source_mass is None:
        p = np.full(n_source, 1.0 / n_source)
    else:
        p = np.asarray(source_mass, dtype=float).ravel()
        p = p / np.maximum(p.sum(), EPS)
    if target_mass is None:
        q = np.full(n_target, 1.0 / n_target)
    else:
        q = np.asarray(target_mass, dtype=float).ravel()
        q = q / np.maximum(q.sum(), EPS)
    if p.shape != (n_source,) or q.shape != (n_target,):
        raise ValueError("mass vectors must match cost dimensions")
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("mass vectors must be non-negative")
    safe_gamma = max(float(gamma), 1e-8)
    kernel = np.exp(-(c - np.min(c)) / safe_gamma)
    kernel = np.maximum(kernel, 1e-300)
    v = np.ones_like(q)
    for _ in range(maxiter):
        u = p / np.maximum(kernel @ v, EPS)
        v = q / np.maximum(kernel.T @ u, EPS)
    return (u[:, None] * kernel) * v[None, :]


def compatibility_from_group_transport(
    source_assignments: np.ndarray,
    group_transport: np.ndarray,
    target_assignments: np.ndarray,
) -> np.ndarray:
    """Return sample compatibility S = A P B^T."""
    a = np.asarray(source_assignments, dtype=float)
    p = np.asarray(group_transport, dtype=float)
    b = np.asarray(target_assignments, dtype=float)
    if a.shape[1] != p.shape[0] or b.shape[1] != p.shape[1]:
        raise ValueError("assignment component counts must match group transport")
    return np.maximum(a @ p @ b.T, 0.0)


def component_conditioned_cost(
    base_cost: np.ndarray,
    compatibility: np.ndarray,
    beta: float,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Combine geometric/sample cost with component compatibility."""
    if beta < 0:
        raise ValueError("beta must be non-negative")
    base = np.asarray(base_cost, dtype=float)
    comp = np.asarray(compatibility, dtype=float)
    if base.shape != comp.shape:
        raise ValueError("base_cost and compatibility must have the same shape")
    if beta == 0:
        return base.copy()
    penalty = -float(beta) * np.log(np.maximum(comp, epsilon))
    return base + penalty


def barycentric_projection(transport: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Map each source sample to the barycenter of target samples under transport."""
    t = np.asarray(transport, dtype=float)
    y = np.asarray(target, dtype=float)
    if t.ndim != 2 or y.ndim != 2 or t.shape[1] != y.shape[0]:
        raise ValueError("transport columns must match target rows")
    row_mass = np.maximum(t.sum(axis=1, keepdims=True), EPS)
    return (t @ y) / row_mass


def coupling_entropy(transport: np.ndarray) -> float:
    """Shannon entropy of a non-negative coupling matrix."""
    values = np.asarray(transport, dtype=float)
    positive = values[values > 0]
    if positive.size == 0:
        return 0.0
    return float(-np.sum(positive * np.log(positive)))


def fit_cc_hiwa(
    source: np.ndarray,
    target: np.ndarray,
    source_assignments: np.ndarray,
    target_assignments: np.ndarray,
    *,
    rotation: np.ndarray | None = None,
    beta: float = 0.25,
    group_gamma: float = 0.1,
    sample_gamma: float = 0.1,
    sinkhorn_maxiter: int = 300,
) -> CCHiWAResult:
    """Run one non-iterative CC-HiWA core pass.

    This is the minimal general backbone. It assumes embeddings are already in a
    comparable space or an optional orthogonal/linear `rotation` is supplied.
    """
    x, y, a, b = validate_inputs(source, target, source_assignments, target_assignments)
    if rotation is None:
        x_aligned = x
    else:
        r = np.asarray(rotation, dtype=float)
        if r.shape != (x.shape[1], x.shape[1]):
            raise ValueError("rotation must be square with the embedding dimension")
        x_aligned = (r @ x.T).T
    source_reps = soft_representatives(x_aligned, a)
    target_reps = soft_representatives(y, b)
    group_cost = squared_euclidean_cost(source_reps, target_reps)
    group_transport = sinkhorn(
        group_cost,
        gamma=group_gamma,
        maxiter=sinkhorn_maxiter,
    )
    compatibility = compatibility_from_group_transport(a, group_transport, b)
    base_cost = squared_euclidean_cost(x_aligned, y)
    conditioned_cost = component_conditioned_cost(base_cost, compatibility, beta=beta)
    sample_transport = sinkhorn(
        conditioned_cost,
        gamma=sample_gamma,
        maxiter=sinkhorn_maxiter,
    )
    return CCHiWAResult(
        source_representatives=source_reps,
        target_representatives=target_reps,
        group_cost=group_cost,
        group_transport=group_transport,
        compatibility=compatibility,
        base_sample_cost=base_cost,
        conditioned_sample_cost=conditioned_cost,
        sample_transport=sample_transport,
        objective=float(np.sum(sample_transport * conditioned_cost)),
    )
