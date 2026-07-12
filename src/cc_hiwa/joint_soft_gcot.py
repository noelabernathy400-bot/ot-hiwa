"""Alignment-aware prototype updates for alternating Soft-GCOT HiWA.

This module deliberately does not differentiate through Sinkhorn or ADMM.
Instead, one outer iteration fixes the current Soft-GCOT group transport and
global rotation, updates the two prototype sets, and then recomputes soft
assignments for the next Soft-GCOT iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

try:
    from .soft_groups import assignment_entropy, assignments_from_prototypes
except ImportError:  # Supports direct execution from experiment scripts.
    from soft_groups import assignment_entropy, assignments_from_prototypes


EPS = 1e-12


@dataclass(frozen=True)
class JointSoftGCOTConfig:
    n_groups: int = 4
    temperature: float = 0.50
    lambda_cross: float = 0.10
    lambda_entropy: float = 0.05
    lambda_l2: float = 1e-4
    maxiter: int = 80
    gtol: float = 1e-5


@dataclass
class JointSoftGCOTUpdate:
    source_prototypes_standardized: np.ndarray
    target_prototypes_standardized: np.ndarray
    source_assignments: np.ndarray
    target_assignments: np.ndarray
    initial_loss: dict[str, float]
    final_loss: dict[str, float]
    diagnostics: dict[str, Any]


def standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float)
    mean = array.mean(axis=0, keepdims=True)
    scale = np.maximum(array.std(axis=0, keepdims=True), EPS)
    return (array - mean) / scale, mean, scale


def minimum_prototype_distance(prototypes: np.ndarray) -> float:
    array = np.asarray(prototypes, dtype=float)
    if array.shape[0] < 2:
        return 0.0
    distance = np.linalg.norm(array[:, None, :] - array[None, :, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    return float(np.min(distance))


def _unpack(flat: np.ndarray, n_groups: int, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    split = n_groups * dimension
    return flat[:split].reshape(n_groups, dimension), flat[split:].reshape(n_groups, dimension)


def _validate(
    source: np.ndarray,
    target: np.ndarray,
    rotation: np.ndarray,
    group_transport: np.ndarray,
    source_initial: np.ndarray,
    target_initial: np.ndarray,
    config: JointSoftGCOTConfig,
) -> None:
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("source and target must be two-dimensional arrays with equal feature dimension")
    expected = (config.n_groups, source.shape[1])
    if source_initial.shape != expected or target_initial.shape != expected:
        raise ValueError(f"initial standardized prototypes must each have shape {expected}")
    if rotation.shape != (source.shape[1], source.shape[1]):
        raise ValueError("rotation must match the shared feature dimension")
    if group_transport.shape != (config.n_groups, config.n_groups):
        raise ValueError("group_transport shape must match the number of groups")
    if np.any(group_transport < -EPS) or not np.isfinite(group_transport).all():
        raise ValueError("group_transport must be finite and non-negative")
    if config.temperature <= 0 or config.lambda_cross < 0 or config.lambda_entropy < 0 or config.lambda_l2 < 0:
        raise ValueError("temperature must be positive and loss weights must be non-negative")


def _loss_terms(
    flat: np.ndarray,
    *,
    source_standardized: np.ndarray,
    target_standardized: np.ndarray,
    source_mean: np.ndarray,
    source_scale: np.ndarray,
    target_mean: np.ndarray,
    target_scale: np.ndarray,
    rotation: np.ndarray,
    group_transport: np.ndarray,
    target_entropy: float,
    cross_reference: float,
    config: JointSoftGCOTConfig,
) -> tuple[float, dict[str, float], dict[str, np.ndarray]]:
    source_proto, target_proto = _unpack(flat, config.n_groups, source_standardized.shape[1])
    source_assignments = assignments_from_prototypes(source_standardized, source_proto, config.temperature)
    target_assignments = assignments_from_prototypes(target_standardized, target_proto, config.temperature)

    reconstruction_source = float(np.mean((source_standardized - source_assignments @ source_proto) ** 2))
    reconstruction_target = float(np.mean((target_standardized - target_assignments @ target_proto) ** 2))
    source_original = source_proto * source_scale + source_mean
    target_original = target_proto * target_scale + target_mean
    rotated_source = (rotation @ source_original.T).T
    cross_raw = float(np.sum(group_transport * np.sum((rotated_source[:, None, :] - target_original[None, :, :]) ** 2, axis=2)))
    cross_normalized = cross_raw / max(cross_reference, EPS)
    source_entropy = float(np.mean(assignment_entropy(source_assignments)))
    target_entropy_mean = float(np.mean(assignment_entropy(target_assignments)))
    entropy_penalty = float((source_entropy - target_entropy) ** 2 + (target_entropy_mean - target_entropy) ** 2)
    l2 = float(np.mean(source_proto**2) + np.mean(target_proto**2))
    total = reconstruction_source + reconstruction_target + config.lambda_cross * cross_normalized + config.lambda_entropy * entropy_penalty + config.lambda_l2 * l2
    terms = {
        "total": float(total),
        "reconstruction_source": reconstruction_source,
        "reconstruction_target": reconstruction_target,
        "cross_raw": cross_raw,
        "cross_normalized": cross_normalized,
        "entropy_penalty": entropy_penalty,
        "source_mean_entropy": source_entropy,
        "target_mean_entropy": target_entropy_mean,
        "l2": l2,
        "source_min_mass": float(source_assignments.mean(axis=0).min()),
        "target_min_mass": float(target_assignments.mean(axis=0).min()),
        "source_min_prototype_distance": minimum_prototype_distance(source_original),
        "target_min_prototype_distance": minimum_prototype_distance(target_original),
    }
    arrays = {"source_assignments": source_assignments, "target_assignments": target_assignments}
    return float(total), terms, arrays


def update_prototypes_from_soft_gcot(
    source: np.ndarray,
    target: np.ndarray,
    *,
    rotation: np.ndarray,
    group_transport: np.ndarray,
    initial_source_prototypes_standardized: np.ndarray,
    initial_target_prototypes_standardized: np.ndarray,
    target_entropy: float,
    config: JointSoftGCOTConfig | None = None,
) -> JointSoftGCOTUpdate:
    """Update prototypes while holding the preceding Soft-GCOT ``P`` and ``R`` fixed."""
    cfg = config or JointSoftGCOTConfig()
    source_array = np.asarray(source, dtype=float)
    target_array = np.asarray(target, dtype=float)
    rotation_array = np.asarray(rotation, dtype=float)
    transport_array = np.asarray(group_transport, dtype=float)
    source_initial = np.asarray(initial_source_prototypes_standardized, dtype=float)
    target_initial = np.asarray(initial_target_prototypes_standardized, dtype=float)
    _validate(source_array, target_array, rotation_array, transport_array, source_initial, target_initial, cfg)
    source_standardized, source_mean, source_scale = standardize(source_array)
    target_standardized, target_mean, target_scale = standardize(target_array)
    x0 = np.concatenate([source_initial.ravel(), target_initial.ravel()])
    # The fixed reference makes lambda_cross dimensionless and therefore stable
    # across the two domains' coordinate scales.
    initial_source_original = source_initial * source_scale + source_mean
    initial_target_original = target_initial * target_scale + target_mean
    initial_rotated = (rotation_array @ initial_source_original.T).T
    cross_reference = float(np.sum(transport_array * np.sum((initial_rotated[:, None, :] - initial_target_original[None, :, :]) ** 2, axis=2)))
    initial_total, initial_terms, _ = _loss_terms(
        x0,
        source_standardized=source_standardized,
        target_standardized=target_standardized,
        source_mean=source_mean,
        source_scale=source_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        rotation=rotation_array,
        group_transport=transport_array,
        target_entropy=target_entropy,
        cross_reference=cross_reference,
        config=cfg,
    )

    def objective(flat: np.ndarray) -> float:
        return _loss_terms(
            flat,
            source_standardized=source_standardized,
            target_standardized=target_standardized,
            source_mean=source_mean,
            source_scale=source_scale,
            target_mean=target_mean,
            target_scale=target_scale,
            rotation=rotation_array,
            group_transport=transport_array,
            target_entropy=target_entropy,
            cross_reference=cross_reference,
            config=cfg,
        )[0]

    optimized = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": cfg.maxiter, "gtol": cfg.gtol})
    final_total, final_terms, arrays = _loss_terms(
        optimized.x,
        source_standardized=source_standardized,
        target_standardized=target_standardized,
        source_mean=source_mean,
        source_scale=source_scale,
        target_mean=target_mean,
        target_scale=target_scale,
        rotation=rotation_array,
        group_transport=transport_array,
        target_entropy=target_entropy,
        cross_reference=cross_reference,
        config=cfg,
    )
    source_proto, target_proto = _unpack(optimized.x, cfg.n_groups, source_array.shape[1])
    return JointSoftGCOTUpdate(
        source_prototypes_standardized=source_proto,
        target_prototypes_standardized=target_proto,
        source_assignments=arrays["source_assignments"],
        target_assignments=arrays["target_assignments"],
        initial_loss=initial_terms,
        final_loss=final_terms,
        diagnostics={
            "success": bool(optimized.success),
            "message": str(optimized.message),
            "iterations": int(getattr(optimized, "nit", 0)),
            "function_evaluations": int(getattr(optimized, "nfev", 0)),
            "cross_reference": cross_reference,
            "initial_total": initial_total,
            "final_total": final_total,
            "configuration": vars(cfg),
        },
    )
