"""Prototype updates constrained by a fixed sample-level OT coupling.

This is a deliberately small alternating-optimization block.  It holds the
Soft-GCOT transport plans and rotations fixed, then updates prototypes through
reconstruction, assignment regularization, and a transport-consistency loss.
It does not differentiate through Sinkhorn or ADMM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

try:
    from .joint_soft_gcot import EPS, minimum_prototype_distance, standardize
    from .soft_groups import assignment_entropy, assignments_from_prototypes
except ImportError:  # Direct execution from experiment scripts.
    from joint_soft_gcot import EPS, minimum_prototype_distance, standardize
    from soft_groups import assignment_entropy, assignments_from_prototypes


@dataclass(frozen=True)
class TransportConsistentConfig:
    n_groups: int = 4
    temperature: float = 0.50
    lambda_tc: float = 0.10
    lambda_entropy: float = 0.05
    lambda_l2: float = 1e-4
    maxiter: int = 80
    gtol: float = 1e-5


@dataclass
class TransportConsistentUpdate:
    source_prototypes_standardized: np.ndarray
    target_prototypes_standardized: np.ndarray
    source_assignments: np.ndarray
    target_assignments: np.ndarray
    initial_loss: dict[str, float]
    final_loss: dict[str, float]
    diagnostics: dict[str, Any]


def global_coupling(local_couplings: tuple[tuple[np.ndarray, ...], ...], group_transport: np.ndarray) -> np.ndarray:
    r"""Return \(\Pi=\sum_{kl}P_{kl}Q_{kl}\) for full-support local couplings."""
    transport = np.asarray(group_transport, dtype=float)
    if transport.ndim != 2 or len(local_couplings) != transport.shape[0]:
        raise ValueError("local coupling and group transport shapes are incompatible")
    first = np.asarray(local_couplings[0][0], dtype=float)
    coupling = np.zeros_like(first)
    for k in range(transport.shape[0]):
        if len(local_couplings[k]) != transport.shape[1]:
            raise ValueError("local coupling and group transport shapes are incompatible")
        for l in range(transport.shape[1]):
            plan = np.asarray(local_couplings[k][l], dtype=float)
            if plan.shape != coupling.shape:
                raise ValueError("transport-consistent updates require full-support local couplings")
            coupling += transport[k, l] * plan
    total = float(coupling.sum())
    if not np.isfinite(coupling).all() or np.any(coupling < -EPS) or total <= EPS:
        raise ValueError("global sample coupling must be finite, non-negative, and non-empty")
    return coupling / total


def group_conditional_map(group_transport: np.ndarray) -> np.ndarray:
    """Map target membership vectors to the source-group coordinate system."""
    transport = np.asarray(group_transport, dtype=float)
    if transport.ndim != 2 or np.any(transport < -EPS):
        raise ValueError("group_transport must be a non-negative matrix")
    return transport / np.maximum(transport.sum(axis=1, keepdims=True), EPS)


def _unpack(flat: np.ndarray, n_groups: int, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    split = n_groups * dimension
    return flat[:split].reshape(n_groups, dimension), flat[split:].reshape(n_groups, dimension)


def transport_consistency_loss(source_assignments: np.ndarray, target_assignments: np.ndarray, coupling: np.ndarray, group_map: np.ndarray) -> float:
    r"""Efficiently compute \(\sum_{ij}\Pi_{ij}\|S_i-T_jM^\top\|^2\)."""
    source = np.asarray(source_assignments, dtype=float)
    target = np.asarray(target_assignments, dtype=float)
    pi = np.asarray(coupling, dtype=float)
    mapped_target = target @ np.asarray(group_map, dtype=float).T
    if pi.shape != (source.shape[0], target.shape[0]) or source.shape[1] != mapped_target.shape[1]:
        raise ValueError("assignment, coupling, and group-map shapes are incompatible")
    row_mass = pi.sum(axis=1)
    col_mass = pi.sum(axis=0)
    squared_source = np.sum(source**2, axis=1)
    squared_target = np.sum(mapped_target**2, axis=1)
    cross = np.sum(source * (pi @ mapped_target))
    return float(row_mass @ squared_source + col_mass @ squared_target - 2.0 * cross)


def update_transport_consistent_prototypes(
    source: np.ndarray,
    target: np.ndarray,
    *,
    local_couplings: tuple[tuple[np.ndarray, ...], ...],
    group_transport: np.ndarray,
    initial_source_prototypes_standardized: np.ndarray,
    initial_target_prototypes_standardized: np.ndarray,
    target_entropy: float,
    config: TransportConsistentConfig | None = None,
) -> TransportConsistentUpdate:
    """Update prototypes using a fixed full-support sample coupling and group map."""
    cfg = config or TransportConsistentConfig()
    source_array, target_array = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if source_array.ndim != 2 or target_array.ndim != 2 or source_array.shape[1] != target_array.shape[1]:
        raise ValueError("source and target must have equal two-dimensional feature shapes")
    expected = (cfg.n_groups, source_array.shape[1])
    if np.asarray(initial_source_prototypes_standardized).shape != expected or np.asarray(initial_target_prototypes_standardized).shape != expected:
        raise ValueError(f"initial prototypes must have shape {expected}")
    if cfg.temperature <= 0 or min(cfg.lambda_tc, cfg.lambda_entropy, cfg.lambda_l2) < 0:
        raise ValueError("temperature must be positive and loss weights non-negative")
    coupling = global_coupling(local_couplings, group_transport)
    group_map = group_conditional_map(group_transport)
    source_standardized, _, _ = standardize(source_array)
    target_standardized, _, _ = standardize(target_array)
    x0 = np.concatenate([np.asarray(initial_source_prototypes_standardized).ravel(), np.asarray(initial_target_prototypes_standardized).ravel()])

    def terms(flat: np.ndarray, tc_reference: float) -> tuple[float, dict[str, float], dict[str, np.ndarray]]:
        source_proto, target_proto = _unpack(flat, cfg.n_groups, source_array.shape[1])
        source_assignment = assignments_from_prototypes(source_standardized, source_proto, cfg.temperature)
        target_assignment = assignments_from_prototypes(target_standardized, target_proto, cfg.temperature)
        reconstruction_source = float(np.mean((source_standardized - source_assignment @ source_proto) ** 2))
        reconstruction_target = float(np.mean((target_standardized - target_assignment @ target_proto) ** 2))
        tc_raw = transport_consistency_loss(source_assignment, target_assignment, coupling, group_map)
        source_entropy = float(np.mean(assignment_entropy(source_assignment)))
        target_entropy_mean = float(np.mean(assignment_entropy(target_assignment)))
        entropy_penalty = float((source_entropy - target_entropy) ** 2 + (target_entropy_mean - target_entropy) ** 2)
        l2 = float(np.mean(source_proto**2) + np.mean(target_proto**2))
        total = reconstruction_source + reconstruction_target + cfg.lambda_tc * tc_raw / max(tc_reference, EPS) + cfg.lambda_entropy * entropy_penalty + cfg.lambda_l2 * l2
        return float(total), {
            "total": float(total), "reconstruction_source": reconstruction_source,
            "reconstruction_target": reconstruction_target, "transport_consistency_raw": tc_raw,
            "transport_consistency_normalized": tc_raw / max(tc_reference, EPS),
            "entropy_penalty": entropy_penalty, "source_mean_entropy": source_entropy,
            "target_mean_entropy": target_entropy_mean, "l2": l2,
            "source_min_mass": float(source_assignment.mean(axis=0).min()),
            "target_min_mass": float(target_assignment.mean(axis=0).min()),
            "source_min_prototype_distance": minimum_prototype_distance(source_proto),
            "target_min_prototype_distance": minimum_prototype_distance(target_proto),
        }, {"source_assignments": source_assignment, "target_assignments": target_assignment}

    initial_total, initial_terms, _ = terms(x0, 1.0)
    tc_reference = max(initial_terms["transport_consistency_raw"], EPS)
    initial_total, initial_terms, _ = terms(x0, tc_reference)
    optimized = minimize(lambda flat: terms(flat, tc_reference)[0], x0, method="L-BFGS-B", options={"maxiter": cfg.maxiter, "gtol": cfg.gtol})
    final_total, final_terms, arrays = terms(optimized.x, tc_reference)
    source_proto, target_proto = _unpack(optimized.x, cfg.n_groups, source_array.shape[1])
    return TransportConsistentUpdate(source_proto, target_proto, arrays["source_assignments"], arrays["target_assignments"], initial_terms, final_terms, {
        "success": bool(optimized.success), "message": str(optimized.message), "iterations": int(getattr(optimized, "nit", 0)),
        "function_evaluations": int(getattr(optimized, "nfev", 0)), "transport_consistency_reference": tc_reference,
        "global_coupling_total_mass": float(coupling.sum()),
        "global_coupling_row_mass_min": float(coupling.sum(axis=1).min()),
        "global_coupling_row_mass_max": float(coupling.sum(axis=1).max()),
        "global_coupling_column_mass_min": float(coupling.sum(axis=0).min()),
        "global_coupling_column_mass_max": float(coupling.sum(axis=0).max()),
        "initial_total": initial_total, "final_total": final_total,
        "configuration": vars(cfg),
    })
