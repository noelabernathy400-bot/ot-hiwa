"""Source-labelled / target-unlabelled class-conditional Soft-GCOT.

The module deliberately accepts source labels and *target prediction logits*,
but it never accepts target labels or synchronized pair IDs.  Source labels
anchor the source-side groups to task semantics; target-side group membership
is a soft pseudo-distribution produced by a source-trained task head.

This is a new transport contract, not a modification of the frozen
prototype-based Soft-GCOT baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

try:
    from .solver_adapter import DetachedSoftGCOTSolution, solve_soft_gcot_detached
except ImportError:  # Supports direct execution of experiment scripts.
    from solver_adapter import DetachedSoftGCOTSolution, solve_soft_gcot_detached


@dataclass(frozen=True)
class ClassConditionalSoftGCOTSolution:
    """Detached transport solution with its semantic group assignments."""

    transport: DetachedSoftGCOTSolution
    source_assignments: np.ndarray
    target_assignments: np.ndarray
    target_prediction_entropy: float
    target_predicted_group_mass: np.ndarray


def source_label_assignments(
    source_labels: np.ndarray,
    *,
    n_classes: int,
    smoothing: float = 0.02,
) -> np.ndarray:
    """Turn zero-based source labels into full-support semantic assignments."""
    labels = np.asarray(source_labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("source_labels must be a non-empty one-dimensional vector")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("source_labels must contain integer class indices")
    if n_classes < 2 or labels.min() < 0 or labels.max() >= n_classes:
        raise ValueError("source label indices must lie in [0, n_classes)")
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("smoothing must lie in [0, 1)")
    base = smoothing / (n_classes - 1)
    assignments = np.full((labels.size, n_classes), base, dtype=float)
    assignments[np.arange(labels.size), labels.astype(int)] = 1.0 - smoothing
    return assignments


def target_prediction_assignments(
    target_logits: np.ndarray,
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    """Convert task-head logits into target soft class memberships."""
    logits = np.asarray(target_logits, dtype=float)
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] < 2:
        raise ValueError("target_logits must have shape (samples, at least two classes)")
    if not np.isfinite(logits).all() or temperature <= 0:
        raise ValueError("target_logits must be finite and temperature must be positive")
    scaled = logits / float(temperature)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(scaled)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def solve_class_conditional_soft_gcot_detached(
    source_representation: Any,
    source_labels: np.ndarray,
    target_representation: Any,
    target_logits: np.ndarray,
    *,
    n_classes: int,
    source_smoothing: float = 0.02,
    target_temperature: float = 1.0,
    lock_group_matching: bool = True,
    solver_kwargs: Mapping[str, Any] | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
) -> ClassConditionalSoftGCOTSolution:
    """Solve full-support Soft-GCOT with source semantics and target pseudo-groups.

    Target labels and pair identifiers are intentionally absent from the API.
    The returned plan is detached numerical evidence, consistent with the
    existing representation-learning inner solver.
    """
    source_assignments = source_label_assignments(
        source_labels,
        n_classes=n_classes,
        smoothing=source_smoothing,
    )
    target_assignments = target_prediction_assignments(
        target_logits,
        temperature=target_temperature,
    )
    if target_assignments.shape[1] != n_classes:
        raise ValueError("target logit columns must equal n_classes")
    parameters = dict(fit_kwargs or {})
    if lock_group_matching:
        locked_transport = np.eye(n_classes, dtype=float) / n_classes
        supplied = parameters.get("fixed_group_transport")
        if supplied is not None and not np.allclose(supplied, locked_transport):
            raise ValueError("class-conditional transport uses the semantic diagonal group matching")
        parameters["fixed_group_transport"] = locked_transport
    transport = solve_soft_gcot_detached(
        source_representation,
        source_assignments,
        target_representation,
        target_assignments,
        solver_kwargs=solver_kwargs,
        fit_kwargs=parameters,
    )
    entropy = -np.sum(target_assignments * np.log(np.maximum(target_assignments, 1e-12)), axis=1)
    return ClassConditionalSoftGCOTSolution(
        transport=transport,
        source_assignments=source_assignments,
        target_assignments=target_assignments,
        target_prediction_entropy=float(np.mean(entropy)),
        target_predicted_group_mass=target_assignments.mean(axis=0),
    )
