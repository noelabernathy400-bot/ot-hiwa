"""Detached adapter between learnable representations and the NumPy Soft-GCOT solver.

The current SoftHiWA implementation is deliberately a NumPy/SciPy inner
solver.  This module makes that boundary explicit: an outer representation
learner may pass tensors, but the transport plans are solved without gradients
and returned as immutable numerical evidence for the following outer update.

Keeping this adapter small is intentional.  It gives the representation MVP a
single, regression-tested entry point without changing the frozen Soft-GCOT
baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

try:
    from .soft_hiwa import SoftHiWA
    from .transport_consistent_joint import global_coupling
except ImportError:  # Supports direct execution of legacy experiment scripts.
    from soft_hiwa import SoftHiWA
    from transport_consistent_joint import global_coupling


def _to_numpy_detached(values: Any) -> np.ndarray:
    """Convert NumPy arrays or tensor-like inputs without retaining gradients."""
    detached = values.detach() if hasattr(values, "detach") else values
    cpu_values = detached.cpu() if hasattr(detached, "cpu") else detached
    array = np.asarray(cpu_values, dtype=float)
    if array.ndim != 2:
        raise ValueError("source and target representations must be two-dimensional")
    if not np.isfinite(array).all():
        raise ValueError("source and target representations must be finite")
    return array.copy()


@dataclass(frozen=True)
class DetachedSoftGCOTSolution:
    """Frozen output of one full-support Soft-GCOT inner solve.

    ``sample_coupling`` is the normalized global plan
    ``Pi = sum_kl P_kl Q_kl``.  The row-vector convention used by
    :meth:`SoftHiWA.transform` is ``aligned_source = source @ rotation.T``.
    """

    source_assignments: np.ndarray
    target_assignments: np.ndarray
    group_transport: np.ndarray
    rotation: np.ndarray
    sample_coupling: np.ndarray
    aligned_source: np.ndarray
    diagnostics: Mapping[str, Any]


def solve_soft_gcot_detached(
    source_representation: Any,
    source_assignments: Any,
    target_representation: Any,
    target_assignments: Any,
    *,
    solver_kwargs: Mapping[str, Any] | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
) -> DetachedSoftGCOTSolution:
    """Run the unchanged full-support solver on detached representations.

    The function intentionally rejects sparse support: variable-size local
    couplings do not define the dense global plan needed by the outer
    representation update.  The caller is responsible for learning
    assignments; labels and pair IDs are never accepted here.
    """
    source = _to_numpy_detached(source_representation)
    target = _to_numpy_detached(target_representation)
    source_soft = _to_numpy_detached(source_assignments)
    target_soft = _to_numpy_detached(target_assignments)
    if source.shape[0] != source_soft.shape[0] or target.shape[0] != target_soft.shape[0]:
        raise ValueError("assignment row counts must match representation sample counts")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target representations must share a latent dimension")

    parameters = dict(solver_kwargs or {})
    if parameters.get("support_mode", "full") != "full":
        raise ValueError("detached representation updates require support_mode='full'")
    parameters["support_mode"] = "full"
    model = SoftHiWA(**parameters).fit(
        source,
        source_soft,
        target,
        target_soft,
        **dict(fit_kwargs or {}),
    )
    sample_coupling = global_coupling(model.local_couplings, model.P)
    return DetachedSoftGCOTSolution(
        source_assignments=source_soft,
        target_assignments=target_soft,
        group_transport=np.asarray(model.P, dtype=float).copy(),
        rotation=np.asarray(model.Rg, dtype=float).copy(),
        sample_coupling=sample_coupling,
        aligned_source=model.transform(source).copy(),
        diagnostics=dict(model.diagnostics),
    )
