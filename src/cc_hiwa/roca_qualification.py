"""Label-free qualification for determinant-constrained ROCA candidates.

The historical selector averaged the group transports of the two candidates
before deriving a group permutation. A non-converged candidate can therefore
change the permutation used to judge the converged candidate. This module
uses each candidate's own transport, rejects unhealthy candidates, and breaks
a healthy-candidate tie only when the model objective separates them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


EPS = 1e-12


@dataclass(frozen=True)
class RocaCandidateEvidence:
    determinant_sign: int
    group_transport: np.ndarray
    admm_converged: bool
    final_local_marginal_error: float
    transport_objective: float
    relative_global_fit_ratio: float


def _representatives(values: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    features = np.asarray(values, dtype=float)
    memberships = np.asarray(assignments, dtype=float)
    if features.ndim != 2 or memberships.ndim != 2 or features.shape[0] != memberships.shape[0]:
        raise ValueError("features and assignments must have matching two-dimensional sample axes")
    standardized = (features - features.mean(axis=0, keepdims=True)) / np.maximum(
        features.std(axis=0, keepdims=True), EPS
    )
    return (memberships / np.maximum(memberships.sum(axis=0, keepdims=True), EPS)).T @ standardized


def _orientation(source_representatives: np.ndarray, target_representatives: np.ndarray, group_transport: np.ndarray) -> dict[str, Any]:
    source = np.asarray(source_representatives, dtype=float)
    target = np.asarray(target_representatives, dtype=float)
    transport = np.asarray(group_transport, dtype=float)
    if source.shape != target.shape or transport.shape != (source.shape[0], target.shape[0]):
        raise ValueError("representative and group-transport shapes are incompatible")
    if source.shape[0] != source.shape[1] + 1:
        raise ValueError("oriented-simplex ROCA requires exactly d + 1 groups in d dimensions")
    rows, cols = linear_sum_assignment(-transport)
    order = cols[np.argsort(rows)]
    source_simplex = (source[1:] - source[0]).T
    matched_target = target[order]
    target_simplex = (matched_target[1:] - matched_target[0]).T
    source_volume = float(np.linalg.det(source_simplex))
    target_volume = float(np.linalg.det(target_simplex))
    product = source_volume * target_volume
    if abs(product) <= 1e-10:
        raise ValueError("representative simplex is degenerate")
    return {
        "orientation_sign": int(np.sign(product)),
        "volume_product_margin": float(abs(product)),
        "source_volume": source_volume,
        "target_volume": target_volume,
        "target_order": order.astype(int).tolist(),
    }


def qualify_roca_candidates(
    source: np.ndarray,
    source_assignments: np.ndarray,
    target: np.ndarray,
    target_assignments: np.ndarray,
    candidates: Iterable[RocaCandidateEvidence],
    *,
    max_local_marginal_error: float = 1e-3,
    max_relative_fit_ratio: float = 1.5,
    min_relative_objective_gap: float = 0.05,
    covariance_spectrum_compatible: bool = True,
    covariance_spectral_mismatch: float | None = None,
    rotation_geometry_identifiable: bool = True,
    covariance_identifiability_margin: float | None = None,
) -> dict[str, Any]:
    """Select exactly one well-posed candidate or abstain.

    No labels, held-out metric, or ground truth enters the decision. Candidate
    transports are never averaged. Oriented-simplex consistency is a health
    check rather than a branch discriminator: either determinant-constrained
    branch may be self-consistent with a different group permutation. When both
    candidates are healthy, their internal objectives must have a material gap.
    """
    evidence = tuple(candidates)
    signs = {item.determinant_sign for item in evidence}
    if signs != {-1, 1} or len(evidence) != 2:
        raise ValueError("ROCA qualification requires exactly one -1 and one +1 candidate")
    if max_local_marginal_error <= 0 or max_relative_fit_ratio <= 0:
        raise ValueError("qualification thresholds must be positive")
    if min_relative_objective_gap < 0:
        raise ValueError("min_relative_objective_gap must be non-negative")
    source_representatives = _representatives(source, source_assignments)
    target_representatives = _representatives(target, target_assignments)
    rows: list[dict[str, Any]] = []
    for item in sorted(evidence, key=lambda row: row.determinant_sign):
        try:
            orientation = _orientation(source_representatives, target_representatives, item.group_transport)
            orientation_error = None
        except ValueError as exc:
            orientation = None
            orientation_error = str(exc)
        valid = bool(
            item.admm_converged
            and item.final_local_marginal_error <= max_local_marginal_error
            and np.isfinite(item.transport_objective)
            and np.isfinite(item.relative_global_fit_ratio)
            and item.relative_global_fit_ratio <= max_relative_fit_ratio
            and covariance_spectrum_compatible
            and rotation_geometry_identifiable
            and orientation is not None
            and orientation["orientation_sign"] == item.determinant_sign
        )
        rows.append(
            {
                "determinant_sign": item.determinant_sign,
                "admm_converged": bool(item.admm_converged),
                "final_local_marginal_error": float(item.final_local_marginal_error),
                "transport_objective": float(item.transport_objective),
                "relative_global_fit_ratio": float(item.relative_global_fit_ratio),
                "covariance_spectrum_compatible": bool(covariance_spectrum_compatible),
                "rotation_geometry_identifiable": bool(rotation_geometry_identifiable),
                "orientation": orientation,
                "orientation_error": orientation_error,
                "qualified": valid,
            }
        )
    qualified = [row for row in rows if row["qualified"]]
    relative_objective_gap: float | None = None
    if not covariance_spectrum_compatible:
        selected_sign = None
        decision = "covariance_spectrum_incompatible"
    elif not rotation_geometry_identifiable:
        selected_sign = None
        decision = "rotation_geometry_weakly_identifiable"
    elif len(qualified) == 1:
        selected_sign: int | None = int(qualified[0]["determinant_sign"])
        decision = "unique_qualified_candidate"
    elif len(qualified) == 0:
        selected_sign = None
        decision = "no_qualified_candidate"
    else:
        ranked = sorted(qualified, key=lambda row: float(row["transport_objective"]))
        best, runner_up = ranked
        relative_objective_gap = float(
            (float(runner_up["transport_objective"]) - float(best["transport_objective"]))
            / max(abs(float(best["transport_objective"])), EPS)
        )
        if relative_objective_gap >= min_relative_objective_gap:
            selected_sign = int(best["determinant_sign"])
            decision = "objective_separated_candidates"
        else:
            selected_sign = None
            decision = "ambiguous_multiple_qualified_candidates"
    return {
        "selected_sign": selected_sign,
        "decision": decision,
        "max_local_marginal_error": float(max_local_marginal_error),
        "max_relative_fit_ratio": float(max_relative_fit_ratio),
        "covariance_spectrum_compatible": bool(covariance_spectrum_compatible),
        "covariance_spectral_mismatch": covariance_spectral_mismatch,
        "rotation_geometry_identifiable": bool(rotation_geometry_identifiable),
        "covariance_identifiability_margin": covariance_identifiability_margin,
        "min_relative_objective_gap": float(min_relative_objective_gap),
        "relative_objective_gap": relative_objective_gap,
        "candidates": rows,
    }
