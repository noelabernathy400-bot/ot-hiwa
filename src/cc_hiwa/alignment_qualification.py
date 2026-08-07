"""Conservative deployment gate for a completed GC-HiWA fit.

This gate deliberately does not claim to prove that two unpaired domains are
semantically equivalent.  It only turns the solver's necessary numerical
health conditions into an explicit decision: return an alignment when every
condition is met, otherwise abstain with machine-readable reasons.
"""

from __future__ import annotations

from math import ceil
from typing import Any, Mapping, Sequence

import numpy as np


def normalized_covariance_spectral_mismatch(source: np.ndarray, target: np.ndarray) -> float:
    """Return a scale- and rotation-invariant mismatch between two point clouds.

    A single orthogonal coordinate map preserves the eigenvalue spectrum of a
    centered covariance matrix.  Each cloud is first normalized by its global
    Frobenius norm, so a shared change of physical units does not cause a
    rejection.  This is only a necessary-condition diagnostic: matching
    spectra do not prove that a shared rotation exists.
    """
    left = np.asarray(source, dtype=float)
    right = np.asarray(target, dtype=float)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("source and target must be two-dimensional with equal feature dimension")
    if left.shape[0] < 2 or right.shape[0] < 2:
        raise ValueError("source and target must each contain at least two samples")
    centered_left = left - left.mean(axis=0, keepdims=True)
    centered_right = right - right.mean(axis=0, keepdims=True)
    left_norm = float(np.linalg.norm(centered_left, "fro"))
    right_norm = float(np.linalg.norm(centered_right, "fro"))
    if left_norm <= 0 or right_norm <= 0:
        return float("inf")
    left_spectrum = np.linalg.eigvalsh((centered_left / left_norm).T @ (centered_left / left_norm))
    right_spectrum = np.linalg.eigvalsh((centered_right / right_norm).T @ (centered_right / right_norm))
    denominator = max(0.5 * (float(np.linalg.norm(left_spectrum)) + float(np.linalg.norm(right_spectrum))), 1e-12)
    return float(np.linalg.norm(left_spectrum - right_spectrum) / denominator)


def normalized_covariance_identifiability_margin(values: np.ndarray) -> float:
    """Return a conservative eigenvalue-separation diagnostic for rotations.

    When a centered point cloud has repeated covariance eigenvalues, rotations
    within the associated eigenspace cannot be distinguished from covariance
    information alone.  A small margin does not prove that the full
    distribution is unidentifiable, but it is a label-free reason to abstain
    rather than assert a unique recovered coordinate frame.
    """
    features = np.asarray(values, dtype=float)
    if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 2:
        raise ValueError("values must have at least two samples and two features")
    centered = features - features.mean(axis=0, keepdims=True)
    norm = float(np.linalg.norm(centered, "fro"))
    if norm <= 0:
        return 0.0
    spectrum = np.linalg.eigvalsh((centered / norm).T @ (centered / norm))
    denominator = max(float(np.linalg.norm(spectrum)), 1e-12)
    return float(np.min(np.diff(spectrum)) / denominator)


def _last(diagnostics: Mapping[str, Any], key: str) -> float:
    values = np.asarray(diagnostics.get(key, ()), dtype=float).ravel()
    if values.size == 0:
        return float("nan")
    return float(values[-1])


def qualify_alignment_fit(
    diagnostics: Mapping[str, Any],
    *,
    max_local_marginal_error: float = 1e-3,
    max_orthogonality_error: float = 1e-6,
    max_relative_fit_ratio: float = 1.5,
    max_covariance_spectral_mismatch: float | None = None,
    min_covariance_identifiability_margin: float | None = None,
    required_rotation_structure: str | None = None,
) -> dict[str, Any]:
    """Accept a numerically compatible fit or explicitly abstain.

    The ADMM stopping rule already binds the global-update and local/global
    consensus residuals to the model's configured tolerance.  This function
    additionally checks local OT marginal feasibility and orthogonality.
    Passing is a *necessary numerical certificate*, not a sufficient proof of
    cross-domain semantic applicability; a structured external validation is
    still required for a scientific claim.
    """
    if max_local_marginal_error <= 0 or max_orthogonality_error <= 0 or max_relative_fit_ratio <= 0:
        raise ValueError("qualification thresholds must be positive")
    if max_covariance_spectral_mismatch is not None and max_covariance_spectral_mismatch <= 0:
        raise ValueError("max_covariance_spectral_mismatch must be positive when provided")
    if min_covariance_identifiability_margin is not None and min_covariance_identifiability_margin <= 0:
        raise ValueError("min_covariance_identifiability_margin must be positive when provided")
    final_local_error = _last(diagnostics, "max_sinkhorn_marginal_error")
    final_global_residual = _last(diagnostics, "Rg_norm")
    final_primal_residual = _last(diagnostics, "admm_primal_residual")
    orthogonality_error = float(diagnostics.get("rotation_orthogonality_error", np.nan))
    relative_fit_ratio = float(diagnostics.get("relative_global_fit_ratio", np.nan))
    covariance_spectral_mismatch = float(diagnostics.get("covariance_spectral_mismatch", np.nan))
    covariance_identifiability_margin = float(
        diagnostics.get("covariance_identifiability_margin", np.nan)
    )
    reasons: list[str] = []
    if not bool(diagnostics.get("admm_converged", False)):
        reasons.append("admm_not_converged")
    if not np.isfinite(final_local_error) or final_local_error > max_local_marginal_error:
        reasons.append("local_ot_marginals_unreliable")
    if not np.isfinite(orthogonality_error) or orthogonality_error > max_orthogonality_error:
        reasons.append("rotation_not_orthogonal")
    if not np.isfinite(relative_fit_ratio) or relative_fit_ratio > max_relative_fit_ratio:
        reasons.append("single_rotation_fit_excessive")
    if max_covariance_spectral_mismatch is not None and (
        not np.isfinite(covariance_spectral_mismatch)
        or covariance_spectral_mismatch > max_covariance_spectral_mismatch
    ):
        reasons.append("covariance_spectrum_incompatible")
    if min_covariance_identifiability_margin is not None and (
        not np.isfinite(covariance_identifiability_margin)
        or covariance_identifiability_margin < min_covariance_identifiability_margin
    ):
        reasons.append("rotation_geometry_weakly_identifiable")
    if required_rotation_structure is not None and diagnostics.get("rotation_structure") != required_rotation_structure:
        reasons.append("rotation_structure_mismatch")
    return {
        "decision": "accepted_numerically_compatible" if not reasons else "abstain",
        "accepted": not reasons,
        "reasons": reasons,
        "final_global_residual": final_global_residual,
        "final_primal_residual": final_primal_residual,
        "final_local_marginal_error": final_local_error,
        "rotation_orthogonality_error": orthogonality_error,
        "relative_global_fit_ratio": relative_fit_ratio,
        "max_relative_fit_ratio": float(max_relative_fit_ratio),
        "covariance_spectral_mismatch": covariance_spectral_mismatch,
        "max_covariance_spectral_mismatch": (
            float(max_covariance_spectral_mismatch)
            if max_covariance_spectral_mismatch is not None
            else None
        ),
        "covariance_identifiability_margin": covariance_identifiability_margin,
        "min_covariance_identifiability_margin": (
            float(min_covariance_identifiability_margin)
            if min_covariance_identifiability_margin is not None
            else None
        ),
        "required_rotation_structure": required_rotation_structure,
        "observed_rotation_structure": diagnostics.get("rotation_structure"),
        "scope": (
            "necessary numerical compatibility check only; acceptance does not prove "
            "semantic equivalence or exact recovery"
        ),
    }


def _rotation_angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    """Geodesic angle between two proper 3-D rotations, in degrees."""
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != (3, 3) or b.shape != (3, 3):
        raise ValueError("restart stability requires 3 by 3 physical rotations")
    relative = a.T @ b
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def qualify_restart_stability(
    physical_rotations: Sequence[np.ndarray],
    individual_acceptance: Sequence[bool],
    *,
    minimum_restarts: int = 3,
    max_rotation_disagreement_degrees: float = 5.0,
    min_consensus_fraction: float = 0.8,
) -> dict[str, Any]:
    """Select a stable multi-start solution or abstain.

    Row order is irrelevant to unpaired OT.  Repeating the fit after different
    target permutations therefore exposes optimizer-specific local minima
    without requiring labels or held-out paired samples.  A medoid is emitted
    only if a predeclared supermajority of individually healthy runs agrees
    within a small physical-rotation angle.
    """
    rotations = tuple(np.asarray(item, dtype=float) for item in physical_rotations)
    accepted = tuple(bool(item) for item in individual_acceptance)
    if len(rotations) != len(accepted):
        raise ValueError("physical_rotations and individual_acceptance must have equal length")
    if minimum_restarts < 2:
        raise ValueError("minimum_restarts must be at least two")
    if max_rotation_disagreement_degrees <= 0:
        raise ValueError("max_rotation_disagreement_degrees must be positive")
    if not 0 < min_consensus_fraction <= 1:
        raise ValueError("min_consensus_fraction must lie in (0, 1]")
    healthy_indices = [index for index, item in enumerate(accepted) if item]
    if len(healthy_indices) < minimum_restarts:
        return {
            "decision": "abstain",
            "accepted": False,
            "reason": "insufficient_individually_qualified_restarts",
            "healthy_indices": healthy_indices,
            "selected_index": None,
        }
    pairwise = np.full((len(healthy_indices), len(healthy_indices)), np.nan)
    for left_position, left_index in enumerate(healthy_indices):
        for right_position, right_index in enumerate(healthy_indices):
            if left_position == right_position:
                pairwise[left_position, right_position] = 0.0
            elif right_position > left_position:
                angle = _rotation_angle_degrees(rotations[left_index], rotations[right_index])
                pairwise[left_position, right_position] = angle
                pairwise[right_position, left_position] = angle
    medoid_position = int(np.argmin(np.mean(pairwise, axis=1)))
    medoid_index = healthy_indices[medoid_position]
    consensus_positions = np.flatnonzero(
        pairwise[medoid_position] <= max_rotation_disagreement_degrees
    )
    consensus_indices = [healthy_indices[int(position)] for position in consensus_positions]
    required = ceil(min_consensus_fraction * len(healthy_indices))
    is_accepted = len(consensus_indices) >= required
    return {
        "decision": "accepted_stable_multistart" if is_accepted else "abstain",
        "accepted": is_accepted,
        "reason": None if is_accepted else "restart_rotation_disagreement",
        "healthy_indices": healthy_indices,
        "selected_index": medoid_index if is_accepted else None,
        "consensus_indices": consensus_indices,
        "required_consensus_count": int(required),
        "max_rotation_disagreement_degrees": float(max_rotation_disagreement_degrees),
        "min_consensus_fraction": float(min_consensus_fraction),
        "pairwise_angle_degrees": pairwise.tolist(),
    }
