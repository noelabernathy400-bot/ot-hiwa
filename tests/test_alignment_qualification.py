from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src" / "cc_hiwa") not in sys.path:
    sys.path.insert(0, str(ROOT / "src" / "cc_hiwa"))

from alignment_qualification import (  # noqa: E402
    normalized_covariance_identifiability_margin,
    normalized_covariance_spectral_mismatch,
    qualify_alignment_fit,
    qualify_restart_stability,
)


def _healthy() -> dict[str, object]:
    return {
        "admm_converged": True,
        "Rg_norm": [0.01],
        "admm_primal_residual": [0.01],
        "max_sinkhorn_marginal_error": [1e-8],
        "rotation_orthogonality_error": 1e-12,
        "relative_global_fit_ratio": 1.01,
        "rotation_structure": "repeated_3d_blocks",
    }


def test_qualification_accepts_a_healthy_expected_structure() -> None:
    result = qualify_alignment_fit(_healthy(), required_rotation_structure="repeated_3d_blocks")
    assert result["accepted"]
    assert result["decision"] == "accepted_numerically_compatible"


def test_qualification_abstains_with_explicit_failure_reasons() -> None:
    unhealthy = _healthy() | {
        "admm_converged": False,
        "max_sinkhorn_marginal_error": [0.1],
        "relative_global_fit_ratio": 2.0,
    }
    result = qualify_alignment_fit(unhealthy, required_rotation_structure="full")
    assert not result["accepted"]
    assert {
        "admm_not_converged",
        "local_ot_marginals_unreliable",
        "single_rotation_fit_excessive",
        "rotation_structure_mismatch",
    } <= set(result["reasons"])


def test_covariance_spectrum_is_invariant_to_rotation_translation_and_global_scale() -> None:
    rng = np.random.default_rng(7)
    source = rng.normal(size=(80, 3))
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target = 3.7 * source @ rotation.T + np.array([2.0, -3.0, 1.0])
    assert normalized_covariance_spectral_mismatch(source, target) < 1e-12


def test_covariance_spectrum_gate_rejects_anisotropic_deformation() -> None:
    rng = np.random.default_rng(8)
    source = rng.normal(size=(100, 3))
    target = source @ np.diag([1.2, 0.8, 1.1])
    mismatch = normalized_covariance_spectral_mismatch(source, target)
    assert mismatch > 0.05
    result = qualify_alignment_fit(
        _healthy() | {"covariance_spectral_mismatch": mismatch},
        max_covariance_spectral_mismatch=0.05,
    )
    assert not result["accepted"]
    assert "covariance_spectrum_incompatible" in result["reasons"]


def test_identifiability_margin_rejects_axisymmetric_geometry() -> None:
    coordinate = np.linspace(-2.0, 2.0, 40)
    nearly_line_like = np.column_stack([coordinate, 1e-4 * coordinate**2, np.zeros_like(coordinate)])
    margin = normalized_covariance_identifiability_margin(nearly_line_like)
    assert margin < 0.02
    result = qualify_alignment_fit(
        _healthy() | {"covariance_identifiability_margin": margin},
        min_covariance_identifiability_margin=0.02,
    )
    assert not result["accepted"]
    assert "rotation_geometry_weakly_identifiable" in result["reasons"]


def test_restart_stability_selects_the_majority_rotation_cluster() -> None:
    angle = np.deg2rad(1.0)
    nearby = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
    far = np.diag([-1.0, -1.0, 1.0])
    result = qualify_restart_stability(
        [np.eye(3), nearby, np.eye(3), np.eye(3), far],
        [True, True, True, True, True],
        minimum_restarts=3,
        max_rotation_disagreement_degrees=5.0,
        min_consensus_fraction=0.8,
    )
    assert result["accepted"]
    assert result["selected_index"] in {0, 2, 3}
    assert set(result["consensus_indices"]) == {0, 1, 2, 3}


def test_restart_stability_abstains_without_a_supermajority() -> None:
    right_angle = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    result = qualify_restart_stability(
        [np.eye(3), np.eye(3), right_angle],
        [True, True, True],
        minimum_restarts=3,
        max_rotation_disagreement_degrees=5.0,
        min_consensus_fraction=0.8,
    )
    assert not result["accepted"]
    assert result["reason"] == "restart_rotation_disagreement"
