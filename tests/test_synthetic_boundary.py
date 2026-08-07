from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "src" / "cc_hiwa"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from synthetic_boundary import (  # noqa: E402
    BoundaryScenario,
    VariableGroupScenario,
    make_synthetic_pair,
    make_variable_group_pair,
    random_orthogonal,
)
from alignment_qualification import normalized_covariance_spectral_mismatch  # noqa: E402


def test_ideal_pair_is_shuffled_but_exactly_orthogonal_before_measurement_noise() -> None:
    scenario = BoundaryScenario("ideal", noise_std=0.0, samples_per_group=5)
    pair = make_synthetic_pair(scenario, seed=11)
    expected = pair.source @ pair.rotation_truth.T
    np.testing.assert_allclose(pair.target_paired_truth, expected, atol=1e-12)
    assert not np.array_equal(pair.target_permutation, np.arange(pair.source.shape[0]))
    np.testing.assert_allclose(pair.rotation_truth.T @ pair.rotation_truth, np.eye(3), atol=1e-12)
    assert np.linalg.det(pair.rotation_truth) > 0


def test_mixed_transform_corruption_changes_only_a_seeded_fraction() -> None:
    scenario = BoundaryScenario("mixed", noise_std=0.0, mixed_transform_fraction=0.5, samples_per_group=10)
    pair = make_synthetic_pair(scenario, seed=12)
    assert 0 < pair.mixed_transform_mask.sum() < pair.source.shape[0]
    assert pair.target.shape == pair.source.shape
    assert pair.target_labels.shape == pair.source_labels.shape


def test_reflection_rotation_has_requested_component() -> None:
    reflection = random_orthogonal(3, seed=13, determinant_sign=-1)
    np.testing.assert_allclose(reflection.T @ reflection, np.eye(3), atol=1e-12)
    assert np.linalg.det(reflection) < 0


def test_cross_modal_scenario_explicitly_has_no_single_rotation_truth() -> None:
    pair = make_synthetic_pair(
        BoundaryScenario("cross_modal", cross_modal_target=True, samples_per_group=5),
        seed=21,
    )
    assert not pair.has_single_rotation_truth
    assert pair.source.shape == pair.target.shape


def test_nonorthogonal_scenarios_do_not_expose_a_spurious_rotation_truth() -> None:
    pair = make_synthetic_pair(
        BoundaryScenario("anisotropic", nonorthogonal_scale=0.1, samples_per_group=5),
        seed=23,
    )
    assert not pair.has_single_rotation_truth


def test_isospectral_nonorthogonal_scenario_preserves_spectrum_but_has_no_rotation_truth() -> None:
    pair = make_synthetic_pair(
        BoundaryScenario(
            "isospectral_nonorthogonal",
            noise_std=0.0,
            isospectral_nonorthogonal=True,
            samples_per_group=5,
        ),
        seed=24,
    )
    assert not pair.has_single_rotation_truth
    assert normalized_covariance_spectral_mismatch(pair.source, pair.target) < 1e-12


def test_variable_group_pair_retains_an_arbitrary_known_group_count() -> None:
    pair = make_variable_group_pair(
        VariableGroupScenario("five_groups", true_groups=5, samples_per_group=6),
        seed=22,
    )
    assert pair.source.shape == (30, 3)
    assert np.unique(pair.source_labels).size == 5
    np.testing.assert_allclose(pair.rotation_truth.T @ pair.rotation_truth, np.eye(3), atol=1e-12)
    assert not np.array_equal(pair.target_permutation, np.arange(pair.source.shape[0]))
    assert pair.has_single_rotation_truth
