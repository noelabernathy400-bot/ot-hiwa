from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src" / "cc_hiwa") not in sys.path:
    sys.path.insert(0, str(ROOT / "src" / "cc_hiwa"))

from soft_groups import _standardize, learn_soft_groups  # noqa: E402


def test_global_scalar_standardization_preserves_unknown_orthogonal_relation() -> None:
    rng = np.random.default_rng(55)
    source = rng.normal(size=(40, 3)) + np.array([2.0, -1.0, 0.5])
    basis, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    target = source @ basis.T

    source_standardized, _, _ = _standardize(source, scaling_mode="global_scalar")
    target_standardized, _, _ = _standardize(target, scaling_mode="global_scalar")
    np.testing.assert_allclose(target_standardized, source_standardized @ basis.T, atol=1e-12)


def test_per_feature_standardization_is_not_rotation_equivariant_in_general() -> None:
    source = np.array([[0.0, 0.0], [2.0, 1.0], [5.0, -2.0], [7.0, 4.0]])
    angle = np.deg2rad(30.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    target = source @ rotation.T
    source_standardized, _, _ = _standardize(source, scaling_mode="per_feature")
    target_standardized, _, _ = _standardize(target, scaling_mode="per_feature")
    assert not np.allclose(target_standardized, source_standardized @ rotation.T)


def test_soft_groups_records_scaling_mode() -> None:
    rng = np.random.default_rng(12)
    result = learn_soft_groups(
        rng.normal(size=(12, 3)),
        n_groups=2,
        seed=3,
        maxiter=10,
        scaling_mode="global_scalar",
    )
    assert result.diagnostics["scaling_mode"] == "global_scalar"
