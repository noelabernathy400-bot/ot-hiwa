from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from soft_groups import (  # noqa: E402
    assignment_entropy,
    assignments_from_prototypes,
    build_sparse_group_supports,
    learn_soft_groups,
)
from soft_hiwa import SoftHiWA, _closed_form_rotation, sinkhorn_weighted  # noqa: E402


class SoftGroupTests(unittest.TestCase):
    def test_assignments_and_group_weights_are_normalized(self) -> None:
        rng = np.random.default_rng(3)
        values = np.vstack((rng.normal(-1, 0.2, (30, 2)), rng.normal(1, 0.2, (30, 2))))
        result = learn_soft_groups(values, n_groups=2, seed=4, maxiter=60)
        np.testing.assert_allclose(result.assignments.sum(axis=1), 1.0, atol=1e-10)
        np.testing.assert_allclose(result.group_weights.sum(axis=0), 1.0, atol=1e-10)
        self.assertGreater(result.diagnostics["min_group_mass"], 0.05)

    def test_lower_temperature_reduces_entropy_for_fixed_prototypes(self) -> None:
        values = np.asarray([[-1.0, 0.0], [-0.2, 0.0], [0.2, 0.0], [1.0, 0.0]])
        prototypes = np.asarray([[-1.0, 0.0], [1.0, 0.0]])
        cold = assignments_from_prototypes(values, prototypes, temperature=0.25)
        warm = assignments_from_prototypes(values, prototypes, temperature=2.0)
        self.assertLess(assignment_entropy(cold).mean(), assignment_entropy(warm).mean())

    def test_one_hot_supports_equal_hard_clusters(self) -> None:
        assignments = np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
        )
        indices, weights, retained = build_sparse_group_supports(
            assignments,
            retain_mass=1.0,
            max_support_factor=3.0,
            min_support=1,
        )
        self.assertEqual(set(indices[0].tolist()), {0, 1})
        self.assertEqual(set(indices[1].tolist()), {2, 3, 4})
        np.testing.assert_allclose(np.sort(weights[0]), [0.5, 0.5])
        np.testing.assert_allclose(np.sort(weights[1]), [1 / 3, 1 / 3, 1 / 3])
        np.testing.assert_allclose(retained, [1.0, 1.0])

    def test_full_support_keeps_every_sample_and_exact_soft_marginals(self) -> None:
        assignments = np.asarray([[0.8, 0.2], [0.3, 0.7], [0.1, 0.9]])
        source = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        model = SoftHiWA(
            normalize=False, maxiter=6, tol=10.0, sa_maxiter=2,
            shorn_maxiter=300, sa_shorn_maxiter=300, support_mode="full",
            random_state=3,
        ).fit(source, assignments, source, assignments)
        self.assertEqual(model.diagnostics["support_mode"], "full")
        self.assertEqual(model.diagnostics["source_support_sizes"], [3, 3])
        np.testing.assert_allclose(model.diagnostics["source_retained_mass"], [1.0, 1.0])
        self.assertLess(model.diagnostics["max_sinkhorn_marginal_error"].max(), 1e-7)
        self.assertLess(model.diagnostics["group_transport_row_marginal_error"], 1e-7)


class WeightedTransportTests(unittest.TestCase):
    def test_sinkhorn_respects_nonuniform_marginals(self) -> None:
        source = np.asarray([[0.0, 1.0, 2.0]])
        target = np.asarray([[0.0, 2.0]])
        p = np.asarray([0.2, 0.3, 0.5])
        q = np.asarray([0.6, 0.4])
        coupling, _, error = sinkhorn_weighted(p, q, source, target, gamma=0.5, maxiter=300)
        np.testing.assert_allclose(coupling.sum(axis=1), p, atol=1e-7)
        np.testing.assert_allclose(coupling.sum(axis=0), q, atol=1e-7)
        self.assertLess(error, 1e-7)

    def test_sinkhorn_accepts_extra_component_cost(self) -> None:
        source = np.asarray([[0.0, 1.0]])
        target = np.asarray([[0.0, 1.0]])
        p = np.asarray([0.5, 0.5])
        q = np.asarray([0.5, 0.5])
        extra = np.asarray([[0.0, 1.0], [1.0, 0.0]])
        coupling, _, error = sinkhorn_weighted(
            p,
            q,
            source,
            target,
            gamma=0.5,
            maxiter=300,
            extra_cost=extra,
            extra_cost_weight=0.2,
        )
        np.testing.assert_allclose(coupling.sum(axis=1), p, atol=1e-7)
        np.testing.assert_allclose(coupling.sum(axis=0), q, atol=1e-7)
        self.assertLess(error, 1e-7)

    def test_closed_form_rotation_is_orthogonal(self) -> None:
        rng = np.random.default_rng(9)
        rotation = _closed_form_rotation(rng.normal(size=(3, 3)))
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)

    def test_closed_form_rotation_respects_requested_component(self) -> None:
        rng = np.random.default_rng(17)
        matrix = rng.normal(size=(3, 3))
        positive = _closed_form_rotation(matrix, determinant_sign=1)
        negative = _closed_form_rotation(matrix, determinant_sign=-1)
        self.assertAlmostEqual(float(np.linalg.det(positive)), 1.0, places=10)
        self.assertAlmostEqual(float(np.linalg.det(negative)), -1.0, places=10)
        np.testing.assert_allclose(positive.T @ positive, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(negative.T @ negative, np.eye(3), atol=1e-10)

    def test_invalid_determinant_sign_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "determinant_sign"):
            SoftHiWA(determinant_sign=0)

    def test_negative_rotation_anchor_weight_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            SoftHiWA(rotation_anchor_weight=-0.1)

    def test_invalid_representative_guidance_weight_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "representative_guidance_weight"):
            SoftHiWA(representative_guidance_weight=-0.1)
        with self.assertRaisesRegex(ValueError, "representative_guidance_weight"):
            SoftHiWA(representative_guidance_weight=1.1)

    def test_negative_representative_rotation_weight_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "representative_rotation_weight"):
            SoftHiWA(representative_rotation_weight=-0.1)

    def test_negative_component_conditioning_weight_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "component_conditioning_weight"):
            SoftHiWA(component_conditioning_weight=-0.1)

    def test_positive_anchor_weight_requires_anchor(self) -> None:
        model = SoftHiWA(
            normalize=False,
            maxiter=1,
            sa_maxiter=1,
            shorn_maxiter=2,
            sa_shorn_maxiter=2,
            rotation_anchor_weight=1.0,
        )
        values = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        assignments = np.asarray([[1.0], [1.0], [1.0]])
        with self.assertRaisesRegex(ValueError, "rotation_anchor is required"):
            model.fit(values, assignments, values, assignments)

    def test_rotation_anchor_pulls_closed_form_update_toward_anchor(self) -> None:
        consensus = np.asarray([[0.0, -1.0], [1.0, 0.0]])
        anchor = np.eye(2)
        unanchored = _closed_form_rotation(consensus)
        anchored = _closed_form_rotation(consensus + 10.0 * anchor)
        self.assertLess(
            np.linalg.norm(anchored - anchor, "fro"),
            np.linalg.norm(unanchored - anchor, "fro"),
        )

    def test_zero_anchor_weight_preserves_legacy_fit(self) -> None:
        rng = np.random.default_rng(11)
        source = rng.normal(size=(12, 2))
        target = source @ np.asarray([[0.0, -1.0], [1.0, 0.0]]).T
        assignments = np.zeros((12, 2))
        assignments[:6, 0] = 1.0
        assignments[6:, 1] = 1.0
        settings = dict(
            normalize=False,
            maxiter=6,
            sa_maxiter=4,
            shorn_maxiter=20,
            sa_shorn_maxiter=10,
            retain_mass=1.0,
            max_support_factor=2.0,
            random_state=5,
            warm_start_local=True,
            rotation_anchor_weight=0.0,
        )
        legacy = SoftHiWA(**settings).fit(
            source,
            assignments,
            target,
            assignments,
            initial_rotation=np.eye(2),
        )
        zero_weight = SoftHiWA(**settings).fit(
            source,
            assignments,
            target,
            assignments,
            initial_rotation=np.eye(2),
            rotation_anchor=np.asarray([[0.0, -1.0], [1.0, 0.0]]),
        )
        np.testing.assert_allclose(zero_weight.Rg, legacy.Rg, atol=1e-12)
        np.testing.assert_allclose(zero_weight.P, legacy.P, atol=1e-12)

    def test_zero_representative_guidance_preserves_legacy_fit(self) -> None:
        rng = np.random.default_rng(13)
        source = rng.normal(size=(16, 2))
        target = source @ np.asarray([[0.0, -1.0], [1.0, 0.0]]).T
        assignments = np.zeros((16, 2))
        assignments[:8, 0] = 1.0
        assignments[8:, 1] = 1.0
        settings = dict(
            normalize=False,
            maxiter=5,
            sa_maxiter=3,
            shorn_maxiter=20,
            sa_shorn_maxiter=10,
            retain_mass=1.0,
            max_support_factor=2.0,
            random_state=7,
            warm_start_local=True,
        )
        legacy = SoftHiWA(**settings).fit(
            source,
            assignments,
            target,
            assignments,
            initial_rotation=np.eye(2),
        )
        guided_zero = SoftHiWA(
            **settings,
            representative_guidance_weight=0.0,
        ).fit(
            source,
            assignments,
            target,
            assignments,
            initial_rotation=np.eye(2),
        )
        np.testing.assert_allclose(guided_zero.Rg, legacy.Rg, atol=1e-12)
        np.testing.assert_allclose(guided_zero.P, legacy.P, atol=1e-12)

    def test_zero_representative_rotation_preserves_legacy_fit(self) -> None:
        rng = np.random.default_rng(23)
        source = rng.normal(size=(16, 2))
        target = source @ np.asarray([[0.0, -1.0], [1.0, 0.0]]).T
        assignments = np.zeros((16, 2))
        assignments[:8, 0] = 1.0
        assignments[8:, 1] = 1.0
        settings = dict(
            normalize=False,
            maxiter=5,
            sa_maxiter=3,
            shorn_maxiter=20,
            sa_shorn_maxiter=10,
            retain_mass=1.0,
            max_support_factor=2.0,
            random_state=11,
            warm_start_local=True,
        )
        legacy = SoftHiWA(**settings).fit(
            source,
            assignments,
            target,
            assignments,
            initial_rotation=np.eye(2),
        )
        rotation_zero = SoftHiWA(
            **settings,
            representative_rotation_weight=0.0,
        ).fit(
            source,
            assignments,
            target,
            assignments,
            initial_rotation=np.eye(2),
        )
        np.testing.assert_allclose(rotation_zero.Rg, legacy.Rg, atol=1e-12)
        np.testing.assert_allclose(rotation_zero.P, legacy.P, atol=1e-12)

    def test_zero_component_conditioning_preserves_legacy_fit(self) -> None:
        rng = np.random.default_rng(37)
        source = rng.normal(size=(18, 2))
        target = source @ np.asarray([[0.0, -1.0], [1.0, 0.0]]).T
        assignments = np.zeros((18, 3))
        assignments[:6, 0] = 1.0
        assignments[6:12, 1] = 1.0
        assignments[12:, 2] = 1.0
        settings = dict(
            normalize=False,
            maxiter=5,
            sa_maxiter=3,
            shorn_maxiter=20,
            sa_shorn_maxiter=10,
            retain_mass=1.0,
            max_support_factor=2.0,
            random_state=41,
            warm_start_local=True,
        )
        legacy = SoftHiWA(**settings).fit(
            source,
            assignments,
            target,
            assignments,
            initial_rotation=np.eye(2),
        )
        conditioned_zero = SoftHiWA(
            **settings,
            component_conditioning_weight=0.0,
        ).fit(
            source,
            assignments,
            target,
            assignments,
            initial_rotation=np.eye(2),
        )
        np.testing.assert_allclose(conditioned_zero.Rg, legacy.Rg, atol=1e-12)
        np.testing.assert_allclose(conditioned_zero.P, legacy.P, atol=1e-12)

    def test_positive_representative_guidance_changes_mixed_group_cost(self) -> None:
        rng = np.random.default_rng(17)
        source = rng.normal(size=(18, 2))
        target = source @ np.asarray([[0.0, -1.0], [1.0, 0.0]]).T
        assignments = np.zeros((18, 3))
        assignments[:6, 0] = 1.0
        assignments[6:12, 1] = 1.0
        assignments[12:, 2] = 1.0
        model = SoftHiWA(
            normalize=False,
            maxiter=3,
            sa_maxiter=2,
            shorn_maxiter=20,
            sa_shorn_maxiter=10,
            retain_mass=1.0,
            max_support_factor=2.0,
            random_state=19,
            warm_start_local=True,
            representative_guidance_weight=0.5,
        ).fit(
            source,
            assignments,
            target,
            assignments,
            initial_rotation=np.eye(2),
        )
        self.assertGreater(model.diagnostics["representative_cost_max"], 0.0)
        self.assertFalse(
            np.allclose(model.diagnostics["mixed_group_cost"], model.diagnostics["C"])
        )

    def test_positive_representative_rotation_keeps_valid_rotation(self) -> None:
        rng = np.random.default_rng(29)
        source = rng.normal(size=(18, 3))
        reflection = np.diag([1.0, 1.0, -1.0])
        target = source @ reflection.T
        assignments = np.zeros((18, 3))
        assignments[:6, 0] = 1.0
        assignments[6:12, 1] = 1.0
        assignments[12:, 2] = 1.0
        model = SoftHiWA(
            normalize=False,
            maxiter=4,
            sa_maxiter=2,
            shorn_maxiter=20,
            sa_shorn_maxiter=10,
            retain_mass=1.0,
            max_support_factor=2.0,
            random_state=31,
            warm_start_local=True,
            determinant_sign=-1,
            representative_rotation_weight=0.1,
        ).fit(
            source,
            assignments,
            target,
            assignments,
            initial_rotation=np.eye(3),
        )
        self.assertEqual(model.diagnostics["representative_rotation_weight"], 0.1)
        self.assertGreater(model.diagnostics["representative_rotation_cross_norm"], 0.0)
        np.testing.assert_allclose(model.Rg.T @ model.Rg, np.eye(3), atol=1e-10)
        self.assertAlmostEqual(float(np.linalg.det(model.Rg)), -1.0, places=10)

    def test_positive_component_conditioning_reports_diagnostics(self) -> None:
        rng = np.random.default_rng(43)
        source = rng.normal(size=(18, 2))
        target = source @ np.asarray([[0.0, -1.0], [1.0, 0.0]]).T
        assignments = np.zeros((18, 3))
        assignments[:6, 0] = 1.0
        assignments[6:12, 1] = 1.0
        assignments[12:, 2] = 1.0
        model = SoftHiWA(
            normalize=False,
            maxiter=3,
            sa_maxiter=2,
            shorn_maxiter=20,
            sa_shorn_maxiter=10,
            retain_mass=1.0,
            max_support_factor=2.0,
            random_state=47,
            warm_start_local=True,
            component_conditioning_weight=0.05,
        ).fit(
            source,
            assignments,
            target,
            assignments,
            initial_rotation=np.eye(2),
        )
        self.assertEqual(model.diagnostics["component_conditioning_weight"], 0.05)
        self.assertGreaterEqual(model.diagnostics["component_conditioning_cost_mean"], 0.0)
        self.assertGreater(model.diagnostics["component_conditioning_cost_max"], 0.0)


if __name__ == "__main__":
    unittest.main()
