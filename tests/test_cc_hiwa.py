from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from cc_hiwa import (  # noqa: E402
    barycentric_projection,
    coupling_entropy,
    compatibility_from_group_transport,
    component_conditioned_cost,
    fit_cc_hiwa,
    soft_representatives,
    squared_euclidean_cost,
)


class CCHiWATests(unittest.TestCase):
    def test_soft_representatives_match_weighted_means(self) -> None:
        values = np.asarray([[0.0, 0.0], [2.0, 0.0], [10.0, 1.0]])
        assignments = np.asarray(
            [
                [1.0, 0.0],
                [0.5, 0.5],
                [0.0, 1.0],
            ]
        )
        representatives = soft_representatives(values, assignments)
        expected_first = np.asarray([2.0 / 3.0, 0.0])
        expected_second = np.asarray([22.0 / 3.0, 2.0 / 3.0])
        np.testing.assert_allclose(representatives[0], expected_first)
        np.testing.assert_allclose(representatives[1], expected_second)

    def test_compatibility_shape_and_values(self) -> None:
        a = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        b = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        group_transport = np.asarray([[0.45, 0.05], [0.05, 0.45]])
        compatibility = compatibility_from_group_transport(a, group_transport, b)
        np.testing.assert_allclose(compatibility, group_transport)
        self.assertGreater(compatibility[0, 0], compatibility[0, 1])

    def test_beta_zero_preserves_base_cost(self) -> None:
        base = np.asarray([[0.0, 1.0], [2.0, 3.0]])
        compatibility = np.asarray([[0.9, 0.1], [0.1, 0.9]])
        conditioned = component_conditioned_cost(base, compatibility, beta=0.0)
        np.testing.assert_allclose(conditioned, base)

    def test_component_conditioning_prefers_compatible_pairs(self) -> None:
        base = np.ones((2, 2))
        compatibility = np.asarray([[0.9, 0.1], [0.1, 0.9]])
        conditioned = component_conditioned_cost(base, compatibility, beta=0.5)
        self.assertLess(conditioned[0, 0], conditioned[0, 1])
        self.assertLess(conditioned[1, 1], conditioned[1, 0])

    def test_fit_cc_hiwa_returns_valid_transports(self) -> None:
        source = np.asarray(
            [
                [-1.0, 0.0],
                [-0.9, 0.1],
                [1.0, 0.0],
                [0.9, -0.1],
            ]
        )
        target = source.copy()
        assignments = np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )
        result = fit_cc_hiwa(
            source,
            target,
            assignments,
            assignments,
            beta=0.25,
            group_gamma=0.05,
            sample_gamma=0.05,
            sinkhorn_maxiter=200,
        )
        self.assertEqual(result.group_transport.shape, (2, 2))
        self.assertEqual(result.sample_transport.shape, (4, 4))
        np.testing.assert_allclose(result.group_transport.sum(axis=1), 0.5, atol=1e-6)
        np.testing.assert_allclose(result.group_transport.sum(axis=0), 0.5, atol=1e-6)
        np.testing.assert_allclose(result.sample_transport.sum(axis=1), 0.25, atol=1e-6)
        np.testing.assert_allclose(result.sample_transport.sum(axis=0), 0.25, atol=1e-6)
        self.assertLess(result.conditioned_sample_cost[0, 0], result.conditioned_sample_cost[0, 2])

    def test_squared_euclidean_cost_is_nonnegative(self) -> None:
        source = np.asarray([[0.0, 0.0], [1.0, 0.0]])
        target = np.asarray([[0.0, 1.0], [1.0, 1.0]])
        cost = squared_euclidean_cost(source, target)
        self.assertTrue(np.all(cost >= 0.0))
        np.testing.assert_allclose(cost, np.asarray([[1.0, 2.0], [2.0, 1.0]]))

    def test_barycentric_projection_uses_row_normalized_transport(self) -> None:
        transport = np.asarray([[0.2, 0.0], [0.1, 0.3]])
        target = np.asarray([[0.0, 0.0], [2.0, 0.0]])
        projected = barycentric_projection(transport, target)
        np.testing.assert_allclose(projected[0], np.asarray([0.0, 0.0]))
        np.testing.assert_allclose(projected[1], np.asarray([1.5, 0.0]))

    def test_coupling_entropy_is_finite(self) -> None:
        transport = np.asarray([[0.5, 0.0], [0.0, 0.5]])
        entropy = coupling_entropy(transport)
        self.assertTrue(np.isfinite(entropy))
        self.assertGreater(entropy, 0.0)


if __name__ == "__main__":
    unittest.main()
