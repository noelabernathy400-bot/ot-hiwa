from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from joint_prototypes import (  # noqa: E402
    JointPrototypeConfig,
    optimize_joint_prototypes,
    standardize,
)
from soft_groups import assignment_entropy, learn_soft_groups  # noqa: E402


class JointPrototypeTests(unittest.TestCase):
    def test_joint_optimization_returns_normalized_assignments(self) -> None:
        rng = np.random.default_rng(13)
        source = np.vstack(
            (
                rng.normal([-1.0, 0.0, 0.0], 0.15, (12, 3)),
                rng.normal([1.0, 0.0, 0.0], 0.15, (12, 3)),
            )
        )
        target = source @ np.diag([1.0, -1.0, 1.0]).T + rng.normal(0, 0.03, source.shape)
        source_init = learn_soft_groups(source, n_groups=2, temperature=0.7, seed=2, maxiter=20)
        target_init = learn_soft_groups(target, n_groups=2, temperature=0.7, seed=3, maxiter=20)
        h0 = float(
            0.5
            * (
                assignment_entropy(source_init.assignments).mean()
                + assignment_entropy(target_init.assignments).mean()
            )
        )

        result = optimize_joint_prototypes(
            source,
            target,
            rotation=np.diag([1.0, -1.0, 1.0]),
            initial_source_prototypes_standardized=source_init.prototypes_standardized,
            initial_target_prototypes_standardized=target_init.prototypes_standardized,
            target_entropy=h0,
            config=JointPrototypeConfig(
                n_groups=2,
                temperature=0.7,
                lambda_align=0.1,
                lambda_balance=0.05,
                lambda_entropy=0.1,
                maxiter=8,
            ),
        )

        np.testing.assert_allclose(result.source_assignments.sum(axis=1), 1.0, atol=1e-10)
        np.testing.assert_allclose(result.target_assignments.sum(axis=1), 1.0, atol=1e-10)
        self.assertEqual(result.source_representatives.shape, (2, 3))
        self.assertEqual(result.target_representatives.shape, (2, 3))
        self.assertLessEqual(
            result.final_loss["total"],
            result.initial_loss["total"] + 1e-8,
        )
        self.assertLess(result.final_loss["group_transport_marginal_error"], 1e-6)

    def test_initial_prototype_shape_is_checked(self) -> None:
        values = np.eye(3)
        standardized, _, _ = standardize(values)
        with self.assertRaises(ValueError):
            optimize_joint_prototypes(
                values,
                values,
                rotation=np.eye(3),
                initial_source_prototypes_standardized=standardized[:2],
                initial_target_prototypes_standardized=standardized[:1],
                target_entropy=0.5,
                config=JointPrototypeConfig(n_groups=2),
            )


if __name__ == "__main__":
    unittest.main()
