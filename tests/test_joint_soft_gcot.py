from __future__ import annotations

import numpy as np

from joint_soft_gcot import JointSoftGCOTConfig, update_prototypes_from_soft_gcot
from soft_groups import learn_soft_groups


def test_alignment_aware_update_preserves_assignments_and_reduces_objective() -> None:
    rng = np.random.default_rng(17)
    source = np.vstack([rng.normal([-1.0, 0.0, 0.0], 0.15, (12, 3)), rng.normal([1.0, 0.0, 0.0], 0.15, (12, 3))])
    rotation = np.diag([1.0, -1.0, 1.0])
    target = source @ rotation.T + rng.normal(0.0, 0.02, source.shape)
    source_init = learn_soft_groups(source, n_groups=2, temperature=0.7, seed=4, maxiter=20)
    target_init = learn_soft_groups(target, n_groups=2, temperature=0.7, seed=5, maxiter=20)
    result = update_prototypes_from_soft_gcot(
        source,
        target,
        rotation=rotation,
        group_transport=np.eye(2) / 2.0,
        initial_source_prototypes_standardized=source_init.prototypes_standardized,
        initial_target_prototypes_standardized=target_init.prototypes_standardized,
        target_entropy=0.5,
        config=JointSoftGCOTConfig(n_groups=2, temperature=0.7, maxiter=12),
    )
    np.testing.assert_allclose(result.source_assignments.sum(axis=1), 1.0, atol=1e-10)
    np.testing.assert_allclose(result.target_assignments.sum(axis=1), 1.0, atol=1e-10)
    assert result.final_loss["total"] <= result.initial_loss["total"] + 1e-8
    assert result.final_loss["source_min_mass"] > 0.0
    assert result.final_loss["target_min_mass"] > 0.0


def test_alignment_aware_update_checks_transport_shape() -> None:
    values = np.eye(3)
    learned = learn_soft_groups(values, n_groups=2, temperature=0.7, seed=1, maxiter=5)
    try:
        update_prototypes_from_soft_gcot(
            values,
            values,
            rotation=np.eye(3),
            group_transport=np.ones((3, 3)),
            initial_source_prototypes_standardized=learned.prototypes_standardized,
            initial_target_prototypes_standardized=learned.prototypes_standardized,
            target_entropy=0.5,
            config=JointSoftGCOTConfig(n_groups=2),
        )
    except ValueError as error:
        assert "group_transport" in str(error)
    else:
        raise AssertionError("expected group transport shape validation")
