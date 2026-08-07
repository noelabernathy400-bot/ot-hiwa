from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from run_taco_faithful_baseline import _assignment_stages, _subsample_indices
from soft_hiwa import SoftHiWA


def test_paired_hard_labels_come_from_final_soft_assignments() -> None:
    values = np.random.default_rng(3).normal(size=(24, 3))
    args = SimpleNamespace(
        temperature_path=[0.25, 0.5], temperature=1.0, groups=4, entropy_weight=0.05
    )
    hard, stages = _assignment_stages(values, args, seed=7)
    np.testing.assert_array_equal(hard, np.argmax(stages[-1], axis=1))


def test_soft_hiwa_defaults_to_full_support() -> None:
    assert SoftHiWA().support_mode == "full"


def test_random_subsampling_is_seeded_unlabeled_and_without_replacement() -> None:
    source_a, target_a, metadata_a = _subsample_indices(803, 623, 96, subset_seed=1001)
    source_b, target_b, metadata_b = _subsample_indices(803, 623, 96, subset_seed=1001)
    source_other, target_other, _ = _subsample_indices(803, 623, 96, subset_seed=1002)

    np.testing.assert_array_equal(source_a, source_b)
    np.testing.assert_array_equal(target_a, target_b)
    assert len(np.unique(source_a)) == len(source_a) == 96
    assert len(np.unique(target_a)) == len(target_a) == 96
    assert not np.array_equal(source_a, source_other)
    assert not np.array_equal(target_a, target_other)
    assert metadata_a == metadata_b == {"mode": "unlabeled_random_without_replacement", "seed": 1001}


def test_legacy_subsampling_remains_evenly_spaced() -> None:
    source, target, metadata = _subsample_indices(9, 7, 4, subset_seed=None)
    np.testing.assert_array_equal(source, np.asarray([0, 2, 5, 8]))
    np.testing.assert_array_equal(target, np.asarray([0, 2, 4, 6]))
    assert metadata == {"mode": "legacy_evenly_spaced", "seed": None}
