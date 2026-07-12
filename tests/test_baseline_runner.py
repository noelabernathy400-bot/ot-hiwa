from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from run_taco_faithful_baseline import _assignment_stages
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
