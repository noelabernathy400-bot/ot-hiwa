from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from representation import (
    DifferentiableVelocityAwareOT,
    MLPDecoder,
    MLPEncoder,
    VelocityAwareTransportConfig,
)


def test_velocity_aware_ot_never_accepts_target_velocity_during_fit() -> None:
    torch.manual_seed(31)
    rng = np.random.default_rng(31)
    source = np.vstack((rng.normal(-1, 0.1, (8, 2)), rng.normal(1, 0.1, (8, 2)))).astype(np.float32)
    target = source @ np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32).T
    velocity = np.column_stack((source[:, 0], -source[:, 1])).astype(np.float32)
    groups = np.asarray([0] * 8 + [1] * 8)
    model = DifferentiableVelocityAwareOT(
        MLPEncoder(2, 2, 6), MLPEncoder(2, 2, 6),
        MLPDecoder(2, 2, 6), MLPDecoder(2, 2, 6),
        nn.Linear(2, 2), nn.Linear(2, 2),
        n_groups=2,
        config=VelocityAwareTransportConfig(source_warmup_epochs=2, epochs=3, sinkhorn_iterations=10),
    )
    result = model.fit(source, velocity, groups, target, source, velocity)
    assert result.sample_transport.shape == (16, 16)
    assert result.group_transport.shape == (2, 2)
    assert result.target_velocity.shape == (16, 2)
    assert result.best_epoch >= 0
    assert np.isfinite(result.target_velocity).all()
