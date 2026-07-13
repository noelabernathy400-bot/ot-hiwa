from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from representation import DifferentiableTaskAwareOT, MLPDecoder, MLPEncoder, TaskAwareTransportConfig
from representation.task_aware_transport import log_sinkhorn


def test_log_sinkhorn_has_uniform_marginals() -> None:
    cost = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0]], dtype=torch.float32)
    plan = log_sinkhorn(cost, epsilon=0.2, iterations=100)
    torch.testing.assert_close(plan.sum(dim=1), torch.full((2,), 0.5), atol=1e-5, rtol=0.0)
    torch.testing.assert_close(plan.sum(dim=0), torch.full((3,), 1.0 / 3.0), atol=1e-5, rtol=0.0)


def test_task_aware_transport_trains_without_target_labels_or_pair_ids() -> None:
    torch.manual_seed(9)
    rng = np.random.default_rng(9)
    source = np.vstack((rng.normal(-1.0, 0.1, (8, 2)), rng.normal(1.0, 0.1, (8, 2)))).astype(np.float32)
    target = source @ np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32).T
    labels = np.asarray([0] * 8 + [1] * 8, dtype=np.int64)
    model = DifferentiableTaskAwareOT(
        MLPEncoder(2, 2, hidden_dimension=6),
        MLPEncoder(2, 2, hidden_dimension=6),
        MLPDecoder(2, 2, hidden_dimension=6),
        MLPDecoder(2, 2, hidden_dimension=6),
        nn.Linear(2, 2),
        n_classes=2,
        config=TaskAwareTransportConfig(source_warmup_epochs=0, epochs=5, sinkhorn_iterations=20),
    )
    result = model.fit(source, labels, target, source, labels)
    assert result.sample_transport.shape == (16, 16)
    assert result.group_transport.shape == (2, 2)
    assert result.target_logits.shape == (16, 2)
    assert result.best_epoch >= 0
    assert np.isfinite(result.sample_transport).all()


def test_task_aware_transport_accepts_explicit_weak_pairs_without_target_labels() -> None:
    torch.manual_seed(10)
    rng = np.random.default_rng(10)
    source = np.vstack((rng.normal(-1.0, 0.1, (6, 2)), rng.normal(1.0, 0.1, (6, 2)))).astype(np.float32)
    paired_target = source @ np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32).T
    labels = np.asarray([0] * 6 + [1] * 6, dtype=np.int64)
    model = DifferentiableTaskAwareOT(
        MLPEncoder(2, 2, hidden_dimension=6),
        MLPEncoder(2, 2, hidden_dimension=6),
        MLPDecoder(2, 2, hidden_dimension=6),
        MLPDecoder(2, 2, hidden_dimension=6),
        nn.Linear(2, 2),
        n_classes=2,
        config=TaskAwareTransportConfig(source_warmup_epochs=0, epochs=5, sinkhorn_iterations=20),
    )
    result = model.fit(source, labels, paired_target, source, labels, paired_target=paired_target)
    assert result.sample_transport.shape == (12, 12)
    assert any("pair_loss" in row for row in result.history)
