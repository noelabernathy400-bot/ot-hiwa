from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "cc_hiwa"))

from representation import MLPDecoder, MLPEncoder
from source_supervised_representation import (
    SourceSupervisedAlternatingSoftGCOT,
    SourceSupervisedRepresentationConfig,
)


def test_source_supervised_trainer_accepts_only_source_labels_and_returns_target_logits() -> None:
    torch.manual_seed(12)
    rng = np.random.default_rng(12)
    source = np.vstack((rng.normal(-1.0, 0.15, (8, 2)), rng.normal(1.0, 0.15, (8, 2)))).astype(np.float32)
    target = source @ np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32).T
    labels = np.asarray([0] * 8 + [1] * 8, dtype=np.int64)
    config = SourceSupervisedRepresentationConfig(
        n_groups=2,
        temperature=0.7,
        outer_steps=2,
        encoder_steps=3,
        source_warmup_steps=2,
        learning_rate=3e-3,
        solver_kwargs={
            "normalize": False,
            "maxiter": 6,
            "tol": 10.0,
            "sa_maxiter": 3,
            "shorn_maxiter": 60,
            "sa_shorn_maxiter": 60,
            "random_state": 12,
            "warm_start_local": True,
            "support_mode": "full",
        },
    )
    model = SourceSupervisedAlternatingSoftGCOT(
        MLPEncoder(2, 2, hidden_dimension=6),
        MLPEncoder(2, 2, hidden_dimension=6),
        MLPDecoder(2, 2, hidden_dimension=6),
        MLPDecoder(2, 2, hidden_dimension=6),
        nn.Linear(2, 2),
        latent_dimension=2,
        config=config,
    )
    result = model.fit(source, target, labels)

    assert len(result.history) == 3
    assert result.target_logits.shape == (16, 2)
    assert result.solution.sample_coupling.shape == (16, 16)
    assert np.isfinite(result.target_logits).all()
    assert result.history[-1]["source_task_loss"] >= 0.0
    assert result.history[0]["source_warmup_accuracy"] >= 0.0
