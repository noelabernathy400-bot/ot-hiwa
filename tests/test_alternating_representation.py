from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "cc_hiwa"))

from alternating_representation import (
    AlternatingRepresentationConfig,
    AlternatingRepresentationSoftGCOT,
)
from representation import MLPDecoder, MLPEncoder, ResidualMLPEncoder


def test_residual_encoder_starts_as_identity_when_dimensions_match() -> None:
    torch.manual_seed(4)
    values = torch.randn(5, 3)
    encoder = ResidualMLPEncoder(3, 3, hidden_dimension=5)
    torch.testing.assert_close(encoder(values), values)


def test_alternating_representation_updates_latents_without_collapse() -> None:
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    source = np.vstack((rng.normal(-1.0, 0.2, (8, 2)), rng.normal(1.0, 0.2, (8, 2)))).astype(np.float32)
    rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32)
    target = source @ rotation.T
    config = AlternatingRepresentationConfig(
        n_groups=2,
        temperature=0.7,
        outer_steps=2,
        encoder_steps=3,
        learning_rate=3e-3,
        solver_kwargs={
            "normalize": False,
            "maxiter": 6,
            "tol": 10.0,
            "sa_maxiter": 3,
            "shorn_maxiter": 60,
            "sa_shorn_maxiter": 60,
            "random_state": 7,
            "warm_start_local": True,
            "support_mode": "full",
        },
    )
    model = AlternatingRepresentationSoftGCOT(
        MLPEncoder(2, 2, hidden_dimension=6),
        MLPEncoder(2, 2, hidden_dimension=6),
        MLPDecoder(2, 2, hidden_dimension=6),
        MLPDecoder(2, 2, hidden_dimension=6),
        latent_dimension=2,
        config=config,
    )
    initial_parameter = next(model.source_encoder.parameters()).detach().clone()
    result = model.fit(source, target)

    assert len(result.history) == 2
    assert result.solution.sample_coupling.shape == (16, 16)
    assert np.isclose(result.solution.sample_coupling.sum(), 1.0)
    assert np.isfinite(result.source_latent).all()
    assert np.isfinite(result.target_latent).all()
    assert result.history[-1]["source_effective_rank"] > 1.0
    assert result.history[-1]["target_effective_rank"] > 1.0
    assert not torch.allclose(initial_parameter, next(model.source_encoder.parameters()).detach())
