"""Latent-health regularizers for the alternating representation MVP."""

from __future__ import annotations

import torch


def variance_floor_penalty(values: torch.Tensor, minimum_std: float = 0.25) -> torch.Tensor:
    """Penalize latent dimensions whose batch standard deviation approaches zero."""
    if minimum_std <= 0:
        raise ValueError("minimum_std must be positive")
    std = torch.sqrt(values.var(dim=0, unbiased=False) + 1e-4)
    return torch.relu(minimum_std - std).mean()


def covariance_penalty(values: torch.Tensor) -> torch.Tensor:
    """Discourage redundant latent coordinates without enforcing whitening exactly."""
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(values.shape[0] - 1, 1)
    off_diagonal = covariance - torch.diag(torch.diag(covariance))
    return off_diagonal.pow(2).mean()
