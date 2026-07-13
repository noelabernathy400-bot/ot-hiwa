"""Small modality-specific decoders used to prevent latent collapse."""

from __future__ import annotations

import torch
from torch import nn


class MLPDecoder(nn.Module):
    """Map a shared latent representation back to its original modality."""

    def __init__(self, latent_dimension: int, output_dimension: int, hidden_dimension: int = 32) -> None:
        super().__init__()
        if min(latent_dimension, output_dimension, hidden_dimension) <= 0:
            raise ValueError("decoder dimensions must be positive")
        self.network = nn.Sequential(
            nn.Linear(latent_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, output_dimension),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class ResidualMLPDecoder(nn.Module):
    """Identity-initialized residual decoder for equal-dimensional MVP inputs."""

    def __init__(self, latent_dimension: int, output_dimension: int, hidden_dimension: int = 32) -> None:
        super().__init__()
        if min(latent_dimension, output_dimension, hidden_dimension) <= 0:
            raise ValueError("decoder dimensions must be positive")
        self.skip = nn.Linear(latent_dimension, output_dimension, bias=False)
        with torch.no_grad():
            self.skip.weight.zero_()
            for index in range(min(latent_dimension, output_dimension)):
                self.skip.weight[index, index] = 1.0
        self.residual = nn.Sequential(
            nn.Linear(latent_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, output_dimension),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.skip(values) + self.residual(values)
