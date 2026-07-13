"""Small modality-specific encoders used by the first representation MVP."""

from __future__ import annotations

import torch
from torch import nn


class MLPEncoder(nn.Module):
    """A deliberately small MLP encoder with a shared latent output dimension."""

    def __init__(self, input_dimension: int, latent_dimension: int, hidden_dimension: int = 32) -> None:
        super().__init__()
        if min(input_dimension, latent_dimension, hidden_dimension) <= 0:
            raise ValueError("encoder dimensions must be positive")
        self.network = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, latent_dimension),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class ResidualMLPEncoder(nn.Module):
    """Identity-initialized residual encoder for controlled baseline adaptation.

    When input and latent dimensions agree, the initial representation is
    exactly the supplied feature matrix.  The residual branch starts at zero,
    so any later change is attributable to the alternating objective rather
    than an arbitrary random projection.
    """

    def __init__(self, input_dimension: int, latent_dimension: int, hidden_dimension: int = 32) -> None:
        super().__init__()
        if min(input_dimension, latent_dimension, hidden_dimension) <= 0:
            raise ValueError("encoder dimensions must be positive")
        self.skip = nn.Linear(input_dimension, latent_dimension, bias=False)
        with torch.no_grad():
            self.skip.weight.zero_()
            for index in range(min(input_dimension, latent_dimension)):
                self.skip.weight[index, index] = 1.0
        self.residual = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, latent_dimension),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.skip(values) + self.residual(values)
