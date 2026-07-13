"""Alternating learnable-representation wrapper around the frozen Soft-GCOT solver.

This is intentionally *not* a claim of end-to-end differentiation through
Sinkhorn or ADMM.  Each outer round first solves the existing full-support
Soft-GCOT problem on detached latent representations.  It then treats the
resulting ``Pi`` and ``R`` as fixed evidence while updating encoders,
decoders, and soft prototypes.  The next round recomputes transport plans.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from representation.regularizers import covariance_penalty, variance_floor_penalty

try:
    from .solver_adapter import DetachedSoftGCOTSolution, solve_soft_gcot_detached
except ImportError:  # Supports direct execution of legacy experiment scripts.
    from solver_adapter import DetachedSoftGCOTSolution, solve_soft_gcot_detached


@dataclass(frozen=True)
class AlternatingRepresentationConfig:
    n_groups: int = 4
    temperature: float = 0.50
    outer_steps: int = 3
    encoder_steps: int = 25
    learning_rate: float = 1e-3
    lambda_alignment: float = 1.0
    lambda_reconstruction: float = 1.0
    lambda_cross_reconstruction: float = 1.0
    lambda_variance: float = 1.0
    lambda_covariance: float = 0.05
    lambda_balance: float = 0.05
    lambda_identity: float = 1.0
    minimum_std: float = 0.25
    solver_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class AlternatingRepresentationResult:
    solution: DetachedSoftGCOTSolution
    source_latent: np.ndarray
    target_latent: np.ndarray
    history: list[dict[str, float]]


class AlternatingRepresentationSoftGCOT:
    """Learn representations while preserving the existing solver as an inner loop."""

    def __init__(
        self,
        source_encoder: nn.Module,
        target_encoder: nn.Module,
        source_decoder: nn.Module,
        target_decoder: nn.Module,
        latent_dimension: int,
        config: AlternatingRepresentationConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self.source_encoder = source_encoder
        self.target_encoder = target_encoder
        self.source_decoder = source_decoder
        self.target_decoder = target_decoder
        self.latent_dimension = int(latent_dimension)
        self.config = config or AlternatingRepresentationConfig()
        if self.latent_dimension <= 0 or self.config.n_groups <= 0:
            raise ValueError("latent_dimension and n_groups must be positive")
        if self.config.temperature <= 0 or self.config.outer_steps <= 0 or self.config.encoder_steps <= 0:
            raise ValueError("temperature, outer_steps, and encoder_steps must be positive")
        if min(
            self.config.lambda_alignment,
            self.config.lambda_reconstruction,
            self.config.lambda_cross_reconstruction,
            self.config.lambda_variance,
            self.config.lambda_covariance,
            self.config.lambda_balance,
            self.config.lambda_identity,
        ) < 0:
            raise ValueError("representation loss weights must be non-negative")
        self.device = torch.device(device)
        for module in self._modules:
            module.to(self.device)
        self.source_prototypes = nn.Parameter(torch.empty(self.config.n_groups, self.latent_dimension, device=self.device))
        self.target_prototypes = nn.Parameter(torch.empty(self.config.n_groups, self.latent_dimension, device=self.device))
        nn.init.orthogonal_(self.source_prototypes)
        nn.init.orthogonal_(self.target_prototypes)
        self.history: list[dict[str, float]] = []

    @property
    def _modules(self) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module]:
        return self.source_encoder, self.target_encoder, self.source_decoder, self.target_decoder

    def _assignments(self, latent: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        scores = latent @ prototypes.T / self.config.temperature
        return torch.softmax(scores, dim=1)

    def _balance_penalty(self, assignments: torch.Tensor) -> torch.Tensor:
        desired = torch.full_like(assignments.mean(dim=0), 1.0 / self.config.n_groups)
        return F.mse_loss(assignments.mean(dim=0), desired)

    def _optimizer(self) -> torch.optim.Optimizer:
        parameters: list[nn.Parameter] = [self.source_prototypes, self.target_prototypes]
        for module in self._modules:
            parameters.extend(module.parameters())
        return torch.optim.Adam(parameters, lr=self.config.learning_rate)

    def fit(self, source: np.ndarray, target: np.ndarray) -> AlternatingRepresentationResult:
        source_tensor = torch.as_tensor(np.asarray(source, dtype=np.float32), device=self.device)
        target_tensor = torch.as_tensor(np.asarray(target, dtype=np.float32), device=self.device)
        if source_tensor.ndim != 2 or target_tensor.ndim != 2:
            raise ValueError("source and target inputs must be two-dimensional")
        optimizer = self._optimizer()
        # These anchors are unlabeled initial representations.  They make the
        # first representation update a controlled deformation of the frozen
        # input geometry rather than an unconstrained re-embedding.
        with torch.no_grad():
            source_anchor = self.source_encoder(source_tensor).detach()
            target_anchor = self.target_encoder(target_tensor).detach()
        solution: DetachedSoftGCOTSolution | None = None
        for outer_step in range(self.config.outer_steps):
            with torch.no_grad():
                source_latent = self.source_encoder(source_tensor)
                target_latent = self.target_encoder(target_tensor)
                if source_latent.shape[1] != self.latent_dimension or target_latent.shape[1] != self.latent_dimension:
                    raise ValueError("both encoders must return the declared latent_dimension")
                source_assignments = self._assignments(source_latent, self.source_prototypes)
                target_assignments = self._assignments(target_latent, self.target_prototypes)
                warm_start = (
                    {}
                    if solution is None
                    else {
                        "initial_rotation": solution.rotation,
                        "initial_transport": solution.group_transport,
                    }
                )
                solution = solve_soft_gcot_detached(
                    source_latent,
                    source_assignments,
                    target_latent,
                    target_assignments,
                    solver_kwargs=self.config.solver_kwargs,
                    fit_kwargs=warm_start,
                )
            fixed_pi = torch.as_tensor(solution.sample_coupling, dtype=source_tensor.dtype, device=self.device)
            fixed_rotation = torch.as_tensor(solution.rotation, dtype=source_tensor.dtype, device=self.device)
            target_barycenter = fixed_pi @ target_tensor / fixed_pi.sum(dim=1, keepdim=True).clamp_min(1e-12)
            source_barycenter = fixed_pi.T @ source_tensor / fixed_pi.sum(dim=0, keepdim=True).T.clamp_min(1e-12)
            for encoder_step in range(self.config.encoder_steps):
                source_latent = self.source_encoder(source_tensor)
                target_latent = self.target_encoder(target_tensor)
                source_assignments = self._assignments(source_latent, self.source_prototypes)
                target_assignments = self._assignments(target_latent, self.target_prototypes)
                aligned_source = source_latent @ fixed_rotation.T
                pair_cost = torch.cdist(aligned_source, target_latent).pow(2)
                alignment = (fixed_pi * pair_cost).sum()
                reconstruction = F.mse_loss(self.source_decoder(source_latent), source_tensor) + F.mse_loss(self.target_decoder(target_latent), target_tensor)
                cross_reconstruction = (
                    F.mse_loss(self.target_decoder(aligned_source), target_barycenter)
                    + F.mse_loss(self.source_decoder(target_latent @ fixed_rotation), source_barycenter)
                )
                variance = variance_floor_penalty(source_latent, self.config.minimum_std) + variance_floor_penalty(target_latent, self.config.minimum_std)
                covariance = covariance_penalty(source_latent) + covariance_penalty(target_latent)
                balance = self._balance_penalty(source_assignments) + self._balance_penalty(target_assignments)
                identity = F.mse_loss(source_latent, source_anchor) + F.mse_loss(target_latent, target_anchor)
                loss = (
                    self.config.lambda_alignment * alignment
                    + self.config.lambda_reconstruction * reconstruction
                    + self.config.lambda_cross_reconstruction * cross_reconstruction
                    + self.config.lambda_variance * variance
                    + self.config.lambda_covariance * covariance
                    + self.config.lambda_balance * balance
                    + self.config.lambda_identity * identity
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if encoder_step == self.config.encoder_steps - 1:
                    self.history.append({
                        "outer_step": float(outer_step),
                        "loss": float(loss.detach().cpu()),
                        "alignment": float(alignment.detach().cpu()),
                        "reconstruction": float(reconstruction.detach().cpu()),
                        "cross_reconstruction": float(cross_reconstruction.detach().cpu()),
                        "variance_penalty": float(variance.detach().cpu()),
                        "covariance_penalty": float(covariance.detach().cpu()),
                        "balance_penalty": float(balance.detach().cpu()),
                        "identity_penalty": float(identity.detach().cpu()),
                        "source_effective_rank": self._effective_rank(source_latent),
                        "target_effective_rank": self._effective_rank(target_latent),
                        "source_assignment_entropy": self._mean_entropy(source_assignments),
                        "target_assignment_entropy": self._mean_entropy(target_assignments),
                        "solver_primal_residual": float(solution.diagnostics["admm_primal_residual"][-1]),
                    })
        with torch.no_grad():
            final_source_tensor = self.source_encoder(source_tensor)
            final_target_tensor = self.target_encoder(target_tensor)
            final_source_assignment = self._assignments(final_source_tensor, self.source_prototypes)
            final_target_assignment = self._assignments(final_target_tensor, self.target_prototypes)
            # Re-solve once after the final encoder update so returned plans
            # always correspond to returned latent representations.
            solution = solve_soft_gcot_detached(
                final_source_tensor,
                final_source_assignment,
                final_target_tensor,
                final_target_assignment,
                solver_kwargs=self.config.solver_kwargs,
                fit_kwargs={
                    "initial_rotation": solution.rotation,
                    "initial_transport": solution.group_transport,
                },
            )
            final_source = final_source_tensor.detach().cpu().numpy()
            final_target = final_target_tensor.detach().cpu().numpy()
        return AlternatingRepresentationResult(solution, final_source, final_target, list(self.history))

    @staticmethod
    def _mean_entropy(assignments: torch.Tensor) -> float:
        entropy = -(assignments * torch.log(assignments.clamp_min(1e-12))).sum(dim=1).mean()
        return float(entropy.detach().cpu())

    @staticmethod
    def _effective_rank(values: torch.Tensor) -> float:
        centered = values - values.mean(dim=0, keepdim=True)
        singular = torch.linalg.svdvals(centered)
        probability = singular / singular.sum().clamp_min(1e-12)
        entropy = -(probability * torch.log(probability.clamp_min(1e-12))).sum()
        return float(torch.exp(entropy).detach().cpu())
