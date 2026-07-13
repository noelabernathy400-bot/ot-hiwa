"""Source-supervised representation learning around detached Soft-GCOT.

This module deliberately keeps the current NumPy Soft-GCOT solver as a
non-differentiable inner loop.  At every outer round, it solves transport on
detached latent representations.  The resulting coupling and rotation are
then treated as fixed evidence while encoders, decoders, prototypes, and a
*source-only* task head are updated.

The public ``fit`` method accepts source labels but intentionally has no
target-label or pair-ID argument.  Those quantities belong to evaluation
code, never to representation or alignment fitting.
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
except ImportError:  # Supports direct execution of experiment scripts.
    from solver_adapter import DetachedSoftGCOTSolution, solve_soft_gcot_detached


@dataclass(frozen=True)
class SourceSupervisedRepresentationConfig:
    n_groups: int = 4
    temperature: float = 0.50
    outer_steps: int = 3
    encoder_steps: int = 25
    source_warmup_steps: int = 100
    learning_rate: float = 1e-3
    lambda_alignment: float = 1.0
    lambda_reconstruction: float = 1.0
    lambda_task: float = 1.0
    lambda_variance: float = 1.0
    lambda_covariance: float = 0.05
    lambda_balance: float = 0.05
    lambda_identity: float = 0.10
    minimum_std: float = 0.25
    solver_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class SourceSupervisedRepresentationResult:
    solution: DetachedSoftGCOTSolution
    source_latent: np.ndarray
    target_latent: np.ndarray
    source_native_logits: np.ndarray
    source_logits: np.ndarray
    target_logits: np.ndarray
    history: list[dict[str, float]]


class SourceSupervisedAlternatingSoftGCOT:
    """Alternate detached OT solves with source-supervised representation updates."""

    def __init__(
        self,
        source_encoder: nn.Module,
        target_encoder: nn.Module,
        source_decoder: nn.Module,
        target_decoder: nn.Module,
        task_head: nn.Module,
        latent_dimension: int,
        config: SourceSupervisedRepresentationConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self.source_encoder = source_encoder
        self.target_encoder = target_encoder
        self.source_decoder = source_decoder
        self.target_decoder = target_decoder
        self.task_head = task_head
        self.latent_dimension = int(latent_dimension)
        self.config = config or SourceSupervisedRepresentationConfig()
        if self.latent_dimension <= 0 or self.config.n_groups <= 0:
            raise ValueError("latent_dimension and n_groups must be positive")
        if (
            self.config.temperature <= 0
            or self.config.outer_steps <= 0
            or self.config.encoder_steps <= 0
            or self.config.source_warmup_steps < 0
        ):
            raise ValueError("temperature and training step counts must be valid")
        self.device = torch.device(device)
        for module in self._modules:
            module.to(self.device)
        self.source_prototypes = nn.Parameter(torch.empty(self.config.n_groups, self.latent_dimension, device=self.device))
        self.target_prototypes = nn.Parameter(torch.empty(self.config.n_groups, self.latent_dimension, device=self.device))
        nn.init.orthogonal_(self.source_prototypes)
        nn.init.orthogonal_(self.target_prototypes)
        self.history: list[dict[str, float]] = []

    @property
    def _modules(self) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:
        return (
            self.source_encoder,
            self.target_encoder,
            self.source_decoder,
            self.target_decoder,
            self.task_head,
        )

    def _assignments(self, latent: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        return torch.softmax(latent @ prototypes.T / self.config.temperature, dim=1)

    def _balance_penalty(self, assignments: torch.Tensor) -> torch.Tensor:
        desired = torch.full_like(assignments.mean(dim=0), 1.0 / self.config.n_groups)
        return F.mse_loss(assignments.mean(dim=0), desired)

    def _optimizer(self) -> torch.optim.Optimizer:
        parameters: list[nn.Parameter] = [self.source_prototypes, self.target_prototypes]
        for module in self._modules:
            parameters.extend(module.parameters())
        return torch.optim.Adam(parameters, lr=self.config.learning_rate)

    def fit(
        self,
        source: np.ndarray,
        target: np.ndarray,
        source_labels: np.ndarray,
    ) -> SourceSupervisedRepresentationResult:
        """Fit using source labels only; target labels/pair IDs are evaluation-only."""
        source_tensor = torch.as_tensor(np.asarray(source, dtype=np.float32), device=self.device)
        target_tensor = torch.as_tensor(np.asarray(target, dtype=np.float32), device=self.device)
        labels = torch.as_tensor(np.asarray(source_labels, dtype=np.int64), device=self.device)
        if source_tensor.ndim != 2 or target_tensor.ndim != 2:
            raise ValueError("source and target inputs must be two-dimensional")
        if labels.ndim != 1 or labels.shape[0] != source_tensor.shape[0]:
            raise ValueError("source_labels must be a one-dimensional vector matching source samples")
        optimizer = self._optimizer()
        with torch.no_grad():
            source_anchor = self.source_encoder(source_tensor).detach()
            target_anchor = self.target_encoder(target_tensor).detach()
        # Establish a meaningful source task representation before its first
        # unlabeled transport solve.  Without this, the task head competes
        # from random initialization against reconstruction and alignment and
        # cannot provide a reliable semantic constraint.
        for warmup_step in range(self.config.source_warmup_steps):
            source_latent = self.source_encoder(source_tensor)
            source_task = F.cross_entropy(self.task_head(source_latent), labels)
            source_reconstruction = F.mse_loss(self.source_decoder(source_latent), source_tensor)
            source_variance = variance_floor_penalty(source_latent, self.config.minimum_std)
            source_covariance = covariance_penalty(source_latent)
            source_identity = F.mse_loss(source_latent, source_anchor)
            warmup_loss = (
                self.config.lambda_task * source_task
                + self.config.lambda_reconstruction * source_reconstruction
                + self.config.lambda_variance * source_variance
                + self.config.lambda_covariance * source_covariance
                + self.config.lambda_identity * source_identity
            )
            optimizer.zero_grad()
            warmup_loss.backward()
            optimizer.step()
            if warmup_step == self.config.source_warmup_steps - 1:
                self.history.append({
                    "outer_step": -1.0,
                    "loss": float(warmup_loss.detach().cpu()),
                    "source_task_loss": float(source_task.detach().cpu()),
                    "source_warmup_accuracy": float((self.task_head(source_latent).argmax(dim=1) == labels).float().mean().detach().cpu()),
                })
        solution: DetachedSoftGCOTSolution | None = None
        for outer_step in range(self.config.outer_steps):
            with torch.no_grad():
                source_latent = self.source_encoder(source_tensor)
                target_latent = self.target_encoder(target_tensor)
                if source_latent.shape[1] != self.latent_dimension or target_latent.shape[1] != self.latent_dimension:
                    raise ValueError("both encoders must return the declared latent_dimension")
                source_assignments = self._assignments(source_latent, self.source_prototypes)
                target_assignments = self._assignments(target_latent, self.target_prototypes)
                warm_start = {} if solution is None else {
                    "initial_rotation": solution.rotation,
                    "initial_transport": solution.group_transport,
                }
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
            for encoder_step in range(self.config.encoder_steps):
                source_latent = self.source_encoder(source_tensor)
                target_latent = self.target_encoder(target_tensor)
                source_assignments = self._assignments(source_latent, self.source_prototypes)
                target_assignments = self._assignments(target_latent, self.target_prototypes)
                aligned_source = source_latent @ fixed_rotation.T
                alignment = (fixed_pi * torch.cdist(aligned_source, target_latent).pow(2)).sum()
                reconstruction = (
                    F.mse_loss(self.source_decoder(source_latent), source_tensor)
                    + F.mse_loss(self.target_decoder(target_latent), target_tensor)
                )
                # The same source labels constrain both coordinate systems.
                # This prevents a detached rotation update from improving the
                # transport objective by erasing activity discrimination in
                # the source encoder's native latent space.
                task_native = F.cross_entropy(self.task_head(source_latent), labels)
                task_aligned = F.cross_entropy(self.task_head(aligned_source), labels)
                task = 0.5 * (task_native + task_aligned)
                variance = (
                    variance_floor_penalty(source_latent, self.config.minimum_std)
                    + variance_floor_penalty(target_latent, self.config.minimum_std)
                )
                covariance = covariance_penalty(source_latent) + covariance_penalty(target_latent)
                balance = self._balance_penalty(source_assignments) + self._balance_penalty(target_assignments)
                identity = F.mse_loss(source_latent, source_anchor) + F.mse_loss(target_latent, target_anchor)
                loss = (
                    self.config.lambda_alignment * alignment
                    + self.config.lambda_reconstruction * reconstruction
                    + self.config.lambda_task * task
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
                        "source_task_loss": float(task.detach().cpu()),
                        "source_native_task_loss": float(task_native.detach().cpu()),
                        "source_aligned_task_loss": float(task_aligned.detach().cpu()),
                        "variance_penalty": float(variance.detach().cpu()),
                        "covariance_penalty": float(covariance.detach().cpu()),
                        "balance_penalty": float(balance.detach().cpu()),
                        "identity_penalty": float(identity.detach().cpu()),
                        "solver_primal_residual": float(solution.diagnostics["admm_primal_residual"][-1]),
                    })
        with torch.no_grad():
            final_source_tensor = self.source_encoder(source_tensor)
            final_target_tensor = self.target_encoder(target_tensor)
            final_source_assignment = self._assignments(final_source_tensor, self.source_prototypes)
            final_target_assignment = self._assignments(final_target_tensor, self.target_prototypes)
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
            fixed_rotation = torch.as_tensor(solution.rotation, dtype=source_tensor.dtype, device=self.device)
            source_native_logits = self.task_head(final_source_tensor)
            source_logits = self.task_head(final_source_tensor @ fixed_rotation.T)
            target_logits = self.task_head(final_target_tensor)
        return SourceSupervisedRepresentationResult(
            solution=solution,
            source_latent=final_source_tensor.detach().cpu().numpy(),
            target_latent=final_target_tensor.detach().cpu().numpy(),
            source_native_logits=source_native_logits.detach().cpu().numpy(),
            source_logits=source_logits.detach().cpu().numpy(),
            target_logits=target_logits.detach().cpu().numpy(),
            history=list(self.history),
        )
