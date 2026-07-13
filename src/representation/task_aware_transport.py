"""Fully differentiable, source-task-aware hierarchical OT.

This does not call the detached NumPy Soft-GCOT solver.  It learns the two
encoders, the source task head, group transport and sample transport together.
Target labels and synchronous pair IDs are intentionally absent from ``fit``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .regularizers import covariance_penalty, variance_floor_penalty


def log_sinkhorn(cost: torch.Tensor, *, epsilon: float, iterations: int) -> torch.Tensor:
    """Differentiable entropic plan with uniform marginals, computed in log space."""
    if cost.ndim != 2 or min(cost.shape) <= 0:
        raise ValueError("cost must be a non-empty two-dimensional tensor")
    if epsilon <= 0 or iterations <= 0:
        raise ValueError("epsilon and iterations must be positive")
    n_source, n_target = cost.shape
    log_kernel = -cost / float(epsilon)
    log_source = torch.full((n_source,), -np.log(n_source), dtype=cost.dtype, device=cost.device)
    log_target = torch.full((n_target,), -np.log(n_target), dtype=cost.dtype, device=cost.device)
    log_u = torch.zeros_like(log_source)
    log_v = torch.zeros_like(log_target)
    for _ in range(iterations):
        log_u = log_source - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        log_v = log_target - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)
    return torch.exp(log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0))


@dataclass(frozen=True)
class TaskAwareTransportConfig:
    source_warmup_epochs: int = 100
    epochs: int = 200
    learning_rate: float = 2e-3
    group_epsilon: float = 0.10
    sample_epsilon: float = 0.15
    sinkhorn_iterations: int = 50
    lambda_task: float = 1.0
    lambda_reconstruction: float = 0.15
    lambda_transport: float = 1.0
    lambda_group: float = 0.25
    lambda_variance: float = 0.10
    lambda_covariance: float = 0.01
    lambda_group_semantic_prior: float = 0.50
    lambda_sample_semantic_cost: float = 0.25
    minimum_std: float = 0.20

    def validate(self) -> None:
        if self.epochs <= 0 or self.source_warmup_epochs < 0 or self.sinkhorn_iterations <= 0:
            raise ValueError("epochs and sinkhorn_iterations must be valid")
        if self.learning_rate <= 0 or self.group_epsilon <= 0 or self.sample_epsilon <= 0:
            raise ValueError("learning rate and Sinkhorn epsilons must be positive")


@dataclass
class TaskAwareTransportResult:
    source_latent: np.ndarray
    target_latent: np.ndarray
    target_logits: np.ndarray
    group_transport: np.ndarray
    sample_transport: np.ndarray
    history: list[dict[str, float]]
    best_epoch: int
    best_source_validation_accuracy: float


class DifferentiableTaskAwareOT:
    """End-to-end task-aware group and sample OT with source labels only."""

    def __init__(
        self,
        source_encoder: nn.Module,
        target_encoder: nn.Module,
        source_decoder: nn.Module,
        target_decoder: nn.Module,
        task_head: nn.Module,
        *,
        n_classes: int,
        config: TaskAwareTransportConfig | None = None,
        device: str = "cpu",
    ) -> None:
        if n_classes < 2:
            raise ValueError("n_classes must be at least two")
        self.source_encoder = source_encoder
        self.target_encoder = target_encoder
        self.source_decoder = source_decoder
        self.target_decoder = target_decoder
        self.task_head = task_head
        self.n_classes = int(n_classes)
        self.config = config or TaskAwareTransportConfig()
        self.config.validate()
        self.device = torch.device(device)
        for module in self._modules:
            module.to(self.device)
        self.history: list[dict[str, float]] = []

    @property
    def _modules(self) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:
        return self.source_encoder, self.target_encoder, self.source_decoder, self.target_decoder, self.task_head

    def _plans(
        self,
        source_latent: torch.Tensor,
        source_labels: torch.Tensor,
        target_latent: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        source_groups = F.one_hot(source_labels, num_classes=self.n_classes).to(source_latent.dtype)
        target_groups = torch.softmax(target_logits, dim=1)
        source_representatives = source_groups.T @ source_latent / source_groups.sum(dim=0).clamp_min(1e-8).unsqueeze(1)
        target_representatives = target_groups.T @ target_latent / target_groups.sum(dim=0).clamp_min(1e-8).unsqueeze(1)
        # OT operates on normalized latent directions, so transport costs stay
        # comparable to task cross-entropy throughout representation learning.
        # The task head still receives the unnormalized latent coordinates.
        group_geometry = torch.cdist(
            F.normalize(source_representatives, dim=1),
            F.normalize(target_representatives, dim=1),
        ).pow(2)
        off_diagonal = 1.0 - torch.eye(self.n_classes, dtype=group_geometry.dtype, device=group_geometry.device)
        group_plan = log_sinkhorn(
            group_geometry + self.config.lambda_group_semantic_prior * off_diagonal,
            epsilon=self.config.group_epsilon,
            iterations=self.config.sinkhorn_iterations,
        )
        compatibility = self.n_classes * (source_groups @ group_plan @ target_groups.T)
        semantic_cost = -torch.log(compatibility.clamp_min(1e-8))
        sample_geometry = torch.cdist(
            F.normalize(source_latent, dim=1),
            F.normalize(target_latent, dim=1),
        ).pow(2)
        sample_cost = sample_geometry + self.config.lambda_sample_semantic_cost * semantic_cost
        sample_plan = log_sinkhorn(sample_cost, epsilon=self.config.sample_epsilon, iterations=self.config.sinkhorn_iterations)
        return group_plan, sample_plan, group_geometry, sample_cost

    def _optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam([parameter for module in self._modules for parameter in module.parameters()], lr=self.config.learning_rate)

    def fit(
        self,
        source: np.ndarray,
        source_labels: np.ndarray,
        target_adaptation: np.ndarray,
        source_validation: np.ndarray,
        source_validation_labels: np.ndarray,
    ) -> TaskAwareTransportResult:
        """Fit without target labels, target-test samples, or pair IDs."""
        source_tensor = torch.as_tensor(np.asarray(source, dtype=np.float32), device=self.device)
        target_tensor = torch.as_tensor(np.asarray(target_adaptation, dtype=np.float32), device=self.device)
        validation_tensor = torch.as_tensor(np.asarray(source_validation, dtype=np.float32), device=self.device)
        source_label_tensor = torch.as_tensor(np.asarray(source_labels, dtype=np.int64), device=self.device)
        validation_label_tensor = torch.as_tensor(np.asarray(source_validation_labels, dtype=np.int64), device=self.device)
        if source_tensor.ndim != 2 or target_tensor.ndim != 2 or validation_tensor.ndim != 2:
            raise ValueError("all feature arrays must be two-dimensional")
        if source_tensor.shape[1] != target_tensor.shape[1] or source_tensor.shape[1] != validation_tensor.shape[1]:
            raise ValueError("feature dimensions must match")
        for labels, rows, name in ((source_label_tensor, source_tensor.shape[0], "source_labels"), (validation_label_tensor, validation_tensor.shape[0], "source_validation_labels")):
            if labels.ndim != 1 or labels.shape[0] != rows or labels.min() < 0 or labels.max() >= self.n_classes:
                raise ValueError(f"{name} must be zero-based class indices matching its rows")
        optimizer = self._optimizer()
        best_accuracy, best_epoch, best_state = float("-inf"), -1, None
        self.history = []
        # Establish task-discriminative source coordinates before the first
        # cross-view transport update.  The source-only control is the
        # justification for this phase; it uses no target data or labels.
        for warmup_epoch in range(self.config.source_warmup_epochs):
            source_latent = self.source_encoder(source_tensor)
            source_logits = self.task_head(source_latent)
            task = F.cross_entropy(source_logits, source_label_tensor)
            reconstruction = F.mse_loss(self.source_decoder(source_latent), source_tensor)
            variance = variance_floor_penalty(source_latent, self.config.minimum_std)
            covariance = covariance_penalty(source_latent)
            warmup_loss = self.config.lambda_task * task + self.config.lambda_reconstruction * reconstruction + self.config.lambda_variance * variance + self.config.lambda_covariance * covariance
            optimizer.zero_grad()
            warmup_loss.backward()
            optimizer.step()
            if warmup_epoch == 0 or warmup_epoch == self.config.source_warmup_epochs - 1:
                with torch.no_grad():
                    validation_logits = self.task_head(self.source_encoder(validation_tensor))
                    validation_accuracy = float((validation_logits.argmax(dim=1) == validation_label_tensor).float().mean().cpu())
                self.history.append({"epoch": float(-(self.config.source_warmup_epochs - warmup_epoch)), "loss": float(warmup_loss.detach().cpu()), "task_loss": float(task.detach().cpu()), "transport_loss": 0.0, "group_loss": 0.0, "source_validation_accuracy": validation_accuracy})
        for epoch in range(self.config.epochs):
            source_latent = self.source_encoder(source_tensor)
            target_latent = self.target_encoder(target_tensor)
            source_logits = self.task_head(source_latent)
            target_logits = self.task_head(target_latent)
            group_plan, sample_plan, group_geometry, sample_cost = self._plans(source_latent, source_label_tensor, target_latent, target_logits)
            task = F.cross_entropy(source_logits, source_label_tensor)
            reconstruction = F.mse_loss(self.source_decoder(source_latent), source_tensor) + F.mse_loss(self.target_decoder(target_latent), target_tensor)
            transport = torch.sum(sample_plan * sample_cost)
            group = torch.sum(group_plan * group_geometry)
            variance = variance_floor_penalty(source_latent, self.config.minimum_std) + variance_floor_penalty(target_latent, self.config.minimum_std)
            covariance = covariance_penalty(source_latent) + covariance_penalty(target_latent)
            loss = self.config.lambda_task * task + self.config.lambda_reconstruction * reconstruction + self.config.lambda_transport * transport + self.config.lambda_group * group + self.config.lambda_variance * variance + self.config.lambda_covariance * covariance
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                validation_logits = self.task_head(self.source_encoder(validation_tensor))
                validation_accuracy = float((validation_logits.argmax(dim=1) == validation_label_tensor).float().mean().cpu())
            if validation_accuracy > best_accuracy:
                best_accuracy, best_epoch = validation_accuracy, epoch
                best_state = {name: deepcopy(module.state_dict()) for name, module in zip(("source_encoder", "target_encoder", "source_decoder", "target_decoder", "task_head"), self._modules)}
            if epoch == 0 or epoch == self.config.epochs - 1 or (epoch + 1) % 25 == 0:
                self.history.append({"epoch": float(epoch), "loss": float(loss.detach().cpu()), "task_loss": float(task.detach().cpu()), "transport_loss": float(transport.detach().cpu()), "group_loss": float(group.detach().cpu()), "source_validation_accuracy": validation_accuracy})
        if best_state is None:
            raise RuntimeError("no task-aware checkpoint was produced")
        for name, module in zip(("source_encoder", "target_encoder", "source_decoder", "target_decoder", "task_head"), self._modules):
            module.load_state_dict(best_state[name])
        with torch.no_grad():
            source_latent = self.source_encoder(source_tensor)
            target_latent = self.target_encoder(target_tensor)
            target_logits = self.task_head(target_latent)
            group_plan, sample_plan, _, _ = self._plans(source_latent, source_label_tensor, target_latent, target_logits)
        return TaskAwareTransportResult(source_latent.cpu().numpy(), target_latent.cpu().numpy(), target_logits.cpu().numpy(), group_plan.cpu().numpy(), sample_plan.cpu().numpy(), list(self.history), best_epoch, best_accuracy)

    def encode_source(self, source: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.source_encoder(torch.as_tensor(np.asarray(source, dtype=np.float32), device=self.device)).cpu().numpy()

    def encode_target(self, target: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.target_encoder(torch.as_tensor(np.asarray(target, dtype=np.float32), device=self.device)).cpu().numpy()

    def predict_target(self, target: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            values = torch.as_tensor(np.asarray(target, dtype=np.float32), device=self.device)
            return self.task_head(self.target_encoder(values)).argmax(dim=1).cpu().numpy()
