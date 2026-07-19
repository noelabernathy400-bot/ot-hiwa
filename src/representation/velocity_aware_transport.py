"""Differentiable source-velocity / target-unlabelled hierarchical OT."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .regularizers import covariance_penalty, variance_floor_penalty
from .task_aware_transport import log_sinkhorn


@dataclass(frozen=True)
class VelocityAwareTransportConfig:
    source_warmup_epochs: int = 50
    epochs: int = 100
    learning_rate: float = 2e-3
    group_epsilon: float = 0.10
    sample_epsilon: float = 0.15
    sinkhorn_iterations: int = 20
    lambda_velocity: float = 1.0
    lambda_reconstruction: float = 0.15
    lambda_transport: float = 1.0
    lambda_group: float = 0.25
    lambda_variance: float = 0.10
    lambda_covariance: float = 0.01
    lambda_group_semantic_prior: float = 0.50
    lambda_sample_semantic_cost: float = 0.25
    minimum_std: float = 0.20
    freeze_source_after_warmup: bool = True


@dataclass
class VelocityAwareTransportResult:
    source_latent: np.ndarray
    target_latent: np.ndarray
    target_velocity: np.ndarray
    group_transport: np.ndarray
    sample_transport: np.ndarray
    history: list[dict[str, float]]
    best_epoch: int
    best_source_validation_mse: float


class DifferentiableVelocityAwareOT:
    """Learn a source velocity representation and transfer it without target velocity.

    Source direction labels define interpretable source groups for hierarchical
    OT.  The target grouping head is learned solely from target neural inputs
    and the OT objective; target velocity is never accepted by ``fit``.
    """

    def __init__(
        self,
        source_encoder: nn.Module,
        target_encoder: nn.Module,
        source_decoder: nn.Module,
        target_decoder: nn.Module,
        velocity_head: nn.Module,
        target_group_head: nn.Module,
        *,
        n_groups: int,
        config: VelocityAwareTransportConfig | None = None,
        device: str = "cpu",
    ) -> None:
        if n_groups < 2:
            raise ValueError("n_groups must be at least two")
        self.source_encoder = source_encoder
        self.target_encoder = target_encoder
        self.source_decoder = source_decoder
        self.target_decoder = target_decoder
        self.velocity_head = velocity_head
        self.target_group_head = target_group_head
        self.n_groups = int(n_groups)
        self.config = config or VelocityAwareTransportConfig()
        self.device = torch.device(device)
        for module in self._modules:
            module.to(self.device)
        self.history: list[dict[str, float]] = []

    @property
    def _modules(self):
        return (
            self.source_encoder,
            self.target_encoder,
            self.source_decoder,
            self.target_decoder,
            self.velocity_head,
            self.target_group_head,
        )

    def _plans(self, source_latent, source_groups, target_latent):
        target_groups = torch.softmax(self.target_group_head(target_latent), dim=1)
        source_rep = source_groups.T @ source_latent / source_groups.sum(dim=0).clamp_min(1e-8).unsqueeze(1)
        target_rep = target_groups.T @ target_latent / target_groups.sum(dim=0).clamp_min(1e-8).unsqueeze(1)
        group_geometry = torch.cdist(F.normalize(source_rep, dim=1), F.normalize(target_rep, dim=1)).pow(2)
        off_diagonal = 1.0 - torch.eye(self.n_groups, dtype=group_geometry.dtype, device=self.device)
        group_transport = log_sinkhorn(
            group_geometry + self.config.lambda_group_semantic_prior * off_diagonal,
            epsilon=self.config.group_epsilon,
            iterations=self.config.sinkhorn_iterations,
        )
        compatibility = self.n_groups * (source_groups @ group_transport @ target_groups.T)
        semantic_cost = -torch.log(compatibility.clamp_min(1e-8))
        geometry = torch.cdist(F.normalize(source_latent, dim=1), F.normalize(target_latent, dim=1)).pow(2)
        sample_cost = geometry + self.config.lambda_sample_semantic_cost * semantic_cost
        sample_transport = log_sinkhorn(sample_cost, epsilon=self.config.sample_epsilon, iterations=self.config.sinkhorn_iterations)
        return group_transport, sample_transport, group_geometry, sample_cost

    def fit(self, source, source_velocity, source_groups, target_adaptation, source_validation, source_validation_velocity):
        source_t = torch.as_tensor(np.asarray(source, dtype=np.float32), device=self.device)
        target_t = torch.as_tensor(np.asarray(target_adaptation, dtype=np.float32), device=self.device)
        validation_t = torch.as_tensor(np.asarray(source_validation, dtype=np.float32), device=self.device)
        velocity_t = torch.as_tensor(np.asarray(source_velocity, dtype=np.float32), device=self.device)
        validation_velocity_t = torch.as_tensor(np.asarray(source_validation_velocity, dtype=np.float32), device=self.device)
        group_t = torch.as_tensor(np.asarray(source_groups, dtype=np.int64), device=self.device)
        if group_t.ndim != 1 or group_t.shape[0] != source_t.shape[0] or group_t.min() < 0 or group_t.max() >= self.n_groups:
            raise ValueError("source_groups must be group indices matching source rows")
        source_group_t = F.one_hot(group_t, num_classes=self.n_groups).to(source_t.dtype)
        source_modules = (self.source_encoder, self.source_decoder, self.velocity_head)
        target_modules = (self.target_encoder, self.target_decoder, self.target_group_head)
        optimizer = torch.optim.Adam([p for module in self._modules for p in module.parameters()], lr=self.config.learning_rate)
        best_mse, best_epoch, best_state = float("inf"), -1, None
        self.history = []
        for epoch in range(self.config.source_warmup_epochs + self.config.epochs):
            if epoch == self.config.source_warmup_epochs and self.config.freeze_source_after_warmup:
                optimizer = torch.optim.Adam(
                    [p for module in target_modules for p in module.parameters()],
                    lr=self.config.learning_rate,
                )
            source_latent = self.source_encoder(source_t)
            source_velocity_pred = self.velocity_head(source_latent)
            velocity_loss = F.mse_loss(source_velocity_pred, velocity_t)
            reconstruction = F.mse_loss(self.source_decoder(source_latent), source_t)
            variance = variance_floor_penalty(source_latent, self.config.minimum_std)
            covariance = covariance_penalty(source_latent)
            transport = torch.zeros((), device=self.device)
            group = torch.zeros((), device=self.device)
            if epoch >= self.config.source_warmup_epochs:
                target_latent = self.target_encoder(target_t)
                group_plan, sample_plan, group_geometry, sample_cost = self._plans(source_latent, source_group_t, target_latent)
                transport = torch.sum(sample_plan * sample_cost)
                group = torch.sum(group_plan * group_geometry)
                reconstruction = reconstruction + F.mse_loss(self.target_decoder(target_latent), target_t)
                variance = variance + variance_floor_penalty(target_latent, self.config.minimum_std)
                covariance = covariance + covariance_penalty(target_latent)
            loss = (
                self.config.lambda_velocity * velocity_loss
                + self.config.lambda_reconstruction * reconstruction
                + self.config.lambda_transport * transport
                + self.config.lambda_group * group
                + self.config.lambda_variance * variance
                + self.config.lambda_covariance * covariance
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                validation_mse = float(F.mse_loss(self.velocity_head(self.source_encoder(validation_t)), validation_velocity_t).cpu())
            if validation_mse <= best_mse:
                best_mse, best_epoch = validation_mse, epoch
                best_state = {name: deepcopy(module.state_dict()) for name, module in zip(("source_encoder", "target_encoder", "source_decoder", "target_decoder", "velocity_head", "target_group_head"), self._modules)}
            if epoch in (0, self.config.source_warmup_epochs - 1, self.config.source_warmup_epochs, self.config.source_warmup_epochs + self.config.epochs - 1):
                self.history.append({"epoch": float(epoch), "loss": float(loss.detach().cpu()), "velocity_loss": float(velocity_loss.detach().cpu()), "transport_loss": float(transport.detach().cpu()), "source_validation_mse": validation_mse})
        for name, module in zip(("source_encoder", "target_encoder", "source_decoder", "target_decoder", "velocity_head", "target_group_head"), self._modules):
            module.load_state_dict(best_state[name])
        with torch.no_grad():
            source_latent = self.source_encoder(source_t)
            target_latent = self.target_encoder(target_t)
            group_plan, sample_plan, _, _ = self._plans(source_latent, source_group_t, target_latent)
            target_velocity = self.velocity_head(target_latent)
        return VelocityAwareTransportResult(source_latent.cpu().numpy(), target_latent.cpu().numpy(), target_velocity.cpu().numpy(), group_plan.cpu().numpy(), sample_plan.cpu().numpy(), self.history, best_epoch, best_mse)

    def predict_target_velocity(self, target: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            values = torch.as_tensor(np.asarray(target, dtype=np.float32), device=self.device)
            return self.velocity_head(self.target_encoder(values)).cpu().numpy()
