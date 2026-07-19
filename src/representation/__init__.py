"""Learnable representation components for alternating Soft-GCOT MVPs."""

from .decoders import MLPDecoder, ResidualMLPDecoder
from .encoders import MLPEncoder, ResidualMLPEncoder
from .regularizers import covariance_penalty, variance_floor_penalty
from .task_aware_transport import DifferentiableTaskAwareOT, TaskAwareTransportConfig, TaskAwareTransportResult
from .velocity_aware_transport import DifferentiableVelocityAwareOT, VelocityAwareTransportConfig, VelocityAwareTransportResult

__all__ = [
    "MLPDecoder",
    "MLPEncoder",
    "ResidualMLPDecoder",
    "ResidualMLPEncoder",
    "DifferentiableTaskAwareOT",
    "TaskAwareTransportConfig",
    "TaskAwareTransportResult",
    "DifferentiableVelocityAwareOT",
    "VelocityAwareTransportConfig",
    "VelocityAwareTransportResult",
    "covariance_penalty",
    "variance_floor_penalty",
]
