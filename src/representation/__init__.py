"""Learnable representation components for alternating Soft-GCOT MVPs."""

from .decoders import MLPDecoder, ResidualMLPDecoder
from .encoders import MLPEncoder, ResidualMLPEncoder
from .regularizers import covariance_penalty, variance_floor_penalty
from .task_aware_transport import DifferentiableTaskAwareOT, TaskAwareTransportConfig, TaskAwareTransportResult

__all__ = [
    "MLPDecoder",
    "MLPEncoder",
    "ResidualMLPDecoder",
    "ResidualMLPEncoder",
    "DifferentiableTaskAwareOT",
    "TaskAwareTransportConfig",
    "TaskAwareTransportResult",
    "covariance_penalty",
    "variance_floor_penalty",
]
