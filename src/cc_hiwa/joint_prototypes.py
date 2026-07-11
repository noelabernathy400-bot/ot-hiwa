from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

try:
    from .soft_groups import assignment_entropy, assignments_from_prototypes
    from .soft_hiwa import sinkhorn_groups
except ImportError:  # Supports direct execution of legacy experiment scripts.
    from soft_groups import assignment_entropy, assignments_from_prototypes
    from soft_hiwa import sinkhorn_groups


EPS = 1e-12


@dataclass(frozen=True)
class JointPrototypeConfig:
    n_groups: int = 4
    temperature: float = 0.50
    lambda_align: float = 0.10
    lambda_balance: float = 0.05
    lambda_entropy: float = 0.10
    group_gamma: float = 0.10
    sinkhorn_maxiter: int = 200
    maxiter: int = 80
    gtol: float = 1e-5


@dataclass
class JointPrototypeResult:
    source_prototypes_standardized: np.ndarray
    target_prototypes_standardized: np.ndarray
    source_assignments: np.ndarray
    target_assignments: np.ndarray
    source_representatives: np.ndarray
    target_representatives: np.ndarray
    group_transport: np.ndarray
    initial_loss: dict[str, float]
    final_loss: dict[str, float]
    diagnostics: dict[str, Any]


def standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=float)
    mean = array.mean(axis=0, keepdims=True)
    scale = array.std(axis=0, keepdims=True)
    scale = np.where(scale < EPS, 1.0, scale)
    return (array - mean) / scale, mean, scale


def representatives(values: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    weights = np.asarray(assignments, dtype=float)
    masses = np.maximum(weights.sum(axis=0), EPS)
    return (weights.T @ np.asarray(values, dtype=float)) / masses[:, None]


def _validate_inputs(
    source: np.ndarray,
    target: np.ndarray,
    rotation: np.ndarray,
    config: JointPrototypeConfig,
) -> None:
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source and target must be 2D arrays")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target must have the same feature dimension")
    if rotation.shape != (source.shape[1], source.shape[1]):
        raise ValueError("rotation must be a square matrix matching feature dimension")
    if config.n_groups <= 1:
        raise ValueError("n_groups must be greater than one")
    if config.temperature <= 0:
        raise ValueError("temperature must be positive")
    if config.group_gamma <= 0:
        raise ValueError("group_gamma must be positive")
    for name in ("lambda_align", "lambda_balance", "lambda_entropy"):
        if getattr(config, name) < 0:
            raise ValueError(f"{name} must be non-negative")


def _unpack(
    flat: np.ndarray,
    n_groups: int,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    split = n_groups * dimension
    return flat[:split].reshape(n_groups, dimension), flat[split:].reshape(
        n_groups, dimension
    )


def _loss_terms(
    flat: np.ndarray,
    *,
    source: np.ndarray,
    target: np.ndarray,
    source_standardized: np.ndarray,
    target_standardized: np.ndarray,
    rotation: np.ndarray,
    config: JointPrototypeConfig,
    target_entropy: float,
) -> tuple[float, dict[str, float], dict[str, np.ndarray]]:
    source_proto, target_proto = _unpack(
        flat,
        config.n_groups,
        source_standardized.shape[1],
    )
    source_assignments = assignments_from_prototypes(
        source_standardized,
        source_proto,
        config.temperature,
    )
    target_assignments = assignments_from_prototypes(
        target_standardized,
        target_proto,
        config.temperature,
    )

    source_reconstruction = source_assignments @ source_proto
    target_reconstruction = target_assignments @ target_proto
    proto_source = float(np.mean((source_standardized - source_reconstruction) ** 2))
    proto_target = float(np.mean((target_standardized - target_reconstruction) ** 2))

    source_reps = representatives(source, source_assignments)
    target_reps = representatives(target, target_assignments)
    rotated_source_reps = (rotation @ source_reps.T).T
    group_cost = np.sum(
        (rotated_source_reps[:, None, :] - target_reps[None, :, :]) ** 2,
        axis=2,
    )
    group_transport = sinkhorn_groups(
        group_cost,
        gamma=config.group_gamma,
        maxiter=config.sinkhorn_maxiter,
    )
    align = float(np.sum(group_transport * group_cost))

    source_mass = source_assignments.mean(axis=0)
    target_mass = target_assignments.mean(axis=0)
    uniform = np.full(config.n_groups, 1.0 / config.n_groups)
    balance = float(np.sum((source_mass - uniform) ** 2) + np.sum((target_mass - uniform) ** 2))

    source_entropy = float(np.mean(assignment_entropy(source_assignments)))
    target_entropy_mean = float(np.mean(assignment_entropy(target_assignments)))
    entropy = float((source_entropy - target_entropy) ** 2 + (target_entropy_mean - target_entropy) ** 2)

    total = (
        proto_source
        + proto_target
        + config.lambda_align * align
        + config.lambda_balance * balance
        + config.lambda_entropy * entropy
    )
    terms = {
        "total": float(total),
        "proto_source": proto_source,
        "proto_target": proto_target,
        "align": align,
        "balance": balance,
        "entropy": entropy,
        "source_mean_entropy": source_entropy,
        "target_mean_entropy": target_entropy_mean,
        "source_min_mass": float(source_mass.min()),
        "target_min_mass": float(target_mass.min()),
        "source_max_mass": float(source_mass.max()),
        "target_max_mass": float(target_mass.max()),
        "group_transport_marginal_error": float(
            max(
                np.abs(group_transport.sum(axis=1) - uniform).max(),
                np.abs(group_transport.sum(axis=0) - uniform).max(),
            )
        ),
    }
    arrays = {
        "source_assignments": source_assignments,
        "target_assignments": target_assignments,
        "source_representatives": source_reps,
        "target_representatives": target_reps,
        "group_transport": group_transport,
    }
    return float(total), terms, arrays


def optimize_joint_prototypes(
    source: np.ndarray,
    target: np.ndarray,
    *,
    rotation: np.ndarray,
    initial_source_prototypes_standardized: np.ndarray,
    initial_target_prototypes_standardized: np.ndarray,
    target_entropy: float,
    config: JointPrototypeConfig | None = None,
) -> JointPrototypeResult:
    """Optimize soft prototypes under a fixed alignment-aware representative loss.

    Assignments are generated in each domain's standardized feature space to match
    the existing Soft-Prototype HiWA initialization. Representatives are computed
    in the original fixed embedding space used by Soft-HiWA.
    """
    cfg = config or JointPrototypeConfig()
    source_array = np.asarray(source, dtype=float)
    target_array = np.asarray(target, dtype=float)
    rotation_array = np.asarray(rotation, dtype=float)
    _validate_inputs(source_array, target_array, rotation_array, cfg)

    source_standardized, _, _ = standardize(source_array)
    target_standardized, _, _ = standardize(target_array)
    initial_source = np.asarray(initial_source_prototypes_standardized, dtype=float)
    initial_target = np.asarray(initial_target_prototypes_standardized, dtype=float)
    expected_shape = (cfg.n_groups, source_array.shape[1])
    if initial_source.shape != expected_shape or initial_target.shape != expected_shape:
        raise ValueError(
            "initial prototypes must have shape "
            f"{expected_shape}; got {initial_source.shape} and {initial_target.shape}"
        )

    x0 = np.concatenate([initial_source.ravel(), initial_target.ravel()])
    initial_total, initial_terms, _ = _loss_terms(
        x0,
        source=source_array,
        target=target_array,
        source_standardized=source_standardized,
        target_standardized=target_standardized,
        rotation=rotation_array,
        config=cfg,
        target_entropy=float(target_entropy),
    )

    def objective(flat: np.ndarray) -> float:
        value, _, _ = _loss_terms(
            flat,
            source=source_array,
            target=target_array,
            source_standardized=source_standardized,
            target_standardized=target_standardized,
            rotation=rotation_array,
            config=cfg,
            target_entropy=float(target_entropy),
        )
        return value

    optimized = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        options={"maxiter": cfg.maxiter, "gtol": cfg.gtol},
    )
    final_total, final_terms, arrays = _loss_terms(
        optimized.x,
        source=source_array,
        target=target_array,
        source_standardized=source_standardized,
        target_standardized=target_standardized,
        rotation=rotation_array,
        config=cfg,
        target_entropy=float(target_entropy),
    )
    source_proto, target_proto = _unpack(
        optimized.x,
        cfg.n_groups,
        source_array.shape[1],
    )
    diagnostics = {
        "success": bool(optimized.success),
        "message": str(optimized.message),
        "iterations": int(getattr(optimized, "nit", 0)),
        "function_evaluations": int(getattr(optimized, "nfev", 0)),
        "initial_total": float(initial_total),
        "final_total": float(final_total),
        "relative_total_change": float((final_total - initial_total) / max(abs(initial_total), EPS)),
        "target_entropy": float(target_entropy),
        "configuration": {
            "n_groups": cfg.n_groups,
            "temperature": cfg.temperature,
            "lambda_align": cfg.lambda_align,
            "lambda_balance": cfg.lambda_balance,
            "lambda_entropy": cfg.lambda_entropy,
            "group_gamma": cfg.group_gamma,
            "sinkhorn_maxiter": cfg.sinkhorn_maxiter,
            "maxiter": cfg.maxiter,
            "gtol": cfg.gtol,
        },
    }
    return JointPrototypeResult(
        source_prototypes_standardized=source_proto,
        target_prototypes_standardized=target_proto,
        source_assignments=arrays["source_assignments"],
        target_assignments=arrays["target_assignments"],
        source_representatives=arrays["source_representatives"],
        target_representatives=arrays["target_representatives"],
        group_transport=arrays["group_transport"],
        initial_loss=initial_terms,
        final_loss=final_terms,
        diagnostics=diagnostics,
    )
