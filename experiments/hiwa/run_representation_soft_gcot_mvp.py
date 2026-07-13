"""MiHiA MVP: learn latent representations around a detached Soft-GCOT inner loop.

This is an exploratory mechanism experiment.  It never supplies direction
labels, movement labels, or hidden neural--movement correspondences to either
the encoders or the OT solver.  Labels are opened only after fitting to report
the same direction-nearest-neighbour diagnostic used by the frozen baseline.
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT / "src", ROOT / "src" / "cc_hiwa"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from alternating_representation import AlternatingRepresentationConfig, AlternatingRepresentationSoftGCOT
from common import RESULTS_DIR, ensure_output_dirs, nearest_neighbor_accuracy, write_json
from representation import MLPDecoder, MLPEncoder, ResidualMLPDecoder, ResidualMLPEncoder
from run_taco_faithful_baseline import _problem
from soft_groups import learn_soft_groups
from solver_adapter import solve_soft_gcot_detached


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--max-samples", type=int, default=96)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.50)
    parser.add_argument("--hidden-dimension", type=int, default=16)
    parser.add_argument("--architecture", choices=("residual", "mlp"), default="residual")
    parser.add_argument("--outer-steps", type=int, default=3)
    parser.add_argument("--encoder-steps", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lambda-identity", type=float, default=1.0)
    parser.add_argument("--lambda-cross-reconstruction", type=float, default=1.0)
    parser.add_argument("--normalize-input", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tag", default="representation_mvp_seed200")
    return parser.parse_args()


def _solver_kwargs(seed: int) -> dict[str, object]:
    """A bounded full-support budget for mechanism validation, not a new baseline."""
    return {
        "normalize": True,
        "maxiter": 80,
        "tol": 0.10,
        "mu": 5e-3,
        "shorn_maxiter": 120,
        "sa_maxiter": 12,
        "sa_tol": 1e-2,
        "sa_shorn_maxiter": 40,
        "sa_shorn_gamma": 1e-1,
        "shorn_gamma": 2e-1,
        "support_mode": "full",
        "random_state": seed,
        "warm_start_local": True,
    }


def _direction_accuracy(aligned: np.ndarray, target: np.ndarray, evaluation: dict) -> float:
    return float(
        nearest_neighbor_accuracy(
            aligned,
            np.asarray(evaluation["neural_labels"]),
            target,
            np.asarray(evaluation["movement_labels"]),
        )
    )


def _standardize(values: np.ndarray) -> tuple[np.ndarray, dict[str, list[float]]]:
    mean = values.mean(axis=0, keepdims=True)
    scale = np.maximum(values.std(axis=0, keepdims=True), 1e-8)
    return (values - mean) / scale, {"mean": mean.ravel().tolist(), "scale": scale.ravel().tolist()}


def main() -> None:
    args = parse_args()
    if args.groups < 2 or args.temperature <= 0:
        raise ValueError("groups must be at least two and temperature must be positive")
    ensure_output_dirs()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    source, target, _, _, evaluation = _problem(args.max_samples)
    solver_kwargs = _solver_kwargs(args.seed)
    source_input, source_scaler = _standardize(source) if args.normalize_input else (source, None)
    target_input, target_scaler = _standardize(target) if args.normalize_input else (target, None)

    # Fixed comparison: the old prototype learner and unchanged solver, with
    # labels absent during both grouping and alignment.
    fixed_source = learn_soft_groups(source, args.groups, args.temperature, seed=args.seed)
    fixed_target = learn_soft_groups(target, args.groups, args.temperature, seed=args.seed)
    fixed_solution = solve_soft_gcot_detached(
        source,
        fixed_source.assignments,
        target,
        fixed_target.assignments,
        solver_kwargs=solver_kwargs,
    )

    encoder = ResidualMLPEncoder if args.architecture == "residual" else MLPEncoder
    decoder = ResidualMLPDecoder if args.architecture == "residual" else MLPDecoder
    model = AlternatingRepresentationSoftGCOT(
        encoder(source_input.shape[1], source.shape[1], args.hidden_dimension),
        encoder(target_input.shape[1], source.shape[1], args.hidden_dimension),
        decoder(source.shape[1], source_input.shape[1], args.hidden_dimension),
        decoder(source.shape[1], target_input.shape[1], args.hidden_dimension),
        latent_dimension=source.shape[1],
        config=AlternatingRepresentationConfig(
            n_groups=args.groups,
            temperature=args.temperature,
            outer_steps=args.outer_steps,
            encoder_steps=args.encoder_steps,
            learning_rate=args.learning_rate,
            lambda_identity=args.lambda_identity,
            lambda_cross_reconstruction=args.lambda_cross_reconstruction,
            solver_kwargs=solver_kwargs,
        ),
    )
    learned = model.fit(source_input, target_input)
    final_solver = learned.solution

    result = {
        "experiment": "mihi_alternating_representation_soft_gcot_mvp",
        "scope": "exploratory mechanism check; this is not a replacement for the frozen Soft-GCOT baseline",
        "label_usage": "direction labels are used only after fitting for evaluation; no labels or pair IDs enter encoders, prototypes, or OT",
        "parameters": vars(args),
        "solver_kwargs": solver_kwargs,
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
        "data": {
            "source_samples": int(source.shape[0]),
            "target_samples": int(target.shape[0]),
            "input_dimension": int(source.shape[1]),
            "encoder_input_standardized": bool(args.normalize_input),
            "source_scaler": source_scaler,
            "target_scaler": target_scaler,
        },
        "evaluation": {
            "unaligned_direction_accuracy": _direction_accuracy(source, target, evaluation),
            "fixed_soft_gcot_direction_accuracy": _direction_accuracy(fixed_solution.aligned_source, target, evaluation),
            "learned_representation_direction_accuracy": _direction_accuracy(final_solver.aligned_source, learned.target_latent, evaluation),
        },
        "fixed_solver": {
            "admm_converged": bool(fixed_solution.diagnostics["admm_converged"]),
            "final_primal_residual": float(fixed_solution.diagnostics["admm_primal_residual"][-1]),
            "rotation_orthogonality_error": float(fixed_solution.diagnostics["rotation_orthogonality_error"]),
        },
        "learned_solver": {
            "admm_converged": bool(final_solver.diagnostics["admm_converged"]),
            "final_primal_residual": float(final_solver.diagnostics["admm_primal_residual"][-1]),
            "rotation_orthogonality_error": float(final_solver.diagnostics["rotation_orthogonality_error"]),
            "group_transport": final_solver.group_transport,
        },
        "outer_history": learned.history,
    }
    output = RESULTS_DIR / f"{args.tag}.json"
    write_json(output, result)
    print(
        "direction accuracy "
        f"unaligned={result['evaluation']['unaligned_direction_accuracy']:.4f} "
        f"fixed={result['evaluation']['fixed_soft_gcot_direction_accuracy']:.4f} "
        f"learned={result['evaluation']['learned_representation_direction_accuracy']:.4f}"
    )
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
