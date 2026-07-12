"""Minimal alternating joint-prototype experiment on the frozen MiHiA subset.

The first block runs the existing full-support Soft-GCOT HiWA solver.  The
second block holds its final group transport and global rotation fixed while
updating soft prototypes.  No Sinkhorn or ADMM gradients are used.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn

from common import FIGURES_DIR, RESULTS_DIR, ensure_output_dirs, write_json
from joint_soft_gcot import JointSoftGCOTConfig, update_prototypes_from_soft_gcot
from run_neural import least_squares_rotation
from run_soft_neural import PROFILES, run_hard, run_soft
from run_taco_faithful_baseline import _problem
from soft_groups import assignment_entropy, assignments_from_prototypes, learn_soft_groups


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="audit")
    parser.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102])
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--temperature-path", nargs="+", type=float, default=[0.25, 0.35, 0.50])
    parser.add_argument("--entropy-weight", type=float, default=0.05)
    parser.add_argument("--joint-outer-iterations", type=int, default=3)
    parser.add_argument("--lambda-cross", type=float, default=0.10)
    parser.add_argument("--lambda-entropy", type=float, default=0.05)
    parser.add_argument("--lambda-l2", type=float, default=1e-4)
    parser.add_argument("--joint-maxiter", type=int, default=80)
    parser.add_argument("--max-samples", type=int, default=96)
    parser.add_argument("--tag", default="joint_soft_gcot_seeds_100_102")
    parser.add_argument("--sync-results", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.groups != 4:
        parser.error("this MiHiA 3D experiment fixes four groups for a matched comparison")
    if args.joint_outer_iterations < 1 or args.joint_outer_iterations > 3:
        parser.error("joint outer iterations must be between one and three")
    if any(value <= 0 for value in args.temperature_path):
        parser.error("all temperatures must be positive")
    if min(args.lambda_cross, args.lambda_entropy, args.lambda_l2) < 0:
        parser.error("joint loss weights must be non-negative")
    return args


def _standardize(values: np.ndarray) -> np.ndarray:
    return (values - values.mean(axis=0, keepdims=True)) / np.maximum(values.std(axis=0, keepdims=True), 1e-12)


def _initial_path(values: np.ndarray, args: argparse.Namespace, seed: int) -> tuple[np.ndarray, list[np.ndarray]]:
    learned = learn_soft_groups(values, args.groups, args.temperature_path[0], args.entropy_weight, seed=seed)
    standardized = _standardize(values)
    stages = [assignments_from_prototypes(standardized, learned.prototypes_standardized, tau) for tau in args.temperature_path]
    return learned.prototypes_standardized, stages


def _stages_from_prototypes(values: np.ndarray, prototypes: np.ndarray, temperatures: list[float]) -> list[np.ndarray]:
    standardized = _standardize(values)
    return [assignments_from_prototypes(standardized, prototypes, tau) for tau in temperatures]


def _run_soft_chain(
    *,
    neural: np.ndarray,
    movement: np.ndarray,
    source_stages: list[np.ndarray],
    target_stages: list[np.ndarray],
    hard: dict,
    target_transform: np.ndarray,
    oracle_rotation: np.ndarray,
    evaluation: dict,
    seed: int,
    profile: str,
    initial_rotation: np.ndarray | None,
    initial_transport: np.ndarray | None,
    method: str,
) -> tuple[dict, list[dict]]:
    rotation = np.asarray(hard["rotation_R"]) if initial_rotation is None else np.asarray(initial_rotation)
    transport = np.asarray(hard["transport_P"]) if initial_transport is None else np.asarray(initial_transport)
    records: list[dict] = []
    for stage, (source, target) in enumerate(zip(source_stages, target_stages)):
        result, _ = run_soft(
            neural,
            source,
            movement,
            target,
            target_transform,
            oracle_rotation,
            seed,
            profile,
            retain_mass=0.90,
            max_support_factor=1.5,
            evaluation_args=evaluation,
            method=method,
            initial_rotation=rotation,
            initial_transport=transport,
            warm_start_local=True,
            support_mode="full",
        )
        result["stage"] = stage
        result["temperature"] = float([0.25, 0.35, 0.50][stage]) if len(source_stages) == 3 else None
        records.append(result)
        rotation = np.asarray(result["rotation_R"])
        transport = np.asarray(result["transport_P"])
    return records[-1], records


def _plot(records: list[dict], path: Path) -> None:
    seeds = sorted({row["seed"] for row in records})
    methods = [("hard_hiwa", "Hard HiWA", "#666666"), ("fixed_soft_gcot", "Fixed Soft-GCOT", "#377eb8"), ("joint_soft_gcot", "Joint Soft-GCOT", "#984ea3")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, key, title in ((axes[0], "after_direction_accuracy", "Direction accuracy"), (axes[1], "after_velocity_r2", "Movement R²")):
        for method, label, color in methods:
            mapping = {row["seed"]: row[key] for row in records if row["method"] == method}
            axis.plot(seeds, [mapping[seed] for seed in seeds], marker="o", label=label, color=color)
        axis.set(title=title, xlabel="seed")
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _summary(records: list[dict]) -> dict:
    output: dict[str, dict] = {}
    for method in ("hard_hiwa", "fixed_soft_gcot", "joint_soft_gcot"):
        rows = [row for row in records if row["method"] == method]
        output[method] = {
            "direction_accuracy_values": [row["after_direction_accuracy"] for row in rows],
            "movement_r2_values": [row["after_velocity_r2"] for row in rows],
            "mean_direction_accuracy": float(np.mean([row["after_direction_accuracy"] for row in rows])),
            "mean_movement_r2": float(np.mean([row["after_velocity_r2"] for row in rows])),
            "converged": int(sum(bool(row["converged"]) for row in rows)),
        }
    fixed = np.asarray(output["fixed_soft_gcot"]["direction_accuracy_values"])
    joint = np.asarray(output["joint_soft_gcot"]["direction_accuracy_values"])
    output["joint_minus_fixed"] = {
        "direction_accuracy_values": (joint - fixed).tolist(),
        "mean_direction_accuracy": float(np.mean(joint - fixed)),
    }
    return output


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    neural, movement, target_transform, oracle_rotation, evaluation = _problem(args.max_samples)
    records: list[dict] = []
    joint_diagnostics: list[dict] = []
    stage_records: list[dict] = []
    for seed in args.seeds:
        source_proto, source_stages = _initial_path(neural, args, seed)
        target_proto, target_stages = _initial_path(movement, args, seed)
        hard, _ = run_hard(
            "hard_hiwa", neural, np.argmax(source_stages[-1], axis=1), movement, np.argmax(target_stages[-1], axis=1), target_transform, oracle_rotation, seed, args.profile, evaluation
        )
        records.append(hard)
        fixed, fixed_stages = _run_soft_chain(
            neural=neural, movement=movement, source_stages=source_stages, target_stages=target_stages, hard=hard,
            target_transform=target_transform, oracle_rotation=oracle_rotation, evaluation=evaluation, seed=seed, profile=args.profile,
            initial_rotation=None, initial_transport=None, method="fixed_soft_gcot"
        )
        records.append(fixed)
        stage_records.extend(fixed_stages)
        prior_rotation = None
        prior_transport = None
        latest = None
        for outer in range(args.joint_outer_iterations):
            latest, soft_stages = _run_soft_chain(
                neural=neural, movement=movement, source_stages=source_stages, target_stages=target_stages, hard=hard,
                target_transform=target_transform, oracle_rotation=oracle_rotation, evaluation=evaluation, seed=seed, profile=args.profile,
                initial_rotation=prior_rotation, initial_transport=prior_transport, method="joint_soft_gcot"
            )
            latest["joint_outer_iteration"] = outer
            stage_records.extend(soft_stages)
            prior_rotation = np.asarray(latest["rotation_R"])
            prior_transport = np.asarray(latest["transport_P"])
            if outer == args.joint_outer_iterations - 1:
                break
            target_entropy = float(0.5 * (assignment_entropy(source_stages[-1]).mean() + assignment_entropy(target_stages[-1]).mean()))
            update = update_prototypes_from_soft_gcot(
                neural, movement, rotation=prior_rotation, group_transport=prior_transport,
                initial_source_prototypes_standardized=source_proto, initial_target_prototypes_standardized=target_proto,
                target_entropy=target_entropy,
                config=JointSoftGCOTConfig(n_groups=args.groups, temperature=args.temperature_path[-1], lambda_cross=args.lambda_cross, lambda_entropy=args.lambda_entropy, lambda_l2=args.lambda_l2, maxiter=args.joint_maxiter),
            )
            joint_diagnostics.append({"seed": seed, "outer_iteration": outer, "initial_loss": update.initial_loss, "final_loss": update.final_loss, "diagnostics": update.diagnostics})
            source_proto = update.source_prototypes_standardized
            target_proto = update.target_prototypes_standardized
            source_stages = _stages_from_prototypes(neural, source_proto, args.temperature_path)
            target_stages = _stages_from_prototypes(movement, target_proto, args.temperature_path)
        assert latest is not None
        records.append(latest)
        print(f"seed={seed} hard={hard['after_direction_accuracy']:.3f} fixed={fixed['after_direction_accuracy']:.3f} joint={latest['after_direction_accuracy']:.3f}", flush=True)

    suffix = f"_{args.tag}" if args.tag else ""
    json_path = RESULTS_DIR / f"alternating_joint_soft_gcot{suffix}.json"
    figure_path = FIGURES_DIR / f"alternating_joint_soft_gcot{suffix}.png"
    payload = {
        "experiment": "alternating_joint_soft_gcot_hiwa",
        "scope": "96x96 MiHiA proof of concept; no ROCA, sparse approximation, or external labels in fitting",
        "parameters": vars(args),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "convergence_rule": "both global and primal ADMM residuals must meet the audit tolerance",
        "results": records,
        "stage_results": stage_records,
        "prototype_updates": joint_diagnostics,
        "summary": _summary(records),
    }
    write_json(json_path, payload)
    _plot(records, figure_path)
    print(f"saved: {json_path}\nsaved: {figure_path}")
    if args.sync_results:
        subprocess.run([sys.executable, str(ROOT / "tools" / "sync_result_artifacts.py"), str(json_path), str(figure_path), "--message", "record alternating joint soft-gcot results"], check=True)


if __name__ == "__main__":
    main()
