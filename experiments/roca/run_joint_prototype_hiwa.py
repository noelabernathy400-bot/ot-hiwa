"""Minimal TACO-style joint prototype optimization for Soft-HiWA.

The optimizer keeps neural and movement embeddings fixed, excludes direction
labels from fitting/selection, and uses a frozen ROCA-selected rotation as the
first alignment-aware signal for prototype/assignment updates.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn

from common import FIGURES_DIR, RESULTS_DIR, ensure_output_dirs, json_ready, write_json
from joint_prototypes import JointPrototypeConfig, optimize_joint_prototypes
from run_component_aware import _hard_baseline, _representative_orientation_selector
from run_rotation_stabilized import ANNEALING_PATH, _load_problem, _run_chain
from run_soft_neural import PROFILES
from soft_groups import assignment_entropy, assignments_from_prototypes, learn_soft_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[50, 51, 52, 53, 54])
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--entropy-weight", type=float, default=0.05)
    parser.add_argument("--retain-mass", type=float, default=0.90)
    parser.add_argument("--max-support-factor", type=float, default=1.5)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=ANNEALING_PATH[-1])
    parser.add_argument("--lambda-align", type=float, default=0.10)
    parser.add_argument("--lambda-balance", type=float, default=0.05)
    parser.add_argument("--lambda-entropy", type=float, default=0.10)
    parser.add_argument("--group-gamma", type=float, default=0.10)
    parser.add_argument("--joint-maxiter", type=int, default=80)
    parser.add_argument("--tag", default="joint_proto_minimal_seeds_50_54")
    args = parser.parse_args()
    if args.groups != 4:
        parser.error("ROCA representative selector currently requires exactly 4 groups")
    if args.temperature not in ANNEALING_PATH:
        parser.error(f"temperature must be one of {ANNEALING_PATH} for this minimal runner")
    if min(args.lambda_align, args.lambda_balance, args.lambda_entropy) < 0:
        parser.error("loss weights must be non-negative")
    return args


def _standardize_values(values: np.ndarray) -> np.ndarray:
    mean = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    return (values - mean) / np.maximum(scale, 1e-12)


def _learn_frozen(
    values: np.ndarray,
    *,
    groups: int,
    entropy_weight: float,
    seed: int,
) -> tuple[Any, dict[float, np.ndarray]]:
    learned = learn_soft_groups(
        values,
        n_groups=groups,
        temperature=ANNEALING_PATH[0],
        entropy_weight=entropy_weight,
        seed=seed,
    )
    standardized = _standardize_values(values)
    assignments = {
        tau: assignments_from_prototypes(
            standardized,
            learned.prototypes_standardized,
            tau,
        )
        for tau in ANNEALING_PATH
    }
    return learned, assignments


def _assignments_from_joint_prototypes(
    values: np.ndarray,
    prototypes_standardized: np.ndarray,
) -> dict[float, np.ndarray]:
    standardized = _standardize_values(values)
    return {
        tau: assignments_from_prototypes(standardized, prototypes_standardized, tau)
        for tau in ANNEALING_PATH
    }


def _run_roca_candidates(
    *,
    label: str,
    neural_3d: np.ndarray,
    movement_3d: np.ndarray,
    neural_assignments: dict[float, np.ndarray],
    movement_assignments: dict[float, np.ndarray],
    hard_result: dict,
    target_transform: np.ndarray,
    oracle_rotation: np.ndarray,
    evaluation_args: dict,
    seed: int,
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict], dict[str, Any], dict]:
    stage_records: list[dict] = []
    final_records: list[dict] = []
    for sign in (-1, 1):
        method = f"{label}_component_{'negative' if sign == -1 else 'positive'}"
        stages = _run_chain(
            method=method,
            neural_3d=neural_3d,
            movement_3d=movement_3d,
            neural_assignments=neural_assignments,
            movement_assignments=movement_assignments,
            hard_result=hard_result,
            target_transform=target_transform,
            oracle_rotation=oracle_rotation,
            evaluation_args=evaluation_args,
            seed=seed,
            profile=args.profile,
            retain_mass=args.retain_mass,
            max_support_factor=args.max_support_factor,
            hard_start=True,
            anchor_weight=args.anchor_weight,
            determinant_sign=sign,
        )
        stage_records.extend(stages)
        final_records.append(stages[-1])
    selector = _representative_orientation_selector(
        neural_3d,
        neural_assignments[ANNEALING_PATH[-1]],
        movement_3d,
        movement_assignments[ANNEALING_PATH[-1]],
        [np.asarray(record["transport_P"]) for record in final_records],
    )
    for record in final_records:
        record.update(selector)
    selected = next(
        record
        for record in final_records
        if record["requested_determinant_sign"] == selector["representative_orientation_sign"]
    )
    return stage_records, final_records, selector, selected


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    write_json(path, payload)


def _safe_write_text(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.write_text(content, encoding="utf-8")


def _summary(seed_records: list[dict]) -> dict[str, Any]:
    frozen_acc = np.asarray([record["frozen_selected"]["after_direction_accuracy"] for record in seed_records])
    joint_acc = np.asarray([record["joint_selected"]["after_direction_accuracy"] for record in seed_records])
    frozen_r2 = np.asarray([record["frozen_selected"]["after_velocity_r2"] for record in seed_records])
    joint_r2 = np.asarray([record["joint_selected"]["after_velocity_r2"] for record in seed_records])
    final_total = np.asarray([record["joint_optimization"]["final_loss"]["total"] for record in seed_records])
    initial_total = np.asarray([record["joint_optimization"]["initial_loss"]["total"] for record in seed_records])
    return {
        "seeds": [int(record["seed"]) for record in seed_records],
        "label_usage": "labels are used only for final evaluation metrics, never for optimization or candidate selection",
        "selection_rule": "ROCA representative orientation sign",
        "frozen_roca": {
            "direction_accuracy_values": frozen_acc.tolist(),
            "mean_direction_accuracy": float(frozen_acc.mean()),
            "movement_r2_values": frozen_r2.tolist(),
            "mean_movement_r2": float(frozen_r2.mean()),
        },
        "joint_prototype_roca": {
            "direction_accuracy_values": joint_acc.tolist(),
            "mean_direction_accuracy": float(joint_acc.mean()),
            "movement_r2_values": joint_r2.tolist(),
            "mean_movement_r2": float(joint_r2.mean()),
        },
        "joint_minus_frozen": {
            "direction_accuracy_values": (joint_acc - frozen_acc).tolist(),
            "mean_direction_accuracy_delta": float((joint_acc - frozen_acc).mean()),
            "movement_r2_values": (joint_r2 - frozen_r2).tolist(),
            "mean_movement_r2_delta": float((joint_r2 - frozen_r2).mean()),
        },
        "joint_objective": {
            "initial_total_values": initial_total.tolist(),
            "final_total_values": final_total.tolist(),
            "mean_total_delta": float((final_total - initial_total).mean()),
        },
    }


def _plot(seed_records: list[dict], out_path: Path) -> None:
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {out_path}")
    seeds = [int(record["seed"]) for record in seed_records]
    frozen_acc = [record["frozen_selected"]["after_direction_accuracy"] for record in seed_records]
    joint_acc = [record["joint_selected"]["after_direction_accuracy"] for record in seed_records]
    frozen_r2 = [record["frozen_selected"]["after_velocity_r2"] for record in seed_records]
    joint_r2 = [record["joint_selected"]["after_velocity_r2"] for record in seed_records]
    x = np.arange(len(seeds))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), facecolor="white", constrained_layout=True)
    axes[0].plot(x, frozen_acc, marker="o", label="Frozen ROCA")
    axes[0].plot(x, joint_acc, marker="s", label="Joint prototypes + ROCA")
    axes[1].plot(x, frozen_r2, marker="o", label="Frozen ROCA")
    axes[1].plot(x, joint_r2, marker="s", label="Joint prototypes + ROCA")
    for axis, title, ylabel in [
        (axes[0], "Direction accuracy", "accuracy"),
        (axes[1], "Movement geometry", "R2"),
    ]:
        axis.set_facecolor("white")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("seed")
        axis.set_xticks(x, seeds)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.savefig(out_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def _markdown_note(payload: dict[str, Any], summary: dict[str, Any]) -> str:
    lines = [
        "# Joint-Prototype HiWA 最小实验记录",
        "",
        "## 实验定位",
        "",
        "本实验只补齐 TACO-inspired HiWA 主线中的最小联合优化环节：固定神经与运动 3D embedding，固定一个由 frozen ROCA 无标签选择得到的 rotation，只优化 soft prototypes 及其诱导的 assignments。",
        "",
        "不使用 direction label 训练，不使用 direction accuracy 或 $R^2$ 选择超参数，不扩展 PBMC / 新数据集 / encoder。",
        "",
        "## 目标函数",
        "",
        "$$",
        "\\mathcal L = \\mathcal L_{proto}^X + \\mathcal L_{proto}^Y + \\lambda_{align}\\mathcal L_{align} + \\lambda_{bal}\\mathcal L_{balance} + \\lambda_{ent}\\mathcal L_{entropy}",
        "$$",
        "",
        "其中 assignment 在各自标准化 embedding 空间由 prototypes 产生；alignment representative 使用原始固定 3D embedding 的加权均值。",
        "",
        "## 参数",
        "",
        "```json",
        json.dumps(json_ready(payload["parameters"]), ensure_ascii=False, indent=2),
        "```",
        "",
        "## 汇总",
        "",
        "| 指标 | Frozen ROCA | Joint prototypes + ROCA | Delta |",
        "|---|---:|---:|---:|",
        f"| mean direction accuracy | {summary['frozen_roca']['mean_direction_accuracy']:.4f} | {summary['joint_prototype_roca']['mean_direction_accuracy']:.4f} | {summary['joint_minus_frozen']['mean_direction_accuracy_delta']:.4f} |",
        f"| mean movement R2 | {summary['frozen_roca']['mean_movement_r2']:.4f} | {summary['joint_prototype_roca']['mean_movement_r2']:.4f} | {summary['joint_minus_frozen']['mean_movement_r2_delta']:.4f} |",
        f"| mean objective delta |  |  | {summary['joint_objective']['mean_total_delta']:.6f} |",
        "",
        "## 每个 seed",
        "",
        "| seed | frozen det | joint det | frozen acc | joint acc | frozen R2 | joint R2 | objective initial | objective final |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in payload["seed_records"]:
        lines.append(
            "| {seed} | {fd:+d} | {jd:+d} | {fa:.4f} | {ja:.4f} | {fr:.4f} | {jr:.4f} | {li:.6f} | {lf:.6f} |".format(
                seed=int(record["seed"]),
                fd=int(record["frozen_selected"]["requested_determinant_sign"]),
                jd=int(record["joint_selected"]["requested_determinant_sign"]),
                fa=float(record["frozen_selected"]["after_direction_accuracy"]),
                ja=float(record["joint_selected"]["after_direction_accuracy"]),
                fr=float(record["frozen_selected"]["after_velocity_r2"]),
                jr=float(record["joint_selected"]["after_velocity_r2"]),
                li=float(record["joint_optimization"]["initial_loss"]["total"]),
                lf=float(record["joint_optimization"]["final_loss"]["total"]),
            )
        )
    lines.extend(
        [
            "",
            "## 初步判断",
            "",
            "- 若 joint objective 稳定下降但 final evaluation 不升，说明 alignment-aware prototypes 在当前固定 $R$ 下学到了目标函数，但该目标还未足以改善 sample-level OT。",
            "- 若 assignment entropy 或 group mass 出现塌缩，需要优先调整 balance/entropy 约束，而不是扩大数据集或引入 encoder。",
            "- 本实验是最小机制验证，不应被解释为完整 TACO 迁移完成。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    neural_3d, movement_3d, target_transform, oracle_rotation, evaluation_args = _load_problem()
    config = JointPrototypeConfig(
        n_groups=args.groups,
        temperature=args.temperature,
        lambda_align=args.lambda_align,
        lambda_balance=args.lambda_balance,
        lambda_entropy=args.lambda_entropy,
        group_gamma=args.group_gamma,
        maxiter=args.joint_maxiter,
    )
    seed_records: list[dict] = []
    all_stage_records: list[dict] = []

    for seed in args.seeds:
        print(f"\nSeed {seed}")
        hard = _hard_baseline(
            neural_3d,
            movement_3d,
            target_transform,
            oracle_rotation,
            evaluation_args,
            seed,
            args,
        )
        neural_frozen, neural_assignments = _learn_frozen(
            neural_3d,
            groups=args.groups,
            entropy_weight=args.entropy_weight,
            seed=seed,
        )
        movement_frozen, movement_assignments = _learn_frozen(
            movement_3d,
            groups=args.groups,
            entropy_weight=args.entropy_weight,
            seed=seed,
        )
        frozen_stages, frozen_final, frozen_selector, frozen_selected = _run_roca_candidates(
            label="frozen",
            neural_3d=neural_3d,
            movement_3d=movement_3d,
            neural_assignments=neural_assignments,
            movement_assignments=movement_assignments,
            hard_result=hard,
            target_transform=target_transform,
            oracle_rotation=oracle_rotation,
            evaluation_args=evaluation_args,
            seed=seed,
            args=args,
        )
        all_stage_records.extend(frozen_stages)
        frozen_rotation = np.asarray(frozen_selected["rotation_R"], dtype=float)
        h0 = float(
            0.5
            * (
                assignment_entropy(neural_assignments[args.temperature]).mean()
                + assignment_entropy(movement_assignments[args.temperature]).mean()
            )
        )
        joint = optimize_joint_prototypes(
            neural_3d,
            movement_3d,
            rotation=frozen_rotation,
            initial_source_prototypes_standardized=neural_frozen.prototypes_standardized,
            initial_target_prototypes_standardized=movement_frozen.prototypes_standardized,
            target_entropy=h0,
            config=config,
        )
        joint_neural_assignments = _assignments_from_joint_prototypes(
            neural_3d,
            joint.source_prototypes_standardized,
        )
        joint_movement_assignments = _assignments_from_joint_prototypes(
            movement_3d,
            joint.target_prototypes_standardized,
        )
        joint_stages, joint_final, joint_selector, joint_selected = _run_roca_candidates(
            label="joint",
            neural_3d=neural_3d,
            movement_3d=movement_3d,
            neural_assignments=joint_neural_assignments,
            movement_assignments=joint_movement_assignments,
            hard_result=hard,
            target_transform=target_transform,
            oracle_rotation=oracle_rotation,
            evaluation_args=evaluation_args,
            seed=seed,
            args=args,
        )
        all_stage_records.extend(joint_stages)
        print(
            "  frozen det={fd:+d} acc={fa:.4f} R2={fr:.4f} | "
            "joint det={jd:+d} acc={ja:.4f} R2={jr:.4f} | "
            "objective {li:.6f}->{lf:.6f}".format(
                fd=int(frozen_selected["requested_determinant_sign"]),
                fa=float(frozen_selected["after_direction_accuracy"]),
                fr=float(frozen_selected["after_velocity_r2"]),
                jd=int(joint_selected["requested_determinant_sign"]),
                ja=float(joint_selected["after_direction_accuracy"]),
                jr=float(joint_selected["after_velocity_r2"]),
                li=float(joint.initial_loss["total"]),
                lf=float(joint.final_loss["total"]),
            )
        )
        seed_records.append(
            {
                "seed": seed,
                "hard_baseline": hard,
                "frozen_selector": frozen_selector,
                "joint_selector": joint_selector,
                "frozen_final_candidates": frozen_final,
                "joint_final_candidates": joint_final,
                "frozen_selected": frozen_selected,
                "joint_selected": joint_selected,
                "joint_optimization": {
                    "initial_loss": joint.initial_loss,
                    "final_loss": joint.final_loss,
                    "diagnostics": joint.diagnostics,
                    "source_group_mass": joint.source_assignments.mean(axis=0),
                    "target_group_mass": joint.target_assignments.mean(axis=0),
                    "group_transport": joint.group_transport,
                },
            }
        )

    summary = _summary(seed_records)
    suffix = f"_{args.tag}" if args.tag else ""
    raw_path = RESULTS_DIR / f"joint_prototype_hiwa{suffix}.json"
    summary_path = RESULTS_DIR / f"joint_prototype_hiwa_summary{suffix}.json"
    figure_path = FIGURES_DIR / f"joint_prototype_hiwa{suffix}.png"
    note_path = (
        Path(__file__).resolve().parents[3]
        / "最优传输"
        / "文本笔记"
        / "代码方面"
        / f"Joint-Prototype-HiWA最小实验记录_{args.tag}.md"
    )
    payload = {
        "experiment": "joint_prototype_hiwa_minimal",
        "status": "minimal_mechanism_test",
        "label_usage": "labels are excluded from optimization and ROCA selection; labels are used only for final evaluation metrics",
        "rotation_source": "frozen ROCA selected rotation for the same seed",
        "parameters": {
            "seeds": args.seeds,
            "profile": args.profile,
            "profile_parameters": PROFILES[args.profile],
            "annealing_path": ANNEALING_PATH,
            "groups": args.groups,
            "entropy_weight": args.entropy_weight,
            "retain_mass": args.retain_mass,
            "max_support_factor": args.max_support_factor,
            "anchor_weight": args.anchor_weight,
            "joint_config": {
                "n_groups": config.n_groups,
                "temperature": config.temperature,
                "lambda_align": config.lambda_align,
                "lambda_balance": config.lambda_balance,
                "lambda_entropy": config.lambda_entropy,
                "group_gamma": config.group_gamma,
                "sinkhorn_maxiter": config.sinkhorn_maxiter,
                "maxiter": config.maxiter,
                "gtol": config.gtol,
            },
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "summary": summary,
        "seed_records": seed_records,
        "stage_records": all_stage_records,
    }
    _safe_write_json(raw_path, payload)
    _safe_write_json(summary_path, {"experiment": payload["experiment"], "summary": summary})
    _plot(seed_records, figure_path)
    _safe_write_text(note_path, _markdown_note(payload, summary))
    print(f"\nSaved {raw_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {figure_path}")
    print(f"Saved {note_path}")


if __name__ == "__main__":
    main()
