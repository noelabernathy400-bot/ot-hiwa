"""Run CC-HiWA on the existing macaque neural-movement task.

This is the first real-data adapter for the dataset-agnostic CC-HiWA core. It
uses the current ROCA pipeline only to obtain a label-free determinant branch
and rotation in the 3D neural-movement setting. The new TACO/GCOT-inspired part
is the component-conditioned sample transport:

    S = A P B^T
    C_cc = C_base - beta * log(S + eps)

Labels are used only after ROCA selection and CC-HiWA fitting for evaluation.
"""

from __future__ import annotations

import argparse
import platform
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn

from cc_hiwa import barycentric_projection, coupling_entropy, fit_cc_hiwa
from common import FIGURES_DIR, RESULTS_DIR, ensure_output_dirs, write_json
from run_component_aware import _hard_baseline, _representative_orientation_selector
from run_rotation_stabilized import ANNEALING_PATH, _fixed_assignments, _load_problem, _run_chain
from run_soft_neural import PROFILES, evaluate


RESULT_DIR = RESULTS_DIR / "cc_hiwa_neural"
FIGURE_DIR = FIGURES_DIR / "cc_hiwa_neural"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[50, 51])
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--entropy-weight", type=float, default=0.05)
    parser.add_argument("--retain-mass", type=float, default=0.90)
    parser.add_argument("--max-support-factor", type=float, default=1.5)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.25])
    parser.add_argument("--group-gamma", type=float, default=0.1)
    parser.add_argument("--sample-gamma", type=float, default=0.1)
    parser.add_argument("--sinkhorn-maxiter", type=int, default=300)
    parser.add_argument("--tag", default="smoke")
    args = parser.parse_args()
    if any(beta < 0 for beta in args.betas):
        parser.error("betas must be non-negative")
    return args


def _select_roca_candidate(
    *,
    neural_3d: np.ndarray,
    movement_3d: np.ndarray,
    neural_assignments: dict[float, np.ndarray],
    movement_assignments: dict[float, np.ndarray],
    hard_result: dict[str, Any],
    target_transform: np.ndarray,
    oracle_rotation: np.ndarray,
    evaluation_args: dict[str, Any],
    seed: int,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = []
    candidate_stage_records = []
    for sign in (-1, 1):
        stages = _run_chain(
            method=f"roca_candidate_det_{sign:+d}",
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
        final = stages[-1]
        final["requested_determinant_sign"] = sign
        candidates.append(final)
        candidate_stage_records.extend(stages)

    selector = _representative_orientation_selector(
        neural_3d,
        neural_assignments[ANNEALING_PATH[-1]],
        movement_3d,
        movement_assignments[ANNEALING_PATH[-1]],
        [np.asarray(candidate["transport_P"]) for candidate in candidates],
    )
    for candidate in candidates:
        candidate.update(selector)
    selected_sign = int(selector["representative_orientation_sign"])
    selected = next(
        candidate for candidate in candidates
        if int(candidate["requested_determinant_sign"]) == selected_sign
    )
    return selected, candidates + candidate_stage_records


def _summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for beta in sorted({float(record["beta"]) for record in records}):
        chosen = sorted(
            [record for record in records if float(record["beta"]) == beta],
            key=lambda record: int(record["seed"]),
        )
        acc = np.asarray([record["after_direction_accuracy"] for record in chosen])
        r2 = np.asarray([record["after_velocity_r2"] for record in chosen])
        objective = np.asarray([record["cc_objective"] for record in chosen])
        sample_entropy = np.asarray([record["sample_transport_entropy"] for record in chosen])
        transfer_acc = np.asarray(
            [record["transport_direction_accuracy"] for record in chosen]
        )
        output[str(beta)] = {
            "seeds": [int(record["seed"]) for record in chosen],
            "mean_direction_accuracy": float(acc.mean()),
            "min_direction_accuracy": float(acc.min()),
            "mean_movement_r2": float(r2.mean()),
            "min_movement_r2": float(r2.min()),
            "mean_cc_objective": float(objective.mean()),
            "mean_sample_transport_entropy": float(sample_entropy.mean()),
            "mean_transport_direction_accuracy": float(transfer_acc.mean()),
            "min_transport_direction_accuracy": float(transfer_acc.min()),
            "selected_determinant_signs": [
                int(record["selected_roca_determinant_sign"]) for record in chosen
            ],
        }
    if "0.0" in output:
        baseline = output["0.0"]
        for value in output.values():
            value["delta_mean_direction_accuracy_vs_beta0"] = (
                value["mean_direction_accuracy"] - baseline["mean_direction_accuracy"]
            )
            value["delta_mean_movement_r2_vs_beta0"] = (
                value["mean_movement_r2"] - baseline["mean_movement_r2"]
            )
            value["delta_mean_transport_direction_accuracy_vs_beta0"] = (
                value["mean_transport_direction_accuracy"]
                - baseline["mean_transport_direction_accuracy"]
            )
    return output


def _transport_direction_accuracy(
    transport: np.ndarray,
    source_labels: np.ndarray,
    target_labels: np.ndarray,
) -> float:
    coupling = np.asarray(transport, dtype=float)
    source_labels = np.asarray(source_labels)
    target_labels = np.asarray(target_labels)
    if coupling.shape[0] != source_labels.shape[0] or coupling.shape[1] != target_labels.shape[0]:
        raise ValueError("transport shape must match source and target labels")
    source_indices = np.argmax(coupling, axis=0)
    return float(np.mean(source_labels[source_indices] == target_labels))


def _plot(summary: dict[str, Any], out_path: Path) -> None:
    betas = sorted(float(key) for key in summary)
    labels = [str(beta) for beta in betas]
    acc = [summary[str(beta)]["mean_direction_accuracy"] for beta in betas]
    r2 = [summary[str(beta)]["mean_movement_r2"] for beta in betas]
    entropy = [summary[str(beta)]["mean_sample_transport_entropy"] for beta in betas]
    transfer = [summary[str(beta)]["mean_transport_direction_accuracy"] for beta in betas]
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.2), facecolor="white", constrained_layout=True)
    axes[0].plot(labels, acc, marker="o")
    axes[0].set(title="CC-HiWA direction accuracy", xlabel="beta", ylabel="accuracy")
    axes[1].plot(labels, r2, marker="o", color="#2ca02c")
    axes[1].set(title="CC-HiWA movement R²", xlabel="beta", ylabel="R²")
    axes[2].plot(labels, entropy, marker="o", color="#9467bd")
    axes[2].set(title="Sample transport entropy", xlabel="beta", ylabel="entropy")
    axes[3].plot(labels, transfer, marker="o", color="#ff7f0e")
    axes[3].set(title="Coupling label transfer", xlabel="beta", ylabel="accuracy")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.set_facecolor("white")
    fig.savefig(out_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    neural_3d, movement_3d, target_transform, oracle_rotation, evaluation_args = _load_problem()

    cc_records: list[dict[str, Any]] = []
    hard_records: list[dict[str, Any]] = []
    roca_records: list[dict[str, Any]] = []

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
        hard_records.append(hard)
        neural_assignments = _fixed_assignments(
            neural_3d,
            args.groups,
            args.entropy_weight,
            seed,
        )
        movement_assignments = _fixed_assignments(
            movement_3d,
            args.groups,
            args.entropy_weight,
            seed,
        )
        selected_roca, all_roca = _select_roca_candidate(
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
        roca_records.extend(all_roca)
        selected_rotation = np.asarray(selected_roca["rotation_R"])
        source_assignments = neural_assignments[ANNEALING_PATH[-1]]
        target_assignments = movement_assignments[ANNEALING_PATH[-1]]
        print(
            f"  ROCA selected det={selected_roca['requested_determinant_sign']:+d} "
            f"acc={selected_roca['after_direction_accuracy']:.4f} "
            f"R2={selected_roca['after_velocity_r2']:.4f}"
        )

        for beta in args.betas:
            cc = fit_cc_hiwa(
                neural_3d,
                movement_3d,
                source_assignments,
                target_assignments,
                rotation=selected_rotation,
                beta=beta,
                group_gamma=args.group_gamma,
                sample_gamma=args.sample_gamma,
                sinkhorn_maxiter=args.sinkhorn_maxiter,
            )
            aligned = barycentric_projection(cc.sample_transport, movement_3d)
            metrics = evaluate(
                aligned=aligned,
                neural_3d=neural_3d,
                movement_3d=movement_3d,
                **evaluation_args,
            )
            record = {
                "method": "cc_hiwa_neural_after_roca_rotation",
                "seed": seed,
                "beta": beta,
                "selected_roca_determinant_sign": int(
                    selected_roca["requested_determinant_sign"]
                ),
                "selected_roca_direction_accuracy": selected_roca[
                    "after_direction_accuracy"
                ],
                "selected_roca_movement_r2": selected_roca["after_velocity_r2"],
                **metrics,
                "cc_objective": cc.objective,
                "group_transport": cc.group_transport,
                "sample_transport_entropy": coupling_entropy(cc.sample_transport),
                "group_transport_entropy": coupling_entropy(cc.group_transport),
                "transport_direction_accuracy": _transport_direction_accuracy(
                    cc.sample_transport,
                    evaluation_args["neural_labels"],
                    evaluation_args["movement_labels"],
                ),
                "compatibility_mean": float(np.mean(cc.compatibility)),
                "compatibility_min": float(np.min(cc.compatibility)),
                "compatibility_max": float(np.max(cc.compatibility)),
                "conditioned_cost_mean": float(np.mean(cc.conditioned_sample_cost)),
                "base_cost_mean": float(np.mean(cc.base_sample_cost)),
            }
            cc_records.append(record)
            print(
                f"  beta={beta:g} acc={record['after_direction_accuracy']:.4f} "
                f"R2={record['after_velocity_r2']:.4f} "
                f"T_acc={record['transport_direction_accuracy']:.4f} "
                f"obj={record['cc_objective']:.6f} "
                f"T_entropy={record['sample_transport_entropy']:.4f}"
            )

    summary = _summarize(cc_records)
    suffix = f"_{args.tag}" if args.tag else ""
    raw_path = RESULT_DIR / f"cc_hiwa_neural_raw{suffix}.json"
    summary_path = RESULT_DIR / f"cc_hiwa_neural_summary{suffix}.json"
    figure_path = FIGURE_DIR / f"cc_hiwa_neural_summary{suffix}.png"
    payload = {
        "experiment": "cc_hiwa_neural",
        "status": "first_real_data_adapter",
        "method_note": (
            "Uses ROCA only to select the 3D determinant branch and rotation without "
            "labels; uses CC-HiWA component-conditioned sample OT after that."
        ),
        "label_usage": (
            "direction labels and movement targets are used only after fitting for "
            "evaluation; they are not used by ROCA selection or CC-HiWA transport."
        ),
        "parameters": {
            "seeds": args.seeds,
            "profile": args.profile,
            "betas": args.betas,
            "groups": args.groups,
            "entropy_weight": args.entropy_weight,
            "annealing_path": ANNEALING_PATH,
            "group_gamma": args.group_gamma,
            "sample_gamma": args.sample_gamma,
            "sinkhorn_maxiter": args.sinkhorn_maxiter,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "hard_records": hard_records,
        "roca_records": roca_records,
        "cc_records": cc_records,
    }
    write_json(raw_path, payload)
    write_json(
        summary_path,
        {
            "experiment": "cc_hiwa_neural",
            "summary": summary,
        },
    )
    _plot(summary, figure_path)
    print(f"\nSaved {raw_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
