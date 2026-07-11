"""Dual-component Soft-HiWA experiment with label-free candidate scores.

For each seed, this script generates one candidate constrained to det(R)=-1
and one constrained to det(R)=+1. Both candidates use identical prototypes,
temperature path, hard-start transport, and compute budget. Labels are used
only after fitting to evaluate candidate coverage and selector diagnostics.
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
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import Isomap

from common import FIGURES_DIR, RESULTS_DIR, ensure_output_dirs, write_json
from run_rotation_stabilized import (
    ANNEALING_PATH,
    _fixed_assignments,
    _load_problem,
    _run_chain,
)
from run_soft_neural import PROFILES, run_hard
from soft_groups import learn_soft_groups
from soft_hiwa import SoftHiWA


BASE_PROXY_KEYS = [
    "transport_objective",
    "local_global_consensus_weighted_rms",
    "final_residual",
]
REVERSE_PROXY_KEYS = [
    "bidirectional_transport_objective",
    "rotation_cycle_error",
    "transport_transpose_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 8, 24])
    parser.add_argument("--profile", choices=PROFILES, default="pilot")
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--entropy-weight", type=float, default=0.05)
    parser.add_argument("--retain-mass", type=float, default=0.90)
    parser.add_argument("--max-support-factor", type=float, default=1.5)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--with-reverse", action="store_true")
    parser.add_argument("--tag", default="mechanism_seeds_7_8_24")
    return parser.parse_args()


def _reverse_diagnostics(
    *,
    forward: dict,
    neural_3d: np.ndarray,
    movement_3d: np.ndarray,
    neural_assignments: np.ndarray,
    movement_assignments: np.ndarray,
    target_transform: np.ndarray,
    oracle_rotation: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    determinant_sign: int,
) -> dict[str, float]:
    """Fit one reverse stage and return label-free cycle diagnostics."""
    model = SoftHiWA(
        dim_red_method=Isomap(n_components=2, n_neighbors=12),
        normalize=True,
        shorn_gamma=2e-1,
        sa_tol=1e-2,
        sa_shorn_gamma=1e-1,
        retain_mass=args.retain_mass,
        max_support_factor=args.max_support_factor,
        random_state=seed,
        warm_start_local=True,
        determinant_sign=determinant_sign,
        **PROFILES[args.profile],
    )
    model.fit(
        movement_3d,
        movement_assignments,
        neural_3d,
        neural_assignments,
        X_transform=target_transform,
        Rgt=np.asarray(oracle_rotation).T,
        initial_rotation=np.asarray(forward["rotation_R"]).T,
        initial_transport=np.asarray(forward["transport_P"]).T,
    )
    forward_rotation = np.asarray(forward["rotation_R"])
    forward_transport = np.asarray(forward["transport_P"])
    rotation_cycle = model.Rg @ forward_rotation
    return {
        "reverse_transport_objective": model.diagnostics["transport_objective"],
        "bidirectional_transport_objective": float(
            forward["transport_objective"] + model.diagnostics["transport_objective"]
        ),
        "rotation_cycle_error": float(
            np.linalg.norm(rotation_cycle - np.eye(rotation_cycle.shape[0]), "fro")
        ),
        "transport_transpose_error": float(
            np.linalg.norm(model.P - forward_transport.T, "fro")
        ),
        "reverse_consensus_weighted_rms": model.diagnostics[
            "local_global_consensus_weighted_rms"
        ],
        "reverse_rotation_determinant": model.diagnostics["rotation_determinant"],
    }


def _representative_orientation_selector(
    source: np.ndarray,
    source_assignments: np.ndarray,
    target: np.ndarray,
    target_assignments: np.ndarray,
    candidate_transports: list[np.ndarray],
) -> dict[str, Any]:
    """Infer the O(3) component from matched soft-group representative orientation."""
    if source.shape[1] != 3 or target.shape[1] != 3:
        raise ValueError("representative orientation currently requires 3D features")
    if source_assignments.shape[1] != 4 or target_assignments.shape[1] != 4:
        raise ValueError("representative orientation currently requires exactly 4 groups")

    def _standardize(values: np.ndarray) -> np.ndarray:
        return (values - values.mean(axis=0, keepdims=True)) / np.maximum(
            values.std(axis=0, keepdims=True), 1e-12
        )

    def _representatives(values: np.ndarray, assignments: np.ndarray) -> np.ndarray:
        normalized = assignments / np.maximum(
            assignments.sum(axis=0, keepdims=True), 1e-12
        )
        return normalized.T @ _standardize(values)

    source_representatives = _representatives(source, source_assignments)
    target_representatives = _representatives(target, target_assignments)
    mean_transport = np.mean(np.asarray(candidate_transports), axis=0)
    rows, columns = linear_sum_assignment(-mean_transport)
    target_order = columns[np.argsort(rows)]
    source_simplex = (source_representatives[1:] - source_representatives[0]).T
    matched_target = target_representatives[target_order]
    target_simplex = (matched_target[1:] - matched_target[0]).T
    source_volume = float(np.linalg.det(source_simplex))
    target_volume = float(np.linalg.det(target_simplex))
    orientation_product = source_volume * target_volume
    if abs(orientation_product) <= 1e-10:
        raise ValueError("representative simplex orientation is numerically degenerate")
    return {
        "representative_orientation_sign": int(np.sign(orientation_product)),
        "representative_source_oriented_volume": source_volume,
        "representative_target_oriented_volume": target_volume,
        "representative_orientation_product": float(orientation_product),
        "representative_target_order": target_order.astype(int).tolist(),
    }


def _hard_baseline(
    neural_3d: np.ndarray,
    movement_3d: np.ndarray,
    target_transform: np.ndarray,
    oracle_rotation: np.ndarray,
    evaluation_args: dict,
    seed: int,
    args: argparse.Namespace,
) -> dict:
    neural_groups = learn_soft_groups(
        neural_3d,
        n_groups=args.groups,
        temperature=0.50,
        entropy_weight=args.entropy_weight,
        seed=seed,
    )
    movement_groups = learn_soft_groups(
        movement_3d,
        n_groups=args.groups,
        temperature=0.50,
        entropy_weight=args.entropy_weight,
        seed=seed,
    )
    result, _ = run_hard(
        "prototype_hard",
        neural_3d,
        np.argmax(neural_groups.assignments, axis=1),
        movement_3d,
        np.argmax(movement_groups.assignments, axis=1),
        target_transform,
        oracle_rotation,
        seed,
        args.profile,
        evaluation_args,
    )
    return result


def _choose_by_majority(candidates: list[dict], proxy_keys: list[str]) -> dict:
    wins = {candidate["method"]: 0 for candidate in candidates}
    for key in proxy_keys:
        winner = min(candidates, key=lambda candidate: candidate[key])
        wins[winner["method"]] += 1
    return min(
        candidates,
        key=lambda candidate: (
            -wins[candidate["method"]],
            candidate["transport_objective"],
        ),
    )


def _selector_summary(final_records: list[dict]) -> dict[str, Any]:
    seeds = sorted({int(record["seed"]) for record in final_records})
    selectors: dict[str, Any] = {}
    proxy_keys = BASE_PROXY_KEYS.copy()
    if final_records and all(
        all(key in record for key in REVERSE_PROXY_KEYS) for record in final_records
    ):
        proxy_keys.extend(REVERSE_PROXY_KEYS)
    has_representative_selector = final_records and all(
        "representative_orientation_sign" in record for record in final_records
    )
    selector_keys = proxy_keys.copy()
    if has_representative_selector:
        selector_keys.append("representative_orientation_sign")
    selector_keys.append("proxy_majority_vote")
    for selector in selector_keys:
        chosen = []
        for seed in seeds:
            candidates = [record for record in final_records if int(record["seed"]) == seed]
            if selector == "proxy_majority_vote":
                winner = _choose_by_majority(candidates, proxy_keys)
            elif selector == "representative_orientation_sign":
                predicted_sign = candidates[0]["representative_orientation_sign"]
                winner = next(
                    candidate for candidate in candidates
                    if candidate["requested_determinant_sign"] == predicted_sign
                )
            else:
                winner = min(candidates, key=lambda candidate: candidate[selector])
            chosen.append(winner)
        accuracy = np.asarray([record["after_direction_accuracy"] for record in chosen])
        r2 = np.asarray([record["after_velocity_r2"] for record in chosen])
        selectors[selector] = {
            "selected_methods": [record["method"] for record in chosen],
            "selected_determinant_signs": [record["requested_determinant_sign"] for record in chosen],
            "mean_direction_accuracy": float(accuracy.mean()),
            "minimum_direction_accuracy": float(accuracy.min()),
            "high_branch_count_accuracy_gt_0.5": int(np.sum(accuracy > 0.5)),
            "mean_movement_r2": float(r2.mean()),
        }

    oracle = []
    for seed in seeds:
        candidates = [record for record in final_records if int(record["seed"]) == seed]
        oracle.append(max(candidates, key=lambda candidate: candidate["after_direction_accuracy"]))
    oracle_accuracy = np.asarray([record["after_direction_accuracy"] for record in oracle])
    selectors["label_using_oracle_not_a_valid_selector"] = {
        "selected_methods": [record["method"] for record in oracle],
        "mean_direction_accuracy": float(oracle_accuracy.mean()),
        "minimum_direction_accuracy": float(oracle_accuracy.min()),
        "high_branch_count_accuracy_gt_0.5": int(np.sum(oracle_accuracy > 0.5)),
    }
    return selectors


def _plot(final_records: list[dict], out_path: Path) -> None:
    seeds = sorted({int(record["seed"]) for record in final_records})
    objective_key = (
        "bidirectional_transport_objective"
        if final_records and all("bidirectional_transport_objective" in record for record in final_records)
        else "transport_objective"
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), facecolor="white", constrained_layout=True)
    x = np.arange(len(seeds))
    for sign, label, color, marker in [(-1, "det(R) = −1", "#1f77b4", "o"), (1, "det(R) = +1", "#d62728", "s")]:
        records = [
            next(record for record in final_records if int(record["seed"]) == seed and record["requested_determinant_sign"] == sign)
            for seed in seeds
        ]
        axes[0].plot(x, [record["after_direction_accuracy"] for record in records], marker=marker, color=color, label=label)
        axes[1].plot(x, [record[objective_key] for record in records], marker=marker, color=color, label=label)
    axes[0].set(title="Candidate coverage", ylabel="direction accuracy")
    axes[1].set(
        title=(
            "Bidirectional label-free objective"
            if objective_key == "bidirectional_transport_objective"
            else "Label-free transport objective"
        ),
        ylabel=(
            "forward + reverse transport cost"
            if objective_key == "bidirectional_transport_objective"
            else "Σ Pᵢⱼ Cᵢⱼ"
        ),
    )
    for axis in axes:
        axis.set_facecolor("white")
        axis.set_xticks(x, seeds)
        axis.set_xlabel("seed")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    fig.savefig(out_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    neural_3d, movement_3d, target_transform, oracle_rotation, evaluation_args = _load_problem()
    hard_results = []
    stage_results = []

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
        hard_results.append(hard)
        neural_assignments = _fixed_assignments(
            neural_3d, args.groups, args.entropy_weight, seed
        )
        movement_assignments = _fixed_assignments(
            movement_3d, args.groups, args.entropy_weight, seed
        )

        seed_final_records = []
        for sign in (-1, 1):
            method = f"component_{'negative' if sign == -1 else 'positive'}"
            stages = _run_chain(
                method=method,
                neural_3d=neural_3d,
                movement_3d=movement_3d,
                neural_assignments=neural_assignments,
                movement_assignments=movement_assignments,
                hard_result=hard,
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
            if args.with_reverse:
                final.update(
                    _reverse_diagnostics(
                        forward=final,
                        neural_3d=neural_3d,
                        movement_3d=movement_3d,
                        neural_assignments=neural_assignments[ANNEALING_PATH[-1]],
                        movement_assignments=movement_assignments[ANNEALING_PATH[-1]],
                        target_transform=target_transform,
                        oracle_rotation=oracle_rotation,
                        seed=seed,
                        args=args,
                        determinant_sign=sign,
                    )
                )
            seed_final_records.append(final)
            stage_results.extend(stages)
            print(
                f"  det={sign:+d} acc={final['after_direction_accuracy']:.4f} "
                f"R2={final['after_velocity_r2']:.4f} "
                f"objective={final['transport_objective']:.6f} "
                f"consensus={final['local_global_consensus_weighted_rms']:.6f}"
                + (
                    f" cycle={final['rotation_cycle_error']:.6f}"
                    if args.with_reverse
                    else ""
                )
            )

        representative_selector = _representative_orientation_selector(
            neural_3d,
            neural_assignments[ANNEALING_PATH[-1]],
            movement_3d,
            movement_assignments[ANNEALING_PATH[-1]],
            [np.asarray(record["transport_P"]) for record in seed_final_records],
        )
        for record in seed_final_records:
            record.update(representative_selector)
        print(
            "  representative orientation selects "
            f"det={representative_selector['representative_orientation_sign']:+d}"
        )

    final_records = [record for record in stage_results if record["temperature"] == ANNEALING_PATH[-1]]
    suffix = f"_{args.tag}" if args.tag else ""
    raw_path = RESULTS_DIR / f"component_aware_soft_hiwa{suffix}.json"
    summary_path = RESULTS_DIR / f"component_aware_soft_hiwa_summary{suffix}.json"
    figure_path = FIGURES_DIR / f"component_aware_soft_hiwa{suffix}.png"
    payload = {
        "experiment": "component_aware_soft_hiwa",
        "status": "mechanism_development",
        "label_usage": "labels are excluded from fitting and proxy selection; used only for evaluation",
        "parameters": {
            "seeds": args.seeds,
            "profile": args.profile,
            "profile_parameters": PROFILES[args.profile],
            "annealing_path": ANNEALING_PATH,
            "determinant_signs": [-1, 1],
            "anchor_weight": args.anchor_weight,
            "with_reverse": args.with_reverse,
            "groups": args.groups,
            "entropy_weight": args.entropy_weight,
            "retain_mass": args.retain_mass,
            "max_support_factor": args.max_support_factor,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "hard_baselines": hard_results,
        "stage_results": stage_results,
    }
    write_json(raw_path, payload)
    write_json(
        summary_path,
        {
            "experiment": payload["experiment"],
            "selectors": _selector_summary(final_records),
        },
    )
    _plot(final_records, figure_path)
    print(f"\nSaved {raw_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
