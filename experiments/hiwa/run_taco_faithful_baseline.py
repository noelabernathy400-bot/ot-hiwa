"""Frozen neural comparison: Hard HiWA, Soft-GCOT HiWA, and Soft-GCOT HiWA + ROCA.

The two Soft-GCOT baselines use only learned prototypes, normalized soft group
measures, per-pair Q_ij, group P, and the inherited HiWA ADMM consensus.  They
set every representative/anchor/component-conditioned term to zero.  ROCA only
chooses between det(R)=-1 and +1 from unlabeled representative orientation.
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
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FactorAnalysis

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ROCA_DIR = HERE.parent / "roca"
if str(ROCA_DIR) not in sys.path:
    sys.path.insert(0, str(ROCA_DIR))

from common import FIGURES_DIR, RESULTS_DIR, ensure_output_dirs, write_json
from run_neural import least_squares_rotation, load_demo, movement_to_3d, remove_constant_columns
from run_soft_neural import PROFILES, run_hard, run_soft
from soft_groups import assignment_entropy, learn_soft_groups

# Historical bounded smoke profile retained only for reproducing its prior output.
PROFILES.setdefault(
    "baseline-mini",
    dict(maxiter=6, tol=1e-2, mu=2e-2, shorn_maxiter=30, sa_maxiter=4, sa_shorn_maxiter=12),
)
# Fixed numerical budget for a comparable baseline audit.  It changes no loss,
# prototype, OT, or ROCA setting; it only gives ADMM enough iterations to meet
# the common stopping rule.
PROFILES.setdefault(
    "audit",
    dict(maxiter=200, tol=1e-2, mu=5e-2, shorn_maxiter=300, sa_maxiter=40, sa_shorn_maxiter=80),
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="audit")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--entropy-weight", type=float, default=0.05)
    parser.add_argument("--retain-mass", type=float, default=0.90)
    parser.add_argument("--max-support-factor", type=float, default=1.5)
    parser.add_argument("--max-samples", type=int, default=96, help="Deterministic per-domain neural smoke subset; 0 uses all samples.")
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--sync-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="commit and push only this run's JSON and figures after a successful run",
    )
    return parser.parse_args()


def _problem(max_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    data = load_demo()
    neural_all = FactorAnalysis(n_components=3, random_state=0).fit_transform(
        remove_constant_columns(data["test_neural"])
    )
    source_index = np.arange(neural_all.shape[0]) if max_samples <= 0 else np.linspace(0, neural_all.shape[0] - 1, min(max_samples, neural_all.shape[0]), dtype=int)
    target_index = np.arange(data["train_movement"].shape[0]) if max_samples <= 0 else np.linspace(0, data["train_movement"].shape[0] - 1, min(max_samples, data["train_movement"].shape[0]), dtype=int)
    neural = neural_all[source_index]
    movement_xy = data["train_movement"][target_index]
    movement = movement_to_3d(movement_xy)
    target_transform = np.linalg.pinv(movement) @ movement_xy
    oracle_rotation = least_squares_rotation(movement_to_3d(data["test_movement"])[source_index], neural)
    evaluation = {
        "movement_xy": data["test_movement"][source_index],
        "neural_labels": data["test_labels"][source_index],
        "movement_labels": data["train_labels"][target_index],
    }
    return neural, movement, target_transform, oracle_rotation, evaluation


def _simplex_diagnostics(values: np.ndarray, assignments: np.ndarray) -> dict:
    standardized = (values - values.mean(axis=0)) / np.maximum(values.std(axis=0), 1e-12)
    representatives = (assignments / np.maximum(assignments.sum(axis=0), 1e-12)).T @ standardized
    simplex = (representatives[1:] - representatives[0]).T
    singular = np.linalg.svd(simplex, compute_uv=False)
    return {
        "representatives": representatives,
        "volume": float(np.linalg.det(simplex)),
        "abs_volume": float(abs(np.linalg.det(simplex))),
        "condition_number": float(singular.max() / max(singular.min(), 1e-12)),
        "mean_assignment_entropy": float(assignment_entropy(assignments).mean()),
    }


def _roca_selector(source: np.ndarray, a: np.ndarray, target: np.ndarray, b: np.ndarray, transports: list[np.ndarray]) -> dict:
    src = _simplex_diagnostics(source, a)
    tgt = _simplex_diagnostics(target, b)
    rows, cols = linear_sum_assignment(-np.mean(transports, axis=0))
    order = cols[np.argsort(rows)]
    source_simplex = (src["representatives"][1:] - src["representatives"][0]).T
    matched = tgt["representatives"][order]
    target_simplex = (matched[1:] - matched[0]).T
    source_volume = float(np.linalg.det(source_simplex))
    target_volume = float(np.linalg.det(target_simplex))
    product = source_volume * target_volume
    if abs(product) <= 1e-10:
        raise ValueError("ROCA representative simplex is numerically degenerate")
    margin = abs(source_volume) * abs(target_volume)
    reasons = []
    if margin < 1.0:
        reasons.append("low_oriented_volume_margin")
    if src["condition_number"] > 50 or tgt["condition_number"] > 50:
        reasons.append("ill_conditioned_simplex")
    if src["mean_assignment_entropy"] > 0.45 or tgt["mean_assignment_entropy"] > 0.45:
        reasons.append("high_assignment_entropy")
    return {
        "selected_determinant_sign": int(np.sign(product)),
        "source_volume": source_volume,
        "target_volume": target_volume,
        "volume_product_margin": margin,
        "source_condition_number": src["condition_number"],
        "target_condition_number": tgt["condition_number"],
        "warning": bool(reasons),
        "warning_reasons": reasons,
        "target_order": order.astype(int).tolist(),
    }


def _plot(records: list[dict], path: Path) -> None:
    names = ["hard_hiwa", "soft_gcot_full", "soft_gcot_sparse", "soft_gcot_roca"]
    labels = ["Hard\nHiWA", "Soft-GCOT\nHiWA", "Sparse\napproximation", "Soft-GCOT HiWA\n+ ROCA"]
    means = {name: [] for name in names}
    for row in records:
        means[row["method"]].append(row["after_direction_accuracy"])
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.bar(labels, [float(np.mean(means[name])) for name in names], color=["#666666", "#377eb8", "#4daf4a", "#984ea3"])
    ax.set(ylabel="direction accuracy", title="Frozen TACO-faithful Soft-GCOT HiWA baseline")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_convergence(records: list[dict], path: Path) -> None:
    """Plot mean outer-loop traces; individual traces remain in the JSON."""
    names = ["hard_hiwa", "soft_gcot_full", "soft_gcot_sparse", "soft_gcot_roca"]
    labels = {
        "hard_hiwa": "Hard HiWA",
        "soft_gcot_full": "Soft-GCOT HiWA",
        "soft_gcot_sparse": "Sparse approximation",
        "soft_gcot_roca": "Soft-GCOT HiWA + ROCA",
    }
    curves = (
        ("admm_global_residual_curve", "global residual"),
        ("admm_primal_residual_curve", "primal residual"),
        ("admm_dual_residual_curve", "dual residual"),
        ("transport_objective_curve", "transport objective"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, curves):
        for name in names:
            values = [np.asarray(row[key], dtype=float) for row in records if row["method"] == name]
            if not values:
                continue
            padded = np.full((len(values), max(map(len, values))), np.nan)
            for index, value in enumerate(values):
                padded[index, :len(value)] = value
            axis.plot(np.nanmean(padded, axis=0), label=labels[name])
        axis.set(title=title, xlabel="outer iteration")
        if "residual" in key:
            axis.set_yscale("log")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.groups != 4:
        raise ValueError("ROCA baseline requires exactly four groups in 3D")
    ensure_output_dirs()
    neural, movement, target_transform, oracle_rotation, evaluation = _problem(args.max_samples)
    records: list[dict] = []
    roca: list[dict] = []
    for seed in args.seeds:
        source_groups = learn_soft_groups(neural, args.groups, args.temperature, args.entropy_weight, seed=seed)
        target_groups = learn_soft_groups(movement, args.groups, args.temperature, args.entropy_weight, seed=seed)
        hard, _ = run_hard("hard_hiwa", neural, np.argmax(source_groups.assignments, axis=1), movement, np.argmax(target_groups.assignments, axis=1), target_transform, oracle_rotation, seed, args.profile, evaluation)
        records.append(hard)
        hard["display_name"] = "Hard HiWA"
        full, _ = run_soft(neural, source_groups.assignments, movement, target_groups.assignments, target_transform, oracle_rotation, seed, args.profile, args.retain_mass, args.max_support_factor, evaluation, method="soft_gcot_full", support_mode="full")
        sparse, _ = run_soft(neural, source_groups.assignments, movement, target_groups.assignments, target_transform, oracle_rotation, seed, args.profile, args.retain_mass, args.max_support_factor, evaluation, method="soft_gcot_sparse", support_mode="sparse")
        records.extend([full, sparse])
        full["display_name"] = "Soft-GCOT HiWA"
        sparse["display_name"] = "Soft-GCOT HiWA (sparse approximation)"
        candidates = []
        for sign in (-1, 1):
            candidate, _ = run_soft(neural, source_groups.assignments, movement, target_groups.assignments, target_transform, oracle_rotation, seed, args.profile, args.retain_mass, args.max_support_factor, evaluation, method=f"soft_gcot_roca_candidate_det_{sign:+d}", determinant_sign=sign, support_mode="full")
            candidates.append(candidate)
        selection = _roca_selector(neural, source_groups.assignments, movement, target_groups.assignments, [row["transport_P"] for row in candidates])
        chosen = next(row for row in candidates if int(round(row["rotation_determinant"])) == selection["selected_determinant_sign"])
        chosen["method"] = "soft_gcot_roca"
        chosen["display_name"] = "Soft-GCOT HiWA + ROCA"
        chosen["roca"] = selection
        records.append(chosen)
        roca.append({"seed": seed, **selection, "candidate_determinants": [row["rotation_determinant"] for row in candidates]})
        print(f"seed={seed} hard={hard['after_direction_accuracy']:.3f} full={full['after_direction_accuracy']:.3f} sparse={sparse['after_direction_accuracy']:.3f} roca={chosen['after_direction_accuracy']:.3f}")
    suffix = f"_{args.tag}" if args.tag else ""
    json_path = RESULTS_DIR / f"taco_faithful_baseline{suffix}.json"
    figure_path = FIGURES_DIR / f"taco_faithful_baseline{suffix}.png"
    convergence_path = FIGURES_DIR / f"taco_faithful_convergence{suffix}.png"
    payload = {
        "experiment": "taco_faithful_soft_gcot_hiwa_neural_baseline",
        "method_names": {
            "hard_hiwa": "Hard HiWA",
            "soft_gcot_full": "Soft-GCOT HiWA",
            "soft_gcot_sparse": "Soft-GCOT HiWA (sparse approximation)",
            "soft_gcot_roca": "Soft-GCOT HiWA + ROCA",
        },
        "label_usage": "labels are used only for final direction-accuracy and movement-R2 evaluation; never for fitting, branch selection, or hyperparameter selection",
        "frozen_configuration": {"representative_guidance_weight": 0.0, "representative_rotation_weight": 0.0, "component_conditioning_weight": 0.0, "rotation_anchor_weight": 0.0, "joint_prototypes": False},
        "parameters": vars(args),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "convergence_rule": "A run is comparable only when both global and primal ADMM residuals are at or below tol; dual residual and objective are diagnostic traces.",
        "results": records,
        "roca": roca,
    }
    write_json(json_path, payload)
    _plot(records, figure_path)
    _plot_convergence(records, convergence_path)
    print(f"saved: {json_path}\nsaved: {figure_path}\nsaved: {convergence_path}")
    if args.sync_results:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools" / "sync_result_artifacts.py"),
                str(json_path),
                str(figure_path),
                str(convergence_path),
                "--message",
                "record soft-gcot audit results",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
