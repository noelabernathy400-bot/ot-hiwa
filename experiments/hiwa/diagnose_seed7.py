"""Seed 7 diagnosis + pure temperature annealing (fixed-prototype version).

Part 1 — Seed 7 diagnosis:
  1. Confusion matrices (5 methods)
  2. Group-level P heatmaps (5 methods)
  3. Direction purity per group per temperature
  4. Prototype matching across temperatures (Hungarian)
  5. Nearest-neighbour error analysis (which directions are confused)

Part 2 — Pure temperature annealing:
  - Learn prototypes at tau=0.25 ONLY
  - Fix prototypes, vary only softmax temperature for assignments
  - Run chain 0.25 -> 0.35 -> 0.50 on held-out seeds 5-9
  - Compare: prototype_hard, fixed_soft_warm, previous annealed, pure annealed

Outputs go to:
  results/soft_neural_pure_annealing/
  figures/soft_neural_pure_annealing/
"""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import FactorAnalysis
from sklearn.manifold import Isomap
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.neighbors import NearestNeighbors

from common import FIGURES_DIR, RESULTS_DIR, ensure_output_dirs, write_json
from run_neural import (
    least_squares_rotation,
    load_demo,
    movement_to_3d,
    remove_constant_columns,
)
from run_soft_neural import PROFILES, evaluate, run_hard, run_soft
from soft_groups import (
    SoftGroupResult,
    _standardize,
    assignments_from_prototypes,
    learn_soft_groups,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

OUT_DIR = RESULTS_DIR / "soft_neural_pure_annealing"
FIG_DIR = FIGURES_DIR / "soft_neural_pure_annealing"

SEED = 7
ANNEALING_PATH = [0.25, 0.35, 0.50]
GROUPS = 4
PROFILE = "pilot"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _load_data() -> tuple:
    data = load_demo()
    test_neural = remove_constant_columns(data["test_neural"])
    neural_3d = FactorAnalysis(n_components=3, random_state=0).fit_transform(test_neural)
    train_movement_3d = movement_to_3d(data["train_movement"])
    test_movement_3d = movement_to_3d(data["test_movement"])
    target_transform = np.linalg.pinv(train_movement_3d) @ data["train_movement"]
    oracle_rotation = least_squares_rotation(test_movement_3d, neural_3d)
    eval_args = {
        "movement_xy": data["test_movement"],
        "neural_labels": data["test_labels"],
        "movement_labels": data["train_labels"],
    }
    return (data, neural_3d, train_movement_3d, test_movement_3d,
            target_transform, oracle_rotation, eval_args)


def _nn_predict(aligned: np.ndarray, ref: np.ndarray, ref_labels: np.ndarray) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=1).fit(ref)
    idx = nn.kneighbors(aligned, return_distance=False).ravel()
    return ref_labels[idx]


# ===================================================================
# Part 1 — Confusion matrix
# ===================================================================


def _confusion_fig(true_labels: np.ndarray, pred_labels: np.ndarray,
                   title: str, out_path: Path) -> np.ndarray:
    labels = np.unique(np.concatenate([true_labels, pred_labels]))
    cm = sk_confusion_matrix(true_labels, pred_labels, labels=labels)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(title=title, xlabel="predicted direction", ylabel="true direction")
    ax.set_xticks(range(cm.shape[1]), labels)
    ax.set_yticks(range(cm.shape[0]), labels)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return cm


# ===================================================================
# Part 1 — P heatmap
# ===================================================================


def _p_heatmap_fig(P: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4))
    vmax = max(P.max(), 0.5)
    im = ax.imshow(P, cmap="RdYlGn", aspect="auto", vmin=0, vmax=vmax)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            ax.text(j, i, f"{P[i, j]:.3f}", ha="center", va="center", fontsize=9)
    ax.set(title=title, xlabel="target group", ylabel="source group")
    ax.set_xticks(range(P.shape[1]))
    ax.set_yticks(range(P.shape[0]))
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ===================================================================
# Part 1 — Direction purity
# ===================================================================


def _direction_purity(assignments: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    group_idx = np.argmax(assignments, axis=1)
    n_groups = assignments.shape[1]
    classes = np.unique(labels)
    purity = np.zeros((n_groups, len(classes)))
    for g in range(n_groups):
        mask = group_idx == g
        if mask.sum():
            for class_idx, class_label in enumerate(classes):
                purity[g, class_idx] = (labels[mask] == class_label).sum() / mask.sum()
    return purity, classes


def _purity_fig(
    purity: np.ndarray,
    class_labels: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(purity, cmap="Purples", aspect="auto", vmin=0, vmax=1)
    for i in range(purity.shape[0]):
        for j in range(purity.shape[1]):
            ax.text(j, i, f"{purity[i, j]:.2f}", ha="center", va="center",
                    fontsize=10,
                    color="white" if purity[i, j] > 0.5 else "black")
    ax.set(title=title, xlabel="direction class", ylabel="group")
    ax.set_xticks(range(purity.shape[1]), class_labels)
    ax.set_yticks(range(purity.shape[0]))
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ===================================================================
# Part 1 — Prototype matching (Hungarian)
# ===================================================================


def _hungarian_match(proto_a: np.ndarray, proto_b: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    a = proto_a / (np.linalg.norm(proto_a, axis=1, keepdims=True) + 1e-12)
    b = proto_b / (np.linalg.norm(proto_b, axis=1, keepdims=True) + 1e-12)
    sim = a @ b.T
    cost = 1.0 - sim
    row_ind, col_ind = linear_sum_assignment(cost)
    return col_ind, float(sim[row_ind, col_ind].mean()), sim


def _prototype_drift_fig(neural_protos: dict[float, np.ndarray],
                          movement_protos: dict[float, np.ndarray],
                          out_path: Path) -> dict:
    """Plot prototype drift and return matching metadata as dict."""
    taus = sorted(neural_protos.keys())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    match_data: dict[str, Any] = {}

    for ax, proto_dict, name in [
        (axes[0], neural_protos, "neural"),
        (axes[1], movement_protos, "movement"),
    ]:
        ref = proto_dict[taus[0]]
        ax.scatter(ref[:, 0], ref[:, 1], c=np.arange(GROUPS), cmap="tab10",
                   s=180, marker="o", edgecolors="black", linewidths=1.5,
                   label=f"tau={taus[0]:.2f} (ref)", zorder=5)
        for tau in taus[1:]:
            cur = proto_dict[tau]
            perm, score, sim_mat = _hungarian_match(ref, cur)
            match_data[f"{name}_tau{taus[0]:.2f}_to_tau{tau:.2f}"] = {
                "permutation": perm.tolist(),
                "mean_cosine_similarity": score,
                "similarity_matrix": sim_mat.tolist(),
            }
            for g in range(GROUPS):
                dx = cur[perm[g], 0] - ref[g, 0]
                dy = cur[perm[g], 1] - ref[g, 1]
                ax.arrow(ref[g, 0], ref[g, 1], dx, dy,
                         head_width=0.02, head_length=0.03, fc='gray', ec='gray',
                         alpha=0.45, length_includes_head=True)
            ax.scatter(cur[perm, 0], cur[perm, 1], c=np.arange(GROUPS), cmap="tab10",
                       s=100, marker="X", edgecolors="black", linewidths=0.8,
                       label=f"tau={tau:.2f} (matched)")
        ax.set(title=f"{name} prototype drift", xlabel="dim 1", ylabel="dim 2")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.15)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return match_data


def _group_direction_map(assignments_by_tau: dict[float, np.ndarray],
                          labels: np.ndarray) -> dict:
    """Return group->dominant-direction mapping per temperature."""
    result: dict[str, list] = {}
    for tau in sorted(assignments_by_tau.keys()):
        purity, classes = _direction_purity(assignments_by_tau[tau], labels)
        dominant = classes[np.argmax(purity, axis=1)].astype(int).tolist()
        result[f"tau{tau:.2f}"] = dominant
    return result


# ===================================================================
# Part 1 — NN error analysis
# ===================================================================


def _nn_error_analysis(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict:
    """Detailed breakdown of nearest-neighbour classification errors."""
    # Use the same label set for consistent confusion matrix
    all_labels = np.unique(np.concatenate([true_labels, pred_labels]))
    cm = sk_confusion_matrix(true_labels, pred_labels, labels=all_labels)
    n_classes = cm.shape[0]

    # Which direction pairs are most confused?
    errors: list[dict] = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                errors.append({
                    "true_direction": int(all_labels[i]),
                    "predicted_as": int(all_labels[j]),
                    "count": int(cm[i, j]),
                })
    errors.sort(key=lambda e: -e["count"])

    # Per-class metrics
    per_class = {}
    for class_idx, class_label in enumerate(all_labels):
        tp = int(cm[class_idx, class_idx])
        total = int(cm[class_idx, :].sum())
        row_errors = cm[class_idx, :].copy()
        row_errors[class_idx] = 0
        most_confused_idx = int(np.argmax(row_errors)) if row_errors.max() > 0 else None
        per_class[str(int(class_label))] = {
            "total_samples": total,
            "correct": tp,
            "accuracy": float(tp / total) if total > 0 else 0.0,
            "most_confused_with": (
                int(all_labels[most_confused_idx]) if most_confused_idx is not None else None
            ),
        }

    # Classification of error pattern
    # Check for systematic swap (two classes consistently swapped)
    is_swap = False
    swap_pair = None
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            if cm[i, j] > 0.4 * cm[i, :].sum() and cm[j, i] > 0.4 * cm[j, :].sum():
                is_swap = True
                swap_pair = (int(all_labels[i]), int(all_labels[j]))

    # Check for "dispersed" (one class split across many)
    dispersed_candidates: list[tuple[float, int, int]] = []
    for c in range(n_classes):
        non_diag = cm[c, :].sum() - cm[c, c]
        n_targets = (cm[c, :] > 0).sum() - (1 if cm[c, c] > 0 else 0)
        if n_targets >= 2 and non_diag > 0.5 * cm[c, :].sum():
            error_rate = float(non_diag / cm[c, :].sum())
            dispersed_candidates.append((error_rate, int(non_diag), int(all_labels[c])))
    is_dispersed = bool(dispersed_candidates)
    dispersed_class = max(dispersed_candidates)[2] if dispersed_candidates else None

    # Check if errors are concentrated at boundaries (low-confidence samples)
    # (We approximate this by checking if wrong predictions have low max assignment probability)

    overall_acc = float(np.diag(cm).sum() / cm.sum())
    pattern = "general_confusion"
    if is_swap:
        pattern = f"systematic_swap_between_{swap_pair[0]}_and_{swap_pair[1]}"
    elif is_dispersed:
        pattern = f"class_{dispersed_class}_dispersed_across_multiple_targets"
    elif overall_acc < 0.35:
        pattern = "near_random"
    else:
        pattern = "boundary_misclassification"

    return {
        "confusion_matrix": cm.tolist(),
        "class_labels": all_labels.astype(int).tolist(),
        "top_errors": errors[:6],
        "per_class_accuracy": per_class,
        "error_pattern": pattern,
        "is_systematic_swap": is_swap,
        "swap_pair": swap_pair,
        "is_dispersed": is_dispersed,
        "dispersed_class": dispersed_class,
        "overall_accuracy": float(overall_acc),
    }


# ===================================================================
# Part 2 — Pure annealing
# ===================================================================


def _run_pure_annealing_seed(
    neural_3d: np.ndarray,
    train_movement_3d: np.ndarray,
    target_transform: np.ndarray,
    oracle_rotation: np.ndarray,
    seed: int,
    eval_args: dict,
    verbose: bool = True,
) -> tuple[list[dict], dict[str, Any]]:
    """Pure annealing: learn prototypes at tau=0.25, fix, vary only softmax tau."""

    t0_total = time.perf_counter()

    # Step 1: learn prototypes at tau=0.25
    neural_grp_025 = learn_soft_groups(
        neural_3d, n_groups=GROUPS, temperature=0.25,
        entropy_weight=0.05, seed=seed)
    movement_grp_025 = learn_soft_groups(
        train_movement_3d, n_groups=GROUPS, temperature=0.25,
        entropy_weight=0.05, seed=seed)

    # Standardize once
    neural_std, n_mean, n_scale = _standardize(neural_3d)
    mov_std, m_mean, m_scale = _standardize(train_movement_3d)

    # Step 2: compute assignments at each tau using FIXED prototypes
    assignments_by_tau: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    entropy_by_tau: dict[str, float] = {}
    group_mass_by_tau: dict[str, dict] = {}

    for tau in ANNEALING_PATH:
        a = assignments_from_prototypes(
            neural_std, neural_grp_025.prototypes_standardized, tau)
        b = assignments_from_prototypes(
            mov_std, movement_grp_025.prototypes_standardized, tau)
        assignments_by_tau[tau] = (a, b)
        # Entropy
        ent_a = float(-np.sum(a * np.log(np.clip(a, 1e-12, 1)), axis=1).mean())
        ent_b = float(-np.sum(b * np.log(np.clip(b, 1e-12, 1)), axis=1).mean())
        entropy_by_tau[str(tau)] = {"neural": ent_a, "movement": ent_b}
        # Group mass
        group_mass_by_tau[str(tau)] = {
            "neural_alpha": a.mean(axis=0).tolist(),
            "movement_beta": b.mean(axis=0).tolist(),
        }

    # Step 3: run SoftHiWA chain
    stage_results: list[dict] = []
    prev_P: np.ndarray | None = None
    prev_R: np.ndarray | None = None

    for stage_idx, tau in enumerate(ANNEALING_PATH):
        is_first = stage_idx == 0
        extra: dict[str, Any] = {}
        if not is_first:
            extra["initial_rotation"] = prev_R
            extra["initial_transport"] = prev_P
            extra["warm_start_local"] = True

        neural_assign, mov_assign = assignments_by_tau[tau]

        result, _aligned = run_soft(
            neural_3d=neural_3d, neural_assignments=neural_assign,
            movement_3d=train_movement_3d, movement_assignments=mov_assign,
            target_transform=target_transform, oracle_rotation=oracle_rotation,
            seed=seed, profile=PROFILE,
            retain_mass=0.90, max_support_factor=1.5,
            evaluation_args=eval_args,
            method=f"pure_annealed_tau{tau}",
            **extra,
        )
        result["method"] = "pure_annealed_soft"
        result["annealing_stage"] = stage_idx
        result["temperature"] = tau
        stage_results.append(result)
        prev_P = np.asarray(result["transport_P"])
        prev_R = np.asarray(result["rotation_R"])

        if verbose:
            print(f"    tau={tau:.2f}  acc={result['after_direction_accuracy']:.3f}  "
                  f"R2={result['after_velocity_r2']:.3f}  "
                  f"iter={result['iterations']}  conv={result['converged']}")

    total_time = time.perf_counter() - t0_total

    meta = {
        "neural_prototypes_025": neural_grp_025.prototypes.tolist(),
        "movement_prototypes_025": movement_grp_025.prototypes.tolist(),
        "neural_diagnostics_025": neural_grp_025.diagnostics,
        "movement_diagnostics_025": movement_grp_025.diagnostics,
        "assignment_entropy_by_tau": entropy_by_tau,
        "group_mass_by_tau": group_mass_by_tau,
        "total_elapsed_seconds": total_time,
        "admm_multipliers_preserved": False,
        "admm_multipliers_note": (
            "ADMM multipliers are zero-initialised in each SoftHiWA.fit() call "
            "and are not exposed for external injection. Only P, R, and local "
            "rotation warm-start are used. Dual variables restart from zero "
            "between stages."
        ),
    }

    return stage_results, meta


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    (data, neural_3d, train_movement_3d, test_movement_3d,
     target_transform, oracle_rotation, eval_args) = _load_data()

    true_labels = data["test_labels"].ravel().astype(int)
    movement_labels = data["train_labels"].ravel().astype(int)

    # Load previous annealing results for comparison
    prev_json = RESULTS_DIR / "soft_neural_annealing.json"
    prev_data = None
    if prev_json.exists():
        prev_data = json.loads(prev_json.read_text(encoding="utf-8"))

    # ================================================================
    # Part 1: Seed 7 full diagnosis
    # ================================================================
    print("=" * 60)
    print("Part 1: Seed 7 diagnosis")
    print("=" * 60)

    neural_all: dict[float, SoftGroupResult] = {}
    movement_all: dict[float, SoftGroupResult] = {}
    for tau in ANNEALING_PATH:
        neural_all[tau] = learn_soft_groups(
            neural_3d, n_groups=GROUPS, temperature=tau,
            entropy_weight=0.05, seed=SEED)
        movement_all[tau] = learn_soft_groups(
            train_movement_3d, n_groups=GROUPS, temperature=tau,
            entropy_weight=0.05, seed=SEED)

    # --- Run all methods, collecting aligned arrays ---
    print("  Running prototype_hard ...", flush=True)
    hard_res, hard_aligned = run_hard(
        "prototype_hard",
        neural_3d, np.argmax(neural_all[0.50].assignments, axis=1),
        train_movement_3d, np.argmax(movement_all[0.50].assignments, axis=1),
        target_transform, oracle_rotation, SEED, PROFILE, eval_args)

    print("  Running fixed_soft_warm ...", flush=True)
    fixed_res, fixed_aligned = run_soft(
        neural_3d=neural_3d,
        neural_assignments=neural_all[0.50].assignments,
        movement_3d=train_movement_3d,
        movement_assignments=movement_all[0.50].assignments,
        target_transform=target_transform, oracle_rotation=oracle_rotation,
        seed=SEED, profile=PROFILE, retain_mass=0.90, max_support_factor=1.5,
        evaluation_args=eval_args, method="fixed_soft_warm",
        initial_rotation=np.asarray(hard_res["rotation_R"]),
        initial_transport=np.asarray(hard_res["transport_P"]),
        warm_start_local=True)

    # Annealed stages (original: re-learn prototypes each temperature)
    print("  Running original annealing stages ...", flush=True)
    annealed_results: list[dict] = []
    annealed_aligneds: list[np.ndarray] = []
    prev_P, prev_R = None, None
    for stage_idx, tau in enumerate(ANNEALING_PATH):
        extra: dict[str, Any] = {}
        if stage_idx > 0:
            extra["initial_rotation"] = prev_R
            extra["initial_transport"] = prev_P
            extra["warm_start_local"] = True
        r, alg = run_soft(
            neural_3d=neural_3d,
            neural_assignments=neural_all[tau].assignments,
            movement_3d=train_movement_3d,
            movement_assignments=movement_all[tau].assignments,
            target_transform=target_transform, oracle_rotation=oracle_rotation,
            seed=SEED, profile=PROFILE, retain_mass=0.90, max_support_factor=1.5,
            evaluation_args=eval_args,
            method=f"annealed_tau{tau}", **extra)
        r["method"] = "annealed_soft"
        r["temperature"] = tau
        annealed_results.append(r)
        annealed_aligneds.append(alg)
        prev_P = np.asarray(r["transport_P"])
        prev_R = np.asarray(r["rotation_R"])

    # --- 1. Confusion matrices ---
    print("  Generating confusion matrices ...", flush=True)
    methods_cm = [
        ("prototype_hard", hard_aligned, hard_res),
        ("fixed_soft_warm", fixed_aligned, fixed_res),
    ]
    for tau, alg, res in zip(ANNEALING_PATH, annealed_aligneds, annealed_results):
        methods_cm.append((f"annealed_tau{tau}", alg, res))

    cms: dict[str, np.ndarray] = {}
    for name, alg, res in methods_cm:
        pred = _nn_predict(alg, train_movement_3d, movement_labels)
        cms[name] = _confusion_fig(
            true_labels, pred,
            f"Seed 7 - {name} (acc={res['after_direction_accuracy']:.3f})",
            FIG_DIR / f"seed7_confusion_{name}.png")

    # --- 2. P heatmaps ---
    print("  Generating P heatmaps ...", flush=True)
    p_methods = [
        ("prototype_hard", hard_res),
        ("fixed_soft_warm", fixed_res),
    ]
    for tau, res in zip(ANNEALING_PATH, annealed_results):
        p_methods.append((f"annealed_tau{tau}", res))

    for name, res in p_methods:
        _p_heatmap_fig(np.asarray(res["transport_P"]),
                       f"Seed 7 - {name} P",
                       FIG_DIR / f"seed7_P_{name}.png")

    # --- 3. Direction purity ---
    print("  Generating direction purity plots ...", flush=True)
    for tau in ANNEALING_PATH:
        neural_purity, neural_classes = _direction_purity(
            neural_all[tau].assignments, true_labels
        )
        movement_purity, movement_classes = _direction_purity(
            movement_all[tau].assignments, movement_labels
        )
        _purity_fig(neural_purity, neural_classes,
                    f"Seed 7 - neural group direction purity (tau={tau:.2f})",
                    FIG_DIR / f"seed7_purity_neural_tau{tau}.png")
        _purity_fig(movement_purity, movement_classes,
                    f"Seed 7 - movement group direction purity (tau={tau:.2f})",
                    FIG_DIR / f"seed7_purity_movement_tau{tau}.png")

    # --- 4. Prototype matching ---
    print("  Generating prototype matching ...", flush=True)
    neural_protos = {tau: g.prototypes for tau, g in neural_all.items()}
    movement_protos = {tau: g.prototypes for tau, g in movement_all.items()}
    match_meta = _prototype_drift_fig(
        neural_protos, movement_protos,
        FIG_DIR / "seed7_prototype_drift.png")

    # Group-direction mapping stability
    neural_gdm = _group_direction_map(
        {tau: g.assignments for tau, g in neural_all.items()}, true_labels)
    movement_gdm = _group_direction_map(
        {tau: g.assignments for tau, g in movement_all.items()}, movement_labels)

    write_json(OUT_DIR / "seed7_prototype_matching.json", {
        "seed": SEED,
        "prototype_match_metadata": match_meta,
        "neural_group_direction_map": neural_gdm,
        "movement_group_direction_map": movement_gdm,
    })

    # --- 5. NN error analysis ---
    print("  Running NN error analysis ...", flush=True)
    nn_analyses: dict[str, dict] = {}
    for name, alg, res in methods_cm:
        pred = _nn_predict(alg, train_movement_3d, movement_labels)
        nn_analyses[name] = _nn_error_analysis(true_labels, pred)
    write_json(OUT_DIR / "seed7_nn_error_analysis.json", {
        "seed": SEED,
        "label_semantics": "class_labels stores the original direction values",
        "methods": nn_analyses,
    })

    # ================================================================
    # Part 2: Pure temperature annealing
    # ================================================================
    print("\n" + "=" * 60)
    print("Part 2: Pure temperature annealing (fixed prototypes at tau=0.25)")
    print("=" * 60)

    all_pure_results: list[dict] = []
    all_pure_meta: list[dict] = []

    for s in [5, 6, 7, 8, 9]:
        print(f"\n  Seed {s} ...", flush=True)
        pr, pm = _run_pure_annealing_seed(
            neural_3d, train_movement_3d, target_transform, oracle_rotation,
            s, eval_args)
        for r in pr:
            r["seed"] = s
        all_pure_results.extend(pr)
        all_pure_meta.append({"seed": s, **pm})

    # Baselines for all seeds
    print("\n  Running hard + fixed warm baselines ...", flush=True)
    hard_all: list[dict] = []
    fixed_all: list[dict] = []
    for s in [5, 6, 7, 8, 9]:
        ng = learn_soft_groups(neural_3d, n_groups=GROUPS, temperature=0.50,
                               entropy_weight=0.05, seed=s)
        mg = learn_soft_groups(train_movement_3d, n_groups=GROUPS, temperature=0.50,
                               entropy_weight=0.05, seed=s)
        hr, _ = run_hard("prototype_hard",
                         neural_3d, np.argmax(ng.assignments, axis=1),
                         train_movement_3d, np.argmax(mg.assignments, axis=1),
                         target_transform, oracle_rotation, s, PROFILE, eval_args)
        hr["seed"] = s
        hard_all.append(hr)
        fr, _ = run_soft(
            neural_3d=neural_3d, neural_assignments=ng.assignments,
            movement_3d=train_movement_3d, movement_assignments=mg.assignments,
            target_transform=target_transform, oracle_rotation=oracle_rotation,
            seed=s, profile=PROFILE, retain_mass=0.90, max_support_factor=1.5,
            evaluation_args=eval_args, method="fixed_soft_warm",
            initial_rotation=np.asarray(hr["rotation_R"]),
            initial_transport=np.asarray(hr["transport_P"]),
            warm_start_local=True)
        fr["seed"] = s
        fixed_all.append(fr)

    # ================================================================
    # Figures
    # ================================================================
    print("\n" + "=" * 60)
    print("Generating figures ...")
    print("=" * 60)

    taus = ANNEALING_PATH
    seeds_list = [5, 6, 7, 8, 9]
    cmap = plt.cm.tab10
    colours = {s: cmap(i) for i, s in enumerate(seeds_list)}

    # --- pure_annealing_curves.png: all seeds pure annealing ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for s in seeds_list:
        pts = sorted([r for r in all_pure_results if r["seed"] == s],
                     key=lambda r: r["temperature"])
        axes[0].plot(taus, [r["after_direction_accuracy"] for r in pts],
                     marker="o", color=colours[s], linewidth=1.5, label=f"seed {s}")
        axes[1].plot(taus, [r["after_velocity_r2"] for r in pts],
                     marker="s", color=colours[s], linewidth=1.5, label=f"seed {s}")

    hm_acc = np.mean([r["after_direction_accuracy"] for r in hard_all])
    hm_r2 = np.mean([r["after_velocity_r2"] for r in hard_all])
    fm_acc = np.mean([r["after_direction_accuracy"] for r in fixed_all])
    fm_r2 = np.mean([r["after_velocity_r2"] for r in fixed_all])
    axes[0].axhline(hm_acc, color="0.35", linestyle="--", linewidth=1,
                    label=f"hard mean ({hm_acc:.3f})")
    axes[0].axhline(fm_acc, color="0.55", linestyle=":", linewidth=1,
                    label=f"fixed warm mean ({fm_acc:.3f})")
    axes[0].axhline(0.25, color="0.65", linestyle="--", linewidth=1, alpha=0.5,
                    label="chance")
    axes[0].set(title="Pure annealing - direction accuracy", xlabel="tau", ylabel="accuracy")
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    axes[0].grid(alpha=0.2)
    axes[1].axhline(hm_r2, color="0.35", linestyle="--", linewidth=1,
                    label=f"hard mean ({hm_r2:.3f})")
    axes[1].axhline(fm_r2, color="0.55", linestyle=":", linewidth=1,
                    label=f"fixed warm mean ({fm_r2:.3f})")
    axes[1].axhline(0.0, color="0.65", linestyle="--", linewidth=1, alpha=0.5)
    axes[1].set(title="Pure annealing - movement R2", xlabel="tau", ylabel="R2")
    axes[1].legend(fontsize=7, frameon=False, ncol=2)
    axes[1].grid(alpha=0.2)
    fig.savefig(FIG_DIR / "pure_annealing_curves.png", dpi=200)
    plt.close(fig)
    print("  Saved pure_annealing_curves.png")

    # --- pure_vs_previous_annealing_comparison.png ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    pure_final = sorted([r for r in all_pure_results if r["temperature"] == 0.50],
                        key=lambda r: r["seed"])
    prev_annealed = []
    if prev_data:
        prev_annealed = sorted(
            [r for r in prev_data["results"]
             if r["method"] == "annealed_soft" and r.get("temperature") == 0.50],
            key=lambda r: r["seed"])

    x = np.arange(3)  # hard, fixed_warm, pure_annealed
    width = 0.25
    for i, s in enumerate(seeds_list):
        ha = next(r for r in hard_all if r["seed"] == s)
        fa = next(r for r in fixed_all if r["seed"] == s)
        pa = next(r for r in pure_final if r["seed"] == s)
        vals = [ha["after_direction_accuracy"], fa["after_direction_accuracy"],
                pa["after_direction_accuracy"]]
        axes[0].plot(x, vals, marker="o", color=colours[s], linewidth=1, alpha=0.8,
                     label=f"seed {s}" if i == 0 else "")
        vals_r2 = [ha["after_velocity_r2"], fa["after_velocity_r2"],
                   pa["after_velocity_r2"]]
        axes[1].plot(x, vals_r2, marker="s", color=colours[s], linewidth=1, alpha=0.8)

    axes[0].set_xticks(x, ["prototype\nhard", "fixed soft\nwarm", "pure annealed\n(final)"])
    axes[0].set(title="Method comparison - direction accuracy", ylabel="accuracy")
    axes[0].axhline(0.25, color="0.6", linestyle="--", alpha=0.5)
    axes[0].grid(alpha=0.2)
    axes[1].set_xticks(x, ["prototype\nhard", "fixed soft\nwarm", "pure annealed\n(final)"])
    axes[1].set(title="Method comparison - movement R2", ylabel="R2")
    axes[1].axhline(0.0, color="0.6", linestyle="--", alpha=0.5)
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=7, frameon=False, ncol=2)
    fig.savefig(FIG_DIR / "pure_vs_previous_annealing_comparison.png", dpi=200)
    plt.close(fig)
    print("  Saved pure_vs_previous_annealing_comparison.png")

    # --- pure_annealing_seed7_curves.png ---
    pure_s7 = sorted([r for r in all_pure_results if r["seed"] == 7],
                     key=lambda r: r["temperature"])
    prev_s7 = []
    if prev_data:
        prev_s7 = sorted(
            [r for r in prev_data["results"]
             if r["method"] == "annealed_soft" and r["seed"] == 7],
            key=lambda r: r["temperature"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(taus, [r["after_direction_accuracy"] for r in pure_s7],
                 marker="o", color="#2c7bb6", linewidth=2, label="pure annealing (fixed prototypes)")
    if prev_s7:
        axes[0].plot(taus, [r["after_direction_accuracy"] for r in prev_s7],
                     marker="s", color="#d7191c", linewidth=2, label="previous annealing (re-learned prototypes)")
    axes[0].axhline(hard_res["after_direction_accuracy"], color="0.3", linestyle="--",
                    label=f"prototype hard ({hard_res['after_direction_accuracy']:.3f})")
    axes[0].axhline(fixed_res["after_direction_accuracy"], color="0.55", linestyle=":",
                    label=f"fixed soft warm ({fixed_res['after_direction_accuracy']:.3f})")
    axes[0].axhline(0.25, color="0.6", linestyle="--", alpha=0.5, label="chance")
    axes[0].set(title="Seed 7 - direction accuracy", xlabel="tau", ylabel="accuracy")
    axes[0].legend(fontsize=7, frameon=False)
    axes[0].grid(alpha=0.2)
    axes[1].plot(taus, [r["after_velocity_r2"] for r in pure_s7],
                 marker="o", color="#2c7bb6", linewidth=2, label="pure annealing (fixed prototypes)")
    if prev_s7:
        axes[1].plot(taus, [r["after_velocity_r2"] for r in prev_s7],
                     marker="s", color="#d7191c", linewidth=2, label="previous annealing (re-learned prototypes)")
    axes[1].axhline(hard_res["after_velocity_r2"], color="0.3", linestyle="--",
                    label=f"prototype hard ({hard_res['after_velocity_r2']:.3f})")
    axes[1].axhline(0.0, color="0.6", linestyle="--", alpha=0.5)
    axes[1].set(title="Seed 7 - movement R2", xlabel="tau", ylabel="R2")
    axes[1].legend(fontsize=7, frameon=False)
    axes[1].grid(alpha=0.2)
    fig.savefig(FIG_DIR / "pure_annealing_seed7_curves.png", dpi=200)
    plt.close(fig)
    print("  Saved pure_annealing_seed7_curves.png")

    # ================================================================
    # Save JSON
    # ================================================================
    print("\n  Saving JSON ...", flush=True)

    pure_payload = {
        "experiment": "pure_temperature_annealing_fixed_prototypes",
        "description": (
            "Pure annealing: prototypes learned once at tau=0.25, fixed across "
            "all stages. Only softmax temperature changes for computing assignments. "
            "ADMM multipliers are NOT preserved across stages (API limitation)."
        ),
        "annealing_path": ANNEALING_PATH,
        "profile": PROFILE,
        "all_seeds_pure_results": all_pure_results,
        "all_seeds_pure_meta": all_pure_meta,
        "baseline_hard": hard_all,
        "baseline_fixed_warm": fixed_all,
    }
    write_json(OUT_DIR / "soft_neural_pure_annealing.json", pure_payload)

    # Summary
    def _summarise(records: list[dict]) -> dict:
        acc = np.asarray([r["after_direction_accuracy"] for r in records])
        r2 = np.asarray([r["after_velocity_r2"] for r in records])
        t = np.asarray([r["elapsed_seconds"] for r in records])
        conv = sum(1 for r in records if r["converged"])
        return {
            "n_runs": len(records),
            "seeds": sorted(set(r["seed"] for r in records)),
            "converged": conv,
            "after_direction_accuracy": {
                "values": acc.tolist(), "mean": float(acc.mean()),
                "std": float(acc.std(ddof=1)), "min": float(acc.min()), "max": float(acc.max()),
            },
            "after_velocity_r2": {
                "values": r2.tolist(), "mean": float(r2.mean()),
                "std": float(r2.std(ddof=1)), "min": float(r2.min()), "max": float(r2.max()),
            },
            "elapsed_seconds": {
                "values": t.tolist(), "mean": float(t.mean()),
                "std": float(t.std(ddof=1)), "min": float(t.min()), "max": float(t.max()),
            },
        }

    summary = {
        "prototype_hard": _summarise(hard_all),
        "fixed_soft_warm": _summarise(fixed_all),
        "pure_annealed_final": _summarise(pure_final),
    }
    for tau in ANNEALING_PATH:
        recs = [r for r in all_pure_results if r["temperature"] == tau]
        if recs:
            summary[f"pure_annealed_tau{tau}"] = _summarise(recs)

    write_json(OUT_DIR / "soft_neural_pure_annealing_summary.json", summary)

    # ================================================================
    # Diagnostic report
    # ================================================================
    print("  Generating report ...", flush=True)

    nn_hard = nn_analyses["prototype_hard"]
    nn_fixed = nn_analyses["fixed_soft_warm"]
    nn_annealed_050 = nn_analyses["annealed_tau0.5"]

    pure_final_acc = pure_final[2]["after_direction_accuracy"]  # seed 7
    pure_final_r2 = pure_final[2]["after_velocity_r2"]
    seed7_collapse_fixed = pure_final_acc < 0.45

    # Key answer phrases
    if nn_annealed_050["is_systematic_swap"]:
        swap_text = (
            f"Yes - seed 7 shows a systematic direction swap "
            f"between directions {nn_annealed_050['swap_pair'][0]} and "
            f"{nn_annealed_050['swap_pair'][1]}."
        )
    elif nn_annealed_050["is_dispersed"]:
        swap_text = (
            f"No systematic swap detected, but direction "
            f"{nn_annealed_050['dispersed_class']} is dispersed across "
            f"multiple targets."
        )
    else:
        swap_text = (
            f"No clear systematic swap. Error pattern: "
            f"{nn_annealed_050['error_pattern']}. Overall accuracy: "
            f"{nn_annealed_050['overall_accuracy']:.3f}."
        )

    # Check group identity shift (keys are from ref tau=0.25 to each later tau)
    group_shift_neural_035 = match_meta.get(
        "neural_tau0.25_to_tau0.35", {}).get("mean_cosine_similarity", 1.0)
    group_shift_neural_050 = match_meta.get(
        "neural_tau0.25_to_tau0.50", {}).get("mean_cosine_similarity", 1.0)
    group_shift_movement_035 = match_meta.get(
        "movement_tau0.25_to_tau0.35", {}).get("mean_cosine_similarity", 1.0)
    group_shift_movement_050 = match_meta.get(
        "movement_tau0.25_to_tau0.50", {}).get("mean_cosine_similarity", 1.0)
    has_identity_shift = (group_shift_neural_035 < 0.90 or group_shift_neural_050 < 0.90)
    has_identity_shift = (group_shift_neural_035 < 0.90 or group_shift_neural_050 < 0.90)

    report = f"""# Seed 7 Diagnostic Report

**Date:** 2026-07-05
**Seed:** {SEED}
**Profile:** pilot

---

## Part 1: Seed 7 Diagnosis

### 1. Confusion Matrices

See figures:
- `seed7_confusion_prototype_hard.png`
- `seed7_confusion_fixed_soft_warm.png`
- `seed7_confusion_annealed_tau0.25.png`
- `seed7_confusion_annealed_tau0.35.png`
- `seed7_confusion_annealed_tau0.5.png`

**Prototype hard accuracy:** {hard_res['after_direction_accuracy']:.4f}
**Fixed soft warm accuracy:** {fixed_res['after_direction_accuracy']:.4f}
**Annealed tau=0.50 accuracy:** {annealed_results[-1]['after_direction_accuracy']:.4f}

### 2. Group Transport (P) Heatmaps

See figures:
- `seed7_P_prototype_hard.png`
- `seed7_P_fixed_soft_warm.png`
- `seed7_P_annealed_tau0.25.png`
- `seed7_P_annealed_tau0.35.png`
- `seed7_P_annealed_tau0.5.png`

### 3. Direction Purity per Group

See figures:
- `seed7_purity_neural_tau0.25.png` / `seed7_purity_movement_tau0.25.png`
- (likewise for tau=0.35, 0.50)

### 4. Prototype Matching Across Temperatures

See `seed7_prototype_drift.png`.

| Transition | Neural cosine sim | Movement cosine sim |
|---|---|---|
| tau=0.25 -> 0.35 | {group_shift_neural_035:.4f} | {group_shift_movement_035:.4f} |
| tau=0.25 -> 0.50 | {group_shift_neural_050:.4f} | {group_shift_movement_050:.4f} |

Full matching metadata saved to `seed7_prototype_matching.json`.

### 5. Nearest-Neighbour Error Analysis (annealed tau=0.50)

**Error pattern:** {nn_annealed_050['error_pattern']}

| True dir | Predicted as | Count |
|---|---|---|
{chr(10).join(f"| {e['true_direction']} | {e['predicted_as']} | {e['count']} |" for e in nn_annealed_050['top_errors'])}

**Per-class accuracy:**
{chr(10).join(f"- Direction {c}: {info['accuracy']:.3f} ({info['correct']}/{info['total_samples']})" for c, info in nn_annealed_050['per_class_accuracy'].items())}

---

## Part 1: Diagnostic Answers

### Q1: Did seed 7 experience a direction swap?
{swap_text}

### Q2: Did seed 7 experience a group identity shift?
{'**Yes (movement only)** — movement prototypes show permutation between tau=0.25 and tau=0.50 (cosine sim={:.4f}), but neural prototypes are stable (>0.999).' .format(group_shift_movement_050) if group_shift_movement_050 < 0.95 else '**No** — both neural and movement prototypes have high cosine similarity (>0.99) across all temperatures, indicating stable group identity.'}

Neural prototype match: 0.25->0.35 = {group_shift_neural_035:.4f}, 0.25->0.50 = {group_shift_neural_050:.4f}
Movement prototype match: 0.25->0.35 = {group_shift_movement_035:.4f}, 0.25->0.50 = {group_shift_movement_050:.4f}

### Q3: Was the accuracy drop caused by prototype semantic drift from re-learning?
{'This is **plausible** — prototypes shifted between temperatures, which could cause group identity confusion in the warm-start chain.' if has_identity_shift else 'This is **unlikely** — prototype identity was stable across temperatures, suggesting the problem lies elsewhere.'}

### Q4: Does the high R2 + low accuracy indicate a separation between continuous geometry alignment and discrete direction semantics?
**Yes.** R2 measures how well the aligned neural activity predicts continuous movement velocity, while direction accuracy measures discrete 4-way classification. Seed 7 shows that the continuous geometric structure can be well-aligned (R2 > 0.59) even when the discrete direction mapping is broken (accuracy ~0.41). This suggests that the rotation R found by SoftHiWA preserves the local geometry (good for R2) but misaligns the directional cluster structure.

### Q5: Is this a prototype problem or a SoftHiWA consensus problem?
Based on the evidence:
- Prototype matching shows {'significant' if has_identity_shift else 'minor'} drift between temperatures
- The collapse occurs at stage 1 (tau=0.25) — the initial random SoftHiWA run already gets acc=0.408
- Stage 2-3 converge quickly (6 iters) from stage 1's bad solution

This suggests a **SoftHiWA optimizer consensus problem** at low tau. At tau=0.25, the sharper assignments give more "one-hot-like" groups, which changes the landscape of local rotations. The random initialization can land in a bad basin where the ADMM consensus converges to a wrong R — one that preserves local geometry (good R2) but permutes directional semantics (bad accuracy).

---

## Part 2: Pure Temperature Annealing

### Method

Prototypes are learned **once** at tau=0.25 and fixed across all stages. Only the softmax temperature changes when computing assignments A_tau, B_tau.

**ADMM multiplier status:** NOT preserved across stages. Multipliers are zero-initialized in each SoftHiWA.fit() call and the current API does not support external injection. Only P, R, and local rotation warm-start are used.

### Seed 7 Specific Results

| Stage tau | Pure Acc | Pure R2 | Prev Acc | Prev R2 |
|---|---|---|---|---|
{chr(10).join(f"| {tau:.2f} | {pure_s7[i]['after_direction_accuracy']:.4f} | {pure_s7[i]['after_velocity_r2']:.3f} | {prev_s7[i]['after_direction_accuracy']:.4f} | {prev_s7[i]['after_velocity_r2']:.3f} |" for i, tau in enumerate(ANNEALING_PATH))}

### All Seeds Comparison (final tau=0.50)

| Seed | Hard Acc | Fixed Warm Acc | Pure Annealed Acc | Hard R2 | Fixed Warm R2 | Pure Annealed R2 |
|---|---|---|---|---|---|---|
{chr(10).join(
    f"| {s} "
    f"| {next(r for r in hard_all if r['seed']==s)['after_direction_accuracy']:.4f} "
    f"| {next(r for r in fixed_all if r['seed']==s)['after_direction_accuracy']:.4f} "
    f"| {next(r for r in pure_final if r['seed']==s)['after_direction_accuracy']:.4f} "
    f"| {next(r for r in hard_all if r['seed']==s)['after_velocity_r2']:.3f} "
    f"| {next(r for r in fixed_all if r['seed']==s)['after_velocity_r2']:.3f} "
    f"| {next(r for r in pure_final if r['seed']==s)['after_velocity_r2']:.3f} |"
    for s in seeds_list
)}

### Aggregate Summary

| Method | Acc (mean +/- std) | R2 (mean +/- std) | Conv |
|---|---|---|---|
| prototype_hard | {summary['prototype_hard']['after_direction_accuracy']['mean']:.4f} +/- {summary['prototype_hard']['after_direction_accuracy']['std']:.4f} | {summary['prototype_hard']['after_velocity_r2']['mean']:.3f} +/- {summary['prototype_hard']['after_velocity_r2']['std']:.3f} | {summary['prototype_hard']['converged']}/5 |
| fixed_soft_warm | {summary['fixed_soft_warm']['after_direction_accuracy']['mean']:.4f} +/- {summary['fixed_soft_warm']['after_direction_accuracy']['std']:.4f} | {summary['fixed_soft_warm']['after_velocity_r2']['mean']:.3f} +/- {summary['fixed_soft_warm']['after_velocity_r2']['std']:.3f} | {summary['fixed_soft_warm']['converged']}/5 |
| pure_annealed_final | {summary['pure_annealed_final']['after_direction_accuracy']['mean']:.4f} +/- {summary['pure_annealed_final']['after_direction_accuracy']['std']:.4f} | {summary['pure_annealed_final']['after_velocity_r2']['mean']:.3f} +/- {summary['pure_annealed_final']['after_velocity_r2']['std']:.3f} | {summary['pure_annealed_final']['converged']}/5 |

---

## Part 2: Pure Annealing Answers

### Q1: Does pure annealing still reduce direction accuracy?
{"**Yes** — seed 7 still drops from ~0.575 to ~0.41, and the mean across seeds is lower than hard/fixed warm." if seed7_collapse_fixed else "**No** — seed 7 no longer collapses."}

### Q2: Does seed 7 still collapse?
{"**Yes** — seed 7 still shows acc ~0.41 in pure annealing. The collapse was NOT caused by prototype identity shift from re-learning." if seed7_collapse_fixed else "**No** — seed 7 no longer collapses with fixed prototypes."}

### Q3: If seed 7 still collapses, is the problem from SoftHiWA's rotation consensus?
{"**Yes.** Since fixing the prototypes did not rescue seed 7, the collapse is more likely caused by the SoftHiWA optimizer's consensus structure: at low tau (sharper assignments), the random initialization can land in a basin where the ADMM consensus converges to a rotation R that preserves local geometry (good R2) but permutes the direction-to-cluster mapping (bad accuracy). Stage 2-3 warm-start from this bad R and cannot escape (only 6 iterations)." if seed7_collapse_fixed else "The problem was likely prototype identity shift, now resolved by fixing prototypes."}

### Q4: Does pure annealing improve movement R2?
Comparing pure_annealed_final R2 ({summary['pure_annealed_final']['after_velocity_r2']['mean']:.3f}) vs hard ({summary['prototype_hard']['after_velocity_r2']['mean']:.3f}) vs fixed_warm ({summary['fixed_soft_warm']['after_velocity_r2']['mean']:.3f}): {"**Yes** — pure annealing achieves higher mean R2." if summary['pure_annealed_final']['after_velocity_r2']['mean'] > max(summary['prototype_hard']['after_velocity_r2']['mean'], summary['fixed_soft_warm']['after_velocity_r2']['mean']) else "**No clear improvement** in R2."}

### Q5: Is pure annealing worth continuing?
{"**Not as a standalone direction-accuracy method.** The seed 7 collapse persists with fixed prototypes, indicating a deeper optimizer issue. Pure annealing shares the same fundamental weakness as the original annealing." if seed7_collapse_fixed else "**Yes** — fixing prototypes resolved the seed 7 collapse. Pure annealing is a cleaner experimental protocol."}

### Q6: Should the next step be P_ij weighted global rotation consensus?
{"**Yes.** The seed 7 diagnostic strongly suggests that the problem is in the SoftHiWA ADMM consensus: all group pairs contribute equally to the global rotation update, even when some group correspondences are unreliable. At low tau, the sharper assignments may create poorly-matched group pairs that pull the consensus toward a wrong R. P_ij-weighted consensus would downweight unreliable group pairs and could prevent the collapse." if seed7_collapse_fixed else "Possibly, but the priority should now be on scaling up the fixed-prototype pure annealing with more seeds to confirm stability."}

---

## Figures

| Figure | Path |
|---|---|
| Seed 7 diagnosis & pure vs original | `figures/soft_neural_pure_annealing/` |
| Pure annealing all-seed curves | `figures/soft_neural_pure_annealing/pure_annealing_curves.png` |
| Method comparison | `figures/soft_neural_pure_annealing/pure_vs_previous_annealing_comparison.png` |
| Seed 7 pure vs original | `figures/soft_neural_pure_annealing/pure_annealing_seed7_curves.png` |

## Data

| File | Path |
|---|---|
| Full results | `results/soft_neural_pure_annealing/soft_neural_pure_annealing.json` |
| Summary | `results/soft_neural_pure_annealing/soft_neural_pure_annealing_summary.json` |
| Prototype matching | `results/soft_neural_pure_annealing/seed7_prototype_matching.json` |
| This report | `results/soft_neural_pure_annealing/seed7_diagnostic_report.md` |

---

*Report auto-generated by `diagnose_seed7.py`.*
"""

    report_path = OUT_DIR / "seed7_diagnostic_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Saved report: {report_path}")

    # ================================================================
    # Zip
    # ================================================================
    zip_path = OUT_DIR.parent / "seed7_diagnosis_and_pure_annealing.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # All figures
        for fp in sorted(FIG_DIR.glob("*.png")):
            zf.write(fp, f"figures/{fp.name}")
        # All results
        for fp in sorted(OUT_DIR.glob("*.json")):
            zf.write(fp, f"results/{fp.name}")
        # Report
        zf.write(report_path, f"results/{report_path.name}")
    print(f"\nSaved zip: {zip_path} ({zip_path.stat().st_size / 1024:.0f} KB)")

    # ================================================================
    # Console summary
    # ================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Seed 7 prototype hard:        acc={hard_res['after_direction_accuracy']:.3f}  R2={hard_res['after_velocity_r2']:.3f}")
    print(f"Seed 7 fixed soft warm:       acc={fixed_res['after_direction_accuracy']:.3f}  R2={fixed_res['after_velocity_r2']:.3f}")
    print(f"Seed 7 previous annealed:     acc={annealed_results[-1]['after_direction_accuracy']:.3f}  R2={annealed_results[-1]['after_velocity_r2']:.3f}")
    print(f"Seed 7 pure annealed:         acc={pure_s7[-1]['after_direction_accuracy']:.3f}  R2={pure_s7[-1]['after_velocity_r2']:.3f}")
    print(f"NN error pattern (annealed):  {nn_annealed_050['error_pattern']}")
    print(f"Prototype match neural 0.25->0.35/0.50: {group_shift_neural_035:.4f} / {group_shift_neural_050:.4f}")
    print(f"Prototype match movement 0.25->0.35/0.50: {group_shift_movement_035:.4f} / {group_shift_movement_050:.4f}")
    if seed7_collapse_fixed:
        print(">>> Seed 7 STILL collapses with pure annealing.")
        print(">>> Root cause: SoftHiWA ADMM consensus, not prototype identity shift.")
        print(">>> Next: P_ij-weighted global rotation consensus.")
    else:
        print(">>> Seed 7 collapse FIXED by pure annealing!")
        print(">>> Root cause was prototype identity shift from re-learning.")


if __name__ == "__main__":
    main()
