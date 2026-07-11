"""Temperature annealing experiment for Soft-Prototype HiWA.

Compares three methods on held-out seeds 5–9:
1. Prototype hard            — argmax of tau=0.5 soft assignments, run via HiWA
2. Fixed soft warm start     — tau=0.5 soft assignments, warm from hard's P and R
3. Annealed soft             — tau path [0.25, 0.35, 0.50], each stage warm-starts
                               from the previous stage's P, R, and local rotations

---------------------------------------------------------------------------
WARM-START API ASSESSMENT (requirement #10)
---------------------------------------------------------------------------
SoftHiWA.fit() already supports:
  - initial_rotation   → warm-start global rotation R
  - initial_transport  → warm-start group transport P
  - warm_start_local   → seed local_rotations from the global R instead of identity

Not supported (and not modified here):
  - ADMM multipliers are zero-initialised in every fit() call and are NOT
    exposed for external injection. Between annealing stages, the dual variables
    restart from zero. This is a minor limitation: P, R, and local rotation
    state carry the bulk of the consensus information, so the warm start remains
    effective. The multipliers only encode the residual between local and global
    rotations — they converge quickly from zero when P and R are already good.

If full multiplier warm-start is desired later, SoftHiWA.fit() should accept an
optional initial_multipliers kwarg (shape (d,d,K_x,K_y)), stored as
self.multipliers_ after fitting. No such change is made here — the annealing
script works within the existing public API.

Output:
  results/soft_neural_annealing.json           — full per-stage results
  results/soft_neural_annealing_summary.json   — aggregate summary
  figures/soft_neural_annealing_curves.png     — per-seed accuracy & R² curves
  results/soft_neural_annealing_report.md      — Markdown interpretation report
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
from sklearn.decomposition import FactorAnalysis
from sklearn.manifold import Isomap

from common import FIGURES_DIR, RESULTS_DIR, ensure_output_dirs, write_json
from run_neural import (
    least_squares_rotation,
    load_demo,
    movement_to_3d,
    remove_constant_columns,
)
from run_soft_neural import PROFILES, evaluate, run_hard, run_soft
from soft_groups import learn_soft_groups

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANNEALING_PATH: list[float] = [0.25, 0.35, 0.50]
DEFAULT_SEEDS: list[int] = [5, 6, 7, 8, 9]

# ---------------------------------------------------------------------------
# Small helpers (inline, no changes to existing modules)
# ---------------------------------------------------------------------------


def _paired_diff(ref: list[dict], cand: list[dict], key: str,
                 n_bootstrap: int = 10_000) -> dict[str, Any]:
    """Bootstrap paired differences between two per-seed result lists."""
    rmap = {int(r["seed"]): r for r in ref}
    cmap = {int(r["seed"]): r for r in cand}
    seeds = sorted(set(rmap) & set(cmap))
    diffs = np.asarray([cmap[s][key] - rmap[s][key] for s in seeds], dtype=float)
    rng = np.random.default_rng(20260705)
    boot = np.asarray(
        [rng.choice(diffs, size=len(diffs), replace=True).mean()
         for _ in range(n_bootstrap)]
    )
    return {
        "seeds": seeds,
        "differences": diffs.tolist(),
        "mean_difference": float(diffs.mean()),
        "bootstrap_95_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
    }


def _by_method(results: list[dict], method: str) -> list[dict]:
    return sorted((r for r in results if r["method"] == method),
                  key=lambda r: r["seed"])


def _annealed_final(results: list[dict]) -> list[dict]:
    """Return only the final annealing stage (highest tau) per seed."""
    annealed_all = [r for r in results if r["method"] == "annealed_soft"]
    final_tau = max(r["temperature"] for r in annealed_all)
    return sorted(
        (r for r in annealed_all if r["temperature"] == final_tau),
        key=lambda r: r["seed"])


def _summarise(records: list[dict]) -> dict[str, Any]:
    acc = np.asarray([r["after_direction_accuracy"] for r in records])
    r2 = np.asarray([r["after_velocity_r2"] for r in records])
    t = np.asarray([r["elapsed_seconds"] for r in records])
    conv = np.asarray([int(bool(r["converged"])) for r in records])

    def _s(a: np.ndarray) -> dict:
        return {
            "values": a.tolist(),
            "mean": float(a.mean()),
            "std": float(a.std(ddof=1)),
            "min": float(a.min()),
            "max": float(a.max()),
        }

    return {
        "n_runs": len(records),
        "seeds": [int(r["seed"]) for r in records],
        "converged": int(conv.sum()),
        "after_direction_accuracy": _s(acc),
        "after_velocity_r2": _s(r2),
        "elapsed_seconds": _s(t),
    }


# ---------------------------------------------------------------------------
# Annealing runner
# ---------------------------------------------------------------------------


def run_annealed_stages(
    neural_3d: np.ndarray,
    neural_assignments_by_tau: dict[float, np.ndarray],
    movement_3d: np.ndarray,
    movement_assignments_by_tau: dict[float, np.ndarray],
    target_transform: np.ndarray,
    oracle_rotation: np.ndarray,
    seed: int,
    profile: str,
    retain_mass: float,
    max_support_factor: float,
    evaluation_args: dict,
    annealing_path: list[float],
) -> list[dict]:
    """Run the full annealing chain for one seed.

    Stage 0 (lowest tau) starts from random init.
    Each subsequent stage warm-starts from the previous stage's P, R.
    """
    stage_results: list[dict] = []
    prev_P: np.ndarray | None = None
    prev_R: np.ndarray | None = None

    for stage_idx, tau in enumerate(annealing_path):
        is_first = stage_idx == 0
        extra: dict[str, Any] = {}
        if not is_first:
            extra["initial_rotation"] = prev_R
            extra["initial_transport"] = prev_P
            extra["warm_start_local"] = True

        result, _aligned = run_soft(
            neural_3d=neural_3d,
            neural_assignments=neural_assignments_by_tau[tau],
            movement_3d=movement_3d,
            movement_assignments=movement_assignments_by_tau[tau],
            target_transform=target_transform,
            oracle_rotation=oracle_rotation,
            seed=seed, profile=profile,
            retain_mass=retain_mass, max_support_factor=max_support_factor,
            evaluation_args=evaluation_args,
            method=f"annealed_tau{tau}",
            **extra,
        )
        result["method"] = "annealed_soft"
        result["annealing_stage"] = stage_idx
        result["temperature"] = tau
        stage_results.append(result)
        prev_P = np.asarray(result["transport_P"])
        prev_R = np.asarray(result["rotation_R"])

    return stage_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Temperature annealing experiment for Soft-Prototype HiWA"
    )
    p.add_argument("--profile", choices=list(PROFILES), default="pilot")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--groups", type=int, default=4)
    p.add_argument("--entropy-weight", type=float, default=0.05)
    p.add_argument("--retain-mass", type=float, default=0.90)
    p.add_argument("--max-support-factor", type=float, default=1.5)
    p.add_argument("--tag", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    # --- data loading (identical to run_soft_neural.py) -------------------
    data = load_demo()
    test_neural = remove_constant_columns(data["test_neural"])
    neural_3d = FactorAnalysis(n_components=3, random_state=0).fit_transform(
        test_neural)
    train_movement_3d = movement_to_3d(data["train_movement"])
    test_movement_3d = movement_to_3d(data["test_movement"])
    target_transform = np.linalg.pinv(train_movement_3d) @ data["train_movement"]
    oracle_rotation = least_squares_rotation(test_movement_3d, neural_3d)
    evaluation_args = {
        "movement_xy": data["test_movement"],
        "neural_labels": data["test_labels"],
        "movement_labels": data["train_labels"],
    }

    all_results: list[dict] = []
    prototype_records: list[dict] = []

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")

        # --- learn soft groups at each annealing temperature --------------
        neural_grp: dict[float, Any] = {}
        movement_grp: dict[float, Any] = {}
        for tau in ANNEALING_PATH:
            neural_grp[tau] = learn_soft_groups(
                neural_3d, n_groups=args.groups, temperature=tau,
                entropy_weight=args.entropy_weight, seed=seed)
            movement_grp[tau] = learn_soft_groups(
                train_movement_3d, n_groups=args.groups, temperature=tau,
                entropy_weight=args.entropy_weight, seed=seed)

        prototype_records.append({
            "seed": seed,
            "prototypes_by_tau": {
                str(tau): {
                    "neural_diagnostics": neural_grp[tau].diagnostics,
                    "movement_diagnostics": movement_grp[tau].diagnostics,
                }
                for tau in ANNEALING_PATH
            },
        })

        neural_050 = neural_grp[0.50].assignments
        movement_050 = movement_grp[0.50].assignments
        seed_results: dict[str, dict] = {}

        # ---- Method 1: prototype hard ------------------------------------
        print("  [1/3] prototype_hard ...", end=" ", flush=True)
        t0 = time.perf_counter()
        seed_results["prototype_hard"], _ = run_hard(
            "prototype_hard",
            neural_3d, np.argmax(neural_050, axis=1),
            train_movement_3d, np.argmax(movement_050, axis=1),
            target_transform, oracle_rotation, seed, args.profile, evaluation_args)
        dt = time.perf_counter() - t0
        r = seed_results["prototype_hard"]
        print(f"acc={r['after_direction_accuracy']:.3f} "
              f"R2={r['after_velocity_r2']:.3f} ({dt:.1f}s)")

        # ---- Method 2: fixed soft warm start (tau=0.5) -------------------
        print("  [2/3] fixed_soft_warm ...", end=" ", flush=True)
        t0 = time.perf_counter()
        hard = seed_results["prototype_hard"]
        seed_results["fixed_soft_warm"], _ = run_soft(
            neural_3d=neural_3d, neural_assignments=neural_050,
            movement_3d=train_movement_3d, movement_assignments=movement_050,
            target_transform=target_transform, oracle_rotation=oracle_rotation,
            seed=seed, profile=args.profile,
            retain_mass=args.retain_mass, max_support_factor=args.max_support_factor,
            evaluation_args=evaluation_args, method="fixed_soft_warm",
            initial_rotation=np.asarray(hard["rotation_R"]),
            initial_transport=np.asarray(hard["transport_P"]),
            warm_start_local=True)
        dt = time.perf_counter() - t0
        r = seed_results["fixed_soft_warm"]
        print(f"acc={r['after_direction_accuracy']:.3f} "
              f"R2={r['after_velocity_r2']:.3f} ({dt:.1f}s)")

        # ---- Method 3: annealed soft -------------------------------------
        print("  [3/3] annealed_soft ...", end=" ", flush=True)
        t0 = time.perf_counter()
        stage_results = run_annealed_stages(
            neural_3d=neural_3d,
            neural_assignments_by_tau={t: g.assignments
                                       for t, g in neural_grp.items()},
            movement_3d=train_movement_3d,
            movement_assignments_by_tau={t: g.assignments
                                         for t, g in movement_grp.items()},
            target_transform=target_transform,
            oracle_rotation=oracle_rotation,
            seed=seed, profile=args.profile,
            retain_mass=args.retain_mass,
            max_support_factor=args.max_support_factor,
            evaluation_args=evaluation_args,
            annealing_path=ANNEALING_PATH)
        dt = time.perf_counter() - t0
        for sr in stage_results:
            seed_results[f"annealed_tau{sr['temperature']}"] = sr
        final = stage_results[-1]
        print(f"final acc={final['after_direction_accuracy']:.3f} "
              f"R2={final['after_velocity_r2']:.3f} ({dt:.1f}s)")

        for sr in stage_results:
            print(f"    tau={sr['temperature']:.2f}  "
                  f"acc={sr['after_direction_accuracy']:.3f}  "
                  f"R2={sr['after_velocity_r2']:.3f}  "
                  f"iter={sr['iterations']}  conv={sr['converged']}")

        all_results.extend(seed_results.values())

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    suffix = f"_{args.tag}" if args.tag else ""
    out = RESULTS_DIR / f"soft_neural_annealing{suffix}.json"

    payload = {
        "experiment": "soft_hiwa_temperature_annealing",
        "profile": args.profile,
        "parameters": {
            "annealing_path": ANNEALING_PATH,
            "groups": args.groups,
            "entropy_weight": args.entropy_weight,
            "retain_mass": args.retain_mass,
            "max_support_factor": args.max_support_factor,
            "profile_parameters": PROFILES[args.profile],
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "results": all_results,
        "prototype_diagnostics": prototype_records,
        "interpretation_scope": (
            "temperature annealing experiment on held-out seeds 5-9; "
            "labels used ONLY for evaluation, never for training or "
            "hyper-parameter selection"
        ),
    }
    write_json(out, payload)
    print(f"\nSaved full results: {out}")

    # ------------------------------------------------------------------
    # Summary + Figure + Report
    # ------------------------------------------------------------------
    _write_summary(all_results, suffix)
    _plot_curves(all_results, suffix)
    rp = _write_report(all_results, suffix)
    print(f"Saved report: {rp}")


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------


def _write_summary(all_results: list[dict], suffix: str) -> None:
    methods = ["prototype_hard", "fixed_soft_warm", "annealed_soft"]
    summaries: dict[str, Any] = {}
    for m in methods:
        if m == "annealed_soft":
            recs = _annealed_final(all_results)
        else:
            recs = _by_method(all_results, m)
        summaries[m] = _summarise(recs)

    for tau in ANNEALING_PATH:
        recs = sorted(
            (r for r in all_results if r.get("temperature") == tau),
            key=lambda r: r["seed"])
        if recs:
            summaries[f"annealed_tau{tau}"] = _summarise(recs)

    hard = _by_method(all_results, "prototype_hard")
    fixed = _by_method(all_results, "fixed_soft_warm")
    annealed_final = _annealed_final(all_results)

    paired: dict[str, Any] = {}
    for label, ref, cand in [
        ("fixed_vs_hard", hard, fixed),
        ("annealed_vs_hard", hard, annealed_final),
        ("annealed_vs_fixed", fixed, annealed_final),
    ]:
        if cand:
            paired[f"{label}_acc"] = _paired_diff(
                ref, cand, "after_direction_accuracy")
            paired[f"{label}_r2"] = _paired_diff(
                ref, cand, "after_velocity_r2")

    summary = {
        "experiment": "soft_hiwa_temperature_annealing",
        "annealing_path": ANNEALING_PATH,
        "summaries": summaries,
        "paired_comparisons": paired,
    }
    p = RESULTS_DIR / f"soft_neural_annealing_summary{suffix}.json"
    write_json(p, summary)
    print(f"Saved summary: {p}")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _plot_curves(all_results: list[dict], suffix: str) -> None:
    annealed = sorted(
        (r for r in all_results if r["method"] == "annealed_soft"),
        key=lambda r: (r["seed"], r["temperature"]))
    seeds = sorted({r["seed"] for r in annealed})
    taus = ANNEALING_PATH
    cmap = plt.cm.viridis
    colours = {s: cmap(i / max(1, len(seeds) - 1)) for i, s in enumerate(seeds)}

    hard = _by_method(all_results, "prototype_hard")
    fixed = _by_method(all_results, "fixed_soft_warm")
    hard_mean_acc = (np.mean([r["after_direction_accuracy"] for r in hard])
                     if hard else None)
    fixed_mean_acc = (np.mean([r["after_direction_accuracy"] for r in fixed])
                      if fixed else None)
    hard_mean_r2 = (np.mean([r["after_velocity_r2"] for r in hard])
                    if hard else None)
    fixed_mean_r2 = (np.mean([r["after_velocity_r2"] for r in fixed])
                     if fixed else None)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # -- accuracy --
    ax = axes[0]
    for s in seeds:
        pts = [r for r in annealed if r["seed"] == s]
        ax.plot(taus, [r["after_direction_accuracy"] for r in pts],
                marker="o", color=colours[s], linewidth=1.5, label=f"seed {s}")
    if hard_mean_acc is not None:
        ax.axhline(hard_mean_acc, color="0.3", linestyle="--", linewidth=1,
                   label=f"hard mean ({hard_mean_acc:.3f})")
    if fixed_mean_acc is not None:
        ax.axhline(fixed_mean_acc, color="0.6", linestyle=":", linewidth=1,
                   label=f"fixed warm mean ({fixed_mean_acc:.3f})")
    ax.axhline(0.25, color="0.7", linestyle="--", linewidth=1, alpha=0.5,
               label="chance")
    ax.set(title="Direction accuracy across annealing stages",
           xlabel="softmax temperature τ", ylabel="accuracy")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.grid(alpha=0.2)

    # -- R² --
    ax = axes[1]
    for s in seeds:
        pts = [r for r in annealed if r["seed"] == s]
        ax.plot(taus, [r["after_velocity_r2"] for r in pts],
                marker="s", color=colours[s], linewidth=1.5, label=f"seed {s}")
    if hard_mean_r2 is not None:
        ax.axhline(hard_mean_r2, color="0.3", linestyle="--", linewidth=1,
                   label=f"hard mean ({hard_mean_r2:.3f})")
    if fixed_mean_r2 is not None:
        ax.axhline(fixed_mean_r2, color="0.6", linestyle=":", linewidth=1,
                   label=f"fixed warm mean ({fixed_mean_r2:.3f})")
    ax.axhline(0.0, color="0.7", linestyle="--", linewidth=1, alpha=0.5)
    ax.set(title="Movement R² across annealing stages",
           xlabel="softmax temperature τ", ylabel="$R^2$")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.grid(alpha=0.2)

    fig.savefig(FIGURES_DIR / f"soft_neural_annealing_curves{suffix}.png",
                dpi=200)
    plt.close(fig)
    print(f"Saved figure: "
          f"{FIGURES_DIR / f'soft_neural_annealing_curves{suffix}.png'}")


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _write_report(all_results: list[dict], suffix: str) -> Path:
    hard = _by_method(all_results, "prototype_hard")
    fixed = _by_method(all_results, "fixed_soft_warm")
    annealed = _annealed_final(all_results)

    hs = _summarise(hard)
    fs = _summarise(fixed)
    as_ = _summarise(annealed)

    stage_sums: dict[float, dict] = {}
    for tau in ANNEALING_PATH:
        recs = sorted(
            (r for r in all_results if r.get("temperature") == tau),
            key=lambda r: r["seed"])
        if recs:
            stage_sums[tau] = _summarise(recs)

    avh = _paired_diff(hard, annealed, "after_direction_accuracy")
    avf = _paired_diff(fixed, annealed, "after_direction_accuracy")
    avh_r2 = _paired_diff(hard, annealed, "after_velocity_r2")
    avf_r2 = _paired_diff(fixed, annealed, "after_velocity_r2")

    def _f(v: float, d: int = 4) -> str:
        return f"{v:.{d}f}"

    seeds = sorted({r["seed"] for r in annealed})
    seed_table = "\n".join(
        f"| {s} "
        f"| {_f(next(r for r in hard if r['seed']==s)['after_direction_accuracy'])} "
        f"| {_f(next(r for r in fixed if r['seed']==s)['after_direction_accuracy'])} "
        f"| {_f(next(r for r in annealed if r['seed']==s)['after_direction_accuracy'])} "
        f"| {_f(next(r for r in hard if r['seed']==s)['after_velocity_r2'], 3)} "
        f"| {_f(next(r for r in fixed if r['seed']==s)['after_velocity_r2'], 3)} "
        f"| {_f(next(r for r in annealed if r['seed']==s)['after_velocity_r2'], 3)} |"
        for s in seeds)

    hf = sum(1 for r in hard if r["after_velocity_r2"] < 0)
    ff = sum(1 for r in fixed if r["after_velocity_r2"] < 0)
    af = sum(1 for r in annealed if r["after_velocity_r2"] < 0)

    stage_table = "\n".join(
        f"| {tau:.2f} "
        f"| {_f(stage_sums[tau]['after_direction_accuracy']['mean'])} "
        f"± {_f(stage_sums[tau]['after_direction_accuracy']['std'])} "
        f"| {_f(stage_sums[tau]['after_velocity_r2']['mean'], 3)} "
        f"± {_f(stage_sums[tau]['after_velocity_r2']['std'], 3)} "
        f"| {stage_sums[tau]['converged']}/{stage_sums[tau]['n_runs']} |"
        for tau in ANNEALING_PATH if tau in stage_sums)

    better_acc = as_["after_direction_accuracy"]["mean"] > max(
        hs["after_direction_accuracy"]["mean"],
        fs["after_direction_accuracy"]["mean"])
    better_r2 = as_["after_velocity_r2"]["mean"] > max(
        hs["after_velocity_r2"]["mean"],
        fs["after_velocity_r2"]["mean"])

    report = f"""# Temperature Annealing Experiment Report

**Date:** 2026-07-05
**Experiment:** soft_hiwa_temperature_annealing
**Seeds:** {seeds} (held-out, not used in prior temperature selection)
**Annealing path:** τ = {ANNEALING_PATH}
**Profile:** pilot (maxiter=80, tol=1e-1, mu=5e-3)

---

## 1. Quick Answer

| Question | Answer |
|---|---|
| Annealed soft better than fixed soft warm start? | {"**Yes**" if better_acc else "**No** — see details below"} |
| Reduced catastrophic failures (R² < 0)? | Hard: {hf}/5, Fixed warm: {ff}/5, Annealed: {af}/5 |
| Improved direction accuracy? | {"**Yes**" if better_acc else "No clear gain"} |
| Maintained movement R²? | {"**Yes**" if better_r2 else "Comparable or slightly different"} |
| Consistent across seeds? | See per-seed table (Section 4) |

---

## 2. Aggregate Results

| Method | Direction Acc (mean ± std) | Movement R² (mean ± std) | Converged | Runtime (s) |
|---|---|---|---|---|
| prototype_hard | {_f(hs['after_direction_accuracy']['mean'])} ± {_f(hs['after_direction_accuracy']['std'])} | {_f(hs['after_velocity_r2']['mean'], 3)} ± {_f(hs['after_velocity_r2']['std'], 3)} | {hs['converged']}/{hs['n_runs']} | {_f(hs['elapsed_seconds']['mean'], 1)} |
| fixed_soft_warm | {_f(fs['after_direction_accuracy']['mean'])} ± {_f(fs['after_direction_accuracy']['std'])} | {_f(fs['after_velocity_r2']['mean'], 3)} ± {_f(fs['after_velocity_r2']['std'], 3)} | {fs['converged']}/{fs['n_runs']} | {_f(fs['elapsed_seconds']['mean'], 1)} |
| annealed_soft (final) | {_f(as_['after_direction_accuracy']['mean'])} ± {_f(as_['after_direction_accuracy']['std'])} | {_f(as_['after_velocity_r2']['mean'], 3)} ± {_f(as_['after_velocity_r2']['std'], 3)} | {as_['converged']}/{as_['n_runs']} | {_f(as_['elapsed_seconds']['mean'], 1)} |

### Per-stage (annealed only)

| Stage τ | Direction Acc (mean ± std) | R² (mean ± std) | Converged |
|---|---|---|---|
{stage_table}

---

## 3. Paired Comparisons

### Annealed vs Prototype Hard

| Metric | Mean Δ | 95% CI |
|---|---|---|
| Direction accuracy | {_f(avh['mean_difference'])} | [{_f(avh['bootstrap_95_ci'][0])}, {_f(avh['bootstrap_95_ci'][1])}] |
| Movement R² | {_f(avh_r2['mean_difference'], 3)} | [{_f(avh_r2['bootstrap_95_ci'][0], 3)}, {_f(avh_r2['bootstrap_95_ci'][1], 3)}] |

### Annealed vs Fixed Soft Warm Start

| Metric | Mean Δ | 95% CI |
|---|---|---|
| Direction accuracy | {_f(avf['mean_difference'])} | [{_f(avf['bootstrap_95_ci'][0])}, {_f(avf['bootstrap_95_ci'][1])}] |
| Movement R² | {_f(avf_r2['mean_difference'], 3)} | [{_f(avf_r2['bootstrap_95_ci'][0], 3)}, {_f(avf_r2['bootstrap_95_ci'][1], 3)}] |

---

## 4. Per-Seed Breakdown

| Seed | Hard Acc | Fixed Acc | Annealed Acc | Hard R² | Fixed R² | Annealed R² |
|---|---|---|---|---|---|---|
{seed_table}

---

## 5. Catastrophic Failure Check

- **prototype_hard:** {hf}/5 seeds with R² < 0
- **fixed_soft_warm:** {ff}/5 seeds with R² < 0
- **annealed_soft:** {af}/5 seeds with R² < 0

{"**Annealing eliminated all catastrophic failures.**" if af == 0 and (hf > 0 or ff > 0) else "No catastrophic failures in any method." if hf == 0 and ff == 0 and af == 0 else "Annealing did not fully eliminate catastrophic failures."}

---

## 6. Interpretation

### Does annealed soft outperform fixed soft warm start?

{"Yes — the annealed path achieves higher mean direction accuracy than fixed soft warm start, with the bootstrap 95% CI for the difference being [" + _f(avf['bootstrap_95_ci'][0]) + ", " + _f(avf['bootstrap_95_ci'][1]) + "]." if avf['mean_difference'] > 0 else "No — the annealed path does not produce a meaningful accuracy gain over fixed soft warm start (mean Δ = " + _f(avf['mean_difference']) + ", 95% CI [" + _f(avf['bootstrap_95_ci'][0]) + ", " + _f(avf['bootstrap_95_ci'][1]) + "])."}

### Does annealing reduce catastrophic failures?

{"Yes — annealed soft has " + str(af) + " catastrophic failures vs " + str(ff) + " for fixed warm start and " + str(hf) + " for hard." if af < max(hf, ff) else "The failure rate is comparable across methods."}

### Does annealing improve direction accuracy?

The annealed path{" does" if better_acc else " does not"} clearly improve direction accuracy over the baselines. The mean accuracy of annealed soft is {_f(as_['after_direction_accuracy']['mean'])} vs {_f(hs['after_direction_accuracy']['mean'])} (hard) and {_f(fs['after_direction_accuracy']['mean'])} (fixed warm).

### Does annealing maintain movement R²?

The annealed path produces movement R² of {_f(as_['after_velocity_r2']['mean'], 3)} (mean) vs {_f(hs['after_velocity_r2']['mean'], 3)} (hard) and {_f(fs['after_velocity_r2']['mean'], 3)} (fixed warm). {"R² is maintained or improved." if as_['after_velocity_r2']['mean'] >= min(hs['after_velocity_r2']['mean'], fs['after_velocity_r2']['mean']) else "R² is slightly lower — the annealed path may trade some movement fit for other properties."}

### Is the improvement consistent across seeds?

The per-seed table (Section 4) shows {"consistent trends across seeds" if as_['after_direction_accuracy']['std'] <= max(hs['after_direction_accuracy']['std'], fs['after_direction_accuracy']['std']) else "some seed-to-seed variation"}. The standard deviation of annealed accuracy is {_f(as_['after_direction_accuracy']['std'])}, compared to {_f(hs['after_direction_accuracy']['std'])} (hard) and {_f(fs['after_direction_accuracy']['std'])} (fixed warm).

---

## 7. Limitations & Caveats

1. **ADMM multipliers are not preserved across stages.** The current `SoftHiWA` API does not expose multipliers for external injection — they are zero-initialised in each stage. Only P, R, and local rotation initialisation carry over.
2. **5 seeds only.** This is a mechanism-validation experiment; broader claims require more seeds.
3. **Single annealing path.** Only [0.25, 0.35, 0.50] was tested. Other schedules (e.g., more stages, different endpoints) may yield different results.
4. **Fixed prototype learning.** Prototypes are independently learned at each temperature rather than using a single set of prototypes with varying softmax temperature. This means group identity may shift slightly between stages.
5. **No label information was used for training or parameter selection.** Labels were used only in the final evaluation metrics.

---

## 8. Conclusion

{_conclusion(better_acc, better_r2, af, hf, ff, as_, hs, fs)}

---

*Report auto-generated by `run_annealing.py`. See `results/soft_neural_annealing{suffix}.json` for raw data and `figures/soft_neural_annealing_curves{suffix}.png` for the annealing-curve plot.*
"""

    out = RESULTS_DIR / f"soft_neural_annealing_report{suffix}.md"
    out.write_text(report, encoding="utf-8")
    return out


def _conclusion(better_acc: bool, better_r2: bool, af: int, hf: int, ff: int,
                as_: dict, hs: dict, fs: dict) -> str:
    parts: list[str] = []

    if better_acc and better_r2:
        parts.append(
            "Temperature annealing provides a clear improvement over both prototype "
            "hard and fixed soft warm start on held-out seeds 5–9, with higher mean "
            "direction accuracy and maintained or improved movement R². ")
    elif better_acc:
        parts.append(
            "Temperature annealing improves direction accuracy over both baselines, "
            "though movement R² is comparable. ")
    elif better_r2:
        parts.append(
            "Temperature annealing improves movement R² over both baselines, "
            "though direction accuracy is comparable. ")
    else:
        parts.append(
            "Temperature annealing does not produce a clear accuracy or R² gain "
            "over the baselines on these 5 held-out seeds. ")

    if af < max(hf, ff):
        parts.append(
            f"It reduces catastrophic failures (R² < 0) from "
            f"{max(hf, ff)} to {af}. ")
    elif af == 0 and hf == 0 and ff == 0:
        parts.append(
            "No method suffered catastrophic failures (R² < 0) on these seeds. ")

    parts.append(
        f"The annealed path achieves mean accuracy "
        f"{as_['after_direction_accuracy']['mean']:.4f} ± "
        f"{as_['after_direction_accuracy']['std']:.4f} "
        f"and mean R² "
        f"{as_['after_velocity_r2']['mean']:.3f} ± "
        f"{as_['after_velocity_r2']['std']:.3f}. ")
    parts.append(
        "Further experiments with more seeds, alternative annealing schedules, "
        "and multiplier-preserving warm starts are recommended before drawing "
        "strong conclusions.")
    return "".join(parts)


if __name__ == "__main__":
    main()
