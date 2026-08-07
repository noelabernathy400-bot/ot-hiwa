"""Frozen known-truth boundary benchmark for the fixed GC-HiWA v2 core.

The benchmark never supplies truth labels, paired target rows, the true
rotation, or the true determinant component to the soft-group learner, OT, or
ROCA selector.  They are retained only for post-fit diagnostics.

``*_oracle_component`` controls expose the correct determinant component only
to isolate membership and consensus recovery.  They are not deployable
unsupervised methods.  ``soft_roca`` is the label-free deployable branch test.
"""

from __future__ import annotations

import argparse
import platform
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT / "src", ROOT / "src" / "cc_hiwa", ROOT / "experiments" / "hiwa"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import FIGURES_DIR, RESULTS_DIR, ensure_output_dirs, write_json  # noqa: E402
from alignment_qualification import (  # noqa: E402
    normalized_covariance_identifiability_margin,
    normalized_covariance_spectral_mismatch,
    qualify_alignment_fit,
)
from roca_qualification import RocaCandidateEvidence, qualify_roca_candidates  # noqa: E402
from soft_groups import learn_soft_groups  # noqa: E402
from soft_hiwa import SoftHiWA  # noqa: E402
from synthetic_boundary import BoundaryScenario, default_boundary_scenarios, make_synthetic_pair  # noqa: E402


SCENARIOS = {scenario.name: scenario for scenario in default_boundary_scenarios()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[901])
    parser.add_argument("--scenarios", nargs="+", choices=tuple(SCENARIOS), default=tuple(SCENARIOS))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("hard_group_oracle_component", "soft_uniform_oracle_component", "soft_transport_oracle_component", "soft_roca"),
        default=("hard_group_oracle_component", "soft_uniform_oracle_component", "soft_transport_oracle_component", "soft_roca"),
    )
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.50)
    parser.add_argument("--entropy-weight", type=float, default=0.05)
    parser.add_argument(
        "--group-scaling-mode",
        choices=("global_scalar", "per_feature"),
        default="global_scalar",
        help=(
            "soft-group preprocessing; global_scalar is required when the "
            "synthetic contract claims an unknown orthogonal coordinate map"
        ),
    )
    parser.add_argument("--samples-per-group", type=int, default=8)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=0.02)
    parser.add_argument("--mu", type=float, default=0.05)
    parser.add_argument(
        "--local-gamma",
        type=float,
        default=0.40,
        help=(
            "fixed local Sinkhorn entropy strength.  The default equals the "
            "historical effective strength for four balanced groups (0.10 / 0.25), "
            "but is never adapted to the learned transport mass."
        ),
    )
    parser.add_argument("--group-sinkhorn-iterations", type=int, default=200)
    parser.add_argument("--local-iterations", type=int, default=24)
    parser.add_argument("--local-sinkhorn-iterations", type=int, default=60)
    parser.add_argument(
        "--max-covariance-spectral-mismatch",
        type=float,
        default=0.05,
        help=(
            "maximum scale-normalized covariance-spectrum mismatch. A single "
            "orthogonal coordinate map must preserve this spectrum."
        ),
    )
    parser.add_argument(
        "--min-covariance-identifiability-margin",
        type=float,
        default=0.02,
        help="minimum covariance eigenvalue-separation margin required to output a rotation",
    )
    parser.add_argument(
        "--roca-min-relative-objective-gap",
        type=float,
        default=0.05,
        help="minimum relative internal-objective gap required to break a two-healthy-candidate ROCA tie",
    )
    parser.add_argument("--tag", default="gc_hiwa_boundary_screen")
    args = parser.parse_args()
    if args.groups != 4:
        parser.error("the current known-truth tetrahedral benchmark fixes four groups")
    if (
        args.temperature <= 0
        or args.local_gamma <= 0
        or args.roca_min_relative_objective_gap < 0
        or args.max_covariance_spectral_mismatch <= 0
        or args.min_covariance_identifiability_margin <= 0
        or args.maxiter < 6
        or args.local_iterations < 1
    ):
        parser.error("temperatures must be positive, ROCA objective gap non-negative, and iteration budgets valid")
    return args


def _one_hot(assignments: np.ndarray) -> np.ndarray:
    labels = np.argmax(assignments, axis=1)
    return np.eye(assignments.shape[1], dtype=float)[labels]


def _fit(
    *,
    source: np.ndarray,
    source_assignments: np.ndarray,
    target: np.ndarray,
    target_assignments: np.ndarray,
    seed: int,
    determinant_sign: int,
    consensus_weighting: str,
    args: argparse.Namespace,
) -> tuple[SoftHiWA, np.ndarray]:
    model = SoftHiWA(
        dim_red_method=PCA(n_components=3),
        normalize=False,
        maxiter=args.maxiter,
        tol=args.tol,
        mu=args.mu,
        shorn_maxiter=args.group_sinkhorn_iterations,
        shorn_gamma=0.20,
        sa_maxiter=args.local_iterations,
        sa_tol=1e-2,
        sa_shorn_maxiter=args.local_sinkhorn_iterations,
        sa_shorn_gamma=args.local_gamma,
        support_mode="full",
        random_state=seed,
        warm_start_local=True,
        consensus_weighting=consensus_weighting,
        inner_entropy_mode="fixed",
        determinant_sign=determinant_sign,
    )
    aligned = model.fit_transform(
        source,
        source_assignments,
        target,
        target_assignments,
        X_transform=np.eye(source.shape[1]),
        Y_transform=np.eye(target.shape[1]),
    )
    return model, aligned


def _metrics(
    *,
    method: str,
    model: SoftHiWA,
    aligned: np.ndarray,
    pair,
    source_assignments: np.ndarray,
    target_assignments: np.ndarray,
    requested_sign: int,
    covariance_spectral_mismatch: float,
    max_covariance_spectral_mismatch: float,
    covariance_identifiability_margin: float,
    min_covariance_identifiability_margin: float,
) -> dict[str, object]:
    diagnostics = model.diagnostics
    rotation_error = (
        float(np.linalg.norm(model.Rg - pair.rotation_truth, "fro"))
        if pair.has_single_rotation_truth
        else None
    )
    qualification_diagnostics = {
        **diagnostics,
        "covariance_spectral_mismatch": covariance_spectral_mismatch,
        "covariance_identifiability_margin": covariance_identifiability_margin,
    }
    return {
        "method": method,
        "requested_determinant_sign": int(requested_sign),
        "estimated_determinant_sign": int(np.sign(np.linalg.det(model.Rg))),
        "rotation_frobenius_error": rotation_error,
        "single_rotation_truth_exists": pair.has_single_rotation_truth,
        "paired_alignment_mse": float(np.mean((aligned - pair.target_paired_truth) ** 2)),
        "source_group_ari": float(adjusted_rand_score(pair.source_labels, np.argmax(source_assignments, axis=1))),
        "target_group_ari": float(adjusted_rand_score(pair.target_labels, np.argmax(target_assignments, axis=1))),
        "admm_converged": bool(diagnostics["admm_converged"]),
        "iterations": int(len(diagnostics["Rg_norm"])),
        "final_global_residual": float(diagnostics["Rg_norm"][-1]),
        "final_primal_residual": float(diagnostics["admm_primal_residual"][-1]),
        "final_dual_residual": float(diagnostics["admm_dual_residual"][-1]),
        "final_local_marginal_error": float(diagnostics["max_sinkhorn_marginal_error"][-1]),
        "group_row_marginal_error": float(diagnostics["group_transport_row_marginal_error"]),
        "group_column_marginal_error": float(diagnostics["group_transport_column_marginal_error"]),
        "rotation_orthogonality_error": float(diagnostics["rotation_orthogonality_error"]),
        "transport_objective": float(diagnostics["transport_objective"]),
        "relative_global_fit_ratio": float(diagnostics["relative_global_fit_ratio"]),
        "covariance_spectral_mismatch": float(covariance_spectral_mismatch),
        "covariance_identifiability_margin": float(covariance_identifiability_margin),
        "consensus_weighting": str(diagnostics["consensus_weighting"]),
        "inner_entropy_mode": str(diagnostics["inner_entropy_mode"]),
        "applicability_qualification": qualify_alignment_fit(
            qualification_diagnostics,
            max_covariance_spectral_mismatch=max_covariance_spectral_mismatch,
            min_covariance_identifiability_margin=min_covariance_identifiability_margin,
        ),
    }


def _run_instance(scenario: BoundaryScenario, seed: int, args: argparse.Namespace) -> list[dict[str, object]]:
    pair = make_synthetic_pair(scenario, seed)
    covariance_spectral_mismatch = normalized_covariance_spectral_mismatch(pair.source, pair.target)
    covariance_identifiability_margin = normalized_covariance_identifiability_margin(pair.source)
    source_groups = learn_soft_groups(
        pair.source,
        args.groups,
        args.temperature,
        args.entropy_weight,
        seed=seed,
        scaling_mode=args.group_scaling_mode,
    )
    target_groups = learn_soft_groups(
        pair.target,
        args.groups,
        args.temperature,
        args.entropy_weight,
        seed=seed,
        scaling_mode=args.group_scaling_mode,
    )
    soft_source, soft_target = source_groups.assignments, target_groups.assignments
    hard_source, hard_target = _one_hot(soft_source), _one_hot(soft_target)
    rows: list[dict[str, object]] = []
    common = {
        "scenario": scenario.name,
        "seed": int(seed),
        "truth": {
            "determinant_sign": int(scenario.determinant_sign),
            "noise_std": scenario.noise_std,
            "mixed_transform_fraction": scenario.mixed_transform_fraction,
            "realized_mixed_transform_fraction": float(pair.mixed_transform_mask.mean()),
            "group_shift": scenario.group_shift,
            "nonorthogonal_scale": scenario.nonorthogonal_scale,
            "cross_modal_target": scenario.cross_modal_target,
            "single_rotation_truth_exists": pair.has_single_rotation_truth,
        },
        "group_learning": {
            "source": source_groups.diagnostics,
            "target": target_groups.diagnostics,
        },
    }

    def append(method: str, model: SoftHiWA, aligned: np.ndarray, a: np.ndarray, b: np.ndarray, sign: int, extra: dict[str, object] | None = None) -> None:
        row = {
            **common,
            **_metrics(
                method=method,
                model=model,
                aligned=aligned,
                pair=pair,
                source_assignments=a,
                target_assignments=b,
                requested_sign=sign,
                covariance_spectral_mismatch=covariance_spectral_mismatch,
                max_covariance_spectral_mismatch=args.max_covariance_spectral_mismatch,
                covariance_identifiability_margin=covariance_identifiability_margin,
                min_covariance_identifiability_margin=args.min_covariance_identifiability_margin,
            ),
        }
        if extra:
            row.update(extra)
        rows.append(row)

    if "hard_group_oracle_component" in args.methods:
        model, aligned = _fit(source=pair.source, source_assignments=hard_source, target=pair.target, target_assignments=hard_target, seed=seed, determinant_sign=scenario.determinant_sign, consensus_weighting="uniform", args=args)
        append("hard_group_oracle_component", model, aligned, hard_source, hard_target, scenario.determinant_sign)
    if "soft_uniform_oracle_component" in args.methods:
        model, aligned = _fit(source=pair.source, source_assignments=soft_source, target=pair.target, target_assignments=soft_target, seed=seed, determinant_sign=scenario.determinant_sign, consensus_weighting="uniform", args=args)
        append("soft_uniform_oracle_component", model, aligned, soft_source, soft_target, scenario.determinant_sign)
    if "soft_transport_oracle_component" in args.methods:
        model, aligned = _fit(source=pair.source, source_assignments=soft_source, target=pair.target, target_assignments=soft_target, seed=seed, determinant_sign=scenario.determinant_sign, consensus_weighting="transport", args=args)
        append("soft_transport_oracle_component", model, aligned, soft_source, soft_target, scenario.determinant_sign)
    if "soft_roca" in args.methods:
        candidates: list[tuple[int, SoftHiWA, np.ndarray]] = []
        for sign in (-1, 1):
            model, aligned = _fit(source=pair.source, source_assignments=soft_source, target=pair.target, target_assignments=soft_target, seed=seed, determinant_sign=sign, consensus_weighting="transport", args=args)
            candidates.append((sign, model, aligned))
        selector = qualify_roca_candidates(
            pair.source,
            soft_source,
            pair.target,
            soft_target,
            [
                RocaCandidateEvidence(
                    candidate_sign,
                    candidate_model.P,
                    bool(candidate_model.diagnostics["admm_converged"]),
                    float(candidate_model.diagnostics["max_sinkhorn_marginal_error"][-1]),
                    float(candidate_model.diagnostics["transport_objective"]),
                    float(candidate_model.diagnostics["relative_global_fit_ratio"]),
                )
                for candidate_sign, candidate_model, _ in candidates
            ],
            min_relative_objective_gap=args.roca_min_relative_objective_gap,
            covariance_spectrum_compatible=(
                covariance_spectral_mismatch <= args.max_covariance_spectral_mismatch
            ),
            covariance_spectral_mismatch=covariance_spectral_mismatch,
            rotation_geometry_identifiable=(
                covariance_identifiability_margin >= args.min_covariance_identifiability_margin
            ),
            covariance_identifiability_margin=covariance_identifiability_margin,
        )
        selected_sign = selector["selected_sign"]
        candidate_rows = [
            {
                "determinant_sign": candidate_sign,
                "admm_converged": bool(candidate_model.diagnostics["admm_converged"]),
                "rotation_frobenius_error": (
                    float(np.linalg.norm(candidate_model.Rg - pair.rotation_truth, "fro"))
                    if pair.has_single_rotation_truth
                    else None
                ),
                "transport_objective": float(candidate_model.diagnostics["transport_objective"]),
            }
            for candidate_sign, candidate_model, _ in candidates
        ]
        if selected_sign is None:
            rows.append(
                {
                    **common,
                    "method": "soft_roca_abstained",
                    "roca_selected_correct_component": False,
                    "roca_qualification": selector,
                    "roca_candidates": candidate_rows,
                }
            )
        else:
            sign, model, aligned = next(candidate for candidate in candidates if candidate[0] == selected_sign)
            append(
                "soft_roca",
                model,
                aligned,
                soft_source,
                soft_target,
                sign,
                {
                    "roca_selected_correct_component": bool(selected_sign == scenario.determinant_sign),
                    "roca_qualification": selector,
                    "roca_candidates": candidate_rows,
                },
            )
    return rows


def _summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if "rotation_frobenius_error" in row:
            groups[(str(row["scenario"]), str(row["method"]))].append(row)
    result: list[dict[str, object]] = []
    for (scenario, method), values in sorted(groups.items()):
        numeric = lambda key: np.asarray(
            [float(value[key]) for value in values if value.get(key) is not None],
            dtype=float,
        )
        rotation_errors = numeric("rotation_frobenius_error")
        result.append(
            {
                "scenario": scenario,
                "method": method,
                "n": len(values),
                "rotation_frobenius_error_mean": (
                    float(rotation_errors.mean()) if rotation_errors.size else None
                ),
                "paired_alignment_mse_mean": float(numeric("paired_alignment_mse").mean()),
                "source_group_ari_mean": float(numeric("source_group_ari").mean()),
                "target_group_ari_mean": float(numeric("target_group_ari").mean()),
                "converged_count": int(sum(bool(value["admm_converged"]) for value in values)),
                "roca_component_correct_count": int(sum(bool(value.get("roca_selected_correct_component", False)) for value in values)),
            }
        )
    return result


def _plot(summary: list[dict[str, object]], path: Path) -> None:
    scenarios = list(dict.fromkeys(item["scenario"] for item in summary))
    methods = list(dict.fromkeys(item["method"] for item in summary))
    colors = {"hard_group_oracle_component": "#666666", "soft_uniform_oracle_component": "#377eb8", "soft_transport_oracle_component": "#4daf4a", "soft_roca": "#984ea3"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    width = 0.8 / max(len(methods), 1)
    x = np.arange(len(scenarios))
    for index, method in enumerate(methods):
        values = {item["scenario"]: item for item in summary if item["method"] == method}
        rotation_values = [
            values.get(scenario, {}).get("rotation_frobenius_error_mean", np.nan)
            for scenario in scenarios
        ]
        rotation_values = [np.nan if value is None else value for value in rotation_values]
        axes[0].bar(x + (index - (len(methods) - 1) / 2) * width, rotation_values, width=width, label=method, color=colors[method])
        axes[1].bar(x + (index - (len(methods) - 1) / 2) * width, [values.get(scenario, {}).get("paired_alignment_mse_mean", np.nan) for scenario in scenarios], width=width, label=method, color=colors[method])
    for axis, title in zip(axes, ("Rotation recovery error", "Withheld paired alignment MSE")):
        axis.set(title=title, xlabel="frozen synthetic condition")
        axis.set_xticks(x, scenarios, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    scenarios = [
        BoundaryScenario(**{**SCENARIOS[name].__dict__, "samples_per_group": args.samples_per_group})
        for name in args.scenarios
    ]
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        for seed in args.seeds:
            rows.extend(_run_instance(scenario, seed, args))
            print(f"completed scenario={scenario.name} seed={seed}", flush=True)
    summary = _summary(rows)
    suffix = f"_{args.tag}" if args.tag else ""
    json_path = RESULTS_DIR / f"gc_hiwa_boundary_benchmark{suffix}.json"
    figure_path = FIGURES_DIR / f"gc_hiwa_boundary_benchmark{suffix}.png"
    payload = {
        "experiment": "gc_hiwa_fixed_core_known_truth_boundary_benchmark",
        "scope": "fixed adapters, rotation-equivariant soft groups, full-support hierarchical OT, ADMM consensus, and ROCA; learned encoders and joint prototype updates are excluded",
        "truth_usage": "truth rotation, paired rows, and group labels are withheld from fitting and used only after fitting for scoring",
        "oracle_component_control": "oracle-component controls receive only the true determinant component to isolate group and consensus recovery; soft_roca is the label-free branch-selection method",
        "parameters": vars(args),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "records": rows,
        "summary": summary,
    }
    write_json(json_path, payload)
    _plot(summary, figure_path)
    print(f"saved: {json_path}\nsaved: {figure_path}")


if __name__ == "__main__":
    main()
