"""Test whether fixed four-group GC-HiWA has a measurable group-count deficit.

The true group count is withheld from all fixed-K fits.  Running with the
true count is an oracle control, not an automatic-selector result.  A DP
extension is justified only if this control exposes a clear and stable gap.
"""

from __future__ import annotations

import argparse
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.decomposition import PCA


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT / "src", ROOT / "src" / "cc_hiwa", ROOT / "experiments" / "hiwa"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import RESULTS_DIR, ensure_output_dirs, write_json  # noqa: E402
from alignment_qualification import (  # noqa: E402
    normalized_covariance_identifiability_margin,
    normalized_covariance_spectral_mismatch,
    qualify_alignment_fit,
)
from soft_groups import learn_soft_groups  # noqa: E402
from soft_hiwa import SoftHiWA  # noqa: E402
from synthetic_boundary import VariableGroupScenario, make_variable_group_pair  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2101, 2102, 2103])
    parser.add_argument("--true-groups", nargs="+", type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument("--fixed-groups", type=int, default=4)
    parser.add_argument("--samples-per-group", type=int, default=10)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.50)
    parser.add_argument("--entropy-weight", type=float, default=0.05)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=0.02)
    parser.add_argument("--mu", type=float, default=0.05)
    parser.add_argument("--local-gamma", type=float, default=0.40)
    parser.add_argument("--local-iterations", type=int, default=24)
    parser.add_argument("--local-sinkhorn-iterations", type=int, default=60)
    parser.add_argument("--max-covariance-spectral-mismatch", type=float, default=0.05)
    parser.add_argument("--min-covariance-identifiability-margin", type=float, default=0.02)
    parser.add_argument("--tag", default="gc_hiwa_group_count_value_gate_2101_2103")
    args = parser.parse_args()
    if not args.true_groups or min(args.true_groups) < 2:
        parser.error("true-groups must contain integers no smaller than two")
    if args.fixed_groups < 2 or args.samples_per_group < 4:
        parser.error("fixed-groups must be at least two and samples-per-group at least four")
    if min(
        args.temperature,
        args.local_gamma,
        args.max_covariance_spectral_mismatch,
        args.min_covariance_identifiability_margin,
    ) <= 0:
        parser.error("temperature, local-gamma and qualification thresholds must be positive")
    return args


def _fit(pair, learned_groups: int, seed: int, args: argparse.Namespace) -> dict[str, object]:
    source_groups = learn_soft_groups(
        pair.source,
        learned_groups,
        args.temperature,
        args.entropy_weight,
        seed=seed,
        scaling_mode="global_scalar",
    )
    target_groups = learn_soft_groups(
        pair.target,
        learned_groups,
        args.temperature,
        args.entropy_weight,
        seed=seed,
        scaling_mode="global_scalar",
    )
    model = SoftHiWA(
        dim_red_method=PCA(n_components=3),
        normalize=False,
        maxiter=args.maxiter,
        tol=args.tol,
        mu=args.mu,
        shorn_maxiter=200,
        shorn_gamma=0.20,
        sa_maxiter=args.local_iterations,
        sa_tol=1e-2,
        sa_shorn_maxiter=args.local_sinkhorn_iterations,
        sa_shorn_gamma=args.local_gamma,
        support_mode="full",
        random_state=seed,
        warm_start_local=True,
        consensus_weighting="transport",
        inner_entropy_mode="fixed",
        determinant_sign=1,
    )
    aligned = model.fit_transform(
        pair.source,
        source_groups.assignments,
        pair.target,
        target_groups.assignments,
        X_transform=np.eye(3),
        Y_transform=np.eye(3),
    )
    diagnostics = model.diagnostics
    spectrum = normalized_covariance_spectral_mismatch(pair.source, pair.target)
    identifiability = normalized_covariance_identifiability_margin(pair.source)
    qualification = qualify_alignment_fit(
        {
            **diagnostics,
            "covariance_spectral_mismatch": spectrum,
            "covariance_identifiability_margin": identifiability,
        },
        max_covariance_spectral_mismatch=args.max_covariance_spectral_mismatch,
        min_covariance_identifiability_margin=args.min_covariance_identifiability_margin,
    )
    return {
        "learned_groups": learned_groups,
        "admm_converged": bool(diagnostics["admm_converged"]),
        "rotation_frobenius_error": float(np.linalg.norm(model.Rg - pair.rotation_truth, "fro")),
        "paired_alignment_mse": float(np.mean((aligned - pair.target_paired_truth) ** 2)),
        "relative_global_fit_ratio": float(diagnostics["relative_global_fit_ratio"]),
        "covariance_spectral_mismatch": float(spectrum),
        "covariance_identifiability_margin": float(identifiability),
        "applicability_qualification": qualification,
        "source_soft_group_diagnostics": source_groups.diagnostics,
        "target_soft_group_diagnostics": target_groups.diagnostics,
    }


def _summarize(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[(int(record["true_groups"]), int(record["learned_groups"]))].append(record)
    summary = []
    for (true_groups, learned_groups), rows in sorted(grouped.items()):
        summary.append(
            {
                "true_groups": true_groups,
                "learned_groups": learned_groups,
                "runs": len(rows),
                "accepted_count": int(
                    sum(bool(row["applicability_qualification"]["accepted"]) for row in rows)
                ),
                "rotation_frobenius_error_mean": float(
                    np.mean([float(row["rotation_frobenius_error"]) for row in rows])
                ),
                "paired_alignment_mse_mean": float(
                    np.mean([float(row["paired_alignment_mse"]) for row in rows])
                ),
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    records: list[dict[str, object]] = []
    for true_groups in args.true_groups:
        for seed in args.seeds:
            scenario = VariableGroupScenario(
                f"valid_{true_groups}_group_rotation",
                true_groups=true_groups,
                samples_per_group=args.samples_per_group,
                noise_std=args.noise_std,
            )
            pair = make_variable_group_pair(scenario, seed)
            for learned_groups in dict.fromkeys((args.fixed_groups, true_groups)):
                record = _fit(pair, learned_groups, seed, args)
                records.append(
                    {
                        "scenario": scenario.name,
                        "seed": seed,
                        "true_groups": true_groups,
                        "oracle_true_group_count_control": learned_groups == true_groups,
                        **record,
                    }
                )
            print(f"completed true_groups={true_groups} seed={seed}", flush=True)
    ensure_output_dirs()
    path = RESULTS_DIR / f"{args.tag}.json"
    payload = {
        "experiment": "fixed_four_group_value_gate_for_adaptive_group_count",
        "scope": (
            "fixed K=4 is compared only to an oracle true-K control on valid orthogonal "
            "synthetic pairs; neither fit receives group labels or pairings"
        ),
        "truth_usage": (
            "true group count, source labels, paired target rows and true rotation are "
            "evaluation-only; true K is supplied only to the explicitly labelled oracle control"
        ),
        "parameters": vars(args),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "records": records,
        "summary": _summarize(records),
    }
    write_json(path, payload)
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
