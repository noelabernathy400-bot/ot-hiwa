from __future__ import annotations

import argparse
import platform
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.manifold import Isomap

from common import (
    FIGURES_DIR,
    PYHIWA_ROOT,
    RESULTS_DIR,
    HiWA,
    ensure_output_dirs,
    nearest_neighbor_accuracy,
    whiten,
    write_json,
)


PROFILES = {
    "quick": dict(
        maxiter=40,
        tol=1e-2,
        mu=2e-2,
        shorn_maxiter=200,
        sa_maxiter=30,
        sa_shorn_maxiter=50,
    ),
    "python-default": dict(
        maxiter=300,
        tol=1e-1,
        mu=5e-3,
        shorn_maxiter=1000,
        sa_maxiter=100,
        sa_shorn_maxiter=150,
    ),
    "matlab-demo": dict(
        maxiter=200,
        tol=1e-3,
        mu=2e-2,
        shorn_maxiter=1000,
        sa_maxiter=100,
        sa_shorn_maxiter=150,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("notebook", "consistent"),
        default=["notebook", "consistent"],
    )
    return parser.parse_args()


def run_variant(
    source: np.ndarray,
    source_labels: np.ndarray,
    target: np.ndarray,
    target_labels: np.ndarray,
    seed: int,
    variant: str,
    profile: str,
) -> tuple[dict, np.ndarray]:
    np.random.seed(seed)
    parameters = PROFILES[profile]

    if variant == "notebook":
        source_fit, target_fit = source.copy(), target.copy()
        normalize = True
    elif variant == "consistent":
        source_fit, target_fit = whiten(source), whiten(target)
        normalize = False
    else:
        raise ValueError(variant)

    model = HiWA(
        # The original demo explicitly uses Isomap. Its disconnected-neighbor
        # warning is retained as reproduction evidence rather than suppressed.
        dim_red_method=Isomap(n_components=2),
        normalize=normalize,
        shorn_gamma=2e-1,
        sa_tol=1e-2,
        sa_shorn_gamma=1e-1,
        **parameters,
    )

    started = time.perf_counter()
    aligned = model.fit_transform(source_fit, source_labels, target_fit, target_labels)
    elapsed = time.perf_counter() - started

    row_ind, col_ind = linear_sum_assignment(-model.P)
    residuals = model.diagnostics["Rg_norm"]
    result = {
        "seed": seed,
        "variant": variant,
        "profile": profile,
        "before_accuracy": nearest_neighbor_accuracy(
            source_fit, source_labels, target_fit, target_labels
        ),
        "after_accuracy": nearest_neighbor_accuracy(
            aligned, source_labels, target_fit, target_labels
        ),
        "assignment_mass": float(model.P[row_ind, col_ind].sum()),
        "identity_mass": float(np.trace(model.P)),
        "iterations": int(len(residuals)),
        "final_residual": float(residuals[-1]),
        "converged": bool(residuals[-1] <= parameters["tol"]),
        "elapsed_seconds": elapsed,
        "transport_P": model.P,
        "rotation_R": model.Rg,
    }
    return result, aligned


def save_figure(
    source: np.ndarray,
    target: np.ndarray,
    aligned: np.ndarray,
    labels: np.ndarray,
    variant: str,
    profile: str,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, data, title in zip(
        axes,
        (source, aligned, target),
        ("Source", "Aligned source", "Target"),
    ):
        axis.scatter(data[:, 0], data[:, 1], c=labels, s=9, cmap="tab10")
        axis.set_title(title)
        axis.set_xlabel("dimension 1")
        axis.set_ylabel("dimension 2")
        axis.set_aspect("equal", adjustable="datalim")
    figure.savefig(FIGURES_DIR / f"synthetic_{profile}_{variant}.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    with np.load(PYHIWA_ROOT / "data" / "sg_demo.npz") as archive:
        source = archive["X_te"]
        source_labels = archive["T_te"]
        target = archive["X_tr"]
        target_labels = archive["T_tr"]

    results = []
    representative = {}
    for variant in args.variants:
        for seed in args.seeds:
            result, aligned = run_variant(
                source,
                source_labels,
                target,
                target_labels,
                seed,
                variant,
                args.profile,
            )
            results.append(result)
            representative.setdefault(variant, aligned)
            print(
                f"{variant=} {seed=} before={result['before_accuracy']:.3f} "
                f"after={result['after_accuracy']:.3f} "
                f"residual={result['final_residual']:.4g} "
                f"converged={result['converged']}",
                flush=True,
            )

    for variant, aligned in representative.items():
        displayed_source = source if variant == "notebook" else whiten(source)
        displayed_target = target if variant == "notebook" else whiten(target)
        save_figure(
            displayed_source,
            displayed_target,
            aligned,
            source_labels,
            variant,
            args.profile,
        )

    payload = {
        "experiment": "synthetic_hiwas_demo",
        "source_file": str(PYHIWA_ROOT / "data" / "sg_demo.npz"),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "results": results,
    }
    output = RESULTS_DIR / f"synthetic_{args.profile}.json"
    write_json(output, payload)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
