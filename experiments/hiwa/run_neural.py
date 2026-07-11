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
from scipy.io import loadmat
from sklearn.decomposition import FactorAnalysis
from sklearn.manifold import Isomap

from common import (
    FIGURES_DIR,
    PYHIWA_ROOT,
    RESULTS_DIR,
    HiWA,
    ensure_output_dirs,
    nearest_neighbor_accuracy,
    original_r2,
    write_json,
)


PROFILES = {
    "quick": dict(
        maxiter=12,
        tol=1e-2,
        mu=2e-2,
        shorn_maxiter=100,
        sa_maxiter=12,
        sa_shorn_maxiter=30,
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
    return parser.parse_args()


def remove_constant_columns(data: np.ndarray) -> np.ndarray:
    return data[:, ~np.all(data[1:] == data[:-1], axis=0)]


def movement_to_3d(movement: np.ndarray) -> np.ndarray:
    return np.column_stack((movement[:, 0], movement[:, 1], np.linalg.norm(movement, axis=1)))


def least_squares_rotation(movement_3d: np.ndarray, neural_3d: np.ndarray) -> np.ndarray:
    return movement_3d.T @ np.linalg.pinv(neural_3d).T


def load_demo() -> dict[str, np.ndarray]:
    raw = loadmat(PYHIWA_ROOT / "data" / "mihi_demo.mat")
    return {
        "train_labels": raw["Ttr"].ravel(),
        "test_labels": raw["Tte"].ravel(),
        "train_movement": raw["Xtr"],
        "test_movement": raw["Xte"],
        "train_neural": raw["Ytr"],
        "test_neural": raw["Yte"],
    }


def run_seed(data: dict[str, np.ndarray], seed: int, profile: str) -> tuple[dict, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    test_neural = remove_constant_columns(data["test_neural"])
    # Keep factor analysis fixed while varying only HiWA's non-convex
    # initialization. The original notebook relies on sklearn's fixed default.
    neural_3d = FactorAnalysis(n_components=3, random_state=0).fit_transform(test_neural)
    train_movement_3d = movement_to_3d(data["train_movement"])
    test_movement_3d = movement_to_3d(data["test_movement"])
    target_transform = np.linalg.pinv(train_movement_3d) @ data["train_movement"]
    oracle_rotation = least_squares_rotation(test_movement_3d, neural_3d)

    model = HiWA(
        dim_red_method=Isomap(n_components=2, n_neighbors=12),
        normalize=True,
        shorn_gamma=2e-1,
        sa_tol=1e-2,
        sa_shorn_gamma=1e-1,
        **PROFILES[profile],
    )
    started = time.perf_counter()
    aligned = model.fit_transform(
        neural_3d,
        data["test_labels"],
        train_movement_3d,
        data["train_labels"],
        Y_transform=target_transform,
        Rgt=oracle_rotation,
    )
    elapsed = time.perf_counter() - started

    residuals = model.diagnostics["Rg_norm"]
    result = {
        "seed": seed,
        "profile": profile,
        "before_direction_accuracy": nearest_neighbor_accuracy(
            neural_3d,
            data["test_labels"],
            train_movement_3d,
            data["train_labels"],
        ),
        "after_direction_accuracy": nearest_neighbor_accuracy(
            aligned,
            data["test_labels"],
            train_movement_3d,
            data["train_labels"],
        ),
        "before_velocity_r2": original_r2(neural_3d[:, :2], data["test_movement"]),
        "after_velocity_r2": original_r2(aligned[:, :2], data["test_movement"]),
        "iterations": int(len(residuals)),
        "final_residual": float(residuals[-1]),
        "converged": bool(residuals[-1] <= PROFILES[profile]["tol"]),
        "elapsed_seconds": elapsed,
        "transport_P": model.P,
        "rotation_R": model.Rg,
        "active_test_neurons": int(test_neural.shape[1]),
    }
    return result, neural_3d, aligned


def save_figure(
    neural_3d: np.ndarray,
    aligned: np.ndarray,
    movement_3d: np.ndarray,
    neural_labels: np.ndarray,
    movement_labels: np.ndarray,
    profile: str,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    panels = (
        (neural_3d, neural_labels, "Neural latent before"),
        (aligned, neural_labels, "Neural latent aligned"),
        (movement_3d, movement_labels, "Movement target"),
    )
    for axis, (values, labels, title) in zip(axes, panels):
        axis.scatter(values[:, 0], values[:, 1], c=labels, s=9, cmap="tab10")
        axis.set_title(title)
        axis.set_xlabel("dimension 1")
        axis.set_ylabel("dimension 2")
        axis.set_aspect("equal", adjustable="datalim")
    figure.savefig(FIGURES_DIR / f"neural_{profile}.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    data = load_demo()
    results = []
    representative = None
    for seed in args.seeds:
        result, neural_3d, aligned = run_seed(data, seed, args.profile)
        results.append(result)
        representative = representative or (neural_3d, aligned)
        print(
            f"seed={seed} direction: {result['before_direction_accuracy']:.3f} -> "
            f"{result['after_direction_accuracy']:.3f}; velocity R2: "
            f"{result['before_velocity_r2']:.3f} -> {result['after_velocity_r2']:.3f}; "
            f"converged={result['converged']}"
        )

    if representative is not None:
        save_figure(
            representative[0],
            representative[1],
            movement_to_3d(data["train_movement"]),
            data["test_labels"],
            data["train_labels"],
            args.profile,
        )

    payload = {
        "experiment": "mihi_neural_demo",
        "source_file": str(PYHIWA_ROOT / "data" / "mihi_demo.mat"),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "data_shapes": {key: list(value.shape) for key, value in data.items()},
        "results": results,
    }
    output = RESULTS_DIR / f"neural_{args.profile}.json"
    write_json(output, payload)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
