"""Label-free alignment comparison for DANDI 000688 cross-day latents."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from threadpoolctl import threadpool_limits

from gcot.alignment import TPAMIGCOT
from gcot.baselines import coral, icp, sa, wa

from .config import FIXED_INITIALIZATION, N_GROUPS, SOLVER



def _centroids(x: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return np.vstack([x[labels == k].mean(axis=0) for k in classes])


def evaluate(
    aligned: np.ndarray,
    source_labels: np.ndarray,
    target: np.ndarray,
    target_labels: np.ndarray,
) -> dict[str, float]:
    classes = np.intersect1d(np.unique(source_labels), np.unique(target_labels))
    source_centers = _centroids(aligned, source_labels, classes)
    target_centers = _centroids(target, target_labels, classes)
    center_cost = np.linalg.norm(source_centers[:, None] - target_centers[None], axis=2)
    rows, cols = linear_sum_assignment(center_cost)
    matching = float(np.mean(rows == cols))
    scale = float(np.sqrt(np.mean(np.sum(target**2, axis=1))))
    centroid_nrmse = float(np.sqrt(np.mean((source_centers - target_centers) ** 2)) / scale)
    one_nn = KNeighborsClassifier(1).fit(target, target_labels).predict(aligned)
    five_nn = KNeighborsClassifier(5, weights="distance").fit(target, target_labels).predict(aligned)
    tree = NearestNeighbors(n_neighbors=1).fit(target)
    chamfer_xy = float(np.mean(tree.kneighbors(aligned, return_distance=True)[0] ** 2))
    tree_reverse = NearestNeighbors(n_neighbors=1).fit(aligned)
    chamfer_yx = float(np.mean(tree_reverse.kneighbors(target, return_distance=True)[0] ** 2))
    return {
        "cluster_matching_accuracy": matching,
        "direction_accuracy_1nn": float(accuracy_score(source_labels, one_nn)),
        "direction_accuracy_5nn": float(accuracy_score(source_labels, five_nn)),
        "centroid_nrmse": centroid_nrmse,
        "symmetric_chamfer": 0.5 * (chamfer_xy + chamfer_yx),
    }


def run(input_path: Path, output_dir: Path) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"Latent feature file not found: {input_path}")
    # One fixed, reproducible initialization; this is a direct run, not a sweep.
    seed = FIXED_INITIALIZATION
    data = np.load(input_path)
    source = np.asarray(data["source"], dtype=float)
    target = np.asarray(data["target"], dtype=float)
    source_labels = np.asarray(data["source_labels"], dtype=int)
    target_labels = np.asarray(data["target_labels"], dtype=int)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError(f"Incompatible domains: source={source.shape}, target={target.shape}")
    if len(source) != len(source_labels) or len(target) != len(target_labels):
        raise ValueError("Every feature row must have exactly one evaluation label")
    rng = np.random.default_rng(seed)
    n_fit = min(len(source), len(target))
    source_fit = source[rng.choice(len(source), n_fit, replace=False)]
    target_fit = target[rng.choice(len(target), n_fit, replace=False)]
    methods: dict[str, tuple[np.ndarray, float]] = {"No adaptation": (source, 0.0)}

    with threadpool_limits(limits=1):
        started = time.monotonic()
        gcot = TPAMIGCOT(
            groups=N_GROUPS,
            seed=seed,
            entropy_weight=1.0,
            prototype_maxiter=250,
            config=SOLVER,
            retain_plans=False,
        ).fit(source_fit, target_fit)
        methods["GCOT"] = (gcot.transform(source), time.monotonic() - started)
        functions = {
            "WA": lambda: wa(source_fit, target_fit, seed=seed, maxiter=250),
            "SA": lambda: sa(source_fit, target_fit),
            "ICP": lambda: icp(source_fit, target_fit, maxiter=100),
            "CORAL": lambda: coral(source_fit, target_fit),
        }
        for name, function in functions.items():
            started = time.monotonic()
            fitted = function()
            elapsed = time.monotonic() - started
            aligned = (
                source @ fitted["R"].T + fitted["translation"]
                if name == "ICP"
                else source @ fitted["R"].T
            )
            methods[name] = (aligned, elapsed)

    rows = []
    arrays = {
        "source": source,
        "target": target,
        "source_labels": source_labels,
        "target_labels": target_labels,
    }
    for name, (aligned, seconds) in methods.items():
        rows.append(
            {"method": name, **evaluate(aligned, source_labels, target, target_labels), "seconds": seconds}
        )
        arrays[f"aligned_{name.lower().replace(' ', '_')}"] = aligned
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "aligned.npz", **arrays)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    run(args.input, args.output_dir)


if __name__ == "__main__":
    main()
