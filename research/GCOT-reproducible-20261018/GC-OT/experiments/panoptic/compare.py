"""Compare label-blind alignment methods on the Panoptic camera domains."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from threadpoolctl import threadpool_limits

from gcot.alignment import TPAMIGCOT
from gcot.baselines import coral, icp, sa, wa

from .config import FIXED_INITIALIZATION, N_GROUPS, SOLVER


def evaluate(
    aligned: np.ndarray,
    target: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    distances, indices = NearestNeighbors(n_neighbors=5).fit(target).kneighbors(aligned)
    truth = np.arange(len(target))[:, None]
    predictions = KNeighborsClassifier(5, weights="distance").fit(target, labels).predict(aligned)
    return {
        "paired_rmse": float(np.sqrt(np.mean((aligned - target) ** 2))),
        "paired_recall_at_1": float(np.mean(indices[:, :1] == truth)),
        "paired_recall_at_5": float(np.mean(np.any(indices == truth, axis=1))),
        "three_group_accuracy_5nn": float(accuracy_score(labels, predictions)),
        "nearest_neighbor_distance": float(np.mean(distances[:, 0])),
    }


def run(input_path: Path, output_dir: Path) -> None:
    data = np.load(input_path)
    source = np.asarray(data["source"], dtype=float)
    target = np.asarray(data["target"], dtype=float)
    labels = np.asarray(data["labels"], dtype=int)
    rng = np.random.default_rng(FIXED_INITIALIZATION)
    shuffled_target = target[rng.permutation(len(target))]

    methods: dict[str, tuple[np.ndarray, float]] = {"No adaptation": (source, 0.0)}
    with threadpool_limits(limits=1):
        started = time.monotonic()
        gcot = TPAMIGCOT(
            groups=N_GROUPS,
            seed=FIXED_INITIALIZATION,
            entropy_weight=1.0,
            prototype_maxiter=250,
            config=SOLVER,
            retain_plans=False,
        ).fit(source, shuffled_target)
        methods["GCOT"] = (gcot.transform(source), time.monotonic() - started)
        functions = {
            "WA": lambda: wa(source, shuffled_target, seed=FIXED_INITIALIZATION, maxiter=250),
            "SA": lambda: sa(source, shuffled_target),
            "ICP": lambda: icp(source, shuffled_target, maxiter=100),
            "CORAL": lambda: coral(source, shuffled_target),
        }
        for name, function in functions.items():
            started = time.monotonic()
            fitted = function()
            elapsed = time.monotonic() - started
            aligned = source @ fitted["R"].T
            if name == "ICP":
                aligned += fitted["translation"]
            methods[name] = (aligned, elapsed)

    arrays = {
        "source": source,
        "target": target,
        "labels": labels,
        "frame_ids": data["frame_ids"],
        "source_tsne": data["source_tsne"],
        "target_tsne": data["target_tsne"],
    }
    rows = []
    for name, (aligned, seconds) in methods.items():
        rows.append({"method": name, **evaluate(aligned, target, labels), "seconds": seconds})
        arrays[f"aligned_{name.lower().replace(' ', '_')}"] = aligned

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "aligned.npz", **arrays)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    run(args.input, args.output_dir)


if __name__ == "__main__":
    main()
