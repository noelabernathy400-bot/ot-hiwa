#!/usr/bin/env python3
"""Fair exact empirical W2² scoring: identical data and uniform mass for all methods.

Requires real full-precision Hyser 250x3 X,Y (from prepare_from_wfdb.py).
Read R from fitted_R_15runs.csv, archived from prior no-leakage experiments.
Only X,Y,R enter the loss: no true time pairings, source/target labels, P or Q.

python experiments/hyser/unsupervised_control/evaluate_common_w2.py \
 --data experiments/hyser/data/paired_full_precision.npz \
 --rotations experiments/hyser/unsupervised_control/fitted_R_15runs.csv \
 --out experiments/hyser/unsupervised_control/common_w2_recalculated.csv
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def common_w2_sq(X, Y, R):
    n, d = X.shape
    if Y.shape != (n, d) or R.shape != (d, d):
        raise ValueError("Expected X/Y matching n*d and square R")
    if np.linalg.norm(R.T @ R - np.eye(d)) > 1e-5:
        raise ValueError("R must be orthogonal")
    C = cdist(X @ R.T / np.sqrt(d), Y / np.sqrt(d), "sqeuclidean")
    source_idx, target_idx = linear_sum_assignment(C)
    return float(C[source_idx, target_idx].sum() / n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--rotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.data, allow_pickle=False) as z:
        X = np.asarray(z["X"], dtype=float)
        Y = np.asarray(z["Y"], dtype=float)
    if X.shape != Y.shape or X.shape != (250, 3):
        raise ValueError("Expected real 250x3 Hyser data")

    baseline = common_w2_sq(X, Y, np.eye(3))
    output = []
    with args.rotations.open(newline="") as f:
        for r in csv.DictReader(f):
            R = np.array([float(r[f"r{i}{j}"]) for i in range(3)
                          for j in range(3)]).reshape(3, 3)
            after = common_w2_sq(X, Y, R)
            output.append({
                "seed": int(r["seed"]), "method": r["method"],
                "common_w2_before": baseline, "common_w2_after": after,
                "relative_reduction_pct": 100 * (baseline - after) / baseline
            })
    if len(output) != 15 or len({(r["seed"], r["method"]) for r in output}) != 15:
        raise ValueError("Expected exactly 15 unique seed/method outcomes")
    fields = list(output[0])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    print("Before:", baseline)
    for method in ("hard", "soft_kmeans", "soft_prototype"):
        v = np.array([r["common_w2_after"] for r in output
                      if r["method"] == method])
        print(method, "mean", v.mean(), "sample SD", v.std(ddof=1))
    print("Saved", args.out)


if __name__ == "__main__":
    main()
