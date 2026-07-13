"""Paired upper-bound audit for PAMAP2 wrist--chest feature geometry.

This is not an unsupervised-method result.  It deliberately uses held-out
synchronous pair IDs to ask a narrower question: after the fixed feature
contract, can a known correspondence be represented by an orthogonal map, or
does it require a substantially more general linear transformation?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from datasets.pamap2 import ACTIVITY_NAMES, build_paired_windows, load_protocol_subject


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path, default=ROOT / "data" / "raw" / "pamap2" / "PAMAP2_Dataset")
    parser.add_argument("--subject", type=int, default=101)
    parser.add_argument("--windows-per-activity", type=int, default=16)
    parser.add_argument("--window-samples", type=int, default=200)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "results" / "pamap2_wrist_chest_eligibility_subject101.json")
    return parser.parse_args()


def _retrieval(source: np.ndarray, target: np.ndarray) -> dict[str, float]:
    order = np.argsort(np.sum((source[:, None, :] - target[None, :, :]) ** 2, axis=2), axis=1)
    truth = np.arange(len(source))[:, None]
    return {
        "recall_at_1": float(np.mean(order[:, :1] == truth)),
        "recall_at_5": float(np.mean(np.any(order[:, :5] == truth, axis=1))),
        "mean_squared_alignment_error": float(np.mean((source - target) ** 2)),
    }


def _orthogonal_map(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    left, _, right_t = np.linalg.svd(source.T @ target, full_matrices=False)
    return left @ right_t


def _distance_correlation(source: np.ndarray, target: np.ndarray) -> float:
    source_distance = np.linalg.norm(source[:, None, :] - source[None, :, :], axis=2)
    target_distance = np.linalg.norm(target[:, None, :] - target[None, :, :], axis=2)
    mask = np.triu_indices(len(source), k=1)
    return float(np.corrcoef(source_distance[mask], target_distance[mask])[0, 1])


def main() -> None:
    args = parse_args()
    rows = load_protocol_subject(args.raw_directory, args.subject)
    prepared = build_paired_windows(
        rows,
        subject=args.subject,
        activities=tuple(ACTIVITY_NAMES),
        window_samples=args.window_samples,
        windows_per_activity=args.windows_per_activity,
    )
    source = StandardScaler().fit_transform(prepared.source_features)
    target = StandardScaler().fit_transform(prepared.target_features)
    train_indices = np.arange(0, len(source), 2)
    test_indices = np.arange(1, len(source), 2)
    source_train, target_train = source[train_indices], target[train_indices]
    source_test, target_test = source[test_indices], target[test_indices]
    rotation = _orthogonal_map(source_train, target_train)
    # A regularized linear comparison is necessary here: the first protocol
    # has as many feature dimensions as paired fitting windows, so unregularized
    # least squares is ill-conditioned and not a meaningful capacity control.
    linear_model = Ridge(alpha=1.0, fit_intercept=False)
    linear_model.fit(source_train, target_train)
    linear = linear_model.coef_.T
    result = {
        "experiment": "pamap2_paired_wrist_chest_eligibility_upper_bound",
        "warning": "uses true pairs to fit upper-bound mappings; it is not an unsupervised Soft-GCOT result",
        "protocol": {
            "subject": args.subject,
            "source_view": "wrist",
            "target_view": "chest",
            "samples": int(len(source)),
            "train_pairs": int(len(train_indices)),
            "held_out_pairs": int(len(test_indices)),
            "window_samples": args.window_samples,
            "activities": list(ACTIVITY_NAMES.values()),
            "split": "even-indexed synchronous pairs fit mapping; odd-indexed pairs evaluate it",
        },
        "held_out_evaluation": {
            "random_recall_at_1": float(1 / len(test_indices)),
            "random_recall_at_5": float(min(5 / len(test_indices), 1.0)),
            "unaligned": _retrieval(source_test, target_test),
            "orthogonal_procrustes": _retrieval(source_test @ rotation, target_test),
            "ridge_linear_alpha_1": _retrieval(source_test @ linear, target_test),
            "paired_distance_correlation": _distance_correlation(source_test, target_test),
            "orthogonal_general_linear_mse_gap": float(
                np.mean((source_test @ rotation - target_test) ** 2)
                - np.mean((source_test @ linear - target_test) ** 2)
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    evaluation = result["held_out_evaluation"]
    print(
        "held-out R@1 "
        f"unaligned={evaluation['unaligned']['recall_at_1']:.4f} "
        f"orthogonal={evaluation['orthogonal_procrustes']['recall_at_1']:.4f} "
        f"ridge={evaluation['ridge_linear_alpha_1']['recall_at_1']:.4f}"
    )
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
