"""Audit the leakage-safe PAMAP2 temporal domain-adaptation split.

This is deliberately a protocol audit, not a Soft-GCOT experiment.  It fits
only source-wrist labels and evaluates a held-out chest time block.  The chest
adaptation block is available only to fit a label-free target normalizer.
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT / "src", ROOT / "src" / "cc_hiwa"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import RESULTS_DIR, ensure_output_dirs, write_json
from datasets.pamap2 import (
    ACTIVITY_NAMES,
    PAMAP2SplitCounts,
    build_temporal_domain_adaptation_split,
    load_protocol_subject,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path, default=ROOT / "data" / "raw" / "pamap2" / "PAMAP2_Dataset")
    parser.add_argument("--subject", type=int, default=101)
    parser.add_argument("--window-samples", type=int, default=200)
    parser.add_argument("--source-train-windows", type=int, default=8)
    parser.add_argument("--source-validation-windows", type=int, default=2)
    parser.add_argument("--target-adaptation-windows", type=int, default=3)
    parser.add_argument("--target-test-windows", type=int, default=3)
    parser.add_argument("--tag", default="pamap2_temporal_protocol_subject101")
    return parser.parse_args()


def _metrics(model: LogisticRegression, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    prediction = model.predict(features)
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro")),
    }


def main() -> None:
    args = parse_args()
    counts = PAMAP2SplitCounts(
        source_train=args.source_train_windows,
        source_validation=args.source_validation_windows,
        target_adaptation=args.target_adaptation_windows,
        target_test=args.target_test_windows,
    )
    counts.validate()
    ensure_output_dirs()
    rows = load_protocol_subject(args.raw_directory, args.subject)
    split = build_temporal_domain_adaptation_split(
        rows,
        subject=args.subject,
        activities=tuple(ACTIVITY_NAMES),
        window_samples=args.window_samples,
        counts=counts,
    )

    source_scaler = StandardScaler().fit(split.source_train_features)
    source_train = source_scaler.transform(split.source_train_features)
    source_validation = source_scaler.transform(split.source_validation_features)
    source_model = LogisticRegression(max_iter=1000, random_state=0).fit(source_train, split.source_train_labels)
    source_only_target = source_scaler.transform(split.evaluation_target_features)

    target_scaler = StandardScaler().fit(split.target_adaptation_features)
    transductive_target = target_scaler.transform(split.evaluation_target_features)
    result = {
        "experiment": "pamap2_temporal_domain_adaptation_protocol_audit",
        "scope": "fixed source-only and label-free target-normalization baselines; no OT, target labels, or pair IDs in fitting",
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "environment": {"python": platform.python_version()},
        "contract": {
            "source_labels": "available only in source train and validation",
            "target_adaptation": "chest features only; no labels or pair IDs exposed to fitting",
            "target_test": "held out from all scalers and fitting; labels and pair IDs are evaluation-only",
            "partition_order_per_activity": ["source_train", "source_validation", "target_adaptation", "target_test"],
        },
        "data": {
            "subject": split.subject,
            "source_view": split.source_view,
            "target_view": split.target_view,
            "window_samples": split.window_samples,
            "counts_per_activity": {
                "source_train": counts.source_train,
                "source_validation": counts.source_validation,
                "target_adaptation": counts.target_adaptation,
                "target_test": counts.target_test,
            },
            "samples": {
                "source_train": int(len(split.source_train_features)),
                "source_validation": int(len(split.source_validation_features)),
                "target_adaptation": int(len(split.target_adaptation_features)),
                "target_test": int(len(split.evaluation_target_features)),
            },
        },
        "evaluation": {
            "source_validation": _metrics(source_model, source_validation, split.source_validation_labels),
            "source_only_target_test": _metrics(source_model, source_only_target, split.evaluation_target_labels),
            "label_free_target_normalization_target_test": _metrics(source_model, transductive_target, split.evaluation_target_labels),
        },
    }
    output = RESULTS_DIR / f"{args.tag}.json"
    write_json(output, result)
    print(
        "accuracy "
        f"source_validation={result['evaluation']['source_validation']['accuracy']:.4f} "
        f"source_only_target={result['evaluation']['source_only_target_test']['accuracy']:.4f} "
        f"target_normalized={result['evaluation']['label_free_target_normalization_target_test']['accuracy']:.4f}"
    )
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
