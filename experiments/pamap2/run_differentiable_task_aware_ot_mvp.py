"""Fully differentiable source-supervised / target-unlabelled PAMAP2 OT MVP.

No detached NumPy solver is used. Source labels enter only source task loss
and source semantic groups. Target labels, synchronized pair IDs, and target
test windows are not supplied to ``fit``.
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT / "src", ROOT / "src" / "cc_hiwa"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import RESULTS_DIR, ensure_output_dirs, write_json
from datasets.pamap2 import ACTIVITY_NAMES, PAMAP2SplitCounts, build_temporal_domain_adaptation_split, load_protocol_subject
from representation import DifferentiableTaskAwareOT, MLPDecoder, MLPEncoder, TaskAwareTransportConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path, default=ROOT / "data" / "raw" / "pamap2" / "PAMAP2_Dataset")
    parser.add_argument("--subject", type=int, default=101)
    parser.add_argument("--window-samples", type=int, default=200)
    parser.add_argument("--source-train-windows", type=int, default=8)
    parser.add_argument("--source-validation-windows", type=int, default=2)
    parser.add_argument("--target-adaptation-windows", type=int, default=3)
    parser.add_argument("--target-test-windows", type=int, default=3)
    parser.add_argument("--latent-dimension", type=int, default=16)
    parser.add_argument("--hidden-dimension", type=int, default=32)
    parser.add_argument("--source-warmup-epochs", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=501)
    parser.add_argument("--tag", default="pamap2_differentiable_task_aware_ot_subject101_seed501")
    return parser.parse_args()


def _paired_recall(source: np.ndarray, target: np.ndarray, maximum_k: int = 5) -> dict[str, float]:
    distances = np.sum((source[:, None, :] - target[None, :, :]) ** 2, axis=2)
    order = np.argsort(distances, axis=1)
    truth = np.arange(len(source))[:, None]
    return {"recall_at_1": float(np.mean(order[:, :1] == truth)), f"recall_at_{maximum_k}": float(np.mean(np.any(order[:, :maximum_k] == truth, axis=1)))}


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    counts = PAMAP2SplitCounts(args.source_train_windows, args.source_validation_windows, args.target_adaptation_windows, args.target_test_windows)
    counts.validate()
    ensure_output_dirs()
    rows = load_protocol_subject(args.raw_directory, args.subject)
    split = build_temporal_domain_adaptation_split(rows, subject=args.subject, activities=tuple(ACTIVITY_NAMES), window_samples=args.window_samples, counts=counts)
    source_scaler = StandardScaler().fit(split.source_train_features)
    target_scaler = StandardScaler().fit(split.target_adaptation_features)
    source_train = source_scaler.transform(split.source_train_features).astype(np.float32)
    source_validation = source_scaler.transform(split.source_validation_features).astype(np.float32)
    target_adaptation = target_scaler.transform(split.target_adaptation_features).astype(np.float32)
    source_evaluation = source_scaler.transform(split.evaluation_source_features).astype(np.float32)
    target_test = target_scaler.transform(split.evaluation_target_features).astype(np.float32)
    label_offset = int(split.source_train_labels.min())
    source_labels = split.source_train_labels - label_offset
    source_validation_labels = split.source_validation_labels - label_offset
    model = DifferentiableTaskAwareOT(
        MLPEncoder(source_train.shape[1], args.latent_dimension, args.hidden_dimension),
        MLPEncoder(target_adaptation.shape[1], args.latent_dimension, args.hidden_dimension),
        MLPDecoder(args.latent_dimension, source_train.shape[1], args.hidden_dimension),
        MLPDecoder(args.latent_dimension, target_adaptation.shape[1], args.hidden_dimension),
        nn.Linear(args.latent_dimension, len(ACTIVITY_NAMES)),
        n_classes=len(ACTIVITY_NAMES),
        config=TaskAwareTransportConfig(source_warmup_epochs=args.source_warmup_epochs, epochs=args.epochs),
    )
    result = model.fit(source_train, source_labels, target_adaptation, source_validation, source_validation_labels)
    target_prediction = model.predict_target(target_test)
    source_evaluation_latent = model.encode_source(source_evaluation)
    target_test_latent = model.encode_target(target_test)
    # First target-label and pair-order use: final held-out evaluation only.
    target_test_labels = split.evaluation_target_labels - label_offset
    payload = {
        "experiment": "pamap2_fully_differentiable_task_aware_hierarchical_ot_mvp",
        "scope": "single fixed-protocol feasibility run; no target labels, pair IDs, or target-test data in fit",
        "label_usage": {"source_activity_labels": "source task loss and source semantic groups only", "target_adaptation": "features only; group probabilities come from task-head outputs", "target_test_labels": "opened after fit only for classification metrics", "pair_ids": "not supplied to fit; held-out order used only for retrieval metrics"},
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
        "data": {"subject": split.subject, "source_view": split.source_view, "target_view": split.target_view, "samples": {"source_train": int(len(source_train)), "source_validation": int(len(source_validation)), "target_adaptation": int(len(target_adaptation)), "target_test": int(len(target_test))}},
        "evaluation": {"best_source_validation_accuracy": result.best_source_validation_accuracy, "target_test_accuracy": float(accuracy_score(target_test_labels, target_prediction)), "target_test_macro_f1": float(f1_score(target_test_labels, target_prediction, average="macro")), "paired_retrieval": _paired_recall(source_evaluation_latent, target_test_latent)},
        "transport_diagnostics": {"best_epoch": result.best_epoch, "group_transport": result.group_transport, "sample_transport_row_error": float(np.max(np.abs(result.sample_transport.sum(axis=1) - 1.0 / len(source_train)))), "sample_transport_column_error": float(np.max(np.abs(result.sample_transport.sum(axis=0) - 1.0 / len(target_adaptation)))), "history": result.history},
    }
    output = RESULTS_DIR / f"{args.tag}.json"
    write_json(output, payload)
    print(f"accuracy source_validation={payload['evaluation']['best_source_validation_accuracy']:.4f} target_test={payload['evaluation']['target_test_accuracy']:.4f}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
