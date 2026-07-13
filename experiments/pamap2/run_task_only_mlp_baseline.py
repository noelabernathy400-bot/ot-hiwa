"""Source-only MLP activity baseline for the locked PAMAP2 temporal protocol.

This is a diagnostic control for the differentiable OT MVP.  It uses only
source wrist features and source labels during fitting.  Target adaptation,
target labels and pair IDs are not used before final evaluation.
"""

from __future__ import annotations

import argparse
import platform
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from cc_hiwa.common import RESULTS_DIR, ensure_output_dirs, write_json
from datasets.pamap2 import ACTIVITY_NAMES, PAMAP2SplitCounts, build_temporal_domain_adaptation_split, load_protocol_subject
from representation import MLPEncoder


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
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=501)
    parser.add_argument("--tag", default="pamap2_task_only_mlp_subject101_seed501")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    counts = PAMAP2SplitCounts(args.source_train_windows, args.source_validation_windows, args.target_adaptation_windows, args.target_test_windows)
    counts.validate()
    ensure_output_dirs()
    split = build_temporal_domain_adaptation_split(
        load_protocol_subject(args.raw_directory, args.subject),
        subject=args.subject,
        activities=tuple(ACTIVITY_NAMES),
        window_samples=args.window_samples,
        counts=counts,
    )
    scaler = StandardScaler().fit(split.source_train_features)
    source_train = torch.as_tensor(scaler.transform(split.source_train_features), dtype=torch.float32)
    source_validation = torch.as_tensor(scaler.transform(split.source_validation_features), dtype=torch.float32)
    target_test = torch.as_tensor(scaler.transform(split.evaluation_target_features), dtype=torch.float32)
    offset = int(split.source_train_labels.min())
    source_labels = torch.as_tensor(split.source_train_labels - offset, dtype=torch.long)
    validation_labels = torch.as_tensor(split.source_validation_labels - offset, dtype=torch.long)
    encoder = MLPEncoder(source_train.shape[1], args.latent_dimension, args.hidden_dimension)
    head = nn.Linear(args.latent_dimension, len(ACTIVITY_NAMES))
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=2e-3)
    best_accuracy, best_epoch, best_state = float("-inf"), -1, None
    history: list[dict[str, float]] = []
    for epoch in range(args.epochs):
        loss = F.cross_entropy(head(encoder(source_train)), source_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            validation_accuracy = float((head(encoder(source_validation)).argmax(dim=1) == validation_labels).float().mean())
        if validation_accuracy > best_accuracy:
            best_accuracy, best_epoch = validation_accuracy, epoch
            best_state = {"encoder": deepcopy(encoder.state_dict()), "head": deepcopy(head.state_dict())}
        if epoch == 0 or epoch == args.epochs - 1 or (epoch + 1) % 25 == 0:
            history.append({"epoch": float(epoch), "loss": float(loss.detach()), "source_validation_accuracy": validation_accuracy})
    if best_state is None:
        raise RuntimeError("no task-only checkpoint was produced")
    encoder.load_state_dict(best_state["encoder"])
    head.load_state_dict(best_state["head"])
    # First target-label access, after all source-only fitting is complete.
    target_labels = split.evaluation_target_labels - offset
    with torch.no_grad():
        target_prediction = head(encoder(target_test)).argmax(dim=1).numpy()
    payload = {
        "experiment": "pamap2_source_only_mlp_task_baseline",
        "scope": "source-only neural task control; no target adaptation, labels, or pair IDs in fit",
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
        "evaluation": {
            "best_source_validation_accuracy": best_accuracy,
            "target_test_accuracy": float(accuracy_score(target_labels, target_prediction)),
            "target_test_macro_f1": float(f1_score(target_labels, target_prediction, average="macro")),
        },
        "best_epoch": best_epoch,
        "history": history,
    }
    output = RESULTS_DIR / f"{args.tag}.json"
    write_json(output, payload)
    print(f"accuracy source_validation={best_accuracy:.4f} target_test={payload['evaluation']['target_test_accuracy']:.4f}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
