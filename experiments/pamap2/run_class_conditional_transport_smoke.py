"""One pre-registered feasibility check for class-conditional Soft-GCOT.

The source wrist labels define semantic source groups.  A source-trained
classifier produces soft chest pseudo-groups on target adaptation features.
Target-test labels and synchronous pair IDs are touched only after transport
and classifiers have been fitted.
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

from class_conditional_transport import solve_class_conditional_soft_gcot_detached
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
    parser.add_argument("--source-smoothing", type=float, default=0.02)
    parser.add_argument("--target-temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--tag", default="pamap2_class_conditional_smoke_subject101_seed401")
    return parser.parse_args()


def _solver_kwargs(seed: int) -> dict[str, object]:
    return {
        "normalize": True,
        "maxiter": 80,
        "tol": 0.10,
        "mu": 5e-3,
        "shorn_maxiter": 120,
        "sa_maxiter": 12,
        "sa_tol": 1e-2,
        "sa_shorn_maxiter": 40,
        "sa_shorn_gamma": 1e-1,
        "shorn_gamma": 2e-1,
        "support_mode": "full",
        "random_state": seed,
        "warm_start_local": True,
    }


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
    if not 0.0 <= args.source_smoothing < 1.0 or args.target_temperature <= 0:
        raise ValueError("source smoothing must be in [0, 1) and target temperature must be positive")
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
    target_scaler = StandardScaler().fit(split.target_adaptation_features)
    source_train = source_scaler.transform(split.source_train_features)
    source_validation = source_scaler.transform(split.source_validation_features)
    target_adaptation = target_scaler.transform(split.target_adaptation_features)
    target_test = target_scaler.transform(split.evaluation_target_features)
    label_offset = int(split.source_train_labels.min())
    source_labels = split.source_train_labels - label_offset

    source_task = LogisticRegression(max_iter=1000, random_state=args.seed).fit(source_train, source_labels)
    validation = _metrics(source_task, source_validation, split.source_validation_labels - label_offset)
    target_adaptation_logits = source_task.decision_function(target_adaptation)
    conditional = solve_class_conditional_soft_gcot_detached(
        source_train,
        source_labels,
        target_adaptation,
        target_adaptation_logits,
        n_classes=len(ACTIVITY_NAMES),
        source_smoothing=args.source_smoothing,
        target_temperature=args.target_temperature,
        solver_kwargs=_solver_kwargs(args.seed),
    )
    aligned_source_task = LogisticRegression(max_iter=1000, random_state=args.seed).fit(
        conditional.transport.aligned_source,
        source_labels,
    )

    # The following line is the first target-label access in this script.
    target_test_labels = split.evaluation_target_labels - label_offset
    target_pseudo_prediction = source_task.predict(target_test)
    result = {
        "experiment": "pamap2_class_conditional_soft_gcot_feasibility",
        "scope": "single pre-registered fixed-representation smoke test; not a final benchmark",
        "label_usage": {
            "source_activity_labels": "anchor source semantic groups and train the source task classifier",
            "target_adaptation": "features only; class memberships come from source-task logits",
            "target_test_labels": "opened after fitting only for evaluation",
            "pair_ids": "not read by this experiment",
        },
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "environment": {"python": platform.python_version()},
        "data": {
            "subject": split.subject,
            "source_view": split.source_view,
            "target_view": split.target_view,
            "samples": {
                "source_train": int(len(source_train)),
                "source_validation": int(len(source_validation)),
                "target_adaptation": int(len(target_adaptation)),
                "target_test": int(len(target_test)),
            },
        },
        "evaluation": {
            "source_validation": validation,
            "source_task_target_test_pseudolabel": {
                "accuracy": float(accuracy_score(target_test_labels, target_pseudo_prediction)),
                "macro_f1": float(f1_score(target_test_labels, target_pseudo_prediction, average="macro")),
            },
            "class_conditional_target_test": _metrics(aligned_source_task, target_test, target_test_labels),
        },
        "conditional_diagnostics": {
            "group_matching": "semantic diagonal lock",
            "target_prediction_entropy": conditional.target_prediction_entropy,
            "target_predicted_group_mass": conditional.target_predicted_group_mass,
            "admm_converged": bool(conditional.transport.diagnostics["admm_converged"]),
            "final_primal_residual": float(conditional.transport.diagnostics["admm_primal_residual"][-1]),
            "group_transport": conditional.transport.group_transport,
        },
    }
    output = RESULTS_DIR / f"{args.tag}.json"
    write_json(output, result)
    print(
        "accuracy "
        f"source_validation={validation['accuracy']:.4f} "
        f"class_conditional_target={result['evaluation']['class_conditional_target_test']['accuracy']:.4f}"
    )
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
