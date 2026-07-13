"""PAMAP2 wrist-to-chest source-supervised Soft-GCOT representation MVP.

Protocol: source wrist activity labels train the task head.  Chest activity
labels and synchronous pair IDs are loaded only after fitting to calculate
evaluation metrics.  Neither can enter the detached Soft-GCOT solver.
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT / "src", ROOT / "src" / "cc_hiwa"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import RESULTS_DIR, ensure_output_dirs, write_json
from datasets.pamap2 import ACTIVITY_NAMES, build_paired_windows, load_protocol_subject
from representation import MLPDecoder, MLPEncoder
from soft_groups import learn_soft_groups
from solver_adapter import solve_soft_gcot_detached
from source_supervised_representation import (
    SourceSupervisedAlternatingSoftGCOT,
    SourceSupervisedRepresentationConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path, default=ROOT / "data" / "raw" / "pamap2" / "PAMAP2_Dataset")
    parser.add_argument("--subject", type=int, default=101)
    parser.add_argument("--windows-per-activity", type=int, default=16)
    parser.add_argument("--window-samples", type=int, default=200)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--latent-dimension", type=int, default=16)
    parser.add_argument("--hidden-dimension", type=int, default=32)
    parser.add_argument("--outer-steps", type=int, default=3)
    parser.add_argument("--encoder-steps", type=int, default=25)
    parser.add_argument("--source-warmup-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=301)
    parser.add_argument("--tag", default="pamap2_wrist_chest_source_supervised_seed301")
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


def _paired_recall(aligned_source: np.ndarray, target: np.ndarray, maximum_k: int = 5) -> dict[str, float]:
    distances = np.sum((aligned_source[:, None, :] - target[None, :, :]) ** 2, axis=2)
    order = np.argsort(distances, axis=1)
    truth = np.arange(len(aligned_source))[:, None]
    return {
        "recall_at_1": float(np.mean(order[:, :1] == truth)),
        f"recall_at_{maximum_k}": float(np.mean(np.any(order[:, :maximum_k] == truth, axis=1))),
    }


def _classification_metrics(train_features: np.ndarray, source_labels: np.ndarray, target_features: np.ndarray, target_labels: np.ndarray) -> dict[str, float]:
    model = LogisticRegression(max_iter=1000, random_state=0)
    model.fit(train_features, source_labels)
    predicted = model.predict(target_features)
    return {
        "source_train_accuracy": float(accuracy_score(source_labels, model.predict(train_features))),
        "target_accuracy": float(accuracy_score(target_labels, predicted)),
        "target_macro_f1": float(f1_score(target_labels, predicted, average="macro")),
    }


def main() -> None:
    args = parse_args()
    if min(args.groups, args.latent_dimension, args.hidden_dimension, args.windows_per_activity) <= 0:
        raise ValueError("groups, dimensions, and windows_per_activity must be positive")
    ensure_output_dirs()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_protocol_subject(args.raw_directory, args.subject)
    prepared = build_paired_windows(
        rows,
        subject=args.subject,
        activities=tuple(ACTIVITY_NAMES),
        window_samples=args.window_samples,
        windows_per_activity=args.windows_per_activity,
    )
    # Per-view scaling is transductive but label-free: target observations are
    # available for adaptation, while target labels remain unopened until the
    # evaluation block below.
    source = StandardScaler().fit_transform(prepared.source_features).astype(np.float32)
    target = StandardScaler().fit_transform(prepared.target_features).astype(np.float32)
    source_labels = prepared.source_labels - prepared.source_labels.min()
    target_labels = prepared.target_labels - prepared.source_labels.min()
    solver_kwargs = _solver_kwargs(args.seed)

    source_only = _classification_metrics(source, source_labels, target, target_labels)
    fixed_source = learn_soft_groups(source, args.groups, 0.50, seed=args.seed)
    fixed_target = learn_soft_groups(target, args.groups, 0.50, seed=args.seed)
    fixed_solution = solve_soft_gcot_detached(
        source,
        fixed_source.assignments,
        target,
        fixed_target.assignments,
        solver_kwargs=solver_kwargs,
    )
    fixed_metrics = _classification_metrics(
        fixed_solution.aligned_source,
        source_labels,
        target,
        target_labels,
    )

    trainer = SourceSupervisedAlternatingSoftGCOT(
        MLPEncoder(source.shape[1], args.latent_dimension, args.hidden_dimension),
        MLPEncoder(target.shape[1], args.latent_dimension, args.hidden_dimension),
        MLPDecoder(args.latent_dimension, source.shape[1], args.hidden_dimension),
        MLPDecoder(args.latent_dimension, target.shape[1], args.hidden_dimension),
        nn.Linear(args.latent_dimension, len(ACTIVITY_NAMES)),
        latent_dimension=args.latent_dimension,
        config=SourceSupervisedRepresentationConfig(
            n_groups=args.groups,
            outer_steps=args.outer_steps,
            encoder_steps=args.encoder_steps,
            source_warmup_steps=args.source_warmup_steps,
            solver_kwargs=solver_kwargs,
        ),
    )
    learned = trainer.fit(source, target, source_labels)
    predicted_target = learned.target_logits.argmax(axis=1)
    learned_metrics = {
        "source_native_train_accuracy": float(accuracy_score(source_labels, learned.source_native_logits.argmax(axis=1))),
        "source_train_accuracy": float(accuracy_score(source_labels, learned.source_logits.argmax(axis=1))),
        "target_accuracy": float(accuracy_score(target_labels, predicted_target)),
        "target_macro_f1": float(f1_score(target_labels, predicted_target, average="macro")),
    }
    learned_retrieval = _paired_recall(learned.solution.aligned_source, learned.target_latent)

    result = {
        "experiment": "pamap2_wrist_to_chest_source_supervised_representation_mvp",
        "scope": "small protocol check; not a final benchmark or a claim of end-to-end differentiation through OT",
        "label_usage": {
            "source_activity_labels": "used only by the source task-head cross-entropy loss",
            "target_activity_labels": "opened only after fitting for accuracy and macro-F1",
            "pair_ids": "opened only after fitting for retrieval evaluation; never supplied to the trainer or solver",
        },
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
        "data": {
            "subject": prepared.subject,
            "source_view": prepared.source_view,
            "target_view": prepared.target_view,
            "samples": int(len(source)),
            "input_dimension": int(source.shape[1]),
            "window_samples": prepared.window_samples,
            "activity_counts": {ACTIVITY_NAMES[label]: int(np.sum(prepared.source_labels == label)) for label in ACTIVITY_NAMES},
        },
        "evaluation": {
            "source_only": source_only,
            "fixed_soft_gcot": fixed_metrics,
            "learned_source_supervised_soft_gcot": learned_metrics,
            "fixed_paired_retrieval": _paired_recall(fixed_solution.aligned_source, target),
            "learned_paired_retrieval": learned_retrieval,
        },
        "fixed_solver": {
            "admm_converged": bool(fixed_solution.diagnostics["admm_converged"]),
            "final_primal_residual": float(fixed_solution.diagnostics["admm_primal_residual"][-1]),
        },
        "learned_solver": {
            "admm_converged": bool(learned.solution.diagnostics["admm_converged"]),
            "final_primal_residual": float(learned.solution.diagnostics["admm_primal_residual"][-1]),
        },
        "outer_history": learned.history,
    }
    output = RESULTS_DIR / f"{args.tag}.json"
    write_json(output, result)
    print(
        "target accuracy "
        f"source_only={source_only['target_accuracy']:.4f} "
        f"fixed={fixed_metrics['target_accuracy']:.4f} "
        f"learned={learned_metrics['target_accuracy']:.4f}"
    )
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
