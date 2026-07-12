"""Compare unaligned, Hard HiWA, and fixed Soft-GCOT on Paderborn data."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "src" / "cc_hiwa", ROOT / "src" / "hiwa"):
    sys.path.insert(0, str(directory))

from hiwa import HiWA  # noqa: E402
from soft_groups import assignments_from_prototypes, learn_soft_groups  # noqa: E402
from soft_hiwa import SoftHiWA  # noqa: E402

PROFILE = dict(maxiter=80, tol=1e-1, mu=5e-3, shorn_maxiter=300, sa_maxiter=40, sa_shorn_maxiter=80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=130)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--temperature-path", nargs="+", type=float, default=[0.25, 0.35, 0.50])
    parser.add_argument("--entropy-weight", type=float, default=0.05)
    parser.add_argument("--tag", default="paderborn_speed_shift_seed130")
    return parser.parse_args()


def _standardize(values: np.ndarray) -> np.ndarray:
    return (values - values.mean(axis=0, keepdims=True)) / np.maximum(values.std(axis=0, keepdims=True), 1e-12)


def _target_transform(values: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(values) @ PCA(n_components=2, random_state=0).fit_transform(values)


def _metrics(source: np.ndarray, source_labels: np.ndarray, target: np.ndarray, target_labels: np.ndarray) -> dict[str, Any]:
    classifier = LogisticRegression(max_iter=1_000, random_state=0).fit(source, source_labels)
    prediction = classifier.predict(target)
    return {
        "target_accuracy": float(accuracy_score(target_labels, prediction)),
        "target_macro_f1": float(f1_score(target_labels, prediction, average="macro")),
        "target_recall_by_class": recall_score(target_labels, prediction, average=None, labels=[0, 1, 2]).tolist(),
    }


def _hard(source: np.ndarray, target: np.ndarray, source_assignment: np.ndarray, target_assignment: np.ndarray, seed: int):
    model = HiWA(dim_red_method=PCA(n_components=2, random_state=0), normalize=True, shorn_gamma=0.2, sa_tol=1e-2, sa_shorn_gamma=0.1, **PROFILE)
    started = time.perf_counter()
    aligned = model.fit_transform(source, np.argmax(source_assignment, axis=1), target, np.argmax(target_assignment, axis=1), Y_transform=_target_transform(target), Rgt=np.eye(source.shape[1]))
    info = {
        "iterations": len(model.diagnostics["Rg_norm"]),
        "converged": bool(model.diagnostics["Rg_norm"][-1] <= PROFILE["tol"] and model.diagnostics["admm_primal_residual"][-1] <= PROFILE["tol"]),
        "elapsed_seconds": time.perf_counter() - started,
        "global_residual": float(model.diagnostics["Rg_norm"][-1]),
        "primal_residual": float(model.diagnostics["admm_primal_residual"][-1]),
    }
    return aligned, model.Rg, model.P, info


def _soft(source: np.ndarray, target: np.ndarray, source_stages: list[np.ndarray], target_stages: list[np.ndarray], rotation: np.ndarray, transport: np.ndarray, seed: int):
    elapsed = 0.0
    for source_assignment, target_assignment in zip(source_stages, target_stages):
        model = SoftHiWA(dim_red_method=PCA(n_components=2, random_state=0), normalize=True, shorn_gamma=0.2, sa_tol=1e-2, sa_shorn_gamma=0.1, support_mode="full", random_state=seed, warm_start_local=True, **PROFILE)
        started = time.perf_counter()
        aligned = model.fit_transform(source, source_assignment, target, target_assignment, Y_transform=_target_transform(target), Rgt=np.eye(source.shape[1]), initial_rotation=rotation, initial_transport=transport)
        elapsed += time.perf_counter() - started
        rotation, transport = model.Rg, model.P
    info = {
        "iterations": len(model.diagnostics["Rg_norm"]),
        "converged": bool(model.diagnostics["admm_converged"]),
        "elapsed_seconds": elapsed,
        "global_residual": float(model.diagnostics["Rg_norm"][-1]),
        "primal_residual": float(model.diagnostics["admm_primal_residual"][-1]),
        "support_mode": "full",
    }
    return aligned, info


def main() -> None:
    args = parse_args()
    data = np.load(args.input, allow_pickle=True)
    source, target = np.asarray(data["source_features"], dtype=float), np.asarray(data["target_features"], dtype=float)
    source_labels, target_labels = np.asarray(data["source_labels"]), np.asarray(data["target_labels"])
    source_learned = learn_soft_groups(source, args.groups, args.temperature_path[0], args.entropy_weight, seed=args.seed)
    target_learned = learn_soft_groups(target, args.groups, args.temperature_path[0], args.entropy_weight, seed=args.seed)
    source_stages = [assignments_from_prototypes(_standardize(source), source_learned.prototypes_standardized, tau) for tau in args.temperature_path]
    target_stages = [assignments_from_prototypes(_standardize(target), target_learned.prototypes_standardized, tau) for tau in args.temperature_path]
    hard_aligned, rotation, transport, hard_info = _hard(source, target, source_stages[-1], target_stages[-1], args.seed)
    soft_aligned, soft_info = _soft(source, target, source_stages, target_stages, rotation, transport, args.seed)
    records = [
        {"method": "unaligned", **_metrics(source, source_labels, target, target_labels)},
        {"method": "paired_hard_hiwa", **hard_info, **_metrics(hard_aligned, source_labels, target, target_labels)},
        {"method": "fixed_soft_gcot", **soft_info, **_metrics(soft_aligned, source_labels, target, target_labels)},
    ]
    output = ROOT / "experiments" / "results" / f"paderborn_soft_gcot_{args.tag}.json"
    parameters = {**vars(args), "input": str(args.input)}
    output.write_text(json.dumps({"experiment": "paderborn_cross_condition_fault_transfer", "input": str(args.input), "label_usage": "labels are used only for final source classifier training and target evaluation", "parameters": parameters, "results": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    for record in records:
        print(f"{record['method']}: accuracy={record['target_accuracy']:.3f}, macro_f1={record['target_macro_f1']:.3f}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
