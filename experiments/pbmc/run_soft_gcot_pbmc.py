"""Minimal paired-cell PBMC baseline for Hard HiWA and fixed Soft-GCOT HiWA.

The supplied NPZ contains paired RNA and ATAC embeddings in the same cell-ID
order. Cell type is never used to fit either method; it is optional final
evaluation only. Full-support OT is deliberately capped to a small,
deterministic paired subset because the current NumPy implementation scales as
the square of the number of cells for each group pair.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "src" / "cc_hiwa", ROOT / "src" / "hiwa"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from hiwa import HiWA  # noqa: E402
from soft_groups import assignments_from_prototypes, learn_soft_groups  # noqa: E402
from soft_hiwa import SoftHiWA  # noqa: E402


PROFILE = dict(
    maxiter=80,
    tol=1e-1,
    mu=5e-3,
    shorn_maxiter=300,
    sa_maxiter=40,
    sa_shorn_maxiter=80,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[100, 101, 102])
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--temperature-path", nargs="+", type=float, default=[0.25, 0.35, 0.50])
    parser.add_argument("--entropy-weight", type=float, default=0.05)
    parser.add_argument("--max-cells", type=int, default=96)
    parser.add_argument("--tag", default="pbmc_paired_96_seeds_100_102")
    return parser.parse_args()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _standardize(values: np.ndarray) -> np.ndarray:
    return (values - values.mean(axis=0, keepdims=True)) / np.maximum(values.std(axis=0, keepdims=True), 1e-12)


def _target_transform(values: np.ndarray) -> np.ndarray:
    low_dimensional = PCA(n_components=2, random_state=0).fit_transform(values)
    return np.linalg.pinv(values) @ low_dimensional


def _scores(aligned: np.ndarray, target: np.ndarray) -> np.ndarray:
    aligned = aligned / np.maximum(np.linalg.norm(aligned, axis=1, keepdims=True), 1e-12)
    target = target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12)
    return aligned @ target.T


def _evaluate(aligned: np.ndarray, target: np.ndarray, labels: np.ndarray | None) -> dict[str, float | None]:
    score = _scores(aligned, target)
    n_cells = score.shape[0]
    order = np.argsort(-score, axis=1)
    output: dict[str, float | None] = {
        "paired_recall_at_1": float(np.mean(order[:, 0] == np.arange(n_cells))),
        "paired_recall_at_5": float(np.mean(np.any(order[:, : min(5, n_cells)] == np.arange(n_cells)[:, None], axis=1))),
        "cell_type_transfer_accuracy": None,
    }
    if labels is not None:
        output["cell_type_transfer_accuracy"] = float(np.mean(labels[order[:, 0]] == labels))
    return output


def _fit_hard(source: np.ndarray, target: np.ndarray, source_labels: np.ndarray, target_labels: np.ndarray, seed: int) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    model = HiWA(dim_red_method=PCA(n_components=2, random_state=0), normalize=True, shorn_gamma=0.2, sa_tol=1e-2, sa_shorn_gamma=0.1, **PROFILE)
    started = time.perf_counter()
    aligned = model.fit_transform(source, source_labels, target, target_labels, Y_transform=_target_transform(target), Rgt=np.eye(source.shape[1]))
    return {
        "iterations": int(len(model.diagnostics["Rg_norm"])),
        "converged": bool(model.diagnostics["Rg_norm"][-1] <= PROFILE["tol"] and model.diagnostics["admm_primal_residual"][-1] <= PROFILE["tol"]),
        "elapsed_seconds": time.perf_counter() - started,
        "global_residual": float(model.diagnostics["Rg_norm"][-1]),
        "primal_residual": float(model.diagnostics["admm_primal_residual"][-1]),
    }, aligned, model.Rg, model.P


def _fit_soft_chain(source: np.ndarray, target: np.ndarray, source_stages: list[np.ndarray], target_stages: list[np.ndarray], hard_rotation: np.ndarray, hard_transport: np.ndarray, seed: int) -> tuple[dict, np.ndarray]:
    rotation, transport, final_model, elapsed = hard_rotation, hard_transport, None, 0.0
    for source_assignment, target_assignment in zip(source_stages, target_stages):
        model = SoftHiWA(dim_red_method=PCA(n_components=2, random_state=0), normalize=True, shorn_gamma=0.2, sa_tol=1e-2, sa_shorn_gamma=0.1, support_mode="full", random_state=seed, warm_start_local=True, **PROFILE)
        started = time.perf_counter()
        aligned = model.fit_transform(source, source_assignment, target, target_assignment, Y_transform=_target_transform(target), Rgt=np.eye(source.shape[1]), initial_rotation=rotation, initial_transport=transport)
        elapsed += time.perf_counter() - started
        rotation, transport, final_model = model.Rg, model.P, model
    assert final_model is not None
    return {
        "iterations": int(len(final_model.diagnostics["Rg_norm"])),
        "converged": bool(final_model.diagnostics["admm_converged"]),
        "elapsed_seconds": elapsed,
        "global_residual": float(final_model.diagnostics["Rg_norm"][-1]),
        "primal_residual": float(final_model.diagnostics["admm_primal_residual"][-1]),
        "support_mode": "full",
    }, aligned


def main() -> None:
    args = parse_args()
    if args.groups < 2 or args.max_cells < 2 or any(tau <= 0 for tau in args.temperature_path):
        raise SystemExit("groups, max-cells, and all temperatures must be positive")
    data = np.load(args.input, allow_pickle=True)
    source_all, target_all = np.asarray(data["rna_embedding"], dtype=float), np.asarray(data["atac_embedding"], dtype=float)
    if source_all.shape != target_all.shape:
        raise SystemExit("paired RNA and ATAC embeddings must have identical shapes")
    indices = np.linspace(0, source_all.shape[0] - 1, min(args.max_cells, source_all.shape[0]), dtype=int)
    source, target = source_all[indices], target_all[indices]
    labels = np.asarray(data["cell_type"])[indices] if "cell_type" in data.files else None
    records: list[dict[str, Any]] = []
    for seed in args.seeds:
        source_learned = learn_soft_groups(source, args.groups, args.temperature_path[0], args.entropy_weight, seed=seed)
        target_learned = learn_soft_groups(target, args.groups, args.temperature_path[0], args.entropy_weight, seed=seed)
        source_stages = [assignments_from_prototypes(_standardize(source), source_learned.prototypes_standardized, tau) for tau in args.temperature_path]
        target_stages = [assignments_from_prototypes(_standardize(target), target_learned.prototypes_standardized, tau) for tau in args.temperature_path]
        hard_info, hard_aligned, hard_rotation, hard_transport = _fit_hard(
            source,
            target,
            np.argmax(source_stages[-1], axis=1),
            np.argmax(target_stages[-1], axis=1),
            seed,
        )
        soft_info, soft_aligned = _fit_soft_chain(
            source, target, source_stages, target_stages, hard_rotation, hard_transport, seed
        )
        records.extend([
            {"method": "paired_hard_hiwa", "seed": seed, **hard_info, **_evaluate(hard_aligned, target, labels)},
            {"method": "fixed_soft_gcot", "seed": seed, **soft_info, **_evaluate(soft_aligned, target, labels)},
        ])
        print(f"seed={seed} hard_R1={records[-2]['paired_recall_at_1']:.3f} soft_R1={records[-1]['paired_recall_at_1']:.3f}", flush=True)
    output = ROOT / "experiments" / "results" / f"soft_gcot_pbmc_{args.tag}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_ready({"experiment": "paired_pbmc_fixed_soft_gcot", "input": str(args.input), "n_cells": int(len(indices)), "label_usage": "cell type is final evaluation only", "parameters": vars(args), "results": records}), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
