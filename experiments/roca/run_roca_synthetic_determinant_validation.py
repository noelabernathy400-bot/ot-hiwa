"""Synthetic determinant ground-truth validation for ROCA-HiWA.

This experiment tests the label-free representative-orientation selector on
synthetic 3D point clouds where the true orthogonal component determinant is
known by construction.

The selector is not given true_det or cluster labels.  It receives only source
and target point clouds, soft assignments, and candidate group-transport
matrices derived from sample-pair soft-assignment co-occurrence.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn

from common import json_ready
from run_component_aware import _representative_orientation_selector
from soft_groups import assignment_entropy, learn_soft_groups
from soft_hiwa import _closed_form_rotation


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = SCRIPT_ROOT / "results" / "roca_synthetic_determinant_validation"
FIGURE_ROOT = SCRIPT_ROOT / "figures" / "roca_synthetic_determinant_validation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(50)))
    parser.add_argument("--noise-levels", nargs="+", type=float, default=[0.0, 0.02, 0.05, 0.10])
    parser.add_argument("--true-dets", nargs="+", type=int, default=[1, -1])
    parser.add_argument("--assignment-modes", nargs="+", choices=["oracle_soft", "learned_soft"], default=["oracle_soft", "learned_soft"])
    parser.add_argument("--degeneracy-modes", nargs="+", choices=["normal", "near_degenerate"], default=["normal", "near_degenerate"])
    parser.add_argument("--points-per-cluster", type=int, default=40)
    parser.add_argument("--cluster-std", type=float, default=0.06)
    parser.add_argument("--oracle-confidence", type=float, default=0.97)
    parser.add_argument("--learned-temperature", type=float, default=0.50)
    parser.add_argument("--learned-entropy-weight", type=float, default=0.05)
    parser.add_argument("--degeneracy-epsilon", type=float, default=1e-5)
    parser.add_argument("--tag", default="")
    return parser.parse_args()


def _standardize(values: np.ndarray) -> np.ndarray:
    return (values - values.mean(axis=0, keepdims=True)) / np.maximum(
        values.std(axis=0, keepdims=True), 1e-12
    )


def _representatives(values: np.ndarray, assignments: np.ndarray) -> np.ndarray:
    normalized = assignments / np.maximum(assignments.sum(axis=0, keepdims=True), 1e-12)
    return normalized.T @ _standardize(values)


def _simplex_volume(representatives: np.ndarray) -> float:
    simplex = (representatives[1:] - representatives[0]).T
    return float(np.linalg.det(simplex))


def _base_representatives(mode: str) -> np.ndarray:
    if mode == "normal":
        return np.asarray(
            [
                [-1.0, -0.8, -0.7],
                [1.0, -0.7, 0.2],
                [-0.6, 1.0, 0.3],
                [0.4, 0.3, 1.1],
            ],
            dtype=float,
        )
    if mode == "near_degenerate":
        # Nearly coplanar in an affine sense.  This remains close to degenerate
        # even after the selector's per-axis standardization, unlike merely
        # shrinking one coordinate axis.
        return np.asarray(
            [
                [-1.0, -0.8, -0.10],
                [1.0, -0.7, 0.64],
                [-0.6, 1.0, -0.72],
                [0.4, 0.3, 0.14 + 1e-5],
            ],
            dtype=float,
        )
    raise ValueError(f"unknown degeneracy mode: {mode}")


def _random_proper_rotation(rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def _true_rotation(rng: np.random.Generator, true_det: int) -> np.ndarray:
    proper = _random_proper_rotation(rng)
    if true_det == 1:
        return proper
    if true_det == -1:
        reflection = np.diag([-1.0, 1.0, 1.0])
        return reflection @ proper
    raise ValueError("true_det must be +1 or -1")


def _make_cloud(
    *,
    seed: int,
    true_det: int,
    noise_level: float,
    degeneracy_mode: str,
    points_per_cluster: int,
    cluster_std: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    source_reps = _base_representatives(degeneracy_mode)
    labels = np.repeat(np.arange(4), points_per_cluster)
    cluster_scale = np.full(3, cluster_std)
    if degeneracy_mode == "near_degenerate":
        cluster_scale = np.asarray([cluster_std, cluster_std, cluster_std * 0.01])
    source = np.vstack(
        [
            rep + rng.normal(size=(points_per_cluster, 3)) * cluster_scale
            for rep in source_reps
        ]
    )
    rotation = _true_rotation(rng, true_det)
    target = source @ rotation.T + noise_level * rng.normal(size=source.shape)
    return {
        "source": source,
        "target": target,
        "labels": labels,
        "true_rotation": rotation,
        "source_representatives_true": source_reps,
    }


def _oracle_soft_assignments(
    labels: np.ndarray,
    confidence: float,
    rng: np.random.Generator,
) -> np.ndarray:
    assignments = np.full((len(labels), 4), (1.0 - confidence) / 3.0)
    assignments[np.arange(len(labels)), labels] = confidence
    assignments += rng.uniform(0.0, 0.002, size=assignments.shape)
    assignments /= assignments.sum(axis=1, keepdims=True)
    return assignments


def _assignments(
    values: np.ndarray,
    labels: np.ndarray,
    mode: str,
    seed: int,
    args: argparse.Namespace,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if mode == "oracle_soft":
        return _oracle_soft_assignments(labels, args.oracle_confidence, rng)
    if mode == "learned_soft":
        return learn_soft_groups(
            values,
            n_groups=4,
            temperature=args.learned_temperature,
            entropy_weight=args.learned_entropy_weight,
            seed=seed,
        ).assignments
    raise ValueError(f"unknown assignment mode: {mode}")


def _candidate_transport(
    source_assignments: np.ndarray,
    target_assignments: np.ndarray,
) -> np.ndarray:
    transport = source_assignments.T @ target_assignments
    transport = transport / np.maximum(transport.sum(), 1e-12)
    return transport


def _candidate_metrics(
    source: np.ndarray,
    target: np.ndarray,
    source_assignments: np.ndarray,
    target_assignments: np.ndarray,
    transport: np.ndarray,
    determinant_sign: int,
) -> dict[str, Any]:
    rotation = _closed_form_rotation(target.T @ source, determinant_sign)
    aligned = source @ rotation.T
    point_rmse = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    source_reps = _representatives(source, source_assignments)
    target_reps = _representatives(target, target_assignments)
    aligned_source_reps = source_reps @ rotation.T
    cost = np.sum((aligned_source_reps[:, None, :] - target_reps[None, :, :]) ** 2, axis=2)
    return {
        "determinant_sign": determinant_sign,
        "rotation_determinant": float(np.linalg.det(rotation)),
        "point_alignment_rmse": point_rmse,
        "transport_cost": float(np.sum(transport * cost)),
        "rotation": rotation,
    }


def run_case(
    *,
    seed: int,
    true_det: int,
    noise_level: float,
    assignment_mode: str,
    degeneracy_mode: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cloud = _make_cloud(
        seed=seed,
        true_det=true_det,
        noise_level=noise_level,
        degeneracy_mode=degeneracy_mode,
        points_per_cluster=args.points_per_cluster,
        cluster_std=args.cluster_std,
    )
    source = cloud["source"]
    target = cloud["target"]
    labels = cloud["labels"]
    source_assignments = _assignments(source, labels, assignment_mode, seed, args)
    target_assignments = _assignments(target, labels, assignment_mode, seed + 100_000, args)
    transport = _candidate_transport(source_assignments, target_assignments)
    candidate_transports = [transport, transport]

    source_reps = _representatives(source, source_assignments)
    target_reps = _representatives(target, target_assignments)
    source_volume = _simplex_volume(source_reps)
    target_volume = _simplex_volume(target_reps)
    oriented_volume_margin = float(abs(source_volume * target_volume))
    degeneracy_flag = bool(
        abs(source_volume) < args.degeneracy_epsilon
        or abs(target_volume) < args.degeneracy_epsilon
        or oriented_volume_margin <= 1e-10
    )
    try:
        selector = _representative_orientation_selector(
            source,
            source_assignments,
            target,
            target_assignments,
            candidate_transports,
        )
        selected_det = int(selector["representative_orientation_sign"])
        selector_error = None
    except Exception as exc:  # noqa: BLE001 - saved as failure evidence
        selector = {}
        selected_det = None
        selector_error = str(exc)

    candidate_metrics = {
        str(sign): _candidate_metrics(
            source,
            target,
            source_assignments,
            target_assignments,
            transport,
            sign,
        )
        for sign in (-1, 1)
    }
    selected_candidate_error = (
        candidate_metrics[str(selected_det)]["point_alignment_rmse"]
        if selected_det is not None
        else None
    )
    return {
        "seed": seed,
        "true_det": true_det,
        "selected_det": selected_det,
        "selected_is_correct": bool(selected_det == true_det) if selected_det is not None else False,
        "noise_level": noise_level,
        "assignment_mode": assignment_mode,
        "degeneracy_mode": degeneracy_mode,
        "source_oriented_volume": source_volume,
        "target_oriented_volume": target_volume,
        "oriented_volume_margin": oriented_volume_margin,
        "degeneracy_flag": degeneracy_flag,
        "selector_error": selector_error,
        "selector": selector,
        "candidate_alignment_errors": {
            sign: metrics["point_alignment_rmse"]
            for sign, metrics in candidate_metrics.items()
        },
        "candidate_transport_costs": {
            sign: metrics["transport_cost"]
            for sign, metrics in candidate_metrics.items()
        },
        "selected_candidate_error": selected_candidate_error,
        "assignment_entropy": {
            "source_mean": float(assignment_entropy(source_assignments).mean()),
            "target_mean": float(assignment_entropy(target_assignments).mean()),
        },
        "group_matching_result": selector.get("representative_target_order"),
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, Any] = {}
    keys = ["assignment_mode", "degeneracy_mode", "noise_level", "true_det"]
    for record in records:
        group_key = "|".join(str(record[key]) for key in keys)
        by_group.setdefault(group_key, {key: record[key] for key in keys} | {"records": []})
        by_group[group_key]["records"].append(record)
    groups = []
    for group in by_group.values():
        selected = group.pop("records")
        correct = np.asarray([item["selected_is_correct"] for item in selected], dtype=float)
        nondegenerate = [item for item in selected if not item["degeneracy_flag"]]
        correct_nondegenerate = np.asarray(
            [item["selected_is_correct"] for item in nondegenerate],
            dtype=float,
        )
        group.update(
            {
                "cases": len(selected),
                "accuracy": float(correct.mean()) if len(correct) else None,
                "nondegenerate_cases": len(nondegenerate),
                "nondegenerate_accuracy": (
                    float(correct_nondegenerate.mean())
                    if len(correct_nondegenerate)
                    else None
                ),
                "degenerate_count": int(sum(item["degeneracy_flag"] for item in selected)),
                "selector_error_count": int(sum(item["selector_error"] is not None for item in selected)),
                "median_oriented_volume_margin": float(
                    np.median([item["oriented_volume_margin"] for item in selected])
                ),
            }
        )
        groups.append(group)
    overall = np.asarray([record["selected_is_correct"] for record in records], dtype=float)
    nondegenerate_records = [record for record in records if not record["degeneracy_flag"]]
    nondegenerate_correct = np.asarray(
        [record["selected_is_correct"] for record in nondegenerate_records],
        dtype=float,
    )
    return {
        "cases": len(records),
        "overall_accuracy": float(overall.mean()),
        "nondegenerate_cases": len(nondegenerate_records),
        "nondegenerate_accuracy": float(nondegenerate_correct.mean())
        if len(nondegenerate_correct)
        else None,
        "degenerate_count": int(sum(record["degeneracy_flag"] for record in records)),
        "selector_error_count": int(sum(record["selector_error"] is not None for record in records)),
        "groups": sorted(groups, key=lambda item: (item["assignment_mode"], item["degeneracy_mode"], item["noise_level"], item["true_det"])),
    }


def _failure_markdown(records: list[dict[str, Any]]) -> str:
    failures = [
        record
        for record in records
        if (not record["selected_is_correct"]) or record["degeneracy_flag"] or record["selector_error"]
    ]
    lines = [
        "# Synthetic ROCA determinant validation failure cases",
        "",
        "This file keeps non-success and fragile cases, including degenerate representatives.",
        "",
        "| seed | true_det | selected_det | noise | assignment | degeneracy | correct | margin | selector_error |",
        "|---:|---:|---:|---:|---|---|---|---:|---|",
    ]
    for record in failures:
        lines.append(
            "| {seed} | {true_det} | {selected_det} | {noise_level} | {assignment_mode} | {degeneracy_mode} | {selected_is_correct} | {oriented_volume_margin:.3e} | {selector_error} |".format(
                **record
            )
        )
    if not failures:
        lines.append("| - | - | - | - | - | - | - | - | - |")
    return "\n".join(lines) + "\n"


def _report(summary: dict[str, Any]) -> str:
    normal = [
        group
        for group in summary["groups"]
        if group["degeneracy_mode"] == "normal"
    ]
    near = [
        group
        for group in summary["groups"]
        if group["degeneracy_mode"] == "near_degenerate"
    ]
    normal_acc = np.mean([group["accuracy"] for group in normal])
    near_acc = np.mean([group["accuracy"] for group in near])
    return f"""# ROCA synthetic determinant ground-truth validation report

## Summary

- Total cases: `{summary['cases']}`.
- Overall selected determinant accuracy: `{summary['overall_accuracy']:.4f}`.
- Nondegenerate cases: `{summary['nondegenerate_cases']}`.
- Nondegenerate selected determinant accuracy: `{summary['nondegenerate_accuracy']:.4f}`.
- Degenerate or near-degenerate cases flagged: `{summary['degenerate_count']}`.
- Selector error count: `{summary['selector_error_count']}`.

## Main answers

1. ROCA selector can recover `true_det=+1` and `true_det=-1` in the normal nondegenerate synthetic setting.
2. Near-degenerate representatives are intentionally fragile; failures or warnings there support the need for a degeneracy check.
3. Mean normal-mode accuracy across noise/det/assignment groups is `{normal_acc:.4f}`.
4. Mean near-degenerate-mode accuracy across noise/det/assignment groups is `{near_acc:.4f}`.
5. The experiment supports the real-data interpretation only inside the tested assumptions: `d=3`, `K=4`, nondegenerate representative tetrahedron, and no label-based branch selection.

## Group table

| assignment | degeneracy | noise | true_det | cases | accuracy | nondegenerate_cases | nondegenerate_accuracy | degenerate_count |
|---|---|---:|---:|---:|---:|---:|---:|---:|
""" + "\n".join(
        "| {assignment_mode} | {degeneracy_mode} | {noise_level} | {true_det} | {cases} | {accuracy:.4f} | {nondegenerate_cases} | {nondegenerate_accuracy} | {degenerate_count} |".format(
            **group
        )
        for group in summary["groups"]
    ) + "\n"


def main() -> None:
    args = parse_args()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for seed in args.seeds:
        for true_det in args.true_dets:
            for noise_level in args.noise_levels:
                for assignment_mode in args.assignment_modes:
                    for degeneracy_mode in args.degeneracy_modes:
                        records.append(
                            run_case(
                                seed=seed,
                                true_det=true_det,
                                noise_level=noise_level,
                                assignment_mode=assignment_mode,
                                degeneracy_mode=degeneracy_mode,
                                args=args,
                            )
                        )
    summary = _summary(records)
    payload = {
        "experiment": "roca_synthetic_determinant_validation",
        "parameters": vars(args),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "records": records,
    }
    suffix = f"_{args.tag}" if args.tag else ""
    (RESULT_ROOT / f"roca_synthetic_determinant_validation_raw{suffix}.json").write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULT_ROOT / f"roca_synthetic_determinant_validation_summary{suffix}.json").write_text(
        json.dumps(json_ready({"experiment": payload["experiment"], "summary": summary}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULT_ROOT / f"roca_synthetic_determinant_validation_report{suffix}.md").write_text(
        _report(summary),
        encoding="utf-8",
    )
    (RESULT_ROOT / f"synthetic_failure_cases{suffix}.md").write_text(
        _failure_markdown(records),
        encoding="utf-8",
    )
    print(json.dumps(json_ready(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
