"""Summarize paired source-only versus differentiable task-aware OT results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT / "src", ROOT / "src" / "cc_hiwa"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import RESULTS_DIR, ensure_output_dirs, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[501, 502, 503])
    parser.add_argument("--tag", default="pamap2_differentiable_task_aware_ot_summary_seeds501_503_20260713")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    rows: list[dict[str, float | int]] = []
    for seed in args.seeds:
        task = _read_json(RESULTS_DIR / f"pamap2_task_only_mlp_subject101_seed{seed}_20260713.json")
        ot = _read_json(RESULTS_DIR / f"pamap2_differentiable_task_aware_ot_normalized_subject101_seed{seed}_20260713.json")
        task_accuracy = float(task["evaluation"]["target_test_accuracy"])
        ot_accuracy = float(ot["evaluation"]["target_test_accuracy"])
        rows.append({
            "seed": int(seed),
            "task_only_target_accuracy": task_accuracy,
            "task_aware_ot_target_accuracy": ot_accuracy,
            "paired_delta": ot_accuracy - task_accuracy,
            "task_only_source_validation_accuracy": float(task["evaluation"]["best_source_validation_accuracy"]),
            "task_aware_ot_source_validation_accuracy": float(ot["evaluation"]["best_source_validation_accuracy"]),
        })
    deltas = np.asarray([row["paired_delta"] for row in rows], dtype=float)
    payload = {
        "experiment": "pamap2_differentiable_task_aware_ot_paired_summary",
        "scope": "three frozen-confirmation seeds under one temporal split; seed 501 is development and seeds 502-503 are confirmation",
        "rows": rows,
        "summary": {
            "task_only_target_accuracy_mean": float(np.mean([row["task_only_target_accuracy"] for row in rows])),
            "task_aware_ot_target_accuracy_mean": float(np.mean([row["task_aware_ot_target_accuracy"] for row in rows])),
            "paired_delta_mean": float(np.mean(deltas)),
            "paired_delta_median": float(np.median(deltas)),
            "task_aware_ot_better_seed_count": int(np.sum(deltas > 0)),
            "seed_count": len(rows),
        },
    }
    output = RESULTS_DIR / f"{args.tag}.json"
    write_json(output, payload)
    print(f"paired delta mean={payload['summary']['paired_delta_mean']:.4f}; saved: {output}")


if __name__ == "__main__":
    main()
