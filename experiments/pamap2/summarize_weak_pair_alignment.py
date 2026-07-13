"""Summarize the frozen paired comparison for weakly supervised PAMAP2 alignment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (ROOT / "src", ROOT / "src" / "cc_hiwa"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import RESULTS_DIR, write_json


def _accuracy(path: Path) -> float:
    with path.open(encoding="utf-8") as handle:
        return float(json.load(handle)["evaluation"]["target_test_accuracy"])


def main() -> None:
    seeds = (601, 602, 603)
    rows = []
    for seed in seeds:
        task_only = _accuracy(RESULTS_DIR / f"pamap2_task_only_mlp_revised_subject101_seed{seed}_20260713.json")
        matched_no_pair = _accuracy(RESULTS_DIR / f"pamap2_no_pair_no_ot_subject101_seed{seed}_20260713.json")
        weak_pair = _accuracy(RESULTS_DIR / f"pamap2_weak_pair_task_only_revised_subject101_seed{seed}_20260713.json")
        rows.append({
            "seed": seed,
            "task_only_target_accuracy": task_only,
            "matched_no_pair_no_ot_target_accuracy": matched_no_pair,
            "weak_pair_no_ot_target_accuracy": weak_pair,
            "weak_pair_vs_task_only_delta": weak_pair - task_only,
            "weak_pair_vs_matched_no_pair_delta": weak_pair - matched_no_pair,
        })
    task_deltas = np.asarray([row["weak_pair_vs_task_only_delta"] for row in rows])
    matched_deltas = np.asarray([row["weak_pair_vs_matched_no_pair_delta"] for row in rows])
    payload = {
        "experiment": "pamap2_weak_pair_alignment_paired_summary",
        "scope": "one development seed and two confirmation seeds under the frozen temporal split; target-test labels were opened only by the individual completed evaluations",
        "rows": rows,
        "summary": {
            "task_only_target_accuracy_mean": float(np.mean([row["task_only_target_accuracy"] for row in rows])),
            "matched_no_pair_no_ot_target_accuracy_mean": float(np.mean([row["matched_no_pair_no_ot_target_accuracy"] for row in rows])),
            "weak_pair_no_ot_target_accuracy_mean": float(np.mean([row["weak_pair_no_ot_target_accuracy"] for row in rows])),
            "weak_pair_vs_task_only_delta_mean": float(np.mean(task_deltas)),
            "weak_pair_vs_matched_no_pair_delta_mean": float(np.mean(matched_deltas)),
            "weak_pair_vs_matched_no_pair_delta_median": float(np.median(matched_deltas)),
            "weak_pair_better_than_matched_no_pair_seed_count": int(np.sum(matched_deltas > 0.0)),
            "seed_count": len(rows),
        },
        "ot_diagnostic": {
            "seed": 601,
            "weak_pair_plus_task_aware_ot_target_accuracy": _accuracy(RESULTS_DIR / "pamap2_weak_pair_task_aware_ot_revised_subject101_seed601_20260713.json"),
            "interpretation": "single diagnostic only; do not treat as a confirmation result",
        },
    }
    output = RESULTS_DIR / "pamap2_weak_pair_alignment_summary_seeds601_603_20260713.json"
    write_json(output, payload)
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
