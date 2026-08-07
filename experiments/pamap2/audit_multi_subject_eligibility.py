"""Audit whether named PAMAP2 subjects satisfy the frozen temporal protocol.

This is a data-availability and protocol-integrity check, not a model run.  It
uses the pre-specified six activity IDs only to verify that each available
subject has enough finite, contiguous wrist/chest windows for the already
locked temporal split.  It performs no train/test metric evaluation.
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

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
    protocol_subject_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-directory",
        type=Path,
        default=ROOT / "data" / "raw" / "pamap2" / "PAMAP2_Dataset",
    )
    parser.add_argument("--subjects", nargs="+", type=int, default=list(range(101, 110)))
    parser.add_argument("--window-samples", type=int, default=200)
    parser.add_argument("--source-train-windows", type=int, default=8)
    parser.add_argument("--source-validation-windows", type=int, default=2)
    parser.add_argument("--target-adaptation-windows", type=int, default=3)
    parser.add_argument("--target-test-windows", type=int, default=3)
    parser.add_argument("--tag", default="pamap2_multi_subject_temporal_eligibility")
    return parser.parse_args()


def assess_subject(
    raw_directory: Path,
    subject: int,
    *,
    window_samples: int,
    counts: PAMAP2SplitCounts,
) -> dict:
    """Return an auditable eligibility record without fitting a model."""
    path = protocol_subject_path(raw_directory, subject)
    if not path.exists():
        return {
            "subject": int(subject),
            "protocol_file": str(path),
            "available": False,
            "eligible": False,
            "reason": "protocol_file_missing",
        }
    try:
        rows = load_protocol_subject(raw_directory, subject)
        split = build_temporal_domain_adaptation_split(
            rows,
            subject=subject,
            activities=tuple(ACTIVITY_NAMES),
            window_samples=window_samples,
            counts=counts,
        )
    except (OSError, ValueError) as exc:
        return {
            "subject": int(subject),
            "protocol_file": str(path),
            "available": True,
            "eligible": False,
            "reason": str(exc),
        }
    return {
        "subject": int(subject),
        "protocol_file": str(path),
        "available": True,
        "eligible": True,
        "reason": None,
        "row_count": int(rows.shape[0]),
        "samples": {
            "source_train": int(len(split.source_train_features)),
            "source_validation": int(len(split.source_validation_features)),
            "target_adaptation": int(len(split.target_adaptation_features)),
            "target_test": int(len(split.evaluation_target_features)),
        },
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
    records = [
        assess_subject(
            args.raw_directory,
            subject,
            window_samples=args.window_samples,
            counts=counts,
        )
        for subject in args.subjects
    ]
    eligible = [row["subject"] for row in records if row["eligible"]]
    payload = {
        "experiment": "pamap2_multi_subject_temporal_eligibility_audit",
        "scope": "fixed-protocol data eligibility only; no model fitting or target-test label evaluation",
        "parameters": {
            "raw_directory": str(args.raw_directory),
            "subjects": args.subjects,
            "activities": list(ACTIVITY_NAMES),
            "window_samples": args.window_samples,
            "counts_per_activity": {
                "source_train": counts.source_train,
                "source_validation": counts.source_validation,
                "target_adaptation": counts.target_adaptation,
                "target_test": counts.target_test,
            },
        },
        "environment": {"python": platform.python_version()},
        "records": records,
        "eligible_subjects": eligible,
        "eligible_count": len(eligible),
    }
    ensure_output_dirs()
    output = RESULTS_DIR / f"{args.tag}.json"
    write_json(output, payload)
    print(f"eligible={eligible}; saved: {output}")


if __name__ == "__main__":
    main()
