"""Verify the frozen evidence chain for the GC-HiWA v2 paper claim.

This is an artifact-integrity check, not a new model fit and not a statistical
guarantee.  It ensures that the exact result files cited by the current paper
contract still support the deliberately narrow claim: recovery in controlled
shared-geometry settings and abstention on the declared violations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "results"

SPECTRAL_AUDIT = "gc_hiwa_boundary_benchmark_gc_hiwa_spectral_gate_independent_holdout_1961_1965_qualification_audit.json"
ISOSPECTRAL_RESULT = "gc_hiwa_boundary_benchmark_gc_hiwa_isospectral_nonorthogonal_independent_2201_2205.json"
ISOSPECTRAL_AUDIT = "gc_hiwa_boundary_benchmark_gc_hiwa_isospectral_nonorthogonal_independent_2201_2205_qualification_audit.json"
PANOPTIC_RESULT = "panoptic_identifiability_gate_smoke_1601.json"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AssertionError(f"required evidence file is missing: {path}")
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise AssertionError(f"expected a JSON object: {path}")
    return value


def _require_equal(label: str, observed: object, expected: object) -> None:
    if observed != expected:
        raise AssertionError(f"{label}: expected {expected!r}, observed {observed!r}")


def audit(results_directory: Path = RESULTS) -> dict[str, Any]:
    """Return a machine-readable verification of the frozen evidence files."""
    spectral = _read_json(results_directory / SPECTRAL_AUDIT)["aggregate"]
    _require_equal("independent recoverable runs", spectral["recoverable_runs"], 15)
    _require_equal("independent core recovery accepts", spectral["recoverable_core_accepted"], 15)
    _require_equal("independent ROCA recovery accepts", spectral["recoverable_roca_accepted"], 15)
    _require_equal("independent nonrecoverable runs", spectral["nonrecoverable_runs"], 25)
    _require_equal("independent core abstentions", spectral["nonrecoverable_core_abstained"], 25)
    _require_equal("independent ROCA abstentions", spectral["nonrecoverable_roca_abstained"], 25)

    isospectral_audit = _read_json(results_directory / ISOSPECTRAL_AUDIT)["aggregate"]
    _require_equal("isospectral nonrecoverable runs", isospectral_audit["nonrecoverable_runs"], 5)
    _require_equal("isospectral core abstentions", isospectral_audit["nonrecoverable_core_abstained"], 5)
    _require_equal("isospectral ROCA abstentions", isospectral_audit["nonrecoverable_roca_abstained"], 5)

    isospectral_records = _read_json(results_directory / ISOSPECTRAL_RESULT)["records"]
    core_records = [row for row in isospectral_records if row.get("method") == "soft_transport_oracle_component"]
    roca_records = [row for row in isospectral_records if row.get("method") == "soft_roca_abstained"]
    _require_equal("isospectral core record count", len(core_records), 5)
    _require_equal("isospectral ROCA record count", len(roca_records), 5)
    for row in core_records:
        qualification = row["applicability_qualification"]
        _require_equal("isospectral core decision", qualification["accepted"], False)
        if float(row["covariance_spectral_mismatch"]) > 1e-10:
            raise AssertionError("isospectral counterexample no longer has matching covariance spectra")
        if row["single_rotation_truth_exists"]:
            raise AssertionError("nonorthogonal counterexample must not expose a rotation truth")
        if "single_rotation_fit_excessive" not in qualification["reasons"]:
            raise AssertionError("isospectral counterexample must fail the global-fit diagnostic")
    for row in roca_records:
        _require_equal("isospectral ROCA decision", row["roca_qualification"]["decision"], "no_qualified_candidate")

    panoptic = _read_json(results_directory / PANOPTIC_RESULT)
    records = panoptic["records"]
    _require_equal("Panoptic qualified record count", len(records), 1)
    panoptic_record = records[0]
    _require_equal("Panoptic numerical qualification", panoptic_record["applicability_qualification"]["accepted"], True)
    _require_equal(
        "Panoptic rotation structure",
        panoptic_record["applicability_qualification"]["observed_rotation_structure"],
        "repeated_3d_blocks",
    )
    if float(panoptic_record["rotation_frobenius_error"]) >= 0.05:
        raise AssertionError("Panoptic controlled recovery error exceeds the documented bound")
    if float(panoptic_record["paired_alignment_mse"]) >= 1e-5:
        raise AssertionError("Panoptic controlled paired alignment error exceeds the documented bound")
    _require_equal("Panoptic paired Recall@1", panoptic_record["paired_recall_at_1"], 1.0)
    if float(panoptic_record["covariance_identifiability_margin"]) < 0.02:
        raise AssertionError("Panoptic physical geometry must pass the declared identifiability gate")

    return {
        "verification": "gc_hiwa_v2_frozen_evidence_integrity",
        "scope": (
            "checks frozen result artifacts only; it does not establish universal transfer, "
            "a distribution-wide error rate, or real semantic-action generalization"
        ),
        "synthetic_independent_holdout": {
            "recoverable_core_accepted": spectral["recoverable_core_accepted"],
            "recoverable_roca_accepted": spectral["recoverable_roca_accepted"],
            "declared_violations_core_abstained": spectral["nonrecoverable_core_abstained"],
            "declared_violations_roca_abstained": spectral["nonrecoverable_roca_abstained"],
        },
        "isospectral_nonorthogonal_counterexample": {
            "runs": len(core_records),
            "core_abstained": isospectral_audit["nonrecoverable_core_abstained"],
            "roca_abstained": isospectral_audit["nonrecoverable_roca_abstained"],
            "maximum_covariance_spectral_mismatch": max(
                float(row["covariance_spectral_mismatch"]) for row in core_records
            ),
        },
        "panoptic_controlled_recovery": {
            "accepted": panoptic_record["applicability_qualification"]["accepted"],
            "rotation_frobenius_error": panoptic_record["rotation_frobenius_error"],
            "paired_alignment_mse": panoptic_record["paired_alignment_mse"],
            "paired_recall_at_1": panoptic_record["paired_recall_at_1"],
            "covariance_identifiability_margin": panoptic_record["covariance_identifiability_margin"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-directory", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = audit(args.results_directory)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
