from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "src" / "cc_hiwa"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from roca_qualification import RocaCandidateEvidence, qualify_roca_candidates  # noqa: E402


def _tetrahedron() -> np.ndarray:
    return np.asarray([[-1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, 1.0], [1.0, 1.0, -1.0]])


def test_qualification_selects_unique_converged_orientation_consistent_candidate() -> None:
    source = _tetrahedron()
    target = source.copy()
    assignments = np.eye(4)
    evidence = [
        RocaCandidateEvidence(-1, np.eye(4), False, 1e-6, 0.2, 1.0),
        RocaCandidateEvidence(1, np.eye(4), True, 1e-6, 0.1, 1.0),
    ]
    result = qualify_roca_candidates(source, assignments, target, assignments, evidence)
    assert result["selected_sign"] == 1
    assert result["decision"] == "unique_qualified_candidate"


def test_qualification_abstains_when_both_candidates_are_qualified() -> None:
    source = _tetrahedron()
    target = source.copy()
    assignments = np.eye(4)
    # Swapping two target groups flips the oriented simplex sign for the -1 branch.
    reflected_transport = np.eye(4)[:, [1, 0, 2, 3]]
    evidence = [
        RocaCandidateEvidence(-1, reflected_transport, True, 1e-6, 0.1, 1.0),
        RocaCandidateEvidence(1, np.eye(4), True, 1e-6, 0.1, 1.0),
    ]
    result = qualify_roca_candidates(source, assignments, target, assignments, evidence)
    assert result["selected_sign"] is None
    assert result["decision"] == "ambiguous_multiple_qualified_candidates"


def test_qualification_uses_internal_objective_to_break_a_healthy_tie() -> None:
    source = _tetrahedron()
    target = source.copy()
    assignments = np.eye(4)
    reflected_transport = np.eye(4)[:, [1, 0, 2, 3]]
    evidence = [
        RocaCandidateEvidence(-1, reflected_transport, True, 1e-6, 0.25, 1.0),
        RocaCandidateEvidence(1, np.eye(4), True, 1e-6, 0.10, 1.0),
    ]
    result = qualify_roca_candidates(source, assignments, target, assignments, evidence)
    assert result["selected_sign"] == 1
    assert result["decision"] == "objective_separated_candidates"
    assert np.isclose(result["relative_objective_gap"], 1.5)


def test_qualification_abstains_when_each_candidate_fits_poorly_despite_convergence() -> None:
    source = _tetrahedron()
    assignments = np.eye(4)
    evidence = [
        RocaCandidateEvidence(-1, np.eye(4), True, 1e-6, 0.1, 3.0),
        RocaCandidateEvidence(1, np.eye(4), True, 1e-6, 0.05, 2.0),
    ]
    result = qualify_roca_candidates(source, assignments, source, assignments, evidence)
    assert result["selected_sign"] is None
    assert result["decision"] == "no_qualified_candidate"


def test_qualification_abstains_when_global_covariance_invariant_fails() -> None:
    source = _tetrahedron()
    assignments = np.eye(4)
    evidence = [
        RocaCandidateEvidence(-1, np.eye(4), False, 1e-6, 0.2, 1.0),
        RocaCandidateEvidence(1, np.eye(4), True, 1e-6, 0.1, 1.0),
    ]
    result = qualify_roca_candidates(
        source,
        assignments,
        source,
        assignments,
        evidence,
        covariance_spectrum_compatible=False,
        covariance_spectral_mismatch=0.08,
    )
    assert result["selected_sign"] is None
    assert result["decision"] == "covariance_spectrum_incompatible"
    assert not result["covariance_spectrum_compatible"]
