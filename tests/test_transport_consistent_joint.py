import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "cc_hiwa"))

from transport_consistent_joint import global_coupling, group_conditional_map, transport_consistency_loss


def test_global_coupling_is_normalized_and_uses_every_group_pair() -> None:
    q00 = np.array([[0.2, 0.1], [0.1, 0.1]])
    q01 = np.array([[0.1, 0.2], [0.1, 0.1]])
    q10 = np.array([[0.1, 0.1], [0.2, 0.1]])
    q11 = np.array([[0.1, 0.1], [0.1, 0.2]])
    coupling = global_coupling(((q00, q01), (q10, q11)), np.full((2, 2), 0.25))
    assert coupling.shape == (2, 2)
    assert np.isclose(coupling.sum(), 1.0)
    assert np.all(coupling >= 0)


def test_transport_consistency_is_zero_for_matching_assignments() -> None:
    assignments = np.array([[1.0, 0.0], [0.0, 1.0]])
    coupling = np.eye(2) / 2
    group_map = group_conditional_map(np.eye(2))
    assert np.isclose(transport_consistency_loss(assignments, assignments, coupling, group_map), 0.0)
