from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "panoptic" / "run_source_pose_state_transfer.py"
SPEC = importlib.util.spec_from_file_location("panoptic_transfer", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_chronological_split_uses_earliest_frame_ids_regardless_of_input_order() -> None:
    mask = MODULE._chronological_split(np.asarray([12, 10, 13, 11]), 0.5)
    np.testing.assert_array_equal(mask, np.asarray([False, True, False, True]))


def test_transfer_loader_rejects_an_unstable_alignment(tmp_path: Path) -> None:
    path = tmp_path / "unstable.json"
    path.write_text(
        json.dumps(
            {
                "experiment": "panoptic_coordinate_alignment_restart_aggregation",
                "restart_stability_qualification": {"accepted": False},
                "records": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="restart-stable"):
        MODULE._load_selected_alignment(path)
