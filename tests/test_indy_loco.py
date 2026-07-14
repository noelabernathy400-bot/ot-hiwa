from __future__ import annotations

import numpy as np

from datasets.indy_loco import IndyLocoSession, bin_indy_loco_session


def _session() -> IndyLocoSession:
    timestamps = np.arange(0.0, 1.01, 0.01)
    return IndyLocoSession(
        timestamps=timestamps,
        spike_times=(np.array([0.10, 0.20, 0.30, 0.90]), np.array([0.50])),
        cursor_position=np.column_stack((timestamps, 2.0 * timestamps)),
        finger_position=np.column_stack((timestamps, timestamps, timestamps)),
        target_position=np.column_stack((timestamps, timestamps)),
        session_id="synthetic",
    )


def test_binning_returns_time_by_unit_rates_and_cursor_velocity() -> None:
    result = bin_indy_loco_session(
        _session(),
        bin_width_seconds=0.1,
        smoothing_sigma_bins=0.0,
        min_firing_rate_hz=2.0,
    )
    assert result.neural_rates.shape == (10, 1)
    assert result.cursor_position.shape == (10, 2)
    assert result.cursor_velocity.shape == (10, 2)
    assert np.allclose(result.cursor_velocity[1:-1], np.array([1.0, 2.0]))
    assert result.retained_unit_indices.tolist() == [0]


def test_binning_rejects_a_unit_threshold_that_removes_everything() -> None:
    try:
        bin_indy_loco_session(_session(), min_firing_rate_hz=100.0)
    except ValueError as error:
        assert "no units" in str(error)
    else:
        raise AssertionError("expected an empty-unit validation error")
