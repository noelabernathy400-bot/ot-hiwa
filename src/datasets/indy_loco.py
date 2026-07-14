"""Indy--Loco session loading and leakage-safe neural binning utilities.

The public O'Doherty--Sabes sessions are MATLAB 7.3 (HDF5) files containing
spike times plus synchronised cursor and finger positions.  This adapter keeps
the raw behavioural traces separate from the neural representation so an
experiment can withhold target-session behaviour until its final evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class IndyLocoSession:
    """One public Indy--Loco session in its original time coordinate."""

    timestamps: np.ndarray
    spike_times: tuple[np.ndarray, ...]
    cursor_position: np.ndarray
    finger_position: np.ndarray
    target_position: np.ndarray
    session_id: str


@dataclass(frozen=True)
class BinnedIndyLocoSession:
    """Fixed-width neural rates and cursor kinematics for one session."""

    bin_centers: np.ndarray
    neural_rates: np.ndarray
    cursor_position: np.ndarray
    cursor_velocity: np.ndarray
    retained_unit_indices: np.ndarray
    bin_width_seconds: float


def _matrix_rows(handle: h5py.File, name: str) -> np.ndarray:
    """Read a MATLAB matrix stored as dimensions-by-time into time-by-dim."""
    values = np.asarray(handle[name], dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional MATLAB matrix")
    return values.T.copy()


def load_indy_loco_session(path: str | Path) -> IndyLocoSession:
    """Load timestamped spikes and synchronised behaviour from an HDF5 MAT file."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Indy--Loco session not found: {source}")
    with h5py.File(source, "r") as handle:
        required = {"t", "spikes", "cursor_pos", "finger_pos", "target_pos"}
        missing = required.difference(handle.keys())
        if missing:
            raise ValueError(f"{source.name} is missing expected fields: {sorted(missing)}")
        timestamps = np.asarray(handle["t"], dtype=float).reshape(-1)
        cursor = _matrix_rows(handle, "cursor_pos")
        finger = _matrix_rows(handle, "finger_pos")
        target = _matrix_rows(handle, "target_pos")
        if any(values.shape[0] != timestamps.size for values in (cursor, finger, target)):
            raise ValueError("behavioural arrays must share the timestamp axis")
        if not np.all(np.diff(timestamps) > 0):
            raise ValueError("session timestamps must be strictly increasing")

        spike_cells = np.asarray(handle["spikes"])
        spike_times: list[np.ndarray] = []
        for reference in spike_cells.flat:
            if not reference:
                spike_times.append(np.empty(0, dtype=float))
                continue
            times = np.asarray(handle[reference], dtype=float).reshape(-1)
            spike_times.append(np.sort(times))

    return IndyLocoSession(
        timestamps=timestamps,
        spike_times=tuple(spike_times),
        cursor_position=cursor,
        finger_position=finger,
        target_position=target,
        session_id=source.stem,
    )


def bin_indy_loco_session(
    session: IndyLocoSession,
    *,
    bin_width_seconds: float = 0.05,
    smoothing_sigma_bins: float = 1.0,
    min_firing_rate_hz: float = 0.5,
) -> BinnedIndyLocoSession:
    """Bin spikes and interpolate cursor kinematics on an evenly spaced grid.

    Unit selection uses spike times only.  Cursor position and velocity are
    returned separately so callers can avoid reading target-session behaviour
    during unsupervised alignment.
    """
    if bin_width_seconds <= 0:
        raise ValueError("bin_width_seconds must be positive")
    if smoothing_sigma_bins < 0:
        raise ValueError("smoothing_sigma_bins must be non-negative")
    if min_firing_rate_hz < 0:
        raise ValueError("min_firing_rate_hz must be non-negative")

    start, stop = float(session.timestamps[0]), float(session.timestamps[-1])
    edges = np.arange(start, stop + bin_width_seconds, bin_width_seconds)
    if edges.size < 3:
        raise ValueError("session is too short for the requested bin width")
    centers = (edges[:-1] + edges[1:]) / 2.0
    duration = edges[-1] - edges[0]
    firing_rates = np.asarray([times.size / duration for times in session.spike_times])
    retained = np.flatnonzero(firing_rates >= min_firing_rate_hz)
    if retained.size == 0:
        raise ValueError("no units meet min_firing_rate_hz")

    rates = np.stack(
        [np.histogram(session.spike_times[index], bins=edges)[0] / bin_width_seconds for index in retained],
        axis=1,
    ).astype(np.float64)
    if smoothing_sigma_bins > 0:
        rates = gaussian_filter1d(rates, sigma=smoothing_sigma_bins, axis=0, mode="nearest")

    cursor = np.column_stack(
        [np.interp(centers, session.timestamps, session.cursor_position[:, axis]) for axis in range(session.cursor_position.shape[1])]
    )
    velocity = np.gradient(cursor, bin_width_seconds, axis=0)
    return BinnedIndyLocoSession(
        bin_centers=centers,
        neural_rates=rates,
        cursor_position=cursor,
        cursor_velocity=velocity,
        retained_unit_indices=retained,
        bin_width_seconds=float(bin_width_seconds),
    )
