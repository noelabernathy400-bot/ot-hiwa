"""Build cross-day trial-level latent features from DANDI 000688 NWB files.

The reach direction is carried to the output only for evaluation.  It is not
used to estimate movement onset, fit PCA, balance trials, or fit an alignment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA

from .config import DATASET_ID, DATASET_VERSION, PREPROCESSING


def _ragged_spikes(group: h5py.Group) -> list[np.ndarray]:
    values = np.asarray(group["spike_times"], dtype=float)
    stops = np.asarray(group["spike_times_index"], dtype=int)
    starts = np.r_[0, stops[:-1]]
    return [values[a:b] for a, b in zip(starts, stops)]


def _decode(values: np.ndarray) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in values]


def _movement_onsets(
    trial_go: np.ndarray,
    trial_stop: np.ndarray,
    velocity: np.ndarray,
    timestamps: np.ndarray,
    threshold_fraction: float = 0.10,
    sustain_ms: float = 30.0,
) -> np.ndarray:
    """Infer onset from cursor speed without looking at reach direction."""
    dt = float(np.median(np.diff(timestamps)))
    sigma = max(1.0, 0.03 / dt)
    speed = gaussian_filter1d(np.linalg.norm(velocity, axis=1), sigma=sigma)
    sustain = max(1, int(round((sustain_ms / 1000.0) / dt)))
    out = np.full(len(trial_go), np.nan)
    for i, (go, stop) in enumerate(zip(trial_go, trial_stop)):
        search = np.flatnonzero((timestamps >= go) & (timestamps <= stop))
        baseline_idx = np.flatnonzero((timestamps >= go - 0.20) & (timestamps < go))
        if len(search) < sustain or not len(baseline_idx):
            continue
        baseline = float(np.median(speed[baseline_idx]))
        peak = float(np.max(speed[search]))
        threshold = baseline + threshold_fraction * max(peak - baseline, 0.0)
        above = speed[search] >= threshold
        run = np.convolve(above.astype(int), np.ones(sustain, dtype=int), mode="valid")
        hits = np.flatnonzero(run == sustain)
        if len(hits):
            out[i] = timestamps[search[hits[0]]]
    return out


def load_session(
    path: Path,
    dimension: int = PREPROCESSING.latent_dimension,
    bin_ms: float = PREPROCESSING.bin_ms,
    smoothing_ms: float = PREPROCESSING.smoothing_ms,
    window_ms: tuple[float, float] = PREPROCESSING.window_ms,
) -> dict[str, np.ndarray | dict]:
    if not path.is_file():
        raise FileNotFoundError(f"NWB session not found: {path}")
    with h5py.File(path, "r") as f:
        trials = f["intervals/trials"]
        go = np.asarray(trials["go_cue_time"], dtype=float)
        stop = np.asarray(trials["stop_time"], dtype=float)
        direction = np.asarray(trials["target_id"], dtype=int)
        result = np.asarray(trials["result"])
        velocity = np.asarray(f["processing/behavior/Velocity/cursor_vel/data"], dtype=float)
        timestamps = np.asarray(
            f["processing/behavior/Velocity/cursor_vel/timestamps"], dtype=float
        )
        onsets = _movement_onsets(
            go,
            stop,
            velocity,
            timestamps,
            threshold_fraction=PREPROCESSING.onset_threshold_fraction,
            sustain_ms=PREPROCESSING.onset_sustain_ms,
        )
        rewarded = np.array([x in (b"R", "R") for x in result])
        keep = rewarded & np.isfinite(onsets)
        spikes = _ragged_spikes(f["units"])

        electrodes = np.asarray(f["units/electrodes"], dtype=int)
        electrode_table = f["general/extracellular_ephys/electrodes"]
        area_by_electrode = _decode(np.asarray(electrode_table["location"]))
        unit_areas = np.array([area_by_electrode[i] for i in electrodes])
        motor = np.array(
            [("Motor" in area) or ("Premotor" in area) for area in unit_areas], dtype=bool
        )
        spikes = [s for s, use in zip(spikes, motor) if use]
        unit_areas = unit_areas[motor]

    onsets = onsets[keep]
    labels = direction[keep]
    if len(np.unique(labels)) != 8:
        raise ValueError(f"Expected 8 reach directions in {path.name}; found {np.unique(labels)}")
    if len(onsets) <= dimension:
        raise ValueError(f"Too few valid trials in {path.name}: {len(onsets)}")
    bin_s = bin_ms / 1000.0
    lo, hi = window_ms[0] / 1000.0, window_ms[1] / 1000.0
    relative_edges = np.arange(lo, hi + bin_s * 0.5, bin_s)
    n_bins = len(relative_edges) - 1
    rates = np.empty((len(onsets), n_bins, len(spikes)), dtype=float)
    for trial_idx, onset in enumerate(onsets):
        edges = onset + relative_edges
        for unit_idx, times in enumerate(spikes):
            rates[trial_idx, :, unit_idx] = np.histogram(times, bins=edges)[0] / bin_s

    rates = gaussian_filter1d(rates, sigma=smoothing_ms / bin_ms, axis=1, mode="nearest")
    rates = np.sqrt(np.maximum(rates, 0.0))
    flat = rates.reshape(-1, rates.shape[-1])
    # Session-local centering and soft normalization are label-free.  PCA is
    # independently fitted in each domain, intentionally leaving a rotation to recover.
    flat -= flat.mean(axis=0, keepdims=True)
    flat /= flat.std(axis=0, keepdims=True) + 1e-4
    pca = PCA(n_components=dimension, svd_solver="full").fit(flat)
    trajectory = pca.transform(flat).reshape(len(onsets), n_bins, dimension)
    summary = trajectory.mean(axis=1)
    summary -= summary.mean(axis=0, keepdims=True)
    rms = np.sqrt(np.mean(np.sum(summary**2, axis=1)))
    summary /= max(rms, np.finfo(float).eps)

    counts = {str(int(k)): int(np.sum(labels == k)) for k in np.unique(labels)}
    areas, area_counts = np.unique(unit_areas, return_counts=True)
    metadata = {
        "file": path.name,
        "rewarded_trials_with_onset": int(len(labels)),
        "original_trials": int(len(result)),
        "units": int(len(spikes)),
        "unit_areas": dict(zip(areas.tolist(), area_counts.astype(int).tolist())),
        "direction_counts": counts,
        "movement_onset_rule": "first 30 ms sustained speed >= pre-go baseline + 10% of trial peak excursion",
        "window_ms": list(window_ms),
        "bin_ms": bin_ms,
        "gaussian_smoothing_ms": smoothing_ms,
        "latent_dimension": dimension,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }
    return {
        "features": summary,
        "trajectories": trajectory,
        "labels": labels,
        "onsets": onsets,
        "metadata": metadata,
    }


def run(source: Path, target: Path, output: Path, dimension: int) -> None:
    x = load_session(source, dimension=dimension)
    y = load_session(target, dimension=dimension)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        source=x["features"],
        target=y["features"],
        source_trajectory=x["trajectories"],
        target_trajectory=y["trajectories"],
        source_labels=x["labels"],
        target_labels=y["labels"],
        source_onsets=x["onsets"],
        target_onsets=y["onsets"],
    )
    manifest = {
        "dataset": f"{DATASET_ID} version {DATASET_VERSION}",
        "experiment": "same monkey M, center-out task, 2015-06-10 to 2015-06-12",
        "labels_used_for_preprocessing_or_fit": False,
        "labels_used_for_evaluation_only": True,
        "source": x["metadata"],
        "target": y["metadata"],
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dimension", type=int, default=PREPROCESSING.latent_dimension)
    args = parser.parse_args()
    run(args.source, args.target, args.output, args.dimension)


if __name__ == "__main__":
    main()
