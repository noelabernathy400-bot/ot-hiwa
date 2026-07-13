"""Minimal, protocol-locked PAMAP2 wrist--chest preparation utilities.

The loader produces paired views for *evaluation*, but downstream alignment
code receives only feature matrices and source activity labels.  Target labels
and pair IDs are deliberately returned separately so callers can keep them out
of model fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


ACTIVITY_NAMES = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
}

# Zero-based columns in the official 54-column file.  We use the 6g
# accelerometer and gyroscope because both sensors expose the same channels.
VIEW_COLUMNS = {
    "wrist": np.r_[7:13],
    "chest": np.r_[24:30],
    "ankle": np.r_[41:47],
}


@dataclass(frozen=True)
class PAMAP2PairedWindows:
    source_features: np.ndarray
    target_features: np.ndarray
    source_labels: np.ndarray
    target_labels: np.ndarray
    pair_ids: np.ndarray
    subject: int
    source_view: str
    target_view: str
    window_samples: int


@dataclass(frozen=True)
class PAMAP2DomainAdaptationSplit:
    """Leakage-safe temporal split for wrist-to-chest adaptation.

    The source training and validation views are labelled. The synchronous
    chest view of source-training windows is retained solely for explicitly
    weakly supervised cross-view experiments; it has no target activity label.
    The chest adaptation view is deliberately returned without labels or pair IDs.
    Labels and pair IDs for the final paired source/chest holdout live in
    ``evaluation_*`` fields and must not be supplied to a trainer.
    """

    source_train_features: np.ndarray
    source_train_labels: np.ndarray
    source_train_paired_target_features: np.ndarray
    source_validation_features: np.ndarray
    source_validation_labels: np.ndarray
    target_adaptation_features: np.ndarray
    evaluation_source_features: np.ndarray
    evaluation_target_features: np.ndarray
    evaluation_source_labels: np.ndarray
    evaluation_target_labels: np.ndarray
    evaluation_pair_ids: np.ndarray
    subject: int
    source_view: str
    target_view: str
    window_samples: int


@dataclass(frozen=True)
class PAMAP2SplitCounts:
    """Per-activity window counts in chronological order."""

    source_train: int = 8
    source_validation: int = 2
    target_adaptation: int = 3
    target_test: int = 3

    @property
    def total(self) -> int:
        return self.source_train + self.source_validation + self.target_adaptation + self.target_test

    def validate(self) -> None:
        if min(self.source_train, self.source_validation, self.target_adaptation, self.target_test) <= 0:
            raise ValueError("all split counts must be positive")


def protocol_subject_path(raw_directory: str | Path, subject: int) -> Path:
    """Return the expected official Protocol file path for a subject code."""
    return Path(raw_directory) / "Protocol" / f"subject{int(subject)}.dat"


def load_protocol_subject(raw_directory: str | Path, subject: int) -> np.ndarray:
    path = protocol_subject_path(raw_directory, subject)
    if not path.exists():
        raise FileNotFoundError(
            f"PAMAP2 Protocol file not found: {path}. Download and unpack the official archive first."
        )
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 54:
        raise ValueError(f"expected 54 columns in {path}, found shape {values.shape}")
    return values


def window_features(values: np.ndarray) -> np.ndarray:
    """Return 48 label-free time/frequency features for a six-channel window."""
    signal = np.asarray(values, dtype=np.float64)
    if signal.ndim != 2 or signal.shape[1] != 6:
        raise ValueError("values must have shape (time, 6)")
    if not np.isfinite(signal).all():
        raise ValueError("window features require finite sensor values")
    time_features = np.concatenate((
        signal.mean(axis=0),
        signal.std(axis=0),
        np.sqrt(np.mean(signal**2, axis=0)),
        np.ptp(signal, axis=0),
    ))
    spectrum = np.abs(np.fft.rfft(signal - signal.mean(axis=0, keepdims=True), axis=0)) ** 2
    bands = np.array_split(spectrum[1:], 4, axis=0)
    band_features = np.concatenate([
        np.log1p(band.mean(axis=0)) if band.size else np.zeros(signal.shape[1])
        for band in bands
    ])
    return np.concatenate((time_features, band_features)).astype(np.float32)


def build_paired_windows(
    rows: np.ndarray,
    *,
    subject: int,
    source_view: str = "wrist",
    target_view: str = "chest",
    activities: Iterable[int] = (1, 2, 3, 4, 5, 6),
    window_samples: int = 200,
    windows_per_activity: int = 16,
) -> PAMAP2PairedWindows:
    """Create a deterministic source-labelled / target-hidden paired dataset.

    Windows are selected using the source activity stream before the two views
    are separated.  The returned target labels and pair IDs are evaluation
    metadata; fitting code must not receive them.
    """
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 54:
        raise ValueError("rows must have the official PAMAP2 shape (n, 54)")
    if source_view not in VIEW_COLUMNS or target_view not in VIEW_COLUMNS:
        raise ValueError(f"views must be one of {tuple(VIEW_COLUMNS)}")
    if source_view == target_view:
        raise ValueError("source_view and target_view must differ")
    if window_samples <= 1 or windows_per_activity <= 0:
        raise ValueError("window_samples must exceed one and windows_per_activity must be positive")
    source_columns = VIEW_COLUMNS[source_view]
    target_columns = VIEW_COLUMNS[target_view]
    activity_values = values[:, 1].astype(np.int64)
    source_features: list[np.ndarray] = []
    target_features: list[np.ndarray] = []
    source_labels: list[int] = []
    pair_ids: list[str] = []
    for activity in activities:
        selected = 0
        indices = np.flatnonzero(activity_values == int(activity))
        for start in range(0, max(0, len(indices) - window_samples + 1), window_samples):
            block = indices[start : start + window_samples]
            if block.size != window_samples or not np.array_equal(block, np.arange(block[0], block[0] + window_samples)):
                continue
            source_window = values[block][:, source_columns]
            target_window = values[block][:, target_columns]
            if not np.isfinite(source_window).all() or not np.isfinite(target_window).all():
                continue
            source_features.append(window_features(source_window))
            target_features.append(window_features(target_window))
            source_labels.append(int(activity))
            pair_ids.append(f"subject{subject}:rows{block[0]}-{block[-1]}")
            selected += 1
            if selected == windows_per_activity:
                break
        if selected != windows_per_activity:
            raise ValueError(
                f"subject {subject}, activity {activity}: found only {selected} valid windows; "
                "reduce windows_per_activity or choose another subject"
            )
    labels = np.asarray(source_labels, dtype=np.int64)
    return PAMAP2PairedWindows(
        source_features=np.stack(source_features),
        target_features=np.stack(target_features),
        source_labels=labels,
        target_labels=labels.copy(),
        pair_ids=np.asarray(pair_ids, dtype=object),
        subject=int(subject),
        source_view=source_view,
        target_view=target_view,
        window_samples=int(window_samples),
    )


def build_temporal_domain_adaptation_split(
    rows: np.ndarray,
    *,
    subject: int,
    source_view: str = "wrist",
    target_view: str = "chest",
    activities: Iterable[int] = (1, 2, 3, 4, 5, 6),
    window_samples: int = 200,
    counts: PAMAP2SplitCounts = PAMAP2SplitCounts(),
) -> PAMAP2DomainAdaptationSplit:
    """Make four disjoint chronological partitions within every activity.

    Windows are never randomly shuffled.  For each activity, earliest windows
    form labelled source training, followed by labelled source validation,
    followed by label-free chest adaptation, and finally paired evaluation.
    This keeps target-test rows out of scaling, alignment and model selection.
    """
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 54:
        raise ValueError("rows must have the official PAMAP2 shape (n, 54)")
    if source_view not in VIEW_COLUMNS or target_view not in VIEW_COLUMNS:
        raise ValueError(f"views must be one of {tuple(VIEW_COLUMNS)}")
    if source_view == target_view:
        raise ValueError("source_view and target_view must differ")
    if window_samples <= 1:
        raise ValueError("window_samples must exceed one")
    counts.validate()

    source_columns = VIEW_COLUMNS[source_view]
    target_columns = VIEW_COLUMNS[target_view]
    activity_values = values[:, 1].astype(np.int64)
    buckets: dict[str, list[object]] = {
        "source_train_features": [], "source_train_labels": [], "source_train_paired_target_features": [],
        "source_validation_features": [], "source_validation_labels": [],
        "target_adaptation_features": [], "evaluation_source_features": [],
        "evaluation_target_features": [], "evaluation_source_labels": [],
        "evaluation_target_labels": [], "evaluation_pair_ids": [],
    }
    for activity in activities:
        indices = np.flatnonzero(activity_values == int(activity))
        candidates: list[tuple[np.ndarray, np.ndarray, str]] = []
        for start in range(0, max(0, len(indices) - window_samples + 1), window_samples):
            block = indices[start : start + window_samples]
            if block.size != window_samples or not np.array_equal(block, np.arange(block[0], block[0] + window_samples)):
                continue
            source_window = values[block][:, source_columns]
            target_window = values[block][:, target_columns]
            if not np.isfinite(source_window).all() or not np.isfinite(target_window).all():
                continue
            candidates.append((
                window_features(source_window),
                window_features(target_window),
                f"subject{subject}:rows{block[0]}-{block[-1]}",
            ))
            if len(candidates) == counts.total:
                break
        if len(candidates) != counts.total:
            raise ValueError(
                f"subject {subject}, activity {activity}: found only {len(candidates)} valid windows; "
                f"need {counts.total} for the temporal split"
            )
        train_end = counts.source_train
        validation_end = train_end + counts.source_validation
        adaptation_end = validation_end + counts.target_adaptation
        for source_feature, target_feature, pair_id in candidates[:train_end]:
            buckets["source_train_features"].append(source_feature)
            buckets["source_train_labels"].append(int(activity))
            buckets["source_train_paired_target_features"].append(target_feature)
        for source_feature, _, _ in candidates[train_end:validation_end]:
            buckets["source_validation_features"].append(source_feature)
            buckets["source_validation_labels"].append(int(activity))
        for _, target_feature, _ in candidates[validation_end:adaptation_end]:
            buckets["target_adaptation_features"].append(target_feature)
        for source_feature, target_feature, pair_id in candidates[adaptation_end:]:
            buckets["evaluation_source_features"].append(source_feature)
            buckets["evaluation_target_features"].append(target_feature)
            buckets["evaluation_source_labels"].append(int(activity))
            buckets["evaluation_target_labels"].append(int(activity))
            buckets["evaluation_pair_ids"].append(pair_id)

    return PAMAP2DomainAdaptationSplit(
        source_train_features=np.stack(buckets["source_train_features"]),
        source_train_labels=np.asarray(buckets["source_train_labels"], dtype=np.int64),
        source_train_paired_target_features=np.stack(buckets["source_train_paired_target_features"]),
        source_validation_features=np.stack(buckets["source_validation_features"]),
        source_validation_labels=np.asarray(buckets["source_validation_labels"], dtype=np.int64),
        target_adaptation_features=np.stack(buckets["target_adaptation_features"]),
        evaluation_source_features=np.stack(buckets["evaluation_source_features"]),
        evaluation_target_features=np.stack(buckets["evaluation_target_features"]),
        evaluation_source_labels=np.asarray(buckets["evaluation_source_labels"], dtype=np.int64),
        evaluation_target_labels=np.asarray(buckets["evaluation_target_labels"], dtype=np.int64),
        evaluation_pair_ids=np.asarray(buckets["evaluation_pair_ids"], dtype=object),
        subject=int(subject),
        source_view=source_view,
        target_view=target_view,
        window_samples=int(window_samples),
    )
