"""Prepare a small, leakage-aware Paderborn cross-condition alignment task.

The labels identify bearing health state only for final classifier evaluation.
They are not used by the feature construction or HiWA grouping procedure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DEFAULT_SOURCE_CODES = ("K001", "KA04", "KI04")
DEFAULT_TARGET_CODES = ("K002", "KA15", "KI14")
LABELS = {
    "K001": 0, "K002": 0,
    "KA04": 1, "KA15": 1,
    "KI04": 2, "KI14": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-condition", default="N15_M07_F10")
    parser.add_argument("--target-condition", default="N09_M07_F10")
    parser.add_argument("--source-codes", nargs=3, default=DEFAULT_SOURCE_CODES)
    parser.add_argument("--target-codes", nargs=3, default=DEFAULT_TARGET_CODES)
    parser.add_argument("--records-per-bearing", type=int, default=4)
    parser.add_argument("--windows-per-record", type=int, default=8)
    parser.add_argument("--spectral-bands", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=20)
    return parser.parse_args()


def _measurement_number(path: Path) -> int:
    return int(path.stem.rsplit("_", maxsplit=1)[-1])


def _files(root: Path, code: str, condition: str, count: int) -> list[Path]:
    candidates = sorted(
        (root / code).rglob(f"{condition}_{code}_*.mat"), key=_measurement_number
    )
    if len(candidates) < count:
        raise ValueError(f"{code} / {condition} has {len(candidates)} records; expected {count}")
    return candidates[:count]


def _vibration(path: Path) -> np.ndarray:
    record = loadmat(path, squeeze_me=True, struct_as_record=False)[path.stem]
    for channel in record.Y:
        if channel.Name == "vibration_1":
            return np.asarray(channel.Data, dtype=float)
    raise ValueError(f"vibration_1 not found in {path}")


def _spectral_windows(signal: np.ndarray, windows: int, bands: int) -> np.ndarray:
    usable = (len(signal) // windows) * windows
    chunks = signal[:usable].reshape(windows, -1)
    spectrum = np.log1p(np.abs(np.fft.rfft(chunks, axis=1))[:, 1:])
    usable_bins = (spectrum.shape[1] // bands) * bands
    if usable_bins == 0:
        raise ValueError("too many spectral bands for the available signal length")
    return spectrum[:, :usable_bins].reshape(windows, bands, -1).mean(axis=2)


def _condition_matrix(root: Path, codes: tuple[str, ...], condition: str, records: int, windows: int, bands: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    record_ids: list[str] = []
    for code in codes:
        for path in _files(root, code, condition, records):
            value = _spectral_windows(_vibration(path), windows, bands)
            features.append(value)
            labels.append(np.full(len(value), LABELS[code], dtype=np.int64))
            record_ids.extend([path.stem] * len(value))
    return np.vstack(features), np.concatenate(labels), record_ids


def main() -> None:
    args = parse_args()
    if args.records_per_bearing < 1 or args.windows_per_record < 1:
        raise SystemExit("records-per-bearing and windows-per-record must be positive")
    source_raw, source_labels, source_ids = _condition_matrix(
        args.input_root, tuple(args.source_codes), args.source_condition, args.records_per_bearing, args.windows_per_record, args.spectral_bands
    )
    target_raw, target_labels, target_ids = _condition_matrix(
        args.input_root, tuple(args.target_codes), args.target_condition, args.records_per_bearing, args.windows_per_record, args.spectral_bands
    )
    combined = np.vstack([source_raw, target_raw])
    standardized = StandardScaler().fit_transform(combined)
    components = min(args.embedding_dim, standardized.shape[0] - 1, standardized.shape[1])
    embedding = PCA(n_components=components, random_state=0).fit_transform(standardized).astype(np.float32)
    split = len(source_raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        source_features=embedding[:split],
        target_features=embedding[split:],
        source_labels=source_labels,
        target_labels=target_labels,
        source_record_ids=np.asarray(source_ids),
        target_record_ids=np.asarray(target_ids),
        source_condition=np.asarray([args.source_condition]),
        target_condition=np.asarray([args.target_condition]),
        source_bearing_codes=np.asarray(args.source_codes),
        target_bearing_codes=np.asarray(args.target_codes),
    )
    print(
        f"saved {args.output}; source={split}, target={len(target_raw)}, "
        f"classes={len(args.source_codes)}, embedding_dim={components}"
    )


if __name__ == "__main__":
    main()
