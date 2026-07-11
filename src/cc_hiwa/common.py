from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import sqrtm
from sklearn.neighbors import NearestNeighbors


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYHIWA_ROOT = PROJECT_ROOT / "最优传输" / "PyHiWA"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
FIGURES_DIR = PROJECT_ROOT / "experiments" / "figures"

if str(PYHIWA_ROOT) not in sys.path:
    sys.path.insert(0, str(PYHIWA_ROOT))

from hiwa import HiWA  # noqa: E402


def ensure_output_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def whiten(data: np.ndarray) -> np.ndarray:
    centered = data - np.mean(data, axis=0, keepdims=True)
    covariance_root = sqrtm(np.cov(centered, rowvar=False))
    covariance_root = np.real_if_close(covariance_root).astype(float)
    return centered @ np.linalg.pinv(covariance_root)


def nearest_neighbor_accuracy(
    reference: np.ndarray,
    reference_labels: np.ndarray,
    queries: np.ndarray,
    query_labels: np.ndarray,
) -> float:
    model = NearestNeighbors(n_neighbors=1).fit(reference)
    indices = model.kneighbors(queries, return_distance=False).ravel()
    return float(np.mean(reference_labels[indices] == query_labels))


def original_r2(aligned_xy: np.ndarray, movement_xy: np.ndarray) -> float:
    aligned_normalized = whiten(aligned_xy)
    movement_normalized = whiten(movement_xy)
    numerator = np.mean((movement_normalized - aligned_normalized) ** 2, axis=0).sum()
    denominator = np.var(movement_normalized, axis=0).sum()
    return float(1.0 - numerator / denominator)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
