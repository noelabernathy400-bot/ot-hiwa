#!/usr/bin/env python3
"""Download the five official Hyser WFDB random-task trials and prepare 100 ms windows.

Does not train any OT alignment or use synchronized row matches during learning.
Only uses paired timestamps to construct the two independently processed domains.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import requests
import wfdb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT_URL = "https://physionet.org/files/hd-semg/2.0.0/random_dataset/subject01_session1"
TYPES = ("random_preprocess", "random_force")
CHANNELS = ("thumb", "middle", "index")
FORCE_INDICES = (0, 2, 1)
WINDOW_S = 0.1
FS_EMG = 2048
FS_FORCE = 100
LICENSE = "Open Data Commons Attribution License v1.0 (ODC-By 1.0)"
CITATION = (
    "Jiang, X., Dai, C., Liu, X., & Fan, J. (2023). Open Access Dataset and "
    "Toolbox of High-Density Surface Electromyogram Recordings, version 2.0.0, "
    "PhysioNet. DOI:10.13026/hxan-pe94"
)

def download(session: requests.Session, url: str, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            with session.get(url, stream=True, timeout=(20, 180)) as response:
                response.raise_for_status()
                expected = response.headers.get("Content-Length")
                digest = hashlib.sha256()
                total = 0
                tmp = destination.with_suffix(destination.suffix + ".partial")
                with tmp.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            digest.update(chunk)
                            total += len(chunk)
                if expected and total != int(expected):
                    raise IOError(f"Incomplete download: {url}: {total}/{expected}")
                tmp.replace(destination)
                return {
                    "filename": destination.name, "source_url": url,
                    "size_bytes": total, "sha256": digest.hexdigest()
                }
        except (requests.RequestException, IOError) as exc:
            destination.with_suffix(destination.suffix + ".partial").unlink(missing_ok=True)
            if attempt == 3:
                raise RuntimeError(f"Unable to download {url}") from exc
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")

def make_windows(emg: np.ndarray, force: np.ndarray, idx: int):
    assert emg.ndim == 2 and emg.shape[1] == 256, emg.shape
    assert force.ndim == 2 and force.shape[1] == 5, force.shape
    n = min(round(emg.shape[0] * 10 / FS_EMG), force.shape[0] // 10)
    x, y = [], []
    for w in range(n):
        lo = round(w * FS_EMG / 10)
        hi = round((w + 1) * FS_EMG / 10)
        # The dataset's preprocessed EMG is already band/notch filtered.
        e = emg[lo:hi, :]
        f = force[10 * w:10 * (w + 1), :]
        x.append(np.sqrt(np.mean(np.square(e), axis=0)))
        y.append(np.mean(f[:, FORCE_INDICES], axis=0))
    return np.log1p(np.asarray(x)), np.asarray(y), np.full(n, idx, dtype=np.int16)

def whitening(train, test):
    mean = train.mean(axis=0)
    centered = train - mean
    cov = np.cov(centered, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    if vals.min() <= 1e-10:
        raise ValueError(f"Singular or nearly singular 3D covariance: {vals}")
    W = (vecs * vals ** -0.5) @ vecs.T
    return (train - mean) @ W, (test - mean) @ W

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, default=Path("hyser_five_trials_work"))
    args = parser.parse_args()
    root = args.workdir.resolve()
    raw = root / "raw_wfdb"
    derived = root / "derived"
    derived.mkdir(parents=True, exist_ok=True)

    files = []
    session = requests.Session()
    session.headers.update({"User-Agent": "Hyser-five-trial-research-preparation/1.0"})
    for trial in range(1, 6):
        for kind in TYPES:
            for extension in ("hea", "dat"):
                filename = f"{kind}_sample{trial}.{extension}"
                files.append(download(session, f"{ROOT_URL}/{filename}", raw / filename))
                print(f"Downloaded {filename}", flush=True)

    xs, ys, ts = [], [], []
    per_trial = []
    for trial in range(1, 6):
        emg_rec = wfdb.rdrecord(str(raw / f"random_preprocess_sample{trial}"))
        force_rec = wfdb.rdrecord(str(raw / f"random_force_sample{trial}"))
        if float(emg_rec.fs) != FS_EMG or float(force_rec.fs) != FS_FORCE:
            raise ValueError(f"Unexpected sampling rate trial {trial}")
        emg = np.asarray(emg_rec.p_signal, dtype=np.float64)
        force = np.asarray(force_rec.p_signal, dtype=np.float64)
        if not np.isfinite(emg).all() or not np.isfinite(force).all():
            raise ValueError(f"Nonfinite signal trial {trial}")
        x, y, t = make_windows(emg, force, trial)
        xs.append(x); ys.append(y); ts.append(t)
        per_trial.append({
            "trial": trial, "emg_samples": len(emg), "force_samples": len(force),
            "windows_100ms": len(t), "emg_channels": emg.shape[1],
            "force_channels": force.shape[1]
        })
        print(f"Trial {trial}: {len(t)} windows", flush=True)

    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)
    trial_ids = np.concatenate(ts)
    assert X.shape == (1250, 256) and Y.shape == (1250, 3)
    assert all(row["windows_100ms"] == 250 for row in per_trial)
    np.savez_compressed(
        derived / "all_five_trials_UNSCALED_windows.npz",
        X_emg_log1p_rms_256=X, Y_force_thumb_middle_index_3=Y,
        trial_id=trial_ids, sample_in_trial=np.tile(np.arange(250), 5)
    )

    # Trial-disjoint CV. Fit scalers / PCA / whitening strictly on training trials.
    for holdout in range(1, 6):
        mask = trial_ids != holdout
        scaler = StandardScaler().fit(X[mask])
        pca = PCA(n_components=3, svd_solver="full").fit(scaler.transform(X[mask]))
        xx_tr = pca.transform(scaler.transform(X[mask]))
        xx_te = pca.transform(scaler.transform(X[~mask]))
        Xtr, Xte = whitening(xx_tr, xx_te)
        Ytr, Yte = whitening(Y[mask], Y[~mask])
        np.savez_compressed(
            derived / f"trial_heldout_{holdout}.npz",
            X_train=Xtr, Y_train=Ytr, X_test=Xte, Y_test=Yte,
            train_trial_id=trial_ids[mask], test_trial_id=trial_ids[~mask],
            emg_scaler_mean=scaler.mean_, emg_scaler_scale=scaler.scale_,
            emg_pca_mean=pca.mean_, emg_pca_components=pca.components_
        )
        print(f"Holdout {holdout}: training {Xtr.shape}, testing {Xte.shape}", flush=True)

    info = {
        "source": ROOT_URL, "license": LICENSE, "citation": CITATION,
        "not_a_model_training_run": True, "dataset": "Hyser random task, subject01 session1",
        "domain_X": "EMG 256ch, 2048Hz, 100ms RMS -> log1p -> training-only scaler/PCA3/whitening",
        "domain_Y": "force 100Hz, thumb/middle/index -> training-only 3D whitening",
        "window_seconds": WINDOW_S, "trials": per_trial,
        "total_windows_per_domain": int(len(X)),
        "folds": "five folds, one whole trial held out; no paired rows in OT training",
        "original_files": files,
        "method_warning": "Generated features use a freshly specified reproducible pipeline; never treat old 250-point baseline scores as directly comparable without rerunning all methods."
    }
    (derived / "SOURCE_AND_VALIDATION.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (root / "SOURCE_LICENSE_AND_CITATION.txt").write_text(
        f"Original source: {ROOT_URL}\nLicense: {LICENSE}\nCitation: {CITATION}\n"
        "WFDB record files are unmodified; sha256 hashes are in derived/SOURCE_AND_VALIDATION.json.\n",
        encoding="utf-8"
    )
    # One archive preserves the full original 20-file waveform subset; the
    # second is small and contains all reproducible three-dimensional folds.
    for output, members in [
        (root / "hyser_s01_s1_random_5trials_raw_wfdb.zip",
         list(raw.glob("*")) + [root / "SOURCE_LICENSE_AND_CITATION.txt"]),
        (root / "hyser_s01_s1_random_5trials_features_cv.zip",
         list(derived.glob("*")) + [root / "SOURCE_LICENSE_AND_CITATION.txt"])
    ]:
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=3, allowZip64=True) as archive:
            for item in members:
                archive.write(item, arcname=item.relative_to(root))
        print(f"Saved {output}: {output.stat().st_size} bytes", flush=True)
    print(json.dumps({"success": True, "total_windows": len(X),
                      "trials": 5, "files": len(files)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
