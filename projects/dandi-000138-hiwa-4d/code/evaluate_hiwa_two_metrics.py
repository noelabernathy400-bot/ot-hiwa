"""HiWA metrics on DANDI 000138: original rMSE formula and corrected held-out R².

IMPORTANT: no ground-truth rotation exists for real neural/hand data.  rMSE is
therefore evaluated against a supervised TRAIN-only Procrustes rotation and
MUST be called a proxy, not ground-truth rMSE.  The original _eval_R2 in the
uploaded PyHiWA erroneously uses signed residual means, not squared errors.
This evaluator implements the standard coefficient of determination instead.
Never pass the supervised reference to the unpaired HiWA fit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from hiwa_original import _rMSE


def r2(y, yhat):
    if y.shape != yhat.shape or y.ndim != 2:
        raise ValueError("R² requires same-sized two-dimensional arrays")
    denom = np.sum((y - y.mean(axis=0)) ** 2)
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("Target test variance must be positive")
    return float(1 - np.sum((y - yhat) ** 2) / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", type=Path, required=True)
    ap.add_argument("--alignment", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    raw = args.clusters.read_bytes()
    h = json.loads(args.alignment.read_text(encoding="utf-8"))
    if hashlib.sha256(raw).hexdigest() != h["data_sha256"]:
        raise ValueError("The clustered input differs from the frozen HiWA run")
    z = np.load(args.clusters, allow_pickle=False)
    train, test = z["train_mask"].astype(bool), z["test_mask"].astype(bool)
    assert not np.any(train & test)
    assert int(train.sum()) == h["training_n"] and int(test.sum()) == h["test_n"]
    x = (np.asarray(z["neural4"], dtype=float) - np.asarray(h["source_mean"])) / h["source_radius"]
    y = (np.asarray(z["hand4"], dtype=float) - np.asarray(h["target_mean"])) / h["target_radius"]
    assert x.shape == y.shape and x.shape[1] == 4
    assert np.isfinite(x).all() and np.isfinite(y).all()

    # Only the evaluation reference may access training time-aligned pairs.
    u, _, vt = np.linalg.svd(y[train].T @ x[train])
    reference = u @ vt
    xt, yt = x[test], y[test]

    def measure(R):
        R = np.asarray(R, dtype=float)
        if R.shape != (4, 4) or np.linalg.norm(R.T @ R - np.eye(4), "fro") > 1e-7:
            raise ValueError("Expected a 4x4 orthogonal HiWA transform")
        return {
            "rMSE_proxy": float(_rMSE(xt, R, reference)),
            "R2": r2(yt, xt @ R.T),
        }

    before = measure(np.eye(4))
    supervised_reference = measure(reference)
    rows = [{"seed": int(s["seed"]), "converged": bool(s["strict_converged"]),
             **measure(s["R"])} for s in h["seed_results"]]
    stats = {}
    for metric in ("rMSE_proxy", "R2"):
        values = np.array([s[metric] for s in rows])
        stats[metric] = {"before": before[metric], "mean": float(values.mean()),
                         "sample_sd": float(values.std(ddof=1)),
                         "reference": supervised_reference[metric]}
    assert np.isclose(supervised_reference["rMSE_proxy"], 0, atol=1e-12)
    assert np.isclose(r2(yt, yt), 1.0, atol=1e-12)
    report = {
        "dataset": h["dataset"],
        "data_version": h["data_version"],
        "ground_truth_rotation_known": False,
        "rMSE_definition": "original PyHiWA _rMSE(X_test, R_estimated, R_ref), R_ref is TRAIN-only supervised Procrustes (proxy; NOT R_gt)",
        "R2_definition": "1 - sum_test ||y_i-Rx_i||² / sum_test ||y_i-mean_test(y)||² (corrected standard R²)",
        "hiwa_solver_retrained": False,
        "training_pairs_used_for_hiwa": False,
        "data_sha256": h["data_sha256"],
        "original_solver_sha256": h["original_solver_sha256"],
        "hiwa_results_sha256": hashlib.sha256(args.alignment.read_bytes()).hexdigest(),
        "before_identity": before,
        "train_supervised_reference_diagnostic": supervised_reference,
        "seeds": rows,
        "statistics": stats
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
