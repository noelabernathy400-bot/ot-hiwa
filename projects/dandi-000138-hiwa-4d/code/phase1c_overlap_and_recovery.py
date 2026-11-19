"""Stage-1C: genuine overlapping mixture + independent Hard-HiWA P/Q/R.

Only source and target unpaired train observations enter KMeans,
covariance-spectrum initial_R selection, and P/Q/R optimization. True
generator labels and R_* enter only the diagnostic/evaluation functions.
All comparisons use the *same* hard HiWA solver and sample indices.
No observed DANDI trials and no cross-domain pairing are used.

Use --preflight-only to audit independently generated overlap before OT fits.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from zip_gcot.alignment import SolverConfig, hard_measures, hierarchical_alignment
from zip_gcot.numerics import polar
from zip_gcot.preprocessing import (
    independent_covariance_sign_candidates, select_spectral_candidate_train_ot
)

D, K = 4, 4
CENTER_TEMPLATE = np.array([
    [-1.80, -1.00,  0.55,  0.15],
    [ 1.55, -0.95, -0.65,  0.75],
    [-0.15,  1.85,  1.00, -0.55],
    [ 0.50,  0.30, -1.90, -1.00],
], dtype=float)
# Chosen using independent generator-only KMeans/true-R oracle diagnostics;
# never from fitted HiWA outcomes. KMeans purity and oracle label transfer
# are expected to be < 1 in the mild and severe regimes.
REGIME_SIGMA = {"separated": 0.38, "mild_overlap": 1.45, "severe_overlap": 1.85}


def create_instance(regime: str, index: int):
    rng = np.random.default_rng(810000 + 71 * index)
    centers = CENTER_TEMPLATE + .15 * rng.normal(size=(K, D))
    centers -= centers.mean(axis=0)
    matrices = []
    for j in range(K):
        u, _ = np.linalg.qr(rng.normal(size=(D, D)))
        matrices.append(u @ np.diag(
            np.array([1., .83, .68, .52]) * (1. + .07 * j)
        ))
    g, _ = np.linalg.qr(rng.normal(size=(D, D)))
    if np.linalg.det(g) < 0:
        g[:, 0] *= -1
    sigma = REGIME_SIGMA[regime]

    def draw(n, apply_rotation):
        points, true_labels = [], []
        for j in range(K):
            a = centers[j] + sigma * rng.normal(size=(n, D)) @ matrices[j].T
            if apply_rotation:
                a = a @ g.T
            points.append(a)
            true_labels.extend([j] * n)
        shuffle = rng.permutation(K * n)
        return np.concatenate(points)[shuffle], np.asarray(true_labels)[shuffle]

    # Independently sampled, never paired even in the simulator.
    x, lx = draw(180, False)
    y, ly = draw(180, True)
    xh, lxh = draw(100, False)
    yh, lyh = draw(100, True)
    # A common scalar preserves the actual orthogonal relation. No target
    # domain whitening or transformation chosen from paired training samples.
    scale = np.sqrt(np.mean(np.sum(x*x, axis=1)))
    return {
        "x": x / scale, "y": y / scale,
        "xh": xh / scale, "yh": yh / scale,
        "true_x_train": lx, "true_y_train": ly,
        "true_x_heldout": lxh, "true_y_heldout": lyh,
        "Rstar": g, "common_scalar_from_source_train": float(scale),
    }


def make_confusion(predicted, truth):
    matrix = np.zeros((K, K), dtype=np.int64)
    np.add.at(matrix, (predicted, truth), 1)
    return matrix


def matched_purity(confusion):
    rr, cc = linear_sum_assignment(-confusion)
    return float(confusion[rr, cc].sum() / confusion.sum())


def group_class_distributions(confusion):
    row_sums = confusion.sum(axis=1, keepdims=True)
    assert np.all(row_sums > 0)
    return confusion / row_sums


def evaluate_overlap(d, predicted_x, predicted_y):
    cx = make_confusion(predicted_x, d["true_x_train"])
    cy = make_confusion(predicted_y, d["true_y_train"])
    oracle_nn = cKDTree(d["yh"]).query(d["xh"] @ d["Rstar"].T, k=1)[1]
    baseline_nn = cKDTree(d["yh"]).query(d["xh"], k=1)[1]
    return {
        "source_KMeans_true_class_purity_Hungarian": matched_purity(cx),
        "target_KMeans_true_class_purity_Hungarian": matched_purity(cy),
        "source_KMeans_true_class_ARI": float(adjusted_rand_score(
            predicted_x, d["true_x_train"])),
        "target_KMeans_true_class_ARI": float(adjusted_rand_score(
            predicted_y, d["true_y_train"])),
        "oracle_Rstar_true_class_1NN": float(np.mean(
            d["true_x_heldout"] == d["true_y_heldout"][oracle_nn])),
        "identity_true_class_1NN": float(np.mean(
            d["true_x_heldout"] == d["true_y_heldout"][baseline_nn])),
        "source_confusion_predicted_by_truth": cx.tolist(),
        "target_confusion_predicted_by_truth": cy.tolist(),
    }


def sampled_groups(indices, hard_labels, n, rng):
    # Unsup KMeans IDs independently learned; no true class used.
    parts = []
    for j in range(K):
        pool = indices[hard_labels[indices] == j]
        if len(pool) < n:
            raise ValueError(f"predicted cluster {j}: too few points")
        parts.append(rng.choice(pool, n, replace=False))
    return rng.permutation(np.concatenate(parts))


def unpaired_W2squared(x, y, r):
    dist = cdist(x @ r.T, y, metric="sqeuclidean")
    i, j = linear_sum_assignment(dist)
    return float(dist[i, j].mean())


def evaluate_fit(d, lx, ly, fit, overlap, w2_x, w2_y):
    r, p = np.asarray(fit["R"]), np.asarray(fit["P"])
    match = cKDTree(d["yh"]).query(d["xh"] @ r.T, k=1)[1]
    cx = group_class_distributions(np.asarray(
        overlap["source_confusion_predicted_by_truth"], dtype=float))
    cy = group_class_distributions(np.asarray(
        overlap["target_confusion_predicted_by_truth"], dtype=float))
    # Interpretable even with mixed/misaligned KMeans clusters.
    # This is expected true-class match probability under learned P, not
    # oracle group-ID permutation or supervised training.
    true_class_pair_prob = cx @ cy.T
    return {
        "heldout_true_class_1NN": float(np.mean(
            d["true_x_heldout"] == d["true_y_heldout"][match])),
        "oracle_Rstar_true_class_1NN": overlap["oracle_Rstar_true_class_1NN"],
        "P_expected_true_class_agreement": float(np.sum(p * true_class_pair_prob)),
        "P_uniform_baseline_true_class_agreement": float(
            np.mean(true_class_pair_prob)),
        "P_true_class_agreement_matrix_POSTFIT_ONLY": true_class_pair_prob.tolist(),
        "Rstar_fro_error": float(np.linalg.norm(r - d["Rstar"], "fro")),
        "train_unpaired_selection_pool_W2sq": unpaired_W2squared(w2_x, w2_y, r),
        "heldout_unpaired_W2sq": unpaired_W2squared(d["xh"][:160], d["yh"][:160], r),
        "P": p.tolist(), "R": r.tolist(),
    }


def output_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--regimes", nargs="+",
                    choices=list(REGIME_SIGMA), default=list(REGIME_SIGMA))
    ap.add_argument("--instances", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--outer", type=int, default=500)
    ap.add_argument("--inner", type=int, default=20)
    ap.add_argument("--per-cluster", type=int, default=40)
    args = ap.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    records = {
        "research_question": "Recoverability under *genuinely overlapping* "
                             "unpaired 4D K4 mixtures",
        "data": "synthetic only, independent X and Y samples; no DANDI used",
        "fit_never_sees": "true class labels, true Rstar, true cross-domain pairs",
        "truth_uses": "postfit geometry/cluster diagnostics only",
        "regimes": {name: REGIME_SIGMA[name] for name in args.regimes},
        "instances": args.instances,
        "optimizer_seeds": args.seeds,
        "budget": {"outer": args.outer, "inner": args.inner,
                   "sample_sinkhorn": 45, "group_sinkhorn": 200,
                   "per_cluster": args.per_cluster,
                   "mu": .05, "epsilon_sample": .1,
                   "epsilon_group": .2, "scale": True},
        "overlap_preflight": [], "fits": [],
    }
    instance_state = []
    for name in args.regimes:
        for instance_id in args.instances:
            d = create_instance(name, instance_id)
            lx = KMeans(K, random_state=42, n_init=20).fit_predict(d["x"])
            ly = KMeans(K, random_state=43, n_init=20).fit_predict(d["y"])
            ov = evaluate_overlap(d, lx, ly)
            record = {"regime": name, "instance_id": instance_id, **ov}
            records["overlap_preflight"].append(record)
            instance_state.append((name, instance_id, d, lx, ly, ov))
            print(
                f"OVERLAP {name} instance{instance_id}: "
                f"purityX={ov['source_KMeans_true_class_purity_Hungarian']:.4f} "
                f"purityY={ov['target_KMeans_true_class_purity_Hungarian']:.4f} "
                f"oracleNN={ov['oracle_Rstar_true_class_1NN']:.4f} "
                f"ARI_X={ov['source_KMeans_true_class_ARI']:.4f}",
                flush=True)
    output_json(root / "RESULTS.json", records)
    if args.preflight_only:
        print("PHASE1C_OVERLAP_PREFLIGHT_COMPLETE", len(instance_state), flush=True)
        return
    cfg = SolverConfig(maxiter=args.outer, tol=1e-9, mu=.05,
                       epsilon_group=.2, epsilon_sample=.1,
                       inner_maxiter=args.inner, inner_tol=.01,
                       sinkhorn_maxiter=45, group_sinkhorn_maxiter=200,
                       sinkhorn_tol=1e-8, stopping="legacy")
    plans = root / "plans"
    plans.mkdir(exist_ok=True)
    for name, instance_id, d, lx, ly, overlap in instance_state:
        x, y = d["x"], d["y"]
        chooser = np.random.default_rng(51000 + instance_id)
        ix = chooser.choice(len(x), 200, replace=False)
        iy = chooser.choice(len(y), 200, replace=False)
        candidates = independent_covariance_sign_candidates(x, y)
        assert len(candidates) == 16
        warm_r, selection = select_spectral_candidate_train_ot(
            candidates, x, y, ix, iy)
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            sx = sampled_groups(np.arange(len(x)), lx, args.per_cluster, rng)
            sy = sampled_groups(np.arange(len(y)), ly, args.per_cluster, rng)
            mx = hard_measures(x[sx], lx[sx])
            my = hard_measures(y[sy], ly[sy])
            assert len(mx) == len(my) == K
            # Explicit initial_R for both modes ensures same subsequent local
            # RNG state and isolates *only* global R initialization.
            initial_random = polar(np.random.default_rng(seed).random((D, D)))
            for mode in ("random", "spectral"):
                start = time.monotonic()
                fit = hierarchical_alignment(
                    mx, my, seed=seed, config=cfg, scale=True,
                    initial_R=initial_random if mode == "random" else warm_r,
                    local_start_from_global=False, retain_plans=True)
                p, r = fit["P"], fit["R"]
                if np.linalg.norm(r.T @ r - np.eye(D)) > 1e-7:
                    raise AssertionError("R is not orthogonal")
                if np.max(abs(p.sum(axis=1) - .25)) > 1e-5:
                    raise AssertionError("P marginals invalid")
                if np.max(abs(p.sum(axis=0) - .25)) > 1e-5:
                    raise AssertionError("P marginals invalid")
                q = {key: fit[key] for key in fit if key.startswith("Q_")}
                if len(q) != K*K:
                    raise AssertionError("Incomplete 16 local Q plans")
                q_err = max(
                    max(np.abs(v.sum(0) - 1/args.per_cluster).max(),
                        np.abs(v.sum(1) - 1/args.per_cluster).max())
                    for v in q.values())
                if q_err > 1e-4:
                    raise AssertionError(f"Q marginal: {q_err}")
                ev = evaluate_fit(d, lx, ly, fit, overlap, x[ix], y[iy])
                run = {
                    "regime": name, "instance_id": instance_id,
                    "method": "Hard-HiWA",
                    "initialization": mode, "optimization_seed": seed,
                    "outer_iterations": len(fit["history"]),
                    "strict_ADMM": bool(fit["criteria_satisfied"]),
                    "stop_reason": fit["stop_reason"],
                    "rotation_delta": float(fit["rotation_delta"]),
                    "consensus_residual": float(fit["consensus_residual"]),
                    "dual_residual": float(fit["dual_residual"]),
                    "Q_max_marginal_error": q_err,
                    "seconds": float(time.monotonic()-start),
                    "spectral_sign_train_only": (
                        selection["chosen_signs"] if mode == "spectral" else None),
                    **ev,
                }
                records["fits"].append(run)
                np.savez_compressed(
                    plans / f"{name}_i{instance_id}_seed{seed}_{mode}.npz",
                    P=p, R=r, x_rows=sx, y_rows=sy, **q)
                output_json(root / "RESULTS.json", records)
                print(
                    f"FIT {name} instance{instance_id} seed{seed} {mode}: "
                    f"acc={run['heldout_true_class_1NN']:.4f} "
                    f"oracle={run['oracle_Rstar_true_class_1NN']:.4f} "
                    f"Ptrue={run['P_expected_true_class_agreement']:.4f} "
                    f"Rerr={run['Rstar_fro_error']:.4f} "
                    f"ADMM={run['strict_ADMM']} iter={run['outer_iterations']}",
                    flush=True)
    lines = [
        "# Phase 1C: Genuine overlap, fixed 4D/K4 Hard-HiWA geometry",
        "",
        "All values below are experimental synthetic results, NOT real DANDI.",
        "Each row averages optimization seeds AND generated instances; "
        "both are shown in RESULTS.json, with original P/R/16 Q archived.",
        "The randomized generator and both KMeans fits never use paired samples.",
        "True labels and Rstar enter evaluation only. No Soft method is claimed.",
        "",
        "| Regime | Init | Fits | Mean class 1NN | Oracle Rstar 1NN | "
        "True-class P agreement | Strict ADMM |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name in args.regimes:
        for mode in ["random", "spectral"]:
            arr = [a for a in records["fits"] if
                   a["regime"] == name and a["initialization"] == mode]
            avg = lambda key: np.mean([v[key] for v in arr])
            lines.append(
                f"| {name} | {mode} | {len(arr)} | "
                f"{avg('heldout_true_class_1NN'):.4f} | "
                f"{avg('oracle_Rstar_true_class_1NN'):.4f} | "
                f"{avg('P_expected_true_class_agreement'):.4f} | "
                f"{sum(v['strict_ADMM'] for v in arr)}/{len(arr)} |")
    lines += [
        "",
        "Important: for overlapping classes the oracle 1NN is below 1 by "
        "construction, and P expected agreement is weighted by posterior "
        "true-class frequencies. Do not apply clear-cluster 0.8 success "
        "gate without conditioning on oracle difficulty.",
        "This is a small pilot, not the >=20 independent-instance confirmation.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf8")
    print("PHASE1C_FULL_COMPLETE", len(records["fits"]), flush=True)


if __name__ == "__main__":
    main()
