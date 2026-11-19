"""Stage-1A: reproducible independent-instance recoverability pilot (HARD only).

This intentionally tests the *same* ZIP hierarchical_alignment solver with
hard one-hot groups. It compares random initial R versus train-unpaired
covariance-sign initial R and changes no algorithm objective or budget.
True generator labels and R_* are kept out of fitting and initialization.
Separate Soft-GCOT experiments begin only after checking this prerequisite.
No DANDI real observations or pre-inspected test trials are used.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from zip_gcot.alignment import SolverConfig, hierarchical_alignment, hard_measures
from zip_gcot.preprocessing import (
    independent_covariance_sign_candidates,
    select_spectral_candidate_train_ot,
)
from run_hard_hiwa import sample_groups


def make_instance(d: int, k: int, overlap: str, instance_id: int):
    """Independently sample both domains from one rotatable distribution.

    Population class centers are explicitly zero-centered before sampling.
    A single scale computed from source TRAIN observations is applied to both
    domains, hence there is no target-specific whitening/nonorthogonal map.
    There is no per-point correspondence even in the generator.
    """
    rng = np.random.default_rng(900000 + 101 * instance_id + 13 * d + k)
    centers = None
    for _ in range(10000):
        z = rng.normal(size=(k, d)) * (2.8 + .2 * np.arange(k))[:, None]
        z -= z.mean(axis=0)
        if np.min(cdist(z, z) + 1e4 * np.eye(k)) > 2.8:
            centers = z
            break
    if centers is None:
        raise RuntimeError("Unable to generate separated non-symmetric centers")
    matrices = []
    for i in range(k):
        u, _ = np.linalg.qr(rng.normal(size=(d, d)))
        scales = np.linspace(.85, .30, d) * (1 + .09 * i)
        matrices.append(u @ np.diag(scales))
    G, _ = np.linalg.qr(rng.normal(size=(d, d)))
    if np.linalg.det(G) < 0:
        G[:, 0] *= -1
    within = .33 if overlap == "separated" else 1.25

    def draw(per_class: int, rotate: bool):
        pts, labels = [], []
        for i in range(k):
            a = centers[i] + within * rng.normal(size=(per_class, d)) @ matrices[i].T
            if rotate:
                a = a @ G.T
            pts.append(a)
            labels.extend([i] * per_class)
        p = rng.permutation(k * per_class)
        return np.concatenate(pts)[p], np.asarray(labels, dtype=int)[p]

    X, tx = draw(90, False)
    Y, ty = draw(90, True)
    Xe, tex = draw(70, False)
    Ye, tey = draw(70, True)
    scale = np.sqrt(np.mean(np.sum(X * X, axis=1)))
    return {
        "X": X / scale, "Y": Y / scale,
        "Xe": Xe / scale, "Ye": Ye / scale,
        "true_x": tx, "true_y": ty,
        "true_x_test": tex, "true_y_test": tey,
        "Rgt": G,
        "scale_from_source_train": float(scale),
    }


def square_w2(X, Y, R):
    c = cdist(X @ R.T, Y, metric="sqeuclidean")
    ii, jj = linear_sum_assignment(c)
    return float(c[ii, jj].mean())


def score(d, labels_x, labels_y, R, P, clx, cly, oracle_acc):
    near = cKDTree(d["Ye"]).query(d["Xe"] @ R.T, k=1)[1]
    acc = float(np.mean(d["true_x_test"] == d["true_y_test"][near]))
    mx = np.array([np.bincount(d["true_x"][labels_x == j], minlength=P.shape[0]).argmax()
                   for j in range(P.shape[0])])
    my = np.array([np.bincount(d["true_y"][labels_y == j], minlength=P.shape[0]).argmax()
                   for j in range(P.shape[0])])
    row_truth = (mx[:, None] == my[None, :]).astype(float)
    purity_x = float(np.mean(mx[labels_x] == d["true_x"]))
    purity_y = float(np.mean(my[labels_y] == d["true_y"]))
    return {
        "test_true_group_1nn_accuracy": acc,
        "oracle_Rstar_test_true_group_1nn_accuracy": float(oracle_acc),
        "true_group_P_mass": float(np.sum(P * row_truth)),
        "train_source_cluster_purity": purity_x,
        "train_target_cluster_purity": purity_y,
        "true_rotation_Frobenius_error": float(np.linalg.norm(R - d["Rgt"], "fro")),
        "heldout_empirical_W2sq": square_w2(d["Xe"][:120], d["Ye"][:120], R),
        "training_empirical_W2sq": square_w2(clx, cly, R),
        "P": P.tolist(),
        "R": R.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instances", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--outer", type=int, default=180)
    parser.add_argument("--inner", type=int, default=20)
    parser.add_argument("--per-cluster", type=int, default=24)
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    plans_dir = root / "plans"
    plans_dir.mkdir(exist_ok=True)
    cfg = SolverConfig(
        maxiter=args.outer, tol=1e-9, mu=.05,
        epsilon_group=.2, epsilon_sample=.1, inner_maxiter=args.inner,
        inner_tol=.01, sinkhorn_maxiter=45, group_sinkhorn_maxiter=200,
        sinkhorn_tol=1e-8, stopping="legacy"
    )
    results = {
        "status": "exploratory_pilot_not_full_20_instance_confirmation",
        "solver": "single ZIP hierarchical_alignment hard measures; NOT a separate GCOT-Hard algorithm",
        "selection": "warm start from source/target train covariance sign candidates; minimum train unpaired Hungarian OT; no truth selection",
        "true_labels_and_Rstar": "never supplied to fitting, initializer, or model selection; only used postfit",
        "real_dandi_used": False,
        "setting": {
            "dimensions": 4, "clusters": 4, "regimes": ["separated", "overlapping"],
            "instances": args.instances, "optimizer_seeds": args.seeds,
            "outer": args.outer, "inner": args.inner,
            "sample_Sinkhorn": 45, "group_Sinkhorn": 200,
            "mu": .05, "epsilon_group": .2, "epsilon_sample": .1,
            "per_cluster_optimizer_sample": args.per_cluster,
            "both_methods_same_local_start_from_global": False
        },
        "runs": [],
    }
    for regime in ["separated", "overlapping"]:
        for instance_id in args.instances:
            d = make_instance(4, 4, regime, instance_id)
            X, Y = d["X"], d["Y"]
            lx = KMeans(4, random_state=42, n_init=20).fit_predict(X)
            ly = KMeans(4, random_state=43, n_init=20).fit_predict(Y)
            selector = np.random.default_rng(5000 + instance_id)
            ii = selector.choice(len(X), 160, replace=False)
            jj = selector.choice(len(Y), 160, replace=False)
            candidates = independent_covariance_sign_candidates(X, Y)
            warm_R, selection = select_spectral_candidate_train_ot(
                candidates, X, Y, ii, jj
            )
            assert len(candidates) == 16
            assert np.linalg.norm(warm_R.T @ warm_R - np.eye(4)) < 1e-7
            oracle_nn = cKDTree(d["Ye"]).query(d["Xe"] @ d["Rgt"].T, k=1)[1]
            oracle_acc = float(np.mean(d["true_x_test"] == d["true_y_test"][oracle_nn]))
            for seed in args.seeds:
                rng = np.random.default_rng(seed)
                px = sample_groups(np.arange(len(X)), lx, args.per_cluster, rng)
                py = sample_groups(np.arange(len(Y)), ly, args.per_cluster, rng)
                sx = hard_measures(X[px], lx[px])
                sy = hard_measures(Y[py], ly[py])
                assert len(sx) == len(sy) == 4
                for mode in ["random", "spectral"]:
                    started = time.monotonic()
                    fit = hierarchical_alignment(
                        sx, sy, seed=seed, config=cfg,
                        scale=True, retain_plans=True,
                        initial_R=(warm_R if mode == "spectral" else None),
                        local_start_from_global=False,
                    )
                    R = np.asarray(fit["R"])
                    P = np.asarray(fit["P"])
                    assert np.linalg.norm(R.T @ R - np.eye(4)) < 1e-7
                    assert np.max(np.abs(P.sum(axis=0) - .25)) < 1e-5
                    assert np.max(np.abs(P.sum(axis=1) - .25)) < 1e-5
                    q = {key: fit[key] for key in fit if key.startswith("Q_")}
                    assert len(q) == 16
                    assert all(v.shape == (args.per_cluster, args.per_cluster)
                               for v in q.values())
                    q_err = max(
                        max(np.abs(v.sum(0) - 1 / args.per_cluster).max(),
                            np.abs(v.sum(1) - 1 / args.per_cluster).max())
                        for v in q.values()
                    )
                    assert q_err < 1e-4
                    run = {
                        "instance": instance_id, "regime": regime,
                        "method": "Hard-HiWA", "initialization": mode,
                        "optimizer_seed": seed,
                        "runtime_seconds": float(time.monotonic() - started),
                        "outer_iterations": int(len(fit["history"])),
                        "strict_ADMM_criteria_met": bool(fit["criteria_satisfied"]),
                        "stop_reason": str(fit["stop_reason"]),
                        "rotation_delta": float(fit["rotation_delta"]),
                        "consensus_residual": float(fit["consensus_residual"]),
                        "Q_max_marginal_error": float(q_err),
                        "group_P_max_marginal_error": float(
                            max(np.abs(P.sum(0) - .25).max(),
                                np.abs(P.sum(1) - .25).max())),
                        "warm_sign_candidate_selected_train_only": (
                            selection if mode == "spectral" else None),
                    }
                    run.update(score(d, lx, ly, R, P, X[ii], Y[jj], oracle_acc))
                    name = f"{regime}_instance{instance_id}_seed{seed}_{mode}"
                    np.savez_compressed(
                        plans_dir / (name + ".npz"),
                        P=P, R=R, source_sample_ids=px, target_sample_ids=py,
                        **q,
                    )
                    results["runs"].append(run)
                    (root / "RESULTS.json").write_text(
                        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    print(
                        f"PHASE1 {name}: accuracy={run['test_true_group_1nn_accuracy']:.4f}"
                        f" oracle={oracle_acc:.4f}"
                        f" Ptrue={run['true_group_P_mass']:.4f}"
                        f" Rerr={run['true_rotation_Frobenius_error']:.4f}"
                        f" admm={run['strict_ADMM_criteria_met']}",
                        flush=True,
                    )
    lines = [
        "# Phase 1A | Independent-generation Hard-HiWA pilot",
        "",
        "**Not the full multi-instance confirmation:** this is a preregistered "
        "small pilot on 4D-K4. No DANDI observations or synchronized training pairs.",
        "Same P/Q/R solver, same samples and budgets; ONLY global R initialization differs.",
        "True group labels and R* are used POSTFIT ONLY. Strict ADMM is separately reported.",
        "",
        "| Regime | Initialization | N | Mean group 1NN | Mean oracle 1NN | Mean true P mass | Strict ADMM |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for regime in ["separated", "overlapping"]:
        for mode in ["random", "spectral"]:
            runs = [r for r in results["runs"]
                    if r["regime"] == regime and r["initialization"] == mode]
            avg = lambda key: float(np.mean([r[key] for r in runs]))
            lines.append(
                f"| {regime} | {mode} | {len(runs)} |"
                f" {avg('test_true_group_1nn_accuracy'):.4f} |"
                f" {avg('oracle_Rstar_test_true_group_1nn_accuracy'):.4f} |"
                f" {avg('true_group_P_mass'):.4f} |"
                f" {sum(r['strict_ADMM_criteria_met'] for r in runs)}/{len(runs)} |"
            )
    lines += [
        "", "Do not interpret training W2 minimization as physical recovery.",
        "P mass is interpreted using postfit cluster-majority true-group IDs; "
        "inspect cluster purity before making claims.",
        "This pilot is insufficient to establish superiority or robust recovery.",
        "Next confirmation requires >=20 independent generated instances per regime, "
        ">=5 optimizer seeds, additional dimension/K regimes, and a frozen success gate.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("PHASE1_PILOT_COMPLETE", len(results["runs"]), flush=True)


if __name__ == "__main__":
    main()
