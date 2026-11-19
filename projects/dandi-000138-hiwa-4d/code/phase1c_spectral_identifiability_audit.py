"""Posthoc diagnostic of spectral initializer under genuine synthetic overlap.

No fitted HiWA model is refit, changed or selected by physical labels.
Truth R* and true class labels are only used in POSTHOC diagnostic columns.
The actual initializer always selects the minimum TRAIN unpaired W2 among
the 16 covariance-eigenvector sign candidates.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from zip_gcot.preprocessing import independent_covariance_sign_candidates
from phase1c_overlap_and_recovery import create_instance, REGIME_SIGMA


def wasserstein(x, y, R):
    a, b=linear_sum_assignment(cdist(x@R.T, y, "sqeuclidean"))
    D=cdist(x@R.T,y,"sqeuclidean")
    return float(D[a,b].mean())


def heldout_nn(d,R):
    ix=cKDTree(d["yh"]).query(d["xh"]@R.T,k=1)[1]
    return float(np.mean(d["true_x_heldout"]==d["true_y_heldout"][ix]))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--regimes",nargs="+",choices=list(REGIME_SIGMA),
                    default=["separated","mild_overlap","severe_overlap"])
    ap.add_argument("--instances",type=int,nargs="+",default=list(range(8)))
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    results={
        "status":"spectral_identifiability_audit_no_fit_or_hyperparameter_selection",
        "truth_never_used_to_pick_actual_initializer":True,
        "true_Rstar_posthoc_only":True,
        "settings":{"regimes":args.regimes,"instances":args.instances,
                    "train_candidate_sample_size":200,
                    "candidate_count":16},
        "instances":[]
    }
    for regime in args.regimes:
        for iid in args.instances:
            d=create_instance(regime,iid)
            X,Y=d["x"],d["y"]
            selector=np.random.default_rng(51000+iid)
            ix=selector.choice(len(X),200,replace=False)
            iy=selector.choice(len(Y),200,replace=False)
            cands=independent_covariance_sign_candidates(X,Y)
            assert len(cands)==16
            cost=np.array([wasserstein(X[ix],Y[iy],R) for signs,R in cands])
            err=np.array([np.linalg.norm(R-d["Rstar"],"fro") for signs,R in cands])
            labels=np.array([heldout_nn(d,R) for signs,R in cands])
            select=int(np.argmin(cost))
            best_error_idx=int(np.argmin(err)) # POSTHOC-only unattainable oracle
            gt_W=wasserstein(X[ix],Y[iy],d["Rstar"])
            eigx=np.linalg.eigvalsh(np.cov(X,rowvar=False))[::-1]
            eigy=np.linalg.eigvalsh(np.cov(Y,rowvar=False))[::-1]
            rel_gap_x=np.diff(eigx)*-1 / max(eigx[0],1e-12)
            rel_gap_y=np.diff(eigy)*-1 / max(eigy[0],1e-12)
            rec={
                "regime":regime,"instance_id":iid,
                "selected_sign_idx_by_unpaired_train_W2":select,
                "selected_train_W2":float(cost[select]),
                "actual_true_rotation_train_W2_POSTHOC":gt_W,
                "selected_Rstar_fro_error_POSTHOC":float(err[select]),
                "selected_heldout_true_label_NN_POSTHOC":float(labels[select]),
                "actual_Rstar_heldout_true_label_NN_POSTHOC":heldout_nn(d,d["Rstar"]),
                "best_Rstar_error_among_16_POSTHOC":float(err[best_error_idx]),
                "best_Rstar_error_sign_idx_POSTHOC":best_error_idx,
                "best_Rstar_error_sign_train_W2_POSTHOC":float(cost[best_error_idx]),
                "best_Rstar_error_sign_heldout_NN_POSTHOC":float(labels[best_error_idx]),
                "selected_vs_posthoc_best_Rstar_error_gap":float(
                    err[select]-err[best_error_idx]),
                "source_eigenvalues":eigx.tolist(),
                "target_eigenvalues":eigy.tolist(),
                "source_relative_eigengaps":rel_gap_x.tolist(),
                "target_relative_eigengaps":rel_gap_y.tolist(),
                "min_relative_eigengap_both":float(min(min(rel_gap_x),min(rel_gap_y))),
                "16_sign_train_W2":cost.tolist(),
                "16_sign_true_Rstar_Fro_error_POSTHOC":err.tolist(),
                "16_sign_heldout_true_label_NN_POSTHOC":labels.tolist(),
            }
            results["instances"].append(rec)
            print(f"SPECTRAL_AUDIT {regime} iid{iid}: "
                  f"selectRerr={err[select]:.4f} best16Rerr={err.min():.4f} "
                  f"sel_nn={labels[select]:.4f} trueR_nn={rec['actual_Rstar_heldout_true_label_NN_POSTHOC']:.4f} "
                  f"gap={rec['min_relative_eigengap_both']:.4f}",flush=True)
    (args.output/"RESULTS.json").write_text(
        json.dumps(results,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    lines=[
        "# Phase 1C spectral orientation identifiability audit",
        "",
        "**Diagnostic only:** the best-true-R candidate is an oracle never used to "
        "select a model; do not confuse this with unsupervised feasibility.",
        "The *actual* candidate is always selected by lowest TRAIN unpaired W2.",
        "",
        "| Geometry | N instances | Selected R error | Best-of-16 true-R error (POSTHOC) | "
        "Selected 1NN | True R* 1NN | Mean min eigengap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in args.regimes:
        sub=[v for v in results["instances"] if v["regime"]==mode]
        mean=lambda key:float(np.mean([r[key] for r in sub]))
        lines.append(f"| {mode} | {len(sub)} | "
                     f"{mean('selected_Rstar_fro_error_POSTHOC'):.4f} | "
                     f"{mean('best_Rstar_error_among_16_POSTHOC'):.4f} | "
                     f"{mean('selected_heldout_true_label_NN_POSTHOC'):.4f} | "
                     f"{mean('actual_Rstar_heldout_true_label_NN_POSTHOC'):.4f} | "
                     f"{mean('min_relative_eigengap_both'):.4f} |")
    lines+=["","Low selected candidate error need not imply the downstream PQR solver converges to the correct rotation; high minimum over all 16 signs implies even the available sign family is insufficient. Large discrepancy selected-vs-best sign implicates unpaired W2 selection, without proving the true transform identifiable.",""]
    (args.output/"REPORT.md").write_text("\n".join(lines),encoding="utf8")
    print("SPECTRAL_AUDIT_COMPLETE",len(results["instances"]),flush=True)

if __name__=="__main__":
    main()
