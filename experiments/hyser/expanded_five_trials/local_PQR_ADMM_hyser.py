#!/usr/bin/env python3
"""FULL LOCAL R_kl / Q_kl / P / ADMM comparison, NOT global-R shortcut.

Uses the repository's existing src.cc_hiwa.soft_hiwa.SoftHiWA, with all
auxiliary objectives set to zero, identical solver hyperparameters, and
full-support local Q for BOTH hard one-hot and soft memberships. Therefore
this is an exact full-Q local-ADMM fit on a stratified REAL TRAINING SUBSAMPLE,
not the prior 96-anchor global-R approximation and NOT a full 1000-point fit.

Train-only 4 trials -> predetermined 250 unpaired training samples each domain,
test the 250 intact fifth trial. Source and target sample selection independent.
Full 1000 training points still fitted the PCA5 or FA5 representation prior.
No target aligned training row labels or synchronized matches enter training.
"""
from __future__ import annotations
import argparse,json,csv,sys
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from scipy.special import softmax

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2] # expanded_five_trials -> hyser -> experiments -> repo
sys.path.insert(0,str(REPO))
from src.cc_hiwa.soft_hiwa import SoftHiWA
from heldout_best2d_and_nn import train_radius,evaluate,random_baseline

K=4
D=5
N_PER_TRAIN_TRIAL=63 # with four training trials: three 63 + one 61 = 250 total
PARAMS=dict(
    normalize=False,
    maxiter=60, tol=.1, mu=.005,
    shorn_maxiter=300,shorn_gamma=.2,
    sa_maxiter=40,sa_tol=.01,
    sa_shorn_maxiter=80,sa_shorn_gamma=.1,
    support_mode="full",warm_start_local=False,
    rotation_anchor_weight=0.0,
    determinant_sign=None,
    representative_guidance_weight=0.0,
    representative_rotation_weight=0.0,
    component_conditioning_weight=0.0,
    temporal_signature_weight=0.0,
    consensus_weighting="uniform",
    inner_entropy_mode="legacy_inverse_group_mass",
    rotation_structure="full",
)

def choose_indices(tids,seed):
    idx=[]
    unique=sorted(np.unique(tids))
    assert len(unique)==4
    for j,t in enumerate(unique):
        pool=np.flatnonzero(tids==t)
        assert len(pool)==250
        rng=np.random.default_rng(seed+1000*int(t))
        n=63 if j<3 else 61
        idx.extend(rng.choice(pool,n,replace=False).tolist())
    assert len(idx)==250 and len(set(idx))==250
    return np.sort(idx)

def per_method_memberships(x,y,mode):
    # independent clustering, no synchronized Y labels inherited from X
    mx=KMeans(n_clusters=K,random_state=42,n_init=30).fit(x/np.sqrt(D))
    my=KMeans(n_clusters=K,random_state=43,n_init=30).fit(y/np.sqrt(D))
    if mode=="hard":
        A=np.eye(K)[mx.labels_];B=np.eye(K)[my.labels_]
    elif mode=="soft05":
        aa=cdist(x/np.sqrt(D),mx.cluster_centers_,metric="sqeuclidean")
        bb=cdist(y/np.sqrt(D),my.cluster_centers_,metric="sqeuclidean")
        # NOTE centers fitted to x/sqrt5, so distances must be on same scale
        A=softmax(-aa/.5,axis=1);B=softmax(-bb/.5,axis=1)
    else:raise ValueError(mode)
    return A,B

def stats_or_null(diag,name):
    v=diag.get(name)
    if v is None:return None
    a=np.asarray(v,float)
    if a.size==0:return None
    return {"first":float(a.flat[0]),"last":float(a.flat[-1]),
            "max":float(np.max(a)),"min":float(np.min(a)),"length":int(a.size)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--features",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--folds",type=int,nargs="+",default=[1,2,3,4,5])
    p.add_argument("--seeds",type=int,nargs="+",default=[42,43])
    p.add_argument("--configs",nargs="+",choices=["PCA5_RAW5","FA5_RAW5"],
                   default=["PCA5_RAW5","FA5_RAW5"])
    p.add_argument("--maxiter",type=int,default=60)
    p.add_argument("--sa-maxiter",type=int,default=40)
    p.add_argument("--save-first-Q",action="store_true")
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for conf in args.configs:
        for fold in args.folds:
            with np.load(args.features/conf/f"trial_heldout_{fold}_5d.npz") as f:
                Xtr=np.asarray(f["X_train"],float)
                Ytr=np.asarray(f["Y_train"],float)
                Xte=np.asarray(f["X_test"],float)/np.sqrt(5)
                Yte=np.asarray(f["Y_test"],float)/np.sqrt(5)
                traintrials=np.asarray(f["train_trial_id"],int)
            assert Xtr.shape==Ytr.shape==(1000,5)
            assert Xte.shape==Yte.shape==(250,5)
            # Different subsamples by domain: no synchronized pair ordering
            iX=choose_indices(traintrials,512)
            iY=choose_indices(traintrials,2048)
            x=Xtr[iX];y=Ytr[iY]
            rad=train_radius(Ytr/np.sqrt(5),traintrials)
            rr=random_baseline(Yte,rad["radius"])
            for seed in args.seeds:
                for mode in ("hard","soft05"):
                    aa,bb=per_method_memberships(x,y,mode)
                    params=dict(PARAMS)
                    params.update(random_state=seed,maxiter=args.maxiter,
                                  sa_maxiter=args.sa_maxiter)
                    model=SoftHiWA(**params).fit(x,aa,y,bb,
                              X_transform=np.eye(5),Y_transform=np.eye(5))
                    R=np.asarray(model.Rg,float)
                    P=np.asarray(model.P,float)
                    if R.shape!=(5,5) or P.shape!=(4,4):
                        raise ValueError("Bad model shapes")
                    if np.max(abs(P.sum(0)-.25))>1e-5 or np.max(abs(P.sum(1)-.25))>1e-5:
                        raise ArithmeticError("P marginal constraint failure")
                    if np.linalg.norm(R.T@R-np.eye(5))>1e-8:
                        raise ArithmeticError("R violates orthogonality")
                    pred=evaluate(Xte,Yte,R,rad["radius"])
                    diagnostic=model.diagnostics
                    # Local matrix and Q are all actually learned, not skipped
                    q=model.local_couplings
                    assert len(q)==K and all(len(row)==K for row in q)
                    qstats=[]
                    for i in range(K):
                        for j in range(K):
                            arr=np.asarray(q[i][j],float)
                            if arr.shape!=(250,250) or not np.isfinite(arr).all():
                                raise ValueError("Unexpected full-Q local block")
                            qstats.append({"i":i,"j":j,
                                 "shape":[250,250],"total_mass":float(arr.sum()),
                                 "nonzero_fraction":float(np.mean(arr>1e-14)),
                                 "max_entry":float(np.max(arr))})
                    row={
                        "config":conf,"heldout_trial":fold,"seed":seed,"method":mode,
                        "fitting":"local-PQR-ADMM full point-support on 250 randomly preselected TRAINING points, no anchors",
                        "n_full_representation_training":1000,
                        "n_OT_train_source":250,"n_OT_train_target":250,
                        "n_heldout_test":250,
                        "n_train_per_trial_chosen":[63,63,63,61],
                        "sampling_indices_selected_before_model":[iX.tolist(),iY.tolist()],
                        "source_and_target_subsample_independent":True,
                        "K":K,"dims":5,"params":params,"R":R.tolist(),"P":P.tolist(),
                        "q_stats":qstats,
                        "local_R_matrices_learned":True,
                        "full_Q_matrices_learned_for_each_16_pairs":True,
                        "diagnostics":{
                           name:stats_or_null(diagnostic,name)
                           for name in ("Rg_norm","admm_primal_residual",
                             "admm_dual_residual","transport_objective",
                             "group_sinkhorn_marginal_error","max_sinkhorn_marginal_error")},
                        "nearest":pred,
                        "mean_threshold":rad["radius"],
                        "threshold_training_calibration":rad,
                        "random_baseline":rr,
                        "true_test_correspondence_held_out_from_training":True,
                    }
                    rows.append(row)
                    (args.out/"partial_results.json").write_text(
                        json.dumps({"status":"PARTIAL","runs":rows},ensure_ascii=False,indent=2)+"\n",
                        encoding="utf-8")
                    if (args.save_first_Q and fold==1 and seed==42):
                        np.savez_compressed(args.out/f"{conf}_{mode}_fold1_seed42_original_ADMM_Q.npz",
                        **{f"Q_{i}_{j}":np.asarray(q[i][j]) for i in range(K) for j in range(K)},
                        R=R,P=P)
                    print("DONE",conf,fold,seed,mode,
                      "NN physical hit",round(pred["nn_retrieval_physical_neighbor_fraction"],5),
                      "coverage",round(pred["target_nn_coverage_fraction"],5),
                      "P error",max(abs(P.sum(0)-.25).max(),abs(P.sum(1)-.25).max()),
                      "last delta",row["diagnostics"]["Rg_norm"],flush=True)
    (args.out/"local_ADMM_results.json").write_text(
      json.dumps({"experiment":"Five-trial heldout full P,Q,Rkl ADMM, 250 TRAINING samples per domain subsampled from 1000",
       "not_full_1000_point_OT":True,
       "different_from_prior_96_anchor_global_R":True,
       "source_and_target_sampled_independently":True,
       "local_solver":"repo.src.cc_hiwa.soft_hiwa.SoftHiWA full support with hard one-hot vs soft K-means",
       "physical_correspondence_only_evaluation":True,
       "runs":rows},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    summary=[]
    for conf in args.configs:
        for meth in ("hard","soft05"):
            rr=[r for r in rows if r["config"]==conf and r["method"]==meth]
            summary.append({"config":conf,"method":meth,"n_runs":len(rr),
              "coverage":float(np.mean([r["nearest"]["target_nn_coverage_fraction"] for r in rr])),
              "physical_hit":float(np.mean([r["nearest"]["nn_retrieval_physical_neighbor_fraction"] for r in rr])),
              "exact_index":float(np.mean([r["nearest"]["nn_retrieval_exact_time_index_fraction"] for r in rr])),
              "random_hit":float(np.mean([r["random_baseline"]["random_target_assignment_hit_mean"] for r in rr])),
              "last_primal":float(np.mean([r["diagnostics"]["admm_primal_residual"]["last"] for r in rr])),
              "last_group_marginal":float(np.mean([r["diagnostics"]["group_sinkhorn_marginal_error"]["last"] for r in rr]))
            })
    (args.out/"local_ADMM_summary.json").write_text(
       json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    report=["# Hyser local P/Q/Rkl ADMM vs 96-anchor global-R pilot",
       "",
       "This code runs the existing repo src.cc_hiwa.soft_hiwa.SoftHiWA with full Q for each of 16 local group pairs, local orthogonal R_kl, group P and ADMM consensus global Rg.",
       "HARD group assignment is one-hot, SOFT group assignment is independent temperature=.5 soft Kmeans; all P/Q/entropy/consensus parameters are otherwise identical.",
       "**Scope limit**: each fold's complete 1000 training points build PCA5/FA5 representations and 4 group K-means is fitted on **250 real stratified training points per domain**, selected independently from each of four training trials (63+63+63+61). The fit fully solves all local Q blocks on those 250 points, not on 1000 original training points; no 96-anchor approximation is used. This is a scientifically honest small-data full-PQR-ADMM CONTROL, NOT claimed as full 1000-point original HiWA run.",
       "Holdout is the entire remaining 250 windows, never used to train PCA/FA/ADMM.",
       "Nearest target accuracy is based on true synchronized heldout Y, and train-only 5% inter-trial quantile radius. Coverage is not physical correctness.",
       "",
       "|Feature|Method|n|Nearest coverage|physical-neighbor hit|random neighbor hit|exact time index|",
       "|---|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        report.append(f"|{r['config']}|{r['method']}|{r['n_runs']}|{r['coverage']:.3%}|{r['physical_hit']:.3%}|{r['random_hit']:.3%}|{r['exact_index']:.3%}|")
    report+=["","**Do not interpret a local-ADMM run with nonconverged primal residual as a successfully converged full HiWA model.** The numerical history, P/Q block statistics and group marginal errors are saved for review.",
      "No inherited source cluster labels have been given to target. No synchronized row pair enters OT fitting."]
    (args.out/"README_ZH.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(json.dumps({"status":"success","n_runs":len(rows),"summary":summary},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
