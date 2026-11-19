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
from scipy.special import softmax,logsumexp

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2] # expanded_five_trials -> hyser -> experiments -> repo
sys.path.insert(0,str(REPO))
import src.cc_hiwa.soft_hiwa as sh
from src.cc_hiwa.soft_hiwa import SoftHiWA
from rectangular_group_ot import group_entropic_transport
from heldout_best2d_and_nn import train_radius,evaluate,random_baseline
import run_fivefinger_5d_K4_K5 as pilot
from run_pca_fa_ot_ablation import physical_eval

# Equivalent cost + entropy + group marginals, numerically stable dual solution.
# Scoped to this script; DOES NOT change the repository-wide solver behavior.
# The original direct kernel scaling may fail because exp(-cost/gamma) underflows.
def stable_group_P(C,gamma,maxiter):
    n,m=C.shape
    alpha=np.full(n,1./n)
    beta=np.full(m,1./m)
    return group_entropic_transport(C,alpha,beta,eps=gamma,tol=1e-9)


# Exact same entropic conditional OT objective and gamma as old kernel scaling.
# Use the original fast solver whenever it satisfies marginal constraints;
# on pathological/imbalanced weights recompute with log-domain Sinkhorn
# without EPS-flooring denominators or violating zero-probability support.
_old_conditional_sinkhorn=sh.sinkhorn_weighted

def stable_conditional_Q(p,q,X,Y,gamma,maxiter,extra_cost=None,extra_cost_weight=0.):
    pa=np.asarray(p,float).ravel();qb=np.asarray(q,float).ravel()
    pa=pa/pa.sum();qb=qb/qb.sum()
    # Fast legacy route: identical objective and exact parameters.
    try:
        plan,cost,err=_old_conditional_sinkhorn(
           pa,qb,X,Y,gamma,maxiter,
           extra_cost=extra_cost,extra_cost_weight=extra_cost_weight)
        if np.isfinite(err) and err<=1e-8 and np.isfinite(plan).all():
            return plan,cost,err
    except (ArithmeticError,FloatingPointError,ValueError,ZeroDivisionError):
        pass
    active_x=np.flatnonzero(pa>0)
    active_y=np.flatnonzero(qb>0)
    if not len(active_x) or not len(active_y):
        raise ArithmeticError("Empty positive point support in conditional Sinkhorn")
    xx=np.asarray(X,float)[:,active_x]
    yy=np.asarray(Y,float)[:,active_y]
    ca=np.maximum(
        np.sum(xx**2,axis=0)[:,None]+
        np.sum(yy**2,axis=0)[None,:]-
        2*(xx.T@yy),0.)
    if extra_cost is not None and extra_cost_weight>0:
        ca=ca+extra_cost_weight*np.asarray(extra_cost,float)[np.ix_(active_x,active_y)]
    temperature=max(float(gamma),1e-8)
    log_kernel=-(ca-ca.min())/temperature
    loga=np.log(pa[active_x])
    logb=np.log(qb[active_y])
    logv=np.zeros(len(active_y))
    logu=np.zeros(len(active_x))
    # Iteration cap is a NUMERICAL accuracy safeguard, not a new regularizer.
    # For difficult Q it may exceed the original fast 80 scaling iterations.
    max_adaptive=max(400,20*int(maxiter))
    err=float("inf")
    for step in range(max_adaptive):
        logu=loga-logsumexp(log_kernel+logv[None,:],axis=1)
        logv=logb-logsumexp(log_kernel+logu[:,None],axis=0)
        if step%25==24 or step==max_adaptive-1:
            row=np.exp(logu+logsumexp(log_kernel+logv[None,:],axis=1))
            col=np.exp(logv+logsumexp(log_kernel+logu[:,None],axis=0))
            err=max(float(np.max(np.abs(row-pa[active_x]))),
                    float(np.max(np.abs(col-qb[active_y]))))
            if err<=1e-8:break
    logQ=logkernel_prob=log_kernel+logu[:,None]+logv[None,:]
    active_Q=np.exp(np.clip(logQ,-745.,700.))
    # Preserve real positive support and exact zero-mass entries.
    Q=np.zeros((len(pa),len(qb)),dtype=float)
    Q[np.ix_(active_x,active_y)]=active_Q
    err=max(float(np.max(np.abs(Q.sum(axis=1)-pa))),
            float(np.max(np.abs(Q.sum(axis=0)-qb))))
    C=np.maximum(
        np.sum(np.asarray(X,float)**2,axis=0)[:,None]+
        np.sum(np.asarray(Y,float)**2,axis=0)[None,:]-
        2*(np.asarray(X,float).T@np.asarray(Y,float)),0.)
    if extra_cost is not None and extra_cost_weight>0:
        C=C+extra_cost_weight*np.asarray(extra_cost,float)
    return Q,float(np.sum(C*Q)),err

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

def inductive_train_bank_retrieval(Xtest,Ytrain,Ytrue,R,radius):
    """Inductive target lookup: candidates are ONLY the 1000 training-force points.

    The true heldout Y_i is used only AFTER nearest-neighbor selection.
    This eliminates transductive access to the fifth trial target distribution.
    """
    dist=cdist(Xtest@R.T,Ytrain,metric="euclidean")
    ind=np.argmin(dist,axis=1)
    geometry=dist[np.arange(len(Xtest)),ind]
    error=np.linalg.norm(Ytrain[ind]-Ytrue,axis=1)
    random=np.random.default_rng(2054)
    chance=[]
    for _ in range(250):
        idx=random.integers(0,len(Ytrain),size=len(Ytrue))
        chance.append(float(np.mean(np.linalg.norm(Ytrain[idx]-Ytrue,axis=1)<=radius)))
    return {
        "train_bank_NN_geometric_coverage":float(np.mean(geometry<=radius)),
        "train_bank_NN_geometric_distance_mean":float(np.mean(geometry)),
        "train_bank_NN_physically_correct":float(np.mean(error<=radius)),
        "train_bank_NN_physical_RMSE_in_whitened_units":float(np.sqrt(np.mean((Ytrain[ind]-Ytrue)**2))),
        "train_bank_NN_physical_error_mean":float(np.mean(error)),
        "train_bank_random_hit_mean":float(np.mean(chance)),
        "train_bank_random_hit_std":float(np.std(chance)),
        "target_test_pool_not_used_for_retrieval":True
    }

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
    p.add_argument("--full-training",action="store_true",
                   help="Fit on all 1000 train windows in both domains, independently permuted.")
    p.add_argument("--numerically-stable-Q",action="store_true",
                   help="Use mathematically identical entropic conditional-Q log-Sinkhorn fallback when legacy scaling violates marginals.")
    p.add_argument("--exact-positive-support",action="store_true",
                   help="Exact zero-mass removal for each Q block, same mathematical OT, applied to BOTH Hard and Soft.")
    p.add_argument("--warm-start-local",action="store_true",
                   help="Opt-in common numerical warm start for Hard & Soft (sensitivity, not historical default).")
    p.add_argument("--mu",type=float,default=.005,
                   help="ADMM penalty; .005 is historical default.")
    p.add_argument("--only-methods",nargs="+",choices=["hard","soft05"],
                   default=["hard","soft05"])
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    sh.sinkhorn_groups=stable_group_P
    if args.numerically_stable_Q:
        sh.sinkhorn_weighted=stable_conditional_Q
    rows=[]
    for conf in args.configs:
        for fold in args.folds:
            with np.load(args.features/conf/f"trial_heldout_{fold}_5d.npz") as f:
                Xtr=np.asarray(f["X_train"],float)
                Ytr=np.asarray(f["Y_train"],float)
                Xte=np.asarray(f["X_test"],float)/np.sqrt(5)
                Yte=np.asarray(f["Y_test"],float)/np.sqrt(5)
                traintrials=np.asarray(f["train_trial_id"],int)
                Yphys_train=np.asarray(f["original_target_force_train"],float)
                Yphys_test=np.asarray(f["original_target_force_test"],float)
            assert Xtr.shape==Ytr.shape==(1000,5)
            assert Xte.shape==Yte.shape==(250,5)
            # Different subsamples by domain: no synchronized pair ordering
            if args.full_training:
                # Independent within-domain permutations make any temporal
                # index-pairing impossible for the unsupervised fit.
                iX=np.random.default_rng(512).permutation(len(Xtr))
                iY=np.random.default_rng(2048).permutation(len(Ytr))
            else:
                iX=choose_indices(traintrials,512)
                iY=choose_indices(traintrials,2048)
            x=Xtr[iX];y=Ytr[iY]
            nfit=len(x)
            rad=train_radius(Ytr/np.sqrt(5),traintrials)
            rr=random_baseline(Yte,rad["radius"])
            for seed in args.seeds:
                for mode in args.only_methods:
                    aa,bb=per_method_memberships(x,y,mode)
                    params=dict(PARAMS)
                    params.update(random_state=seed,maxiter=args.maxiter,
                                  sa_maxiter=args.sa_maxiter,
                                  warm_start_local=args.warm_start_local,
                                  support_mode=("positive" if args.exact_positive_support else "full"),
                                  mu=args.mu)
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
                    w2sq=pilot.w2(Xte,Yte,R)
                    physical=physical_eval(Xte,Ytr/np.sqrt(D),
                        Yphys_train,Yphys_test,R)
                    inductive=inductive_train_bank_retrieval(
                        Xte,Ytr/np.sqrt(5),Yte,R,rad["radius"])
                    diagnostic=model.diagnostics
                    # Local matrix and Q are all actually learned, not skipped
                    q=model.local_couplings
                    assert len(q)==K and all(len(row)==K for row in q)
                    qstats=[]
                    for i in range(K):
                        for j in range(K):
                            arr=np.asarray(q[i][j],float)
                            expected_x=int(np.count_nonzero(aa[:,i])) if args.exact_positive_support else nfit
                            expected_y=int(np.count_nonzero(bb[:,j])) if args.exact_positive_support else nfit
                            if arr.shape!=(expected_x,expected_y) or not np.isfinite(arr).all():
                                raise ValueError(f"Unexpected exact local-Q block shape {arr.shape} vs {(expected_x,expected_y)}")
                            qstats.append({"i":i,"j":j,
                                 "shape":[expected_x,expected_y],"total_mass":float(arr.sum()),
                                 "nonzero_fraction":float(np.mean(arr>1e-14)),
                                 "max_entry":float(np.max(arr))})
                    row={
                        "config":conf,"heldout_trial":fold,"seed":seed,"method":mode,
                        "fitting":("Full 1000 unpaired TRAINING points each domain; exact conditional-support local Q, no anchors"
                                   if args.full_training else "250 preselected unpaired training points; exact local Q, no anchors"),
                        "n_full_representation_training":1000,
                        "n_OT_train_source":nfit,"n_OT_train_target":nfit,
                        "n_heldout_test":250,
                        "n_train_per_trial_chosen":([250]*4 if args.full_training else [63,63,63,61]),
                        "sampling_indices_selected_before_model":[iX.tolist(),iY.tolist()],
                        "source_and_target_subsample_independent":True,
                        "K":K,"dims":5,"params":params,"R":R.tolist(),"P":P.tolist(),
                        "q_stats":qstats,
                        "local_rotations_Rkl":model.local_rotations.tolist(),
                        "ADMM_dual_multipliers_final":model.admm_dual_multipliers.tolist(),
                        "membership_source_mean_entropy":float(np.mean(-np.sum(aa*np.log(np.maximum(aa,1e-300)),axis=1))),
                        "membership_target_mean_entropy":float(np.mean(-np.sum(bb*np.log(np.maximum(bb,1e-300)),axis=1))),
                        "source_mixed_fraction_max_membership_below_0p8":float(np.mean(aa.max(axis=1)<.8)),
                        "target_mixed_fraction_max_membership_below_0p8":float(np.mean(bb.max(axis=1)<.8)),
                        "P_mean_row_max":float(np.mean(P.max(axis=1))),
                        "ADMM_iteration_history":{
                            "rotation_change":diagnostic["Rg_norm"].tolist(),
                            "primal_residual":diagnostic["admm_primal_residual"].tolist(),
                            "dual_residual":diagnostic["admm_dual_residual"].tolist(),
                            "transport_objective":diagnostic["transport_objective_history"].tolist(),
                            "local_Q_marginal_error":diagnostic["max_sinkhorn_marginal_error"].tolist()
                        },
                        "local_R_matrices_learned":True,
                        "full_Q_matrices_learned_for_each_16_pairs":True,
                        "numeric_convergence_ADMM_primal_and_rotation":False,
                        "exact_nonzero_support_no_mass_removed":bool(args.exact_positive_support),
                        "stable_conditional_Q_solver_if_needed":bool(args.numerically_stable_Q),
                        "diagnostics":{
                           name:stats_or_null(diagnostic,name)
                           for name in ("Rg_norm","admm_primal_residual",
                             "admm_dual_residual","transport_objective_history",
                             "max_sinkhorn_marginal_error")},
                        "nearest":pred,
                        "heldout_W2_squared":w2sq,
                        "heldout_physical_force_prediction_POSTHOC":physical,
                        "train_bank_inductive_nearest":inductive,
                        "mean_threshold":rad["radius"],
                        "threshold_training_calibration":rad,
                        "random_baseline":rr,
                        "true_test_correspondence_held_out_from_training":True,
                    }
                    dpr=row["diagnostics"].get("admm_primal_residual") or {}
                    dr=row["diagnostics"].get("Rg_norm") or {}
                    row["numeric_convergence_ADMM_primal_and_rotation"]=(
                        dpr.get("last",float("inf"))<=.1 and
                        dr.get("last",float("inf"))<=.1)
                    # Verify ALL local conditional OT row/column marginals, not
                    # only P. A numerical failure is recorded in row and NEVER
                    # silently called a converged run.
                    row["max_final_Q_marginal_error"] = float(
                        diagnostic["max_sinkhorn_marginal_error"][-1])
                    row["local_rotation_consensus_max_error"]=float(
                        np.linalg.norm(model.local_rotations -
                            R[:,:,None,None],axis=(0,1)).max())
                    row["max_Rkl_orthogonality_error"]=float(max(
                        np.linalg.norm(model.local_rotations[:,:,i,j].T @
                                       model.local_rotations[:,:,i,j]-np.eye(D))
                        for i in range(K) for j in range(K)))
                    row["Q_constraints_passed"]=(
                        row["max_final_Q_marginal_error"]<=1e-5 and
                        all(abs(t["total_mass"]-1)<1e-6 for t in qstats))
                    row["numeric_convergence_ADMM_primal_and_rotation"] &= bool(
                        row["Q_constraints_passed"])
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
      json.dumps({"experiment":("Five-trial heldout true 1000 points/domain full dense P,Q,Rkl ADMM"
                                 if args.full_training else
                                 "Five-trial heldout P,Q,Rkl ADMM with 250 TRAINING points/domain"),
       "not_full_1000_point_OT":not args.full_training,
       "different_from_prior_96_anchor_global_R":True,
       "source_and_target_sampled_independently":True,
       "local_solver":"repo.src.cc_hiwa.soft_hiwa.SoftHiWA full support with hard one-hot vs soft K-means",
       "numerical_P_solver":"stable dual-root/log-Sinkhorn for same entropic group OT objective and same gamma; original direct-kernel solver underflowed",
       "exact_zero_mass_support_mode":args.exact_positive_support,
       "numerically_stable_conditional_Q":args.numerically_stable_Q,
       "physical_correspondence_only_evaluation":True,
       "runs":rows},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    summary=[]
    for conf in args.configs:
        for meth in args.only_methods:
            rr=[r for r in rows if r["config"]==conf and r["method"]==meth]
            summary.append({"config":conf,"method":meth,"n_runs":len(rr),
              "coverage":float(np.mean([r["nearest"]["target_nn_coverage_fraction"] for r in rr])),
              "physical_hit":float(np.mean([r["nearest"]["nn_retrieval_physical_neighbor_fraction"] for r in rr])),
              "exact_index":float(np.mean([r["nearest"]["nn_retrieval_exact_time_index_fraction"] for r in rr])),
              "train_bank_physical_hit":float(np.mean([r["train_bank_inductive_nearest"]["train_bank_NN_physically_correct"] for r in rr])),
              "train_bank_random_hit":float(np.mean([r["train_bank_inductive_nearest"]["train_bank_random_hit_mean"] for r in rr])),
              "test_W2_squared":float(np.mean([r["heldout_W2_squared"] for r in rr])),
              "test_physical_force_RMSE":float(np.mean([r["heldout_physical_force_prediction_POSTHOC"]["paired_force_RMSE_EVAL_ONLY"] for r in rr])),
              "test_physical_force_R2":float(np.mean([r["heldout_physical_force_prediction_POSTHOC"]["paired_force_R2_vs_trainmean_EVAL_ONLY"] for r in rr])),
              "random_hit":float(np.mean([r["random_baseline"]["random_target_assignment_hit_mean"] for r in rr])),
              "last_primal":float(np.mean([(r["diagnostics"].get("admm_primal_residual") or {}).get("last",float("nan")) for r in rr])),
              "converged_count":sum(r["numeric_convergence_ADMM_primal_and_rotation"] for r in rr),
              "last_group_marginal":float(np.mean([max(np.abs(np.sum(r["P"],axis=0)-.25).max(),
                                                          np.abs(np.sum(r["P"],axis=1)-.25).max()) for r in rr]))
            })
    (args.out/"local_ADMM_summary.json").write_text(
       json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    report=["# Hyser local P/Q/Rkl ADMM vs 96-anchor global-R pilot",
       "",
       "This code runs the existing repo src.cc_hiwa.soft_hiwa.SoftHiWA with full Q for each of 16 local group pairs, local orthogonal R_kl, group P and ADMM consensus global Rg.",
       "HARD group assignment is one-hot, SOFT group assignment is independent temperature=.5 soft Kmeans; all P/Q/entropy/consensus parameters are otherwise identical.",
       ("**FULL TRAINING**: 1000 training points/domain per fold, independent domain permutations, 16 full local Q matrices each 1000x1000, no anchors, no test pairs in model fit."
        if args.full_training else
        "**SUBSAMPLE**: only 250 training points/domain per fold, full Q on subsample, NOT the 1000-point formal fit."),
       "Holdout is the entire remaining 250 windows, never used to train PCA/FA/ADMM.",
       "Nearest target accuracy includes BOTH transductive 250-target heldout pool retrieval and INDUCTIVE 1000-point training-target bank retrieval; labels are used only AFTER nearest selection to judge physical correctness. Train-only 5% inter-trial quantile radius; geometry coverage is not physical correctness.",
       "",
       "|Feature|Method|n|TEST-cloud coverage|TEST-cloud physical hit|TEST-cloud random hit|TRAIN-bank physical hit|TRAIN-bank random hit|W2²|physical RMSE|exact time index|",
       "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        report.append(f"|{r['config']}|{r['method']}|{r['n_runs']}|{r['coverage']:.3%}|{r['physical_hit']:.3%}|{r['random_hit']:.3%}|{r['train_bank_physical_hit']:.3%}|{r['train_bank_random_hit']:.3%}|{r['test_W2_squared']:.5f}|{r['test_physical_force_RMSE']:.5f}|{r['exact_index']:.3%}|")
    report+=["",f"Solver initialization: warm_start_local={args.warm_start_local}; ADMM mu={args.mu}; outer limit={args.maxiter}; per-pair inner limit={args.sa_maxiter}; exact positive mass support={args.exact_positive_support}; stable conditional log-Sinkhorn fallback={args.numerically_stable_Q}.",
        "**Do not interpret a local-ADMM run with nonconverged primal residual as a successfully converged full HiWA model.** The numerical history, P/Q block statistics and group marginal errors are saved for review.",
      "No inherited source cluster labels have been given to target. No synchronized row pair enters OT fitting."]
    (args.out/"README_ZH.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(json.dumps({"status":"success","n_runs":len(rows),"summary":summary},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
