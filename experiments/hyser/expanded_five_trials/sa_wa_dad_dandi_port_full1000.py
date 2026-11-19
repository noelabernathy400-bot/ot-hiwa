#!/usr/bin/env python3
"""HySeR all-1000-train-window SA/WA/DAD-like comparisons.

DANDI code source (READ ONLY):
  projects/dandi-000138-hiwa-4d/code/closeout_dandi_sa_wa_dad_baselines.py
  at commit 38fb630001e11f5045a2a1055b2aa2686e7bfd34 on
  experiment/dandi-pca3-fa3-vs-pca4-fa4-20261119.

SA: exactly port the DANDI full-basis standardized-PCA formula.
WA: port the 200-step 100->0.1 entropic OT / orthogonal Procrustes
    annealing and random initialization; adapt 1000-point fitting pool to
    *stochastic* balanced 250-point minibatches. Every training point appears
    once per four iterations and 50 times over 200 iterations. This adaptation
    is disclosed, and is not identical to DANDI's 160-point full batch.
DAD-like: replicate DANDI's 3D symmetric kNN-density KL proxy, 128
    Euler-grid rotations/reflections and Powell max45. This is NOT the official
    DAD implementation; the DANDI method is ONLY defined for 3D.
No "5D DAD" result is invented.

All methods use the real trial-disjoint HySeR 1000 training points per
modality, 250 complete test points, and train-bank-only inductive retrieval.
No synchronized pair information is used in any fit. Five-finger true force
is exposed exclusively after each nearest-neighbour prediction is frozen.
"""
from __future__ import annotations
import argparse, json, math, sys, time, traceback
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from sklearn.decomposition import PCA

REPS=("PCA3_PCA3","FA3_FA3","PCA5_RAW5","FA5_RAW5")
WA_STEPS=200
WA_BATCH=250
WA_GAMMA_START=100.0
WA_GAMMA_MIN=0.1
WA_SINKHORN_MAX=1000

def polar(C):
    u,_,vh=np.linalg.svd(C,full_matrices=False)
    return u@vh

def sa_dandi_port(x,y):
    """Same row-vector formula as archived DANDI port."""
    def basis(z):
        std=np.std(z,axis=0,ddof=1)
        zz=(z-z.mean(axis=0))/np.where(std>0,std,1.)
        return PCA(n_components=z.shape[1],svd_solver="full").fit(zz).components_.T
    return (basis(x).T@basis(y)).T

def sinkhorn_dandi(a,b,cost,eps,maxiter=WA_SINKHORN_MAX,tol=1e-8):
    """Direct copy of the DANDI kernel-stabilized Sinkhorn structure, with
    early stopping and log-domain fallback. No fabricated zero-marginal rows."""
    a=np.asarray(a,float);b=np.asarray(b,float);cost=np.asarray(cost,float)
    assert eps>0 and cost.shape==(len(a),len(b)) and np.all(a>0) and np.all(b>0)
    shifted=cost-cost.min(axis=1,keepdims=True)
    shifted-=shifted.min(axis=0,keepdims=True)
    z=-shifted/eps
    kernel=np.exp(z)
    v=np.ones_like(b)
    for step in range(maxiter):
        with np.errstate(divide="ignore",invalid="ignore",over="ignore"):
            u=a/(kernel@v)
            v=b/(kernel.T@u)
        if not np.isfinite(v).all():
            break
        if (step+1)%10==0:
            if np.max(np.abs(u*(kernel@v)-a))<=tol:
                break
    else:
        step=maxiter-1
    with np.errstate(divide="ignore",invalid="ignore",over="ignore"):
        Q=(u[:,None]*kernel)*v[None,:]
    if np.isfinite(Q).all() and np.max(np.abs(Q.sum(1)-a))<2e-6 and np.max(np.abs(Q.sum(0)-b))<2e-6:
        return Q
    la,lb=np.log(a),np.log(b)
    vv=np.zeros_like(b)
    for step in range(maxiter):
        uu=la-logsumexp(z+vv[None,:],axis=1)
        vv=lb-logsumexp(z+uu[:,None],axis=0)
        if (step+1)%10==0:
            rows=np.exp(uu+logsumexp(z+vv[None,:],axis=1))
            if np.max(abs(rows-a))<=tol:break
    Q=np.exp(z+uu[:,None]+vv[None,:])
    return Q

def wa_dandi_port_stochastic1000(x,y,seed):
    assert x.shape==y.shape and x.shape[0]==1000
    d=x.shape[1]
    rng=np.random.default_rng(seed)
    # DANDI original author-port initialization: U of svd(random([d,d])).
    R=np.linalg.svd(rng.random((d,d)),full_matrices=False)[0]
    gamma=WA_GAMMA_START
    idsx=idsy=None
    max_marg=0.0
    last_delta=np.nan
    visited_x=np.zeros(1000,int)
    visited_y=np.zeros(1000,int)
    for it in range(WA_STEPS):
        if it%4==0:
            idsx=rng.permutation(1000)
            idsy=rng.permutation(1000)
        shift=(it%4)*WA_BATCH
        ix=idsx[shift:shift+WA_BATCH]
        iy=idsy[shift:shift+WA_BATCH]
        visited_x[ix]+=1;visited_y[iy]+=1
        xx=x[ix];yy=y[iy]
        if gamma>WA_GAMMA_MIN: gamma*=.95
        C=cdist(xx@R.T,yy,metric="sqeuclidean")
        Q=sinkhorn_dandi(np.full(WA_BATCH,1/WA_BATCH),
                        np.full(WA_BATCH,1/WA_BATCH),C,gamma)
        marg=max(float(np.max(abs(Q.sum(0)-1/WA_BATCH))),
                 float(np.max(abs(Q.sum(1)-1/WA_BATCH))))
        if not np.isfinite(Q).all() or marg>2e-5:
            raise ArithmeticError(f"WA step {it} marginal residual {marg}")
        max_marg=max(max_marg,marg)
        newR=polar(yy.T@Q.T@xx)
        last_delta=float(np.linalg.norm(newR-R))
        R=newR
    assert np.all(visited_x==50) and np.all(visited_y==50)
    return R,{"algorithm":"DANDI WA entropic alternating OT+Procrustes",
              "adaptation":"Stochastic all1000 pool: 250 batch/iteration, 4 batches/epoch, 50 epochs",
              "steps":WA_STEPS,"batch_size":WA_BATCH,
              "train_points_each_domain":1000,
              "each_train_point_visits":50,
              "gamma_start":WA_GAMMA_START,"gamma_final":gamma,
              "sinkhorn_max_steps":WA_SINKHORN_MAX,
              "max_batch_marginal_residual":max_marg,
              "last_rotation_delta":last_delta}

def euler_r(theta):
    a,b,c=theta
    sa,ca=np.sin(a),np.cos(a)
    sb,cb=np.sin(b),np.cos(b)
    sc,cc=np.sin(c),np.cos(c)
    rx=np.array([[1.,0,0],[0,ca,-sa],[0,sa,ca]])
    ry=np.array([[cb,0,sb],[0,1.,0],[-sb,0,cb]])
    rz=np.array([[cc,-sc,0],[sc,cc,0],[0,0,1.]])
    return rz@ry@rx

def dad_like_dandi_3d(x,y):
    """Explicitly *adapted* density-KL proxy; not original KordingLab DAD."""
    assert x.shape[1]==y.shape[1]==3 and x.shape[0]==y.shape[0]==1000
    k=5
    sx=np.maximum(cKDTree(x).query(x,k=k+1)[0][:,k],1e-8)
    sy=np.maximum(cKDTree(y).query(y,k=k+1)[0][:,k],1e-8)
    tree_y=cKDTree(y)
    mirrorR=np.diag([-1.,1.,1.])
    def score(theta,mirror):
        RR=euler_r(theta)@mirrorR if mirror else euler_r(theta)
        xx=x@RR.T
        xy=np.maximum(tree_y.query(xx,k=k)[0][:,k-1],1e-8)
        yx=np.maximum(cKDTree(xx).query(y,k=k)[0][:,k-1],1e-8)
        return float(3*np.mean(np.log(xy/sx))+3*np.mean(np.log(yx/sy)))
    angles=np.arange(4)*np.pi/2
    best=(float("inf"),None,None)
    for reflection in (False,True):
        for aa in angles:
            for bb in angles:
                for cc in angles:
                    theta=np.array([aa,bb,cc])
                    val=score(theta,reflection)
                    if val<best[0]:best=(val,theta,reflection)
    val,th,m=best
    opt=minimize(lambda z:score(z,m),th,method="Powell",
                 options={"maxiter":45,"xtol":1e-3,"ftol":2e-4})
    if float(opt.fun)<val:val=float(opt.fun);th=opt.x
    RR=euler_r(th)@mirrorR if m else euler_r(th)
    return RR,{"algorithm":"modified-DAD-like-3D DANDI density-KL proxy",
               "official_DAD_reproduction":False,
               "three_dimensional_only":True,
               "kNN":5,"coarse_grid":128,
               "powell_maxiter":45,"density_objective":val,
               "use_reflection":bool(m)}

def physical_score(Xtest,Ytrain,Ytest,Yphys_train,Yphys_test,R,eps_by_q):
    distances=cdist(Xtest@R.T,Ytrain)
    idx=np.argmin(distances,axis=1)
    pred=Yphys_train[idx]
    error=np.linalg.norm(pred-Yphys_test,axis=1)
    corr=pred-Yphys_test
    physical_r2=1-float(np.sum(corr*corr))/max(float(np.sum((Yphys_test-Yphys_train.mean(axis=0))**2)),1e-16)
    return {"physical_errors_250":error.tolist(),
            "mean_physical_l2":float(np.mean(error)),
            "median_physical_l2":float(np.median(error)),
            "physical_r2_train_mean_baseline":physical_r2,
            "hit_q05":float(np.mean(error<=eps_by_q["0.05"])),
            "hit_q20":float(np.mean(error<=eps_by_q["0.2"])),
            "hit_eps_0_325":float(np.mean(error<=.325)),
            "selected_target_train_indices_sha256":__import__("hashlib").sha256(idx.tobytes()).hexdigest(),
            "nearest_target_bank_is_training_only":True,
            "n_test":250}

def free_w2_heldout(Xtest,Ytest,R):
    C=cdist(Xtest@R.T,Ytest,metric="sqeuclidean")
    a,b=linear_sum_assignment(C)
    return float(np.mean(C[a,b]))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--features",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--representation",choices=REPS,required=True)
    parser.add_argument("--folds",type=int,nargs="+",default=list(range(1,6)))
    parser.add_argument("--wa-seeds",type=int,nargs="+",default=[42,43])
    parser.add_argument("--methods",nargs="+",choices=["SA","WA","DAD_like_3D"],
                        default=["SA","WA","DAD_like_3D"])
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    d=3 if args.representation.endswith(("PCA3","FA3")) else 5
    assert all(1<=i<=5 for i in args.folds)
    records=[]
    for fold in args.folds:
        filename=f"trial_heldout_{fold}"+("_5d" if d==5 else "")+".npz"
        with np.load(args.features/args.representation/filename) as z:
            Xtrain=np.asarray(z["X_train"],float)/np.sqrt(d)
            Ytrain=np.asarray(z["Y_train"],float)/np.sqrt(d)
            Xtest=np.asarray(z["X_test"],float)/np.sqrt(d)
            Ytest=np.asarray(z["Y_test"],float)/np.sqrt(d)
            Yphytrain=np.asarray(z["original_target_force_train"],float)
            Yphytest=np.asarray(z["original_target_force_test"],float)
            trial_ids=np.asarray(z["train_trial_id"],int)
        assert Xtrain.shape==Ytrain.shape==(1000,d)
        assert Xtest.shape==Ytest.shape==(250,d)
        assert Yphytrain.shape==(1000,5) and Yphytest.shape==(250,5)
        assert len(np.unique(trial_ids))==4 and all(np.sum(trial_ids==t)==250 for t in np.unique(trial_ids))
        # Same frozen training-only physical thresholds used for all HiWA runs.
        sourcejson=args.reference/("3D" if d==3 else "5D")/"full_physical_distance_curves.json"
        rs=json.loads(sourcejson.read_text())["records"]
        eps=next(x["train_only_thresholds"] for x in rs
                 if x["config"]==args.representation and x["fold"]==fold and x["method"]=="hard")
        for method in args.methods:
            if method=="DAD_like_3D" and d!=3:continue
            seeds=args.wa_seeds if method=="WA" else (42,)
            for seed in seeds:
                started=time.perf_counter()
                row={"representation":args.representation,"fold":fold,
                     "seed":seed,"method":method,
                     "source_train_n":1000,"target_train_n":1000,
                     "test_n":250,
                     "train_trial_ids":sorted([int(z) for z in np.unique(trial_ids)]),
                     "uses_training_cross_modal_true_pairs":False,
                     "uses_test_truth_in_fitting":False,
                     "trained_on_full_1000_pool":True,
                     "qualified":False,"error":None}
                try:
                    if method=="SA":
                        R=sa_dandi_port(Xtrain,Ytrain)
                        config={"algorithm":"DANDI SA full-basis standardized PCA port",
                                "deterministic_per_fold":True}
                    elif method=="WA":
                        R,config=wa_dandi_port_stochastic1000(Xtrain,Ytrain,seed)
                    else:
                        R,config=dad_like_dandi_3d(Xtrain,Ytrain)
                    if not np.isfinite(R).all() or R.shape!=(d,d):raise ArithmeticError("bad rotation")
                    orth=float(np.linalg.norm(R.T@R-np.eye(d)))
                    if orth>1e-6:raise ArithmeticError("non-orthogonal baseline rotation")
                    score=physical_score(Xtest,Ytrain,Ytest,Yphytrain,Yphytest,R,eps)
                    row.update({"R":R.tolist(),"config":config,
                                "orthogonality_error":orth,
                                "heldout_unpaired_W2sq":free_w2_heldout(Xtest,Ytest,R),
                                **score,"qualified":True})
                except Exception as ex:
                    row["error"]=f"{type(ex).__name__}: {ex}"
                    row["traceback"]=traceback.format_exc(limit=2)
                row["runtime_seconds"]=time.perf_counter()-started
                records.append(row)
                print("HYSER_DANDI_BASELINE",args.representation,fold,method,seed,
                      row["qualified"],row.get("hit_q05"),
                      "time",round(row["runtime_seconds"],2),flush=True)
                # partial atomic write per run prevents losing the complete record
                tmp=args.out/"runs_partial.json.tmp"
                tmp.write_text(json.dumps(records,ensure_ascii=False,indent=2)+"\n")
                tmp.replace(args.out/"runs.json")
    out={"representation":args.representation,"n_records":len(records),
         "dandi_reference":"projects/dandi-000138-hiwa-4d/code/closeout_dandi_sa_wa_dad_baselines.py at 38fb630001e11f5045a2a1055b2aa2686e7bfd34",
         "full1000_train_data":True,
         "wa_dandi_stochastic_adaptation_not_identical_full_batch":True,
         "not_official_DAD":True,
         "all_results_including_failures":records}
    (args.out/"runs_bundle.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
    print("HYSER_REP_COMPLETED",args.representation,len(records),
          sum(x["qualified"] for x in records),flush=True)

if __name__=="__main__":
    main()
