#!/usr/bin/env python3
"""PRE-REGISTERED nested/inner validation PARAMETER PILOT (not outer-test tuning).

On real Hyser S01/s1 five trials: outer trial 1 excluded ENTIRELY; inner trial
2 as validation; trials 3,4,5 supply 750 training windows. PCA/FA/scaling are
fitted ONLY on these 750 inner training points, never on inner validation
or outer testing.

Each candidate fits exact local P/Q/Rkl+ADMM on 250 TRAINING samples/domain
independently sampled from the 750 inner-train windows; validates on all 250
inner-validation windows against raw 5-finger physical force.
This is a low-cost COMPUTE preflight, not full750 x four-innerfold nested
selection, and NOT an outer trial performance result.

Fixed candidate grid per feature: 2 K x 3 tau x 3 mu = 18 Soft; 2 K x 3 mu=6
Hard controls; same gamma1=.2, gamma2=.1/P, local rotation/ADMM structure.
Do not choose candidates based on heldout outer fold1; that data is never read.
"""
import argparse,json,csv,time,sys
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA,FactorAnalysis
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
import src.cc_hiwa.soft_hiwa as sh
from src.cc_hiwa.soft_hiwa import SoftHiWA
from local_PQR_ADMM_hyser import stable_group_P,stable_conditional_Q,PARAMS
from pca_vs_fa_representations import whiter
from common_force_distance_curves import quantiles_for_training

REPS=("PCA3_PCA3","FA3_FA3","PCA5_RAW5","FA5_RAW5")
TAUS=(.25,.5,1.)
KS=(4,5)
MUS=(.02,.05,.1)
FINGERS=("thumb","index","middle","ring","little")
OUTER_HOLD=1
INNER_HOLD=2
FIXED_SEED=42

def latent_fit(train,test,d,method):
    if method=="PCA":
        model=PCA(n_components=d,svd_solver="full").fit(train)
    elif method=="FA":
        model=FactorAnalysis(n_components=d,svd_method="randomized",
            iterated_power=5,random_state=42,max_iter=300,tol=1e-3).fit(train)
    else:raise ValueError(method)
    return whiter(model.transform(train),model.transform(test))[:2]

def make_data(Xraw,Yraw,tid,rep):
    # Outer fold1 not touched. Inner trial2 is excluded from latent/scaler fitting.
    m=(tid!=OUTER_HOLD)&(tid!=INNER_HOLD)
    v=tid==INNER_HOLD
    assert m.sum()==750 and v.sum()==250
    d=3 if "3" in rep else 5
    xs=StandardScaler().fit(Xraw[m]);ys=StandardScaler().fit(Yraw[m])
    Xtr=xs.transform(Xraw[m]);Xval=xs.transform(Xraw[v])
    Ytr=ys.transform(Yraw[m]);Yval=ys.transform(Yraw[v])
    x_method="PCA" if rep.startswith("PCA") else "FA"
    a,b=latent_fit(Xtr,Xval,d,x_method)
    if rep.endswith("RAW5"):
        c,e=whiter(Ytr,Yval)[:2]
    else:
        y_method="PCA" if rep.endswith("PCA3") else "FA"
        c,e=latent_fit(Ytr,Yval,d,y_method)
    return (a,b,c,e,Yraw[m],Yraw[v],tid[m],d)

def stratified_250_indices(training_ids,seed):
    rng=np.random.default_rng(seed)
    chosen=[]
    for n,t in zip((84,83,83),sorted(set(training_ids))):
        ids=np.flatnonzero(training_ids==t)
        assert len(ids)==250
        chosen.extend(rng.choice(ids,size=n,replace=False).tolist())
    return np.array(sorted(chosen),dtype=int)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--representation",choices=REPS,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--maxiter",type=int,default=120)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    with np.load(args.input) as dat:
        X=np.asarray(dat["X_emg_log1p_rms_256"],float)
        Y=np.asarray(dat["Y_force_5_raw"],float)
        t=np.asarray(dat["trial_id"],int)
        names=dat["force_names"].tolist()
    assert X.shape==(1250,256) and Y.shape==(1250,5)
    assert names==list(FINGERS)
    a,xval,c,yval,rawtr,rawval,training_tid,d=make_data(X,Y,t,args.representation)
    ix=stratified_250_indices(training_tid,111)
    iy=stratified_250_indices(training_tid,222)
    x=a[ix];y=c[iy]
    # Calibration strictly on the 750 inner-training ORIGINAL physical values.
    eps,_=quantiles_for_training(rawtr,training_tid) if len(set(training_tid))==4 else (None,None)
    # 3 training trials; use exclusively those disjoint real trial IDs.
    rng=np.random.default_rng(20261119)
    dd=[];trials=[int(x) for x in sorted(set(training_tid))]
    for _ in range(7500):
        i,j=rng.choice(trials,size=2,replace=False)
        ii=rng.choice(np.flatnonzero(training_tid==i))
        jj=rng.choice(np.flatnonzero(training_tid==j))
        dd.append(np.linalg.norm(rawtr[ii]-rawtr[jj]))
    q05=float(np.quantile(dd,.05))
    sh.sinkhorn_groups=stable_group_P
    sh.sinkhorn_weighted=stable_conditional_Q
    rows=[]
    for k in KS:
        kmx=KMeans(k,n_init=30,random_state=42).fit(x/np.sqrt(d))
        kmy=KMeans(k,n_init=30,random_state=43).fit(y/np.sqrt(d))
        dxa=cdist(x/np.sqrt(d),kmx.cluster_centers_,metric="sqeuclidean")
        dyb=cdist(y/np.sqrt(d),kmy.cluster_centers_,metric="sqeuclidean")
        for mu in MUS:
            for method,tau in [("hard",None)]+[("soft",q) for q in TAUS]:
                if method=="hard":
                    membx=np.eye(k)[kmx.labels_]
                    memby=np.eye(k)[kmy.labels_]
                else:
                    membx=softmax(-dxa/tau,axis=1)
                    memby=softmax(-dyb/tau,axis=1)
                params=dict(PARAMS)
                params.update(random_state=FIXED_SEED,maxiter=args.maxiter,
                       sa_maxiter=40,mu=mu,support_mode="positive",
                       warm_start_local=True)
                started=time.monotonic()
                record={"representation":args.representation,"dimension":d,
                    "outer_holdout_UNTOUCHED":OUTER_HOLD,"inner_validation_trial":INNER_HOLD,
                    "inner_training_trials":trials,"train_inner_available_n":750,
                    "n_fit_source":250,"n_fit_target":250,"n_inner_validation":250,
                    "K":k,"tau":tau,"ADMM_mu":mu,"method":method,"gamma_group":.2,
                    "gamma_local_reference":.1,"seed":FIXED_SEED}
                try:
                    model=SoftHiWA(**params).fit(x,membx,y,memby,
                               X_transform=np.eye(d),Y_transform=np.eye(d))
                    R=np.asarray(model.Rg,float)
                    P=np.asarray(model.P,float)
                    Q=model.local_couplings
                    p_error=float(max(np.max(np.abs(P.sum(axis=0)-1/k)),
                                        np.max(np.abs(P.sum(axis=1)-1/k))))
                    q_error=float(model.diagnostics["max_sinkhorn_marginal_error"][-1])
                    prim=float(model.diagnostics["admm_primal_residual"][-1])
                    dr=float(model.diagnostics["Rg_norm"][-1])
                    ok=(p_error<1e-5 and q_error<1e-5 and prim<=.1 and
                        dr<=.1 and np.linalg.norm(R.T@R-np.eye(d))<1e-7 and
                        len(Q)==k and all(len(row)==k for row in Q))
                    # Evaluate only on inner trial2. Candidate target lookup
                    # is exclusively the 750 TRAIN target latent+raw force.
                    dist=cdist(xval@R.T/np.sqrt(d),c/np.sqrt(d))
                    near=dist.argmin(axis=1)
                    err=np.linalg.norm(rawtr[near]-rawval,axis=1)
                    # Same null for all candidates, no model-specific tuning.
                    record.update({"success":True,"qualified":bool(ok),
                         "P_marginal_error":p_error,"Q_marginal_error":q_error,
                         "primal_residual":prim,"R_update":dr,
                         "inner_physical_error_mean":float(err.mean()),
                         "inner_physical_Hit_at_5pct":float(np.mean(err<=q05)),
                         "physical_training_only_epsilon_5pct":q05,
                         "validation_only_used_postfit":True,
                         "full_750_point_PQR_NOT_run_in_this_pilot":True})
                except Exception as e:
                    record.update({"success":False,"qualified":False,
                        "failure_type":type(e).__name__,
                        "failure_message":str(e)[:500]})
                record["elapsed_seconds"]=float(time.monotonic()-started)
                rows.append(record)
                (args.out/"partial_preflight.json").write_text(
                  json.dumps({"status":"PARTIAL","runs":rows},ensure_ascii=False,indent=2)+"\n")
                print("CANDIDATE",args.representation,k,mu,method,tau,
                      "valid",record["qualified"],"Hit5",record.get("inner_physical_Hit_at_5pct"),
                      "primal",record.get("primal_residual"),
                      "sec",round(record["elapsed_seconds"],2),flush=True)
    assert len(rows)==24
    (args.out/"inner_preflight_all24.json").write_text(json.dumps({
      "status":"DONE","outer_holdout_has_never_entered_any_fit_or_parameter_selection":True,
      "selection_scope":"ONE inner heldout trial; 250-point independent training subsample, does not constitute full 5-fold nested selection",
      "candidate_count":24,"representation":args.representation,
      "runs":rows},ensure_ascii=False,indent=2)+"\n")
    with (args.out/"inner_preflight_all24.csv").open("w",newline="",encoding="utf-8") as f:
        names=list(dict.fromkeys(key for rr in rows for key in rr))
        w=csv.DictWriter(f,fieldnames=names);w.writeheader();w.writerows(rows)
    valid=[r for r in rows if r["qualified"]]
    best=max(valid,key=lambda r:(r["inner_physical_Hit_at_5pct"],
                    -r["inner_physical_error_mean"],-r["K"])) if valid else None
    (args.out/"INNER_PREFLIGHT_DECISION.json").write_text(json.dumps({
       "valid_candidates":len(valid),"total":len(rows),
       "best_inner_trial_only":best,
       "warning":"NOT selected for outer test; full nested four-inner-trial evaluation required before selecting final hyperparameters. Do not infer improvement from test.",
    },ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"done":True,"valid":len(valid),"best_inner_trial_only":best},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
