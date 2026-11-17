#!/usr/bin/env python3
"""Official project HiWA-class entrypoint (NOT the previous audit reimplementation)."""
import argparse,sys,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.hiwa.hiwa import HiWA

def whiten_train(train):
    mean=train.mean(axis=0);v,u=np.linalg.eigh(np.cov(train,rowvar=False))
    return mean,(u*(np.maximum(v,1e-8)**(-.5)))@u.T

def assess(x,y,x_id,y_id,x_lab,y_lab,R,perm):
    index={s:i for i,s in enumerate(y_id)}
    target=np.array([index[s] for s in x_id])
    aligned=x@R.T
    dis=np.linalg.norm(aligned[:,None]-y[None],axis=2)
    order=np.argsort(dis,axis=1)
    shuff=np.random.default_rng(321).permutation(len(target))
    return dict(n=len(x),cluster_agreement=float(np.mean(perm[x_lab]==y_lab[target])),
        mean_paired_distance=float(dis[np.arange(len(x)),target].mean()),
        mean_shuffled_paired_distance=float(np.linalg.norm(aligned-y[target[shuff]],axis=1).mean()),
        top1=float(np.mean(order[:,0]==target)),
        top5=float(np.mean(np.any(order[:,:min(5,len(y))]==target[:,None],axis=1))))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--mu",type=float,default=.05)
    p.add_argument("--maxiter",type=int,default=200)
    p.add_argument("--tol",type=float,default=.01)
    a=p.parse_args()
    z=np.load(a.out_dir/"domains.npz",allow_pickle=True)
    g=np.load(a.out_dir/"clusters.npz",allow_pickle=True)
    mx,sx=whiten_train(z["Xtrain"]);my,sy=whiten_train(z["Ytrain"])
    transform=lambda q,m,s:(q-m)@s
    X,Y=transform(z["Xtrain"],mx,sx),transform(z["Ytrain"],my,sy)
    Xt,Yt=transform(z["Xtest"],mx,sx),transform(z["Ytest"],my,sy)
    np.random.seed(a.seed)
    # Explicit train-only whitening means normalize=False in the source HiWA class.
    # identity projections yield the reference X_mbed = X/sqrt(3).
    solver=HiWA(dim_red_method=PCA(n_components=3),normalize=False,
       maxiter=a.maxiter,tol=a.tol,mu=a.mu,
       shorn_maxiter=300,shorn_gamma=.2,sa_maxiter=40,sa_tol=.01,
       sa_shorn_maxiter=80,sa_shorn_gamma=.1)
    solver.fit(X,g["cx"],Y,g["cy"],X_transform=np.eye(3),Y_transform=np.eye(3))
    P,R=solver.P,solver.Rg
    r,c=linear_sum_assignment(-P);perm=np.empty(len(r),dtype=int);perm[r]=c
    d=solver.diagnostics
    result={"implementation":"DIRECT src.hiwa.hiwa.HiWA", "seed":a.seed,
       "parameters":{"mu":a.mu,"maxiter":a.maxiter,"tol":a.tol},
       "iterations":len(d["Rg_norm"]),"R":R.tolist(),"P":P.tolist(),
       "permutation":perm.tolist(),
       "last_global":float(d["Rg_norm"][-1]),
       "last_primal":float(d["admm_primal_residual"][-1]),
       "last_group_sinkhorn_marginal_error":float(d["group_sinkhorn_marginal_error"][-1]),
       "converged":bool(d["Rg_norm"][-1]<=a.tol and d["admm_primal_residual"][-1]<=a.tol),
       "train":assess(X,Y,z["tr_pair"],z["tr_y_pair"],g["cx"],g["cy"],R,perm),
       "holdout":assess(Xt,Yt,z["te_pair"],z["te_y_pair"],g["cx_test"],g["cy_test"],R,perm),
       "diagnostics":{"global":d["Rg_norm"].tolist(),"primal":d["admm_primal_residual"].tolist(),
          "dual":d["admm_dual_residual"].tolist(),
          "group_marginal":d["group_sinkhorn_marginal_error"].tolist()},
       "warning":"Single-trial holdout is not independent-trial generalization"}
    path=a.out_dir/"hard_hiwa_original.json"
    path.write_text(json.dumps(result,indent=2),encoding="utf8")
    print("saved",path,"converged",result["converged"])
if __name__=="__main__":main()
