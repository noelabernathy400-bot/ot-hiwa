#!/usr/bin/env python3
"""Historical Hyser trial1 Hard-HiWA reimplementation (not original project HiWA).

This is the prior experiment solver, included for provenance. Its defaults and
preprocessing differ from src.hiwa.hiwa.HiWA. Do NOT use it as the official baseline.
"""
from pathlib import Path
import json, time
import numpy as np
from scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment
from scipy.linalg import orthogonal_procrustes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT=Path(__file__).resolve().parents[1]/"output"

def polar(A):
    U,S,Vt=np.linalg.svd(A,full_matrices=False)
    return U@Vt

def sinkhorn(A,B,gamma,steps=80):
    C=np.maximum(np.sum(A*A,axis=1)[:,None]+np.sum(B*B,axis=1)[None,:]-2*A@B.T,0)
    n,m=C.shape;lp=np.full(n,-np.log(n));lq=np.full(m,-np.log(m))
    logK=-C/max(float(gamma),1e-10)
    u=np.zeros(n);v=np.zeros(m)
    for _ in range(steps):
        u=lp-logsumexp(logK+v[None,:],axis=1)
        v=lq-logsumexp(logK+u[:,None],axis=0)
    Q=np.exp(u[:,None]+logK+v[None,:])
    return Q,float(np.sum(Q*C)),float(np.max(np.abs(Q.sum(axis=1)-1/n)))

def sinkhorn_cost(C,gamma,steps=160):
    n,m=C.shape;lp=np.full(n,-np.log(n));lq=np.full(m,-np.log(m))
    logK=-C/max(float(gamma),1e-10)
    u=np.zeros(n);v=np.zeros(m)
    for _ in range(steps):
        u=lp-logsumexp(logK+v[None,:],axis=1)
        v=lq-logsumexp(logK+u[:,None],axis=0)
    return np.exp(u[:,None]+logK+v[None,:])

def hiwa_fit(X,Y,cx,cy,seed=42,iterations=25,sa_iterations=12,mu=.005,
             cluster_gamma=.2,local_gamma=.1):
    rng=np.random.default_rng(seed)
    n,d=X.shape;k=len(np.unique(cx));j=len(np.unique(cy))
    assert d==Y.shape[1] and d==3 and k==j==4
    ids_x=[np.flatnonzero(cx==i) for i in range(k)]
    ids_y=[np.flatnonzero(cy==b) for b in range(j)]
    Rg=polar(rng.normal(size=(d,d)))
    R=np.tile(np.eye(d)[:,:,None,None],(1,1,k,j))
    L=np.zeros((d,d,k,j));P=np.full((k,j),1/(k*j));history=[]
    for iteration in range(iterations):
        t=time.time();costs=np.zeros((k,j));errors=[]
        for a,idx in enumerate(ids_x):
            x=X[idx]
            for b,idy in enumerate(ids_y):
                y=Y[idy]
                T=(mu/d)*(Rg-L[:,:,a,b]);rij=polar(rng.normal(size=(d,d)))
                Q=np.full((len(x),len(y)),1/(len(x)*len(y)))
                dist=np.nan
                for inner in range(sa_iterations):
                    old=rij
                    rij=polar(2*P[a,b]*(y.T@Q.T@x)+T)
                    Q,dist,err=sinkhorn(x@rij.T,y,local_gamma/max(P[a,b],1e-12),steps=50)
                    if np.linalg.norm(old-rij,ord=2)<1e-3:break
                R[:,:,a,b]=rij
                costs[a,b]=dist;errors.append(err)
        P=sinkhorn_cost(costs,cluster_gamma,steps=200)
        prev=Rg.copy()
        Rg=polar(np.mean(R+L,axis=(2,3)))
        residual=R-Rg[:,:,None,None]
        L=L+residual
        primal=float(np.linalg.norm(residual))
        change=float(np.linalg.norm(Rg-prev,"fro"))
        dual=float(mu*np.sqrt(k*j)*change)
        history.append(dict(iter=iteration+1,primal_residual=primal,
                            dual_residual=dual,rotation_change=change,
                            max_instance_marginal_error=float(max(errors)),
                            elapsed_seconds=float(time.time()-t),
                            cost_min=float(costs.min()),cost_max=float(costs.max())))
        print("ADMM %02d: primal=%.4g deltaR=%.4g"%(iteration+1,primal,change),flush=True)
        if change<1e-3 and primal<5e-2:break
    return Rg,P,history

def align_eval(X,Y,cx,cy,xids,yids,R,P,seed=123):
    perm=np.zeros(P.shape[0],dtype=int)
    a,b=linear_sum_assignment(-P);perm[a]=b
    idx={v:i for i,v in enumerate(yids)}
    y_true=np.array([idx[v] for v in xids])
    Ypair=Y[y_true];y_cluster=cy[y_true]
    transformed=X@R.T
    actual=np.linalg.norm(transformed-Ypair,axis=1)
    rng=np.random.default_rng(seed);wrong=Ypair[rng.permutation(len(X))]
    wrongdist=np.linalg.norm(transformed-wrong,axis=1)
    C=np.sum((transformed[:,None,:]-Y[None,:,:])**2,axis=2)
    idx_nn=np.argmin(C,axis=1)
    top1=float(np.mean(idx_nn==y_true))
    hits5=np.argpartition(C,kth=min(4,len(Y)-1),axis=1)[:,:min(5,len(Y))]
    top5=float(np.mean((hits5==y_true[:,None]).any(axis=1)))
    agreement=float(np.mean(perm[cx]==y_cluster))
    return dict(n=int(len(X)),mapping=perm.tolist(),cluster_agreement=agreement,
                same_time_mean_distance=float(actual.mean()),
                random_time_mean_distance=float(wrongdist.mean()),
                time_retrieval_top1=top1,time_retrieval_top5=top5)

def oracle_fit(X,Y,xids,yids):
    idx={v:i for i,v in enumerate(yids)}
    yy=Y[[idx[v] for v in xids]]
    A,_=orthogonal_procrustes(X,yy)
    return A

def main():
    z=np.load(OUT/"domains.npz",allow_pickle=True)
    q=np.load(OUT/"clusters.npz")
    X,Y=z["Xtrain"],z["Ytrain"];cx,cy=q["cx"],q["cy"]
    R,P,history=hiwa_fit(X,Y,cx,cy)
    train=align_eval(X,Y,cx,cy,z["tr_pair"],z["tr_y_pair"],R,P)
    test=align_eval(z["Xtest"],z["Ytest"],q["cx_test"],q["cy_test"],z["te_pair"],z["te_y_pair"],R,P)
    A=oracle_fit(X,Y,z["tr_pair"],z["tr_y_pair"])
    it={v:i for i,v in enumerate(z["tr_y_pair"])}
    yy=Y[[it[v] for v in z["tr_pair"]]]
    it={v:i for i,v in enumerate(z["te_y_pair"])}
    yy_test=z["Ytest"][[it[v] for v in z["te_pair"]]]
    result=dict(source="Real Hyser trial1",method="legacy reimplementation NOT original",
       train=train,test=test,R_column_convention=R.tolist(),cluster_ot_P=P.tolist(),
       oracle_train_relative_error=float(np.linalg.norm(X@A-yy)/np.linalg.norm(yy)),
       oracle_holdout_relative_error=float(np.linalg.norm(z["Xtest"]@A-yy_test)/np.linalg.norm(yy_test)),
       hiwa_train_relative_error=float(np.linalg.norm(X@R.T-yy)/np.linalg.norm(yy)),
       hiwa_holdout_relative_error=float(np.linalg.norm(z["Xtest"]@R.T-yy_test)/np.linalg.norm(yy_test)),
       history=history,
       convergence=bool(history[-1]["primal_residual"]<.05 and history[-1]["rotation_change"]<.001))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"hiwa_results.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(train=train,test=test,convergence=result["convergence"]),indent=2))
if __name__=="__main__":main()
