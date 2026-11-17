#!/usr/bin/env python3
"""Local HiWA numerical port for diagnostic matched-group Hyser experiments.
NOT a byte-for-byte run of the GitHub GCOT baseline.
Requires paired_full_precision.npz from prepare_from_wfdb.py.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.spatial.distance import cdist

def polar(a):
    u,_,v=np.linalg.svd(a,full_matrices=False)
    return u@v

def sinkhorn(a,b,cost,eps,iterations,tol=1e-8):
    Z=cost-cost.min(axis=1,keepdims=True)
    Z=Z-Z.min(axis=0,keepdims=True);Z=-Z/eps
    K=np.exp(Z);v=np.ones(len(b))
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        for t in range(iterations):
            u=a/(K@v);v=b/(K.T@u)
            if not np.isfinite(v).all():break
            if (t+1)%10==0 and tol and np.max(np.abs(u*(K@v)-a))<=tol:break
        Q=(u[:,None]*K)*v[None,:]
    if np.isfinite(Q).all():return Q
    la,lb=np.log(a),np.log(b);v=np.zeros(len(b))
    for t in range(iterations):
        u=la-logsumexp(Z+v[None,:],axis=1)
        v=lb-logsumexp(Z+u[:,None],axis=0)
        if (t+1)%10==0 and tol:
            rows=np.exp(u+logsumexp(Z+v[None,:],axis=1))
            if np.max(np.abs(rows-a))<=tol:break
    return np.exp(Z+u[:,None]+v[None,:])

def align(X,Y,labels,fixed=False,mu=.005,seed=42,maxiter=300):
    groups=4;d=X.shape[1];div=np.sqrt(d)
    rng=np.random.default_rng(seed)
    xs=[X[labels==i]/div for i in range(groups)]
    ys=[Y[labels==i]/div for i in range(groups)]
    R=polar(rng.random((d,d)))
    L=np.tile(np.eye(d),(groups,groups,1,1));D=np.zeros_like(L)
    P=np.eye(groups)/groups if fixed else np.full((groups,groups),1/groups**2)
    history=[];pi=np.full(groups,1/groups)
    for it in range(maxiter):
        costs=np.zeros((groups,groups));err=0
        for i in range(groups):
            xx=xs[i];a=np.full(len(xx),1/len(xx))
            for j in range(groups):
                if fixed and i!=j:
                    L[i,j]=R;continue
                yy=ys[j];b=np.full(len(yy),1/len(yy))
                p=max(P[i,j],np.finfo(float).tiny)
                penalty=(mu/d)*(R-D[i,j]);q=np.outer(a,b);local=polar(rng.random((d,d)))
                for k in range(100):
                    prev=local
                    local=polar(2*p*(yy.T@q.T@xx)+penalty)
                    C=cdist(xx@local.T,yy,'sqeuclidean')
                    q=sinkhorn(a,b,C,.1/p,150)
                    if np.linalg.norm(local-prev,2)<=.01:break
                L[i,j]=local;costs[i,j]=np.sum(q*C)
                err=max(err,float(np.max(abs(q.sum(0)-b))),float(np.max(abs(q.sum(1)-a))))
        if not fixed:P=sinkhorn(pi,pi,costs,.2,1000)
        prev=R.copy()
        if fixed:
            R=polar(np.mean(np.array([L[i,i]+D[i,i] for i in range(groups)]),axis=0))
            for i in range(groups):D[i,i]+=L[i,i]-R
            primal=float(np.linalg.norm(np.array([L[i,i]-R for i in range(groups)])))
        else:
            R=polar(np.mean(L+D,axis=(0,1)));D+=L-R
            primal=float(np.linalg.norm(L-R))
        delta=float(np.linalg.norm(R-prev))
        err=max(err,float(np.max(abs(P.sum(0)-pi))),float(np.max(abs(P.sum(1)-pi))))
        history.append(dict(iteration=it+1,delta_R=delta,primal=primal,marginal=err))
        if it>=5 and ((fixed and delta<=.01 and primal<=.01 and err<=1e-5)
                      or (not fixed and delta<=.1)):break
    return R,P,history

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path('experiments/hyser/data/paired_full_precision.npz'))
    parser.add_argument('--method',choices=['free','fixed'],default='free')
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    z=np.load(args.data);X,Y,labels=z['X'],z['Y'],z['labels']
    if not(np.array_equal(np.unique(labels),np.arange(4)) and np.bincount(labels).tolist()==[46,53,91,60]):
        raise ValueError('This experiment expects inherited labels from source-only KMeans')
    mu=.05 if args.method=='fixed' else .005
    R,P,history=align(X,Y,labels,fixed=args.method=='fixed',mu=mu,seed=args.seed)
    diff=X@R.T-Y;SSE=float(np.sum(diff**2));SST=float(np.sum((Y-Y.mean(0))**2))
    info=dict(seed=args.seed,method=args.method,mu=mu,iterations=len(history),last=history[-1],
        R=R.tolist(),P=P.tolist(),R2=1-SSE/SST,relative_squared_error=SSE/float(np.sum(Y**2)),
        mean_euclidean=float(np.mean(np.linalg.norm(diff,axis=1))))
    print(json.dumps(info,indent=2))
if __name__=='__main__':main()
