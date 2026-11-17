#!/usr/bin/env python3
"""Full-precision Hyser source-only soft memberships and HiWA matched-group OT.

Run after:
  python experiments/hyser/scripts/prepare_from_wfdb.py \
    --raw-dir /path/to/WFDB --output experiments/hyser/data/paired_full_precision.npz
  python experiments/hyser/scripts/run_soft_inherited.py \
    --data experiments/hyser/data/paired_full_precision.npz --mode prototype

This is an auditable *local numerical port* of gcot/grouping.py Eq. (2)
and gcot/alignment.py's local OT / Procrustes / ADMM update, not a direct
execution of the original repository GCOT class. The target inherits source
membership vectors using real synchronized row correspondence. Hence this is
a GROUP-SUPERVISED diagnostic experiment, not an unsupervised GCOT baseline.
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax, xlogy
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from solver import sinkhorn, polar

def prototypes(x, groups=4, seed=42, entropy=1.0):
    # Exact TPAMI Eq. 2, as transcribed in user-provided GC-OT/gcot/grouping.py
    c0 = KMeans(groups, n_init=1, random_state=seed).fit(x).cluster_centers_
    def loss_grad(flat):
        c=flat.reshape(groups, x.shape[1])
        a=softmax(x@c.T,axis=1)
        e=a@c-x
        f=float(np.sum(e*e)+entropy*np.sum(xlogy(a,a))/len(x))
        da=2*e@c.T+entropy/len(x)*(np.log(np.maximum(a,1e-300))+1)
        dz=a*(da-np.sum(da*a,axis=1,keepdims=True))
        g=2*a.T@e+dz.T@x
        return f,g.ravel()
    res=minimize(loss_grad,c0.ravel(),jac=True,method='L-BFGS-B',
                 options={'maxiter':300,'ftol':1e-10,'gtol':1e-7})
    c=res.x.reshape(groups,x.shape[1])
    a=softmax(x@c.T,axis=1)
    a=np.maximum(a,np.finfo(float).tiny)
    a/=a.sum(1,keepdims=True)
    return a,{'optimizer_success':bool(res.success),'optimizer_iters':int(res.nit),
              'prototype_centers':c.tolist(),'objective':float(res.fun)}

def memberships(x, labels, mode, seed, tau):
    if mode=='hard':
        return np.eye(4)[labels],{'mode':'hard_kmeans'}
    if mode=='soft_kmeans':
        c=np.asarray([x[labels==i].mean(0) for i in range(4)])
        cost=cdist(x,c,'sqeuclidean')
        return softmax(-cost/tau,axis=1),{'mode':'soft_kmeans','tau':tau}
    return prototypes(x,4,seed,entropy=1.0)

def fit_matched(x,y,a,mu=.05,seed=42,maxiter=160):
    # Same model class for hard and soft. B_ik=A_ik by synchronized sample
    # index; four full-support matched blocks, P=diag(1/4).
    k,d=4,x.shape[1]
    xs=x/np.sqrt(d);ys=y/np.sqrt(d)
    weights=[a[:,i]/np.sum(a[:,i]) for i in range(k)]
    rng=np.random.default_rng(seed)
    rg=polar(rng.random((d,d)))
    local=np.tile(np.eye(d),(k,1,1))
    dual=np.zeros((k,d,d));history=[]
    for step in range(maxiter):
        marginal=0;local_cost=0
        for g in range(k):
            w=weights[g]
            q=np.outer(w,w)
            r=polar(rng.random((d,d)))
            penalty=mu/d*(rg-dual[g])
            for _ in range(100):
                old_r=r
                r=polar(2*.25*(ys.T@q.T@xs)+penalty)
                C=cdist(xs@r.T,ys,'sqeuclidean')
                q=sinkhorn(w,w,C,.1/.25,150,1e-8)
                if np.linalg.norm(r-old_r,2)<=.01:break
            local[g]=r
            local_cost+=.25*float(np.sum(q*C))
            marginal=max(marginal,float(np.max(abs(q.sum(0)-w))),
                         float(np.max(abs(q.sum(1)-w))))
        old_rg=rg.copy()
        rg=polar(np.mean(local+dual,axis=0))
        dual+=local-rg
        dr=float(np.linalg.norm(rg-old_rg))
        primal=float(np.linalg.norm(local-rg))
        history.append({'iteration':step+1,'rotation_delta':dr,
                        'primal_residual':primal,'marginal_error':marginal,
                        'local_transport_cost':local_cost})
        if step>=5 and dr<=.01 and primal<=.01 and marginal<=1e-6:break
    return rg,np.eye(4)/4,history

def metrics(x,y,r):
    e=x@r.T-y;ss=float(np.sum(e**2))
    return {'r2':float(1-ss/np.sum((y-y.mean(0))**2)),
            'relative_squared_error':float(ss/np.sum(y*y)),
            'mean_euclidean_distance':float(np.mean(np.linalg.norm(e,axis=1)))}

def global_transport(x,y,a,r):
    # Global R, entropic OT plans recomputed; entropy terms excluded.
    xs=x/np.sqrt(3);ys=y/np.sqrt(3);C=cdist(xs@r.T,ys,'sqeuclidean')
    total=0.;margin=0.
    for g in range(4):
        w=a[:,g]/a[:,g].sum()
        q=sinkhorn(w,w,C,.1/.25,150,1e-8)
        total+=.25*float(np.sum(q*C))
        margin=max(margin,float(np.abs(q.sum(0)-w).max()),
                   float(np.abs(q.sum(1)-w).max()))
    return {'cost':total,'marginal_error':margin}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path('experiments/hyser/data/paired_full_precision.npz'))
    p.add_argument('--output',type=Path,default=Path('experiments/hyser/results/soft_rerun_local.json'))
    p.add_argument('--mode',choices=['hard','soft_kmeans','prototype'],default='prototype')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--tau',type=float,default=.5)
    p.add_argument('--mu',type=float,default=.05)
    args=p.parse_args()
    z=np.load(args.data);x,y,labels=z['X'],z['Y'],z['labels']
    if x.shape!=(250,3) or y.shape!=(250,3) or len(labels)!=250:
        raise ValueError('Expected aligned, full precision, real Hyser 250x3 data')
    if np.bincount(labels,minlength=4).tolist()!=[46,53,91,60]:
        raise ValueError('Source-only KMeans reference labels changed')
    a,fit=memberships(x,labels,args.mode,args.seed,args.tau)
    r,P,history=fit_matched(x,y,a,args.mu,args.seed)
    last=history[-1]
    result={
      'mode':args.mode,'seed':args.seed,'mu':args.mu,'source_fit':fit,
      'n_pairs':250,'target_memberships':'inherited A by same true time-window index',
      'supervision':'group correspondence known, not fully unsupervised',
      'membership_entropy_mean':float(np.mean(-np.sum(xlogy(a,a),axis=1))),
      'source_membership_mass':a.sum(axis=0).tolist(),
      'converged_checked':bool(last['rotation_delta']<=.01 and
                                last['primal_residual']<=.01 and
                                last['marginal_error']<=1e-6),
      'iterations':len(history),'last':last,'R':r.tolist(),'P':P.tolist(),
      'ot_before':global_transport(x,y,a,np.eye(3)),
      'ot_after':global_transport(x,y,a,r),
      'paired_before':metrics(x,y,np.eye(3)),
      'paired_after':metrics(x,y,r),
      'history':history}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print('Written',args.output)
    print(json.dumps({k:result[k] for k in ['mode','converged_checked',
       'iterations','ot_before','ot_after','paired_after']},indent=2))
if __name__=='__main__':
    main()
