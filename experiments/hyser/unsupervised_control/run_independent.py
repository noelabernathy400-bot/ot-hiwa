#!/usr/bin/env python3
"""Independent (unpaired) Hard/Soft-GCOT-style Hyser control.

The two domains learn their groupings independently. Synchrony is used ONLY
after fitting to compute R2 and paired group overlap. Scaling X,Y by sqrt(d),
all other OT and ADMM hyperparameters are shared among comparison methods.

Run:
 python experiments/hyser/scripts/prepare_from_wfdb.py --raw-dir PATH/TO/WFDB
 python experiments/hyser/unsupervised_control/run_independent.py \
   --data experiments/hyser/data/paired_full_precision.npz \
   --out experiments/hyser/unsupervised_control/reproduced.json

Numerical port of user-uploaded GC-OT formulas. Four-by-four group Sinkhorn
is replaced by equivalent higher-accuracy entropic dual Newton/root solver.
NOT byte-identical execution of the original GCOT class.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import root,linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.special import softmax,xlogy
from sklearn.cluster import KMeans

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'scripts'))
from solver import polar,sinkhorn
from run_soft_inherited import prototypes

K=4
def groups(X,Y,method,seed):
    if method=='hard':
        lx=KMeans(K,n_init=30,random_state=seed).fit_predict(X)
        ly=KMeans(K,n_init=30,random_state=seed+1).fit_predict(Y)
        return np.eye(K)[lx],np.eye(K)[ly]
    if method=='soft_kmeans':
        cx=KMeans(K,n_init=30,random_state=seed).fit(X).cluster_centers_
        cy=KMeans(K,n_init=30,random_state=seed+1).fit(Y).cluster_centers_
        return softmax(-cdist(X,cx,'sqeuclidean')/.5,axis=1),softmax(-cdist(Y,cy,'sqeuclidean')/.5,axis=1)
    if method=='soft_prototype':
        # TPAMI Eq. 2; fit separately with independent random seeds
        ax,_=prototypes(X,K,seed,entropy=1.0)
        by,_=prototypes(Y,K,seed+1,entropy=1.0)
        return ax,by
    raise ValueError(method)

def measures(X,A):
    d=X.shape[1];out=[]
    for g in range(K):
        ids=np.flatnonzero(A[:,g]>0)
        a=A[ids,g]/A[ids,g].sum()
        out.append((X[ids]/np.sqrt(d),a))
    return out

def precise_group_entropic_OT(cost,eps=.2):
    # Dual root for the SAME entropy-regularized 4x4 transportation problem.
    c=cost-cost.min(axis=1,keepdims=True)
    c=c-c.min(axis=0,keepdims=True)
    def plan(z):
        f=z[:K];g=np.r_[z[K:],0]
        return np.exp(np.clip((f[:,None]+g[None,:]-c)/eps,-700,700))
    def res(z):
        p=plan(z)
        return np.r_[p.sum(axis=1)-1/K,p.sum(axis=0)[:K-1]-1/K]
    def jac(z):
        p=plan(z);J=np.zeros((2*K-1,2*K-1))
        J[:K,:K]=np.diag(p.sum(axis=1)/eps)
        J[:K,K:]=p[:,:K-1]/eps
        J[K:,:K]=p[:,:K-1].T/eps
        J[K:,K:]=np.diag(p.sum(axis=0)[:K-1]/eps)
        return J
    sol=root(res,np.zeros(2*K-1),jac=jac,method='hybr',options={'xtol':1e-12,'maxfev':400})
    p=plan(sol.x)
    if np.max(np.abs(res(sol.x)))>5e-8 or not np.isfinite(p).all():
        p=sinkhorn(np.full(K,1/K),np.full(K,1/K),c,eps,100000,1e-10)
    return p

def fit(X,Y,A,B,seed,mu=.05,maxiter=180):
    d=X.shape[1];mx,my=measures(X,A),measures(Y,B)
    rng=np.random.default_rng(seed);R=polar(rng.random((d,d)))
    P=np.ones((K,K))/K**2;dual=np.zeros((K,K,d,d));hist=[]
    for t in range(maxiter):
        L=np.empty((K,K,d,d));costs=np.empty((K,K));marg=0.
        for i,(x,a) in enumerate(mx):
            for j,(y,b) in enumerate(my):
                pij=max(P[i,j],np.finfo(float).tiny)
                penalty=mu/d*(R-dual[i,j])
                q=np.outer(a,b);r=polar(rng.random((d,d)))
                for inner in range(100):
                    old=r
                    r=polar(2*pij*(y.T@q.T@x)+penalty)
                    C=cdist(x@r.T,y,'sqeuclidean')
                    q=sinkhorn(a,b,C,.1/pij,150,1e-8)
                    if np.linalg.norm(r-old,2)<=.01:break
                L[i,j]=r;costs[i,j]=np.sum(q*C)
                marg=max(marg,float(np.max(np.abs(q.sum(axis=1)-a))),
                              float(np.max(np.abs(q.sum(axis=0)-b))))
        oldP=P.copy();P=precise_group_entropic_OT(costs)
        oldR=R.copy();R=polar(np.mean(L+dual,axis=(0,1)))
        dual+=L-R
        dr=float(np.linalg.norm(R-oldR));pr=float(np.linalg.norm(L-R))
        marg=max(marg,float(np.abs(P.sum(axis=0)-1/K).max()),float(np.abs(P.sum(axis=1)-1/K).max()))
        hist.append(dict(iteration=t+1,rotation_delta=dr,consensus=pr,marginal=marg,
                         group_P_delta=float(np.linalg.norm(P-oldP))))
        if t>=5 and dr<=.01 and pr<=.01 and marg<=1e-6:break
    return R,P,hist,mx,my

def transportation(mx,my,R,P):
    value=0.;err=0
    for i,(x,a) in enumerate(mx):
        for j,(y,b) in enumerate(my):
            pij=P[i,j]
            C=cdist(x@R.T,y,'sqeuclidean')
            q=sinkhorn(a,b,C,.1/max(pij,1e-15),150,1e-8)
            value+=pij*np.sum(q*C)
            err=max(err,float(np.abs(q.sum(axis=1)-a).max()),float(np.abs(q.sum(axis=0)-b).max()))
    return dict(cost=float(value),marginal_error=float(err))

def paired_metrics(X,Y,R):
    E=X@R.T-Y;sse=float(np.sum(E**2))
    return dict(R2=1-sse/np.sum((Y-Y.mean(axis=0))**2),
                paired_rel_squared_error=sse/np.sum(Y**2),
                paired_mean_distance=float(np.mean(np.linalg.norm(E,axis=1))))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[40,41,42,43,44])
    p.add_argument('--methods',nargs='+',default=['hard','soft_kmeans','soft_prototype'])
    args=p.parse_args()
    arr=np.load(args.data);X=np.asarray(arr['X'],dtype=float);Y=np.asarray(arr['Y'],dtype=float)
    if X.shape!=Y.shape or X.shape!=(250,3):raise ValueError('Expected Hyser 250x3 paired windows')
    output={'note':'True synchrony is EVALUATION ONLY; all fits independent',
            'parameters':dict(K=4,mu=.05,eps_sample=.1,eps_group=.2,scale='1/sqrt(3)',
                               local_steps=100,sample_sinkhorn_steps=150,maxiter=180,
                               termination='delta_R/primal < .01; marginal < 1e-6'),
            'runs':[]}
    args.out.parent.mkdir(exist_ok=True,parents=True)
    for seed in args.seeds:
        for method in args.methods:
            A,B=groups(X,Y,method,seed)
            R,P,h,mx,my=fit(X,Y,A,B,seed)
            overlap=A.T@B/len(X)
            rows,cols=linear_sum_assignment(-overlap)
            rec=dict(seed=seed,method=method,iterations=len(h),last=h[-1],
                passed_numerical_thresholds=h[-1]['rotation_delta']<=.01
                  and h[-1]['consensus']<=.01 and h[-1]['marginal']<=1e-6,
                R=R.tolist(),P=P.tolist(),
                X_membership_entropy=float(np.mean(-np.sum(xlogy(A,A),axis=1))),
                Y_membership_entropy=float(np.mean(-np.sum(xlogy(B,B),axis=1))),
                OT_before=transportation(mx,my,np.eye(3),P),
                OT_after=transportation(mx,my,R,P),
                paired_EVAL_ONLY=paired_metrics(X,Y,R),
                P_mass_on_true_best_group_permutation_EVAL_ONLY=float(P[rows,cols].sum()),
                true_pair_overlap_EVAL_ONLY=overlap.tolist())
            output['runs'].append(rec)
            args.out.write_text(json.dumps(output,ensure_ascii=False,indent=2))
            print(seed,method,rec['paired_EVAL_ONLY']['R2'],rec['OT_after']['cost'],
                  rec['passed_numerical_thresholds'],flush=True)
if __name__=='__main__':main()
