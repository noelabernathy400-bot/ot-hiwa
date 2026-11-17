#!/usr/bin/env python3
"""Hyser all-250 hard HiWA, DIRECT imports of user's GCOT branch.

Example from main checkout:
  git worktree add /tmp/ot-hiwa-gcot experiments/dandi-000138-4d-hard-clustering-20261117
  python experiments/hyser/scripts/run_hiwa_postpca_normalized.py \
    --data-dir /path/to/HYSER_WFDB \
    --gcot-root /tmp/ot-hiwa-gcot/research/GCOT-reproducible-20261018/GC-OT \
    --output experiments/hyser/output/hiwa_postpca_direct.json
Needs wfdb, numpy, scipy, scikit-learn and the supplied branch sources.
This entrypoint is independently auditable but NOT yet executed in this chat.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import wfdb
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def metric(x,truth,R):
    error=x@R.T-truth
    sse=float(np.sum(error**2))
    sst=float(np.sum((truth-truth.mean(0))**2))
    return {'r2':1-sse/sst,'paired_relative_mse':sse/float(np.sum(truth**2)),
            'mean_euclidean_distance':float(np.mean(np.linalg.norm(error,axis=1)))}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",required=True,type=Path)
    p.add_argument("--gcot-root",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    sys.path.insert(0,str(args.gcot_root.resolve()))
    from gcot.alignment import SolverConfig,hierarchical_alignment,hard_measures
    from gcot.baselines import hiwa_gmm
    from gcot.numerics import whiten

    emg=wfdb.rdrecord(str(args.data_dir/"random_preprocess_sample1"))
    record=wfdb.rdrecord(str(args.data_dir/"random_force_sample1"))
    if emg.p_signal.shape!=(51200,256) or record.p_signal.shape!=(2500,5):
        raise ValueError("Hyser unexpected signal dimensions")
    fingers=['thumb','index','middle','ring','little']
    force=record.p_signal[:,[record.sig_name.index(name) for name in fingers]]
    rms=[]; five_finger=[]
    for i in range(250):
        a=emg.p_signal[round(i*.1*emg.fs):round((i+1)*.1*emg.fs)]
        b=force[round(i*.1*record.fs):round((i+1)*.1*record.fs)]
        rms.append(np.sqrt(np.mean((a-a.mean(axis=0))**2,axis=0)))
        five_finger.append(np.mean(b,axis=0))
    rms=np.asarray(rms);five_finger=np.asarray(five_finger)
    features=StandardScaler().fit_transform(np.log1p(rms*1e5))
    pca=PCA(n_components=3,random_state=args.seed).fit(features)
    xp=pca.transform(features)
    selected=np.argsort(-five_finger.var(axis=0))[:3]
    yp=five_finger[:,selected]

    cfg=SolverConfig(stopping='legacy',mu=.005,tol=.1,maxiter=300,
            epsilon_group=.2,epsilon_sample=.1,inner_maxiter=100,
            inner_tol=.01,sinkhorn_maxiter=150,group_sinkhorn_maxiter=1000,sinkhorn_tol=0)
    out={'dataset':'Real Hyser sample1','windows':250,'fit_uses_pairing':False,
         'fingers':[fingers[j] for j in selected],
         'pca_variance':pca.explained_variance_ratio_.tolist(),
         'code':'DIRECT imports from GC-OT user branch',
         'seed':args.seed,'K':4,'modes':{}}
    for mode in ('whiten','zscore'):
        if mode=='whiten':
            x,_,_=whiten(xp)
            y,_,_=whiten(yp)
        else:
            x=StandardScaler().fit_transform(xp)
            y=StandardScaler().fit_transform(yp)
        permutation=np.random.default_rng(114).permutation(250)
        shuffled=y[permutation]
        truth=shuffled[np.argsort(permutation)]
        oracle_row,_=orthogonal_procrustes(x,truth)
        result={'before':metric(x,truth,np.eye(3)),
            'oracle':metric(x,truth,oracle_row.T),'methods':{}}
        for name in ('kmeans_3D','gmm_isomap'):
            if name=='kmeans_3D':
                lx=KMeans(4,random_state=args.seed,n_init=30).fit_predict(x)
                ly=KMeans(4,random_state=args.seed,n_init=30).fit_predict(shuffled)
                fitted=hierarchical_alignment(hard_measures(x,lx),hard_measures(shuffled,ly),
                    args.seed,cfg,scale=True)
            else:
                fitted=hiwa_gmm(x,shuffled,groups=4,seed=args.seed,config=cfg)
            R=fitted['R']
            result['methods'][name]={
                'metrics':metric(x,truth,R),'iterations':int(len(fitted['history'])),
                'legacy_converged':bool(fitted['converged']),
                'delta_R':float(fitted['rotation_delta']),
                'consensus_F':float(fitted['consensus_residual']),
                'marginal_error':float(fitted['marginal_error']),
                'R':R.tolist(),'P':fitted['P'].tolist()
            }
        out['modes'][mode]=result
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf8")
    print('Saved',args.output)

if __name__=='__main__':main()
