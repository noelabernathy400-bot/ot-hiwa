#!/usr/bin/env python3
"""Real Hyser all-250-window, no-split Hard HiWA; import repo original src.hiwa.hiwa.HiWA."""
from pathlib import Path
import argparse, json, sys
import numpy as np
import wfdb
from scipy.linalg import sqrtm, orthogonal_procrustes
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.hiwa.hiwa import HiWA
FINGERS=['thumb','index','middle','ring','little']

def white(a):
    a=a-a.mean(axis=0,keepdims=True)
    return a@np.linalg.pinv(np.real_if_close(sqrtm(np.cov(a,rowvar=False))).astype(float))

def score(a,b):
    e=a-b
    sse=float(np.sum(e**2))
    aa,bb=white(a),white(b)
    return {'R2_original_neural_3D':float(1-np.mean((bb-aa)**2,axis=0).sum()/np.var(bb,axis=0).sum()),
            'R2_standard':float(1-sse/np.sum((b-b.mean(axis=0))**2)),
            'relative_squared_error':float(sse/np.sum(b**2)),
            'relative_Frobenius_error':float(np.sqrt(sse/np.sum(b**2))),
            'mean_paired_Euclidean':float(np.linalg.norm(e,axis=1).mean())}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[42])
    a=p.parse_args()
    a.out_dir.mkdir(parents=True,exist_ok=True)
    emg=wfdb.rdrecord(str(a.data_dir/'random_preprocess_sample1'))
    force=wfdb.rdrecord(str(a.data_dir/'random_force_sample1'))
    assert emg.p_signal.shape==(51200,256) and force.p_signal.shape==(2500,5)
    assert abs(len(emg.p_signal)/emg.fs-len(force.p_signal)/force.fs)<.01
    ef=emg.p_signal
    ff=force.p_signal[:,[force.sig_name.index(f) for f in FINGERS]]
    rms=[]; forces=[]
    for k in range(250):
        e=ef[round(k*.1*emg.fs):round((k+1)*.1*emg.fs)]
        f=ff[round(k*.1*force.fs):round((k+1)*.1*force.fs)]
        rms.append(np.sqrt(np.mean((e-e.mean(axis=0))**2,axis=0)))
        forces.append(f.mean(axis=0))
    channels=StandardScaler().fit_transform(np.log1p(np.asarray(rms)*1e5))
    pca=PCA(n_components=3,random_state=42).fit(channels)
    X=white(pca.transform(channels))
    forces=np.asarray(forces)
    selected=np.argsort(-forces.var(axis=0))[:3]
    Y=white(forces[:,selected])
    perm=np.random.default_rng(114).permutation(250)
    Yshuffled=Y[perm]
    Ytrue=Yshuffled[np.argsort(perm)]
    cx=KMeans(n_clusters=4,random_state=42,n_init=30).fit_predict(X)
    cy=KMeans(n_clusters=4,random_state=42,n_init=30).fit_predict(Yshuffled)
    oracle_row,_=orthogonal_procrustes(X,Ytrue)
    result={'dataset':'Real Hyser sample1','samples_per_domain':250,'split':None,
       'selected_fingers':[FINGERS[i] for i in selected],
       'PCA_explained_variance':pca.explained_variance_ratio_.tolist(),
       'K':4,'params':{'mu':.05,'tol':.01,'maxiter':200,'group_gamma':.2,'group_steps':300,
           'SA_maxiter':40,'SA_tol':.01,'SA_gamma':.1,'SA_sinkhorn_steps':80},
       'paired_indices_used_in_fit':False,
       'code':'Direct import src.hiwa.hiwa.HiWA',
       'warning':'Do not use HiWA._eval_R2 (missing squared difference), or rotation rMSE without true Rgt',
       'before':score(X,Ytrue),'oracle':score(X@oracle_row,Ytrue),'runs':[]}
    for seed in a.seeds:
        np.random.seed(seed)
        m=HiWA(dim_red_method=PCA(n_components=3),normalize=False,
            maxiter=200,tol=.01,mu=.05,shorn_gamma=.2,shorn_maxiter=300,
            sa_maxiter=40,sa_tol=.01,sa_shorn_gamma=.1,sa_shorn_maxiter=80)
        m.fit(X,cx,Yshuffled,cy,X_transform=np.eye(3),Y_transform=np.eye(3))
        d=m.diagnostics
        result['runs'].append({'seed':seed,'iters':len(d['Rg_norm']),
           'ADMM_converged':bool(d['Rg_norm'][-1]<=.01 and d['admm_primal_residual'][-1]<=.01),
           'last_delta_R':float(d['Rg_norm'][-1]),
           'last_primal':float(d['admm_primal_residual'][-1]),
           'last_group_marginal_error':float(d['group_sinkhorn_marginal_error'][-1]),
           'R':m.Rg.tolist(),'metrics':score(X@m.Rg.T,Ytrue)})
    out=a.out_dir/'all250_hard_hiwa.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({'before':result['before'],'oracle':result['oracle'],
       'runs':[{k:v for k,v in r.items() if k!='R'} for r in result['runs']]},ensure_ascii=False,indent=2))
    print('saved',out)

if __name__=='__main__':main()
