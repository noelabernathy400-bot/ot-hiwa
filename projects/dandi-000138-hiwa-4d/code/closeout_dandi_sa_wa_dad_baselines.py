"""Close-out real DANDI matched-protocol SA/WA/DAD-like baselines.

Real original NWB 162-channel signals, original 4D hand pos+velocity.
Same 320 inner TRAIN / 80 internal VALID split as archived FA3/FA4 Hard/Soft,
same exact 160 train sampled rows per seed/method (source and target independent).
No original 100 TEST trial or synchronized row labels used in fitting.
SA: full-basis subspace alignment port from archived gcot/baselines.py.
WA: stochastic orthogonal Wasserstein alignment with author annealing
gamma(100 -> 0.1), 200 outer steps, entropic Sinkhorn; original paper port.
DAD-like: explicitly MODIFIED 3D only, train-sample nearest-neighbor
density-KL rotation grid+local optimizer; NOT original KordingLab MATLAB DAD.
No false claim of official DAD reproduction. Scores physical matching in
unchanged original 4D hand position/velocity after train-only fitting.
"""
from __future__ import annotations
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.decomposition import FactorAnalysis,PCA
from sklearn.preprocessing import StandardScaler
from run_hard_clustering import load_trial_windows
from zip_gcot.numerics import polar,squared_cost,sinkhorn
from preflight_geometry_semantics_retrieval import cross_trial_nearest,retrieval_metrics
from compare_pca4_fa4_inner_train_ot import raw_behavior_metrics,score_global_w2

def sa_author_port(x,y):
    def basis(z):
        std=np.std(z,axis=0,ddof=1)
        q=(z-z.mean(axis=0))/np.where(std>0,std,1.)
        return PCA(n_components=z.shape[1],svd_solver="full").fit(q).components_.T
    return (basis(x).T @ basis(y)).T

def wa_author_port(x,y,seed=42,maxiter=200,sinkhorn_maxiter=1000):
    d=x.shape[1]; rng=np.random.default_rng(seed)
    r=np.linalg.svd(rng.random((d,d)),full_matrices=False)[0]
    a=np.full(len(x),1./len(x));b=np.full(len(y),1./len(y))
    gamma=100.
    for it in range(maxiter):
        if gamma>0.1:gamma*=.95
        Q=sinkhorn(a,b,squared_cost(x@r.T,y),gamma,sinkhorn_maxiter)
        r=polar(y.T@Q.T@x)
    return r,gamma

def euler_r(a):
    a,b,c=a
    sa,ca=np.sin(a),np.cos(a)
    sb,cb=np.sin(b),np.cos(b)
    sc,cc=np.sin(c),np.cos(c)
    rx=np.array([[1,0,0],[0,ca,-sa],[0,sa,ca]])
    ry=np.array([[cb,0,sb],[0,1,0],[-sb,0,cb]])
    rz=np.array([[cc,-sc,0],[sc,cc,0],[0,0,1]])
    return rz@ry@rx

def dad_like_3d(x,y):
    """Density-KL 3D rotation search inspired by DAD, not official DAD."""
    assert x.shape[1]==y.shape[1]==3
    refk=5
    sx=cKDTree(x).query(x,k=refk+1)[0][:,refk]
    sy=cKDTree(y).query(y,k=refk+1)[0][:,refk]
    sx=np.maximum(sx,1e-8);sy=np.maximum(sy,1e-8)
    tree_y=cKDTree(y)
    def score(params,mirror):
        r=euler_r(params)@np.diag([-1.,1.,1.]) if mirror else euler_r(params)
        xx=x@r.T
        crossx=np.maximum(tree_y.query(xx,k=refk)[0][:,refk-1],1e-8)
        crossy=np.maximum(cKDTree(xx).query(y,k=refk)[0][:,refk-1],1e-8)
        return float(3*np.mean(np.log(crossx/sx))+
                     3*np.mean(np.log(crossy/sy)))
    angles=np.arange(4)*np.pi/2.
    best=(float("inf"),None,None)
    for m in (False,True):
        for a in angles:
            for b in angles:
                for c in angles:
                    th=np.array([a,b,c])
                    v=score(th,m)
                    if v<best[0]:best=(v,th,m)
    val,init,m=best
    opt=minimize(lambda z:score(z,m),init,method="Powell",
                 options={"maxiter":45,"xtol":1e-3,"ftol":2e-4})
    if float(opt.fun)<val:val=float(opt.fun);init=opt.x
    r=euler_r(init)@np.diag([-1.,1.,1.]) if m else euler_r(init)
    return r,{"fitted_symmetric_density_KL_proxy":val,
              "rotation_grid_euler_4x4x4_reflections_2":128,
              "k_nearest_density":refk,
              "local_optimization":"Powell max45",
              "original_DAD_exact_reproduction":False}

def run():
    p=argparse.ArgumentParser()
    for name in ("nwb","four-d-results","three-d-results","three-d-plan-dir","four-d-plan-dir","output"):
        p.add_argument("--"+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    d3=json.loads(args.three_d_results.read_text())
    d4=json.loads(args.four_d_results.read_text())
    nr,br,valid,meta=load_trial_windows(args.nwb)
    assert meta["sha256"]==d3["raw_nwb_sha256"]==d4["file_sha256"]
    original_test=set(np.arange(500))-set(d4["inner_train_trials"])-set(d4["inner_validation_trials"])
    assert len(original_test)==100
    trial=np.repeat(np.arange(500),35)
    trmask=np.isin(trial,d4["inner_train_trials"])
    vamask=np.isin(trial,d4["inner_validation_trials"])
    assert trmask.sum()==11200 and vamask.sum()==2800 and not np.any(trmask&vamask)
    neural=gaussian_filter1d(np.sqrt(np.maximum(nr,0)),sigma=2.,axis=1,mode="nearest").reshape(-1,162)
    hand=br.reshape(-1,4).astype(float)
    n_std=StandardScaler().fit(neural[trmask]).transform(neural)
    hand_std=StandardScaler().fit(hand[trmask]).transform(hand)
    hpca=PCA(n_components=3,svd_solver="full",random_state=42).fit(hand_std[trmask])
    path,low,_,_=raw_behavior_metrics(hand,trmask,vamask,trial)
    val_trials=trial[vamask]
    baseline=d4["random_cross_trial_baseline"]
    rng=np.random.default_rng(3026)
    valind=np.flatnonzero(vamask)
    scoreidx_x=rng.choice(len(valind),size=200,replace=False)
    scoreidx_y=rng.choice(len(valind),size=200,replace=False)
    _=rng.choice(len(valind),size=128,replace=False)
    _=rng.choice(len(valind),size=128,replace=False)
    data={"method_status":"DANDI actual SA WA and DAD-inspired modified 3D",
          "data_source":"DANDI:000138 NWB SHA256 "+meta["sha256"],
          "raw_sha256":meta["sha256"],
          "inner_train_trials":320,"inner_val_trials":80,
          "original_test100_not_used":True,
          "original_physical_4d_labels_used_only_after_train":True,
          "all_baseline_training_unpaired":True,
          "random_baseline":baseline,
          "reference_results_3d4d_match_original_trial_split":True,
          "results":{}}
    for dim in (3,4):
        feature=f"FA{dim}"
        fa=FactorAnalysis(n_components=dim,svd_method="randomized",
                          random_state=42,max_iter=120,tol=1e-3,iterated_power=4).fit(n_std[trmask])
        latent=fa.transform(n_std)
        mx=latent[trmask].mean(0)
        scale=np.sqrt(np.mean(np.sum((latent[trmask]-mx)**2,axis=1)))
        X=(latent-mx)/scale
        y0=hpca.transform(hand_std) if dim==3 else hand_std
        my=y0[trmask].mean(0)
        yscale=np.sqrt(np.mean(np.sum((y0[trmask]-my)**2,axis=1)))
        Y=(y0-my)/yscale
        vX=X[vamask];vY=Y[vamask]
        data["results"][feature]={}
        plan_dir=args.three_d_plan_dir if dim==3 else args.four_d_plan_dir
        for method in (["SA","WA","DAD_like_3D"] if dim==3 else ["SA","WA"]):
            results=[]
            seeds=[42] if method=="SA" else [42,43,44]
            for seed in seeds:
                with np.load(plan_dir/f"DANDI_{feature}_Hard_seed{seed}_FULL_P_Q_R.npz",allow_pickle=False) as plan:
                    a=plan["sampled_source_features"]
                    b=plan["sampled_target_features"]
                    indices_a=plan["source_train_row_indices"]
                    indices_b=plan["target_train_row_indices"]
                assert a.shape==b.shape==(160,dim)
                assert set(trial[indices_a]).issubset(d4["inner_train_trials"])
                assert set(trial[indices_b]).issubset(d4["inner_train_trials"])
                start=time.perf_counter()
                if method=="SA":
                    R=sa_author_port(a,b);extra={"SA_full_basis_train_only":True}
                elif method=="WA":
                    R,gamma=wa_author_port(a,b,seed=seed)
                    extra={"anneal_gamma_start":100.,"anneal_gamma_end":gamma,
                           "outer":200,"sinkhorn_maxiter":1000}
                else:
                    R,extra=dad_like_3d(a,b)
                runtime=time.perf_counter()-start
                assert R.shape==(dim,dim) and np.linalg.norm(R.T@R-np.eye(dim))<1e-7
                ix=cross_trial_nearest(vX@R.T,vY,val_trials)
                physical=retrieval_metrics(ix,hand[vamask],low,path)
                w2=score_global_w2(vX,vY,R,scoreidx_x,scoreidx_y)
                row={"method":method,"feature":feature,"seed":seed,
                     "validation_free_W2sq":w2,
                     "validation_physical_cross_trial":physical,
                     "runtime_sec":runtime,
                     "train_source_indices_sha256":hashlib.sha256(indices_a.tobytes()).hexdigest(),
                     "train_target_indices_sha256":hashlib.sha256(indices_b.tobytes()).hexdigest(),
                     "R":R.tolist(),
                     "orthogonality_error":float(np.linalg.norm(R.T@R-np.eye(dim))),
                     "group_hierarchy_PQ":False,
                     "config":extra}
                results.append(row)
                print("CLOSEOUT_BASELINE",feature,method,seed,
                      "w2",round(w2,5),"pos",round(physical["pos_hit10pct"],4),
                      "dir",round(physical["direction_hit30deg"],4),
                      "sec",round(runtime,2),flush=True)
            data["results"][feature][method]=results
            (args.output/"DANDI_SA_WA_DAD_ADAPTED_RESULTS.json").write_text(
                json.dumps(data,indent=2,ensure_ascii=False)+"\n")
    assert data["results"]["FA3"]["DAD_like_3D"][0]["config"]["original_DAD_exact_reproduction"] is False
    (args.output/"DANDI_SA_WA_DAD_ADAPTED_RESULTS.json").write_text(
        json.dumps(data,indent=2,ensure_ascii=False)+"\n")
    print("CLOSEOUT_BASELINES_COMPLETE",flush=True)

if __name__=="__main__":run()
