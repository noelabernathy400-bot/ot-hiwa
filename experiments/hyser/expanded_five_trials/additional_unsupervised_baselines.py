#!/usr/bin/env python3
"""Five-trial full1000 Hyser extra unsupervised controls.

No paired EMG/force index in learning. Inductive prediction searches the
1000-point target TRAIN bank; test forces are used ONLY for post-fit scores.

Controls are NOT original author SA/WA/DAD implementations:
- identity/no rotation
- unpaired nearest-neighbour orthogonal ICP (trimmed, multi-start)
- single-level entropic OT with orthogonal Procrustes alternating updates.

No tuning on heldout trials. Report failures, seed sensitivity and numerical
residuals. The 3D vs 5D results share physical original-force units but are
not comparable by latent W2 in absolute scale.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.linalg import svd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPS=("PCA3_PCA3","FA3_FA3","PCA5_RAW5","FA5_RAW5")
SEEDS=(42,43)
EPS=.2
MAX_STEPS=16
SINKHORN_STEPS=110

def rot(C):
    U,_,Vh=svd(C,full_matrices=False)
    R=U@Vh
    assert np.all(np.isfinite(R))
    return R

def random_rotation(d,seed):
    rng=np.random.default_rng(seed)
    return rot(rng.normal(size=(d,d)))

def solve_icp(X,Y,seed):
    d=X.shape[1]
    R=random_rotation(d,seed)
    tree=cKDTree(Y)
    history=[]
    for it in range(MAX_STEPS):
        dist,ids=tree.query(X@R.T,k=1)
        keep=dist<=np.quantile(dist,.80)
        # Trimming is fixed a priori, not selected from heldout response.
        C=Y[ids[keep]].T@X[keep]
        new=rot(C)
        delta=float(np.linalg.norm(new-R))
        R=new
        history.append(delta)
        if delta<1e-5:break
    return R,{"iterations":len(history),"last_rotation_change":history[-1],
              "solver":"unpaired trimmed nearest-neighbor ICP"}

def sinkhorn_equal(C,eps=EPS,niter=SINKHORN_STEPS):
    # Single-layer entropic OT, n x m equally weighted, no class/true correspondence.
    n,m=C.shape
    K=np.exp(-np.minimum((C-C.min())/eps,700.))
    K=np.maximum(K,1e-300)
    a=np.full(n,1./n);b=np.full(m,1./m)
    v=np.ones(m);u=np.ones(n)
    for _ in range(niter):
        u=a/np.maximum(K@v,1e-300)
        v=b/np.maximum(K.T@u,1e-300)
    Q=(u[:,None]*K)*v[None,:]
    err=max(float(np.max(abs(Q.sum(axis=1)-a))),
            float(np.max(abs(Q.sum(axis=0)-b))))
    return Q,err

def solve_single_ot(X,Y,seed):
    d=X.shape[1]
    R=random_rotation(d,seed)
    hist=[];maxerr=0.
    for it in range(MAX_STEPS):
        D=cdist(X@R.T,Y,metric="sqeuclidean")
        Q,err=sinkhorn_equal(D)
        maxerr=max(maxerr,err)
        C=Y.T@Q.T@X
        new=rot(C)
        delta=float(np.linalg.norm(new-R))
        R=new;hist.append(delta)
        if delta<1e-5:break
    return R,{"iterations":len(hist),"last_rotation_change":hist[-1],
              "max_sinkhorn_marginal_error":maxerr,
              "solver":"single-layer entropic unpaired OT-Procrustes"}

def evaluate(Xtest,Ytrain,Yphys_train,Yphys_test,R,thresholds):
    idx=np.argmin(cdist(Xtest@R.T,Ytrain),axis=1)
    pred=Yphys_train[idx]
    e=np.linalg.norm(pred-Yphys_test,axis=1)
    return {"hit_q05":float(np.mean(e<=thresholds["0.05"])),
            "hit_q20":float(np.mean(e<=thresholds["0.2"])),
            "hit_eps_0325":float(np.mean(e<=.325)),
            "mean_fivefinger_l2":float(e.mean()),
            "rmse_fivefinger_components":float(np.sqrt(np.mean((pred-Yphys_test)**2))),
            "R2_vs_training_finger_mean":float(1-
              np.sum((pred-Yphys_test)**2)/
              np.maximum(np.sum((Yphys_test-Yphys_train.mean(axis=0))**2),1e-12)),
            "n_test":int(len(e))}

def savecsv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as fp:
        wr=csv.DictWriter(fp,fieldnames=list(rows[0]))
        wr.writeheader();wr.writerows(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--features",type=Path,required=True)
    p.add_argument("--existing",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    ref={}
    for dim in ("3D","5D"):
        obj=json.loads((a.existing/dim/"full_physical_distance_curves.json").read_text())
        assert obj["n_runs"]==40
        for r in obj["records"]:
            ref[r["config"],r["method"],r["fold"],r["seed"]]=r
    assert len(ref)==80
    rows=[]
    for conf in REPS:
        d=3 if conf.endswith("_PCA3") or conf.endswith("_FA3") else 5
        for fold in range(1,6):
            fname=f"trial_heldout_{fold}"+("_5d" if d==5 else "")+".npz"
            with np.load(a.features/conf/fname) as z:
                X=np.asarray(z["X_train"],float)/np.sqrt(d)
                Y=np.asarray(z["Y_train"],float)/np.sqrt(d)
                Xt=np.asarray(z["X_test"],float)/np.sqrt(d)
                fy=np.asarray(z["original_target_force_train"],float)
                ft=np.asarray(z["original_target_force_test"],float)
            assert X.shape==Y.shape==(1000,d) and Xt.shape==(250,d)
            assert fy.shape==(1000,5) and ft.shape==(250,5)
            qs=ref[conf,"hard",fold,42]["train_only_thresholds"]
            assert ref[conf,"soft05",fold,43]["train_only_thresholds"]==qs
            for method in ("identity","icp","single_ot"):
                seeds=(0,) if method=="identity" else SEEDS
                for seed in seeds:
                    try:
                        if method=="identity":
                            R=np.eye(d);info={"solver":"identity,no fit"}
                        elif method=="icp":
                            R,info=solve_icp(X,Y,seed)
                        else:
                            R,info=solve_single_ot(X,Y,seed)
                        assert R.shape==(d,d)
                        orth=float(np.linalg.norm(R.T@R-np.eye(d)))
                        if orth>1e-7:raise ValueError("nonorthogonal R")
                        if info.get("max_sinkhorn_marginal_error",0)>1e-5:
                            raise ValueError("single-level Q marginals failed")
                        sc=evaluate(Xt,Y,fy,ft,R,qs)
                        row={"representation":conf,"fold":fold,"method":method,
                             "seed":seed,"qualified":True,"error":"","orthogonality":orth,
                             **sc,**info}
                    except Exception as ex:
                        row={"representation":conf,"fold":fold,"method":method,
                             "seed":seed,"qualified":False,"error":str(ex)}
                    rows.append(row)
                    print(conf,fold,method,seed,row["qualified"],
                          row.get("hit_q05"),flush=True)
    # all rows retained, including failures
    fields=sorted({k for r in rows for k in r})
    with (a.out/"all_baselines_by_fold_seed.csv").open("w",newline="",encoding="utf-8") as fp:
        w=csv.DictWriter(fp,fieldnames=fields);w.writeheader();w.writerows(rows)
    summary=[]
    for conf in REPS:
        for method in ("identity","icp","single_ot","hiwa_hard","hiwa_soft"):
            if method.startswith("hiwa_"):
                typ="hard" if method=="hiwa_hard" else "soft05"
                r=[ref[conf,typ,fold,seed] for fold in range(1,6) for seed in SEEDS]
                pts=[{"hit_q05":x["results"]["physical_hit"]["0.05"],
                      "hit_q20":x["results"]["physical_hit"]["0.2"],
                      "mean_fivefinger_l2":x["results"]["physical_error_mean"]}
                      for x in r]
                n_total=10;n_valid=10
            else:
                r=[x for x in rows if x["representation"]==conf and x["method"]==method]
                pts=[x for x in r if x["qualified"]];n_total=len(r);n_valid=len(pts)
            summary.append({"representation":conf,"method":method,
                  "n_run":n_total,"n_qualified":n_valid,
                  "mean_hit_q05":float(np.mean([x["hit_q05"] for x in pts])) if pts else None,
                  "mean_hit_q20":float(np.mean([x["hit_q20"] for x in pts])) if pts else None,
                  "mean_physical_l2":float(np.mean([x["mean_fivefinger_l2"] for x in pts])) if pts else None})
    savecsv(a.out/"unpaired_baseline_summary.csv",summary)
    (a.out/"provenance.json").write_text(json.dumps({
      "n_full1000_HiWA_frozen_models":len(ref),
      "references":"frozen existing 3D/5D physical errors and original fivefinger arrays",
      "training":"unpaired source and target clouds; target TRAIN bank only",
      "inductive":"test source only; true test force exclusively used in scoring",
      "baseline_parameters":{"single_layer_epsilon":EPS,"sinkhorn_iterations":SINKHORN_STEPS,
                             "outer_steps":MAX_STEPS,"ICP_trimming_fraction":.8,
                             "seeds":list(SEEDS)},
      "controls_implemented":["identity","unpaired ICP","single-layer entropic OT-Procrustes"],
      "not_claimed":["original SA/WA/DAD implementations","optimal single-level solver",
                     "independent subjects","statistical population significance"]},
      ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    for ax,reps in zip(axes,(REPS[:2],REPS[2:])):
        group=[x for x in summary if x["representation"] in reps]
        x=np.arange(len(group))
        labels=[x["representation"].split("_")[0]+" "+x["method"] for x in group]
        ax.barh(x,[100*(z["mean_hit_q05"] or 0) for z in group])
        ax.set_yticks(x,labels,fontsize=8)
        ax.set(xlabel="Physical Hit@training q05 (%)",
               title=f"{3 if reps[0].startswith(('PCA3','FA3')) else 5}D full1000 unpaired baselines")
        ax.invert_yaxis()
    fig.savefig(a.out/"hyser_additional_baselines_hit_q05.png",dpi=170)
    plt.close(fig)
    md=["# Hyser S01/S1 五折全1000点：额外无监督对照",
        "","这些是新增实际拟合的基线；不能声称为 SA、WA 或 DAD 原论文的复现。",
        "源、目标均仅用当前外折的四个训练试次拟合，每种方法对250测试肌电进行归纳式目标训练库检索。",
        "测试真实五指力量只用于事后评分。表中两随机种子在同一被试同一试次上不是独立样本。",
        "","|表示|方法|有效/拟合|Hit@5%|Hit@20%|五指平均物理误差|","|---|---|---:|---:|---:|---:|"]
    for r in summary:
        pp=lambda x:"NA" if x is None else f"{x:.2%}"
        em="NA" if r["mean_physical_l2"] is None else f"{r['mean_physical_l2']:.5f}"
        md.append(f"|{r['representation']}|{r['method']}|{r['n_qualified']}/{r['n_run']}|{pp(r['mean_hit_q05'])}|{pp(r['mean_hit_q20'])}|{em}|")
    md+=["","## 解释","1. 这些是额外无监督控制，初始化与优化不等同原HiWA；非凸性导致不同种子结果不稳定。",
         "2. 本对照的公平性在于完全相同的四试次训练、同一外折和物理评价口径；不是所有算法具备同样数学归纳偏置或计算预算。",
         "3. 证明优于随机并不意味着优于单层OT；以表中真实数值为准。",
         "4. 第五次试次应单独检查，且五折不能当作五位不同受试者。",
         "","![新增无监督基线](hyser_additional_baselines_hit_q05.png)"]
    (a.out/"ADDITIONAL_UNSUPERVISED_BASELINES_ZH.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    print(json.dumps({"done":True,"runs":len(rows),"qualified":sum(r["qualified"] for r in rows),
                      "summary":summary},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
