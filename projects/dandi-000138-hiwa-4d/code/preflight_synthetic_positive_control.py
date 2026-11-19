"""Strict unpaired known-rotation POSITIVE CONTROL for original Hard/Soft solver.

Independent Gaussian samples in two domains with predeclared true global
orthogonal rotation and shared *distribution* of asymmetric K=4 groups.
All true group labels and Rgt are hidden from fitting and used POSTFIT only.
KMeans and Soft prototype groups are independently learned from X or Y.
No elementwise source-target sample pairing exists, by construction.

Two regimes: well-separated vs overlapping groups, all solver settings
matched to DANDI original. This is a mechanism sanity check, NOT evidence
that DANDI has the same geometry.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
from sklearn.cluster import KMeans
from zip_gcot.grouping import soft_groups
from zip_gcot.alignment import hierarchical_alignment,SolverConfig
from run_hard_hiwa import sample_groups
from run_controlled_soft_hard_ablation import build_measures

def synthesise(within_scale,seed=90210):
    rng=np.random.default_rng(seed)
    centers=np.array([[-2.6,-1.0,-0.6,-.4],
                      [.4,2.4,-1.4,-.8],
                      [2.4,-1.4,1.2,1.7],
                      [-1.4,.5,2.9,-1.6]])
    blocks=[]
    for k in range(4):
        V,_=np.linalg.qr(rng.normal(size=(4,4)))
        blocks.append(V@np.diag([.8,.6,.5,.35])*within_scale)
    G,_=np.linalg.qr(rng.normal(size=(4,4)))
    if np.linalg.det(G)<0:G[:,0]*=-1
    def sample(n,rotate=False):
        xx=[];lab=[]
        for k in range(4):
            data=centers[k]+rng.normal(size=(n,4))@blocks[k].T
            if rotate:data=data@G.T
            xx.append(data);lab.extend([k]*n)
        order=rng.permutation(4*n)
        return np.concatenate(xx)[order],np.asarray(lab)[order]
    xtrain,xtrue=sample(150)
    ytrain,ytrue=sample(150,True)
    xtest,lxtest=sample(90)
    ytest,lytest=sample(90,True)
    meanx=xtrain.mean(0);meany=ytrain.mean(0)
    radx=np.sqrt(np.mean(np.sum((xtrain-meanx)**2,axis=1)))
    rady=np.sqrt(np.mean(np.sum((ytrain-meany)**2,axis=1)))
    xtr=(xtrain-meanx)/radx
    ytr=(ytrain-meany)/rady
    xte=(xtest-meanx)/radx
    yte=(ytest-meany)/rady
    return dict(xtrain=xtr,ytrain=ytr,xtest=xte,ytest=yte,
                source_gt_train=xtrue,target_gt_train=ytrue,
                source_gt_test=lxtest,target_gt_test=lytest,
                true_rotation=G)

def info_for_P(P,cx,cy,xtruth,ytruth):
    """True labels used postfit to interpret unsupervised cluster IDs."""
    majx=[int(np.bincount(xtruth[cx==k],minlength=4).argmax()) for k in range(4)]
    majy=[int(np.bincount(ytruth[cy==k],minlength=4).argmax()) for k in range(4)]
    truth=np.array([[a==b for b in majy] for a in majx])
    return dict(true_group_matching_mass=float(np.sum(P*truth)),
                random_uniform_group_matching_mass=float(truth.mean()),
                source_kmeans_majority_true_group=majx,target_kmeans_majority_true_group=majy,
                source_kmeans_true_group_purity=float(np.mean([majx[k]==t for k,t in zip(cx,xtruth)])),
                target_kmeans_true_group_purity=float(np.mean([majy[k]==t for k,t in zip(cy,ytruth)])))

def apply_eval(X,Y,src_true,tgt_true,R,Rgt):
    nn=cKDTree(Y).query(X@R.T,k=1)[1]
    b=cKDTree(Y).query(X,k=1)[1]
    rng=np.random.default_rng(20261119)
    ix=rng.choice(len(X),120,replace=False)
    iy=rng.choice(len(Y),120,replace=False)
    D0=cdist(X[ix],Y[iy],"sqeuclidean")
    D1=cdist((X@R.T)[ix],Y[iy],"sqeuclidean")
    a,b1=linear_sum_assignment(D0);c,d=linear_sum_assignment(D1)
    return dict(
       test_true_group_nearest_neighbor_acc=float(np.mean(src_true==tgt_true[nn])),
       test_true_group_identity_nearest_neighbor_acc=float(np.mean(src_true==tgt_true[b])),
       test_global_free_W2sq=float(np.mean(D1[c,d])),
       test_global_free_W2sq_before=float(np.mean(D0[a,b1])),
       rotation_frobenius_error=float(np.linalg.norm(R-Rgt,"fro")),
       group_majority_accuracy_100pct_reference=1.)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seeds",nargs="+",type=int,default=[42,43,44])
    p.add_argument("--outer",type=int,default=500)
    p.add_argument("--inner",type=int,default=20)
    p.add_argument("--sinkhorn",type=int,default=45)
    p.add_argument("--group-sinkhorn",type=int,default=200)
    p.add_argument("--prototype-maxiter",type=int,default=300)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    cfg=SolverConfig(maxiter=a.outer,tol=1e-9,mu=.05,epsilon_group=.2,
                     epsilon_sample=.1,inner_maxiter=a.inner,inner_tol=.01,
                     sinkhorn_maxiter=a.sinkhorn,group_sinkhorn_maxiter=a.group_sinkhorn,
                     sinkhorn_tol=1e-8,stopping="legacy")
    out={"synthetic_only":True,"true_rotation_known_only_postfit":True,
         "true_group_labels_known_only_postfit":True,
         "unpaired_independent_generation":True,
         "source_target_cluster_estimation_independent":True,
         "P_freely_optimized":True,
         "no_real_DANDI_data_used":True,
         "settings":{"seeds":a.seeds,"outer":a.outer,"inner":a.inner,
                     "sample_sinkhorn":a.sinkhorn,"group_sinkhorn":a.group_sinkhorn,
                     "mu":.05,"epsilon_sample":.1,"epsilon_group":.2,
                     "n_train_per_group":150,"n_test_per_group":90,
                     "ot_subsample_per_hard_group":40},"regimes":{}}
    for mode,scale in [("well_separated",.28),("overlapping",1.3)]:
        d=synthesise(scale)
        X=d["xtrain"];Y=d["ytrain"]
        kh=KMeans(n_clusters=4,random_state=42,n_init=20).fit(X)
        kt=KMeans(n_clusters=4,random_state=42,n_init=20).fit(Y)
        hx=kh.labels_;hy=kt.labels_
        ax,cx,infox=soft_groups(X,4,seed=42,entropy_weight=1.0,maxiter=a.prototype_maxiter)
        ay,cy,infoy=soft_groups(Y,4,seed=43,entropy_weight=1.0,maxiter=a.prototype_maxiter)
        assert np.allclose(ax.sum(axis=1),1) and np.allclose(ay.sum(axis=1),1)
        assert not np.array_equal(hx,hy)
        runs={"hard":[],"soft":[]}
        for seed in a.seeds:
            g=np.random.default_rng(seed)
            ix=sample_groups(np.arange(len(X)),hx,40,g)
            iy=sample_groups(np.arange(len(Y)),hy,40,g)
            assert len(ix)==len(iy)==160
            for meth in ["hard","soft"]:
                soft=meth=="soft"
                sx=ax[ix] if soft else np.eye(4)[hx[ix]]
                sy=ay[iy] if soft else np.eye(4)[hy[iy]]
                fit=hierarchical_alignment(build_measures(X[ix],sx,not soft),
                                           build_measures(Y[iy],sy,not soft),
                                           seed=seed,config=cfg,scale=True,
                                           retain_plans=False)
                R=fit["R"];P=fit["P"]
                assert np.max(abs(P.sum(0)-.25))<1e-6 and np.max(abs(P.sum(1)-.25))<1e-6
                e=apply_eval(d["xtest"],d["ytest"],d["source_gt_test"],
                             d["target_gt_test"],R,d["true_rotation"])
                e.update(info_for_P(P,hx,hy,d["source_gt_train"],d["target_gt_train"]))
                e.update(seed=int(seed),method=meth,strict_admm_pass=bool(fit["criteria_satisfied"]),
                         iteration_count=len(fit["history"]),residual_primal=float(fit["consensus_residual"]),
                         residual_rotation=float(fit["rotation_delta"]),
                         P=P.tolist(),R=R.tolist())
                runs[meth].append(e)
                print(f"POSITIVE_CONTROL {mode}/{meth}/seed{seed} class_before="
                      f"{e['test_true_group_identity_nearest_neighbor_acc']:.3f} "
                      f"class_after={e['test_true_group_nearest_neighbor_acc']:.3f} "
                      f"P_mass_correct={e['true_group_matching_mass']:.3f} "
                      f"Rgt_err={e['rotation_frobenius_error']:.3f} "
                      f"strict={e['strict_admm_pass']}",flush=True)
        out["regimes"][mode]=dict(within_cluster_scale=scale,
            source_hard_counts=np.bincount(hx,minlength=4).tolist(),
            target_hard_counts=np.bincount(hy,minlength=4).tolist(),
            source_soft_prototypes_success=infox["success"],target_soft_prototypes_success=infoy["success"],
            runs=runs)
        (a.output/"SYNTHETIC_POSITIVE_CONTROL_RESULTS.json").write_text(
            json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    methods=["hard","soft"]
    fig,axs=plt.subplots(2,2,figsize=(12,8.2),constrained_layout=True)
    for row,(mode,v) in enumerate(out["regimes"].items()):
        for col,key in enumerate(["test_true_group_nearest_neighbor_acc","true_group_matching_mass"]):
            ax=axs[row,col]
            data=[[r[key] for r in v["runs"][method]] for method in methods]
            means=[np.mean(t) for t in data]
            sds=[np.std(t,ddof=1) if len(t)>1 else 0 for t in data]
            ax.bar([0,1],means,yerr=sds,capsize=3,color=["#3982b9","#dc9248"])
            ax.set(xticks=[0,1],xticklabels=["Hard","Soft"],ylim=(0,1.05),
                   title=f"{mode}, {['physical true-group transfer NN','correct true-group mass in P'][col]}")
            ax.axhline(.25,color="black",linestyle="--",alpha=.6,label="4-group chance")
            ax.grid(axis="y",alpha=.12)
    fig.suptitle("Synthetic control: TRUE common orthogonal distribution exists by construction\n"
                 "Independent X/Y draws, KMeans/Soft independent; P is FREE; heldout true groups used AFTER fitting")
    fig.savefig(a.output/"SYNTHETIC_POSITIVE_CONTROL_HARD_SOFT.png",
                dpi=190,bbox_inches="tight")
    plt.close(fig)
    lines=["# Synthetic positive-control: before any expensive new real-data model",
           "",
           "**The two domains are independent samples from the SAME four asymmetric Gaussian cluster distributions, up to one known orthogonal rotation R*.** Ground truth is used ONLY after fitting; independent K-means / soft prototypes are learned on each domain, and P is freely optimized. This is a necessary algorithm positive control, not DANDI evidence.",
           "",
           "Solver control: original DANDI budgets (outer500/inner20/sample Sinkhorn45/group Sinkhorn200, eps_s=.1,eps_g=.2,mu=.05); same sampling 40 per hard group, 160/domain, seeds 42–44.",
           "",
           "| Geometry regime | Method | Heldout true-group 1NN acc↑ | Identity 1NN acc | P true-group mass↑ (chance=.25) | Rgt Frobenius error↓ | Strict ADMM |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for mode,rs in out["regimes"].items():
        for method in methods:
            runs=rs["runs"][method]
            avg=lambda k:float(np.mean([r[k] for r in runs]))
            lines.append(f"| {mode} | {method} | {avg('test_true_group_nearest_neighbor_acc'):.4f} | "
                f"{avg('test_true_group_identity_nearest_neighbor_acc'):.4f} | "
                f"{avg('true_group_matching_mass'):.4f} | "
                f"{avg('rotation_frobenius_error'):.4f} | "
                f"{sum(t['strict_admm_pass'] for t in runs)}/{len(runs)} |")
    lines+=["","![Hard-vs-Soft true structure positive control](SYNTHETIC_POSITIVE_CONTROL_HARD_SOFT.png)",
      "",
      "**Interpretation**: The ground-truth class labels do NOT enter training. Synthetic recovery demonstrates only behavior under this designed known structure. Compare to the real DANDI read-only preflight to separate algorithm pathologies from poor real-domain representation.",
      "",
      "[Seed-level P, R, ground truth accuracy audit](SYNTHETIC_POSITIVE_CONTROL_RESULTS.json) | [Generator and code](../../code/preflight_synthetic_positive_control.py)",
      ""]
    (a.output/"SYNTHETIC_POSITIVE_CONTROL_REPORT.md").write_text("\n".join(lines),encoding="utf8")
    print("POSITIVE_CONTROL_COMPLETED",flush=True)

if __name__=="__main__":
    main()
