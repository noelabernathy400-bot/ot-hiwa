#!/usr/bin/env python3
"""3D PCA/FA controls: TRAIN-only best 2D plus 3D heldout NN validation.

Uses four previously frozen 3D (both domains) K4 Hard/Soft runs.
No new OT optimization, no labels in training. All 250 heldout windows used.
"""
from __future__ import annotations
import argparse,json
from itertools import combinations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from heldout_best2d_and_nn import train_radius,evaluate,random_baseline

CONFS=("PCA3_PCA3","FA3_PCA3","PCA3_FA3","FA3_FA3")
COLORS=["#2677AA","#DF9953","#4F9C83","#9357A5"]
def best(X,lab):
    return max([(i,j,float(silhouette_score(X[:,[i,j]],lab)))
                for i,j in combinations(range(3),2)],key=lambda x:x[2])
def draw(x,y,z,cx,cy,bx,by,out):
    fig,axs=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for ax,pts,labels,pair,name in zip(axs,(x,y,z),(cx,cy,cx),(bx,by,by),
        ("Source heldout","Target heldout","Aligned source heldout")):
        for k,color in enumerate(COLORS):
            m=labels==k
            ax.scatter(pts[m,pair[0]],pts[m,pair[1]],c=color,s=18,alpha=.73,
                       edgecolors="none",rasterized=True)
        ax.set(xlabel=f"dim{pair[0]+1}",ylabel=f"dim{pair[1]+1}",title=name)
        ax.set_aspect("equal",adjustable="datalim");ax.grid(alpha=.15)
    fig.suptitle("3D baseline: best 2-coordinate training K4 views, chosen WITHOUT test data")
    fig.savefig(out/"best_3D_to_2D_views_independent.png",dpi=175,bbox_inches="tight")
    plt.close(fig)
    fig=plt.figure(figsize=(15,5))
    for j,(pts,labels,name) in enumerate(zip((x,y,z),(cx,cy,cx),
                  ("Source","Target","Transformed source")),1):
        ax=fig.add_subplot(1,3,j,projection="3d")
        for k,color in enumerate(COLORS):
            m=labels==k
            ax.scatter(pts[m,0],pts[m,1],pts[m,2],c=color,alpha=.73,
                       s=13,depthshade=False)
        ax.set(xlabel="dim1",ylabel="dim2",zlabel="dim3",title=name)
    fig.suptitle("ALL three latent dimensions; independent domain figures")
    fig.savefig(out/"threeD_complete_point_clouds.png",dpi=175,bbox_inches="tight")
    plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--features",type=Path,required=True)
    p.add_argument("--frozen-results",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    frozen=json.loads(a.frozen_results.read_text())["runs"]
    rows=[]
    for conf in CONFS:
        for fold in range(1,6):
            with np.load(a.features/conf/f"trial_heldout_{fold}.npz") as z:
                Xtr=np.asarray(z["X_train"])/np.sqrt(3)
                Ytr=np.asarray(z["Y_train"])/np.sqrt(3)
                Xt=np.asarray(z["X_test"])/np.sqrt(3)
                Yt=np.asarray(z["Y_test"])/np.sqrt(3)
                trials=np.asarray(z["train_trial_id"])
            kmx=KMeans(4,n_init=30,random_state=42).fit(Xtr)
            kmy=KMeans(4,n_init=30,random_state=43).fit(Ytr)
            bx=best(Xtr,kmx.labels_);by=best(Ytr,kmy.labels_)
            radius=train_radius(Ytr,trials)
            chance=random_baseline(Yt,radius["radius"])
            for seed in (42,43):
                for mode in ("hard","soft05"):
                    row=next(x for x in frozen if x["representation"]==conf
                             and x["fold"]==fold and x["seed"]==seed
                             and x["method"]==mode)
                    R=np.asarray(row["R"])
                    assert R.shape==(3,3)
                    e=evaluate(Xt,Yt,R,radius["radius"])
                    rows.append({"config":conf,"fold":fold,"seed":seed,
                        "method":mode,"d":3,
                        "training_best2D_source":[bx[0]+1,bx[1]+1,bx[2]],
                        "training_best2D_target":[by[0]+1,by[1]+1,by[2]],
                        "train_only_threshold":radius["radius"],
                        "random_null_hit":chance["random_target_assignment_hit_mean"],
                        "metrics":e,"uses_test_pairs_only_after_training":True})
                    if conf=="FA3_FA3" and fold==1 and seed==42 and mode=="soft05":
                        draw(Xt,Yt,Xt@R.T,kmx.predict(Xt),kmy.predict(Yt),
                             bx[:2],by[:2],a.out)
    assert len(rows)==80
    summary=[]
    for conf in CONFS:
        for method in ("hard","soft05"):
            sample=[q for q in rows if q["config"]==conf and q["method"]==method]
            summary.append({"conf":conf,"method":method,
                "coverage":float(np.mean([q["metrics"]["target_nn_coverage_fraction"] for q in sample])),
                "physical_neighbor_correct":float(np.mean([q["metrics"]["nn_retrieval_physical_neighbor_fraction"] for q in sample])),
                "exact_index":float(np.mean([q["metrics"]["nn_retrieval_exact_time_index_fraction"] for q in sample])),
                "null_hit":float(np.mean([q["random_null_hit"] for q in sample])),
                "train_only_radius":float(np.mean([q["train_only_threshold"] for q in sample]))})
    (a.out/"3D_all_heldout_NN.json").write_text(json.dumps({"runs":rows,"summary":summary,
        "note":"3D force representations PCA3/FA3 were fitted on all five original finger force channels, NOT old three selected fingers."},
        ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    report=["# 3D Hard/Soft nearest-neighbor and best training-only 2D projections",
        "3D features X/Y fitted only on four training trials. These are new three-dimensional reductions of full five-finger target, NOT old hand-picked thumb/middle/index three-dimensional datasets.",
        "Threshold calibrated from training target domain unrelated trial pair distances (5% quantile); score both mere NN distance coverage and true synchronized physical-neighborhood retrieval.",
        "Cannot compare absolute tolerance or W2 across 3D versus 5D representation.",
        "|Config|Method|train radius|cloud coverage|correct physical-neighborhood retrieval|random null|exact time index|",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        report.append(f"|{r['conf']}|{r['method']}|{r['train_only_radius']:.4f}|{r['coverage']:.3%}|{r['physical_neighbor_correct']:.3%}|{r['null_hit']:.3%}|{r['exact_index']:.3%}|")
    (a.out/"README_ZH.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(json.dumps({"success":True,"n_runs":len(rows),"summary":summary},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
