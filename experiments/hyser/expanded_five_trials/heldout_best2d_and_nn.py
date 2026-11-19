#!/usr/bin/env python3
"""Heldout nearest-neighbor validation and objective 5D->2D visuals.

Inputs are AUTHORITATIVE real Hyser trial-heldout PCA5/FA5 feature archives and
previously frozen 120-iter full-5D Hard/Soft global-R pilots. This script:
* NEVER trains R, never adjusts hyperparameters/selection by heldout accuracy.
* Selects the most K=4 separable 2-coordinate view (out of 10), FOR EACH DOMAIN,
  using its own TRAINING points only, separately.
* Draws all ten 2D views (as a 5x5 scatter matrix), best 2D, and 3D.
* Compares the nearest target TEST point to the truly synchronized target
  ONLY as external evaluation. "Nearest distance under radius" separately
  reported as mere coverage and NOT used as prediction correctness.
* Calibrates the five-dimensional success radius on TRAINING target
  random unrelated distinct-window distances at a fixed 5% quantile.
* Reports time-index top-1, target-target physical-neighborhood retrieval,
  unpaired target null, identity, per-subject/trial stratification.
"""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from itertools import combinations

CONFS=("PCA5_RAW5","FA5_RAW5")
METHODS=("hard","soft05")
PAIRS=list(combinations(range(5),2))
DIM=5
DEFAULT_QUANTILE=.05
RADIUS_SEED=2077
FIXED_FOLD=1
FIXED_SEED=42

def train_radius(Ytrain, trials, q=DEFAULT_QUANTILE):
    # independent random pairs; reject exact same observation as well as
    # same-record near-time windows (serial autocorrelation)
    rng=np.random.default_rng(RADIUS_SEED)
    indices=np.arange(len(Ytrain))
    ix=[];jx=[]
    it=0
    while len(ix)<20000 and it<100000:
        a,b=rng.integers(0,len(Ytrain),2)
        it+=1
        if a==b or trials[a]==trials[b]:
            continue
        ix.append(a);jx.append(b)
    vals=np.linalg.norm(Ytrain[np.asarray(ix)]-Ytrain[np.asarray(jx)],axis=1)
    r=float(np.quantile(vals,q))
    return {"radius":r,"quantile":q,"train_random_nonmatching_pairs":len(vals),
            "train_random_distribution_p10":float(np.quantile(vals,.1)),
            "train_random_distribution_median":float(np.median(vals)),
            "calibration":"0.05 quantile of training TARGET pairs from different trials; trained ONLY once per fold; same for Hard/Soft within representation"}

def best_pair(Ztrain,cluster):
    scores=[]
    for aa,bb in PAIRS:
        score=float(silhouette_score(Ztrain[:,[aa,bb]],cluster,
            metric="euclidean"))
        scores.append((aa,bb,score))
    return sorted(scores,key=lambda x:(-x[2],x[0],x[1]))[0],scores

def best_triplet(Ztrain,cluster):
    scores=[]
    for ids in combinations(range(5),3):
        score=float(silhouette_score(Ztrain[:,list(ids)],cluster))
        scores.append((*ids,score))
    return sorted(scores,key=lambda x:(-x[3],x[0],x[1],x[2]))[0],scores

def scatter(ax,Z,labs,coords,title):
    cols=["#2677AA","#DF9953","#4F9C83","#9357A5"]
    for k in range(4):
        mask=labs==k
        ax.scatter(Z[mask,coords[0]],Z[mask,coords[1]],color=cols[k],
                   alpha=.70,s=17,edgecolors="none",rasterized=True,label=f"C{k+1}")
    ax.set(xlabel=f"dimension {coords[0]+1}",ylabel=f"dimension {coords[1]+1}",
           title=title)
    ax.grid(alpha=.12)
    ax.set_aspect("equal",adjustable="datalim")

def draw(Zx,Zy,Zxrot,labx,laby,selected,triplets,reportdir,info):
    reportdir.mkdir(parents=True,exist_ok=True)
    ix,iy=selected
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    scatter(axes[0],Zx,labx,ix,"Source: best TRAIN-only 2D")
    scatter(axes[1],Zy,laby,iy,"Target: best TRAIN-only 2D")
    scatter(axes[2],Zxrot,labx,iy,"Transformed source in TARGET view")
    fig.suptitle("Hyser heldout trial 1, seed 42; separate source / target views, NOT overlaid")
    fig.savefig(reportdir/"fiveD_best_train_2D_three_independent_panels.png",
                dpi=165,bbox_inches="tight")
    plt.close(fig)
    # 5D display as every coordinate pair: 10 views, explicitly NOT a 5D plane
    for nam,Z,lab in [("source",Zx,labx),("target",Zy,laby),
                     ("transformed_source",Zxrot,labx)]:
        fig,axes=plt.subplots(5,5,figsize=(14,14),constrained_layout=True)
        for a in range(5):
            for b in range(5):
                ax=axes[a,b]
                if a==b:
                    ax.hist(Z[:,a],bins=25,alpha=.85,color="#6A879B")
                else:
                    for k,c in enumerate(["#2677AA","#DF9953","#4F9C83","#9357A5"]):
                        m=lab==k
                        ax.scatter(Z[m,b],Z[m,a],c=c,s=4,alpha=.5,rasterized=True)
                if a==4:ax.set_xlabel(f"d{b+1}")
                else:ax.set_xticklabels([])
                if b==0:ax.set_ylabel(f"d{a+1}")
                else:ax.set_yticklabels([])
                ax.grid(alpha=.08)
        fig.suptitle("Hyser "+nam+": ALL 5D coordinate-pair views; heldout, independent",fontsize=15)
        fig.savefig(reportdir/f"{nam}_all_5D_pairwise_views.png",dpi=145,bbox_inches="tight")
        plt.close(fig)
    fig=plt.figure(figsize=(16,5))
    from mpl_toolkits.mplot3d import Axes3D
    for idx,(name,Z,lab,coord) in enumerate([
        ("source",Zx,labx,triplets[0]),
        ("target",Zy,laby,triplets[1]),
        ("transformed source",Zxrot,labx,triplets[1])],start=1):
        ax=fig.add_subplot(1,3,idx,projection="3d")
        for k,c in enumerate(["#2677AA","#DF9953","#4F9C83","#9357A5"]):
            m=lab==k
            ax.scatter(Z[m,coord[0]],Z[m,coord[1]],Z[m,coord[2]],
                       color=c,s=14,alpha=.7,depthshade=False)
        ax.set(xlabel=f"d{coord[0]+1}",ylabel=f"d{coord[1]+1}",
               zlabel=f"d{coord[2]+1}",title=name)
    fig.suptitle("3D coordinate triplets selected by TRAIN-only silhouette")
    fig.savefig(reportdir/"best_3D_views.png",dpi=175,bbox_inches="tight")
    plt.close(fig)
    (reportdir/"figure_selection_training_only.json").write_text(
        json.dumps(info,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def evaluate(Xtest,Ytest,R,r):
    Z=Xtest@R.T
    dist=cdist(Z,Ytest,metric="euclidean")
    nn=np.argmin(dist,axis=1)
    best=dist[np.arange(len(Xtest)),nn]
    real=np.linalg.norm(Ytest[nn]-Ytest[np.arange(len(Ytest))],axis=1)
    paired=np.linalg.norm(Z-Ytest,axis=1)
    k5=np.argpartition(dist,4,axis=1)[:,:5]
    return {
       "target_nn_coverage_fraction":float(np.mean(best <= r)),
       "target_nn_coverage_distance_mean":float(np.mean(best)),
       "target_nn_coverage_distance_median":float(np.median(best)),
       "nn_retrieval_exact_time_index_fraction":float(np.mean(nn==np.arange(len(nn)))),
       "nn_retrieval_top5_time_index_fraction":float(np.mean(np.any(k5==np.arange(len(nn))[:,None],axis=1))),
       "nn_retrieval_physical_neighbor_fraction":float(np.mean(real<=r)),
       "nn_retrieval_physical_neighbor_error_mean":float(np.mean(real)),
       "nn_retrieval_physical_neighbor_error_median":float(np.median(real)),
       "paired_aligned_distance_lt_r_fraction":float(np.mean(paired<=r)),
       "paired_aligned_distance_mean":float(np.mean(paired)),
       "n_test":int(len(nn)),
    }

def random_baseline(Ytest,r,repeats=500):
    rng=np.random.default_rng(3201)
    true=np.arange(len(Ytest))
    hits=[]
    for _ in range(repeats):
        rr=rng.permutation(len(Ytest))
        real=np.linalg.norm(Ytest[rr]-Ytest[true],axis=1)
        hits.append(np.mean(real<=r))
    return {
      "random_target_assignment_hit_mean":float(np.mean(hits)),
      "random_target_assignment_hit_p5":float(np.percentile(hits,5)),
      "random_target_assignment_hit_p95":float(np.percentile(hits,95)),
      "random_top1_exact_index_expected":float(1/len(Ytest)),
      "random_perm_count":repeats
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--features",type=Path,required=True)
    p.add_argument("--frozen-results",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    obj=json.loads(args.frozen_results.read_text())
    runs=[r for r in obj["runs"] if r["representation"] in CONFS]
    assert len(runs)==40
    record=[];selected_fig=None
    figdir=args.out/"figures"
    for conf in CONFS:
        for fold in range(1,6):
            file=args.features/conf/f"trial_heldout_{fold}_5d.npz"
            with np.load(file) as f:
                Xtrain=np.asarray(f["X_train"])/np.sqrt(DIM)
                Ytrain=np.asarray(f["Y_train"])/np.sqrt(DIM)
                Xtest=np.asarray(f["X_test"])/np.sqrt(DIM)
                Ytest=np.asarray(f["Y_test"])/np.sqrt(DIM)
                tids=np.asarray(f["train_trial_id"])
            radius=train_radius(Ytrain,tids)
            kmx=KMeans(n_clusters=4,n_init=30,random_state=42).fit(Xtrain)
            kmy=KMeans(n_clusters=4,n_init=30,random_state=43).fit(Ytrain)
            bestx,allx=best_pair(Xtrain,kmx.labels_)
            besty,ally=best_pair(Ytrain,kmy.labels_)
            tripx,tripxs=best_triplet(Xtrain,kmx.labels_)
            tripy,tripys=best_triplet(Ytrain,kmy.labels_)
            rand=random_baseline(Ytest,radius["radius"])
            for seed in (42,43):
                for meth in METHODS:
                    row=next(r for r in runs if r["representation"]==conf
                         and r["fold"]==fold and r["seed"]==seed and r["method"]==meth)
                    R=np.asarray(row["R"])
                    assert R.shape==(5,5) and np.linalg.norm(R.T@R-np.eye(5))<1e-7
                    result=evaluate(Xtest,Ytest,R,radius["radius"])
                    identity=evaluate(Xtest,Ytest,np.eye(5),radius["radius"])
                    record.append({
                     "config":conf,"heldout_trial":fold,"seed":seed,"method":meth,
                     "fiveD_train_only_threshold":radius,
                     "training_source_best_2D_pair":[int(bestx[0]+1),int(bestx[1]+1)],
                     "training_source_pair_silhouette":bestx[2],
                     "training_target_best_2D_pair":[int(besty[0]+1),int(besty[1]+1)],
                     "training_target_pair_silhouette":besty[2],
                     "training_source_best_3D_triple":[int(x+1) for x in tripx[:3]],
                     "training_target_best_3D_triple":[int(x+1) for x in tripy[:3]],
                     **result,**{f"identity_{k}":v for k,v in identity.items()
                                 if k not in ("n_test",)},
                     **rand,
                     "training_never_uses_test_synchrony":True,
                     "model_estimator":"frozen global-R weighted-anchor hierarchical OT",
                     "not_original_local_ADMM":True
                    })
                    if conf=="FA5_RAW5" and fold==FIXED_FOLD and seed==FIXED_SEED and meth=="soft05":
                        meta={"source_pair_scan":[[int(x[0]+1),int(x[1]+1),x[2]] for x in allx],
                              "target_pair_scan":[[int(x[0]+1),int(x[1]+1),x[2]] for x in ally],
                              "source_best":[int(bestx[0]+1),int(bestx[1]+1),bestx[2]],
                              "target_best":[int(besty[0]+1),int(besty[1]+1),besty[2]],
                              "source_triple_best":[int(x+1) if i<3 else x for i,x in enumerate(tripx)],
                              "target_triple_best":[int(x+1) if i<3 else x for i,x in enumerate(tripy)],
                              "selection":"K4 silhouette on each training domain independently; not test; visualization only",
                              "selected_example":"FA5 / trial1 heldout / seed42 Soft; fixed, not best-performing run",
                              "all_ten_coordinate_pair_projections_saved":True}
                        draw(Xtest,Ytest,Xtest@R.T,kmx.predict(Xtest),kmy.predict(Ytest),
                             ((bestx[0],bestx[1]),(besty[0],besty[1])),
                             (tripx[:3],tripy[:3]),figdir,meta)
    assert len(record)==40
    (args.out/"nearest_neighbor_evaluation_all.json").write_text(
        json.dumps({"evaluation":"train calibrated five-dimensional nearest-neighbor",
          "methodology":"Nearest target TEST sample to transformed source TEST sample; coverage and true-synchrony physical neighborhood are distinct",
          "radius":"training TARGET inter-trial unrelated-pair distance 5th percentile; never chosen using heldout performance",
          "scores_cannot_be_computed_from_best_2D_projection":True,
          "no_posthoc_parameter_cherry_picking":True,
          "runs":record},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    keys=[
      "config","heldout_trial","seed","method",
      "target_nn_coverage_fraction","nn_retrieval_physical_neighbor_fraction",
      "nn_retrieval_exact_time_index_fraction","nn_retrieval_top5_time_index_fraction",
      "nn_retrieval_physical_neighbor_error_mean","paired_aligned_distance_lt_r_fraction",
      "identity_nn_retrieval_physical_neighbor_fraction",
      "random_target_assignment_hit_mean"
    ]
    with (args.out/"nearest_neighbor_evaluation_all.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader()
        w.writerows({k:r[k] for k in keys} for r in record)
    summary=[]
    for conf in CONFS:
        for meth in METHODS:
            rr=[r for r in record if r["config"]==conf and r["method"]==meth]
            s={"config":conf,"method":meth,"n_runs":len(rr)}
            for k in ("target_nn_coverage_fraction","nn_retrieval_physical_neighbor_fraction",
                      "nn_retrieval_exact_time_index_fraction","nn_retrieval_top5_time_index_fraction",
                      "nn_retrieval_physical_neighbor_error_mean",
                      "paired_aligned_distance_lt_r_fraction",
                      "identity_nn_retrieval_physical_neighbor_fraction","random_target_assignment_hit_mean"):
                s[k]=float(np.mean([r[k] for r in rr]))
            s["mean_threshold"]=float(np.mean([r["fiveD_train_only_threshold"]["radius"] for r in rr]))
            summary.append(s)
    (args.out/"nearest_neighbor_summary.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    fig,axes=plt.subplots(1,2,figsize=(12.4,4.8),constrained_layout=True)
    for i,k in enumerate(("target_nn_coverage_fraction","nn_retrieval_physical_neighbor_fraction")):
        ax=axes[i]
        for conf in CONFS:
            r=[x for x in summary if x["config"]==conf]
            for ix,method in enumerate(METHODS):
                x=0 if conf=="PCA5_RAW5" else 1
                ax.bar(x+(-.16 if method=="hard" else .16),
                    next(y[k] for y in r if y["method"]==method),
                    width=.3,label=method if conf=="PCA5_RAW5" else None)
        ax.set(xticks=[0,1],xticklabels=list(CONFS),
               ylabel="fraction",ylim=(0,1),title=("mere cloud coverage (NOT prediction accuracy)"
                 if i==0 else "correct physical nearest-target neighborhood (heldout)"))
        ax.legend(frameon=False);ax.grid(axis="y",alpha=.12)
    fig.savefig(args.out/"coverage_vs_physical_retrieval.png",dpi=170,bbox_inches="tight")
    plt.close(fig)
    report=[
       "# Hyser 5D: best train-only 2D views and correct nearest-neighbor evaluation",
       "",
       "Nearest target neighbor retrieved among 250 heldout Y windows: j*(i)=argmin_j||R x_i-y_j||. Radius eps is calibrated on TRAIN-only unrelated inter-trial target distances, 5th percentile; same for Hard and Soft. Full-dimensional Euclidean distances, never 2D distances, are used in evaluation.",
       "",
       "**WARNING**: d(Rx,Y_nn)<eps is NOT evidence of correct sample prediction; this is point-cloud coverage only. The real external check is ||Y_nn - Y_true(i)||<eps, using synchrony only after model fitting. Report exact synchronized time index retrieval and random-baseline hit as independent measures.",
       "",
       "**Not original paper's identical metric**: the paper's real neural decoding accuracy uses reach-direction cluster labels, which Hyser Random Task does not contain.",
       "",
       "## Fivefold fixed-seed means",
       "",
       "| Embedding | Method | Train-only eps | NN cloud coverage | Correct physical neighborhood | Exact time index | Random target hit |",
       "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        report.append(f"|{r['config']}|{r['method']}|{r['mean_threshold']:.4f}|{r['target_nn_coverage_fraction']:.3%}|{r['nn_retrieval_physical_neighbor_fraction']:.3%}|{r['nn_retrieval_exact_time_index_fraction']:.3%}|{r['random_target_assignment_hit_mean']:.3%}|")
    report+=["","## Figure details",
        "The best two-coordinate view is chosen among the 10 coordinate pairs using TRAINING Kmeans 4 labels and TRAINING silhouette, for each independent domain. The figure is fixed to FA5, heldout trial1, seed42 Soft (not chosen by performance).",
        "All 10 coordinate pairs and the best 3-coordinate projection are saved; separate source and target figures are never overlaid. The five-dimensional geometry CANNOT be faithfully represented by one two-dimensional figure; the 5x5 matrix is a view of all coordinate pairs.",
        "Using test labels to choose the 'nicest' view would be visualization selection bias.",
        "No additional R/P/Q optimization is performed in this stage, only frozen 120step global-R models.",
        "## Limitation",
        "This does not substitute for a full original-local-R-ADMM HiWA rerun. The physical labels here are actual synchronized force measurements used only in heldout evaluation.",
        ]
    (args.out/"README_ZH.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(json.dumps({"success":True,"runs":len(record),"summary":summary},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
