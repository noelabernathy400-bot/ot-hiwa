"""Evaluate a new Hard HiWA run that uses ONE KMeans label family across domains.

Run after source_only_fixed_group_pipeline.py and the unmodified
code/run_hard_hiwa.py.  All scores are held-out and use original paired windows.
Target is *never* KMeans-clustered.  The 1NN query direction mirrors the
reference GCOT/HiWA evaluation: target queries an aligned source point.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from hiwa_original import _rMSE


def evaluate(x, y, lab, R, Rref):
    prediction=x@R.T
    nearest=cKDTree(prediction).query(y,k=1)[1]
    # Note: target lab is inherited from the *paired* source: this measures
    # shared-label neighborhood preservation, not ground-truth class accuracy.
    accuracy=float(np.mean(lab[nearest]==lab))
    residual=prediction-y
    sse=float(np.sum(residual*residual))
    energy=float(np.sum(y*y))
    sst=float(np.sum((y-y.mean(axis=0))**2))
    return dict(one_nn_shared_group=accuracy,
                MAE=float(np.mean(np.abs(residual))),
                RMSE=float(np.sqrt(np.mean(residual*residual))),
                paired_relative_MSE=float(sse/energy),
                rMSE_rotation_proxy=float(_rMSE(x,R,Rref)),
                R2=float(1-sse/sst))


def plot_six(arr, labels, name, path, limits):
    cmap=plt.get_cmap("tab10")
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    for ax,(i,j) in zip(axes.flat,combinations(range(4),2)):
        for k in range(4):
            mask=labels==k
            ax.scatter(arr[mask,i],arr[mask,j],s=3,alpha=.5,c=[cmap(k)],
                       label=f"Shared group {k}",rasterized=True)
        ax.set(xlabel=f"coordinate {i+1}",ylabel=f"coordinate {j+1}",
               xlim=limits[i],ylim=limits[j])
        ax.grid(alpha=.13)
    handles,leg=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,leg,loc="lower center",ncol=4,bbox_to_anchor=(.5,-.012))
    fig.suptitle(name,fontsize=15)
    fig.savefig(path,dpi=180,bbox_inches="tight")
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--alignment",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(exist_ok=True,parents=True)
    h=json.loads(args.alignment.read_text(encoding="utf-8"))
    if hashlib.sha256(args.input.read_bytes()).hexdigest()!=h["data_sha256"]:
        raise ValueError("HiWA run used different input: refusing mismatch")
    with np.load(args.input,allow_pickle=False) as z:
        x0=np.asarray(z["neural4"],dtype=float)
        y0=np.asarray(z["hand4"],dtype=float)
        train=np.asarray(z["train_mask"],dtype=bool)
        test=np.asarray(z["test_mask"],dtype=bool)
        c=np.asarray(z["shared_labels"],dtype=int)
        source_labels=np.asarray(z["neural_groups"],dtype=int)
        target_labels=np.asarray(z["hand_groups"],dtype=int)
        trial=np.asarray(z["trial_of_window"],dtype=int)
    assert x0.shape==y0.shape==(17500,4) and int(train.sum())==14000
    assert int(test.sum())==3500 and not (train&test).any()
    assert np.array_equal(c,source_labels) and np.array_equal(c,target_labels)
    assert len(np.unique(c[train]))==4
    # Independent normalizations match the exact HiWA runner output, not new
    # per-dimension whitening.
    x=(x0-np.asarray(h["source_mean"]))/float(h["source_radius"])
    y=(y0-np.asarray(h["target_mean"]))/float(h["target_radius"])
    U,_,VT=np.linalg.svd(y[train].T@x[train])
    Rref=U@VT
    xt=x[test]; yt=y[test]; lt=c[test]
    identity=evaluate(xt,yt,lt,np.eye(4),Rref)
    oracle=evaluate(xt,yt,lt,Rref,Rref)
    scores=[]
    for r in h["seed_results"]:
        R=np.asarray(r["R"])
        assert R.shape==(4,4) and np.linalg.norm(R.T@R-np.eye(4),"fro")<1e-7
        metrics=evaluate(xt,yt,lt,R,Rref)
        scores.append(dict(seed=r["seed"],iterations=r["iterations"],
                           converged=bool(r["strict_converged"]),
                           R_det=float(np.linalg.det(R)),metrics=metrics))
    if len(scores)!=5:
        raise ValueError("Five frozen seeds required")

    keys=list(identity.keys())
    averages={key:dict(mean=float(np.mean([s["metrics"][key] for s in scores])),
                  sd_sample=float(np.std([s["metrics"][key] for s in scores],ddof=1)))
              for key in keys}

    # Visualize pre-registered seed 42, not the best test seed.  All stages use
    # identical labels, indices, projection axes and normalized coordinates.
    chosen=[s for s in h["seed_results"] if int(s["seed"])==42]
    if len(chosen)!=1:raise ValueError("Seed 42 missing")
    R42=np.asarray(chosen[0]["R"],dtype=float)
    before=x[test];after=before@R42.T;target=y[test]
    bounds=[]
    for j in range(4):
        three=np.r_[before[:,j],after[:,j],target[:,j]]
        lo,hi=float(np.min(three)),float(np.max(three))
        padding=.045*(hi-lo)
        bounds.append((lo-padding,hi+padding))
    plot_six(before,lt,"Source BEFORE alignment | same source-only labels",
             args.output/"source_before_test_six.png",bounds)
    plot_six(target,lt,"TARGET | inherited source-only labels, NO target KMeans",
             args.output/"target_shared_test_six.png",bounds)
    plot_six(after,lt,"Source AFTER original Hard HiWA, seed 42 | labels unchanged",
             args.output/"source_after_test_six.png",bounds)
    fig,axes=plt.subplots(6,3,figsize=(13.8,21),constrained_layout=True)
    cmap=plt.get_cmap("tab10")
    for row,(i,j) in enumerate(combinations(range(4),2)):
        for col,(arr,title) in enumerate(((before,"Source before"),(after,"Source after"),
                                         (target,"Target (same labels)"))):
            ax=axes[row,col]
            for k in range(4):
                mask=lt==k
                ax.scatter(arr[mask,i],arr[mask,j],s=3,alpha=.48,
                           c=[cmap(k)],rasterized=True)
            ax.set(title=f"{title}: ({i+1},{j+1})",
                   xlabel=f"coordinate {i+1}",ylabel=f"coordinate {j+1}",
                   xlim=bounds[i],ylim=bounds[j])
            ax.grid(alpha=.1)
    fig.suptitle("ONE source KMeans / ONE label family / SAME test samples / seed 42",fontsize=15)
    fig.savefig(args.output/"before_after_target_six_pairs.png",dpi=185,bbox_inches="tight")
    plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(14.4,4.75),constrained_layout=True)
    for ax,(arr,title) in zip(axes,((before,"Source before"),
                                    (after,"Source after, seed 42"),
                                    (target,"Target; same labels"))):
        for k in range(4):
            t=lt==k
            ax.scatter(arr[t,0],arr[t,1],s=5,alpha=.5,
                       c=[cmap(k)],rasterized=True)
        ax.set(xlim=bounds[0],ylim=bounds[1],title=title,
               xlabel="coordinate 1",ylabel="coordinate 2")
        ax.grid(alpha=.12)
    fig.suptitle("Original source group labels retained across all three clouds")
    fig.savefig(args.output/"before_after_target_xy.png",dpi=220,bbox_inches="tight")
    plt.close(fig)

    fig,ax=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    keys=("one_nn_shared_group","MAE","R2")
    for a,k in zip(ax,keys):
        v=np.array([s["metrics"][k] for s in scores])
        a.axhline(identity[k],linestyle="--",color="grey",label="Identity")
        a.scatter([s["seed"] for s in scores],v,label="HiWA seeds")
        a.axhline(v.mean(),linestyle=":",color="black",label="HiWA mean")
        a.set(xlabel="seed",ylabel=k,title=k)
        a.grid(alpha=.14)
    ax[0].legend(fontsize=8)
    fig.savefig(args.output/"frozen_label_metrics.png",dpi=190)
    plt.close(fig)

    report=dict(
        experiment="2026-11-17 source-only KMeans, paired label propagation, original Hard HiWA",
        dataset="DANDI:000138/0.220113.0407",
        protocol="KMeans fitted on source training features ONCE; all target labels inherited via known paired time-window ID; no second clustering, no post-alignment relabeling",
        paired_labels_use_supervision=True,
        original_hiwa_implementation_unchanged=True,
        original_solver_sha256=h["original_solver_sha256"],
        source_and_target_labels_identical=True,
        training_n=int(train.sum()),test_n=int(test.sum()),
        training_trials=int(len(np.unique(trial[train]))),
        test_trials=int(len(np.unique(trial[test]))),
        group_size_train=np.bincount(c[train],minlength=4).tolist(),
        group_size_test=np.bincount(c[test],minlength=4).tolist(),
        before_identity=identity,
        supervised_train_pairing_procrustes_diagnostic=oracle,
        seed_results=scores,statistics=averages,
        graph_seed=42,
        one_nn_label_policy="target queries closest aligned source test point; target labels inherited from its paired source; no permutation",
        MAE_policy="mean abs difference over paired four-dimensional heldout coordinates",
        paired_relative_MSE_policy="SSE / sum_target_squared; no square root (different from unknown-groundtruth rotation-rMSE)",
        rMSE_rotation_proxy_policy="original PyHiWA _rMSE using TRAIN-only Procrustes as surrogate R_gt; no actual R_gt exists",
        R2_policy="1 - SSE/SST on held-out paired four-dimensional windows",
        caveat="Target labels transferred via pairing; benchmark has correspondence supervision. DO NOT compare as unsupervised group-discovery HiWA.")
    (args.output/"frozen_label_metrics.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("SOURCE-ONLY FINAL",json.dumps(dict(before=identity,
        mean=averages,each=[{"seed":s["seed"],**s["metrics"]} for s in scores]),
        indent=2),flush=True)


if __name__=="__main__":
    main()
