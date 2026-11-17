"""Single-source KMeans protocol on frozen DANDI:000138 4D observations.

One (and only one) KMeans is fitted on training neural features.
Its labels are copied to hand-kinematic samples by observed time/trial pairing;
test labels are predicted using the frozen source KMeans.  These labels stay
unchanged before and after Hard HiWA; there is NO target-domain KMeans.
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def plot_six(arr, labels, title, path, limits=None, centers=None):
    pairs=list(combinations(range(4), 2))
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    cmap=plt.get_cmap("tab10")
    for ax,(i,j) in zip(axes.flat,pairs):
        for k in range(4):
            t=labels==k
            ax.scatter(arr[t,i],arr[t,j],s=3,alpha=.48,color=cmap(k),
                       rasterized=True,label=f"Shared group {k}")
        if centers is not None:
            ax.scatter(centers[:,i],centers[:,j],marker="x",c="black",s=70,linewidths=1)
        if limits:
            ax.set_xlim(*limits[i])
            ax.set_ylim(*limits[j])
        ax.set(xlabel=f"coordinate {i+1}",ylabel=f"coordinate {j+1}")
        ax.grid(alpha=.15)
    handles,legends=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,legends,ncol=4,loc="lower center",bbox_to_anchor=(.5,-.015))
    fig.suptitle(title,fontsize=15)
    fig.savefig(path,dpi=180,bbox_inches="tight")
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--k",type=int,default=4)
    args=p.parse_args()
    if args.k!=4:
        raise ValueError("The frozen four-group benchmark requires K=4")
    args.output.mkdir(parents=True,exist_ok=True)
    with np.load(args.input,allow_pickle=False) as z:
        x=np.array(z["neural4"],dtype=np.float64)
        y=np.array(z["hand4"],dtype=np.float64)
        tr=np.array(z["train_mask"],dtype=bool)
        te=np.array(z["test_mask"],dtype=bool)
        trial=np.array(z["trial_of_window"],dtype=np.int64)
        original_ids=np.array(z["original_trial_ids"],dtype=np.int64)
    if x.shape!=y.shape or x.ndim!=2 or x.shape[1]!=4:
        raise ValueError("Need identically indexed 4D time-window arrays")
    if np.any(tr&te) or not np.all(tr|te) or len(x)!=17500:
        raise ValueError("Unexpected 400/100 trial split or time windows")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Non-finite four-dimensional samples")
    # Time pairing is inherited from the original official NWB window index.
    # KMeans only uses source *training* points.
    model=KMeans(n_clusters=args.k,n_init=20,random_state=args.seed)
    model.fit(x[tr])
    labels=np.asarray(model.predict(x),dtype=np.int32)
    assert len(labels)==len(y) and set(labels[tr])==set(range(4))

    # Strict label preservation: target obtains source labels by paired row
    # identity, NOT by fitting or applying a KMeans on target measurements.
    target_labels=labels.copy()
    assert np.array_equal(target_labels,labels)
    np.savez_compressed(args.output/"fixed_groups_4d.npz",
        neural4=x.astype(np.float32),hand4=y.astype(np.float32),
        train_mask=tr,test_mask=te,
        trial_of_window=trial,original_trial_ids=original_ids,
        neural_groups=labels,hand_groups=target_labels,
        shared_labels=labels,source_centers=model.cluster_centers_)

    # Matching physical geometry for plots and HiWA: center and use one RMS
    # radius per domain, both estimated only from *training* observations.
    mx=x[tr].mean(0);my=y[tr].mean(0)
    sx=float(np.sqrt(np.mean(np.sum((x[tr]-mx)**2,axis=1))))
    sy=float(np.sqrt(np.mean(np.sum((y[tr]-my)**2,axis=1))))
    xn=(x-mx)/sx;yn=(y-my)/sy
    limits=[(min(xn[tr,j].min(),yn[tr,j].min()),max(xn[tr,j].max(),yn[tr,j].max()))
            for j in range(4)]
    # Add padding while guaranteeing identical axes for corresponding panels.
    limits=[(float(lo-.05*(hi-lo)),float(hi+.05*(hi-lo))) for lo,hi in limits]

    plot_six(xn[tr],labels[tr],"SOURCE BEFORE | one KMeans on neural training only",
             args.output/"source_before_6projections.png",limits=limits,
             centers=(model.cluster_centers_-mx)/sx)
    plot_six(yn[tr],target_labels[tr],"TARGET BEFORE | inherited source-group labels, no clustering",
             args.output/"target_same_groups_6projections.png",limits=limits)
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for ax,arr,title in ((axes[0],xn[tr],"Source before"),
                         (axes[1],yn[tr],"Target (paired source labels)")):
        for k in range(4):
            sl=labels[tr]==k
            ax.scatter(arr[sl,0],arr[sl,1],c=[plt.get_cmap("tab10")(k)],
                       s=4,alpha=.45,rasterized=True)
        ax.set(title=title,xlabel="coordinate 1",ylabel="coordinate 2",
               xlim=limits[0],ylim=limits[1])
        ax.grid(alpha=.12)
    fig.suptitle("ONE fixed four-group assignment | training time windows")
    fig.savefig(args.output/"before_source_target_side_by_side.png",dpi=200)
    plt.close(fig)

    # The only label family is shared throughout, for all test windows too.
    sizes=np.bincount(labels[tr],minlength=4).tolist()
    r=np.random.default_rng(args.seed)
    sampled=r.choice(np.flatnonzero(tr),size=min(int(tr.sum()),3000),replace=False)
    sil=float(silhouette_score(x[sampled],labels[sampled]))
    report=dict(dataset="DANDI:000138",version="0.220113.0407",
        frozen_input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        protocol="KMeans(source TRAIN only); propagate labels by true neural/hand time-window index; NEVER cluster target; NEVER recluster after alignment",
        shared_label_uses_known_training_pairing=True,
        source_training_only_kmeans=True,target_clustering_performed=False,
        labels_reassigned_after_alignment=False,
        samples=int(len(x)),training_windows=int(tr.sum()),test_windows=int(te.sum()),
        K=args.k,seed=args.seed,source_train_silhouette_sampled=sil,
        source_group_sizes=sizes,target_group_sizes=sizes,
        source_norm_mean=mx.tolist(),target_norm_mean=my.tolist(),
        source_norm_radius=sx,target_norm_radius=sy,
        label_sha256=hashlib.sha256(labels.tobytes()).hexdigest())
    (args.output/"group_protocol.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print("FROZEN SOURCE GROUPS",json.dumps({k:report[k] for k in (
        "source_group_sizes","target_group_sizes","source_train_silhouette_sampled",
        "training_windows","test_windows")},indent=2),flush=True)


if __name__=="__main__":
    main()
