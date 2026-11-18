"""Plot real heldout source-vs-target cloud overlays after train-only transforms.

This is POSTFIT VISUALIZATION ONLY. Reuses the already stored 5-seed strict
unpaired Hard/Soft R and train-only target transform definitions. No solver
is called, no test score used to choose a transform, no paired labels used
to determine any source or target group.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from itertools import combinations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_target_only_geometry_ablation import target_transform
plt.rcParams.update({"font.family":"DejaVu Sans","axes.titlesize":9,"axes.labelsize":8})

def overlay(x,y,Rs,modes,method,out):
    pairs=list(combinations(range(4),2))
    fig,axs=plt.subplots(6,len(modes),figsize=(4.05*len(modes),20.0),constrained_layout=True)
    plotted=[(x@Rs[m].T,y[m]) for m in modes]
    for row,(i,j) in enumerate(pairs):
        x0=min(float(min(v[:,i].min(),w[:,i].min())) for v,w in plotted)
        x1=max(float(max(v[:,i].max(),w[:,i].max())) for v,w in plotted)
        y0=min(float(min(v[:,j].min(),w[:,j].min())) for v,w in plotted)
        y1=max(float(max(v[:,j].max(),w[:,j].max())) for v,w in plotted)
        px=max((x1-x0)*.055,.08);py=max((y1-y0)*.055,.08)
        for col,m in enumerate(modes):
            a=axs[row,col]
            sx,ty=plotted[col]
            a.scatter(sx[:,i],sx[:,j],s=2,alpha=.28,linewidths=0,
                      c="#2677b5",label="Source @ learned R",rasterized=True)
            a.scatter(ty[:,i],ty[:,j],s=2,alpha=.23,linewidths=0,
                      c="#e18436",label="Target after train-only transform",rasterized=True)
            a.set(xlim=(x0-px,x1+px),ylim=(y0-py,y1+py),
                  xlabel=f"coordinate {i+1}",ylabel=f"coordinate {j+1}",
                  title=f"{m}, pair {i+1}-{j+1}")
            a.grid(alpha=.09)
    ax=axs[0,0]
    handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles[:2],labels[:2],loc="lower center",ncol=2,
               bbox_to_anchor=(.5,-.015),frameon=False)
    fig.suptitle(f"HELDOUT TEST 3500 points each: SOURCE (blue) vs transformed TARGET (orange)\n"
                 f"{method.upper()}: fixed unpaired learned seed42 R for each target representation. "
                 "Six 2D coordinate projections.",fontsize=13,y=1.014)
    fig.savefig(out/f"HELDOUT_SOURCE_TARGET_OVERLAY_{method.upper()}_six_projections.png",
                dpi=175,bbox_inches="tight")
    plt.close(fig)

def geometry(results,out):
    modes=list(results["variants"])
    diag=[results["variants"][m]["geometry_shape_diagnostic_train_only"] for m in modes]
    src=np.asarray(diag[0]["source_cov_eigs"])
    fig,axs=plt.subplots(1,2,figsize=(13,5.2),constrained_layout=True)
    axs[0].plot(range(1,5),src,marker="o",label="Source unchanged",color="black",lw=2)
    for m,d in zip(modes,diag):
        axs[0].plot(range(1,5),d["target_cov_eigs"],marker="o",label=f"Target {m}")
    axs[0].set(xticks=range(1,5),xlabel="Eigenvalue rank",ylabel="Covariance eigenvalue / trace",
               title="Rotation-invariant covariance spectra on 14,000 TRAIN points")
    axs[0].legend(fontsize=8)
    axs[0].grid(alpha=.15)
    q=diag[0]["quantile_levels"]
    axs[1].plot(q,diag[0]["source_dist_quantiles"],"k-o",lw=2,
               label="Source within-domain pair distances")
    for m,d in zip(modes,diag):
        axs[1].plot(q,d["target_dist_quantiles"],marker="o",
                    label=f"Target {m}")
    axs[1].set(xlabel="Empirical within-domain pair-distance quantile",
               ylabel="4D Euclidean distance",title="Train-only within-domain distance distributions")
    axs[1].legend(fontsize=8)
    axs[1].grid(alpha=.15)
    fig.suptitle("Geometric similarities can occur without recovering the TRUE cross-modal paired relation")
    fig.savefig(out/"TRAIN_ONLY_COVARIANCE_SPECTRA_AND_DISTANCE_QUANTILES.png",
                dpi=200,bbox_inches="tight")
    plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--clusters",required=True,type=Path)
    p.add_argument("--baseline",required=True,type=Path)
    p.add_argument("--results",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    with np.load(args.clusters,allow_pickle=False) as z:
        assert "shared_labels" not in z.files
        X=np.asarray(z["neural4"],float)
        Y=np.asarray(z["hand4"],float)
        train=np.asarray(z["train_mask"],bool)
        test=np.asarray(z["test_mask"],bool)
    baseline=json.loads(args.baseline.read_text())
    result=json.loads(args.results.read_text())
    assert result["same_original_source"]
    assert result["target_groups_refit_from_TARGET_FEATURES_only"]
    assert result["true_synchronous_pairing_is_postfit_evaluation_only"]
    x=(X-np.asarray(baseline["source_mean"]))/baseline["source_radius"]
    y=(Y-np.asarray(baseline["target_mean"]))/baseline["target_radius"]
    modes=list(result["variants"])
    ytest={}
    for m in modes:
        z,inv,settings=target_transform(y,train,m)
        assert np.max(abs(inv(z[train])-y[train]))<1e-8
        ytest[m]=z[test]
    for method in ("hard","soft"):
        R={m:np.asarray(next(t["R"] for t in result["variants"][m]["runs"][method]
                  if t["seed"]==42)) for m in modes}
        overlay(x[test],ytest,R,modes,method,args.output)
    geometry(result,args.output)
    for name in ["HELDOUT_SOURCE_TARGET_OVERLAY_HARD_six_projections.png",
                 "HELDOUT_SOURCE_TARGET_OVERLAY_SOFT_six_projections.png",
                 "TRAIN_ONLY_COVARIANCE_SPECTRA_AND_DISTANCE_QUANTILES.png"]:
        assert (args.output/name).stat().st_size>40000
    print("POSTFIT SOURCE VS TARGET OVERLAYS AND GEOMETRY PLOTS COMPLETE",flush=True)
if __name__=="__main__":
    main()
