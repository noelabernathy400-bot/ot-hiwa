#!/usr/bin/env python3
"""Only Fisher 2D visuals; 1000 TRAIN structure + 250 TEST Hard/Soft alignment.

A Fisher subspace is train-fitted in each domain INDEPENDENTLY. It is not an
unbiased representation of cluster separation, nor a shared source-target
basis. For alignment, we depict (R@source) in the TARGET-trained Fisher plane.

Full original dimension d=3 or d=5 remains unchanged for all quantitative
metrics; no arbitrary coordinate-pair plots.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from best_fisher_2d import fisher2

COLORS=("#2677AA","#DA9051","#4A9376","#A06CB3")
def scatter(ax,Z,labels,small=False):
    for k,color in enumerate(COLORS):
        m=labels==k
        ax.scatter(Z[m,0],Z[m,1],color=color,s=7 if small else 12,
                   alpha=.55 if small else .68,
                   label=f"Cluster {k+1}",edgecolors="none",rasterized=True)
    ax.set(xlabel="Fisher component 1",ylabel="Fisher component 2")
    ax.grid(alpha=.12)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--features",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--configs",nargs="+",required=True)
    p.add_argument("--runs-dir",type=Path,default=None)
    p.add_argument("--fold",type=int,default=1)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    runs=[]
    if args.runs_dir:
        for pth in sorted(args.runs_dir.rglob("local_ADMM_results.json")):
            runs+=json.loads(pth.read_text())["runs"]
    output=[]
    for rep in args.configs:
        d=5 if "5" in rep else 3
        file=args.features/rep/(f"trial_heldout_{args.fold}"+("_5d" if d==5 else "")+".npz")
        with np.load(file) as f:
            sx=np.asarray(f["X_train"],float)/np.sqrt(d)
            sy=np.asarray(f["Y_train"],float)/np.sqrt(d)
            tx=np.asarray(f["X_test"],float)/np.sqrt(d)
            ty=np.asarray(f["Y_test"],float)/np.sqrt(d)
        assert sx.shape==sy.shape==(1000,d), (rep,sx.shape,sy.shape)
        assert tx.shape==ty.shape==(250,d)
        kx=KMeans(4,n_init=30,random_state=42).fit(sx)
        ky=KMeans(4,n_init=30,random_state=43).fit(sy)
        fx=fisher2(sx,kx.labels_)
        fy=fisher2(sy,ky.labels_)
        src_train=(sx-fx["mean"])@fx["basis"]
        tgt_train=(sy-fy["mean"])@fy["basis"]
        fig,ax=plt.subplots(1,2,figsize=(12.5,5.3),constrained_layout=True)
        scatter(ax[0],src_train,kx.labels_,small=True)
        scatter(ax[1],tgt_train,ky.labels_,small=True)
        ax[0].set_title(f"EMG source: ALL 1000 TRAIN windows\n(d={d}, Fisher-trained-on-1000)")
        ax[1].set_title(f"Force target: ALL 1000 TRAIN windows\n(d={d}, Fisher-trained-on-1000)")
        ax[1].legend(frameon=False,fontsize=8,loc="best")
        fig.suptitle(f"{rep} · 4 training trials × 250 windows · independent four-cluster Fisher subspaces")
        fig.savefig(args.out/f"{rep}_TRAIN_1000_Fisher2D.png",dpi=185,bbox_inches="tight")
        plt.close(fig)
        row={"feature":rep,"d":d,"fold":args.fold,"train_n_source":1000,"train_n_target":1000,
             "test_n_source":250,"test_n_target":250,
             "source_train_Fisher_silhouette_2D":fx["train2d_silhouette"],
             "source_train_silhouette_full_d":fx["whole_trainK4_silhouette"],
             "target_train_Fisher_silhouette_2D":fy["train2d_silhouette"],
             "target_train_silhouette_full_d":fy["whole_trainK4_silhouette"],
             "source_train_basis":fx["basis"].tolist(),
             "target_train_basis":fy["basis"].tolist(),
             "training_cluster_labels_NOT_physical_finger_classes":True,
             "no_test_data_used_for_Fisher_selection":True}
        if runs:
            rows=[r for r in runs if r["config"]==rep and r["heldout_trial"]==args.fold
                  and r["seed"]==args.seed]
            assert {r["method"] for r in rows}=={"hard","soft05"},(rep,rows)
            h=next(r for r in rows if r["method"]=="hard")
            s=next(r for r in rows if r["method"]=="soft05")
            assert all(x["Q_constraints_passed"] and x["numeric_convergence_ADMM_primal_and_rotation"] for x in (h,s))
            axsrc=(tx-fx["mean"])@fx["basis"]
            axtgt=(ty-fy["mean"])@fy["basis"]
            fig,ax=plt.subplots(2,2,figsize=(12.5,11.2),constrained_layout=True)
            scatter(ax[0,0],axsrc,kx.predict(tx))
            ax[0,0].set_title("250 TEST source windows in SOURCE Fisher basis")
            scatter(ax[0,1],axtgt,ky.predict(ty))
            ax[0,1].set_title("250 TEST target windows in TARGET Fisher basis")
            for axis,method,title in ((ax[1,0],h,"Hard"),(ax[1,1],s,"Soft")):
                R=np.asarray(method["R"])
                rotated=(tx@R.T-fy["mean"])@fy["basis"]
                axis.scatter(axtgt[:,0],axtgt[:,1],s=12,c="black",alpha=.23,
                             label="250 target TEST points",rasterized=True)
                axis.scatter(rotated[:,0],rotated[:,1],s=11,c="#C75B65" if title=="Hard" else "#3475A9",
                             alpha=.42,label=f"250 transformed source TEST ({title})",rasterized=True)
                axis.set(xlabel="Target Fisher component 1",ylabel="Target Fisher component 2")
                axis.set_title(title+" · target-trained Fisher plane, 250+250 test points")
                axis.legend(fontsize=8,frameon=False)
                axis.grid(alpha=.12)
            fig.suptitle(f"{rep} · identical TRAIN-Fisher bases · full {d}D original local P/Q/Rkl ADMM · holdout fold {args.fold}")
            fig.savefig(args.out/f"{rep}_TEST_250_Fisher2D_Hard_Soft.png",dpi=180,bbox_inches="tight")
            plt.close(fig)
            row["model_2D_test_plot_from_frozen_full_d_R"]=True
        else:
            row["model_2D_test_plot_from_frozen_full_d_R"]=False
        output.append(row)
    (args.out/"FISHER_ONLY_VISUAL_PROVENANCE.json").write_text(
        json.dumps({"all_configurations":output,
         "note":"Two-dimensional Fisher charts are visual-only and chosen using 1000 training points, never evaluate accuracy in 2D.",
         "training_points_rendered_each_domain":1000,
         "test_points_rendered_each_domain":250,
         "test_alignment_status":"frozen existing full local-ADMM R used iff runs-dir provided"},ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"success":True,"representations":[r["feature"] for r in output],
                      "n_train":1000,"n_test":250,"alignment_plots":bool(runs)},ensure_ascii=False))
if __name__=="__main__":main()
