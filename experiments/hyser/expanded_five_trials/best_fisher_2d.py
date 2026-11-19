#!/usr/bin/env python3
"""2D optimal Fisher subspace of 5D (beyond ten coordinate pairs), EDA only.

It learns two discriminant axes maximizing trace of between-cluster scatter
relative to within-cluster scatter from INDEPENDENT KMeans pseudo-labels on
TRAINING source and target. KMeans labels DO NOT imply physical gesture truth.
Does not fit or evaluate OT; uses frozen FA5 fold1 seed42 Soft only to depict
source after alignment in the TRAIN-fitted TARGET Fisher coordinate plane.

This is not globally maximizing silhouette over all linear 2D subspaces. LDA
maximizes Fisher's generalized eigen criterion, a different precise objective.
"""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.linalg import eigh

COLS=["#2677AA","#DF9953","#4F9C83","#9357A5"]
def fisher2(X,labs,reg=1e-5):
    mu=X.mean(0)
    W=np.zeros((X.shape[1],X.shape[1]))
    B=np.zeros_like(W)
    for k in np.unique(labs):
        group=X[labs==k]
        cm=group.mean(0)
        W+=(group-cm).T@(group-cm)
        v=(cm-mu).reshape(-1,1)
        B+=len(group)*(v@v.T)
    # Regularization chosen independent of test information.
    W=W/len(X);B=B/len(X)
    W=W+reg*np.trace(W)/X.shape[1]*np.eye(X.shape[1])
    val,vec=eigh(B,W)
    ii=np.argsort(val)[::-1][:2]
    v=vec[:,ii]
    proj=(X-mu)@v
    return {"mean":mu,"basis":v,
            "eigenvalues":val[ii],
            "train2d_silhouette":float(silhouette_score(proj,labs)),
            "whole_trainK4_silhouette":float(silhouette_score(X,labs))}
def scatter(ax,data,label,title):
    for k,color in enumerate(COLS):
        m=label==k
        ax.scatter(data[m,0],data[m,1],c=color,s=17,alpha=.72,
                   edgecolors="none",rasterized=True)
    ax.set(xlabel="Fisher discriminant dimension 1",
           ylabel="Fisher discriminant dimension 2",title=title)
    ax.grid(alpha=.15);ax.set_aspect("equal",adjustable="datalim")
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fold",type=Path,required=True)
    p.add_argument("--frozen-results",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with np.load(a.fold) as z:
        Xtr=np.array(z["X_train"])/np.sqrt(5)
        Ytr=np.array(z["Y_train"])/np.sqrt(5)
        Xte=np.array(z["X_test"])/np.sqrt(5)
        Yte=np.array(z["Y_test"])/np.sqrt(5)
    kmx=KMeans(4,n_init=30,random_state=42).fit(Xtr)
    kmy=KMeans(4,n_init=30,random_state=43).fit(Ytr)
    fs=fisher2(Xtr,kmx.labels_)
    ft=fisher2(Ytr,kmy.labels_)
    runs=json.loads(a.frozen_results.read_text())["runs"]
    r=next(v for v in runs if v["representation"]=="FA5_RAW5"
           and v["fold"]==1 and v["seed"]==42 and v["method"]=="soft05")
    R=np.array(r["R"])
    # Independent latent views, not a common 2D alignment metric
    sx=(Xte-fs["mean"])@fs["basis"]
    ty=(Yte-ft["mean"])@ft["basis"]
    aligned=(Xte@R.T-ft["mean"])@ft["basis"]
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    scatter(axes[0],sx,kmx.predict(Xte),"Source train-fitted Fisher 2D")
    scatter(axes[1],ty,kmy.predict(Yte),"Target train-fitted Fisher 2D")
    scatter(axes[2],aligned,kmx.predict(Xte),"Aligned source in TARGET Fisher axes")
    fig.suptitle("5D Hyser FA5 fold1 seed42 Soft: max Fisher 2D views, heldout points")
    fig.savefig(a.out/"fiveD_to_fisher_optimal_2D_independent.png",dpi=185,bbox_inches="tight")
    plt.close(fig)
    doc={"mathematical_criterion":"maximize between-cluster scatter / within-cluster scatter using top 2 generalized eigenvectors; not global silhouette maximization",
         "trained_only_on_four_training_trials":True,
         "used_pseudo_labels_for_visualization_ONLY":True,
         "source":{"train_fisher_eigenvalues":fs["eigenvalues"].tolist(),
                   "train_2D_silhouette":fs["train2d_silhouette"],
                   "train_5D_silhouette":fs["whole_trainK4_silhouette"],
                   "basis":fs["basis"].tolist()},
         "target":{"train_fisher_eigenvalues":ft["eigenvalues"].tolist(),
                   "train_2D_silhouette":ft["train2d_silhouette"],
                   "train_5D_silhouette":ft["whole_trainK4_silhouette"],
                   "basis":ft["basis"].tolist()},
         "validation":"plotting only, not training/OT metric; showing best view can visually exaggerate cluster separability."}
    (a.out/"fisher_2D_basis_training_only.json").write_text(json.dumps(doc,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"source2D_sil":fs["train2d_silhouette"],
                      "target2D_sil":ft["train2d_silhouette"],
                      "original_src5D":fs["whole_trainK4_silhouette"],
                      "original_tgt5D":ft["whole_trainK4_silhouette"]},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
