#!/usr/bin/env python3
"""Hyser WFDB (real EMG/force) -> 0.1s features -> 3D clusters and plots."""
import argparse, json
from pathlib import Path
import numpy as np
import wfdb
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
FINGERS=["thumb","index","middle","ring","little"]

def load_trial(folder,trial,step):
    emg=wfdb.rdrecord(str(folder / ("random_preprocess_sample%d"%trial)))
    force=wfdb.rdrecord(str(folder / ("random_force_sample%d"%trial)))
    a,b=emg.p_signal,force.p_signal
    assert a.shape[1]==256 and b.shape[1]==5
    if abs(len(a)/emg.fs-len(b)/force.fs)>.02: raise ValueError("Duration mismatch")
    fi=[force.sig_name.index(f) for f in FINGERS]
    n=int(min(len(a)/emg.fs,len(b)/force.fs)/step)
    X,Y,ids=[],[],[]
    for j in range(n):
        a0,a1=round(j*step*emg.fs),round((j+1)*step*emg.fs)
        b0,b1=round(j*step*force.fs),round((j+1)*step*force.fs)
        e=a[a0:a1]; f=b[b0:b1]
        if not len(e) or not len(f) or not np.isfinite(e).all() or not np.isfinite(f).all():continue
        X.append(np.sqrt(np.mean((e-e.mean(axis=0))**2,axis=0)))
        Y.append(f[:,fi].mean(axis=0))
        ids.append("t%d_w%d"%(trial,j))
    return np.array(X),np.array(Y),np.array(ids)

def plot(Z,labels,centers,out,prefix):
    fig=plt.figure(figsize=(7,5.5));ax=fig.add_subplot(111,projection="3d")
    for c in range(len(centers)):
        p=Z[labels==c]
        ax.scatter(p[:,0],p[:,1],p[:,2],s=14,alpha=.65,label="Cluster %d"%c)
        ax.scatter(*centers[c],marker="X",s=140,edgecolors="black")
    ax.legend();ax.set_title(prefix+" (3D)");fig.tight_layout()
    fig.savefig(out/(prefix+"_3d.png"),dpi=160);plt.close(fig)
    pca=PCA(n_components=2).fit(Z);v=pca.transform(Z);cent=pca.transform(centers)
    fig,ax=plt.subplots(figsize=(7,5))
    for c in range(len(centers)):
        p=v[labels==c];ax.scatter(p[:,0],p[:,1],s=15,alpha=.65,label="Cluster %d"%c)
        ax.scatter(*cent[c],marker="X",s=140,edgecolors="black")
    ax.legend();ax.set_title(prefix+" (2D PCA)");fig.tight_layout()
    fig.savefig(out/(prefix+"_2d.png"),dpi=160);plt.close(fig)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--trials",type=int,nargs="+",default=[1])
    parser.add_argument("--k",type=int,default=4)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--window",type=float,default=.1)
    args=parser.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    blocks=[load_trial(args.data_dir,i,args.window) for i in args.trials]
    xr=np.vstack([b[0] for b in blocks]);yr=np.vstack([b[1] for b in blocks])
    ids=np.concatenate([b[2] for b in blocks]);trial_index=np.concatenate([np.full(len(b[0]),i) for b,i in zip(blocks,args.trials)])
    train=(trial_index!=args.trials[-1]) if len(args.trials)>1 else (np.arange(len(ids))<int(.8*len(ids)))
    test=~train
    xlog=np.log1p(xr*1e5)
    scaler=StandardScaler().fit(xlog[train]);pca=PCA(n_components=3,random_state=args.seed).fit(scaler.transform(xlog[train]))
    X=pca.transform(scaler.transform(xlog))
    selected=np.argsort(-np.var(yr[train],axis=0))[:3];Y=yr[:,selected]
    def normalize(z):
        mean=z[train].mean(axis=0);scale=np.sqrt(np.mean(np.sum((z[train]-mean)**2,axis=1)))
        return (z-mean)/scale
    X,Y=normalize(X),normalize(Y)
    rng=np.random.default_rng(args.seed)
    ptr=rng.permutation(train.sum());pte=rng.permutation(test.sum())
    np.savez_compressed(args.out_dir/"domains.npz",Xtrain=X[train],Xtest=X[test],Ytrain=Y[train][ptr],Ytest=Y[test][pte],
        tr_pair=ids[train],tr_y_pair=ids[train][ptr],te_pair=ids[test],te_y_pair=ids[test][pte])
    mx=KMeans(n_clusters=args.k,n_init=30,random_state=args.seed).fit(X[train])
    my=KMeans(n_clusters=args.k,n_init=30,random_state=args.seed).fit(Y[train][ptr])
    np.savez_compressed(args.out_dir/"clusters.npz",cx=mx.labels_,cy=my.labels_,cx_test=mx.predict(X[test]),
        cy_test=my.predict(Y[test][pte]),centers_x=mx.cluster_centers_,centers_y=my.cluster_centers_)
    plot(X[train],mx.labels_,mx.cluster_centers_,args.out_dir,"emg")
    plot(Y[train][ptr],my.labels_,my.cluster_centers_,args.out_dir,"force")
    report={"n_total":len(ids),"train":int(train.sum()),"holdout":int(test.sum()),
        "independent_trial_holdout":len(args.trials)>1,
        "selected_fingers":[FINGERS[i] for i in selected],
        "pca_ratio":pca.explained_variance_ratio_.tolist(),
        "silhouette":[float(silhouette_score(X[train],mx.labels_)),float(silhouette_score(Y[train][ptr],my.labels_))],
        "cluster_sizes":[np.bincount(mx.labels_,minlength=args.k).tolist(),np.bincount(my.labels_,minlength=args.k).tolist()]}
    (args.out_dir/"manifest.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
