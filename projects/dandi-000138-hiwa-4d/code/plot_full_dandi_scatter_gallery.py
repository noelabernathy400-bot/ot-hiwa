"""Postfit FULL DANDI000138 4D scatter gallery, 17500/14000/3500/160 points.

READ-ONLY: use same frozen original independently clustered neural/hand 4D
input and archived unpaired Hard/Soft GCOT seed42 R. Never refit anything.
Shared source/target physical-window colors are applied ONLY in plots after fit.
"""
from __future__ import annotations
import argparse, hashlib, json
from itertools import combinations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import softmax

PALETTE=["#387abb","#e59140","#2c9a7c","#9b7ac4"]
PAIRS=list(combinations(range(4),2))
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,
                     "axes.labelsize":8,"axes.titlesize":9})

def scatter(ax,coords,group,i,j,n):
    size=0.75 if n>10000 else (2.0 if n>1000 else 14)
    alpha=.20 if n>10000 else (.38 if n>1000 else .78)
    for k in range(4):
        sub=group==k
        ax.scatter(coords[sub,i],coords[sub,j],s=size,alpha=alpha,
                   color=PALETTE[k],rasterized=True,linewidths=0)
    ax.grid(alpha=.08)

def four_panels(X,Y,lx,RH,RS,section,out):
    clouds=[("Source X BEFORE",X),("Source X @ R_H.T",X@RH.T),
            ("Source X @ R_S.T",X@RS.T),("Target Y",Y)]
    n=len(X)
    fig,axes=plt.subplots(6,4,figsize=(17.4,22.0),constrained_layout=True)
    for row,(i,j) in enumerate(PAIRS):
        xmin=min(v[:,i].min() for _,v in clouds);xmax=max(v[:,i].max() for _,v in clouds)
        ymin=min(v[:,j].min() for _,v in clouds);ymax=max(v[:,j].max() for _,v in clouds)
        px=max((xmax-xmin)*.04,.05);py=max((ymax-ymin)*.04,.05)
        for col,(title,coords) in enumerate(clouds):
            ax=axes[row,col]
            scatter(ax,coords,lx,i,j,n)
            ax.set(xlim=(xmin-px,xmax+px),ylim=(ymin-py,ymax+py),
                   title=f"{title} | {i+1},{j+1}",
                   xlabel=f"coordinate {i+1}",ylabel=f"coordinate {j+1}")
    fig.suptitle(f"{section}: ALL {n:,} points per domain | 6 pairs of original 4D coordinates\n"
        "POSTFIT physically synchronized coloring ONLY: target colors from its paired source hard group; NEVER used in fit",
        fontsize=13,y=1.015)
    fig.savefig(out/f"{section}_all_4D_projections_common_postfit_colors.png",dpi=165,bbox_inches="tight")
    plt.close(fig)

def independent_panels(X,Y,lx,ly,section,out,soft=False,centers=None):
    if soft:
        cx,cy=centers
        a=softmax(X@cx.T,axis=1);b=softmax(Y@cy.T,axis=1)
        lx=a.argmax(1);ly=b.argmax(1)
    fig,axes=plt.subplots(6,2,figsize=(11.8,20.6),constrained_layout=True)
    for row,(i,j) in enumerate(PAIRS):
        for col,(coords,labels,name) in enumerate([
               (X,lx,"NEURAL X"),(Y,ly,"HAND Y")]):
            ax=axes[row,col]
            scatter(ax,coords,labels,i,j,len(coords))
            ax.set(xlabel=f"coordinate {i+1}",ylabel=f"coordinate {j+1}",
                   title=f"{name} own {'SOFT argmax' if soft else 'HARD KMeans'} group | {i+1},{j+1}")
    fig.suptitle(f"{section} : {len(X):,} points per domain, six 2D projections\n"
       "Source and Target groups learned INDEPENDENTLY; same color is NOT known correspondence",
       fontsize=13,y=1.018)
    label="independent_SOFT_prototype" if soft else "independent_HARD_Kmeans"
    fig.savefig(out/f"{section}_all_4D_projections_{label}.png",dpi=165,bbox_inches="tight")
    plt.close(fig)
    return np.bincount(lx,minlength=4).tolist(),np.bincount(ly,minlength=4).tolist()

def count_comparison(X,Y,train,test,LX,LY,ix,iy,out):
    fig,axes=plt.subplots(2,3,figsize=(16.2,9.0),constrained_layout=True)
    for row,(coords,group,idx,name) in enumerate([
         (X,LX,ix,"Neural source X"),(Y,LY,iy,"Hand target Y")]):
        lo=coords[:,0].min();hi=coords[:,0].max()
        low=coords[:,1].min();high=coords[:,1].max()
        dx=(hi-lo)*.05;dy=(high-low)*.05
        selections=[np.flatnonzero(train),np.flatnonzero(test),idx]
        for col,indices in enumerate(selections):
            ax=axes[row,col]
            scatter(ax,coords[indices],group[indices],0,1,len(indices))
            ax.set(xlim=(lo-dx,hi+dx),ylim=(low-dy,high+dy),
               xlabel="dimension 1",ylabel="dimension 2",
               title=f"{name} | {len(indices):,} points")
    fig.suptitle("The P/Q point-cloud panel contains only 160 source and 160 target TRAIN windows,\n"
       "whereas this dataset has 14,000 train and 3,500 heldout windows per domain.",
       fontsize=13,y=1.025)
    fig.savefig(out/"POINT_COUNTS_14000_TRAIN_3500_TEST_160_OT.png",
                dpi=195,bbox_inches="tight")
    plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--clusters",type=Path,required=True)
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--pq-arrays",type=Path,required=True)
    p.add_argument("--clustering-report",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.clusters.name!="clustered_4d.npz":
        raise ValueError("Copied-label dataset forbidden")
    j=json.loads(args.archive.read_text(encoding="utf8"))
    d=json.loads(args.clustering_report.read_text(encoding="utf8"))
    assert j["input_sha256"]==hashlib.sha256(args.clusters.read_bytes()).hexdigest()
    assert j["no_target_membership_copy"] and j["no_synchronized_pair_ids_used_in_fitting"]
    with np.load(args.clusters,allow_pickle=False) as z:
        assert "shared_labels" not in z.files
        X0=np.asarray(z["neural4"],float);Y0=np.asarray(z["hand4"],float)
        train=np.asarray(z["train_mask"],bool);test=np.asarray(z["test_mask"],bool)
        lx=np.asarray(z["neural_groups"],int);ly=np.asarray(z["hand_groups"],int)
        trials=np.asarray(z["trial_of_window"],int)
    assert X0.shape==Y0.shape==(17500,4)
    assert train.sum()==14000 and test.sum()==3500
    assert len(np.unique(trials[train]))==400 and len(np.unique(trials[test]))==100
    assert not np.array_equal(lx,ly)
    X=(X0-np.asarray(j["source_mean"]))/j["source_radius"]
    Y=(Y0-np.asarray(j["target_mean"]))/j["target_radius"]
    RH=np.asarray(next(r for r in j["runs"]["hard"] if r["seed"]==42)["R"])
    RS=np.asarray(next(r for r in j["runs"]["soft"] if r["seed"]==42)["R"])
    CX=np.asarray(j["soft_prototypes"]["source_centers"])
    CY=np.asarray(j["soft_prototypes"]["target_centers"])
    with np.load(args.pq_arrays,allow_pickle=False) as z:
        ix=np.asarray(z["source_global_train_indices"],int)
        iy=np.asarray(z["target_global_train_indices"],int)
        assert ix.shape==iy.shape==(160,) and train[ix].all() and train[iy].all()
        assert np.unique(ix).size==np.unique(iy).size==160
        assert np.array_equal(z["source_training_hard_labels"],lx[ix])
        assert np.array_equal(z["target_training_hard_labels"],ly[iy])
        assert np.allclose(X[ix],z["normalized_source_train_points"])
        assert np.allclose(Y[iy],z["normalized_target_train_points"])
    print("VERIFIED 500 trials, 17500 windows, 400/100 split, 14000/3500 points; 160+160 actually used in OT",flush=True)
    four_panels(X[train],Y[train],lx[train],RH,RS,"TRAIN_14000",args.output)
    print("DRAWN full train, common source colors",flush=True)
    four_panels(X[test],Y[test],lx[test],RH,RS,"TEST_3500",args.output)
    print("DRAWN full test, common source colors",flush=True)
    independent_panels(X[train],Y[train],lx[train],ly[train],"TRAIN_14000",args.output)
    independent_panels(X[test],Y[test],lx[test],ly[test],"TEST_3500",args.output)
    independent_panels(X,Y,lx,ly,"ALL_17500",args.output)
    soft_counts=independent_panels(X[test],Y[test],lx[test],ly[test],
                         "TEST_3500",args.output,soft=True,centers=(CX,CY))
    print("DRAWN 3 independent HARD group projections and TEST independent SOFT groups",flush=True)
    count_comparison(X,Y,train,test,lx,ly,ix,iy,args.output)
    imgs=[
      "TRAIN_14000_all_4D_projections_common_postfit_colors.png",
      "TEST_3500_all_4D_projections_common_postfit_colors.png",
      "TRAIN_14000_all_4D_projections_independent_HARD_Kmeans.png",
      "TEST_3500_all_4D_projections_independent_HARD_Kmeans.png",
      "ALL_17500_all_4D_projections_independent_HARD_Kmeans.png",
      "TEST_3500_all_4D_projections_independent_SOFT_prototype.png",
      "POINT_COUNTS_14000_TRAIN_3500_TEST_160_OT.png",
    ]
    meta={
      "dataset":"DANDI000138 v0.220113.0407",
      "NWB_name":d["nwb_name"],"NWB_sha256":d["sha256"],
      "neurons":d["neuron_count"],"event":d["event"],
      "event_window_ms":d["event_window_ms"],"bin_ms":d["bin_ms"],
      "windows_per_trial":d["samples_per_trial"],"trials":d["usable_trials"],
      "all_points_per_domain":17500,"train_points_per_domain":14000,
      "test_points_per_domain":3500,"ot_train_sample_per_domain":160,
      "ot_sample_per_hard_group":40,"train_trials":400,"test_trials":100,
      "source_hard_train_counts":np.bincount(lx[train],minlength=4).tolist(),
      "target_hard_train_counts":np.bincount(ly[train],minlength=4).tolist(),
      "source_hard_test_counts":np.bincount(lx[test],minlength=4).tolist(),
      "target_hard_test_counts":np.bincount(ly[test],minlength=4).tolist(),
      "soft_test_argmax_counts":soft_counts,
      "neural_PCA_top4_variance_explained":d["neural_pca_explained_variance_ratio"],
      "neural_PCA_top4_total":sum(d["neural_pca_explained_variance_ratio"]),
      "original_preprocessing":"sqrt binned neural firing rate, Gaussian 40ms temporal smoothing, train-only zscore, train-only PCA4 (NOT whiten=True); target 4D hand pos xy and vel xy train-only zscore",
      "alignment_scale":"centered by train mean and divided by train RMS radius per modality",
      "reconstruction_from_archived_seed42":True,
      "no_training_refit":True,"no_paired_labels_used_for_fit":True,
      "paired_colors_applied_only_postfit":True,
      "source_ot_original_indices":ix.tolist(),
      "target_ot_original_indices":iy.tolist(),
      "figures":imgs
    }
    (args.output/"FULL_SCATTER_DATA_AUDIT.json").write_text(
       json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    lines=["# 原DANDI四维数据散点图库 — 全样本核查",
      "",
      "本图集**仅重绘**现有严格无监督实验，不重新拟合模型、不改变聚类或传输矩阵。之前 P-Q 图每域只画160个训练点，不能代表本项目的全部17,500个时间窗。",
      "",
      "500有效试次，按运动开始时间对齐，各35个20ms窗口（-100到+600ms），162个神经元；原400/100试次拆分为14,000训练、3,500测试窗口。神经域放电频率sqrt、按时间σ=40ms高斯平滑、按训练zscore、PCA4（未白化）；手部位置xy和速度xy按训练zscore。之后两域分别用训练RMS半径归一化。",
      f"神经四维累计解释标准化神经方差：{100*meta['neural_PCA_top4_total']:.2f}%。",
      "",
      "## 160点、3500点、14000点同尺度对比",
      "",
      "![count](POINT_COUNTS_14000_TRAIN_3500_TEST_160_OT.png)",
      "",
      "## 3500个完整测试点，共同同步物理样本着色（仅训练后）",
      "",
      "![test](TEST_3500_all_4D_projections_common_postfit_colors.png)",
      "",
      "## 14000个完整训练点，共同同步物理样本着色（仅训练后）",
      "",
      "![train](TRAIN_14000_all_4D_projections_common_postfit_colors.png)",
      "",
      "## 3500测试点各域独立Hard分组",
      "",
      "![test hard](TEST_3500_all_4D_projections_independent_HARD_Kmeans.png)",
      "",
      "## 14000训练点各域独立Hard分组",
      "",
      "![train hard](TRAIN_14000_all_4D_projections_independent_HARD_Kmeans.png)",
      "",
      "## 17500全部点各域独立Hard分组",
      "",
      "![all](ALL_17500_all_4D_projections_independent_HARD_Kmeans.png)",
      "",
      "## 3500完整测试点各域独立Soft原型最大归属分组",
      "",
      "![test soft](TEST_3500_all_4D_projections_independent_SOFT_prototype.png)",
      "",
      "注意：共同同步颜色**仅用于绘图**，Target训练簇不是Source复制标签；独立Hard与Soft图中的同编号组不等于已知组对应。左右所有点都在已有四维归一化坐标的二维投影；Source对齐列为原同一批点乘不同的正交R。每个点对应20ms窗口，不能把17500个窗当17500个独立试次。",
      "",
      "[计数、PCA、样本行索引及来源核验](FULL_SCATTER_DATA_AUDIT.json)",
      ""]
    (args.output/"FULL_SCATTER_GALLERY.md").write_text("\n".join(lines),encoding="utf8")
    print("FULL_SCATTER_GALLERY_GENERATED",flush=True)

if __name__=="__main__":
    main()
