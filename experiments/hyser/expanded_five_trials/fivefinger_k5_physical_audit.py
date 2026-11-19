#!/usr/bin/env python3
"""Exploratory K5 physical interpretation on ORIGINAL 5-finger Hyser force.

Zero cross-domain correspondence is used for KMeans. Force values are inspected
AFTER clustering to label dominant-finger force patterns, not actual motion
classes. Pooled feature transformation is visualization-only; never OT training.
"""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from scipy.spatial.distance import cdist

FINGERS=["thumb","index","middle","ring","little"]
COLORS=["#2776ae","#dc9456","#4f9c83","#9062b0","#d25171"]

def white(x):
    z=x-x.mean(0)
    vals,vecs=np.linalg.eigh(np.cov(z,rowvar=False))
    if vals.min()<1e-10:raise ValueError(f"Singular 5-finger covariance {vals}")
    return z@(vecs*vals**-.5)@vecs.T

def scatter_3(z,labs,title,file):
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True)
    for ax,(a,b) in zip(axes,[(0,1),(0,2),(1,2)]):
        for k,col in enumerate(COLORS):
            m=labs==k
            ax.scatter(z[m,a],z[m,b],s=13,c=col,alpha=.72,edgecolors="none")
        ax.set(xlabel=f"dim {a+1}",ylabel=f"dim {b+1}",title=f"projection {a+1}-{b+1}")
        ax.set_aspect("equal",adjustable="datalim");ax.grid(alpha=.15)
    fig.suptitle(title)
    fig.savefig(file,dpi=160,bbox_inches="tight");plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    with np.load(a.data) as z:
        X256=np.array(z["X_emg_log1p_rms_256"],float)
        Y5=np.array(z["Y_force_5_raw"],float)
        trial=np.array(z["trial_id"],int)
        names=z["force_names"].tolist()
    assert names==FINGERS and X256.shape==(1250,256) and Y5.shape==(1250,5)
    X5=white(PCA(n_components=5,svd_solver="full").fit_transform(StandardScaler().fit_transform(X256)))/np.sqrt(5)
    Yw=white(Y5)/np.sqrt(5)
    summaries=[]
    for domain,z in (("EMG_source_PCA5",X5),("five_finger_force",Yw)):
        for K in (4,5):
            km=KMeans(n_clusters=K,random_state=42,n_init=30).fit(z)
            lab=km.labels_
            d={"domain":domain,"K":K,"n":len(z),
               "silhouette":float(silhouette_score(z,lab)),
               "sizes":np.bincount(lab,minlength=K).tolist(),
               "trial_vs_cluster_NMI":float(normalized_mutual_info_score(trial,lab))}
            if K==5:
                scatter_3(z,lab,f"{domain}: K=5 independent clusters, all 1250 windows",a.out/f"{domain}_K5_2D.png")
                fig=plt.figure(figsize=(8,7));ax=fig.add_subplot(111,projection="3d")
                for i,c in enumerate(COLORS):
                    m=lab==i
                    ax.scatter(z[m,0],z[m,1],z[m,2],s=12,c=c,alpha=.72,depthshade=False)
                ax.set(xlabel="latent dim 1",ylabel="latent dim 2",zlabel="latent dim 3",
                       title=f"{domain}: K5 first 3 latent dimensions")
                fig.savefig(a.out/f"{domain}_K5_3D.png",dpi=175,bbox_inches="tight")
                plt.close(fig)
            if domain=="five_finger_force":
                means=np.array([Y5[lab==i].mean(0) for i in range(K)])
                medians=np.array([np.median(Y5[lab==i],axis=0) for i in range(K)])
                zs=(Y5-np.median(Y5,axis=0))/np.maximum(Y5.std(0),1e-10)
                absv=np.array([np.abs(zs[lab==i]).mean(0) for i in range(K)])
                dom=np.argmax(absv,axis=1)
                d.update({"cluster_mean_raw_force":means.tolist(),
                         "cluster_median_raw_force":medians.tolist(),
                         "cluster_mean_abs_standardized_deviation":absv.tolist(),
                         "largest_deviation_finger_by_cluster":[FINGERS[j] for j in dom],
                         "num_distinct_dominant_fingers":int(len(set(dom)))})
                fig,axs=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
                labels=np.arange(K)
                w=.13
                for j,fn in enumerate(FINGERS):
                    offset=(j-2)*w
                    axs[0].bar(labels+offset,means[:,j],width=w,label=fn)
                    axs[1].bar(labels+offset,absv[:,j],width=w,label=fn)
                axs[0].set(title="raw five-finger mean force by independent cluster",xticks=labels,
                           xticklabels=[f"C{k+1}" for k in labels],ylabel="raw force (instrument units)")
                axs[1].set(title="mean |force-median| / channel SD",xticks=labels,
                           xticklabels=[f"C{k+1}" for k in labels],ylabel="standardized force deviation")
                axs[1].legend(frameon=False,ncol=3,fontsize=8)
                fig.suptitle(f"Hyser original FIVE fingers: target K={K} physical profiles")
                fig.savefig(a.out/f"fivefinger_target_K{K}_cluster_force_profiles.png",dpi=185,bbox_inches="tight")
                plt.close(fig)
                if K==5:
                    fig,axes=plt.subplots(1,3,figsize=(14.5,4.5),constrained_layout=True)
                    for ax,(u,v) in zip(axes,[(0,1),(2,3),(3,4)]):
                        for k,col in enumerate(COLORS):
                            m=lab==k
                            ax.scatter(Y5[m,u],Y5[m,v],s=15,c=col,alpha=.72,edgecolors="none")
                        ax.set(xlabel=FINGERS[u],ylabel=FINGERS[v],title=f"{FINGERS[u]} vs {FINGERS[v]}")
                        ax.grid(alpha=.16)
                    fig.suptitle("Target K5 clusters in ORIGINAL physical finger force coordinates")
                    fig.savefig(a.out/"fivefinger_target_K5_original_finger_pairs.png",dpi=180,bbox_inches="tight")
                    plt.close(fig)
            summaries.append(d)
            print(domain,"K",K,"silhouette",d["silhouette"],"sizes",d["sizes"],flush=True)
    final={"source":"Hyser original 5 finger force records: thumb,index,middle,ring,little",
           "sample_count":1250,"study_type":"Pooled descriptive EDA only; no model fit/test selection",
           "physical_finger_to_cluster_labels_are_NOT_ground_truth":True,
           "clustering":"K-means independently fitted in 5D standardized-whitened space",
           "summary":summaries}
    (a.out/"fivefinger_K4_K5_physical_summary.json").write_text(json.dumps(final,ensure_ascii=False,indent=2)+"\n")
    lines=["# Hyser 真正五根手指：四簇与五簇物理意义诊断","",
           "Random Task没有五个单指动作真标签。以下使用原始五指力量测量对无监督统计簇作训练后描述。"
           "不把源域簇编号直接对到目标域簇编号。",
           ""]
    for d in summaries:
        lines.append(f"- {d['domain']} K={d['K']}: silhouette={d['silhouette']:.4f}; size={d['sizes']}; trial-NMI={d['trial_vs_cluster_NMI']:.4f}")
        if d['domain']=="five_finger_force":
            lines.append(f"  - 最大稳健化力量偏差手指：{d['largest_deviation_finger_by_cluster']}，其中不同手指总数={d['num_distinct_dominant_fingers']}")
    lines+=["","**判定**：同一个手指可能参与多个簇，多个手指也可共同决定一个簇；这里的最大力量偏差是描述性标签，不能当作经实验标注的真实动作。"]
    (a.out/"PHYSICAL_INTERPRETATION_ZH.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"done":True,"summary":summaries},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
