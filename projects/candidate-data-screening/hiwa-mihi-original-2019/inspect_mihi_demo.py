"""Dataset-selection PREFLIGHT only. No HiWA or Soft-GCOT training.

Public authors' `nerdslab/PyHiWA/data/mihi_demo.mat`.
Question: do genuine neural FA3 and hand movement 3D hold visible shared
cluster / distance structure prior to optimizing OT?
Source/target KMeans are independently fitted WITHOUT labels; labels used
only in post hoc audit. Mat data may contain old train/test in author names;
we do not assert unseen trial independence or call resulting diagnostic a
heldout test of our method.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr,pearsonr
from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def arr(v):return np.asarray(v)
def plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    return v
def norm(z):
    mu=z.mean(0)
    rms=np.sqrt(np.mean(np.sum((z-mu)**2,axis=1)))
    return (z-mu)/rms
def stats(x,y):
    a=x.ravel();b=y.ravel()
    if len(a)<3 or np.std(a)==0 or np.std(b)==0:return None
    return {"Pearson":float(pearsonr(a,b).statistic),"Spearman":float(spearmanr(a,b).statistic)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    d=loadmat(args.input)
    keys=["Tte","Ttr","Xte","Xtr","Yte","Ytr"]
    for k in keys:
        assert k in d,k
    shapes={k:list(d[k].shape) for k in keys}
    labels_src=d["Tte"].ravel()
    labels_tgt=d["Ttr"].ravel()
    neural0=np.asarray(d["Yte"],float)
    hand0=np.asarray(d["Xtr"],float)
    assert len(labels_src)==len(neural0)
    assert len(labels_tgt)==len(hand0)
    assert neural0.ndim==2 and hand0.ndim==2 and hand0.shape[1]>=2
    assert np.isfinite(neural0).all() and np.isfinite(hand0).all()
    neural0=neural0[:,~np.all(neural0==neural0[0],axis=0)]
    fa=FactorAnalysis(n_components=3,random_state=42,max_iter=200,tol=1e-4)
    neural3=norm(fa.fit_transform(neural0))
    hand3=norm(np.column_stack([hand0[:,0],hand0[:,1],
                                np.linalg.norm(hand0[:,:2],axis=1)]))
    ncl_src=np.unique(labels_src);ncl_tgt=np.unique(labels_tgt)
    assert len(ncl_src)==len(ncl_tgt)==4,(ncl_src,ncl_tgt)
    hard_src=KMeans(n_clusters=4,n_init=20,random_state=42).fit(neural3)
    hard_tgt=KMeans(n_clusters=4,n_init=20,random_state=43).fit(hand3)
    cc_src=np.vstack([neural3[labels_src==k].mean(0) for k in ncl_src])
    cc_tgt=np.vstack([hand3[labels_tgt==k].mean(0) for k in ncl_src])
    centre_cor=stats(pdist(cc_src),pdist(cc_tgt))
    synced=None
    if d["Xte"].shape[0]==len(neural3) and d["Xte"].shape[1]>=2:
        hte=np.asarray(d["Xte"],float)
        if np.isfinite(hte).all():
            hte=norm(np.column_stack([hte[:,0],hte[:,1],
                                 np.linalg.norm(hte[:,:2],axis=1)]))
            rng=np.random.default_rng(20261119)
            idx0=rng.integers(0,len(neural3),size=15000)
            idx1=rng.integers(0,len(neural3),size=15000)
            keep=idx0!=idx1;idx0=idx0[keep];idx1=idx1[keep]
            src_dist=np.linalg.norm(neural3[idx0]-neural3[idx1],axis=1)
            hand_dist=np.linalg.norm(hte[idx0]-hte[idx1],axis=1)
            synced=stats(src_dist,hand_dist)
    score={
        "dataset":"Authors original PyHiWA real monkey center-out reaching (mihi_demo.mat)",
        "source":"https://github.com/nerdslab/PyHiWA/blob/master/data/mihi_demo.mat",
        "original_file_sha256":hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "file_bytes":args.input.stat().st_size,
        "keys_and_shapes":shapes,
        "source_label_counts":{str(i):int(np.sum(labels_src==i)) for i in ncl_src},
        "target_label_counts":{str(i):int(np.sum(labels_tgt==i)) for i in ncl_tgt},
        "same_direction_class_set":bool(np.array_equal(np.sort(ncl_src),np.sort(ncl_tgt))),
        "finite_source_and_target":True,
        "neural_nonconstant_input_channels":int(neural0.shape[1]),
        "source_3d":"authors FA3 on nonconstant Yte neural source, norm by unpaired RMS",
        "target_3d":"authors hand 2D (x,y,sqrt(x2+y2)), norm by unpaired RMS",
        "neural_FA3_niter":fa.n_iter_,
        "neural_KMeans_four_groups":{
            "counts":np.bincount(hard_src.labels_,minlength=4).tolist(),
            "silhouette":float(silhouette_score(neural3,hard_src.labels_)),
            "posthoc_nmi_vs_real_direction":float(normalized_mutual_info_score(labels_src,hard_src.labels_)),
            "posthoc_ARI_vs_real_direction":float(adjusted_rand_score(labels_src,hard_src.labels_)),
        },
        "hand_KMeans_four_groups":{
            "counts":np.bincount(hard_tgt.labels_,minlength=4).tolist(),
            "silhouette":float(silhouette_score(hand3,hard_tgt.labels_)),
            "posthoc_nmi_vs_real_direction":float(normalized_mutual_info_score(labels_tgt,hard_tgt.labels_)),
            "posthoc_ARI_vs_real_direction":float(adjusted_rand_score(labels_tgt,hard_tgt.labels_)),
        },
        "posthoc_four_true_direction_centroid_distance_geometry_correlation":centre_cor,
        "posthoc_synchronized_Xte_Yte_pairwise_distance_correlation_15k_pairs":synced,
        "LABELS_NEVER_USED_FOR_OT_FITTING":True,
        "only_unsupervised_kmeans_and_fa_fitted":True,
        "author_original_demo_was_not_strict_unpaired_group_discovery":True,
        "no_trial_metadata_verified":True,
        "not_claiming_generalized_performance":True,
    }
    fig=plt.figure(figsize=(12,8),layout="constrained")
    panels=[("Source neural, independent KMeans",neural3,hard_src.labels_),
            ("Hand movement, independent KMeans",hand3,hard_tgt.labels_),
            ("Source neural, TRUE direction (postfit)",neural3,labels_src),
            ("Hand movement, TRUE direction (postfit)",hand3,labels_tgt)]
    for i,(name,pts,lab) in enumerate(panels):
        ax=fig.add_subplot(2,2,i+1,projection="3d")
        for k in np.unique(lab):
            pp=pts[lab==k]
            ax.scatter(*pp.T,s=4,alpha=.48,label=str(k))
        ax.set_title(name)
        ax.set_xlabel("1");ax.set_ylabel("2");ax.set_zlabel("3")
        ax.view_init(20,-63)
    fig.suptitle("HiWA original real monkey-reaching data: unsupervised vs postfit TRUE label visualization")
    fig.savefig(out/"MIHI_REAL_DATA_PREFLIGHT_K4_SOURCE_TARGET_3D.png",dpi=160,bbox_inches="tight")
    plt.close(fig)
    score["plot"]="MIHI_REAL_DATA_PREFLIGHT_K4_SOURCE_TARGET_3D.png"
    (out/"REAL_MIHIDEMO_PREFLIGHT.json").write_text(json.dumps(plain(score),ensure_ascii=False,indent=2)+"\n")
    print("MIHI_PREFLIGHT_RESULT",json.dumps(plain(score),ensure_ascii=False),flush=True)
if __name__=="__main__":main()
