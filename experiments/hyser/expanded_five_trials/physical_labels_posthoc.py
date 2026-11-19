#!/usr/bin/env python3
"""POST-HOC force interpretability audit for real Hyser five-trial data.

Both X and Y KMeans clusterings use only their own 3D domain features.
True synchronized row identities appear *only in this reporting script* and
are NEVER passed into the unsupervised OT experiment.
The force-derived state proxy is diagnostic, not a ground-truth action label.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

FINGERS = ["thumb", "middle", "index"]
COLORS = ["#2776ae", "#dc9456", "#4f9c83", "#9062b0"]
from diagnose_five_trials import whiten

def state_proxy(force):
    med = np.median(force, axis=0)
    mad = np.median(np.abs(force - med), axis=0)
    # Relative scale can mask force channel physical dominance; raw forces reported alongside.
    sc = np.maximum(1.4826 * mad, 0.1 * np.std(force, axis=0))
    sc = np.maximum(sc, 1e-9)
    z = (force - med) / sc
    mag = np.linalg.norm(z, axis=1)
    threshold = float(np.quantile(mag, .20))
    label = np.argmax(np.abs(z), axis=1) + 1
    label[mag <= threshold] = 0
    return label, z, {"median_force": med.tolist(), "MAD": mad.tolist(),
                       "robust_scale": sc.tolist(), "quiet_score_20th_pct": threshold,
                       "definition": "Low deviation (bottom 20% robust relative norm), or dominant absolute robust-scaled finger deviation; NOT verified gesture labels."}

def association(cluster, state, trial, n_perm=500):
    observed = float(normalized_mutual_info_score(cluster, state))
    rng = np.random.default_rng(519)
    null = []
    for _ in range(n_perm):
        shifted = state.copy()
        # Circular-shift each 25s trial independently, keeping serial structure
        for t in range(1,6):
            idx = np.flatnonzero(trial==t)
            shifted[idx] = np.roll(state[idx], rng.integers(1, len(idx)))
        null.append(normalized_mutual_info_score(cluster, shifted))
    p = (sum(s >= observed for s in null)+1)/(n_perm+1)
    return {"NMI": observed, "null_shift_mean": float(np.mean(null)),
            "null_shift_95pct": float(np.quantile(null, .95)),
            "circular_shift_p_exploratory": float(p),
            "n_null": n_perm}

def one_domain(domain, labs, force, physical_state, z, trial, out):
    tab = confusion_matrix(labs, physical_state, labels=np.arange(4))
    # physical state values 0..3; cluster values 0..3
    raw_stats=[]
    for k in range(4):
        vals=force[labs==k]
        zvals=z[labs==k]
        freqs=np.bincount(physical_state[labs==k], minlength=4)
        raw_stats.append({
            "cluster": int(k+1), "n": int(len(vals)),
            "mean_raw_force": vals.mean(0).tolist(),
            "median_raw_force": np.median(vals,axis=0).tolist(),
            "std_raw_force": vals.std(0).tolist(),
            "mean_absolute_robust_scaled_force": np.abs(zvals).mean(0).tolist(),
            "proxy_label_counts": freqs.tolist(),
            "proxy_label_fractions": (freqs/freqs.sum()).tolist(),
        })
    assoc=association(labs, physical_state, trial)
    fig,ax=plt.subplots(figsize=(9,5.4))
    im=ax.imshow(tab,aspect="auto",cmap="Blues")
    ax.set(xticks=range(4), yticks=range(4),
        xticklabels=["low-relative-force","thumb deviation","middle deviation","index deviation"],
        yticklabels=[f"cluster {k+1}" for k in range(4)],
        xlabel="Post-hoc physical force proxy only (not a training label)",
        ylabel=f"{domain} independently fitted K-means 4 clusters",
        title=f"Hyser {domain}: cluster-to-force-proxy counts")
    for (i,j),v in np.ndenumerate(tab):
        ax.text(j,i,str(v),ha="center",va="center",fontsize=11,
                color="white" if v>tab.max()*.65 else "#17293b")
    fig.colorbar(im,ax=ax,label="number of 100ms windows")
    fig.savefig(out/f"{domain}_force_proxy_confusion.png",dpi=175,bbox_inches="tight")
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5.2))
    means=np.array([row["mean_absolute_robust_scaled_force"] for row in raw_stats])
    xx=np.arange(4);w=.23
    for finger in range(3):
        ax.bar(xx+(finger-1)*w,means[:,finger],width=w,
               label=FINGERS[finger])
    ax.set(xticks=xx,xticklabels=[f"cluster {i+1}" for i in xx],
           xlabel=f"{domain} independent cluster",
           ylabel="mean |force - median| / robust scale",
           title=f"Hyser {domain}: post-hoc finger-force profile")
    ax.legend(frameon=False)
    fig.savefig(out/f"{domain}_force_magnitude_by_cluster.png",dpi=175,bbox_inches="tight")
    plt.close(fig)
    return {"domain":domain,"cluster_stats":raw_stats,
            "force_proxy_association":assoc,"cluster_to_proxy_table":tab.tolist()}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args();out=args.out;out.mkdir(exist_ok=True,parents=True)
    with np.load(args.data) as z:
        X=np.array(z["X_emg_log1p_rms_256"],float)
        Y=np.array(z["Y_force_thumb_middle_index_3"],float)
        trial=np.array(z["trial_id"],int)
    assert X.shape==(1250,256) and Y.shape==(1250,3)
    scaler=StandardScaler()
    X3=whiten(PCA(n_components=3,svd_solver="full").fit_transform(scaler.fit_transform(X)))/np.sqrt(3)
    Y3=whiten(Y)/np.sqrt(3)
    lx=KMeans(n_clusters=4,n_init=30,random_state=42).fit_predict(X3)
    ly=KMeans(n_clusters=4,n_init=30,random_state=42).fit_predict(Y3)
    proxy,z,defn=state_proxy(Y)
    # "Y", "trial", and "physical_state" cannot appear in the
    # unsupervised source membership initialization: report only.
    xres=one_domain("source_EMG",lx,Y,proxy,z,trial,out)
    yres=one_domain("target_force",ly,Y,proxy,z,trial,out)
    # Create explicit per-window reality-derived proxy labels.
    # These labels are POSTHOC descriptive metadata only, NOT ground-truth
    # actions and NEVER model fitting, tuning, or classifier supervision.
    names=["low_relative_force","thumb_deviation_dominant",
           "middle_deviation_dominant","index_deviation_dominant"]
    with (out/"physical_force_labels_POSTHOC_ONLY.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["trial_id","window_start_seconds","force_thumb","force_middle",
                         "force_index","force_proxy_state","force_proxy_code",
                         "source_kmeans_cluster_POSTHOC","target_kmeans_cluster_POSTHOC",
                         "REAL_ACTION_GROUND_TRUTH"])
        for i in range(len(Y)):
            writer.writerow([int(trial[i]),round((i%250)*.1,6),
                             *[float(v) for v in Y[i]],names[int(proxy[i])],
                             int(proxy[i]),int(lx[i])+1,int(ly[i])+1,"NO"])

    result={"status":"post_hoc_only","n":1250,"physics_label_is_ground_truth":False,
            "dataset_known_semantics":"Random contractions of arbitrary finger combinations; no prescribed four directions",
            "force_channel_order":"thumb,middle,index (old 3D extraction uses columns 0,2,1 from official thumb,index,middle,ring,little header)",
            "force_summary":{"mins":Y.min(0).tolist(),"p10":np.percentile(Y,10,axis=0).tolist(),
                             "median":np.median(Y,axis=0).tolist(),"p90":np.percentile(Y,90,axis=0).tolist(),
                             "maxs":Y.max(0).tolist()},
            "proxy_definition":defn,
            "proxy_counts":np.bincount(proxy,minlength=4).tolist(),
            "source":xres,"target":yres}
    (out/"physical_label_diagnostics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["# Hyser 1250 点四簇的现实解释（仅训练后诊断）","",
           "**Random Task 没有原始的四个动作方向标签。不能把 K-means 的四簇当作真实动作四类。**",
           "其真实记录是五个手指各自的力量。我们从中提取 thumb/index/middle 三通道，以簇内平均力量模式解释统计簇。"]
    for res in [xres,yres]:
        a=res["force_proxy_association"]
        lines+=["",f"## {res['domain']}",
                f"- NMI(四簇, 代理力量状态): {a['NMI']:.4f}",
                f"- 25秒试次内循环移位零假设的 NMI 95%分位: {a['null_shift_95pct']:.4f}",
                f"- 探索性循环移位 p: {a['circular_shift_p_exploratory']:.4f}",
                "- 此代理标签由力量中位数偏差构成，低力量近似取最低20%（人为阈值），其他三类取最大稳健标准化力量偏差的手指。",
                "- 这种代理不是实验者提供的真实动作标注，统计关联不能证明类别有因果或动作方向对应。"]
    lines+=["","## 训练信息隔离","训练输入仅分别为 X、Y 特征，各域独立聚类；源域簇与目标力量的对照只在事后按真实时间索引查看，不参与任何训练、P/Q/R选择、调参或随机种子选择。"]
    (out/"INTERPRETABILITY_ZH.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"source":xres["force_proxy_association"],"target":yres["force_proxy_association"],"force_summary":result["force_summary"],"proxy_counts":result["proxy_counts"]},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
