#!/usr/bin/env python3
"""Select cluster counts from Hyser TRAINING trials only, independently by domain.

Algorithm: candidates K=2..8, silhouette (maximize) with train trial-drop
stability and minimum mass constraints; simple and inspectable, no OT
performance used in selection. Works on 3D or 5D trial-disjoint feature folds.

This is a model-selection PROTOTYPE, not a proof of statistically 'true' K.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score,normalized_mutual_info_score

K_CANDIDATES=tuple(range(2,9))
RANDOM_STATE=42
MIN_CLUSTER_FRAC=.04
MIN_TRIAL_DROP_ARI=.80
TIE_WINDOW=.01
THREADS_NOTE="OpenBLAS/OMP threads set by Github Action for reproducibility."

def stats_for_k(X,trials,k):
    full=KMeans(n_clusters=k,n_init=15,random_state=42)
    full_label=full.fit_predict(X)
    sil=float(silhouette_score(X,full_label))
    sizes=np.bincount(full_label,minlength=k)
    minfrac=float(sizes.min()/len(X))
    drop_aris=[]
    for trial in sorted(set(trials)):
        mod=KMeans(n_clusters=k,n_init=10,random_state=42).fit(X[trials!=trial])
        drop_aris.append(float(adjusted_rand_score(full_label,mod.predict(X))))
    start_aris=[]
    for random_state in (43,44,45):
        lab=KMeans(n_clusters=k,n_init=10,random_state=random_state).fit_predict(X)
        start_aris.append(float(adjusted_rand_score(full_label,lab)))
    eligible=minfrac>=MIN_CLUSTER_FRAC and min(drop_aris)>=MIN_TRIAL_DROP_ARI
    return dict(
        K=k,silhouette=sil,smallest_cluster_fraction=minfrac,
        smallest_cluster_size=int(sizes.min()),
        trial_drop_ARI_mean=float(np.mean(drop_aris)),
        trial_drop_ARI_min=float(min(drop_aris)),
        seed_ARI_mean=float(np.mean(start_aris)),
        seed_ARI_min=float(min(start_aris)),
        trial_vs_cluster_NMI=float(normalized_mutual_info_score(trials,full_label)),
        eligible=bool(eligible),sizes=sizes.tolist(),
    )

def choose(stats):
    accepted=[s for s in stats if s["eligible"]]
    fallback=False
    if not accepted:
        accepted=stats
        fallback=True
    peak=max(s["silhouette"] for s in accepted)
    close=[s for s in accepted if peak-s["silhouette"]<=TIE_WINDOW]
    result=min(close,key=lambda x:x["K"])
    return result["K"],fallback

def save_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--three-d-dir",type=Path,required=True)
    p.add_argument("--five-d-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    diagnostics=[]; decisions=[]
    for dimension,path,suffix in [
        (3,args.three_d_dir,""),
        (5,args.five_d_dir,"_5d"),
    ]:
        for heldout in range(1,6):
            with np.load(path/f"trial_heldout_{heldout}{suffix}.npz") as z:
                X=np.asarray(z["X_train"],float)/np.sqrt(dimension)
                Y=np.asarray(z["Y_train"],float)/np.sqrt(dimension)
                tids=np.asarray(z["train_trial_id"],int)
            assert X.shape==Y.shape==(1000,dimension)
            for domain,z in (("source_EMG",X),("target_force",Y)):
                rows=[stats_for_k(z,tids,k) for k in K_CANDIDATES]
                selected,fallback=choose(rows)
                for s in rows:
                    diagnostics.append(dict(dimension=dimension,heldout_trial=heldout,
                        domain=domain,selected_K=selected,used_fallback=fallback,
                        **{k:(json.dumps(v) if isinstance(v,list) else v) for k,v in s.items()}))
                decisions.append(dict(dimension=dimension,heldout_trial=heldout,
                    domain=domain,auto_selected_K=selected,selection_fallback=fallback,
                    best_silhouette_K=max(rows,key=lambda s:s["silhouette"])["K"],
                    silhouette_K4=next(s["silhouette"] for s in rows if s["K"]==4),
                    silhouette_K5=next(s["silhouette"] for s in rows if s["K"]==5),
                    chosen_trial_drop_ARI_min=next(s["trial_drop_ARI_min"] for s in rows if s["K"]==selected),
                    chosen_min_mass_frac=next(s["smallest_cluster_fraction"] for s in rows if s["K"]==selected),
                ))
                print("SELECT",dimension,heldout,domain,"K",selected,
                      "K4",round(decisions[-1]["silhouette_K4"],4),
                      "K5",round(decisions[-1]["silhouette_K5"],4),
                      "fallback",fallback,flush=True)
    save_csv(args.out/"candidate_K_per_fold.csv",diagnostics)
    save_csv(args.out/"selected_K_per_fold.csv",decisions)
    summary={"rule":{"K_candidates":list(K_CANDIDATES),
        "primary_metric":"max training mean silhouette",
        "min_fraction":MIN_CLUSTER_FRAC,"min_leave_one_training_trial_out_ARI":MIN_TRIAL_DROP_ARI,
        "silhouette_tie_within":TIE_WINDOW,
        "tie_break":"smallest K within tolerance",
        "fallback_if_none_stable":"highest silhouette without eligibility; flag fallback",
        "fitted_only_training":True,"source_target_separate_K_selection":True,
        "no_physical_labels_no_heldout_no_OT_metric_used_for_selection":True},
        "decisions":decisions}
    (args.out/"automatic_K_selection.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    fig,axs=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    for row,dimension in enumerate((3,5)):
        for col,domain in enumerate(("source_EMG","target_force")):
            ax=axs[row,col]
            for held in range(1,6):
                r=[x for x in diagnostics if x["dimension"]==dimension and x["domain"]==domain and x["heldout_trial"]==held]
                ax.plot([x["K"] for x in r],[x["silhouette"] for x in r],
                        marker="o",markersize=3,alpha=.65,label=f"heldout {held}")
            ax.axvline(4,linestyle="--",alpha=.5)
            ax.axvline(5,linestyle=":",alpha=.5)
            ax.set(title=f"{dimension}D {domain}",xlabel="K (train only)",
                   ylabel="silhouette",xticks=K_CANDIDATES)
            ax.grid(alpha=.15)
            if row==0 and col==1:ax.legend(frameon=False,ncol=2,fontsize=8)
    fig.suptitle("Hyser: per-trial training-only K selection, old 3D vs all-fingers 5D")
    fig.savefig(args.out/"K_silhouette_3D_5D_fivefold.png",dpi=160,bbox_inches="tight")
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    for ax,dim in zip(axes,(3,5)):
        z=[v for v in decisions if v["dimension"]==dim]
        labels=["X","Y"]
        for col,domain in enumerate(("source_EMG","target_force")):
            selected=[s["auto_selected_K"] for s in z if s["domain"]==domain]
            ax.scatter(range(1,6),selected,marker="o" if col==0 else "s",
                s=65,label=labels[col])
        ax.set(ylim=(1.5,8.5),xticks=range(1,6),yticks=K_CANDIDATES,
            xlabel="heldout trial",ylabel="automatic K (TRAIN ONLY)",
            title=f"{dim}D independent cluster selection")
        ax.legend(frameon=False);ax.grid(alpha=.12)
    fig.savefig(args.out/"automatic_K_choices_fivefold.png",dpi=170,bbox_inches="tight")
    plt.close(fig)
    chosen3=[(z["auto_selected_K"]) for z in decisions if z["dimension"]==3]
    chosen5=[(z["auto_selected_K"]) for z in decisions if z["dimension"]==5]
    lines=["# Hyser 自动选择组数K：五次独立试次训练内筛选","",
           "## 当前可运行的模型选择模块",
           "分别对每折训练数据的源/目标域进行K=2..8扫描。主要指标为轮廓系数；同时要求",
           "最小簇权重至少4%，且四种「去掉一个训练试次重新拟合」聚类与全训练簇标签",
           "的最小ARI至少0.80。若多候选的轮廓系数与最佳相差不超过0.01，选更小K。",
           "若没有满足约束的K，则回退到轮廓系数最高者，并记录fallback。",
           "测试试次严格禁止进入K选择、PCA、白化和KMeans训练。",
           "",
           "## 统计汇总",
           f"- 三维训练折中的十个域选择：{chosen3}",
           f"- 五指五维训练折中的十个域选择：{chosen5}",
           "- 如果X/Y选择出不同K，层级OT必须支持**非方形 P (Kx×Ky)**、各自边缘1/Kx和1/Ky。",
           "- 不可以强迫另一个域继承源域K，也不能用真实同步组标签选择K。",
           "- 自动K只是无监督结构启发式，不保证动作标签个数、物理可解释性或OT最优值。",
           "- 没有正式训练自动K HiWA的部分不得写成“自动K模型已经优于固定K模型”。"]
    (args.out/"AUTO_K_METHOD_ZH.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"status":"success","selected3D":chosen3,"selected5D":chosen5},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
