#!/usr/bin/env python3
"""Aggregate SA/WA/modified DAD-like real 1000-point Hyser controls.

20 fold-by-representation TRAIN Fisher montages show ALL 1000 source
and ALL 1000 target points for EVERY model panel. Full-dimensional
numeric evaluation is never performed in the Fisher 2D plot plane.
"""
from __future__ import annotations
import argparse,csv,json,math,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from best_fisher_2d import fisher2
from additional_unsupervised_baselines import solve_icp,solve_single_ot

REPS=("PCA3_PCA3","FA3_FA3","PCA5_RAW5","FA5_RAW5")
METHODS=("Identity","ICP","SingleOT","SA","WA","DAD_like_3D","Hard HiWA","Soft HiWA")
COLORS=("#2677AA","#DA9051","#4A9376","#A06CB3")

def csvout(path,rows):
    rows=list(rows)
    with path.open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

def frozen_run(root,rep,fold,seed):
    dim3=rep.endswith(("PCA3","FA3"))
    folder="results_formal_full1000_3D" if dim3 else "results_formal_full1000_Qfixed"
    prefix="hyser-threeD-" if dim3 else "hyser-qfixed-"
    f=root/folder/"individual_runs"/(f"{prefix}{rep}-f{fold}-s{seed}")/"local_ADMM_results.json"
    z=json.loads(f.read_text())["runs"]
    ret={r["method"]:r for r in z}
    assert set(ret)=={"hard","soft05"}
    assert all(r["n_OT_train_source"]==r["n_OT_train_target"]==1000
               and r["Q_constraints_passed"] for r in z)
    return ret

def features(root,rep,fold):
    dim3=rep.endswith(("PCA3","FA3"))
    d=3 if dim3 else 5
    f=root/rep/(f"trial_heldout_{fold}"+("" if dim3 else "_5d")+".npz")
    with np.load(f) as a:
        x=np.asarray(a["X_train"],float)/np.sqrt(d)
        y=np.asarray(a["Y_train"],float)/np.sqrt(d)
    assert x.shape==y.shape==(1000,d)
    return x,y

def method_rotations(rep,fold,x,y,full,frozen):
    m={"Identity":np.eye(x.shape[1]),
       "ICP":solve_icp(x,y,42)[0],
       "SingleOT":solve_single_ot(x,y,42)[0]}
    for method in ("SA","WA","DAD_like_3D"):
        z=[r for r in full if r["representation"]==rep and r["fold"]==fold
           and r["method"]==method and r["seed"]==42 and r["qualified"]]
        if z:
            assert len(z)==1
            m[method]=np.asarray(z[0]["R"],float)
    h=frozen_run(frozen,rep,fold,42)
    m["Hard HiWA"]=np.asarray(h["hard"]["R"],float)
    m["Soft HiWA"]=np.asarray(h["soft05"]["R"],float)
    assert set(m)==set(METHODS)-({"DAD_like_3D"} if x.shape[1]==5 else set())
    assert all(np.linalg.norm(R.T@R-np.eye(x.shape[1]))<1e-6 for R in m.values())
    return m

def own(ax,z,labels,title):
    for k in range(4):
        mask=labels==k
        ax.scatter(z[mask,0],z[mask,1],s=8,alpha=.52,c=COLORS[k],
                   linewidths=0,rasterized=True)
    ax.set_title(title,fontsize=9)
    ax.set_xlabel("Fisher 1");ax.set_ylabel("Fisher 2");ax.grid(alpha=.1)

def overlay(ax,z,tgt,labs,title,xlim,ylim,legend=False):
    assert len(z)==len(tgt)==len(labs)==1000
    ax.scatter(tgt[:,0],tgt[:,1],s=10,c="#414952",alpha=.32,
               linewidths=0,rasterized=True,label="1000 target")
    for k in range(4):
        mask=labs==k
        ax.scatter(z[mask,0],z[mask,1],s=8,alpha=.6,c=COLORS[k],
                   linewidths=0,rasterized=True,label=f"Source group{k+1}")
    ax.set(xlim=xlim,ylim=ylim,xlabel="Target Fisher 1",ylabel="Target Fisher 2")
    ax.set_title(title,fontsize=9);ax.grid(alpha=.09)
    if legend:ax.legend(frameon=False,fontsize=7,ncol=2)

def all_scatter(out,full,frozen,feat):
    p=out/"figures";p.mkdir(exist_ok=True)
    indiv=p/"individual_fold1";indiv.mkdir(exist_ok=True)
    audit=[]
    for rep in REPS:
        for fold in range(1,6):
            x,y=features(feat,rep,fold)
            m=method_rotations(rep,fold,x,y,full,frozen)
            kmx=KMeans(4,n_init=30,random_state=42).fit(x)
            kmy=KMeans(4,n_init=30,random_state=43).fit(y)
            fx=fisher2(x,kmx.labels_)
            fy=fisher2(y,kmy.labels_)
            ownsrc=(x-fx["mean"])@fx["basis"]
            owntgt=(y-fy["mean"])@fy["basis"]
            zs={name:(x@R.T-fy["mean"])@fy["basis"] for name,R in m.items()}
            before=(x-fy["mean"])@fy["basis"]
            both=np.vstack([owntgt,before,*zs.values()])
            minimum=both.min(axis=0);maximum=both.max(axis=0)
            pad=np.maximum((maximum-minimum)*.04,.08)
            xlim=(minimum[0]-pad[0],maximum[0]+pad[0])
            ylim=(minimum[1]-pad[1],maximum[1]+pad[1])
            items=[("Before alignment",before),*list(zs.items())]
            n=2+len(items)
            nrows=int(math.ceil(n/3))
            fig,axes=plt.subplots(nrows,3,figsize=(17.5,4.8*nrows),
                                  constrained_layout=True)
            aa=list(axes.ravel())
            own(aa[0],ownsrc,kmx.labels_,"EMG source 1000 · SOURCE-Fisher")
            own(aa[1],owntgt,kmy.labels_,"Force target 1000 · TARGET-Fisher")
            for ix,(name,z) in enumerate(items,start=2):
                overlay(aa[ix],z,owntgt,kmx.labels_,
                        name+" · 1000 source + 1000 target",xlim,ylim)
            for ax in aa[n:]:ax.axis("off")
            fig.suptitle(
                f"Hyser {rep} | fold {fold} | all 1000 TRAIN points per domain | seed42\n"
                "Source colors are independent TRAIN KMeans clusters; target gray. "
                "Fisher is ONLY a display, not a model input.",fontsize=10)
            name=f"{rep}_FOLD{fold}_ALL1000_Fisher_METHODS.png"
            fig.savefig(p/name,dpi=145)
            plt.close(fig)
            audit.append({"rep":rep,"fold":fold,"file":name,
                          "source_points_each_method":1000,
                          "target_points_each_method":1000,
                          "method_names":list(m),
                          "source_cluster_counts":np.bincount(kmx.labels_,minlength=4).tolist(),
                          "target_cluster_counts":np.bincount(kmy.labels_,minlength=4).tolist(),
                          "test_not_used_to_fit_fisher":True,
                          "source_target_labels_independent":True})
            if fold==1:
                for method,z in zs.items():
                    fig,ax=plt.subplots(figsize=(7.5,6.0),constrained_layout=True)
                    overlay(ax,z,owntgt,kmx.labels_,
                            f"{rep} {method} | ALL 1000 + 1000 TRAIN | Target Fisher2D",
                            xlim,ylim,legend=True)
                    name=f"{rep}_FOLD1_{method.replace(' ','_')}_ALL1000_Fisher.png"
                    fig.savefig(indiv/name,dpi=170)
                    plt.close(fig)
            print("FULL1000_FISHER_FIG",rep,fold,flush=True)
    assert len(audit)==20
    (out/"scatter_count_audit.json").write_text(json.dumps({
        "figures":audit,"Fisher2D_only_for_plotting":True,
        "each_overlay_contains_1000_source_and_1000_target":True,
        "n_fold_gallery":20
    },ensure_ascii=False,indent=2)+"\n",encoding="utf8")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--baselines-dir",required=True,type=Path)
    ap.add_argument("--features",required=True,type=Path)
    ap.add_argument("--frozen",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    new=[]
    for rep in REPS:
        obj=json.loads((a.baselines_dir/rep/"runs_bundle.json").read_text())
        rs=obj["all_results_including_failures"]
        assert obj["representation"]==rep and obj["full1000_train_data"]
        assert len(rs)==(20 if rep.endswith(("PCA3","FA3")) else 15)
        assert all(r["source_train_n"]==r["target_train_n"]==1000 for r in rs)
        new.extend(rs)
    assert len(new)==70 and len({(r["representation"],r["fold"],r["seed"],r["method"]) for r in new})==70
    old={}
    for d in ("3D","5D"):
        p=a.frozen/"results_3D_5D_unified"/d/"full_physical_distance_curves.json"
        obj=json.loads(p.read_text())
        assert obj["n_runs"]==40
        for z in obj["records"]:
            k=(z["config"],z["fold"],z["seed"],z["method"])
            assert k not in old
            old[k]=z
    assert len(old)==80
    (a.out/"all_SA_WA_DAD_run_records.json").write_text(json.dumps({
        "DANDI_algorithm_source_commit":"38fb630001e11f5045a2a1055b2aa2686e7bfd34",
        "full_1000_training_pool":True,
        "WA_stochastic_batch_size":250,
        "DAD_is_modified_3D_not_official":True,
        "results":new
    },ensure_ascii=False,indent=2)+"\n",encoding="utf8")

    summary=[];perfold=[];curve=[]
    grouped={}
    for rep in REPS:
        methods=("Hard HiWA","Soft HiWA","SA","WA")+(() if rep.endswith("RAW5") else ("DAD_like_3D",))
        for method in methods:
            if method.startswith(("Hard","Soft")):
                tag="hard" if method.startswith("Hard") else "soft05"
                rows=[{"fold":f,"seed":seed,"qualified":True,
                       "hit5":old[rep,f,seed,tag]["results"]["physical_hit"]["0.05"],
                       "hit20":old[rep,f,seed,tag]["results"]["physical_hit"]["0.2"],
                       "L2":old[rep,f,seed,tag]["results"]["physical_error_mean"],
                       "errors":old[rep,f,seed,tag]["results"]["physical_errors_250"]}
                      for f in range(1,6) for seed in (42,43)]
            else:
                rows=[{"fold":z["fold"],"seed":z["seed"],"qualified":z["qualified"],
                       "hit5":z.get("hit_q05"),"hit20":z.get("hit_q20"),
                       "L2":z.get("mean_physical_l2"),
                       "errors":z.get("physical_errors_250")}
                      for z in new if z["representation"]==rep and z["method"]==method]
            good=[z for z in rows if z["qualified"]]
            grouped[(rep,method)]=good
            summary.append({"representation":rep,"method":method,"n_runs":len(rows),
                "n_qualified":len(good),"n_valid_folds":len(set(z["fold"] for z in good)),
                "hit5":float(np.mean([z["hit5"] for z in good])) if good else None,
                "hit20":float(np.mean([z["hit20"] for z in good])) if good else None,
                "physical_mean_L2":float(np.mean([z["L2"] for z in good])) if good else None})
            for fold in range(1,6):
                r=[z for z in good if z["fold"]==fold]
                perfold.append({"representation":rep,"method":method,"fold":fold,
                    "n_qualified":len(r),"n_attempted":len([z for z in rows if z["fold"]==fold]),
                    "hit5":float(np.mean([z["hit5"] for z in r])) if r else None,
                    "hit20":float(np.mean([z["hit20"] for z in r])) if r else None,
                    "physical_L2":float(np.mean([z["L2"] for z in r])) if r else None})
            if good:
                errors=np.concatenate([np.asarray(z["errors"],float) for z in good])
                assert len(errors)==250*len(good)
                for eps in np.arange(0,.601,.01):
                    curve.append({"representation":rep,"method":method,
                                  "epsilon":round(float(eps),4),
                                  "hit":float(np.mean(errors<=eps))})
    csvout(a.out/"summary_SA_WA_DAD_HiWA.csv",summary)
    csvout(a.out/"all_fivefold_methods.csv",perfold)
    csvout(a.out/"dense_physical_tolerance_all_methods.csv",curve)
    assert all(z["n_qualified"]==z["n_runs"] for z in summary),summary

    all_scatter(a.out,new,a.frozen,a.features)
    graphs=a.out/"figures"
    for rep in REPS:
        subset=[z for z in summary if z["representation"]==rep]
        fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
        ax.barh(np.arange(len(subset)),[z["hit5"]*100 for z in subset])
        ax.set(yticks=np.arange(len(subset)),yticklabels=[z["method"] for z in subset],
               xlabel="True 5-finger Hit@train 5%-quantile (%)",
               title=f"{rep}: full1000 unpaired SA WA DAD-like versus HiWA")
        ax.invert_yaxis()
        fig.savefig(graphs/f"{rep}_Hit5_all_methods.png",dpi=170)
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
        for method in [z["method"] for z in subset]:
            part=[z for z in curve if z["representation"]==rep and z["method"]==method]
            ax.plot([z["epsilon"] for z in part],[100*z["hit"] for z in part],
                    label=method,lw=1.7)
        ax.axhline(70,ls=":",color="gray")
        ax.set(xlim=(0,.6),ylim=(0,100),
               xlabel="Raw 5-finger force L2 threshold (not Newton)",
               ylabel="Physical hit (%)",
               title=f"{rep}: matching-distance tolerance curves")
        ax.legend(fontsize=8);ax.grid(alpha=.14)
        fig.savefig(graphs/f"{rep}_eps_curves_all_methods.png",dpi=170)
        plt.close(fig)

    lines=["# HySeR：SA、WA、DAD-like 与 Hard/Soft HiWA 真实1000点实验",
      "","## 已实际运行的对照及限制",
      "正式对照以单一受试者 HySeR S01/S1 Random Task 五个试次做留一训练验证，四次训练试次各域1000点，剩余一次各域250测试点；源和目标无配对拟合，单点检索只访问1000目标训练点，原始五指真值只用于事后评分。",
      "",
      "- SA 是 DANDI 实现的全基 PCA Subspace Alignment 转写，每折确定性运行一次。",
      "- WA 为 DANDI 的退火 Sinkhorn/正交 Procrustes（200外迭代、温度100至0.1），在1000训练池采用250/轮随机小批量：每4轮完整覆盖1000点、重复50轮。这是1000池上的随机 WA 变体，并非原DANDI 160点全批次的逐步重现。",
      "- DAD-like 仅对3D实现 DANDI 的 k=5 对称密度KL代理、128网格加Powell局部优化，明确不是KordingLab原版DAD；不编造5D DAD。",
      "- Hard/Soft HiWA使用先前冻结的每折全1000训练点局部P/Q/Rkl模型。",
      "- 匹配成功是五指原始物理力量距离低于各外折训练内非对应力距离的5%或20%分位。Fisher二维只用于可视化，不影响评价。",
      "- 1000个点是每个域的真正训练点数；每张叠加图展示1000源点和1000目标点，不进行稀疏采样。",
      "",
      "## 全五折物理评价（越高越好；L2越低越好）",
      "",
      "|特征表示|方法|合格/运行|Hit@5%|Hit@20%|平均五指力量L2|",
      "|---|---|---:|---:|---:|---:|"]
    for z in summary:
        def fnum(x,fmt):return "NA" if x is None else format(x,fmt)
        lines.append(f"|{z['representation']}|{z['method']}|{z['n_qualified']}/{z['n_runs']}|"+
          f"{fnum(z['hit5'],'.2%')}|{fnum(z['hit20'],'.2%')}|{fnum(z['physical_mean_L2'],'.5f')}|")
    lines+=["","## 所有真实1000点Fisher散点图",
       "原始两域各自用训练组独立训练Fisher子空间；对齐后都投到目标域训练Fisher子空间。四种颜色代表源域独立训练的四个伪簇，灰色代表目标域背景点。颜色不意味着真实动作类别/跨域真配对。评分使用完整d维与五指原始力量。",
       ""]
    for rep in REPS:
        lines.append(f"### {rep}：五折每个方法全部1000点")
        for fold in range(1,6):
            fn=f"figures/{rep}_FOLD{fold}_ALL1000_Fisher_METHODS.png"
            lines.append(f"- [留出试次 {fold}：全部方法 Fisher 1000+1000 对照]({fn})")
        lines.append(f"- [该表示多方法严格Hit@5%图](figures/{rep}_Hit5_all_methods.png)")
        lines.append(f"- [该表示真实距离阈值曲线](figures/{rep}_eps_curves_all_methods.png)")
        lines.append("")
    lines+=["## 单独散点图（Fold1，显示全部1000个训练样本/域）",""]
    for rep in REPS:
        for method in METHODS:
            if method=="DAD_like_3D" and rep.endswith("RAW5"):continue
            fn=f"figures/individual_fold1/{rep}_FOLD1_{method.replace(' ','_')}_ALL1000_Fisher.png"
            lines.append(f"- [{rep} / {method} / 1000源+1000目标]({fn})")
    lines+=["","## 数值存档和严谨性",
        "全部新模型的正交R、每条测试样本真实五指误差、训练配置和数值状态都在下列JSON。上表按相同试次折比较，不能把同一被试的两种子当独立受试者；也不能将事后阈值70%作为预注册泛化准确率。",
        "",
        "- [所有SA/WA/DAD-like逐折逐种子原始数据](all_SA_WA_DAD_run_records.json)",
        "- [逐试次完整对照CSV](all_fivefold_methods.csv)",
        "- [汇总表CSV](summary_SA_WA_DAD_HiWA.csv)",
        "- [实际物理误差阈值曲线CSV](dense_physical_tolerance_all_methods.csv)",
        "- [1000点图像数目及K4颜色审计](scatter_count_audit.json)",
        "- [此前额外Identity/ICP/单层OT](../additional_baselines/ADDITIONAL_UNSUPERVISED_BASELINES_ZH.md)",
        "- [此前80次HiWA共享P/Q层级目标](../shared_hierarchical_full250/SHARED_HIERARCHICAL_EVALUATION_ZH.md)",
        "- [DANDI原迁移代码源分支](https://github.com/noelabernathy400-bot/ot-hiwa/blob/experiment/dandi-pca3-fa3-vs-pca4-fa4-20261119/projects/dandi-000138-hiwa-4d/code/closeout_dandi_sa_wa_dad_baselines.py)",
        "",
        "这是一组特定参数的实际基线复核，不是作者原SA/WA/DAD所有超参的最优性或统计跨人群显著性证明。"]
    (a.out/"SA_WA_DAD_HYSER_1000_FINAL_REPORT_ZH.md").write_text(
        "\n".join(lines)+"\n",encoding="utf8")
    print("HYSER_SAWAD_1000_DONE",len(new),sum(z["qualified"] for z in new),
          "n_figures",len(list(graphs.rglob("*.png"))),
          "SUMMARY",json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
