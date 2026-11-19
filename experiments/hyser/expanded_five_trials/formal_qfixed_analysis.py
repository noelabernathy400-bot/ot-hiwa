#!/usr/bin/env python3
"""Audit 40 genuine Hyser K4 Hard/Soft PQR ADMM runs; Fisher-only plots.

No model fitted and no test-selected hyperparameters. Fold is statistical unit.
"""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from best_fisher_2d import fisher2

KEYS={
"W2":"heldout_W2_squared",
"NN_train":"train_bank_inductive_nearest.train_bank_NN_physically_correct",
"NN_random":"train_bank_inductive_nearest.train_bank_random_hit_mean",
"NN_test":"nearest.nn_retrieval_physical_neighbor_fraction",
"coverage":"train_bank_inductive_nearest.train_bank_NN_geometric_coverage",
"RMSE":"heldout_physical_force_prediction_POSTHOC.paired_force_RMSE_EVAL_ONLY",
"R2":"heldout_physical_force_prediction_POSTHOC.paired_force_R2_vs_trainmean_EVAL_ONLY",
}
REPS=("PCA5_RAW5","FA5_RAW5")
def get(x,k):
    for q in k.split("."):x=x[q]
    return float(x)
def csvsave(path,rows):
    if not rows:return
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
def fisher_figures(out,features,runs):
    # Predeclared fold 1 and seed42, never select "best" successful fold.
    for rep in REPS:
        with np.load(features/rep/"trial_heldout_1_5d.npz") as f:
            xtr=np.asarray(f["X_train"],float)/np.sqrt(5)
            ytr=np.asarray(f["Y_train"],float)/np.sqrt(5)
            xe=np.asarray(f["X_test"],float)/np.sqrt(5)
            ye=np.asarray(f["Y_test"],float)/np.sqrt(5)
        kmx=KMeans(4,n_init=30,random_state=42).fit(xtr)
        kmy=KMeans(4,n_init=30,random_state=43).fit(ytr)
        fx=fisher2(xtr,kmx.labels_)
        fy=fisher2(ytr,kmy.labels_)
        source=(xe-fx["mean"])@fx["basis"]
        target=(ye-fy["mean"])@fy["basis"]
        fig,axes=plt.subplots(1,3,figsize=(15.2,5),constrained_layout=True)
        colors=["#2E78A0","#D8884B","#528B71","#9263A9"]
        cx=kmx.predict(xe);cy=kmy.predict(ye)
        for k,c in enumerate(colors):
            axes[0].scatter(*source[cx==k].T,c=c,s=12,alpha=.62,rasterized=True)
            axes[1].scatter(*target[cy==k].T,c=c,s=12,alpha=.62,rasterized=True)
        axes[2].scatter(*target.T,c="#333",s=9,alpha=.21,label="Target",rasterized=True)
        for meth,color in (("hard","#C3504D"),("soft05","#3262A2")):
            r=next(r for r in runs if r["config"]==rep and
                   r["heldout_trial"]==1 and r["seed"]==42 and r["method"]==meth)
            z=(xe@np.asarray(r["R"]).T-fy["mean"])@fy["basis"]
            axes[2].scatter(*z.T,c=color,s=8,alpha=.3,label=meth,rasterized=True)
        for ax,title in zip(axes,["Source: source-trained Fisher","Target: target-trained Fisher",
                                   "Aligned source + target: TARGET Fisher"]):
            ax.set(title=title,xlabel="Fisher axis 1",ylabel="Fisher axis 2")
            ax.grid(alpha=.12)
        axes[2].legend(fontsize=9)
        fig.suptitle(rep+" | Full1000 local P/Q/R ADMM | trial1 seed42, train-only Fisher")
        fig.savefig(out/(rep+"_formal_ADMM_Fisher2D.png"),dpi=180,bbox_inches="tight")
        plt.close(fig)
        (out/(rep+"_Fisher_provenance.json")).write_text(
          json.dumps({"train_selected_2D_Fisher_only":True,
            "representation":rep,"fold":1,"seed":42,
            "src_train_silhouette_2D":fx["train2d_silhouette"],
            "target_train_silhouette_2D":fy["train2d_silhouette"],
            "trained_with_Kmeans_pseudo_labels_not_hand_movement_truth":True,
            "source_basis":fx["basis"].tolist(),"target_basis":fy["basis"].tolist(),
            "all_metrics_evaluated_in_full_5D":True},ensure_ascii=False,indent=2)+"\n")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--runs-dir",type=Path,required=True)
    p.add_argument("--features",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    z=p.parse_args();z.out.mkdir(exist_ok=True,parents=True)
    runs=[]
    for f in sorted(z.runs_dir.rglob("local_ADMM_results.json")):
        runs+=json.loads(f.read_text())["runs"]
    assert len(runs)==40
    index={(r["config"],r["heldout_trial"],r["seed"],r["method"]):r for r in runs}
    assert len(index)==40
    for r in runs:
        assert r["n_OT_train_source"]==r["n_OT_train_target"]==1000
        assert r["Q_constraints_passed"] and r["numeric_convergence_ADMM_primal_and_rotation"]
        assert r["max_final_Q_marginal_error"]<=1e-5
        assert r["max_Rkl_orthogonality_error"]<1e-7
        assert r["params"]["mu"]==.05 and r["params"]["support_mode"]=="positive"
        assert r["stable_conditional_Q_solver_if_needed"]
    pair=[];folds=[];overall=[]
    for rep in REPS:
        for fold in range(1,6):
            for seed in (42,43):
                a=index[(rep,fold,seed,"hard")]
                b=index[(rep,fold,seed,"soft05")]
                row={"representation":rep,"fold":fold,"seed":seed}
                for k,path in KEYS.items():
                    row["Hard_"+k]=get(a,path)
                    row["Soft_"+k]=get(b,path)
                    row["delta_"+k]=row["Soft_"+k]-row["Hard_"+k]
                pair.append(row)
            rr=[r for r in pair if r["representation"]==rep and r["fold"]==fold]
            row={"representation":rep,"fold":fold}
            for name in KEYS:
                for pre in ("Hard_","Soft_","delta_"):
                    row[pre+name]=float(np.mean([r[pre+name] for r in rr]))
            folds.append(row)
        rows=[r for r in folds if r["representation"]==rep]
        result={"representation":rep,"n_folds":5,"n_paired_seeds":10}
        for name in KEYS:
            dif=np.array([r["delta_"+name] for r in rows])
            result["delta_"+name]=float(np.mean(dif))
            result["delta_"+name+"_sd_across_folds"]=float(np.std(dif,ddof=1))
            result["folds_soft_better_"+name]=int(np.sum(dif>0 if name in ("NN_train","NN_test","R2") else dif<0))
            result["Hard_"+name]=float(np.mean([r["Hard_"+name] for r in rows]))
            result["Soft_"+name]=float(np.mean([r["Soft_"+name] for r in rows]))
        overall.append(result)
    csvsave(z.out/"ten_paired_seed_metrics.csv",pair)
    csvsave(z.out/"five_fold_averages.csv",folds)
    (z.out/"fold_summary.json").write_text(json.dumps({
        "all40_qualified":True,"fivefold_two_seeds_each":True,
        "physical_NN_threshold_training_only":True,
        "source":"actual Hyser S01 S1 five trials",
        "results":overall},ensure_ascii=False,indent=2)+"\n")
    for k in ("W2","NN_train","RMSE"):
        fig,axes=plt.subplots(1,2,figsize=(12.2,4.3),constrained_layout=True)
        for ax,rep in zip(axes,REPS):
            ff=[r for r in folds if r["representation"]==rep]
            for typ in ("Hard","Soft"):
                ax.plot(range(1,6),[r[typ+"_"+k] for r in ff],"-o",label=typ)
            if k=="NN_train":
                ax.plot(range(1,6),[r["Hard_NN_random"] for r in ff],
                        ":",color="#777",label="Random bank")
            ax.set(title=rep,xlabel="Heldout trial",ylabel=k,xticks=[1,2,3,4,5])
            ax.grid(alpha=.15);ax.legend(fontsize=9)
        fig.suptitle("Two fixed seeds averaged in each of five heldout trials")
        fig.savefig(z.out/("fivefold_"+k+".png"),dpi=175,bbox_inches="tight")
        plt.close(fig)
    fisher_figures(z.out,z.features,runs)
    md=[
    "# Hyser 完整五折 Hard/Soft 局部 HiWA 正式结果",
    "",
    "全部 40 次运行的 P/Q 边缘约束、局部旋转正交性和预先确定的 ADMM 收敛条件均已通过。",
    "五个真实试次各一次留出（每折源域和目标域分别训练1000点、留出250点），两随机种子，PCA5或FA5肌电以及完整五指力量5D。无监督训练不使用时间配对。",
    "四簇K=4，Soft温度0.5，簇级gamma1=0.2，簇内gamma2=0.1/P，ADMM数值罚参数mu=0.05，最多300外层迭代。严格零质量支持处理+必要时对数域Sinkhorn，均未改变原有熵正则运输目标。",
    "",
    "## 正式五折均值",
    "",
    "|特征|分组|完整5D W2²↓|训练目标库真实力量最近邻↑|随机训练目标库|测试目标库真实邻域↑|五指力量RMSE↓|力量预测R²↑|",
    "|---|---|---:|---:|---:|---:|---:|---:|"
    ]
    for row in overall:
        for typ in ("Hard","Soft"):
            md.append("|{}|{}|{:.5f}|{:.2%}|{:.2%}|{:.2%}|{:.5f}|{:.4f}|".format(
                row["representation"],typ,row[typ+"_W2"],
                row[typ+"_NN_train"],row[typ+"_NN_random"],
                row[typ+"_NN_test"],row[typ+"_RMSE"],row[typ+"_R2"]))
    md+=["","## 配对效应：Soft − Hard，五个留出试次各先平均两种子",
      "|特征|Δ W2²↓|Δ训练库正确率(百分点)↑|Δ RMSE↓|最近邻Soft改善的折数|W2² Soft改善的折数|",
      "|---|---:|---:|---:|---:|---:|"]
    for row in overall:
        md.append("|{}|{:+.5f}|{:+.2f}|{:+.5f}|{}/5|{}/5|".format(
            row["representation"],row["delta_W2"],100*row["delta_NN_train"],
            row["delta_RMSE"],row["folds_soft_better_NN_train"],
            row["folds_soft_better_W2"]))
    md+=["",
    "## 科学解释与局限",
    "- 40次真正收敛的局部P/Q/Rkl实验没有支持Soft稳定优于Hard。不可沿用此前96锚点全局旋转模型的正向结论。",
    "- 最近邻以目标训练库为候选，选取距离最小点后，才用真实留出目标力量判定命中。阈值来自训练试次之间不对应样本距离的第5百分位数。几何覆盖率不等于预测正确率。",
    "- 尽管最近邻命中率高于随机基线，整体五指力量R²仍为负，说明仍未达到有用的连续力量预测。",
    "- PCA与FA是不同潜在坐标几何：不能以两种表示的绝对W2²直接宣称FA优于PCA。可分别比较同表示内部Hard和Soft。",
    "- ADMM误差阈值0.1不保证非凸问题的全局最优；0.05是为了数值收敛改变的罚参数，不是原文默认值0.005。",
    "- 五折来自同一受试者同一会话，不能当作五个人的独立重复。",
    "- 所有散点图仅使用TRAIN伪簇标签的Fisher判别二维子空间，训练和定量评价在完整五维中完成。",
    "",
    "数据：ten_paired_seed_metrics.csv、five_fold_averages.csv、fold_summary.json。Fisher绘图两张、五折指标图三张。"
    ]
    (z.out/"FINAL_FULL1000_FISHER_ONLY_HARD_SOFT_ZH.md").write_text("\n".join(md)+"\n")
    print(json.dumps({"success":True,"all40_qualified":True,"summary":overall},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
