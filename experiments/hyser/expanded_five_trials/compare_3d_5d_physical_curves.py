#!/usr/bin/env python3
"""Compare 3D and 5D heldout physical-force NN on common 5-finger raw units.

Predeclared 5 folds x 2 seeds; compare within PCA or FA, matched method,
fold and seed, same original 5-finger target physical tolerance. No refitting
or threshold selection. Do NOT compare latent W2 across dimensions.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Q=(.01,.025,.05,.10,.20)
MAPPING={"PCA":("PCA3_PCA3","PCA5_RAW5"),"FA":("FA3_FA3","FA5_RAW5")}
def savecsv(p,rr):
    if not rr:return
    with p.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]))
        w.writeheader();w.writerows(rr)
def load(root,dim):
    f=root/dim/"full_physical_distance_curves.json"
    o=json.loads(f.read_text())
    assert o["n_runs"]==40
    return {(r["config"],r["fold"],r["seed"],r["method"]):r
            for r in o["records"]}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results-root",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    m3=load(a.results_root,"3D");m5=load(a.results_root,"5D")
    rows=[];summary=[]
    for kind,(c3,c5) in MAPPING.items():
        for method in ("hard","soft05"):
            for fold in range(1,6):
                for seed in (42,43):
                    a3=m3[c3,fold,seed,method];a5=m5[c5,fold,seed,method]
                    assert a3["train_only_thresholds"]==a5["train_only_thresholds"]
                    d3=a3["results"];d5=a5["results"]
                    for q in Q:
                        rows.append({
                            "representation_family":kind,"method":method,
                            "fold":fold,"seed":seed,"q":q,
                            "physical_tolerance_original_fivefinger":a3["train_only_thresholds"][str(q)],
                            "hit_3D":d3["physical_hit"][str(q)],
                            "hit_5D":d5["physical_hit"][str(q)],
                            "delta_hit_5D_minus_3D":d5["physical_hit"][str(q)]-d3["physical_hit"][str(q)],
                            "random_hit":d3["random_hit"][str(q)],
                            "mean_physical_error_3D":d3["physical_error_mean"],
                            "mean_physical_error_5D":d5["physical_error_mean"],
                            "delta_physical_error_5D_minus_3D":d5["physical_error_mean"]-d3["physical_error_mean"],
                        })
            at=[r for r in rows if r["representation_family"]==kind and r["method"]==method]
            fold5=[float(np.mean([r["delta_hit_5D_minus_3D"] for r in at if r["q"]==.05 and r["fold"]==fold])) for fold in range(1,6)]
            foldErr=[float(np.mean([r["delta_physical_error_5D_minus_3D"] for r in at if r["q"]==.05 and r["fold"]==fold])) for fold in range(1,6)]
            summary.append({"representation_family":kind,"method":method,
                "delta_hit_5pct_5D_minus_3D":float(np.mean(fold5)),
                "delta_error_5D_minus_3D":float(np.mean(foldErr)),
                "folds_hit_improved_in_5D":int(sum(x>0 for x in fold5)),
                "folds_error_improved_in_5D":int(sum(x<0 for x in foldErr)),
                "mean_hit3":float(np.mean([r["hit_3D"] for r in at if r["q"]==.05])),
                "mean_hit5":float(np.mean([r["hit_5D"] for r in at if r["q"]==.05])),
                "fold_delta_Hit5":fold5,"fold_delta_error":foldErr})
    savecsv(a.out/"matched_3D_vs_5D_fivefinger_physical_distances.csv",rows)
    (a.out/"matched_dim_effects.json").write_text(json.dumps({
      "comparison":"only physical five-finger force error at shared raw-unit training quantile eps",
      "no_cross_latent_W2_comparison":True,"five_folds_not_independent_subjects":True,
      "summary":summary},ensure_ascii=False,indent=2)+"\n")
    fig,axes=plt.subplots(1,2,figsize=(13.5,5),constrained_layout=True)
    for ax,(family,(c3,c5)) in zip(axes,MAPPING.items()):
        for dim,config,mm in ((3,c3,m3),(5,c5,m5)):
            for method,marker in (("hard","o"),("soft05","s")):
                xx=[];yy=[]
                for q in Q:
                    entries=[v for k,v in mm.items() if k[0]==config and k[3]==method]
                    xx.append(q*100)
                    yy.append(100*np.mean([v["results"]["physical_hit"][str(q)] for v in entries]))
                ax.plot(xx,yy,marker=marker,label=f"{dim}D {method}")
        z=next(v for k,v in m3.items() if k[0]==c3)
        baseline=[100*np.mean([v["results"]["random_hit"][str(q)] for k,v in m3.items() if k[0]==c3 and k[3]=="hard"]) for q in Q]
        ax.plot([100*q for q in Q],baseline,":",color="#777",label="random training target")
        ax.set(title=family+" source representation",xlabel="Physical train-only distance percentile (%)",
               ylabel="True force-neighbor hit rate (%)",xticks=[1,2.5,5,10,20])
        ax.grid(alpha=.15);ax.legend(fontsize=8,frameon=False)
    fig.suptitle("Real Hyser 3D vs 5D: same original 5-finger physical correctness units, 5 heldout trials")
    fig.savefig(a.out/"combined_3D_5D_physical_hit_curves.png",dpi=170,bbox_inches="tight")
    plt.close(fig)
    md=["# Hyser：真正的三维 / 五维统一物理指标对照","",
      "均使用五根手指原始记录单位评估，而非比较不可比的3D和5D潜空间W2。全部模型的外层五折+两种子已提前冻结，物理阈值由各折训练四个试次的不同试次无关力量样本决定。",
      "最近邻在完整d维潜变量训练目标库中搜索，结果再映射到其所对应真实5指物理力，与真实同步留出力量比较；真实同步标签不会回流至训练或阈值选择。",
      "## 可直接比较的统一阈值",
      "随机基线训练库Hit@5%约5.245%；平均训练物理力距离的5%分位阈值约0.05915（原始记录单位）。增加阈值会同时提高模型与随机的命中率。",
      "",
      "|特征家族|方法|3D Hit@5%|5D Hit@5%|5D−3D（百分点）|五维有改善的试次数|5D−3D平均物理误差|",
      "|---|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        md.append("|{}|{}|{:.2%}|{:.2%}|{:+.2f}|{}/5|{:+.5f}|".format(
           r["representation_family"],r["method"],r["mean_hit3"],
           r["mean_hit5"],100*r["delta_hit_5pct_5D_minus_3D"],
           r["folds_hit_improved_in_5D"],r["delta_error_5D_minus_3D"]))
    md+=["","**不应误读**：5D看似保留更多源信息/目标物理通道，仍未保证HiWA可更准确地检索单样本。某些5D与3D的差异来自目标表示，需进一步分离源与目标降维方法。",
         "这份比较仅为已有固定方法的外部评价，未根据测试集选择新参数。",""]
    (a.out/"COMBINED_3D_5D_PHYSICAL_REPORT_ZH.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    print(json.dumps({"success":True,"summary":summary},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
