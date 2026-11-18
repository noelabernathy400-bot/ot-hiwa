#!/usr/bin/env python3
"""Real five-trial Hyser Soft-GCOT prototype grouping control.

Numerical model = same weighted-anchor GLOBAL-R hierarchical P/Q pilot as the
previous Hard and Soft KMeans run. Grouping = independent, genuinely learned
soft prototypes (GCOT original prototype approach) with entropy parameter 1.
No synchronized pairing in training; the shared holdout reference is identical.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"scripts"))
from run_soft_inherited import prototypes
from run_hard_soft_pilot import quantize_group_measure,fit_orthogonal_hierarchical,test_shared_reference,w2,DIV

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--baseline-results",type=Path,required=True)
    p.add_argument("--seeds",type=int,nargs="+",default=[42,43])
    p.add_argument("--folds",type=int,nargs="+",default=[1,2,3,4,5])
    p.add_argument("--anchors",type=int,default=96)
    p.add_argument("--outer",type=int,default=45)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    prior=json.loads(a.baseline_results.read_text())["runs"]
    previous={(r["holdout_fold"],r["seed"],r["method"]):r for r in prior}
    for fold in a.folds:
        for seed in a.seeds:
            assert (fold,seed,"hard") in previous
    runs=[]
    for fold in a.folds:
        with np.load(a.data_dir/f"trial_heldout_{fold}.npz") as z:
            Xtr=np.asarray(z["X_train"],float)/DIV;Ytr=np.asarray(z["Y_train"],float)/DIV
            Xte=np.asarray(z["X_test"],float)/DIV;Yte=np.asarray(z["Y_test"],float)/DIV
        assert Xtr.shape==(1000,3) and Ytr.shape==(1000,3)
        cx=KMeans(n_clusters=4,n_init=30,random_state=42).fit(Xtr).cluster_centers_
        cy=KMeans(n_clusters=4,n_init=30,random_state=43).fit(Ytr).cluster_centers_
        for seed in a.seeds:
            A,metaX=prototypes(Xtr,groups=4,seed=seed,entropy=1.0)
            B,metaY=prototypes(Ytr,groups=4,seed=seed+1,entropy=1.0)
            Xa,gx,xd=quantize_group_measure(Xtr,A,a.anchors,seed+101)
            Ya,gy,yd=quantize_group_measure(Ytr,B,a.anchors,seed+202)
            R,P,train,history=fit_orthogonal_hierarchical(Xa,Ya,gx,gy,seed,max_iter=a.outer)
            held=test_shared_reference(Xtr,Ytr,Xte,Yte,R,cx,cy)
            rec={
                "holdout_fold":fold,"seed":seed,"method":"soft_GCOT_prototypes",
                "prototype_entropy":1.0,
                "prototype_metadata_X":metaX,
                "prototype_metadata_Y":metaY,
                "hard_reference_same_as_control":True,
                "n_train_per_domain":1000,"n_test_per_domain":250,
                "n_weighted_anchors":a.anchors,
                "quantization_error_X":xd,"quantization_error_Y":yd,
                "ambiguous_max_lt_0p8_X":float(np.mean(A.max(1)<.8)),
                "ambiguous_max_lt_0p8_Y":float(np.mean(B.max(1)<.8)),
                "R":R.tolist(),"P":P.tolist(),"train":train,
                "heldout_common_hard_K4_reference":held,
                "test_W2_identity":w2(Xte,Yte,np.eye(3)),
                "test_W2_after":w2(Xte,Yte,R),
                "pair_synchrony_never_used_in_training":True,
            }
            runs.append(rec)
            (a.out/"partial_results.json").write_text(
                json.dumps({"status":"PARTIAL","method":"soft_GCOT_prototypes","runs":runs},
                           ensure_ascii=False,indent=2)+"\n")
            print("DONE",fold,seed,"prototype X/Y optimizer success",
                  metaX["optimizer_success"],metaY["optimizer_success"],
                  "sharedT",held["after"]["T"],"W2",rec["test_W2_after"],flush=True)
    vals=[]
    for method in ("hard","soft05","soft_GCOT_prototypes"):
        group=( [r for r in prior if r["method"]==method] if method!="soft_GCOT_prototypes" else runs )
        vals.append({"method":method,
          "n":len(group),
          "heldout_shared_T":float(np.mean([r["heldout_common_hard_K4_reference"]["after"]["T"] for r in group])),
          "heldout_shared_J":float(np.mean([r["heldout_common_hard_K4_reference"]["after"]["J"] for r in group])),
          "heldout_W2_sq":float(np.mean([r["test_W2_after"] for r in group])),
          "numeric_converged":sum(r["train"]["simple_numeric_convergence"] for r in group),
        })
    (a.out/"results_all.json").write_text(
        json.dumps({"method":"soft_GCOT_prototypes","baseline_methods":["hard","soft05"],
                    "solver_is_original_HiWA_ADMM":False,
                    "common_optimizer":"same weighted-anchor GLOBAL-R constrained hierarchical OT",
                    "trained_on_unsupervised_independent_domains":True,
                    "runs":runs,"combined_comparison":vals},ensure_ascii=False,indent=2)+"\n")
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,(metric,extract) in zip(axs,[
        ("Heldout shared hierarchical T",lambda r:r["heldout_common_hard_K4_reference"]["after"]["T"]),
        ("Heldout global W2 squared",lambda r:r["test_W2_after"])]):
        for method,label in [("hard","Hard"),("soft05","Soft KMeans tau=.5"),
                             ("soft_GCOT_prototypes","Soft GCOT prototype")]:
            group=([r for r in prior if r["method"]==method] if method!="soft_GCOT_prototypes" else runs)
            values=[np.mean([extract(r) for r in group if r["holdout_fold"]==fold]) for fold in a.folds]
            ax.plot(a.folds,values,marker="o",label=label)
        ax.set(xlabel="heldout trial",xticks=a.folds,ylabel="Lower is better",title=metric)
        ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
    fig.suptitle("Hyser: 3-way independent grouping control, same weighted-anchor solver")
    fig.savefig(a.out/"three_way_heldout_comparison.png",dpi=175,bbox_inches="tight")
    plt.close(fig)
    lines=["# Hyser 1250点：Soft-GCOT原型方法 vs Hard/Soft K-means",
           "", "**严格说明：三者均使用相同的加权96锚点全局R双层OT近似求解器，不是原版HiWA局部R ADMM。**",
           "全部1000个训练点参与分组和锚点估计，最终P/Q在96个锚点上优化。",
           "原型软分组使用GCOT Eq.(2) 的prototypes优化，两个域严格独立、原型熵权重固定1；P自由学习；不使用同步对应或动作代理标签。",
           "测试集同一Hard KMeans四簇参考重新求P/Q；整体W2²不使用真实对应。",
           "","| 方法 | 留出层级T ↓ | 留出完整J ↓ | 留出整体W2² ↓ | 数值收敛次数/10 |",
           "|---|---:|---:|---:|---:|"]
    for r in vals:
        lines.append(f"| {r['method']} | {r['heldout_shared_T']:.6f} | {r['heldout_shared_J']:.6f}"
                     f" | {r['heldout_W2_sq']:.6f} | {r['numeric_converged']}/{r['n']} |")
    lines+=["","**解释边界**：Hard与Soft自身原生测度不同，故上表仅比较固定KMeans参考下已学得R的质量；不构成软分组原生层级目标的直接绝对比较。",
            "原型优化器收敛标志记录在JSON；如未通过绝不能以其结果做成功断言。","五折为同一被试同一天五次试次，两个种子不是独立被试；结论仅代表探索性实证。"]
    (a.out/"REPORT_ZH.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"n_runs":len(runs),"results":vals},ensure_ascii=False),flush=True)

if __name__=="__main__": main()
