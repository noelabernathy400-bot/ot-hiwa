#!/usr/bin/env python3
"""Postfit Hyser audit: Is heldout trial 5 harder due to physical target distribution shift?

No re-fitting, no model/threshold selection. Use true 5-finger forces ONLY after
the OT fits are frozen. Compute exhaustive target-bank oracle and random chance,
and compare descriptive heldout per-trial actual prediction errors.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from dense_70pct_force_threshold_audit import training_null_distances

def mean(xs):
    return float(np.mean(xs))

def savecsv(p,rows):
    with p.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features",type=Path,required=True)
    ap.add_argument("--results-root",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    recs=[]
    for d in ["3D","5D"]:
        obj=json.loads((a.results_root/d/"full_physical_distance_curves.json").read_text())
        assert obj["n_runs"]==40
        recs+=obj["records"]
    rows=[]
    for fold in range(1,6):
        with np.load(a.features/"PCA5_RAW5"/f"trial_heldout_{fold}_5d.npz") as z:
            train=z["original_target_force_train"].astype(float)
            test=z["original_target_force_test"].astype(float)
            ids=z["train_trial_id"].astype(int)
        assert train.shape==(1000,5) and test.shape==(250,5)
        dist=cdist(test,train)
        oracle=dist.min(axis=1)
        chance=dist.ravel()
        cal=training_null_distances(train,ids)
        eps5=float(np.quantile(cal,.05))
        eps50=float(np.quantile(cal,.5))
        tr_std=np.std(train,axis=0,ddof=0)
        mu_shift=float(np.linalg.norm((np.mean(test,axis=0)-np.mean(train,axis=0))/np.maximum(tr_std,1e-8)))
        rms_tr=float(np.sqrt(np.mean(train**2)))
        rms_te=float(np.sqrt(np.mean(test**2)))
        out_of_train_range=float(np.mean(np.any((test<train.min(axis=0)) | (test>train.max(axis=0)),axis=1)))
        row={"fold":fold,"n_train":1000,"n_test":250,"train_q05_epsilon":eps5,
             "train_median_unrelated_epsilon":eps50,
             "mean_finger_standardized_shift":mu_shift,
             "train_finger_rms":rms_tr,"test_finger_rms":rms_te,
             "test_over_train_force_rms":rms_te/max(rms_tr,1e-12),
             "test_outside_any_train_finger_range_fraction":out_of_train_range,
             "oracle_train_bank_hit_at_epsilon_0_325":mean(oracle<=.325),
             "random_train_bank_hit_at_epsilon_0_325":mean(chance<=.325),
             "oracle_train_bank_hit_at_train_q05":mean(oracle<=eps5),
             "random_train_bank_hit_at_train_q05":mean(chance<=eps5),
             "oracle_minimum_mean_distance":mean(oracle)}
        for rep in ("PCA3_PCA3","PCA5_RAW5"):
            for method in ("hard","soft05"):
                matches=[r for r in recs if r["config"]==rep and r["method"]==method and r["fold"]==fold]
                assert len(matches)==2 and {r["seed"] for r in matches}=={42,43}
                e=np.concatenate([np.asarray(r["results"]["physical_errors_250"],float) for r in matches])
                k=rep.split("_")[0]+("_H" if method=="hard" else "_S")
                row[k+"_hit_at_eps_0_325"]=mean(e<=.325)
                row[k+"_hit_at_train_q05"]=mean(e<=eps5)
                row[k+"_mean_force_error"]=mean(e)
        rows.append(row)
    savecsv(a.out/"physical_bank_drift_oracle_random_by_fold.csv",rows)
    fig,ax=plt.subplots(figsize=(9.5,5.4),constrained_layout=True)
    labs=[str(x["fold"]) for x in rows]
    x=np.arange(5);w=.19
    for offset,key,label in [(-1.5,"PCA3_H_hit_at_eps_0_325","PCA3 Hard"),
                              (-.5,"PCA5_H_hit_at_eps_0_325","PCA5 Hard"),
                              (.5,"random_train_bank_hit_at_epsilon_0_325","Random-bank"),
                              (1.5,"oracle_train_bank_hit_at_epsilon_0_325","Oracle-bank")]:
        ax.bar(x+offset*w,[100*r[key] for r in rows],w,label=label)
    ax.axhline(70,color="gray",ls=":")
    ax.set(xlabel="Completely held-out trial (one subject/session)",ylabel="Physical force hit at ε=0.325 (%)",
           title="Hyser: fold instability vs random and oracle target-bank baselines",
           xticks=x,xticklabels=labs,ylim=(0,105))
    ax.legend(fontsize=9)
    fig.savefig(a.out/"trial5_bank_oracle_random_control.png",dpi=185)
    plt.close(fig)
    out=["# Hyser 原始五指力量留出试次诊断：试次5困难是否源于候选库覆盖不足？",
         "",
         "冻结的HiWA模型仅作事后检验。oracle是已知真实测试力量后在训练目标库中找到的物理最近邻；只代表训练库的理想上限，**不是**无监督模型或合理的预测方法。",
         "",
         "随机是完整250×1000测试—训练目标力量组合的期望，不使用模型，不是 Monte-Carlo 估计。",
         "",
         "|留出试次|PCA3 Hard Hit@ε=.325|PCA5 Hard Hit@ε=.325|随机Hit|oracle Hit|oracle平均最小误差|训练内五指中位距离|力量RMS test/train|",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        out.append(f"|{r['fold']}|{r['PCA3_H_hit_at_eps_0_325']:.2%}|{r['PCA5_H_hit_at_eps_0_325']:.2%}|{r['random_train_bank_hit_at_epsilon_0_325']:.2%}|{r['oracle_train_bank_hit_at_epsilon_0_325']:.2%}|{r['oracle_minimum_mean_distance']:.5f}|{r['train_median_unrelated_epsilon']:.5f}|{r['test_over_train_force_rms']:.3f}|")
    out +=["","![各试次随机和候选库oracle对比](trial5_bank_oracle_random_control.png)","",
          "限制：测试试次与模型同属于同一被试/会话，不能将试次5差异外推到其他被试；原始力量 RMS 和均值漂移是描述性证据，不能单独证明造成模型失配的因果机制。",
          "若试次5的oracle仍接近100%而实际模型显著下降，表明不是训练目标库缺乏邻域，后续应重点检查跨模态映射、试次间EMG表示漂移、局部旋转与运动时滞。"]
    (a.out/"TRIAL5_PHYSICAL_BANK_DIAGNOSTIC_ZH.md").write_text("\n".join(out)+"\n",encoding="utf-8")
    assert len(rows)==5
    print(json.dumps({"done":True,"rows":rows},ensure_ascii=False))

if __name__=="__main__":
    main()
