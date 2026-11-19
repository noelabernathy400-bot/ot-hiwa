#!/usr/bin/env python3
"""Common-five-physical-force error curves for 3D and 5D local HiWA models.

All returned records are from frozen original local P/Q/Rkl ADMM. This program
NEVER fits any OT model or tunes gamma/tau/K/mu.
Original FIVE physical finger force values define a common across-dimension
physical error. Source and target latent nearest-neighbor search is performed
in the model's full (d=3 or d=5) latent space, NEVER the visual Fisher plane.
Per fold, the physical tolerance is the 1/2.5/5/10/20% quantile of
UNRELATED cross-training-trial force vectors, computed solely on the 1000
training physical targets, with a fixed calibration seed.
"""
from __future__ import annotations
import argparse,json,csv,collections
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Q=(.01,.025,.05,.10,.20)
FINGERS=("thumb","index","middle","ring","little")
CAL_SEED=20261119
BOOT_SEED=20261120
N_PAIRS=24000
N_RANDOM=500

def quantiles_for_training(Y, trial_id):
    # Strictly samples from *different* training trials.
    rng=np.random.default_rng(CAL_SEED)
    parts=[]
    available=np.unique(trial_id)
    assert len(available)==4
    for t in available:
        assert np.sum(trial_id==t)==250
    for _ in range(N_PAIRS//4):
        a,b=rng.choice(available,size=2,replace=False)
        ia=rng.choice(np.flatnonzero(trial_id==a))
        ib=rng.choice(np.flatnonzero(trial_id==b))
        parts.append(np.linalg.norm(Y[ia]-Y[ib]))
    # Continuous random collection, length=6000; don't imply 24k draws.
    parts=np.asarray(parts)
    eps={str(q):float(np.quantile(parts,q)) for q in Q}
    return eps,{"random_cross_trial_training_pairs":len(parts),
                "train_distance_median":float(np.median(parts)),
                "train_distance_p90":float(np.quantile(parts,.9)),
                "units":"original five-channel force recording units",
                "calibration_seed":CAL_SEED}

def metrics(Ytrain_phys,Ytest_phys,Ytrain_latent,Xtest_latent,R,eps):
    Z=Xtest_latent@R.T
    d=cdist(Z,Ytrain_latent)
    idx=np.argmin(d,axis=1)
    nearest_d=d[np.arange(len(d)),idx]
    phys_pred=Ytrain_phys[idx]
    errors=np.linalg.norm(phys_pred-Ytest_phys,axis=1)
    rng=np.random.default_rng(BOOT_SEED)
    ri=rng.integers(0,len(Ytrain_phys),size=(N_RANDOM,len(Ytest_phys)))
    err_rand=np.linalg.norm(Ytrain_phys[ri]-Ytest_phys[None,:,:],axis=2)
    # True-target TRAIN-bank nearest is oracle achievable coverage only,
    # calculated AFTER fitting, not a supervised training signal.
    nearest_to_truth=cdist(Ytest_phys,Ytrain_phys)
    oracle_err=nearest_to_truth.min(axis=1)
    e5=eps[str(.05)]
    result={
      "sample_n":len(Ytest_phys),
      "physical_error_mean":float(errors.mean()),
      "physical_error_median":float(np.median(errors)),
      "physical_error_p90":float(np.quantile(errors,.9)),
      "physical_error_physical_RMSE_per_finger":np.sqrt(
        np.mean((phys_pred-Ytest_phys)**2,axis=0)).tolist(),
      "physical_error_root_mean_squared_5D_distance":float(np.sqrt(np.mean(errors**2))),
      "error_over_training_5pct_tolerance_mean":float((errors/e5).mean()) if e5>0 else None,
      "random_match_error_mean":float(err_rand.mean()),
      "random_match_error_median":float(np.median(err_rand)),
      "oracle_match_error_mean_EVAL_ONLY":float(oracle_err.mean()),
      "selected_latent_nearest_distance_mean":float(nearest_d.mean()),
      "selected_latent_nearest_distance_median":float(np.median(nearest_d)),
      "physical_hit":{str(q):float(np.mean(errors<=eps[str(q)])) for q in Q},
      "random_hit":{str(q):float(np.mean(err_rand<=eps[str(q)])) for q in Q},
      "oracle_hit_EVAL_ONLY":{str(q):float(np.mean(oracle_err<=eps[str(q)])) for q in Q},
      "random_replicates":N_RANDOM,
      "physical_errors_250":errors.tolist(),
      "candidate_bank_train_only":True,
      "used_true_time_pairs_only_after_freezing_R":True,
    }
    return result

def savecsv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader();writer.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features",type=Path,required=True)
    ap.add_argument("--frozen-runs",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--configs",nargs="+",required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    runs=[]
    for f in sorted(args.frozen_runs.rglob("local_ADMM_results.json")):
        runs.extend(json.loads(f.read_text())["runs"])
    assert len(runs)==len(args.configs)*5*2*2,(len(runs),args.configs)
    assert len({(r["config"],r["heldout_trial"],r["seed"],r["method"]) for r in runs})==len(runs)
    entries=[]
    for rep in args.configs:
        d=3 if "3" in rep else 5
        assert rep.startswith(("PCA","FA"))
        for fold in range(1,6):
            filename=f"trial_heldout_{fold}" + ("_5d" if d==5 else "")+".npz"
            with np.load(args.features/rep/filename) as z:
                Xtest=np.asarray(z["X_test"],float)/np.sqrt(d)
                Ytrain=np.asarray(z["Y_train"],float)/np.sqrt(d)
                Yphys_train=np.asarray(z["original_target_force_train"],float)
                Yphys_test=np.asarray(z["original_target_force_test"],float)
                tids=np.asarray(z["train_trial_id"],int)
                assert Xtest.shape==(250,d) and Ytrain.shape==(1000,d)
                assert Yphys_train.shape==(1000,5) and Yphys_test.shape==(250,5)
            eps,cal=quantiles_for_training(Yphys_train,tids)
            for seed in (42,43):
                for method in ("hard","soft05"):
                    r=next(r for r in runs if r["config"]==rep and
                           r["heldout_trial"]==fold and
                           r["seed"]==seed and r["method"]==method)
                    assert r["n_OT_train_source"]==1000 and r["n_OT_train_target"]==1000
                    if not (r.get("Q_constraints_passed") and
                            r.get("numeric_convergence_ADMM_primal_and_rotation") and
                            r["max_final_Q_marginal_error"]<1e-5):
                        # Record failed runs elsewhere, but do not silently compare.
                        raise RuntimeError(f"Unqualified local ADMM run: {rep}, {fold}, {seed}, {method}")
                    R=np.asarray(r["R"],float)
                    assert R.shape==(d,d) and np.linalg.norm(R.T@R-np.eye(d))<1e-7
                    m=metrics(Yphys_train,Yphys_test,Ytrain,Xtest,R,eps)
                    entries.append({"config":rep,"dim":d,"fold":fold,"seed":seed,
                        "method":method,"train_only_thresholds":eps,
                        "train_only_distance_reference":cal,"results":m})
    (args.out/"full_physical_distance_curves.json").write_text(
        json.dumps({"data":"real Hyser five trials, one subject / session",
            "metric":"5D physical FORCE difference, common units across 3D and 5D model representations",
            "no_trial_test_leak_in_threshold":True,
            "n_runs":len(entries),"q":list(Q),"records":entries},ensure_ascii=False,indent=2)+"\n")
    rows=[];paired=[]
    for entry in entries:
        rr=entry["results"]
        for q in Q:
            rows.append({"config":entry["config"],"dim":entry["dim"],"fold":entry["fold"],
                "seed":entry["seed"],"method":entry["method"],"q":q,
                "physical_tolerance_5finger_units":entry["train_only_thresholds"][str(q)],
                "hit_rate":rr["physical_hit"][str(q)],
                "random_baseline_hit_rate":rr["random_hit"][str(q)],
                "hit_minus_random":rr["physical_hit"][str(q)]-rr["random_hit"][str(q)],
                "oracle_hit_EVAL_ONLY":rr["oracle_hit_EVAL_ONLY"][str(q)],
                "physical_error_mean":rr["physical_error_mean"],
                "physical_error_median":rr["physical_error_median"],
                "physical_error_p90":rr["physical_error_p90"],
                "random_match_error_mean":rr["random_match_error_mean"],
                "normalized_error_5pct_mean":rr["error_over_training_5pct_tolerance_mean"]})
    savecsv(args.out/"all_fivefold_physical_hit_threshold_curve.csv",rows)
    report=["# 统一五指原始力量物理误差：3D/5D 同一量纲的多阈值检索评价","",
       "原三维/五维R、P/Q、ADMM训练模型已冻结；所有最近邻检索在它们各自完整d维潜在目标训练库中执行；最后利用**真实五根手指原始力量**比较被选中目标训练点和完整留出测试点。不是在Fisher二维图上算距离。",
       "阈值采用仅训练4试次、不同训练试次之间随机无关的五指物理力向量距离的1/2.5/5/10/20%分位；无外测试调参。阈值放宽提高Hit，但不表示对齐本身变好了；所以总是同时展示随机基线和5根手指物理误差。",
       "",
       "|表示|方法|Hit@1%|Hit@2.5%|Hit@5%|Hit@10%|Hit@20%|平均物理五指误差|随机平均物理误差|",
       "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    agg=[]
    for rep in args.configs:
        for method in ("hard","soft05"):
            rr=[x for x in entries if x["config"]==rep and x["method"]==method]
            meanhits={str(q):float(np.mean([x["results"]["physical_hit"][str(q)] for x in rr])) for q in Q}
            randomhits={str(q):float(np.mean([x["results"]["random_hit"][str(q)] for x in rr])) for q in Q}
            am=float(np.mean([x["results"]["physical_error_mean"] for x in rr]))
            rm=float(np.mean([x["results"]["random_match_error_mean"] for x in rr]))
            agg.append({"config":rep,"method":method,"n_seed_fold":len(rr),
                "mean_hits":meanhits,"mean_random_hits":randomhits,
                "mean_physical_error":am,"mean_random_error":rm,
                "mean_thresholds":{str(q):float(np.mean([x["train_only_thresholds"][str(q)] for x in rr])) for q in Q},
                "fivefold_independent_units":5})
            report.append("|{}|{}|{}|{:.5g}|{:.5g}|".format(rep,method,
                "|".join(f"{meanhits[str(q)]:.2%}" for q in Q),am,rm))
    for rep in args.configs:
        for fold in range(1,6):
            for seed in (42,43):
                a=next(x for x in entries if x["config"]==rep and x["fold"]==fold and x["seed"]==seed and x["method"]=="hard")
                b=next(x for x in entries if x["config"]==rep and x["fold"]==fold and x["seed"]==seed and x["method"]=="soft05")
                diff={"config":rep,"fold":fold,"seed":seed,
                      "soft_minus_hard_physical_error_mean":b["results"]["physical_error_mean"]-a["results"]["physical_error_mean"]}
                for q in Q:
                    diff["soft_minus_hard_Hit_"+str(q)]=b["results"]["physical_hit"][str(q)]-a["results"]["physical_hit"][str(q)]
                paired.append(diff)
    savecsv(args.out/"matched_soft_minus_hard_physical_curves.csv",paired)
    (args.out/"mean_physical_distance_curves.json").write_text(
        json.dumps({"means":agg,"n_runs":len(entries),"physical_errors_common_5D":True},ensure_ascii=False,indent=2)+"\n")
    # Plot physically commensurate thresholds, never arbitrary 2D coordinate plot.
    fig,axes=plt.subplots(1,len(args.configs),figsize=(6.2*len(args.configs),4.8),constrained_layout=True)
    if len(args.configs)==1:axes=[axes]
    for rep,ax in zip(args.configs,axes):
        rr=[z for z in agg if z["config"]==rep]
        for method,marker in (("hard","o"),("soft05","s")):
            z=next(z for z in rr if z["method"]==method)
            ax.plot([q*100 for q in Q],[100*z["mean_hits"][str(q)] for q in Q],
                    marker=marker,label=method)
        base=next(z for z in rr if z["method"]=="hard")
        ax.plot([q*100 for q in Q],[100*base["mean_random_hits"][str(q)] for q in Q],
                 ":",label="random target bank",color="#666")
        ax.set(xlabel="Train-only unrelated-force distance quantile (%)",
               ylabel="True heldout physical 5-finger neighborhood hit (%)",
               xticks=[q*100 for q in Q],title=rep)
        ax.grid(alpha=.16);ax.legend(frameon=False)
    fig.suptitle("Hyser physical force-neighbor retrieval; 5 folds x 2 matched seeds; same original 5-finger units")
    fig.savefig(args.out/"true_physical_matching_hit_vs_distance_tolerance.png",dpi=155,bbox_inches="tight")
    plt.close(fig)
    report+=["","**解释**：表中物理误差始终比较原始5根手指，所以3D/5D的误差单位一致；不过训练/测试选择和模型表示有区别，仍不能仅凭单一阈值下不同模型Hit百分比断言维度因果效应。",
      "原五指物理量可能有不同量级，欧氏范数相当于按原量纲加权；后续可加训练拟合的标准化五指误差作为敏感性分析，但不得为提高命中率事后更改定义。",
      "随机命中率与阈值一起上升，必须看Hit−Random或错误距离分布，不能通过加大阈值宣称模型更准确。",
      "两次种子同属一个留出试次，不能当作两个独立样本。",""]
    (args.out/"PHYSICAL_DISTANCE_CURVES_ZH.md").write_text("\n".join(report),encoding="utf-8")
    print(json.dumps({"success":True,"n":len(entries),"summary":agg},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
