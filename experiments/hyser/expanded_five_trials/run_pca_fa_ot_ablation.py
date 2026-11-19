#!/usr/bin/env python3
"""Hyser PCA/FA factorial OT PILOT, K=4; six representations, all heldout.

This is the SAME 96 weighted-anchor, global-orthogonal-R, P/Q hierarchical OT
numerical pilot as the existing 3D/5D Hard/Soft experiment. NOT the original
HiWA local-Rij ADMM.

X and Y train/test embeddings are already fit on 4 TRAINING trials. No
cross-domain synchronized pairs or hand-force labels enter OT training.

Original five force vectors are used POSTHOC for a physically meaningful
common evaluation: learn a force decoder from *target-domain training
embeddings only* (Y embedding -> its original five force measurements), then
decode R(Xtest). All paired-target comparisons occur AFTER R,P,Q are fitted.
The decoder is never used to choose an OT model, hyperparameter, or seed.
"""
from __future__ import annotations
import argparse,csv,json,sys,math
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_hard_soft_pilot as s3
import run_fivefinger_5d_K4_K5 as s5

CONFS=[
    ("PCA3_PCA3",3),
    ("FA3_PCA3",3),
    ("PCA3_FA3",3),
    ("FA3_FA3",3),
    ("PCA5_RAW5",5),
    ("FA5_RAW5",5),
]
FINGERS=("thumb","index","middle","ring","little")
SEEDS=(42,43)
FOLDS=(1,2,3,4,5)
K=4
TEMP=.5

def stats(arr):
    vals=np.asarray(arr,float)
    good=vals[np.isfinite(vals)]
    return {"n":int(len(vals)),"n_defined":int(len(good)),
            "mean":float(np.mean(good)) if len(good) else None,
            "std":float(np.std(good,ddof=1)) if len(good)>1 else None}

def ref_metric(rec,name):
    return rec.get("after",{}).get(name,None)

def physical_eval(xt, yt, original_Y_train, original_Y_test, R):
    """No cross-domain pairs used until AFTER unsupervised R is frozen."""
    # Fit only on target samples: Y-embedding -> its OWN raw force signal.
    decoder=Ridge(alpha=1e-6,fit_intercept=True)
    decoder.fit(yt,original_Y_train)
    # xt is the source test embedding transformed by the learnt orthogonal R
    pred=decoder.predict(xt@R.T)
    if not np.isfinite(pred).all():
        raise ArithmeticError("invalid physical force post-hoc prediction")
    baseline=np.broadcast_to(original_Y_train.mean(axis=0),original_Y_test.shape)
    denom=float(np.sum((original_Y_test-baseline)**2))
    return {
        "paired_force_R2_vs_trainmean_EVAL_ONLY":float(1-
            np.sum((pred-original_Y_test)**2)/denom),
        "paired_force_RMSE_EVAL_ONLY":float(np.sqrt(np.mean((pred-original_Y_test)**2))),
        "paired_force_MAE_EVAL_ONLY":float(np.mean(np.abs(pred-original_Y_test))),
        "paired_force_RMSE_each_finger_EVAL_ONLY":
            np.sqrt(np.mean((pred-original_Y_test)**2,axis=0)).tolist(),
        "physical_train_mean_RMSE_baseline":float(
            np.sqrt(np.mean((baseline-original_Y_test)**2))),
        "raw_target_in_domain_decoder_RMSE_EVAL_ONLY":None,
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--seeds",type=int,nargs="+",default=list(SEEDS))
    parser.add_argument("--folds",type=int,nargs="+",default=list(FOLDS))
    parser.add_argument("--anchors",type=int,default=96)
    parser.add_argument("--outer",type=int,default=45)
    parser.add_argument("--configs",nargs="+",choices=[z[0] for z in CONFS],
                        default=[z[0] for z in CONFS])
    args=parser.parse_args();args.out.mkdir(exist_ok=True,parents=True)
    selected=[z for z in CONFS if z[0] in args.configs]
    runs=[]
    for conf,d in selected:
        mod=s3 if d==3 else s5
        if d==5:
            mod.K=K
        for fold in args.folds:
            folder=args.data_dir/conf
            file=folder/(f"trial_heldout_{fold}" + ("_5d" if d==5 else "") + ".npz")
            with np.load(file) as z:
                xt=np.asarray(z["X_train"],float)/np.sqrt(d)
                yt=np.asarray(z["Y_train"],float)/np.sqrt(d)
                xe=np.asarray(z["X_test"],float)/np.sqrt(d)
                ye=np.asarray(z["Y_test"],float)/np.sqrt(d)
                yphys_train=np.asarray(z["original_target_force_train"],float)
                yphys_test=np.asarray(z["original_target_force_test"],float)
                testtrial=np.unique(z["test_trial_id"]).tolist()
                traintrial=np.unique(z["train_trial_id"]).tolist()
            if xt.shape!=yt.shape!=(1000,d) or xe.shape!=ye.shape!=(250,d):
                raise AssertionError(f"shape error {conf} fold {fold}")
            if testtrial!=[fold] or fold in traintrial:raise AssertionError("trial leakage")
            # Same train-only independent 4-centroid splits for both methods.
            cx=KMeans(n_clusters=K,random_state=42,n_init=30).fit(xt)
            cy=KMeans(n_clusters=K,random_state=43,n_init=30).fit(yt)
            for seed in args.seeds:
                source_anchors=None;target_anchors=None
                for meth in ("hard","soft05"):
                    A=mod.make_domain_memberships(xt,meth,cx.cluster_centers_)
                    B=mod.make_domain_memberships(yt,meth,cy.cluster_centers_)
                    Xa,gx,dx=mod.quantize_group_measure(xt,A,args.anchors,seed+101)
                    Ya,gy,dy=mod.quantize_group_measure(yt,B,args.anchors,seed+202)
                    if meth=="hard":
                        source_anchors=Xa.copy();target_anchors=Ya.copy()
                    else:
                        if not np.allclose(Xa,source_anchors) or not np.allclose(Ya,target_anchors):
                            raise AssertionError("Different anchors between hard and soft")
                    R,P,train,history=mod.fit_orthogonal_hierarchical(
                        Xa,Ya,gx,gy,seed,max_iter=args.outer)
                    hold=mod.test_shared_reference(xt,yt,xe,ye,R,
                                                   cx.cluster_centers_,cy.cluster_centers_)
                    phys=physical_eval(xe,yt,yphys_train,yphys_test,R)
                    targetdecoder=Ridge(alpha=1e-6).fit(yt,yphys_train)
                    phys["raw_target_in_domain_decoder_RMSE_EVAL_ONLY"]=float(
                        np.sqrt(np.mean((targetdecoder.predict(ye)-yphys_test)**2)))
                    result={
                      "representation":conf,"dimension":d,"fold":fold,"seed":seed,
                      "method":meth,"K":K,"n_train":1000,"n_test":250,
                      "n_anchors":args.anchors,"train_trials":traintrial,
                      "heldout_trial":testtrial,"R":R.tolist(),"P":P.tolist(),
                      "regularization":{"eps_s":mod.ES,"eps_g":mod.EG,"soft_tau":mod.TAU},
                      "train":train,"shared_hard_test_reference":hold,
                      "test_W2_sq":mod.w2(xe,ye,R),
                      "identity_test_W2_sq":mod.w2(xe,ye,np.eye(d)),
                      "ambiguous_X":float(np.mean(A.max(1)<.8)),
                      "ambiguous_Y":float(np.mean(B.max(1)<.8)),
                      "n_train_group_membership_mass_X":np.sum(A,axis=0).tolist(),
                      "n_train_group_membership_mass_Y":np.sum(B,axis=0).tolist(),
                      "anchor_MSE_source":dx,"anchor_MSE_target":dy,
                      "physical_evaluation":phys,
                      "paired_force_labels_and_true_times_NOT_USED_FOR_TRAINING":True,
                      "is_original_HiWA_ADMM":False,
                    }
                    runs.append(result)
                    print("DONE",conf,fold,seed,meth,"W2",round(result["test_W2_sq"],5),
                          "phys_rmse",round(phys["paired_force_RMSE_EVAL_ONLY"],5),
                          "sharedT",round(ref_metric(hold,"T"),5) if ref_metric(hold,"T") is not None else "N/A",
                          "convergence",train["simple_numeric_convergence"],flush=True)
        (args.out/"partial_result.json").write_text(
          json.dumps({"status":"PARTIAL","runs":runs},ensure_ascii=False,indent=2)+"\n")
    assert len(runs)==len(selected)*len(args.folds)*len(args.seeds)*2

    summary=[];pairs=[]
    for conf,d in selected:
        for method in ("hard","soft05"):
            rr=[r for r in runs if r["representation"]==conf and r["method"]==method]
            summary.append({
                "representation":conf,"dimension":d,"method":method,
                "n":len(rr),"converged_count":sum(r["train"]["simple_numeric_convergence"] for r in rr),
                **{label:stats([func(r) for r in rr]) for label,func in [
                    ("heldout_T",lambda r:ref_metric(r["shared_hard_test_reference"],"T") if ref_metric(r["shared_hard_test_reference"],"T") is not None else float("nan")),
                    ("heldout_J",lambda r:ref_metric(r["shared_hard_test_reference"],"J") if ref_metric(r["shared_hard_test_reference"],"J") is not None else float("nan")),
                    ("heldout_W2_sq",lambda r:r["test_W2_sq"]),
                    ("paired_physical_force_R2_EVAL_ONLY",lambda r:r["physical_evaluation"]["paired_force_R2_vs_trainmean_EVAL_ONLY"]),
                    ("paired_physical_force_RMSE_EVAL_ONLY",lambda r:r["physical_evaluation"]["paired_force_RMSE_EVAL_ONLY"]),
                    ("heldout_decoder_rmse",lambda r:r["physical_evaluation"]["raw_target_in_domain_decoder_RMSE_EVAL_ONLY"]),
                ]}
            })
        for fold in args.folds:
            for seed in args.seeds:
                a=next(r for r in runs if r["representation"]==conf and r["fold"]==fold and r["seed"]==seed and r["method"]=="hard")
                b=next(r for r in runs if r["representation"]==conf and r["fold"]==fold and r["seed"]==seed and r["method"]=="soft05")
                pairs.append({
                    "representation":conf,"dimension":d,"fold":fold,"seed":seed,
                    "soft_minus_hard_W2_sq":b["test_W2_sq"]-a["test_W2_sq"],
                    "soft_minus_hard_shared_T":(
                        ref_metric(b["shared_hard_test_reference"],"T")-
                        ref_metric(a["shared_hard_test_reference"],"T")
                        if ref_metric(b["shared_hard_test_reference"],"T") is not None and
                           ref_metric(a["shared_hard_test_reference"],"T") is not None else None),
                    "soft_minus_hard_physical_RMSE":(
                        b["physical_evaluation"]["paired_force_RMSE_EVAL_ONLY"]-
                        a["physical_evaluation"]["paired_force_RMSE_EVAL_ONLY"]),
                    "soft_minus_hard_physical_R2":(
                        b["physical_evaluation"]["paired_force_R2_vs_trainmean_EVAL_ONLY"]-
                        a["physical_evaluation"]["paired_force_R2_vs_trainmean_EVAL_ONLY"]),
                })
    (args.out/"OT_results_all.json").write_text(
      json.dumps({"study":"FA/PCA factorial with predeclared representation subset x heldout folds x fixed seeds x both groupings",
                  "source": "Hyser s01/s1 Random Task, five actual trials",
                  "solver": "96 weighted-anchor two-level OT with a shared global orthogonal rotation; NOT original local Rij ADMM",
                  "no_synchronized_pairs_in_OT_training":True,
                  "force_decoder_trained_ONLY_in_target_domain":True,
                  "meta":{"configs":[z[0] for z in selected],"K":4,"eps_s":.1,"eps_g":.2,"soft_tau":.5,
                          "trial_heldout":True,"n_method_runs":len(runs)},
                  "runs":runs,"summaries":summary,"paired_differences":pairs},
                ensure_ascii=False,indent=2)+"\n")
    rows=[]
    for s in summary:
        rows.append({"representation":s["representation"],"d":s["dimension"],
            "method":s["method"],"n":s["n"],"converged_count":s["converged_count"],
            **{k+"_mean":v["mean"] for k,v in s.items() if isinstance(v,dict)},
            **{k+"_defined":v["n_defined"] for k,v in s.items() if isinstance(v,dict)}})
    with (args.out/"OT_comparison_6configs.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (args.out/"OT_pairwise_hard_soft_5fold.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    # Plot paired differences in physically comparable force units and within-representation W2
    fig,axes=plt.subplots(1,2,figsize=(13,5.3),constrained_layout=True)
    pos=np.arange(len(selected));labels=[x[0].replace("_","\n") for x in selected]
    for ax,k,title in [
        (axes[0],"soft_minus_hard_W2_sq",
            "Soft - Hard heldout W2² (only within each representation)"),
        (axes[1],"soft_minus_hard_physical_RMSE",
            "Soft - Hard actual physical force RMSE (all five fingers)")]:
        mu=[];sd=[]
        for conf,_ in selected:
            folds=[]
            for fold in args.folds:
                vv=[r[k] for r in pairs if r["representation"]==conf and r["fold"]==fold]
                folds.append(float(np.mean(vv)))
            mu.append(float(np.mean(folds)))
            sd.append(float(np.std(folds,ddof=1)))
        ax.bar(pos,mu,alpha=.82)
        ax.errorbar(pos,mu,yerr=sd,fmt="none",color="#444",capsize=3,lw=1)
        ax.axhline(0,color="#888",ls="--",lw=1)
        ax.set(xticks=pos,xticklabels=labels,ylabel="Soft minus Hard; negative is better",title=title)
        ax.tick_params(axis="x",labelsize=8);ax.grid(axis="y",alpha=.15)
    fig.suptitle("Hyser PCA/FA K4: 5 heldout trials, 2 fixed seeds, no paired OT fitting")
    fig.savefig(args.out/"OT_6config_paired_differences.png",dpi=175,bbox_inches="tight");plt.close(fig)
    fig,ax=plt.subplots(figsize=(12.8,5),constrained_layout=True)
    w=.34
    for i,(meth,offset) in enumerate([("hard",-w/2),("soft05",w/2)]):
        means=[next(r for r in summary if r["representation"]==conf and r["method"]==meth)["paired_physical_force_R2_EVAL_ONLY"]["mean"]
               for conf,d in selected]
        ax.bar(pos+offset,means,width=w,label=meth)
    ax.set(xticks=pos,xticklabels=labels,ylabel="R² vs training-mean force baseline",
           title="POSTHOC only: R(Xtest) -> train-only target decoder -> five physical forces")
    ax.axhline(0,color="#999",ls="--",lw=1);ax.legend(frameon=False);ax.grid(axis="y",alpha=.15)
    fig.savefig(args.out/"OT_true_fivefinger_physical_transfer_R2.png",dpi=175,bbox_inches="tight");plt.close(fig)

    md=["# Hyser PCA / FA factor representation on K4 Hard vs Soft-GCOT",
      "",
      "**This file contains a specified subset of the predeclared six representations, evaluated on the indicated heldout trials and fixed seeds.**",
      "Both source and target factor methods are fitted independently on the four train trials.",
      "Two-level P/Q OT + global orthogonal R with 96 weighted anchors, not original full local R_kl HiWA ADMM.",
      "FA and PCA output coordinates are train-whitened before training both hard and soft.",
      "No synchronized pairings/force labels in R,P,Q fitting, at any stage.",
      "",
      "For 3D the four combos implement X PCA/FA and Y PCA/FA from ALL FIVE original fingers.",
      "For 5D X is PCA5 or FA5; the target is FULL FIVE FORCE CHANNELS with train whitening, not reduced.",
      "This is not comparable to the historical Y3 experiment that discarded ring/little.",
      "",
      "| Setup | Hard W2² | Soft W2² | Hard shared T | Soft shared T | Hard physical RMSE | Soft physical RMSE |",
      "|---|---:|---:|---:|---:|---:|---:|"]
    for conf,d in selected:
        h=next(r for r in summary if r["representation"]==conf and r["method"]=="hard")
        so=next(r for r in summary if r["representation"]==conf and r["method"]=="soft05")
        fmt=lambda z:str(round(z,5)) if z is not None else "N/A"
        md.append(f"| {conf} | {fmt(h['heldout_W2_sq']['mean'])} | {fmt(so['heldout_W2_sq']['mean'])}"
                  f" | {fmt(h['heldout_T']['mean'])} | {fmt(so['heldout_T']['mean'])}"
                  f" | {fmt(h['paired_physical_force_RMSE_EVAL_ONLY']['mean'])} | {fmt(so['paired_physical_force_RMSE_EVAL_ONLY']['mean'])} |")
    md+=["",
      "## Correct inference",
      "",
      "- Absolute W2 or T between different FA/PCA embeddings is NOT a common metric because the learned geometries differ.",
      "- Compare Soft−Hard differences WITHIN each representation; the physical five-finger RMSE is also in common physical units.",
      "- Heldout shared T is evaluated after re-solving P/Q on a SAME Hard Kmeans cluster reference within each representation; it is NOT the cost of freezing the training P/Q.",
      "- The physical test RMSE uses paired timestamps ONLY AFTER unpaired R has been learned and target-only force decoder has been fitted on train data.",
      "- A target-only decoder is not a supervised EMG-to-force model: it does not learn paired Xtrain/Ytrain relationships.",
      "- Tiny differences based on five trials of one person cannot establish a universal effect or statistical significance.",
      "- Check convergence and any undefined T per run in OT_results_all.json; no missing values are imputed.",
      "- Baseline trial splits share a subject and session; external replication must include other sessions or subjects."
      ]
    (args.out/"OT_PCA_FA_REPORT_ZH.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    print(json.dumps({"status":"success","runs":len(runs),
      "summary":[{"representation":s["representation"],"method":s["method"],
                  "test_W2":s["heldout_W2_sq"]["mean"],
                  "shared_T":s["heldout_T"]["mean"],
                  "physical_RMSE":s["paired_physical_force_RMSE_EVAL_ONLY"]["mean"],
                  "physical_R2":s["paired_physical_force_R2_EVAL_ONLY"]["mean"],
                  "converged":s["converged_count"]} for s in summary]},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
