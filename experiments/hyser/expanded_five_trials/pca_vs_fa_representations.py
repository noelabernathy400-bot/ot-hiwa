#!/usr/bin/env python3
"""Hyser PCA vs FactorAnalysis (FA) paired-sensor diagnostic and OT inputs.

NO leakage: five leave-one-trial-out folds; each preprocessing, PCA, FA,
whitening, and Ridge probe is fitted only on 4 TRAINING trials. Evaluation is
on the entire heldout fifth trial.

The Ridge probe is a POST-HOC SUPERVISED PHYSICS DIAGNOSTIC; its target values
and synchronization are NEVER seen by unsupervised HiWA/Soft-GCOT. No model
or hyperparameter is selected from heldout probe scores.

Inputs are the FULL five real force channels, not old selected-three-channel Y.
3D experiments transform BOTH domains to 3D, separately by PCA/FA. 5D
experiments change the X representation and keep Y=original all-fingers 5D
train-whitened; FA5(Y5) is computed only as a diagnostic (not needed for OT
because full-dimension whitening is redundant up to orthogonal coordinates).
"""
from __future__ import annotations
import argparse,csv,json,warnings,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA,FactorAnalysis
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score,r2_score,mean_squared_error,adjusted_rand_score
from sklearn.exceptions import ConvergenceWarning

FINGERS=["thumb","index","middle","ring","little"]
RANDOM_STATE=42
RIDGE_ALPHA=10.0
SOURCE_D=(3,5,10)
TARGET_D=(3,4,5)
OT_SETTINGS={
  "PCA3_PCA3":("PCA",3,"PCA",3),
  "FA3_PCA3":("FA",3,"PCA",3),
  "PCA3_FA3":("PCA",3,"FA",3),
  "FA3_FA3":("FA",3,"FA",3),
  "PCA5_RAW5":("PCA",5,"RAW",5),
  "FA5_RAW5":("FA",5,"RAW",5),
}

def whiter(train,test):
    mu=train.mean(axis=0)
    cov=np.cov(train-mu,rowvar=False)
    vals,vec=np.linalg.eigh(cov)
    if vals.min() < 1e-9:
        raise ValueError(f"Nearly singular embedded training covariance: min {vals.min():g}")
    w=(vec*vals**(-0.5))@vec.T
    ztr=(train-mu)@w;zte=(test-mu)@w
    if not np.all(np.isfinite(ztr)) or not np.all(np.isfinite(zte)):
        raise ValueError("Nonfinite whitened representation")
    return ztr,zte,{"mean":mu.tolist(),"eig_min":float(vals.min()),
                   "eig_max":float(vals.max()),
                   "condition":float(vals.max()/vals.min())}

def fits(kind,d,Xtr,Xte):
    if kind=="PCA":
        model=PCA(n_components=d,svd_solver="full")
    elif kind=="FA":
        model=FactorAnalysis(n_components=d,svd_method="randomized",
              iterated_power=5,random_state=RANDOM_STATE,
              max_iter=300,tol=1e-3)
    else:raise ValueError(kind)
    with warnings.catch_warnings(record=True) as warns:
        warnings.simplefilter("always",ConvergenceWarning)
        model.fit(Xtr)
    ztr=model.transform(Xtr);zte=model.transform(Xte)
    if kind=="PCA":
        re=model.inverse_transform(zte)
        desc={"PCA_explained_variance_train":float(sum(model.explained_variance_ratio_)),
              "PC_explained_variance_ratios":model.explained_variance_ratio_.tolist()}
    else:
        # posterior expectation of common-signal component, not error-inclusive
        re=zte@model.components_+model.mean_
        desc={"FA_noise_variance_mean":float(np.mean(model.noise_variance_)),
              "FA_noise_variance_min":float(np.min(model.noise_variance_)),
              "FA_noise_variance_max":float(np.max(model.noise_variance_)),
              "FA_iterations":int(model.n_iter_),
              "FA_converged_by_warning":not any(isinstance(w.message,ConvergenceWarning) for w in warns)}
    residual=float(np.mean((Xte-re)**2))
    baseline=float(np.mean(Xte**2))
    desc.update({"heldout_reconstruction_MSE_std":residual,
                 "heldout_reconstruction_R2_global":float(1-residual/baseline),
                 "heldout_model_avg_log_likelihood_per_window":None,
                 "heldout_model_loglike_error":None})
    try:
        ll=float(model.score(Xte))
        if np.isfinite(ll):desc["heldout_model_avg_log_likelihood_per_window"]=ll
    except Exception as e:desc["heldout_model_loglike_error"]=str(e)
    return ztr,zte,re,desc

def probe(Ztr,Zte,Ytr,Yte):
    # A check of source task-relevant information; synchronized labels are
    # used ONLY for this supervised probe, NEVER for OT embedding/transport.
    sc=StandardScaler().fit(Ztr)
    rt=sc.transform(Ztr);re=sc.transform(Zte)
    reg=Ridge(alpha=RIDGE_ALPHA,fit_intercept=True).fit(rt,Ytr)
    pred=reg.predict(re)
    baseline=np.broadcast_to(Ytr.mean(axis=0),Yte.shape)
    ss=np.sum((Yte-pred)**2)
    den=np.sum((Yte-baseline)**2)
    return {
        "test_R2_vs_training_mean_force":float(1-ss/den),
        "test_R2_per_finger_vs_test_mean":[float(x) for x in r2_score(Yte,pred,multioutput="raw_values")],
        "test_RMSE_per_finger":[float(x) for x in np.sqrt(np.mean((Yte-pred)**2,axis=0))],
        "training_mean_baseline_RMSE_per_finger":[float(x) for x in np.sqrt(np.mean((Yte-baseline)**2,axis=0))],
        "supervised_physical_probe_ONLY":True,"ridge_alpha_fixed":RIDGE_ALPHA
    }

def csvout(path,rows):
    if not rows:return
    with path.open("w",newline="",encoding="utf-8") as f:
        fields=list(dict.fromkeys(key for row in rows for key in row))
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--ot-input-dir",type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(exist_ok=True,parents=True)
    a.ot_input_dir.mkdir(exist_ok=True,parents=True)
    with np.load(a.input) as data:
        X=np.asarray(data["X_emg_log1p_rms_256"],float)
        Y=np.asarray(data["Y_force_5_raw"],float)
        tid=np.asarray(data["trial_id"],int)
        fingers=data["force_names"].tolist()
    assert X.shape==(1250,256) and Y.shape==(1250,5)
    assert fingers==FINGERS
    assert np.array_equal(np.bincount(tid)[1:],np.full(5,250))
    assert np.isfinite(X).all() and np.isfinite(Y).all()
    diagnostics=[];sensorR2=[];ridges=[];embeds=[]
    for holdout in range(1,6):
        m=tid!=holdout
        Xtr,Xte=X[m],X[~m]
        Ytr,Yte=Y[m],Y[~m]
        sx=StandardScaler().fit(Xtr)
        sy=StandardScaler().fit(Ytr)
        A,B=sx.transform(Xtr),sx.transform(Xte)
        C,D=sy.transform(Ytr),sy.transform(Yte)
        em={};ym={}
        for domain,src,dims,store in [
            ("source_EMG", (A,B), SOURCE_D, em),
            ("target_fivefinger",(C,D),TARGET_D,ym)
        ]:
            for dim in dims:
                for typ in ("PCA","FA"):
                    ztr,zte,re,desc=fits(typ,dim,*src)
                    # A saturated 5-factor FA fit to five physical forces
                    # may have rank-deficient factor scores. Never arbitrarily
                    # whiten a zero eigenvalue or make up clusters for it.
                    whiten_error=None
                    try:
                        wt,we,winfo=whiter(ztr,zte)
                        km=KMeans(n_clusters=4,n_init=20,random_state=42).fit(wt/np.sqrt(dim))
                        lab=km.predict(we/np.sqrt(dim))
                        sil=float(silhouette_score(wt/np.sqrt(dim),km.labels_))
                    except ValueError as exc:
                        if (domain=="target_fivefinger" and typ=="FA" and dim==5) or (domain=="source_EMG" and typ=="FA" and dim==10):
                            whiten_error=str(exc)
                            wt=we=None
                            winfo={"condition":None,"eig_min":None}
                            lab=None
                            sil=None
                        else:
                            raise
                    reconMSE=np.mean((src[1]-re)**2,axis=0)
                    reconstructR2=1-np.sum((src[1]-re)**2,axis=0)/np.maximum(np.sum(src[1]**2,axis=0),1e-10)
                    info={"fold":holdout,"domain":domain,"method":typ,"d":dim,
                          "train_n":int(len(ztr)),"test_n":int(len(zte)),
                          "K4_train_silhouette":sil,
                          "test_K4_cluster_sizes":np.bincount(lab,minlength=4).tolist() if lab is not None else None,
                          "embedding_whitening_status":"ok" if whiten_error is None else "SINGULAR_ABSTAIN",
                          "embedding_whitening_error":whiten_error,
                          "embedding_train_white_eig_condition":winfo["condition"],
                          "embedding_train_white_eig_min":winfo["eig_min"],
                          "heldout_sensor_R2_each_input_channel":reconstructR2.tolist(),
                          **desc}
                    diagnostics.append(info)
                    store[typ,dim]={"train":ztr,"test":zte,"trwhite":wt,"tewhite":we,
                                     "recon":re,"info":info,"winfo":winfo}
                    if domain=="source_EMG":
                        result=probe(ztr,zte,Ytr,Yte)
                        ridges.append({"fold":holdout,"method":typ,"d":dim,**result})
                    else:
                        # per original physical finger: R² of reduced factors'
                        # expected reconstruction on fifth heldout trial
                        physical_re=sy.inverse_transform(re)
                        r2per=r2_score(Yte,physical_re,multioutput="raw_values")
                        sensorR2.append({"fold":holdout,"method":typ,"d":dim,
                            "original_finger_reconstruction_R2":r2per.tolist(),
                            "original_finger_reconstruction_RMSE":
                                np.sqrt(np.mean((Yte-physical_re)**2,axis=0)).tolist()})
        # Train-only full-finger standardize+whitened 5D, with no need
        # for full-rank PCA/FA on target. No dimensional loss on force side.
        yrtr,yrte,winfo=whiter(C,D)
        if max(abs(yrtr.mean(0)))>1e-7:raise AssertionError("whitening not centered")
        for conf,(xs,xd,ys,yd) in OT_SETTINGS.items():
            if ys=="RAW":
                Ytrain,Ytest=yrtr,yrte
            else:
                Ytrain,Ytest=ym[ys,yd]["trwhite"],ym[ys,yd]["tewhite"]
            Xtrain,Xtest=em[xs,xd]["trwhite"],em[xs,xd]["tewhite"]
            assert Xtrain.shape==(1000,xd) and Ytrain.shape==(1000,yd)
            assert Xtest.shape==(250,xd) and Ytest.shape==(250,yd)
            assert xd==yd
            folder=a.ot_input_dir/conf
            folder.mkdir(exist_ok=True,parents=True)
            filename=f"trial_heldout_{holdout}" + ("_5d" if xd==5 else "") + ".npz"
            np.savez_compressed(folder/filename,
                X_train=Xtrain,Y_train=Ytrain,
                X_test=Xtest,Y_test=Ytest,
                train_trial_id=tid[m],test_trial_id=tid[~m],
                original_target_force_train=Ytr,original_target_force_test=Yte,
                metadata=np.array([conf],dtype="U40"),
                feature_finger_names=np.array(FINGERS))
        for dim in (3,5):
            aP=em["PCA",dim]["trwhite"]/np.sqrt(dim)
            aF=em["FA",dim]["trwhite"]/np.sqrt(dim)
            lP=KMeans(4,n_init=20,random_state=42).fit_predict(aP)
            lF=KMeans(4,n_init=20,random_state=42).fit_predict(aF)
            embeds.append({"fold":holdout,"domain":"source_EMG","d":dim,
                "K4_PCA_vs_FA_train_ARI":float(adjusted_rand_score(lP,lF))})
            # FA5 target scores may have rank < 5: no artificial white geometry.
            if ym["FA",dim]["trwhite"] is not None:
                pP=ym["PCA",dim]["trwhite"]/np.sqrt(dim)
                pF=ym["FA",dim]["trwhite"]/np.sqrt(dim)
                lP=KMeans(4,n_init=20,random_state=42).fit_predict(pP)
                lF=KMeans(4,n_init=20,random_state=42).fit_predict(pF)
                embeds.append({"fold":holdout,"domain":"target_fivefinger","d":dim,
                    "K4_PCA_vs_FA_train_ARI":float(adjusted_rand_score(lP,lF))})
        print("PREPARED heldout",holdout,"PCA/FA both domains (3,5,10 X; 3,4,5 Y)",
              "OT configs",list(OT_SETTINGS),flush=True)
    # Detail all heldout validation values; files are metadata, not OT inputs
    (a.out/"PCA_FA_diagnostics_full.json").write_text(json.dumps({
      "study":"Hyser subject01/session1 random, five heldout trials",
      "source_shape":[1250,256],"target_shape":[1250,5],
      "physical_force_channel_order":FINGERS,
      "decision":"Only train-fitted features are prepared; no heldout leak",
      "PCA_FA_fits":diagnostics,"supervised_physical_probes_POSTHOC":ridges,
      "fivefinger_physical_reconstruction_POSTHOC":sensorR2,
      "cluster_PCA_vs_FA_ARI":embeds,
      "OT_configs":{k:{"source_method":v[0],"d":v[1],
                       "target_method":v[2],"target_d":v[3]} for k,v in OT_SETTINGS.items()},
      "warnings":["PCA explained variance ratio is not an FA explained variance ratio.",
                  "FA5 of a 5-channel target can produce singular posterior factor covariance; record SINGULAR_ABSTAIN and keep lossless full5 target for OT.",
                  "FA factor scores are posterior estimates and can shrink common-signal reconstruction.",
                  "Y full 5D is directly whitened; after full-rank whitening PCA5 vs FA5 of Y can differ by an orthogonal basis if invertible; do not advertise 5D FA as recovering more data.",
                  "Ridge probes use paired timestamps on TRAINING trials ONLY; they are supervised diagnostics separate from HiWA OT and hyperparameter selection.",
                  "No cross-dimensional W2/T absolute value comparison is valid."]
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    rows=[]
    for q in diagnostics:
        rows.append({k:(json.dumps(v) if isinstance(v,list) else v)
                     for k,v in q.items() if k!="heldout_sensor_R2_each_input_channel"})
    csvout(a.out/"representation_metrics_per_fold.csv",rows)
    rows=[]
    for r in ridges:
        for j,name in enumerate(FINGERS):
            rows.append({"fold":r["fold"],"method":r["method"],"dim":r["d"],
                         "finger":name,"probe_test_R2":r["test_R2_per_finger_vs_test_mean"][j],
                         "probe_test_RMSE":r["test_RMSE_per_finger"][j],
                         "training_mean_baseline_RMSE":r["training_mean_baseline_RMSE_per_finger"][j],
                         "probe_global_R2":r["test_R2_vs_training_mean_force"]})
    csvout(a.out/"emg_to_fivefinger_Ridge_supervised_probe_ONLY.csv",rows)
    rows=[]
    for r in sensorR2:
        for j,name in enumerate(FINGERS):
            rows.append({"fold":r["fold"],"method":r["method"],"dim":r["d"],
                         "finger":name,"reconstruction_R2":r["original_finger_reconstruction_R2"][j],
                         "reconstruction_RMSE":r["original_finger_reconstruction_RMSE"][j]})
    csvout(a.out/"physical_fivefinger_force_reconstruction_per_fold.csv",rows)
    csvout(a.out/"PCA_FA_cluster_ARI_per_fold.csv",embeds)

    # Plots: distinct axes for different questions, no hard-coded cherry-pick
    fig,axs=plt.subplots(1,2,figsize=(12.7,4.5),constrained_layout=True)
    for ax,domain,dims in zip(axs,["source_EMG","target_fivefinger"],[SOURCE_D,TARGET_D]):
        for typ,mark in [("PCA","o"),("FA","s")]:
            vals=[(d,np.mean([x["heldout_reconstruction_R2_global"]
                  for x in diagnostics if x["domain"]==domain and x["method"]==typ and x["d"]==d]))
                  for d in dims]
            ax.plot([x[0] for x in vals],[x[1] for x in vals],marker=mark,label=typ)
        ax.set(title=domain,xticks=dims,xlabel="latent dimension",
               ylabel="heldout reconstruction R² (standardized features)")
        ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.suptitle("Hyser PCA vs FA: 5 trial-heldout folds, independent domains")
    fig.savefig(a.out/"representation_reconstruction_5fold.png",dpi=170,bbox_inches="tight")
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.4,5),constrained_layout=True)
    for typ,mark in [("PCA","o"),("FA","s")]:
        vals=[(d,np.mean([x["test_R2_vs_training_mean_force"] for x in ridges
                          if x["method"]==typ and x["d"]==d]))
              for d in SOURCE_D]
        ax.plot([x[0] for x in vals],[x[1] for x in vals],marker=mark,label=typ,linewidth=2)
    ax.set(xticks=SOURCE_D,xlabel="EMG latent dimension",ylabel="heldout five-finger Ridge R²",
           title="Supervised information-preservation diagnostic ONLY (never used for OT)")
    ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.savefig(a.out/"source_EMG_fivefinger_signal_probe.png",dpi=175,bbox_inches="tight")
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,4.7),constrained_layout=True)
    for ax,dim in zip(axs,(3,4)):
        xx=np.arange(5);w=.32
        for idx,(typ,offset) in enumerate([("PCA",-w/2),("FA",w/2)]):
            heights=[np.mean([r["original_finger_reconstruction_R2"][j]
                      for r in sensorR2 if r["method"]==typ and r["d"]==dim])
                     for j in range(5)]
            ax.bar(xx+offset,heights,width=w,label=typ)
        ax.set(xticks=xx,xticklabels=FINGERS,title=f"Y5 -> {dim}D -> five physical forces",
               ylabel="heldout physical force reconstruction R²")
        ax.tick_params(axis="x",rotation=35);ax.grid(alpha=.14);ax.legend(frameon=False)
    fig.suptitle("How much EACH real finger is lost by target PCA vs FA?")
    fig.savefig(a.out/"target_fivefinger_physical_reconstruction.png",dpi=175,bbox_inches="tight")
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12.7,4.7),constrained_layout=True)
    for ax,domain in zip(axs,["source_EMG","target_fivefinger"]):
        dims=SOURCE_D if domain=="source_EMG" else TARGET_D
        for typ,mark in [("PCA","o"),("FA","s")]:
            vals=[(float(np.mean(v)) if len(v)>0 else float("nan")) for d in dims
                  for v in [[r["K4_train_silhouette"] for r in diagnostics
                              if r["domain"]==domain and r["method"]==typ and r["d"]==d
                              and r["K4_train_silhouette"] is not None]]]
            ax.plot(dims,vals,marker=mark,label=typ)
        ax.set(title=domain,xticks=dims,xlabel="latent dimension",
               ylabel="K4 train silhouette in whitened features")
        ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.suptitle("PCA/FA structure: no target labels used for grouping")
    fig.savefig(a.out/"representation_K4_silhouette.png",dpi=170,bbox_inches="tight")
    plt.close(fig)
    # Summary helpful for GitHub review
    report=["# Hyser PCA / Factor Analysis: five-trial independent dimensionality-reduction experiment",
        "",
        "原HiWA论文的NHP实验确实使用FA把神经放电数据降为3D，再与3D运动目标对齐。",
        "此处既在EMG源域，也在真实五根手指力量目标域单独比较PCA/FA。",
        "5折完整试次留出，每折1000训练/250测试；标准化、PCA、FA、白化全部只用4训练试次拟合。",
        "FA用独立噪声模型，PCA最大化几何方差，二者目的不同。",
        "不可给FA编造PCA解释方差指标；应比较同一折的重建质量、heldout高斯似然、物理通道恢复、聚类稳定性。",
        "",
        "## Fivefold means: global test standardized reconstruction R²",
        "",
        "| Domain | d | PCA | FA | ΔFA−PCA |",
        "|---|---:|---:|---:|---:|"]
    for domain,dims in (("source_EMG",SOURCE_D),("target_fivefinger",TARGET_D)):
        for dim in dims:
            v={t:np.mean([r["heldout_reconstruction_R2_global"] for r in diagnostics
                          if r["domain"]==domain and r["d"]==dim and r["method"]==t])
               for t in ("PCA","FA")}
            report.append(f"| {domain} | {dim} | {v['PCA']:.5f} | {v['FA']:.5f} | {v['FA']-v['PCA']:+.5f} |")
    report+=["","## Heldout paired-supervised source information probe (not OT training)",
               "Five-finger physical force from EMG latent via TRAINING trial-only Ridge (fixed alpha=10).",
               "| EMG dimension | PCA heldout force R² | FA heldout force R² | Δ FA−PCA |",
               "|---:|---:|---:|---:|"]
    for d in SOURCE_D:
        v={t:np.mean([r["test_R2_vs_training_mean_force"] for r in ridges
                        if r["d"]==d and r["method"]==t]) for t in ("PCA","FA")}
        report.append(f"| {d} | {v['PCA']:.5f} | {v['FA']:.5f} | {v['FA']-v['PCA']:+.5f} |")
    report+=["","## Target five-finger reconstruction",
            "A reduction of Y from five physical fingers to 3D or 4D loses information by definition.",
            "The channel-by-channel heldout R² lives in physical_fivefinger_force_reconstruction_per_fold.csv.",
            "For OT, the 5D target is retained losslessly up to a train-fitted full-rank whitening.",
            "",
            "## OT input configurations",
            *[f"- {k}: source={v[0]}({v[1]}D), target={v[2]}({v[3]}D)"
              for k,v in OT_SETTINGS.items()],
            "",
            "OT comparison must remain INSIDE each representation because alternative FA/PCA feature spaces have different geometry.",
            "The source Ridge probe is SUPERVISED and MUST NOT be used to tune, train, label, or compare unsupervised OT.",
            "The old 3D experiment used physical thumb,middle,index, whereas new PCA3/FA3 target uses all five fingers: do not mix scores.",
            ""]
    (a.out/"PCA_FA_DIMENSION_DIAGNOSTICS_ZH.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(json.dumps({"status":"success","n_reductions":len(diagnostics),
            "n_probe_evaluations":len(ridges),"n_OT_features":len(OT_SETTINGS)*5,
            "report":str(a.out/"PCA_FA_DIMENSION_DIAGNOSTICS_ZH.md")},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
