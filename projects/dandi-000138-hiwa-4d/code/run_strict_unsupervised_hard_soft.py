"""Strictly UNPAIRED 4D Hard/Soft comparison on the original DANDI000138 cache.

Only the PREEXISTING original independently trained KMeans dataset
results/clustering/clustered_4d.npz is valid input; fixed_groups_4d.npz is
explicitly REJECTED. Keeps original 17,500 windows, trial-disjoint 400/100
train/test, normalization, original full HiWA/GCOT solver, numeric parameters.

Hard: independent existing source KMeans labels and target KMeans labels.
Soft: run UNMODIFIED ZIP soft_groups(X_train) and soft_groups(Y_train)
independently, no pairing, no label alignment or permutation matching.
Group P is FREE, learned by hierarchical OT, never identity-constrained.
For solver-control, also fit hard onehot via exact SAME ZIP GCOT
hierarchical_alignment with all parameters equal to Soft.
Original PyHiWA Hard benchmark remains an independent historical reference.

ONLY after ALL seeds are fitted use row-time pairing for held-out paired
metrics and common source-identity plotting colors. Test pairing is never
an argument to KMeans, soft_groups, train-time sample selection or the OT
solver. Source/target training selections use independently fitted groups.

Input is already the frozen 4D 400/100 trial split; not a new 250-pair experiment.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import softmax,xlogy
from scipy.spatial import cKDTree
from zip_gcot.grouping import soft_groups
from zip_gcot.alignment import hierarchical_alignment,SolverConfig
from run_hard_hiwa import sample_groups
from run_controlled_soft_hard_ablation import build_measures,heldout_nested,fit_components
from run_soft_fixedgroup import ot_sq


def memberships_from_prototypes(x,c):
    v=softmax(x@c.T,axis=1)
    v=np.maximum(v,np.finfo(float).tiny)
    v/=v.sum(axis=1,keepdims=True)
    if not np.all(np.isfinite(v)):raise FloatingPointError("Non-finite soft memberships")
    return v


def metrics_after_fitting(x,y,R,ix,iy,labels_for_eval):
    """All methods call this AFTER training. True sync pairing ONLY HERE."""
    xt=x@R.T
    err=xt-y
    sse=float(np.sum(err**2))
    denom=float(np.sum((y-y.mean(0))**2))
    ref=float(np.sum(y**2))
    D=ot_sq(xt[ix],y[iy])
    # source label as color of its paired target is evaluation-only.
    nn=cKDTree(xt).query(y,k=1)[1]
    return dict(
      global_free_W2sq=D,
      MAE=float(np.abs(err).mean()),
      mean_sync_euclidean=float(np.linalg.norm(err,axis=1).mean()),
      sync_relative_MSE=float(sse/ref),
      sync_R2=float(1-sse/denom),
      visualization_label_1NN=float(np.mean(labels_for_eval[nn]==labels_for_eval)))


def fixed_eval_indices(test_count):
    rng=np.random.default_rng(20261117)
    ix=rng.choice(test_count,200,replace=False)
    iy=rng.choice(test_count,200,replace=False)
    rg=np.random.default_rng(80261117)
    # Independent 128-point source and 128-point target test samples; NOT
    # balanced by any group identity or paired trial indices.
    px=rg.choice(test_count,128,replace=False)
    py=rg.choice(test_count,128,replace=False)
    return ix,iy,px,py


def plot_frozen(x,y,labels_x,labels_y,rhard,rsoft,out):
    """Colors are copied ONLY in this function, once all models are trained."""
    xx=x@rhard.T;ss=x@rsoft.T
    clouds=[("Neural source (before)",x),
            ("Hard aligned source",xx),
            ("Soft aligned source",ss),
            ("Hand target | paired source colors",y)]
    from itertools import combinations
    c=plt.get_cmap("tab10")
    fig,axs=plt.subplots(6,4,figsize=(17,19),constrained_layout=True)
    for row,(i,j) in enumerate(combinations(range(4),2)):
        lo=min(v[:,i].min() for _,v in clouds);hi=max(v[:,i].max() for _,v in clouds)
        low=min(v[:,j].min() for _,v in clouds);high=max(v[:,j].max() for _,v in clouds)
        dx=.05*(hi-lo);dy=.05*(high-low)
        for col,(title,arr) in enumerate(clouds):
            a=axs[row,col]
            for k in range(4):
                m=labels_x==k  # X and Y point same color for VISUALIZATION only
                a.scatter(arr[m,i],arr[m,j],color=c(k),s=3,alpha=.5,rasterized=True)
            a.set(title=f"{title}, ({i+1},{j+1})",xlabel=f"dim{i+1}",ylabel=f"dim{j+1}",
                  xlim=(lo-dx,hi+dx),ylim=(low-dy,high+dy))
            a.grid(alpha=.1)
    fig.suptitle("UNSUPERVISED TRAINING | post-fit paired colors only for viewing")
    fig.savefig(out/"unsupervised_hard_soft_target_six_views.png",dpi=160,bbox_inches="tight")
    plt.close(fig)
    fig,axs=plt.subplots(1,4,figsize=(17.5,4.7),constrained_layout=True)
    for a,(title,arr) in zip(axs,clouds):
        for k in range(4):
            m=labels_x==k
            a.scatter(arr[m,0],arr[m,1],color=c(k),s=4,alpha=.48,rasterized=True)
        a.set(title=title,xlabel="coordinate 1",ylabel="coordinate 2")
    fig.savefig(out/"unsupervised_hard_soft_target_xy.png",dpi=190,bbox_inches="tight")
    plt.close(fig)
    # Target-domain INDEPENDENT groups shown in separate, differently
    # labeled chart so they are never mistaken for paired source colors.
    fig,axs=plt.subplots(1,2,figsize=(11,4.7),constrained_layout=True)
    for a,cl,title in [
        (axs[0],labels_x,"Target colored by PAIRED source labels (visualization only)"),
        (axs[1],labels_y,"Target colored by INDEPENDENT target KMeans labels")]:
        for k in range(4):
            mask=cl==k
            a.scatter(y[mask,0],y[mask,1],color=c(k),s=4,alpha=.48,rasterized=True)
        a.set(title=title,xlabel="Hand coord 1",ylabel="Hand coord 2")
    fig.savefig(out/"target_visual_labels_vs_independent_training_groups.png",
                dpi=200,bbox_inches="tight")
    plt.close(fig)


def summarize(runs):
    fields={
       "P_Q_total_OT_W2sq":("heldout_after","total_transport_W2sq"),
       "P_Q_total_before":("heldout_before","total_transport_W2sq"),
       "global_free_W2sq":("metrics","global_free_W2sq"),
       "MAE":("metrics","MAE"),
       "sync_mean_Euclidean":("metrics","mean_sync_euclidean"),
       "sync_relative_MSE":("metrics","sync_relative_MSE"),
       "sync_R2":("metrics","sync_R2"),
       "posthoc_color_1NN":("metrics","visualization_label_1NN"),
       "training_P_Q_geometry":("train","global_geometry_total_W2sq"),
       "training_original_entropy_objective":("train","ZIP_original_entropy_objective_scaled"),
    }
    out={}
    for name,rr in runs.items():
        out[name]={}
        for alias,(group,key) in fields.items():
            v=np.array([r[group][key] for r in rr],float)
            out[name][alias]={"mean":float(v.mean()),
                             "sd_sample":float(v.std(ddof=1)) if len(v)>1 else 0.,
                             "each":v.tolist()}
        out[name]["admm_criteria_passed"]=sum(bool(z["admm_criteria_passed"]) for z in rr)
        out[name]["legacy_residual_audit_passed"]=sum(bool(z["legacy_residual_audit_passed"]) for z in rr)
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--clusters",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seeds",nargs="+",type=int,default=[42,43,44,45,46])
    p.add_argument("--outer",type=int,default=500)
    p.add_argument("--inner",type=int,default=20)
    p.add_argument("--sample-sinkhorn",type=int,default=45)
    p.add_argument("--group-sinkhorn",type=int,default=200)
    p.add_argument("--eval-Q-iters",type=int,default=120)
    p.add_argument("--prototype-iters",type=int,default=300)
    p.add_argument("--entropy-weight",type=float,default=1.0)
    p.add_argument("--mu",type=float,default=.05)
    p.add_argument("--epsilon-group",type=float,default=.2)
    p.add_argument("--epsilon-sample",type=float,default=.1)
    p.add_argument("--pilot",action="store_true")
    args=p.parse_args()
    out=args.output
    out.mkdir(parents=True,exist_ok=True)

    # Explicitly disallow known group-supervised cached arrays.
    if args.clusters.name=="fixed_groups_4d.npz":
        raise ValueError("Forbidden: fixed_groups_4d.npz derives Y groups from paired X labels")
    with np.load(args.clusters,allow_pickle=False) as data:
        if "shared_labels" in data.files:
            raise ValueError("Leakage: shared_labels field is forbidden in solver input")
        X=np.array(data["neural4"],float)
        Y=np.array(data["hand4"],float)
        train=np.array(data["train_mask"],bool)
        test=np.array(data["test_mask"],bool)
        lx=np.array(data["neural_groups"],int)
        ly=np.array(data["hand_groups"],int)
        trial=np.array(data["trial_of_window"],int)
    assert X.shape==Y.shape==(17500,4)
    assert train.sum()==14000 and test.sum()==3500 and not np.any(train&test)
    assert len(np.unique(trial[train]))==400 and len(np.unique(trial[test]))==100
    assert set(trial[train]).isdisjoint(set(trial[test]))
    assert len(np.unique(lx[train]))==len(np.unique(ly[train]))==4
    assert not np.array_equal(lx,ly),"Target Hard labels were copied from source labels"
    assert np.isfinite(X).all() and np.isfinite(Y).all()
    dataset_hash=hashlib.sha256(args.clusters.read_bytes()).hexdigest()
    # Same original per-domain RMS normalization in both earlier experiments.
    mx=X[train].mean(0);my=Y[train].mean(0)
    sx=float(np.sqrt(np.mean(np.sum((X[train]-mx)**2,axis=1))))
    sy=float(np.sqrt(np.mean(np.sum((Y[train]-my)**2,axis=1))))
    x=(X-mx)/sx;y=(Y-my)/sy
    # Absolutely no synchronized paired rows are used in training:
    # model is a product of source-only and target-only preprocessing.
    start=time.perf_counter()
    AX,CX,infoX=soft_groups(x[train],4,seed=42,
          entropy_weight=args.entropy_weight,maxiter=args.prototype_iters)
    BY,CY,infoY=soft_groups(y[train],4,seed=43,
          entropy_weight=args.entropy_weight,maxiter=args.prototype_iters)
    ax=memberships_from_prototypes(x,CX)
    by=memberships_from_prototypes(y,CY)
    if not (np.allclose(ax[train],AX,atol=1e-8) and
            np.allclose(by[train],BY,atol=1e-8)):
        raise AssertionError("Soft training memberships not reproducible from prototypes")
    if np.array_equal(AX,BY):raise AssertionError("B was copied from A")
    # Only independent train features fitted prototypes; test used via prediction.
    soft_group_summary={
        "source":infoX,"target":infoY,
        "source_mass":AX.mean(0).tolist(),"target_mass":BY.mean(0).tolist(),
        "source_entropy_normalized":float(
            -np.mean(np.sum(xlogy(AX,AX),axis=1))/np.log(4)),
        "target_entropy_normalized":float(
            -np.mean(np.sum(xlogy(BY,BY),axis=1))/np.log(4)),
        "source_centers":CX.tolist(),"target_centers":CY.tolist(),
        "prototype_fit_duration_s":float(time.perf_counter()-start),
    }
    np.savez_compressed(out/"independent_soft_groups.npz",
        source_prototypes=CX,target_prototypes=CY,
        source_train_soft=ax[train].astype(np.float32),
        target_train_soft=by[train].astype(np.float32),
        source_test_soft=ax[test].astype(np.float32),
        target_test_soft=by[test].astype(np.float32))
    cfg=SolverConfig(maxiter=args.outer,tol=1e-9,mu=args.mu,
          epsilon_group=args.epsilon_group,epsilon_sample=args.epsilon_sample,
          inner_maxiter=args.inner,inner_tol=.01,
          sinkhorn_maxiter=args.sample_sinkhorn,
          group_sinkhorn_maxiter=args.group_sinkhorn,
          sinkhorn_tol=1e-8,stopping="legacy")
    tridx=np.flatnonzero(train)
    xt=x[test];yt=y[test]
    ix,iy,ix2,iy2=fixed_eval_indices(len(xt))
    # Test data are not referenced until after a fitted R/P exists.
    runs={"hard":[],"soft":[]}
    for seed in args.seeds:
        rg=np.random.default_rng(seed)
        # Source and target hard clusters were fitted INDEPENDENTLY.
        # Shared train row indices are NOT used to select corresponding pairs.
        tx=sample_groups(tridx,lx,40,rg)
        ty=sample_groups(tridx,ly,40,rg)
        for method in ("hard","soft"):
            if method=="hard":
                Atr=np.eye(4)[lx[tx]];Btr=np.eye(4)[ly[ty]]
                Ate=np.eye(4)[lx[test]][ix2];Bte=np.eye(4)[ly[test]][iy2]
                ishard=True
            else:
                Atr=ax[tx];Btr=by[ty]
                Ate=ax[test][ix2];Bte=by[test][iy2]
                ishard=False
            # Assertions are on independent domain weights, not label overlap.
            if method=="soft" and np.array_equal(Atr,Btr):
                raise ValueError("Source/target memberships unexpectedly equal")
            st=time.perf_counter()
            fit=hierarchical_alignment(
                build_measures(x[tx],Atr,ishard),
                build_measures(y[ty],Btr,ishard),
                seed=seed,config=cfg,scale=True,retain_plans=True)
            elapsed=time.perf_counter()-st
            R=np.asarray(fit["R"]);P=np.asarray(fit["P"])
            ortho=float(np.linalg.norm(R.T@R-np.eye(4),"fro"))
            mass=float(max(abs(P.sum(0)-.25).max(),
                           abs(P.sum(1)-.25).max()))
            assert ortho<1e-7 and mass<1e-5
            traincomp=fit_components(fit,x[tx],y[ty],
                     Atr,Btr,args.epsilon_sample,ishard)
            assert traincomp["max_sample_Q_marginal_error"]<1e-5
            # Train fit is COMPLETE; now and only now read paired test arrays.
            ev=metrics_after_fitting(xt,yt,R,ix,iy,lx[test])
            before=heldout_nested(xt[ix2],yt[iy2],Ate,Bte,P,np.eye(4),
                                  args.epsilon_sample,args.eval_Q_iters,ishard)
            after=heldout_nested(xt[ix2],yt[iy2],Ate,Bte,P,R,
                                 args.epsilon_sample,args.eval_Q_iters,ishard)
            criteria=bool(fit["criteria_satisfied"])
            relaxed=bool(fit["rotation_delta"]<=.01 and
                   fit["consensus_residual"]<=.04 and
                   mass<1e-5 and ortho<1e-7 and
                   fit["marginal_error"]<1e-5)
            record=dict(seed=int(seed),method=method,
               fitted_using_pairing=False,
               training_domain_prototypes_independent=True,
               sample_source_rows_sha256=hashlib.sha256(tx.tobytes()).hexdigest(),
               sample_target_rows_sha256=hashlib.sha256(ty.tobytes()).hexdigest(),
               P= P.tolist(),R=R.tolist(),
               P_diagonal_mass=float(np.trace(P)),P_marginal_error=mass,
               iterations=len(fit["history"]),stop_reason=fit["stop_reason"],
               fit_runtime_s=float(elapsed),
               admm_criteria_passed=criteria,legacy_residual_audit_passed=relaxed,
               rotation_change=float(fit["rotation_delta"]),
               consensus_residual=float(fit["consensus_residual"]),
               dual_residual=float(fit["dual_residual"]),
               primal_tolerance=float(fit["primal_tolerance"]),
               dual_tolerance=float(fit["dual_tolerance"]),
               coupling_change=float(fit["coupling_delta"]),
               Q_train_max_marginal=float(traincomp["max_sample_Q_marginal_error"]),
               Q_test_max_marginal=float(after["max_Q_marginal_error"]),
               orthogonality=ortho,metrics=ev,
               train=traincomp,heldout_before=before,heldout_after=after)
            runs[method].append(record)
            (out/f"{method}_seed{seed}.json").write_text(
                json.dumps(record,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
            print(f"STRICT UNSUPERVISED seed={seed} method={method}: "
                f"iterations={record['iterations']} admm_pass={criteria} "
                f"relaxed_pass={relaxed} P-Qtotal={after['total_transport_W2sq']:.6f} "
                f"globalOT={ev['global_free_W2sq']:.6f} "
                f"R2={ev['sync_R2']:.5f} t={elapsed:.1f}s",flush=True)
    summary=summarize(runs)
    data={
        "dataset":"DANDI 000138, original 4D cache", "input_sha256":dataset_hash,
        "description":"Strict unpaired Hard-vs-Soft GCOT solver-control with independently learned membership in each domain",
        "source_and_target_hard_kmeans_independent":True,
        "source_and_target_soft_prototypes_independent":True,
        "no_synchronized_pair_ids_used_in_fitting":True,
        "no_shared_labels_field_used":True,
        "no_target_membership_copy":True,
        "P_freely_optimized":True,
        "R_initialized_by_zip_original_random_seed_only":True,
        "target_group_permutation_fixed_from_paired_data":False,
        "true_pairing_used_only_postfit":True,
        "old_group_supervised_diagnostics_excluded":True,
        "source_code":"code/zip_gcot/{alignment.py,grouping.py,numerics.py} unmodified; code/run_strict_unsupervised_hard_soft.py comparison wrapper",
        "algorithm":"same original ZIP GCOT alignment kernel for both hard and soft; hard onehot vs separately fitted source/target soft prototype grouping",
        "training_n":int(train.sum()),"test_n":int(test.sum()),
        "training_trials":400,"test_trials":100,
        "config":{k:v for k,v in vars(args).items() if not isinstance(v,Path)},
        "source_mean":mx.tolist(),"target_mean":my.tolist(),
        "source_radius":sx,"target_radius":sy,
        "soft_prototypes":soft_group_summary,
        "evaluation":"Global 200x200 independently drawn exact W2²; 128x128 independently selected test pool for heldout P-Q Sinkhorn OT cost; sync metrics AFTER training only",
        "summary":summary,"runs":runs
    }
    (out/"STRICT_UNSUPERVISED_RESULTS.json").write_text(
       json.dumps(data,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    if 42 in args.seeds:
        rh=np.array(next(r for r in runs["hard"] if r["seed"]==42)["R"])
        rs=np.array(next(r for r in runs["soft"] if r["seed"]==42)["R"])
        plot_frozen(xt,yt,lx[test],ly[test],rh,rs,out)
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    fields=[("P_Q_total_OT_W2sq","Total hierarchical P-Q geometry"),
            ("global_free_W2sq","Global free exact W2²"),
            ("sync_R2","Synchronized R² (postfit only)"),
            ("mean_sync_euclidean","Synchronized average distance (postfit only)")]
    for ax,(k,title) in zip(axes.flat,fields):
        for i,method in enumerate(("hard","soft")):
            v=np.array([r["heldout_after"]["total_transport_W2sq"]
                        if k=="P_Q_total_OT_W2sq" else r["metrics"][k]
                        for r in runs[method]])
            ax.scatter(np.full(len(v),i)+np.linspace(-.12,.12,len(v)),v,s=22)
            ax.errorbar(i,v.mean(),yerr=v.std(ddof=1) if len(v)>1 else 0,
                        fmt="D",capsize=4)
        ax.set(xticks=[0,1],xticklabels=["Hard independent","Soft independent"],
               title=title)
        ax.grid(axis="y",alpha=.13)
    fig.suptitle("Strict UNSUPERVISED group training | 4D DANDI000138 | common solver")
    fig.savefig(out/"unsupervised_hard_soft_errors.png",dpi=200,bbox_inches="tight")
    plt.close(fig)

    lines=[
      "# Strict unpaired Hard-vs-Soft DANDI:000138 4D experiment",
      "",
      "## Training grouping and correspondence isolation",
      "",
      "Uses OLD original 17,500-window/500-trial four-dimensional dataset, NOT the source-only label copied file.",
      "Hard: X and Y KMeans independently fitted by original run_hard_clustering.py.",
      "Soft: directly use unchanged ZIP grouping.soft_groups(X_train) and grouping.soft_groups(Y_train) with distinct seeds.",
      "The target soft B is predicted from **Y features and Y prototypes**; it is never copied from A.",
      "Both methods call the same unchanged ZIP hierarchical_alignment with **free P**, same sample pools, same 500/20/45/200 limits, mu=.05, epsilon=.1/.2, scale=True, 5 seeds.",
      "True synchronized indices are used exclusively *after* R/P fitting for paired errors and shared scatter colors.",
      "No group label ordering/permutation chosen using paired identities; P learned by Sinkhorn.",
      "",
      "## Heldout five-seed summary (mean ± sample SD)",
      "",
      "| Metric | Unsupervised Hard | Unsupervised Soft |",
      "|---|---:|---:|",
    ]
    labels=[
      ("P-Q hierarchical geometry W2sq","P_Q_total_OT_W2sq"),
      ("Exact free global W2sq","global_free_W2sq"),
      ("Paired relative MSE (POSTFIT)","sync_relative_MSE"),
      ("Paired mean Euclidean (POSTFIT)","sync_mean_Euclidean"),
      ("MAE (POSTFIT)","MAE"),
      ("R² (POSTFIT)","sync_R2"),
      ("1NN shared visualization color (POSTFIT)","posthoc_color_1NN"),
      ("Training P-Q geometry W2sq","training_P_Q_geometry"),
      ("Training raw entropy objective scaled","training_original_entropy_objective"),
    ]
    for label,key in labels:
        vals=[summary[t][key] for t in ["hard","soft"]]
        lines.append(f"| {label} | "+ " | ".join(
           f"{v['mean']:.6f} ± {v['sd_sample']:.6f}" for v in vals)+" |")
    lines.extend(["","## Residual/convergence audit",
      "Passing a stop event or 500 iterations DOES NOT establish ADMM convergence.",
      "Original ZIP residual fields: rotation_delta, consensus_residual, dual_residual,",
      "primal_tolerance, dual_tolerance, coupling_delta, and sample/group marginal errors.",
      "Two flags are reported: strict original ADMM criteria; comparable relaxed historical 0.01/0.04 thresholds.",
      "",
      "| Method | Seed | Outer iters | ADMM pass | Relaxed pass | rotation | consensus | dual | Q train marginal | P marginal |",
      "|---|---:|---:|:---:|:---:|---:|---:|---:|---:|---:|"])
    for method in ("hard","soft"):
        for r in runs[method]:
            lines.append(f"| {method} | {r['seed']} | {r['iterations']} | "+
                  ("Yes" if r["admm_criteria_passed"] else "No")+" | "+
                  ("Yes" if r["legacy_residual_audit_passed"] else "No")+" | "+
                  f"{r['rotation_change']:.4g} | {r['consensus_residual']:.4g} | "+
                  f"{r['dual_residual']:.4g} | {r['Q_train_max_marginal']:.2g} | "+
                  f"{r['P_marginal_error']:.2g} |")
    lines+=["","## Essential interpretation",
      "- The test P-Q geometry excludes entropy terms, but evaluates an entropically obtained plan. It differs from original regularized training objective.",
      "- The Hard and Soft group-conditional probability distributions themselves differ, so the numeric P-Q cost alone is not a fixed-measure proof of better predictive alignment.",
      "- Global free W2² is a common, membership-independent distribution metric. Paired R²/MAE are postfit only and do not enter any model selection.",
      "- Original HiWA/PyHiWA Hard is a separate solver, NOT the same-solver control. Its original independently clustered results/hard_hiwa/ can be examined separately; previously shared-label Hard/Soft studies are GROUP-SUPERVISED DIAGNOSTICS.",
      "- Train/test trial split remains the original 400/100. All five seeds are reported; no best test R² seed selection.",
      "",
      "![Independent group Hard/Soft unpaired outcomes](unsupervised_hard_soft_errors.png)",
      "",
      "![Source before, Hard/Soft after, target — paired colors only after fit](unsupervised_hard_soft_target_six_views.png)",
      "",
      "![Different target plotting colors versus target-independent cluster IDs](target_visual_labels_vs_independent_training_groups.png)",
      ""]
    (out/"STRICT_UNSUPERVISED_REPORT.md").write_text("\n".join(lines),encoding="utf8")
    print("FINAL STRICT UNSUPERVISED SUMMARY",json.dumps({
        "summary":summary,
        "prototype_fit":soft_group_summary,
        "admm_pass":{k:v["admm_criteria_passed"] for k,v in summary.items()},
        "relaxed_pass":{k:v["legacy_residual_audit_passed"] for k,v in summary.items()},
    },ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
