"""Full 500-outer P/Q/R DANDI138 train-only spectral-start controlled trial.

Compare archived original random-start Hard and Soft with:
  hard_warm_full: original same K4 identity feature, spectral train-only start
  hard_warm_subspace2: rank2 independent per-hard-group local subspace geometry
  hard_warm_subspace3: rank3 independent per-hard-group local subspace geometry
  soft_warm_full: same soft prototype membership and exact full-support Q
 All models learn P,Q,R, no pairs/true labels in fit. All methods share
 same sampled 160 original row identities per source and target seed.
 Cross-trial physical 1NN and full unpaired W2 assessed only POSTFIT;
 nested P/Q transport on independent heldout samples assessed separately.
 Prior 100-test trials already inspected: this is exploratory, not a new
 independent confirmation set.
"""
from __future__ import annotations
import argparse,json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.special import softmax,xlogy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from zip_gcot.alignment import hierarchical_alignment,SolverConfig
from zip_gcot.preprocessing import (
    independent_group_projectors,independent_covariance_sign_candidates,
    select_spectral_candidate_train_ot)
from run_hard_hiwa import sample_groups
from run_controlled_soft_hard_ablation import build_measures,heldout_nested
from run_strict_unsupervised_hard_soft import fixed_eval_indices
from preflight_geometry_semantics_retrieval import cross_trial_nearest,retrieval_metrics

def globalW2(xx,yy,R,ix,iy):
    D=cdist((xx@R.T)[ix],yy[iy],"sqeuclidean")
    a,b=linear_sum_assignment(D)
    return float(np.mean(D[a,b]))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--clusters",type=Path,required=True)
    ap.add_argument("--archive",type=Path,required=True)
    ap.add_argument("--clustering-report",type=Path,required=True)
    ap.add_argument("--preflight",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seeds",nargs="+",type=int,default=[42,43,44])
    ap.add_argument("--outer",type=int,default=500)
    ap.add_argument("--inner",type=int,default=20)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    old=json.loads(args.archive.read_text())
    prec=json.loads(args.clustering_report.read_text())
    prior=json.loads(args.preflight.read_text())
    assert old["input_sha256"]==hashlib.sha256(args.clusters.read_bytes()).hexdigest()
    assert old["no_synchronized_pair_ids_used_in_fitting"]
    with np.load(args.clusters,allow_pickle=False) as z:
        assert "shared_labels" not in z.files
        X=np.asarray(z["neural4"],float)
        Y=np.asarray(z["hand4"],float)
        train=np.asarray(z["train_mask"],bool)
        test=np.asarray(z["test_mask"],bool)
        lx=np.asarray(z["neural_groups"],int)
        ly=np.asarray(z["hand_groups"],int)
        trial=np.asarray(z["trial_of_window"],int)
    assert X.shape==Y.shape==(17500,4)
    assert train.sum()==14000 and test.sum()==3500
    assert len(np.unique(trial[train]))==400 and len(np.unique(trial[test]))==100
    x=(X-np.asarray(old["source_mean"]))/float(old["source_radius"])
    y=(Y-np.asarray(old["target_mean"]))/float(old["target_radius"])
    # independently fitted source/target soft training prototypes from previous
    # training only; labels from neither side copied to the other.
    CX=np.asarray(old["soft_prototypes"]["source_centers"])
    CY=np.asarray(old["soft_prototypes"]["target_centers"])
    AX=softmax(x@CX.T,axis=1);BY=softmax(y@CY.T,axis=1)
    assert not np.allclose(AX,BY)
    # Author MATLAB per-domain rank2/3 group projectors fitted ONLY on the
    # original 14000 train, for ALL (including test) predicted group labels.
    projections={}
    for rank in (2,3):
        sx,opsx=independent_group_projectors(x,lx,rank,train)
        sy,opsy=independent_group_projectors(y,ly,rank,train)
        assert len(opsx)==len(opsy)==4
        projections[rank]=(sx,sy)
    g=np.random.default_rng(20261119)
    tridx=np.flatnonzero(train)
    left=g.choice(tridx,400,replace=False)
    right=g.choice(tridx,400,replace=False)
    candidates=independent_covariance_sign_candidates(x,y,train,train)
    R0,selection=select_spectral_candidate_train_ot(candidates,x,y,left,right)
    assert len(selection["all_train_costs"])==16
    cfg=SolverConfig(maxiter=args.outer,tol=1e-9,mu=.05,epsilon_group=.2,
                     epsilon_sample=.1,inner_maxiter=args.inner,inner_tol=.01,
                     sinkhorn_maxiter=45,group_sinkhorn_maxiter=200,
                     sinkhorn_tol=1e-8,stopping="legacy")
    raw_physical=Y*np.asarray(prec["scalers"]["behavior_scale"])+np.asarray(prec["scalers"]["behavior_mean"])
    ptest=raw_physical[test]
    paths=float(prior["preregistered_scoring_rules"]["train_median_700ms_path_length_native_units"])
    low=float(prior["preregistered_scoring_rules"]["speed_minimum_train_20percent_quantile_native_units"])
    i200,j200,i128,j128=fixed_eval_indices(int(test.sum()))
    xt=x[test];yt=y[test];tt=trial[test]
    specs={"hard_warm_full":("hard",None),
           "hard_warm_subspace2":("hard",2),
           "hard_warm_subspace3":("hard",3),
           "soft_warm_full":("soft",None)}
    methods={method:[] for method in specs}
    all_original_old_seed_records={
       meth:{r["seed"]:r for r in old["runs"][meth]} for meth in ("hard","soft")}
    out=dict(
       protocol="Strict unpaired original P/Q/R full GCOT with optional train-only spectral+local warm initial R and original HiWA group subspace projections",
       existing_baseline_unchanged=True,
       all_original_training_rotations_unmodified=True,
       source_target_true_point_pairs_used_in_training=False,
       source_target_behavior_labels_used_in_training=False,
       train_only_initial_R_selected_by_unpaired_W2=True,
       target_soft_prototypes_independent=True,
       source_soft_prototypes_independent=True,
       tested_on_previously_inspected_heldout_trials=True,
       data_levels=dict(all=17500,train=14000,test=3500,solver=160),
       original_input_sha256=old["input_sha256"],
       training_spectral_candidate_selection=selection,
       solver_config=dict(outer=args.outer,inner=args.inner,mu=.05,epsilon_group=.2,
                          epsilon_sample=.1,sample_sinkhorn=45,group_sinkhorn=200,
                          seeds=args.seeds),
       inherited_physical_position_threshold_train_path=paths,
       inherited_physical_speed_low_threshold_train=low,
       methods=methods,legacy_baseline_reference=dict(
           hard={str(k):{"R":r["R"],"P":r["P"],
                        "global_free_W2sq":r["metrics"]["global_free_W2sq"],
                        "strict_admm":r["admm_criteria_passed"]}
                 for k,r in all_original_old_seed_records["hard"].items()},
           soft={str(k):{"R":r["R"],"P":r["P"],
                        "global_free_W2sq":r["metrics"]["global_free_W2sq"],
                        "strict_admm":r["admm_criteria_passed"]}
                 for k,r in all_original_old_seed_records["soft"].items()}))
    # deterministic physically unpaired observation for test scoring
    # output P/Q for first seed in each variant to make results inspectable
    for seed in args.seeds:
        rg=np.random.default_rng(seed)
        ix=sample_groups(tridx,lx,40,rg)
        iy=sample_groups(tridx,ly,40,rg)
        assert len(np.unique(ix))==len(np.unique(iy))==160
        for name,(membership,rank) in specs.items():
            a=np.eye(4)[lx[ix]] if membership=="hard" else AX[ix]
            b=np.eye(4)[ly[iy]] if membership=="hard" else BY[iy]
            if rank is None:
                xinput,yinput=x,y
            else:
                xinput,yinput=projections[rank]
            t=time.perf_counter()
            fit=hierarchical_alignment(
                build_measures(xinput[ix],a,membership=="hard"),
                build_measures(yinput[iy],b,membership=="hard"),
                seed=seed,config=cfg,scale=True,retain_plans=True,
                initial_R=R0,local_start_from_global=True)
            R=fit["R"];P=fit["P"]
            assert np.linalg.norm(R.T@R-np.eye(4),"fro")<1e-7
            assert np.max(abs(P.sum(0)-.25))<1e-6 and np.max(abs(P.sum(1)-.25))<1e-6
            # All physical matching in the common unprojected original hand
            # representation; local train projections are solver-only geometry.
            from scipy.spatial import cKDTree
            test_match=cKDTree(yt).query(xt@R.T,k=1)[1]
            cross_match=cross_trial_nearest(xt@R.T,yt,tt)
            hit_full=retrieval_metrics(test_match,ptest,low,paths)
            hit_cross=retrieval_metrics(cross_match,ptest,low,paths)
            unpairedW2=globalW2(xt,yt,R,i200,j200)
            test_AX=np.eye(4)[lx[test][i128]] if membership=="hard" else AX[test][i128]
            test_BY=np.eye(4)[ly[test][j128]] if membership=="hard" else BY[test][j128]
            comp=heldout_nested(xinput[test][i128],yinput[test][j128],
                        test_AX,test_BY,P,R,.1,120,membership=="hard")
            result=dict(seed=seed,method=name,rank=rank,
                fitted_on_independent_Hard_or_Soft_groups=membership,
                original_X_sample_hash=hashlib.sha256(ix.tobytes()).hexdigest(),
                original_Y_sample_hash=hashlib.sha256(iy.tobytes()).hexdigest(),
                full_test_nearest=hit_full,cross_trial_test_nearest=hit_cross,
                original_unpaired_test_free_W2sq=unpairedW2,
                P_Q_nested_heldout_test=comp,
                model_P=P.tolist(),model_R=R.tolist(),
                P_entropy=float(-np.sum(xlogy(P,P))),
                strict_admm_pass=bool(fit["criteria_satisfied"]),
                consensus=float(fit["consensus_residual"]),
                rotation_delta=float(fit["rotation_delta"]),
                dual_residual=float(fit["dual_residual"]),
                P_marginal_error=float(fit["marginal_error"]),
                sample_Q_marginal_max=float(fit["max_marginal_error"]),
                train_original_global_objective=float(fit["objective"]),
                fit_runtime_sec=float(time.perf_counter()-t),
                iteration_count=len(fit["history"]))
            if seed==args.seeds[0]:
                pack={"R":fit["R"],"P":fit["P"],"local_R":fit["local_R"],
                      "dual":fit["dual"],"source_indices":ix,"target_indices":iy,
                      "source_raw_sample":x[ix],"target_raw_sample":y[iy]}
                for i in range(4):
                    for j in range(4):
                        pack[f"Q_{i}_{j}"]=fit[f"Q_{i}_{j}"]
                np.savez_compressed(args.output/f"{name}_seed{seed}_COMPLETE_P_Q_R.npz",**pack)
            methods[name].append(result)
            (args.output/"HIWA_DANDI_REAL_SOURCE_CODE_CORRECTION_RESULTS.json").write_text(
                json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
            print(f"DANDI_FIXED_PQR {name} seed={seed}: "
                  f"pos10cross={hit_cross['pos_hit10pct']:.4f} "
                  f"dir30cross={hit_cross['direction_hit30deg']:.4f} "
                  f"W2original={unpairedW2:.5f} "
                  f"heldout_nested={comp['total_transport_W2sq']:.5f} "
                  f"ADMM={result['strict_admm_pass']} "
                  f"iter={result['iteration_count']} sec={result['fit_runtime_sec']:.2f}",flush=True)
    # coherent audit report if all runs completed
    lines=["# DANDI000138 4D: author-compatible group subspaces and train-only warm starts with FULL P/Q/R fit",
      "",
      "**New controlled exploratory models, not an overwrite of any archived results.** "
      "All use original 17500/14000/3500/160 layers, K4, same 40/Hard-group training sample rows for all methods and each seed, original mu/epsilon/local/global iteration budgets, independent domains. All learn P,Q,R with original HiWA update equations; only optional R start and independently fitted local group projections differ.",
      "",
      "Initial R selected BEFORE test on unpaired 400 source and 400 target training coordinates using covariance eigenspace 16-sign candidates and train Hungarian W2; within-domain projections fitted on 14000 training points separately for source and target without pairs.",
      "",
      "| Method | Heldout cross-trial position hit10% | Velocity direction hit30° | Original full-domain test W2² | Projected P-Q heldout nested OT | Strict ADMM pass |",
      "|---|---:|---:|---:|---:|---:|"]
    def five(vals):
        a=np.asarray(vals,float)
        return f"{np.mean(a):.5f} ± {np.std(a,ddof=1):.5f}" if len(a)>1 else f"{np.mean(a):.5f}"
    for name,rs in methods.items():
        lines.append(f"| {name} | {five([r['cross_trial_test_nearest']['pos_hit10pct'] for r in rs])} | "
                     f"{five([r['cross_trial_test_nearest']['direction_hit30deg'] for r in rs])} | "
                     f"{five([r['original_unpaired_test_free_W2sq'] for r in rs])} | "
                     f"{five([r['P_Q_nested_heldout_test']['total_transport_W2sq'] for r in rs])} | "
                     f"{sum(r['strict_admm_pass'] for r in rs)}/{len(rs)} |")
    lines+=["",
      "**Original archived original Hard/Soft**: comparison values and their R/P remain accessible under strict_unsupervised_hard_soft; do not mix the projected nested OT geometry with raw nested OT when ranking methods because the metric differs.",
      "",
      "Any source-target physical success claim is exploratory; original heldout 100 trials have been examined previously. This trial did not tune using true test outcomes. The source groups remain neural Hard Kmeans, the target groups remain target-only Hard Kmeans; the Soft version keeps full-support independent prototypes and does not secretly inherit Hard labels.",
      "",
      "A correct implementation of the original optional projector path now exists in zip_gcot.preprocessing.py. Historical source code default behavior was regression tested bit-for-bit.",
      "",
      "[All rotations, 4x4 group P and per-seed tests](HIWA_DANDI_REAL_SOURCE_CODE_CORRECTION_RESULTS.json) | "
      "[Exact full Q plans and original row IDs](hard_warm_full_seed42_COMPLETE_P_Q_R.npz) | "
      "[Full archived original](../strict_unsupervised_hard_soft/STRICT_UNSUPERVISED_REPORT.md) | "
      "[This controlled script](../../code/compare_corrected_original_hiwa_on_dandi.py)",
      ""]
    (args.output/"HIWA_DANDI_REAL_SOURCE_CODE_CORRECTION_REPORT.md").write_text(
        "\n".join(lines),encoding="utf8")
    fig,axs=plt.subplots(1,3,figsize=(15,4.9),constrained_layout=True)
    keys=[("cross_trial_test_nearest","pos_hit10pct","Physical position hit ≤10% train path"),
          ("cross_trial_test_nearest","direction_hit30deg","Physical speed angle hit ≤30°"),
          ("_self","original_unpaired_test_free_W2sq","Physical raw 4D free W2²")]
    for ax,(section,key,label) in zip(axs,keys):
        names=list(methods)
        means=[];sds=[]
        for name in names:
            vals=np.asarray([r[key] if section=="_self" else r[section][key] for r in methods[name]])
            means.append(float(vals.mean()));sds.append(float(vals.std(ddof=1)) if len(vals)>1 else 0)
        ax.bar(range(len(names)),means,color=["#3977a6","#579b83","#7754a2","#d39742"],yerr=sds,capsize=3)
        ax.set(xticks=range(len(names)),xticklabels=["Hard full","Hard rank2","Hard rank3","Soft full"],
               ylabel="Heldout score",title=label)
        ax.grid(axis="y",alpha=.12)
        ax.tick_params(axis="x",rotation=20)
    fig.suptitle("DANDI: same train-only spectral orientation, now FULL HiWA P/Q/R optimized\n"
                 "Same strict unpaired sample IDs and original raw physical evaluation")
    fig.savefig(args.output/"DANDI_FULL_PQR_TRAIN_ONLY_WARMSTART_GEOMETRY.png",
               dpi=185,bbox_inches="tight")
    plt.close(fig)
    print("DANDI_FULL_PQR_COMPARISON_COMPLETE",flush=True)
if __name__=="__main__":
    main()
