"""DANDI:000138 actual 3D (PCA3 / FA3 neural, train-PCA3 target) vs archived 4D.

Existing 4D experiment from original NWB is used as unchanged comparison:
320 train trials / 80 internal validation, seeds 42/43/44, original test100
unobserved in this run. No hand-neural pairing or physical retrieval enters OT
training / model selection. Uses identical ZIP hierarchical_alignment P/Q/R and
all original K4 budgets. Full physical matching is evaluated AFTER fitting on
the same heldout trial-validation pool using original physical 4D kinematics.

Do not interpret different-dimensional native OT costs as directly comparable.
Use train-only identity-normalized free-OT improvement + same physical 4D
matching metrics. 3D hand latent is PCA3 of train-standardized 4D hand,
not dropping velocity or position coordinates by hand.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax, xlogy
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler

from run_hard_clustering import load_trial_windows
from zip_gcot.alignment import hierarchical_alignment, SolverConfig
from zip_gcot.grouping import soft_groups
from zip_gcot.numerics import sinkhorn
from run_hard_hiwa import sample_groups
from run_controlled_soft_hard_ablation import build_measures
from preflight_geometry_semantics_retrieval import (
    cross_trial_nearest, retrieval_metrics,
)
from compare_pca4_fa4_inner_train_ot import (
    score_global_w2, raw_behavior_metrics,
)


def nested_val_3d(x, y, A, B, P, R, eps_sample=0.1, q_iter=120, hard=False):
    """Heldout P/Q evaluation in d3 with correctly scaled squared cost.

    The actual Q blocks here are freshly optimized on the heldout unpaired
    candidate subsets; they are NOT the 160-row training Q matrices.
    """
    d=x.shape[1]
    assert d==y.shape[1]==3 and R.shape==(d,d)
    mx=build_measures(x,A,hard)
    my=build_measures(y,B,hard)
    total=0.;ent=0.;err=0.; C=np.zeros((4,4))
    for i,(u,a) in enumerate(mx):
        for j,(v,b) in enumerate(my):
            cost=cdist(u@R.T/np.sqrt(d),v/np.sqrt(d),"sqeuclidean")
            q=sinkhorn(a,b,cost,eps_sample/max(P[i,j],1e-300),
                       maxiter=q_iter,tol=1e-9)
            err=max(err,float(np.max(abs(q.sum(0)-b))),
                    float(np.max(abs(q.sum(1)-a))))
            ci=float(np.sum(q*cost))
            total+=float(P[i,j])*ci
            C[i,j]=ci*d
            ent+=float(np.sum(xlogy(q,q)))
    if err>=1e-5: raise ArithmeticError(f"heldout Q marginal error {err}")
    return {
        "total_transport_W2sq":float(total*d),
        "total_transport_W2sq_per_dimension":float(total),
        "Q_max_marginal_error":err,
        "C_matrix":C.tolist(),
        "postfit_entropy_objective":float(total+.1*ent+.2*np.sum(xlogy(P,P)))
    }


def plain(x):
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [plain(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    return x


def avg(rows,key):
    return float(np.mean([float(r[key]) for r in rows]))


def summarize_entry(rows):
    def physical(name):
        return float(np.mean([r["validation_cross_trial_physical_retrieval"][name]
                              for r in rows]))
    return {
        "n_seeds":len(rows),
        "validation_free_W2sq_mean":avg(rows,"validation_free_W2sq"),
        "validation_free_W2sq_per_dimension_mean":avg(rows,"validation_free_W2sq_per_dimension"),
        "validation_free_W2sq_relative_improvement_vs_same_rep_identity_mean":avg(
            rows,"validation_W2_relative_improvement_vs_identity"),
        "validation_nested_PQ_transport_W2sq_mean":avg(
            rows,"validation_nested_PQ_transport_W2sq"),
        "validation_nested_PQ_transport_W2sq_per_dimension_mean":avg(
            rows,"validation_nested_PQ_transport_W2sq_per_dimension"),
        "physical_position_mean_error":physical("position_mean"),
        "physical_velocity_vector_mean_error":physical("speed_vector_mean_error"),
        "physical_position_hit10pct":physical("pos_hit10pct"),
        "physical_direction_hit30deg":physical("direction_hit30deg"),
        "physical_position_hit20pct":physical("pos_hit20pct"),
        "physical_direction_hit45deg":physical("direction_hit45deg"),
        "admm_passed":int(sum(bool(r["strict_admm_passed"]) for r in rows))
    }


def summarize_4d(ref):
    out={}
    for src in ("PCA4","FA4"):
        out[src]={}
        identity=ref["methods"][src]["identity_control"]
        w0=float(identity["heldout_unpaired_W2sq"])
        for meth in ("Hard","Soft"):
            rows=ref["methods"][src][meth]
            values=[]
            for r in rows:
                z={
                  "seed":r["seed"],
                  "validation_free_W2sq":r["validation_free_W2sq"],
                  "validation_free_W2sq_per_dimension":r["validation_free_W2sq"]/4.,
                  "validation_W2_relative_improvement_vs_identity":1.-r["validation_free_W2sq"]/w0,
                  "validation_nested_PQ_transport_W2sq":r["validation_nested_PQ_transport_W2sq"],
                  "validation_nested_PQ_transport_W2sq_per_dimension":r["validation_nested_PQ_transport_W2sq"]/4.,
                  "validation_cross_trial_physical_retrieval":
                        r["validation_cross_trial_physical_retrieval"],
                  "strict_admm_passed":r["strict_admm_passed"]
                }
                values.append(z)
            out[src][meth]=dict(per_seed=values,summary=summarize_entry(values))
        out[src]["identity"]=identity
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--nwb",type=Path,required=True)
    p.add_argument("--original-clusters",type=Path,required=True)
    p.add_argument("--four-d-reference",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seeds",nargs="+",type=int,default=[42,43,44])
    p.add_argument("--outer",type=int,default=500)
    p.add_argument("--inner",type=int,default=20)
    p.add_argument("--fa-max-iter",type=int,default=120)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    ref=json.loads(args.four_d_reference.read_text(encoding="utf8"))
    assert ref["dataset"]=="DANDI:000138 original NWB"
    assert ref["no_original_test_evaluated_or_optimized"]
    assert ref["original_data_hierarchy"]["internal_fit_trials"]==320
    assert ref["original_data_hierarchy"]["internal_validation_trials"]==80
    assert ref["split_seed"]==20261119
    for x in ("PCA4","FA4"):
        for m in ("Hard","Soft"):
            assert [r["seed"] for r in ref["methods"][x][m]]==args.seeds
    nr,br,valid,meta=load_trial_windows(args.nwb)
    assert meta["sha256"]==ref["file_sha256"]==(
        "1188ddf5b822dd9ac49afb8916f4b42a85867613cf7f169d0b04a1d1aeecd700")
    assert nr.shape==(500,35,162) and br.shape==(500,35,4)
    with np.load(args.original_clusters,allow_pickle=False) as cache:
        outer_train=np.asarray(cache["train_mask"],bool)
        outer_test=np.asarray(cache["test_mask"],bool)
        trial=np.asarray(cache["trial_of_window"],int)
        old_trials=np.asarray(cache["train_trials"],int)
        assert "shared_labels" not in cache.files
    assert outer_train.sum()==14000 and outer_test.sum()==3500
    assert np.array_equal(trial,np.repeat(np.arange(500),35))
    inner_train=np.asarray(ref["inner_train_trials"],int)
    inner_val=np.asarray(ref["inner_validation_trials"],int)
    assert len(inner_train)==320 and len(inner_val)==80
    assert set(map(int,inner_train))|set(map(int,inner_val))==set(map(int,old_trials))
    assert set(map(int,inner_train)).isdisjoint(map(int,inner_val))
    trmask=outer_train&np.isin(trial,inner_train)
    vamask=outer_train&np.isin(trial,inner_val)
    assert trmask.sum()==11200 and vamask.sum()==2800
    assert not np.any((trmask|vamask)&outer_test)
    neural=gaussian_filter1d(np.sqrt(np.maximum(nr,0)),sigma=2.,
                             axis=1,mode="nearest").reshape(-1,162)
    physical=br.reshape(-1,4).astype(float)
    source_scaler=StandardScaler().fit(neural[trmask])
    std=source_scaler.transform(neural)
    target_scaler=StandardScaler().fit(physical[trmask])
    behavior_std=target_scaler.transform(physical)
    target_pca=PCA(n_components=3,svd_solver="full",random_state=42)
    target_pca.fit(behavior_std[trmask])
    hand3=target_pca.transform(behavior_std)
    my=hand3[trmask].mean(0)
    sy=np.sqrt(np.mean(np.sum((hand3[trmask]-my)**2,axis=1)))
    y=(hand3-my)/sy
    assert y.shape==(17500,3)
    truephysical=physical[vamask]
    val_trial=trial[vamask]
    path_length,low_speed,_,_=raw_behavior_metrics(
        physical,trmask,vamask,trial)
    # Identical unpaired validation point selections to original 4D run.
    eval_rng=np.random.default_rng(3026)
    valrows=np.flatnonzero(vamask)
    ex=eval_rng.choice(len(valrows),size=200,replace=False)
    ey=eval_rng.choice(len(valrows),size=200,replace=False)
    qx=eval_rng.choice(len(valrows),size=128,replace=False)
    qy=eval_rng.choice(len(valrows),size=128,replace=False)
    train_rows=np.flatnonzero(trmask)
    cfg=SolverConfig(
        maxiter=args.outer,tol=1e-9,mu=.05,
        epsilon_group=.2,epsilon_sample=.1,
        inner_maxiter=args.inner,inner_tol=.01,
        sinkhorn_maxiter=45,group_sinkhorn_maxiter=200,
        sinkhorn_tol=1e-8,stopping="legacy")
    ky=KMeans(n_clusters=4,n_init=20,random_state=42).fit(y[trmask])
    labels_y=ky.predict(y)
    B,proto_y,_=soft_groups(y[trmask],4,seed=43,
                             entropy_weight=1.,maxiter=300)
    b=softmax(y@proto_y.T,axis=1)
    assert np.allclose(b[trmask],B,atol=1e-8)
    result={
        "research_question":"Does 3D reduce OT and physical matching errors vs fixed PCA4/FA4?",
        "dataset":"DANDI:000138, same original NWB SHA and original train/val split as archived 4D",
        "raw_nwb_sha256":meta["sha256"],
        "test100_trials_untouched":True,
        "training_never_uses_sync_pairs":True,
        "original_kinematics_for_physical_scoring":True,
        "target_3D":"PCA3 fitted on train-standardized physical pos_x,pos_y,vel_x,vel_y; same target for PCA3 and FA3 source",
        "target_PCA3_explained_variance_fraction":float(
            target_pca.explained_variance_ratio_.sum()),
        "target_PCA3_components":target_pca.components_.tolist(),
        "target_train_KMeans_counts":np.bincount(labels_y[trmask],minlength=4).tolist(),
        "target_3D_RMS":float(sy),
        "physical_match_position_threshold_fraction_training_path":.10,
        "physical_match_position_threshold_native_units":float(.1*path_length),
        "physical_direction_threshold_degrees":30.,
        "physical_low_speed_cutoff_20pct_train":float(low_speed),
        "training_trials":320,"validation_trials":80,
        "original_100_test_trials_never_used":True,
        "OT_training_samples_per_domain":160,
        "group_count":4,"group_sample_n_hard":40,
        "outer":args.outer,"inner":args.inner,
        "sample_sinkhorn":45,"group_sinkhorn":200,
        "mu":.05,"eps_sample":.1,"eps_group":.2,
        "source_feature_preprocessing":"raw neural 162 channels sqrt, 40ms smooth, train-only standard scale, PCA3 or FA3, train-only RMS scale",
        "match_pool":"2800 validation samples, nearest target candidate restricted to a different trial; same underlying physical4D labels for 3D/4D",
        "all_hard_soft_train_fits_independent_source_target":True,
        "physical_true_pair_only_used_after_fitting":True,
        "four_d_archived_same_trial_split_reference":{
            "reference_file":str(args.four_d_reference),
            "methods":summarize_4d(ref),
            "random_cross_trial_baseline":ref["random_cross_trial_baseline"]},
        "methods":{},"summary_3d":{}
    }
    for src in ("PCA3","FA3"):
        if src=="PCA3":
            estimator=PCA(n_components=3,svd_solver="full",random_state=42)
        else:
            estimator=FactorAnalysis(n_components=3,svd_method="randomized",
                        random_state=42,max_iter=args.fa_max_iter,tol=1e-3,
                        iterated_power=4)
        tic=time.monotonic()
        estimator.fit(std[trmask])
        neural3=estimator.transform(std)
        xxmean=neural3[trmask].mean(0)
        xxscale=np.sqrt(np.mean(np.sum((neural3[trmask]-xxmean)**2,axis=1)))
        x=(neural3-xxmean)/xxscale
        assert x.shape==(17500,3)
        km=KMeans(n_clusters=4,n_init=20,random_state=42).fit(x[trmask])
        labels_x=km.predict(x)
        A,proto_x,_=soft_groups(x[trmask],4,seed=42,
                                 entropy_weight=1.,maxiter=300)
        a=softmax(x@proto_x.T,axis=1)
        assert np.allclose(a[trmask],A,atol=1e-8)
        xv=x[vamask]; yv=y[vamask]
        identity_w2=score_global_w2(xv,yv,np.eye(3),ex,ey)
        identity_match=cross_trial_nearest(xv,yv,val_trial)
        identity_phys=retrieval_metrics(identity_match,truephysical,low_speed,path_length)
        result["methods"][src]={
            "feature_fit_duration_s":float(time.monotonic()-tic),
            "neural_PCA3_train_variance_fraction":float(
                estimator.explained_variance_ratio_.sum()) if src=="PCA3" else None,
            "neural_FA_iterations":int(getattr(estimator,"n_iter_",0)) if src=="FA3" else None,
            "train_neural_KMeans_counts":np.bincount(labels_x[trmask],minlength=4).tolist(),
            "identity_control":{
                "validation_W2sq":identity_w2,
                "cross_trial_physical_retrieval":identity_phys},
            "Hard":[],"Soft":[]
        }
        for method in ("Hard","Soft"):
            for seed in args.seeds:
                rng=np.random.default_rng(seed)
                idsx=sample_groups(train_rows,labels_x,40,rng)
                idsy=sample_groups(train_rows,labels_y,40,rng)
                if method=="Hard":
                    atr=np.eye(4)[labels_x[idsx]]
                    btr=np.eye(4)[labels_y[idsy]]
                    aval=np.eye(4)[labels_x[vamask][qx]]
                    bval=np.eye(4)[labels_y[vamask][qy]]
                else:
                    atr=a[idsx];btr=b[idsy]
                    aval=a[vamask][qx];bval=b[vamask][qy]
                hard=(method=="Hard")
                t0=time.monotonic()
                fit=hierarchical_alignment(
                    build_measures(x[idsx],atr,hard),
                    build_measures(y[idsy],btr,hard),
                    seed=seed,config=cfg,scale=True,retain_plans=True)
                elapsed=time.monotonic()-t0
                R=np.asarray(fit["R"]);P=np.asarray(fit["P"])
                assert R.shape==(3,3) and P.shape==(4,4)
                assert np.linalg.norm(R.T@R-np.eye(3))<1e-7
                assert max(abs(P.sum(0)-.25).max(),abs(P.sum(1)-.25).max())<1e-5
                qs={f"Q_{i}_{j}":np.asarray(fit[f"Q_{i}_{j}"])
                    for i in range(4) for j in range(4)}
                actual_x=build_measures(x[idsx],atr,hard)
                actual_y=build_measures(y[idsy],btr,hard)
                train_qerr=0.
                for i in range(4):
                    for j in range(4):
                        Q=qs[f"Q_{i}_{j}"]
                        train_qerr=max(train_qerr,
                            float(np.max(abs(Q.sum(1)-actual_x[i][1]))),
                            float(np.max(abs(Q.sum(0)-actual_y[j][1]))))
                assert train_qerr<1e-5
                match=cross_trial_nearest(xv@R.T,yv,val_trial)
                phys=retrieval_metrics(match,truephysical,low_speed,path_length)
                w2=score_global_w2(xv,yv,R,ex,ey)
                hier=nested_val_3d(xv[qx],yv[qy],aval,bval,P,R,
                                 hard=hard)
                run={
                    "method":method,"source_feature":src,"seed":seed,
                    "validation_free_W2sq":w2,
                    "validation_free_W2sq_per_dimension":w2/3.,
                    "validation_W2_relative_improvement_vs_identity":1.-w2/identity_w2,
                    "validation_nested_PQ_transport_W2sq":hier["total_transport_W2sq"],
                    "validation_nested_PQ_transport_W2sq_per_dimension":
                        hier["total_transport_W2sq_per_dimension"],
                    "validation_nested_PQ_max_marginal_error":
                        hier["Q_max_marginal_error"],
                    "validation_cross_trial_physical_retrieval":phys,
                    "strict_admm_passed":bool(fit["criteria_satisfied"]),
                    "legacy_stop_reason":fit["stop_reason"],
                    "iterations":len(fit["history"]),
                    "rotation_delta":float(fit["rotation_delta"]),
                    "consensus_residual":float(fit["consensus_residual"]),
                    "dual_residual":float(fit["dual_residual"]),
                    "P_change":float(fit["coupling_delta"]),
                    "train_Q_max_marginal_error":train_qerr,
                    "P_entropy":float(-np.sum(xlogy(P,P))),
                    "runtime_sec":elapsed,
                    "source_train_sample_indices_sha256":
                      hashlib.sha256(idsx.tobytes()).hexdigest(),
                    "target_train_sample_indices_sha256":
                      hashlib.sha256(idsy.tobytes()).hexdigest(),
                    "P":P.tolist(),"R":R.tolist(),
                }
                result["methods"][src][method].append(run)
                np.savez_compressed(
                    args.output/f"DANDI_{src}_{method}_seed{seed}_FULL_P_Q_R.npz",
                    P=P,R=R,local_R=np.asarray(fit["local_R"]),
                    dual=np.asarray(fit["dual"]),
                    source_train_row_indices=idsx,
                    target_train_row_indices=idsy,
                    sampled_source_features=x[idsx],
                    sampled_target_features=y[idsy],
                    **qs)
                (args.output/"DANDI_3D_VS_4D_RESULTS.json").write_text(
                    json.dumps(plain(result),ensure_ascii=False,indent=2)+"\n",
                    encoding="utf8")
                print(
                    f"DANDI_3D {src} {method} seed{seed}: "
                    f"W2={w2:.5f} nestedPQ={hier['total_transport_W2sq']:.5f} "
                    f"poserr={phys['position_mean']:.4f} poshit={phys['pos_hit10pct']:.4f} "
                    f"dirhit={phys['direction_hit30deg']:.4f} "
                    f"admm={run['strict_admm_passed']} iter={run['iterations']}",
                    flush=True)
        result["summary_3d"][src]={
            method:summarize_entry(result["methods"][src][method])
            for method in ("Hard","Soft")}
    # Persist the complete final summary, including the last FA3 block.
    (args.output/"DANDI_3D_VS_4D_RESULTS.json").write_text(
        json.dumps(plain(result),ensure_ascii=False,indent=2)+"\\n",
        encoding="utf8")
    lines=[
        "# DANDI 3D vs archived 4D: real neural–hand matching and OT errors",
        "",
        "**Same raw NWB SHA and same trial-disjoint 320/80 internal split; "
        "original 100 test trials never used.**",
        "3D: neural 162ch -> train PCA3 or train FA3; target original hand "
        "pos_x,pos_y,vel_x,vel_y -> train-standardize -> train PCA3. "
        "4D archival comparator: neural PCA4 or FA4, original 4-component "
        "target. For each method, K4,40/group, original 500/20/45/200, "
        "mu.05,eps_s.1,eps_g.2, same seed 42-44 and same independent domain "
        "grouping algorithms. No paired labels go into fitting.",
        "",
        "**Primary comparison on exactly identical original 4D physical "
        "movement**: cross-trial position and velocity vector mean error, "
        "position hit <=10% train median trial path, velocity angle <=30deg "
        "(train speed exclusion). All nearest neighbours chosen only after fit "
        "and exclude the source window's trial.",
        "",
        "| Source / target | Group | N seeds | Free W2² (native d) | "
        "Relative drop vs same-d identity | Nested P/Q (native d) | "
        "Position mean error (native) | Velocity vector mean error (native) | "
        "Position hit10% | Direction hit30deg | Strict ADMM |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for nd in (3,4):
        for src in (f"PCA{nd}",f"FA{nd}"):
            for m in ("Hard","Soft"):
                v=(result["summary_3d"][src][m] if nd==3 else
                   result["four_d_archived_same_trial_split_reference"]["methods"][src][m]["summary"])
                lines.append(
                    f"| {src} / hand{'PCA3' if nd==3 else 'original4'} | {m} | "
                    f"{v['n_seeds']} | {v['validation_free_W2sq_mean']:.4f} | "
                    f"{v['validation_free_W2sq_relative_improvement_vs_same_rep_identity_mean']:.2%} | "
                    f"{v['validation_nested_PQ_transport_W2sq_mean']:.4f} | "
                    f"{v['physical_position_mean_error']:.4f} | "
                    f"{v['physical_velocity_vector_mean_error']:.4f} | "
                    f"{v['physical_position_hit10pct']:.2%} | "
                    f"{v['physical_direction_hit30deg']:.2%} | "
                    f"{v['admm_passed']}/{v['n_seeds']} |")
    rb=result["four_d_archived_same_trial_split_reference"]["random_cross_trial_baseline"]
    lines += [
        "",
        f"Same physical random cross-trial baseline: position hit10%={rb['pos_hit10pct']:.2%}; "
        f"direction hit30deg={rb['direction_hit30deg']:.2%}.",
        f"Train median path 10% physical position radius = "
        f"{.1*path_length:.5g} original stored units (do not invent cm/mm).",
        "",
        "**Critical interpretation:** native latent free W2² and groupwise "
        "nested P/Q transport use different 3D versus 4D feature spaces. "
        "The absolute numbers cannot alone determine which embedding is "
        "better. Within-dim relative OT improvement and matching errors "
        "evaluated in ORIGINAL physical4D coordinates are more comparable.",
        "Hard versus Soft native nested OT costs use different group probability "
        "measures; do not rank Soft solely by these unlike objective values.",
        "Numerical Soft ADMM failures must be reported; not optimized means "
        "no unconditional win claim.",
        "This is a **single internal 320/80 split with 3 optimizer seeds**, "
        "not an independent final confirmation. Do not use the previously "
        "inspected original 100-test trials to select 3D versus 4D.",
        "",
        "[Full JSON with all seeds](DANDI_3D_VS_4D_RESULTS.json) | "
        "[3D full P/Q/R blocks](.) | "
        "[Underlying archived 4D results](../PCA4_FA4_inner_validation/DANDI_PCA4_FA4_REPORT_ZH.md)",
        ""
    ]
    (args.output/"DANDI_3D_VS_4D_REPORT_ZH.md").write_text(
        "\n".join(lines),encoding="utf8")
    print("DANDI_3D_VS_4D_COMPLETE",flush=True)


if __name__=="__main__":
    main()
