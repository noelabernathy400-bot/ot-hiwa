"""Verify backward compatibility of HiWA repairs, then test optional warm starts.

- imports EXACT pre-edit ZIP alignment.py from an immutable repository commit
  using git show, compares old default R/P/history to current default.
- exercises the previously broken projection option; identity projectors must
  reproduce the same solver and nontrivial rank2 subspaces must be valid.
- tests newly introduced train-only spectral R starting point and optional
  local consensus starts, with original full P/Q/R optimization and known-R*
  synthetic positive controls, no source-target sample pairing or truth
  used in fitting/initialization/selection.
"""
from __future__ import annotations
import argparse,json,subprocess,types,time,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from scipy.special import softmax
from zip_gcot.alignment import SolverConfig,hierarchical_alignment,GCOT
from zip_gcot.preprocessing import (project_for_alignment,independent_group_projectors,
    independent_covariance_sign_candidates,select_spectral_candidate_train_ot)
from zip_gcot.grouping import soft_groups
from run_controlled_soft_hard_ablation import build_measures
from run_hard_hiwa import sample_groups
from preflight_synthetic_positive_control import synthesise,info_for_P,apply_eval

ORIGINAL_SHA="fc217c23217c7c25c5e8322fe7b2e8f98ba29eb9"
ORIGINAL_PATH="projects/dandi-000138-hiwa-4d/code/zip_gcot/alignment.py"

def load_original():
    txt=subprocess.check_output(["git","show",f"{ORIGINAL_SHA}:{ORIGINAL_PATH}"],text=True)
    module=types.ModuleType("zip_gcot.pre_edit_alignment")
    module.__package__="zip_gcot"
    exec(compile(txt,f"{ORIGINAL_SHA}:{ORIGINAL_PATH}","exec"),module.__dict__)
    return module,txt

def regression():
    old,source=load_original()
    d=synthesise(.28)
    x,y=d["xtrain"],d["ytrain"]
    lx=KMeans(4,random_state=42,n_init=20).fit_predict(x)
    ly=KMeans(4,random_state=42,n_init=20).fit_predict(y)
    rg=np.random.default_rng(42)
    ix=sample_groups(np.arange(len(x)),lx,40,rg)
    iy=sample_groups(np.arange(len(y)),ly,40,rg)
    sx=build_measures(x[ix],np.eye(4)[lx[ix]],True)
    sy=build_measures(y[iy],np.eye(4)[ly[iy]],True)
    cfg=SolverConfig(maxiter=45,tol=1e-9,mu=.05,inner_maxiter=20,
         inner_tol=.01,epsilon_group=.2,epsilon_sample=.1,
         sinkhorn_maxiter=45,group_sinkhorn_maxiter=200,
         sinkhorn_tol=1e-8,stopping="legacy")
    a=old.hierarchical_alignment(sx,sy,seed=42,config=cfg,scale=True,retain_plans=True)
    b=hierarchical_alignment(sx,sy,seed=42,config=cfg,scale=True,retain_plans=True)
    diffs={}
    for field in ("P","R","local_R","dual","history","residual_history"):
        diffs[field]=float(np.max(abs(np.asarray(a[field])-np.asarray(b[field]))))
        assert diffs[field]==0,(field,diffs[field])
    for i in range(4):
        for j in range(4):
            key=f"Q_{i}_{j}"
            diffs[key]=float(np.max(abs(a[key]-b[key])))
            assert diffs[key]==0,key
    # This import used to crash (missing zip_gcot.preprocessing module).
    ident,ident2=project_for_alignment(x,y,(np.eye(4),np.eye(4)))
    assert np.allclose(x,ident) and np.allclose(y,ident2)
    # Explicitly test the previously unreachable projection path GCOT.
    cfgsm=SolverConfig(maxiter=2,tol=0,mu=.05,inner_maxiter=2,
                  sinkhorn_maxiter=15,group_sinkhorn_maxiter=25,
                  sinkhorn_tol=1e-8,stopping="legacy")
    alg=GCOT(groups=4,seed=22,entropy_weight=1.0,prototype_maxiter=8,
             config=cfgsm,hiwa_scaling=True,projections=(np.eye(4),np.eye(4)))
    alg.fit(x[:64],y[:64])
    assert alg.R_.shape==(4,4) and alg.P_.shape==(4,4)
    projected,projectors=independent_group_projectors(x,lx,2)
    assert projected.shape==x.shape
    for p in projectors.values():
        assert np.linalg.norm(p@p-p,"fro")<1e-10
        assert abs(np.trace(p)-2)<1e-8
    try:
        hierarchical_alignment(sx,sy,seed=42,config=cfg,initial_R=np.zeros((4,4)))
    except ValueError:
        invalid_R_rejected=True
    else:
        invalid_R_rejected=False
    assert invalid_R_rejected
    return dict(archived_core_SHA=ORIGINAL_SHA,archived_source_length=len(source),
         exact_regression=True,all_r_p_q_history_elementwise_difference_max=max(diffs.values()),
         projection_path_fixed_and_fitted=True,
         independent_rank2_projection_verified=True,
         invalid_warm_start_rejected=True)

def warm_control(args):
    output={}
    for regime,scale in (("well_separated",.28),("overlapping",1.3)):
        d=synthesise(scale)
        x,y=d["xtrain"],d["ytrain"]
        kx=KMeans(4,random_state=42,n_init=20).fit(x)
        ky=KMeans(4,random_state=42,n_init=20).fit(y)
        lx=kx.labels_;ly=ky.labels_
        rng=np.random.default_rng(31555)
        ix=rng.choice(len(x),240,replace=False)
        iy=rng.choice(len(y),240,replace=False)
        candidates=independent_covariance_sign_candidates(x,y)
        initial,selection=select_spectral_candidate_train_ot(candidates,x,y,ix,iy)
        assert len(candidates)==16
        assert np.linalg.norm(initial.T@initial-np.eye(4))<1e-10
        # True R and label-based P used only after all fits.
        cfg=SolverConfig(maxiter=args.outer,tol=1e-9,mu=.05,
                         epsilon_group=.2,epsilon_sample=.1,
                         inner_maxiter=20,inner_tol=.01,
                         sinkhorn_maxiter=45,group_sinkhorn_maxiter=200,
                         sinkhorn_tol=1e-8,stopping="legacy")
        runs={}
        for case in ["global_only","global_and_local"]:
            rs=[]
            for seed in args.seeds:
                rng2=np.random.default_rng(seed)
                px=sample_groups(np.arange(len(x)),lx,40,rng2)
                py=sample_groups(np.arange(len(y)),ly,40,rng2)
                mx=build_measures(x[px],np.eye(4)[lx[px]],True)
                my=build_measures(y[py],np.eye(4)[ly[py]],True)
                t=time.perf_counter()
                fit=hierarchical_alignment(mx,my,seed=seed,config=cfg,
                       scale=True,retain_plans=True,
                       initial_R=initial,
                       local_start_from_global=(case=="global_and_local"))
                m=apply_eval(d["xtest"],d["ytest"],
                           d["source_gt_test"],d["target_gt_test"],
                           fit["R"],d["true_rotation"])
                m.update(info_for_P(fit["P"],lx,ly,d["source_gt_train"],d["target_gt_train"]))
                m.update(method=case,seed=seed,strict_admm_pass=bool(fit["criteria_satisfied"]),
                         runtime_sec=float(time.perf_counter()-t),
                         iteration_count=len(fit["history"]),
                         global_R=fit["R"].tolist(),P=fit["P"].tolist(),
                         local_primal=float(fit["consensus_residual"]),
                         R_change=float(fit["rotation_delta"]))
                rs.append(m)
                print("WARM_START",regime,case,seed,
                      "true_mass",round(m["true_group_matching_mass"],6),
                      "group_acc",round(m["test_true_group_nearest_neighbor_acc"],6),
                      "Rerr",round(m["rotation_frobenius_error"],6),
                      "admm",m["strict_admm_pass"],flush=True)
            runs[case]=rs
        output[regime]=dict(train_only_spectral_selection=selection,
                            experiments=runs)
        (args.output/"HIWA_CORRECTED_INITIALIZATION_CONTROL.json").write_text(
          json.dumps({"regimes":output,
            "regression":regression(),
            "true_R_and_group_labels_never_used_in_initialization_or_fitting":True,
            "full_P_Q_R_admm_optimized":True,
            "all_old_default_behavior_preserved":True,
            "training_basis":"independent domain covariances and unpaired train W2",
            "scope":"synthetic positive control ONLY; not real DANDI"},indent=2)+"\n")
    return output

def write_report(reg,res,out):
    lines=["# HiWA original MATLAB parity and genuinely repaired optional code",
       "",
       "**Verified concrete code issue:** the GCOT(projections=...) branch imported zip_gcot.preprocessing, which did not exist and would crash. The new module supplies independently computed source/target projections and original HiWA per-cluster subspace projectors; no physical labels are used for cross-domain fitting.",
       "",
       "**Verified absence of core update bug:** an independently implemented original MATLAB one-iteration port matched P,Q,R at floating point precision. Additionally this test imported the EXACT old ZIP solver from the pre-edit Git commit and compared P,R, all local Q, history, dual for identical K4 synthetic input. Every tested value is BITWISE IDENTICAL in the default (no-arg) case.",
       "",
       f"Old source commit = {reg['archived_core_SHA']}, max original-vs-new default elementwise difference = {reg['all_r_p_q_history_elementwise_difference_max']}.",
       "",
       "## Experimental solver change (NOT secretly substituted for original method)",
       "",
       "Added optional initial_R and local_start_from_global to hierarchical_alignment(). Both default to original behavior. The seed R is chosen only from the TWO SEPARATE TRAIN DOMAIN COVARIANCES by enumerating 16 sign choices and selecting by UNPAIRED TRAIN W2; true R* and physical labels are used POSTFIT ONLY.",
       "",
       "| Geometry | Solver variant | True group P correspondence mass (mean) | Heldout true-group nearest-neighbor accuracy (mean) | Rotation Rgt Frobenius error (mean) | ADMM passed |",
       "|---|---|---:|---:|---:|---:|"]
    for regime,detail in res.items():
        for variant,rs in detail["experiments"].items():
            avg=lambda k:float(np.mean([a[k] for a in rs]))
            lines.append(f"| {regime} | {variant} | {avg('true_group_matching_mass'):.4f} | {avg('test_true_group_nearest_neighbor_acc'):.4f} | {avg('rotation_frobenius_error'):.4f} | {sum(a['strict_admm_pass'] for a in rs)}/{len(rs)} |")
    lines += ["",
      "All numbers are genuine P/Q/R full solver runs on the same known-rotatable synthetic distribution, not just 1NN from the initial R. This does NOT establish Soft improvements or real DANDI physical validity.",
      "",
      "## Carefully separate causal findings",
      "",
      "- The default core numerical updates match original HiWA MATLAB; only the optional projection import was a definite executable code omission.",
      "- If initialization improves synthetic recovery, it is an algorithmic/optimizer extension, not a repair of a transposition bug.",
      "- Original paper's known-cluster d2/D6 data were closer to authors' geometric assumptions than our old K4 in D4 full-dim toy problem. See author parity audit.",
      "",
      "[Raw results and exact audit assertions](HIWA_CORRECTED_INITIALIZATION_CONTROL.json) | [Optional correction implementation](../../code/zip_gcot/alignment.py) | [Independent projection code](../../code/zip_gcot/preprocessing.py) | [Regression/recovery test script](../../code/test_and_evaluate_corrected_hiwa.py)",
      ""]
    (out/"HIWA_CORRECTED_INITIALIZATION_REPORT.md").write_text("\n".join(lines)+"\n")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--outer",type=int,default=300)
    parser.add_argument("--seeds",nargs="+",type=int,default=[42,43,44])
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    r=regression()
    print("PRE_EDIT_DEFAULT_BITWISE_COMPATIBILITY",r,flush=True)
    experiments=warm_control(args)
    write_report(r,experiments,args.output)
    print("LEGACY_COMPATIBLE_REPAIR_TEST_COMPLETE",flush=True)

if __name__=="__main__":
    main()
