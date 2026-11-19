#!/usr/bin/env python3
"""Consolidate every exact local P/Q/Rkl ADMM Hyser K4 full-training run.

The main table has strict paired qualification: BOTH Hard and Soft runs on
the same feature representation, heldout trial and seed must pass:
- 1000 x 1000 full point-support conditional Q (16 blocks each)
- group P and conditional Q marginals
- global/local rotation orthogonality and ADMM consensus
- frozen cutoff max(||local_R-Rglobal||, ||Rnew-Rprev||) <= 0.1
Only eligible matched runs are used for an interpretable algorithm comparison.

Results for FAILED runs are retained, not hidden. A missing job does not get
filled in from old/global-R experiments.
"""
from __future__ import annotations
import argparse,json,csv
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURES=("PCA5_RAW5","FA5_RAW5")
METHODS=("hard","soft05")
FOLDS=tuple(range(1,6))
SEEDS=(42,43)
KEYS=(
"heldout_W2_squared",
"train_bank_inductive_nearest.train_bank_NN_physically_correct",
"train_bank_inductive_nearest.train_bank_NN_geometric_coverage",
"train_bank_inductive_nearest.train_bank_random_hit_mean",
"nearest.nn_retrieval_physical_neighbor_fraction",
"nearest.target_nn_coverage_fraction",
"nearest.nn_retrieval_exact_time_index_fraction",
"heldout_physical_force_prediction_POSTHOC.paired_force_RMSE_EVAL_ONLY",
"heldout_physical_force_prediction_POSTHOC.paired_force_R2_vs_trainmean_EVAL_ONLY",
"membership_source_mean_entropy","membership_target_mean_entropy",
"source_mixed_fraction_max_membership_below_0p8",
"target_mixed_fraction_max_membership_below_0p8",
"P_mean_row_max",
)

def retrieve(obj,path):
    for key in path.split("."):obj=obj[key]
    return float(obj)
def average(v):
    v=np.asarray(v,float)
    return float(v.mean()) if len(v)>0 else None
def flatten(out,keys):
    return [retrieve(out,k) for k in keys]
def save_csv(path,rows):
    if not rows:return
    fieldnames=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",encoding="utf-8",newline="") as fp:
        d=csv.DictWriter(fp,fieldnames=fieldnames);d.writeheader();d.writerows(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--artifacts-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    z=p.parse_args();z.out.mkdir(exist_ok=True,parents=True)
    files=sorted(z.artifacts_dir.rglob("local_ADMM_results.json"))
    runs=[];issues=[]
    for file in files:
        try:
            v=json.loads(file.read_text())["runs"]
            runs.extend(v)
        except Exception as err:
            issues.append("UNREADABLE "+str(file)+" "+str(err))
    index=defaultdict(list)
    for x in runs:
        k=(x["config"],x["heldout_trial"],x["seed"],x["method"])
        index[k].append(x)
        if x.get("n_OT_train_source")!=1000 or x.get("n_OT_train_target")!=1000:
            issues.append("NOT FULL 1000 "+str(k))
    expected={(c,f,s,m) for c in FEATURES for f in FOLDS for s in SEEDS for m in METHODS}
    observed=set(index)
    missing=sorted(expected-observed)
    duplicate=[k for k,v in index.items() if len(v)!=1]
    extra=sorted(observed-expected)
    if missing:issues.append("MISSING "+str(missing))
    if duplicate:issues.append("DUPLICATE "+str(duplicate))
    if extra:issues.append("EXTRA "+str(extra))
    def qualified(r):
        try:
            P=np.asarray(r["P"])
            R=np.asarray(r["R"])
            qs=r["q_stats"]
            par=r["params"]
            return (r["Q_constraints_passed"] and
                    r["numeric_convergence_ADMM_primal_and_rotation"] and
                    len(qs)==16 and all(x["shape"]==[1000,1000] for x in qs) and
                    np.isfinite(R).all() and
                    np.linalg.norm(R@R.T-np.eye(5))<1e-7 and
                    max(abs(P.sum(axis=0)-.25).max(),
                        abs(P.sum(axis=1)-.25).max())<1e-7 and
                    par["mu"]==.05 and par["support_mode"]=="full" and
                    par["sa_shorn_gamma"]==.1 and par["shorn_gamma"]==.2 and
                    par["warm_start_local"] and
                    r["heldout_trial"] in FOLDS and
                    r["seed"] in SEEDS)
        except (KeyError,TypeError,ValueError):
            return False
    typed=[]
    for r in runs:
        q=qualified(r)
        row={
            "feature":r["config"],"fold":r["heldout_trial"],
            "seed":r["seed"],"method":r["method"],
            "qualified":q,
            "admm_primal_final":r["diagnostics"]["admm_primal_residual"]["last"],
            "admm_R_change_final":r["diagnostics"]["Rg_norm"]["last"],
            "max_Q_marginal_error":r["max_final_Q_marginal_error"],
            "max_Rkl_orthogonality_error":r["max_Rkl_orthogonality_error"],
            **{key.replace(".","__"):retrieve(r,key) for key in KEYS}
        }
        typed.append(row)
    save_csv(z.out/"full1000_all40_per_run.csv",typed)

    paired=[]
    for c in FEATURES:
        for f in FOLDS:
            for s in SEEDS:
                h=index.get((c,f,s,"hard"),[])
                b=index.get((c,f,s,"soft05"),[])
                if len(h)!=1 or len(b)!=1:
                    continue
                hard,soft=h[0],b[0]
                ok=qualified(hard) and qualified(soft)
                paired.append({
                    "feature":c,"heldout_trial":f,"seed":s,
                    "both_solver_converged":ok,
                    "hard_converged":qualified(hard),"soft_converged":qualified(soft),
                    **{"soft_minus_hard__"+key.replace(".","__"):
                       retrieve(soft,key)-retrieve(hard,key) for key in KEYS},
                })
    save_csv(z.out/"matched_hard_soft_pairwise_results.csv",paired)
    summary=[]
    for c in FEATURES:
        for m in METHODS:
            rr=[r for r in runs if r["config"]==c and r["method"]==m]
            good=[r for r in rr if qualified(r)]
            row={"feature":c,"method":m,
                 "n_expected":len(FOLDS)*len(SEEDS),
                 "n_observed":len(rr),"n_converged":len(good),
                 "max_final_primal":max((r["diagnostics"]["admm_primal_residual"]["last"] for r in rr),default=None)}
            for key in KEYS:
                row[key.replace(".","__")+"_all_descriptive"]=average([retrieve(r,key) for r in rr])
                row[key.replace(".","__")+"_qualified_only"]=average([retrieve(r,key) for r in good])
            summary.append(row)
    (z.out/"full1000_summary.json").write_text(json.dumps({
       "expectation":"40 full-data runs; five trials x two seeds x two features x Hard/Soft",
       "runs_observed":len(runs),"runs_expected":40,
       "eligible_matched_pairs":sum(x["both_solver_converged"] for x in paired),
       "paired_expected":20,
       "missing":missing,"duplicates":duplicate,"extras":extra,"issues":issues,
       "full1000_means":summary,
       "qualified_matches":paired,
       "status":("COMPLETE_ALL_CONVERGED" if len(runs)==40 and
                  len(paired)==20 and not issues and
                  all(x["both_solver_converged"] for x in paired)
                  else "PARTIAL_OR_UNCONVERGED")
    },ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    rows=[]
    for c in FEATURES:
        rr=[x for x in paired if x["feature"]==c and x["both_solver_converged"]]
        row={"feature":c,"qualified_pairs":len(rr)}
        for key in KEYS:
            k="soft_minus_hard__"+key.replace(".","__")
            row[k]=average([x[k] for x in rr])
        rows.append(row)
    save_csv(z.out/"only_solver_qualified_paired_effects.csv",rows)
    # A residual plot is a numerical audit, not a 2D scatter projection.
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    for ax,feature in zip(axes,FEATURES):
        for method in METHODS:
            hist=[r["ADMM_iteration_history"]["primal_residual"]
                  for r in runs if r["config"]==feature and r["method"]==method
                  and "ADMM_iteration_history" in r]
            if len(hist)==0:continue
            minlen=min(len(x) for x in hist)
            values=np.array([x[:minlen] for x in hist])
            med=np.median(values,axis=0)
            ax.plot(np.arange(1,minlen+1),med,label=method)
        ax.axhline(.1,ls="--",color="#666",label="predeclared threshold 0.1")
        ax.set(title=feature,xlabel="ADMM outer iteration",
               ylabel="median primal residual, observed runs",ylim=(0,None))
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("Full-1000 local Rkl consensus numeric convergence; no performance cherry-picking")
    fig.savefig(z.out/"ADMM_primal_convergence_diagnostic.png",dpi=165,bbox_inches="tight")
    plt.close(fig)
    md=["# Hyser full 1000-point per-domain Hard/Soft K=4 P/Q/Rkl ADMM formal benchmark",
        "",
        "Study: Hyser S01 S1 Random Task, 5 actual trials x 250 time windows, 5-fold leave-one-trial-out.",
        "X=256-channel EMG mapped to 5D PCA or FA (train-only); Y=all 5 actual finger forces whitened train-only, with physical channels retained.",
        "Train 1000 source + 1000 target windows, separately shuffled, no synchronized row pair used to fit R/P/Q. Holdout intact 250 windows.",
        "K4 both domains independently; Hard one-hot vs Soft τ=0.5; dense 1000x1000 local Q for all 16 block pairs, local Rkl, P, ADMM; no anchors.",
        "Original author default μ=.005 did NOT numerically converge on Hyser after 300 iterations. The predeclared numeric repair here uses *the same μ=.05 and local warm-start for BOTH methods*, selected using inner training-only ADMM primal/rotation convergence, not test success. The actual method remains the same local PQR/entropic constraints and γ1=.2, γ2=.1/P, max outer=300, inner=40, group Sinkhorn same mathematical objective computed stably.",
        "Fisher 2D subspaces are presentation only, selected by TRAIN pseudolabels; no 2D coordinate plane or 2D transport metric is used.",
        "",
        f"Number of full runs found: {len(runs)}/40. Eligible matched Hard/Soft seed-fold pairs: {sum(x['both_solver_converged'] for x in paired)}/20.",
        "Status: "+("COMPLETE_ALL_CONVERGED" if len(runs)==40 and len(paired)==20 and not issues and all(x["both_solver_converged"] for x in paired) else "PARTIAL_OR_UNCONVERGED"),
        "",
        "## All run observations (UNQUALIFIED rows are not valid model-performance claims)",
        "",
        "| Representation | Method | Observed/expected | Numerically converged | Heldout W2² (all descriptive) | Inductive target-train-bank physical NN (all descriptive) | Transductive test-bank NN (all descriptive) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        fmt=lambda x:f"{x:.5f}" if x is not None else "NA"
        md.append("|{}|{}|{}/{}|{}|{}|{}|{}|".format(r["feature"],r["method"],r["n_observed"],r["n_expected"],
            r["n_converged"],
            fmt(r["heldout_W2_squared_all_descriptive"]),
            fmt(r["train_bank_inductive_nearest__train_bank_NN_physically_correct_all_descriptive"]),
            fmt(r["nearest__nn_retrieval_physical_neighbor_fraction_all_descriptive"])))
    md+=["","## Matched paired effects on BOTH-converged runs (Soft - Hard, negative is better for errors)",
           "| Representation | eligible fold-seed pairs | Δ heldout W2² | Δ inductive physical NN correct fraction | Δ transductive physical NN correct fraction | Δ physical RMSE |",
           "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        fmt=lambda x:f"{x:+.6f}" if x is not None else "NA"
        md.append("|{}|{}|{}|{}|{}|{}|".format(row["feature"],row["qualified_pairs"],
           fmt(row["soft_minus_hard__heldout_W2_squared"]),
           fmt(row["soft_minus_hard__train_bank_inductive_nearest__train_bank_NN_physically_correct"]),
           fmt(row["soft_minus_hard__nearest__nn_retrieval_physical_neighbor_fraction"]),
           fmt(row["soft_minus_hard__heldout_physical_force_prediction_POSTHOC__paired_force_RMSE_EVAL_ONLY"])))
    md+=["","## Scientific inference restrictions",
         "- If any paired runs do not meet both solver residuals, full-sample formal comparison is numerically INCONCLUSIVE. Never report cherry-picked converged subsets as if they represented five folds.",
         "- Training target-bank retrieval uses only Y training reference cloud; transductive retrieval uses the heldout Y cloud but not its row-to-row pairing in training; NN correctness uses true synchronized Y only after training.",
         "- The NN radius is fixed by the 5th percentile of unrelated target TRAIN inter-trial distances; coverage under this radius is not successful physical prediction.",
         "- Full sample local Rkl/Qkl solver is a numerically stabilized, train-only feature-initialized version of the repository SoftHiWA, not a byte-for-byte rerun of the original NeurIPS PyHiWA implementation.",
         "- Some data may not support common orthogonal alignment even if soft grouping helps distribution W2. A lack of convergence can be a model-mismatch signal, not just compute precision.",
         "- Only compare Soft-vs-Hard WITHIN the same FA/PCA embedding. Comparing absolute cross-feature latent-space W2 is invalid.",
         "- Five folds from one person/session, not five independent people: no claim of population significance.",
         "- Individual time-window force regression, retrieval and distribution alignment remain distinct scientific tasks."
         ]
    if issues:md+=["","## Submission/logging issues",*["- "+q for q in issues]]
    (z.out/"FORMAL_COMPARISON_REPORT_ZH.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    print(json.dumps({"runs_found":len(runs),"eligible_pairs":sum(x["both_solver_converged"] for x in paired),
                      "issues":issues,"summaries":summary},ensure_ascii=False),flush=True)
    # Report must persist even if incomplete; do not fail the artifact packaging stage.
if __name__=="__main__":main()
