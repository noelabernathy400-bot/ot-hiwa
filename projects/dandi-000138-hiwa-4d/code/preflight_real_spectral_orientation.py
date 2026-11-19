"""Real-DANDI preflight for 16 train-only marginal spectral sign orientations.

No P/Q training, no supervised calibration. Select best rotation only by
UNPAIRED subset TRAIN empirical W2sq. Score test physical retrieval only
AFTER orientation is frozen. Compare identical heldout candidate pools and
frozen original Hard/Soft rotations. Results are exploratory because the
heldout trials were previously inspected; NEVER use these scores to tune.
"""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from preflight_spectral_alignment_control import candidate_from_cov,measure
from preflight_geometry_semantics_retrieval import cross_trial_nearest,retrieval_metrics

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--clusters",type=Path,required=True)
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--clustering-report",type=Path,required=True)
    p.add_argument("--preflight",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    orig=json.loads(a.archive.read_text(encoding="utf8"))
    a0=json.loads(a.clustering_report.read_text(encoding="utf8"))
    prev=json.loads(a.preflight.read_text(encoding="utf8"))
    assert orig["no_synchronized_pair_ids_used_in_fitting"]
    assert prev["model_retraining_occurred"] is False
    assert orig["input_sha256"]==hashlib.sha256(a.clusters.read_bytes()).hexdigest()
    with np.load(a.clusters,allow_pickle=False) as z:
        x0=np.asarray(z["neural4"],float)
        y0=np.asarray(z["hand4"],float)
        train=np.asarray(z["train_mask"],bool)
        test=np.asarray(z["test_mask"],bool)
        tr=np.asarray(z["trial_of_window"],int)
        assert "shared_labels" not in z.files
    assert test.sum()==3500 and train.sum()==14000
    x=(x0-np.asarray(orig["source_mean"]))/orig["source_radius"]
    y=(y0-np.asarray(orig["target_mean"]))/orig["target_radius"]
    orig_phys=y0*np.asarray(a0["scalers"]["behavior_scale"])+np.asarray(a0["scalers"]["behavior_mean"])
    rng=np.random.default_rng(20261119)
    idx=np.flatnonzero(train)
    ix=rng.choice(idx,400,replace=False)
    iy=rng.choice(idx,400,replace=False)
    C=candidate_from_cov(x[train],y[train])
    costs=np.array([measure(x[ix],y[iy],R) for sign,R in C])
    choose=int(np.argmin(costs))
    signs,R=C[choose]
    xt=x[test];yt=y[test];trt=tr[test]
    paths=float(prev["preregistered_scoring_rules"]["train_median_700ms_path_length_native_units"])
    slow=float(prev["preregistered_scoring_rules"]["speed_minimum_train_20percent_quantile_native_units"])
    phys=orig_phys[test]
    tree=cKDTree(yt)
    def evalR(R):
        source=xt@R.T
        idx1=tree.query(source,k=1)[1]
        cross=cross_trial_nearest(source,yt,trt)
        return {"full_pool":retrieval_metrics(idx1,phys,slow,paths),
                "cross_trial_pool":retrieval_metrics(cross,phys,slow,paths)}
    spectral=evalR(R)
    methods={}
    for name in ("hard","soft"):
        method=[]
        for run in orig["runs"][name]:
            rot=np.asarray(run["R"])
            trainw2=measure(x[ix],y[iy],rot)
            method.append(dict(seed=run["seed"],train_W2sq=trainw2,strict_admm_pass=run["admm_criteria_passed"],
                       retrieval=evalR(rot)))
        methods[name]=method
    out=dict(fitting_uses_synchronized_pairs=False,
        training_selection_uses_test_labels=False,
        training_selection_uses_true_group_labels=False,
        selection="minimum exact 400x400 unpaired TRAIN Wasserstein assignment among 16 covariance sign candidate rotations",
        method_is_new_rigid_initializer_not_HiWA_or_Soft=False,
        strict_no_R_P_Q_refit=True,
        heldout_100_trials_previously_inspected=True,
        rotation_candidate_count=16,
        current_source_target_dimensions=4,
        position_thresholds_percent_train_path=[5,10,20],
        speed_angular_thresholds_degrees=[30,45],
        selected_signs=list(signs),
        selected_candidate_index=choose,
        selected_train_W2sq=float(costs[choose]),
        all_candidates_train_W2sq=costs.tolist(),
        spectral_source_target_retrieval=spectral,
        identity_baseline=prev["retrieval"]["baseline"],
        original_hard_soft_methods=methods)
    # Correct explicit flag, no claim of having actually refit HiWA.
    out["method_is_new_rigid_initializer_not_HiWA_or_Soft"]=True
    (a.output/"DANDI_TRAIN_ONLY_SPECTRAL_GEOMETRY_PREFLIGHT.json").write_text(
        json.dumps(out,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    entries=[("Random",prev["retrieval"]["baseline"]["cross_trial_pool"]["random"]),
             ("Identity",prev["retrieval"]["baseline"]["cross_trial_pool"]["identity"])]
    entries.append(("Spectral16",spectral["cross_trial_pool"]))
    entries.extend([(f"Hard{s['seed']}",s["retrieval"]["cross_trial_pool"]) for s in methods["hard"]])
    entries.extend([(f"Soft{s['seed']}",s["retrieval"]["cross_trial_pool"]) for s in methods["soft"]])
    fig,ax=plt.subplots(1,2,figsize=(16,5.2),constrained_layout=True)
    y0s=[v["pos_hit10pct"] for _,v in entries]
    y1s=[v["direction_hit30deg"] for _,v in entries]
    for a2,vals,title in zip(ax,[y0s,y1s],
              ["Position hit ≤10% train-trial path", "Speed direction error ≤30°"]):
        a2.bar(range(len(vals)),vals,color=["#aaaaaa","#555555","#298b72"]+
                            ["#3475ad"]*5+["#e58f37"]*5)
        a2.set(xticks=range(len(vals)),xticklabels=[k for k,_ in entries],
               ylabel="Heldout cross-trial physical hit fraction",title=title)
        a2.tick_params(axis="x",rotation=60)
        a2.grid(axis="y",alpha=.13)
    fig.suptitle("Real DANDI: covariance sign orientation selected by TRAIN-ONLY unpaired W2\n"
                 "No target physical labels or true pairs used in any orientation selection")
    fig.savefig(a.output/"DANDI_SPECTRAL_VS_HARD_SOFT_PHYSICAL_MATCHING.png",
                dpi=180,bbox_inches="tight")
    plt.close(fig)
    lines=[
      "# DANDI138: can train-only unpaired covariance orientation give a better starting rotation?",
      "",
      "**Fully read-only**; it does not refit or change R/P/Q in archived HiWA. All candidate orientation selection occurs on 400 independent source+target train observations per domain; test physical labels are consulted only after choosing one fixed sign candidate by train W2.",
      "",
      f"- 16 covariance-eigenaxis sign candidates; selected sign={list(signs)}, train W2²={costs[choose]:.6f}",
      "- This is a cheap initializer feasibility check, not a full hierarchical optimizer; it does not include P or Q. Must NEVER claim it is directly comparable to HiWA as a complete model.",
      "",
      "| Rotation / method | Heldout CROSS-TRIAL position hit10% | Speed direction hit30° |",
      "|---|---:|---:|"]
    for name,v in entries:
        lines.append(f"| {name} | {v['pos_hit10pct']:.4f} | {v['direction_hit30deg']:.4f} |")
    lines+=["",
      "![Cross-trial 3500-point physical retrieval](DANDI_SPECTRAL_VS_HARD_SOFT_PHYSICAL_MATCHING.png)",
      "",
      "The heldout trial set was already examined previously; do not use these observed scores to tune candidate families or select future hyperparameters. If spectral initialization is to be integrated, use strictly training-internal validation for algorithm decisions, then confirm on a new independent trial/session when possible.",
      "",
      "[Complete per-seed original solver and train-only candidate score audit](DANDI_TRAIN_ONLY_SPECTRAL_GEOMETRY_PREFLIGHT.json) | [Reproducible script](../../code/preflight_real_spectral_orientation.py)"
    ]
    (a.output/"DANDI_TRAIN_ONLY_SPECTRAL_GEOMETRY_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("DANDI_SPECTRAL_PREFLIGHT",json.dumps({
      "selected_train_W2sq":float(costs[choose]),
      "position_hit10pct_cross":spectral["cross_trial_pool"]["pos_hit10pct"],
      "speed_direction_hit30deg_cross":spectral["cross_trial_pool"]["direction_hit30deg"]}),flush=True)

if __name__=="__main__":
    main()
