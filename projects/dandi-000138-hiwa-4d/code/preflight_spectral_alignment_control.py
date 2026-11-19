"""Source/target marginal-only covariance orientation + exhaustive sign initialization.

No true rotation, paired samples or physical group labels enter candidate
selection. Uses synthetic generator from prior positive control. Evaluates
whether an easy known-rotation problem was not identifiable or the random
Hard/Soft P-Q-R solver simply found poor local optima.
"""
from __future__ import annotations
import argparse,itertools,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from preflight_synthetic_positive_control import synthesise

def candidate_from_cov(A,B):
    _,U=np.linalg.eigh(np.cov(A,rowvar=False))
    _,V=np.linalg.eigh(np.cov(B,rowvar=False))
    U=U[:,::-1];V=V[:,::-1]
    return [(v,V@np.diag(v)@U.T) for v in itertools.product((-1,1),repeat=A.shape[1])]

def measure(X,Y,R):
    D=cdist(X@R.T,Y,"sqeuclidean")
    a,b=linear_sum_assignment(D)
    return float(np.mean(D[a,b]))

def test(d,R):
    xx=d["xtest"];yy=d["ytest"]
    match=cKDTree(yy).query(xx@R.T,k=1)[1]
    return float(np.mean(d["source_gt_test"]==d["target_gt_test"][match]))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--positive-control-results",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    old=json.loads(a.positive_control_results.read_text())
    rng=np.random.default_rng(31555)
    out=dict(training_uses_synchronized_pairing=False,
       training_uses_true_rotation=False,
       labels_used_only_postfit=True,
       candidates="16 signs of independently train-fitted source-target covariance eigenbasis orthogonal mapping",
       candidate_selection="minimum exact Hungarian empirical W2sq on 240+240 UNPAIRED train samples",
       old_solver_selected_by_same_unsupervised_240x240_training_W2_not_test_labels=True,
       regimes={})
    for name,scale in [("well_separated",.28),("overlapping",1.3)]:
        d=synthesise(scale)
        X,Y=d["xtrain"],d["ytrain"]
        ix=rng.choice(len(X),240,replace=False)
        iy=rng.choice(len(Y),240,replace=False)
        candidates=candidate_from_cov(X,Y)
        costs=np.array([measure(X[ix],Y[iy],R) for _,R in candidates])
        chosen=int(np.argmin(costs))
        sign,R=candidates[chosen]
        truth_R=d["true_rotation"]
        base=[0.,0.]
        identity=np.eye(4)
        base[0]=measure(X[ix],Y[iy],identity)
        base[1]=test(d,identity)
        self_scores=dict(sign=list(sign),training_W2sq=float(costs[chosen]),
                         heldout_group_1NN_accuracy=test(d,R),
                         heldout_Rgt_Frob_error=float(np.linalg.norm(R-truth_R,"fro")),
                         all_sign_candidate_train_W2sq=costs.tolist())
        gt=dict(train_W2sq=measure(X[ix],Y[iy],truth_R),
                test_group_accuracy=test(d,truth_R))
        prev={}
        for method,records in old["regimes"][name]["runs"].items():
            scores=[]
            for trial in records:
                RR=np.asarray(trial["R"])
                scores.append(dict(seed=trial["seed"],
                   train_W2sq=measure(X[ix],Y[iy],RR),
                   test_group_accuracy=test(d,RR),
                   rotation_error=float(np.linalg.norm(RR-truth_R,"fro"))))
            prev[method]=dict(select_by_lowest_train_W2sq=sorted(scores,key=lambda z:z["train_W2sq"])[0],
                              all_original_solver_runs=scores)
        out["regimes"][name]=dict(eigen_sign_alignment=self_scores,
                    true_rotation_oracle_POSTFIT_ONLY=gt,
                    identity_baseline=dict(train_W2sq=base[0],test_group_accuracy=base[1]),
                    original_Hard_Soft_solver=prev)
        print(f"SPECTRAL {name}: best_train_W2={costs[chosen]:.4f} "
              f"oracle_train_W2={gt['train_W2sq']:.4f} "
              f"heldout_group_acc={self_scores['heldout_group_1NN_accuracy']:.4f} "
              f"oracle_heldout_group_acc={gt['test_group_accuracy']:.4f}",flush=True)
    (a.output/"SPECTRAL_GEOMETRY_POSITIVE_CONTROL.json").write_text(
      json.dumps(out,indent=2)+"\n")
    f,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
    for ax,(name,rec) in zip(axes,out["regimes"].items()):
        items=[
         ("Identity",rec["identity_baseline"]["test_group_accuracy"]),
         ("Hard train-select",rec["original_Hard_Soft_solver"]["hard"]["select_by_lowest_train_W2sq"]["test_group_accuracy"]),
         ("Soft train-select",rec["original_Hard_Soft_solver"]["soft"]["select_by_lowest_train_W2sq"]["test_group_accuracy"]),
         ("Spectral 16-sign",rec["eigen_sign_alignment"]["heldout_group_1NN_accuracy"]),
         ("Oracle R*",rec["true_rotation_oracle_POSTFIT_ONLY"]["test_group_accuracy"])]
        ax.bar(range(len(items)),[t[1] for t in items],
               color=["#999999","#2a78ad","#d79135","#529a72","#333333"])
        ax.set(xticks=range(len(items)),xticklabels=[v[0] for v in items],
               ylim=(0,1.1),ylabel="True 4-group transfer accuracy, heldout",
               title=name)
        ax.tick_params(axis="x",rotation=18)
        ax.axhline(.25,color="black",linestyle="--",alpha=.4)
    f.suptitle("Known rotation exists: does unpaired 4D geometry identify the rotation?\n"
        "Selection by TRAIN empirical OT only; all physical labels and true R* used POSTFIT",fontsize=12)
    f.savefig(a.output/"SPECTRAL_VS_RANDOM_SOLVER_POSITIVE_CONTROL.png",
              dpi=180,bbox_inches="tight")
    plt.close(f)
    lines=["# Preflight unpaired covariance sign search: isolate geometry identifiability vs solver initialization",
      "",
      "Original 4-cluster synthetic positive control with known 4D true R*; neither R* nor any true group labels or per-point correspondence are used for training. Both domains independently sampled from common mixtures.",
      "",
      "We calculate covariance principal eigenvectors separately on X,Y TRAIN-only, enumerate 16 sign choices, select one by the MINIMUM 240×240 unpaired training empirical squared Wasserstein assignment cost, then evaluate the heldout true-group labels. This search is not the HiWA solver and never uses P/Q.",
      "",
      "| Simulation | Procedure | Train unpaired W2²↓ | Heldout true 4-group 1NN accuracy↑ |",
      "|---|---|---:|---:|"]
    for name,regime in out["regimes"].items():
        for title,rr in [
            ("Identity",regime["identity_baseline"]),
            ("Hard random solver, selected train-only",regime["original_Hard_Soft_solver"]["hard"]["select_by_lowest_train_W2sq"]),
            ("Soft random solver, selected train-only",regime["original_Hard_Soft_solver"]["soft"]["select_by_lowest_train_W2sq"]),
            ("Covariance eigen-sign, train-only",regime["eigen_sign_alignment"]),
            ("True R* oracle (evaluation only)",regime["true_rotation_oracle_POSTFIT_ONLY"])]:
            vv=rr.get("test_group_accuracy",rr.get("heldout_group_1NN_accuracy"))
            ww=rr.get("train_W2sq",rr.get("training_W2sq"))
            lines.append(f"| {name} | {title} | {ww:.6f} | {vv:.4f} |")
    lines+=["",
      "![Spectral-init and original random-seed solver](SPECTRAL_VS_RANDOM_SOLVER_POSITIVE_CONTROL.png)",
      "",
      "If train-only spectral orientation recovers physical truth but original solver misses it, optimization/initialization is implicated, rather than absence of shared orthogonal geometry. If neither recovers truth, this experiment alone cannot separate noisy eigenbasis estimation from intrinsic geometry ambiguity.",
      "",
      "[All 16 candidates and original seed train-only selection](SPECTRAL_GEOMETRY_POSITIVE_CONTROL.json) | [Code](../../code/preflight_spectral_alignment_control.py)",
      ""]
    (a.output/"SPECTRAL_GEOMETRY_POSITIVE_CONTROL_REPORT.md").write_text("\n".join(lines))
    print("SPECTRAL_CONTROL_COMPLETE",flush=True)
if __name__=="__main__":
    main()
