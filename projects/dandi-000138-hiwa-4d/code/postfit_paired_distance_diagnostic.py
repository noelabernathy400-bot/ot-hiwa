"""POST-FIT-ONLY paired distance preservation diagnostic; never train on pairs.

If Y_i ≈ R X_i for an orthogonal R, then ||Y_i-Y_j|| ≈ ||X_i-X_j||
for any pair of synchronized physical windows i,j. Evaluate this necessary
but not sufficient condition on original 3500 TEST windows only, with
original target/three train-only transformed targets. Use different-trial
pairs and no model selection, no fitting from paired correspondences.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from run_target_only_geometry_ablation import target_transform

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--clusters",type=Path,required=True)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    with np.load(a.clusters,allow_pickle=False) as z:
        assert "shared_labels" not in z.files
        X=np.asarray(z["neural4"],float)
        Y=np.asarray(z["hand4"],float)
        train=np.asarray(z["train_mask"],bool)
        test=np.asarray(z["test_mask"],bool)
        trials=np.asarray(z["trial_of_window"],int)
    base=json.loads(a.baseline.read_text())
    record=json.loads(a.results.read_text())
    assert base["no_synchronized_pair_ids_used_in_fitting"]
    assert record["no_test_hyperparameter_selection"]
    assert set(record["variants"])=={"identity","block_zca","full_zca","arcsinh"}
    x=(X-np.asarray(base["source_mean"]))/base["source_radius"]
    y=(Y-np.asarray(base["target_mean"]))/base["target_radius"]
    xx=x[test];trial=trials[test]
    n=len(xx);assert n==3500
    rng=np.random.default_rng(24680246)
    aa=rng.integers(n,size=110000)
    bb=rng.integers(n,size=110000)
    good=(trial[aa]!=trial[bb])
    aa=aa[good][:100000];bb=bb[good][:100000]
    assert len(aa)>90000
    dx=np.linalg.norm(xx[aa]-xx[bb],axis=1)
    fig,axes=plt.subplots(2,2,figsize=(12,10),constrained_layout=True)
    results={}
    for ax,mode in zip(axes.flat,record["variants"]):
        yy,inv,det=target_transform(y,train,mode)
        yy=yy[test]
        dy=np.linalg.norm(yy[aa]-yy[bb],axis=1)
        pearson=float(np.corrcoef(dx,dy)[0,1])
        spear=float(spearmanr(dx,dy).statistic)
        stress=float(np.sqrt(np.mean((dx-dy)**2)/np.mean(dy**2)))
        bins=ax.hexbin(dx,dy,gridsize=65,mincnt=1,cmap="viridis")
        tmax=float(max(np.quantile(dx,.997),np.quantile(dy,.997)))
        ax.plot([0,tmax],[0,tmax],linestyle="--",color="tomato",lw=1.2,alpha=.8)
        ax.set(xlabel="d_X(i,j) between two synchronized test indices",
               ylabel=f"d_T(Y)(i,j) | {mode}",
               xlim=(0,tmax),ylim=(0,tmax),
               title=f"{mode}: Pearson={pearson:.3f} Spearman={spear:.3f} stress={stress:.3f}")
        fig.colorbar(bins,ax=ax,label="Number of cross-trial sample pairs",shrink=.82)
        results[mode]=dict(Pearson_distance_corr=pearson,
            Spearman_distance_corr=spear,relative_distance_stress=stress)
    fig.suptitle("Postfit diagnostic ONLY: does TRUE synchronized pairing preserve 4D distances?\n"
                 "100k random different-trial index pairs from original 3500 heldout samples, NOT used in optimization",
                 fontsize=13)
    fig.savefig(a.output/"POSTFIT_ONLY_PAIRED_DISTANCE_PRESERVATION.png",
                dpi=195,bbox_inches="tight")
    plt.close(fig)
    audit={"postfit_evaluation_only":True,
           "used_for_any_grouping_training_or_parameter_selection":False,
           "num_heldout_windows":n,
           "comparison":"Rotation-invariant necessary condition for original synchronized rigid map",
           "pair_type":"randomly sampled different-trial heldout window-index pairs",
           "pair_samples":len(aa),
           "all_four_modes":results,
           "caution":"Overlapping samples not independent; no p-values. High distance correlation necessary but not sufficient for orthogonal pointwise mapping. Low correlation does not disprove any nonlinear correspondence."}
    (a.output/"POSTFIT_DISTANCE_GEOMETRY_AUDIT.json").write_text(
        json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("POSTFIT PAIRED DISTANCE AUDIT",json.dumps(results,indent=2),flush=True)
if __name__=="__main__":
    main()
