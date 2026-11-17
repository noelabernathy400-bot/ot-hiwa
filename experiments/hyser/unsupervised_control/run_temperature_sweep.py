#!/usr/bin/env python3
"""Unpaired Soft K-means temperature sweep using shared exact empirical W2².

Source and target separately fit K-means with seeds seed and seed+1;
then convert their distances to independent soft memberships.
No true pair indices, labels, or source-to-target copied membership enter fit.
Same K=4, mu=.05, eps_sample=.1, eps_group=.2, 180 outer max.

  python experiments/hyser/unsupervised_control/run_temperature_sweep.py \
    --data experiments/hyser/data/paired_full_precision.npz \
    --out experiments/hyser/unsupervised_control/temperature_rerun.json

The fit implementation is a local GC-OT mathematical port with high precision
4x4 entropic dual solver, NOT a byte-level run of the uploaded original repo.
Only final independent uniform point-cloud W2² is used as the primary score.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from run_independent import fit
from evaluate_common_w2 import common_w2_sq


def memberships_independent(X, Y, tau, seed):
    source_centers = KMeans(4, n_init=30, random_state=seed).fit(X).cluster_centers_
    target_centers = KMeans(4, n_init=30, random_state=seed+1).fit(Y).cluster_centers_
    A = softmax(-cdist(X, source_centers, "sqeuclidean") / tau, axis=1)
    B = softmax(-cdist(Y, target_centers, "sqeuclidean") / tau, axis=1)
    return A, B


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seeds",type=int,nargs="+",default=[40,41,42,43,44])
    p.add_argument("--taus",type=float,nargs="+",default=[.25,1.0])
    args=p.parse_args()
    with np.load(args.data,allow_pickle=False) as z:
        # The file can also contain pairing labels, but NEVER access them here.
        X=np.asarray(z["X"],dtype=float)
        Y=np.asarray(z["Y"],dtype=float)
    if X.shape!=Y.shape or X.shape!=(250,3):
        raise ValueError("Expected real Hyser full precision 250x3")
    before=common_w2_sq(X,Y,np.eye(3))
    out={
      "protocol":"Independent X/Y grouping; no true pairing in fit or metric",
      "common_w2_before":before,
      "mu":.05,"eps_sample":.1,"eps_group":.2,
      "outer_maxiter":180,"local_maxiter":100,
      "numeric_tolerance":{"rotation_change":.01,"consensus":.01,"marginal":1e-6},
      "runs":[]
    }
    args.out.parent.mkdir(parents=True,exist_ok=True)
    for tau in args.taus:
        if tau<=0:raise ValueError("Temperature must be positive")
        for seed in args.seeds:
            A,B=memberships_independent(X,Y,tau,seed)
            R,P,h,_,_=fit(X,Y,A,B,seed,mu=.05,maxiter=180)
            m=h[-1];cost=common_w2_sq(X,Y,R)
            rec={"tau":tau,"seed":seed,"R":R.tolist(),"P":P.tolist(),
                "iterations":len(h),"last":m,
                "passed_numeric_checks":bool(
                    m["rotation_delta"]<=.01 and m["consensus"]<=.01
                    and m["marginal"]<=1e-6),
                "W2_after":cost,
                "W2_reduction_pct":100*(before-cost)/before}
            out["runs"].append(rec)
            args.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
            print("tau",tau,"seed",seed,"W2",cost,"checked",rec["passed_numeric_checks"],flush=True)
    print("Saved",args.out)


if __name__=="__main__":
    main()
