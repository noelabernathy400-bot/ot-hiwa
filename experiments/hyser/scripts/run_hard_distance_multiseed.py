#!/usr/bin/env python3
"""Run original repo HiWA for 10 seeds and evaluate distance-based alignment.

Run from repo root after generating domains.npz/clusters.npz:
    python experiments/hyser/scripts/run_hard_distance_multiseed.py --out-dir experiments/hyser/output

UNSUPERVISED FIT: pair identifiers only used after fitting, for metrics + oracle.
Single trial last 5s is not an independent trial test.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes

from run_hard_original import whiten_train


def whitened(z):
    ax, wx=whiten_train(z["Xtrain"])
    ay, wy=whiten_train(z["Ytrain"])
    return {
        "Xtrain": (z["Xtrain"]-ax)@wx, "Ytrain": (z["Ytrain"]-ay)@wy,
        "Xtest": (z["Xtest"]-ax)@wx, "Ytest": (z["Ytest"]-ay)@wy
    }


def true_indices(x_ids,y_ids):
    mapping={s:i for i,s in enumerate(y_ids)}
    return np.array([mapping[v] for v in x_ids])


def distances(x,y,indices,R=None):
    moved=x if R is None else x@R.T
    counterpart=y[indices]
    d=np.linalg.norm(moved-counterpart,axis=1)
    candidates=np.linalg.norm(moved[:,None,:]-y[None,:,:],axis=2)
    rank=np.argsort(candidates,axis=1)
    random=np.random.default_rng(1147)
    shuffled=[]
    for _ in range(200):
        shuffled.append(np.mean(np.linalg.norm(moved-counterpart[random.permutation(len(d))],axis=1)))
    return dict(
        mean=float(d.mean()),median=float(np.median(d)),rmse=float(np.sqrt(np.mean(d**2))),
        relative_frobenius=float(np.linalg.norm(moved-counterpart)/np.linalg.norm(counterpart)),
        randomized_mean=float(np.mean(shuffled)),
        top1=float(np.mean(rank[:,0]==indices)),
        top5=float(np.mean(np.any(rank[:,:5]==indices[:,None],axis=1)))
    )


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--seeds",nargs="+",type=int,default=list(range(10)))
    parser.add_argument("--mu",type=float,default=0.05)
    parser.add_argument("--maxiter",type=int,default=200)
    parser.add_argument("--tol",type=float,default=.01)
    args=parser.parse_args()
    out=args.out_dir.resolve()
    z=np.load(out/"domains.npz",allow_pickle=True)
    data=whitened(z)
    tr=true_indices(z["tr_pair"],z["tr_y_pair"])
    te=true_indices(z["te_pair"],z["te_y_pair"])
    oracle_row,_=orthogonal_procrustes(data["Xtrain"],data["Ytrain"][tr])
    oracle=oracle_row.T
    result={
        "source":"Real Hyser trial1, source repo class imported directly",
        "parameters":dict(mu=args.mu,maxiter=args.maxiter,tol=args.tol, seeds=args.seeds),
        "before_train":distances(data["Xtrain"],data["Ytrain"],tr),
        "before_holdout":distances(data["Xtest"],data["Ytest"],te),
        "oracle_train":distances(data["Xtrain"],data["Ytrain"],tr,oracle),
        "oracle_holdout":distances(data["Xtest"],data["Ytest"],te,oracle),
        "holdout_warning":"Last 50 points come from same trial, NOT independent test.",
        "seeds":[]
    }
    src=Path(__file__).with_name("run_hard_original.py")
    for seed in args.seeds:
        command=[sys.executable,str(src),"--out-dir",str(out),"--seed",str(seed),
                 "--mu",str(args.mu),"--maxiter",str(args.maxiter),"--tol",str(args.tol)]
        subprocess.run(command,check=True)
        run=json.loads((out/"hard_hiwa_original.json").read_text())
        R=np.asarray(run["R"])
        row={
            "seed":seed,"converged":run["converged"],"iterations":run["iterations"],
            "delta_R":run["last_global"],"primal_max":run["last_primal"],
            "group_marginal_error":run["last_group_sinkhorn_marginal_error"],
            "P":run["P"],"R":run["R"],
            "train_after":distances(data["Xtrain"],data["Ytrain"],tr,R),
            "holdout_after":distances(data["Xtest"],data["Ytest"],te,R),
        }
        result["seeds"].append(row)
        print("seed",seed,"converged",row["converged"],"holdout mean distance",row["holdout_after"]["mean"],flush=True)
    (out/"hard_hiwa_original_multiseed_distance.json").write_text(json.dumps(result,indent=2))
    print("Saved",out/"hard_hiwa_original_multiseed_distance.json")
if __name__=="__main__":main()
