#!/usr/bin/env python3
"""Run the exact user-uploaded GC-OT branch's Hard HiWA on a fixed all250_input.npz.

Requires a checkout/worktree of branch:
  experiments/dandi-000138-4d-hard-clustering-20261117
and --gcot-root set to its research/GCOT-reproducible-20261018/GC-OT directory.

Use --input output/all250_input.npz from run_all250_hard.py.
This file imports gcot.alignment and gcot.baselines from the source; it does
NOT embed a manually transcribed solver.
"""
from pathlib import Path
import argparse, json, sys
import numpy as np

def score(x,y,R):
    diff=x@R.T-y
    denom=np.sum((y-y.mean(axis=0))**2)
    return {"R2":float(1-np.sum(diff**2)/denom),
            "relative_squared_error":float(np.sum(diff**2)/np.sum(y**2)),
            "mean_paired_distance":float(np.linalg.norm(diff,axis=1).mean())}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--gcot-root",type=Path,required=True)
    ap.add_argument("--input",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--stopping",choices=["legacy","admm"],default="legacy")
    args=ap.parse_args()
    sys.path.insert(0,str(args.gcot_root.resolve()))
    from gcot.alignment import SolverConfig,hierarchical_alignment,hard_measures
    from gcot.baselines import hiwa_gmm
    data=np.load(args.input)
    X=data["X"];Y=data["Yshuffle"];P=data["pair"]
    true_y=Y[P]
    lx=data["cluster_x"];ly=data["cluster_y"]
    cfg=SolverConfig(stopping=args.stopping,mu=.005,maxiter=300,tol=.1,
       epsilon_group=.2,epsilon_sample=.1,inner_maxiter=100,inner_tol=.01,
       sinkhorn_maxiter=150,group_sinkhorn_maxiter=1000,sinkhorn_tol=0)
    rows=[]
    for name,func in [
        ("hard_measures_KMeans",lambda:hierarchical_alignment(
            hard_measures(X,lx),hard_measures(Y,ly),args.seed,cfg,scale=True)),
        ("HiWA_GMM_Isomap",lambda:hiwa_gmm(X,Y,groups=4,seed=args.seed,config=cfg)),
    ]:
        model=func()
        rr=np.asarray(model["R"])
        record={"method":name,"seed":args.seed,"iterations":int(len(model["history"])),
                "converged":bool(model["converged"]),
                "rotation_delta":float(model["rotation_delta"]),
                "consensus_residual":float(model["consensus_residual"]),
                "marginal_error":float(model["marginal_error"]),
                "stop_reason":model["stop_reason"],
                "R":rr.tolist(),"P":np.asarray(model["P"]).tolist(),
                "metrics":score(X,true_y,rr)}
        rows.append(record)
        print(json.dumps({k:v for k,v in record.items() if k not in ("R","P")},indent=2))
    output={"dataset":"Real Hyser all250 preprocessed", "source":"Exact imported GC-OT branch code",
       "input":str(args.input),"gcot_root":str(args.gcot_root),"stopping":args.stopping,
       "before":score(X,true_y,np.eye(3)),"results":rows}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print("saved",args.output)

if __name__=="__main__":main()
