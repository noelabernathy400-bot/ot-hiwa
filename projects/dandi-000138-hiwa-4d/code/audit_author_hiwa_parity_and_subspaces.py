"""Audit original HiWA source vs repo and benchmark author-like geometry.

This script compares an INDEPENDENT MATLAB-line-by-line one-outer-step Python
reference to the existing repo implementation with fixed shared random-number
stream, identifying any matrix orientation, P/Q/Sinkhorn-temperature, or ADMM
consensus update discrepancy.

Separately benchmark d=2 intrinsic in D=6 ambient with K=5 asymmetric
subspaces/latent Gaussian clusters, source/target INDEPENDENT latent draws,
true global orthogonal R* and shuffled target group IDs. No paired samples.
TRUE group IDs as group partitions only for explicitly author-known-groups
positive control (separate from fully unsupervised DANDI). P freely learned.

K=5, d=2, D=6 more closely follows paper Figure 1cd than former K=4,d=D=4.
Compare raw vs independent train-fitted per-cluster rank-2 projectors, and
old DANDI vs original author's budgets as distinct factors.
"""
from __future__ import annotations
import argparse,json,time
from itertools import product
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from zip_gcot.alignment import hierarchical_alignment,SolverConfig
from zip_gcot.numerics import polar
from scipy.optimize import linear_sum_assignment

def ref_sinkhorn(a,b,cost,eps,iters):
    # Literal MATLAB SinkhornC/Sinkhorn kernel, no data-dependent tuning.
    k=np.exp(-cost/eps)
    v=np.ones(len(b))/len(b)
    for _ in range(iters):
        u=a/(k@v)
        v=b/(k.T@u)
    return (u[:,None]*k)*v[None,:]

def literal_one_step(groupsx,groupsy,cfg,seed):
    rng=np.random.default_rng(seed)
    D=groupsx[0][0].shape[1];K=len(groupsx)
    Rglobal=polar(rng.random((D,D)))
    L=np.zeros((K,K,D,D))
    P=np.full((K,K),1/(K*K))
    rlocal=np.tile(np.eye(D),(K,K,1,1))
    cost_matrix=np.zeros((K,K))
    divisor=np.sqrt(D)
    for i,(x0,a) in enumerate(groupsx):
        x=x0/divisor
        for j,(y0,b) in enumerate(groupsy):
            y=y0/divisor
            p=P[i,j]
            T=cfg.mu/D*(Rglobal-L[i,j])
            q=np.outer(a,b)
            r=polar(rng.random((D,D)))
            for _ in range(cfg.inner_maxiter):
                before=r
                r=polar(2*p*(y.T@q.T@x)+T)
                cost=cdist(x@r.T,y,"sqeuclidean")
                q=ref_sinkhorn(a,b,cost,cfg.epsilon_sample/p,cfg.sinkhorn_maxiter)
                if np.linalg.norm(r-before,ord=2)<=cfg.inner_tol:
                    break
            rlocal[i,j]=r
            cost_matrix[i,j]=float(np.sum(q*cost))
    marginal=np.full(K,1/K)
    P=ref_sinkhorn(marginal,marginal,cost_matrix,cfg.epsilon_group,cfg.group_sinkhorn_maxiter)
    Rglobal=polar((rlocal+L).mean(axis=(0,1)))
    L=L+rlocal-Rglobal
    return dict(R=Rglobal,P=P,local_R=rlocal,cost=cost_matrix,dual=L)

def one_iteration_parity():
    rng=np.random.default_rng(20261119)
    a=[(.13*rng.normal(size=(5,3))+shift,np.full(5,.2))
       for shift in [np.array([.5,.2,-.1]),np.array([-.3,.1,.5])]]
    b=[(.13*rng.normal(size=(5,3))+shift,np.full(5,.2))
       for shift in [np.array([-.2,.3,.6]),np.array([.1,-.2,-.1])]]
    cfg=SolverConfig(maxiter=1,tol=0,mu=.05,
                     epsilon_group=.2,epsilon_sample=.1,inner_maxiter=4,
                     inner_tol=-1,sinkhorn_maxiter=100,
                     group_sinkhorn_maxiter=180,sinkhorn_tol=0,stopping="legacy")
    ref=literal_one_step(a,b,cfg,seed=27)
    new=hierarchical_alignment(a,b,seed=27,config=cfg,scale=True,retain_plans=True)
    diffs={}
    for key in ["R","P","local_R","cost","dual"]:
        diffs[key]=float(np.max(abs(ref[key]-new[key])))
        assert diffs[key]<5e-9,(key,diffs[key])
    assert np.max(abs(new["P"].sum(axis=0)-.5))<1e-8
    return dict(parity_passed=True,maximum_elementwise_errors=diffs,
       reference="Original MATLAB HiWA.m lines 65-76,95-123,162-185,191-225 translated into an independent, fixed-RNG, fixed-iteration oracle",
       code_under_test="zip_gcot.alignment.hierarchical_alignment and zip_gcot.numerics.sinkhorn",
       conclusion="For a complete outer iteration, source-target transpose, epsilon/Pij, mu/D, P and ADMM updates match literal author equations within numerical tolerance.")


def original_like_sample(seed,scale=.85,noise=.08,K=5,D=6,d=2,n=35):
    rg=np.random.default_rng(seed)
    G=polar(rg.normal(size=(D,D)))
    # independently shuffled source and target domain group identifiers
    shuffled=rg.permutation(K)
    X=[];Y=[];LX=[];LY=[];g_true_x=[];g_true_y=[]
    for group in range(K):
        C=rg.normal(size=(D,d))
        U,_=np.linalg.qr(C)
        mu=rg.normal(size=d)*2.
        v=rg.normal(size=(d,d))
        cov=v@v.T+.35*np.eye(d)
        L=np.linalg.cholesky(cov)
        u=rg.normal(size=(n,d))@L.T + mu
        w=rg.normal(size=(n,d))@L.T + mu
        px=u@U.T+noise*rg.normal(size=(n,D))
        py=(w@U.T)@G.T+noise*rg.normal(size=(n,D))
        X.append(px);Y.append(py)
        LX.extend([group]*n);LY.extend([int(shuffled[group])]*n)
        g_true_x.extend([group]*n);g_true_y.extend([group]*n)
    X=np.concatenate(X);Y=np.concatenate(Y)
    LX=np.asarray(LX);LY=np.asarray(LY)
    permx=rg.permutation(len(X));permy=rg.permutation(len(Y))
    return X[permx],Y[permy],LX[permx],LY[permy],G,shuffled

def cluster_project(x,labels,d):
    Y=np.zeros_like(x)
    for k in np.unique(labels):
        sl=(labels==k)
        # Original HiWA projects through origin: A A^T X, no centering.
        _,_,vh=np.linalg.svd(x[sl],full_matrices=False)
        basis=vh[:d].T
        Y[sl]=x[sl]@basis@basis.T
    return Y

def measurements(x,labs):
    vals=[]
    for k in range(5):
        arr=x[labs==k]
        assert len(arr)==35
        vals.append((arr,np.full(len(arr),1./len(arr))))
    return vals

def eval_run(out,rg,ref,perm):
    P=out["P"];R=out["R"]
    # data order: target group labels shuffle[group]; P target column is perm[source true label].
    mass=sum(float(P[i,perm[i]]) for i in range(5))
    return dict(true_correspondence_mass=mass,
       correct_group_mass_reference_1=1.0,
       random_group_mass_reference=.20,
       Rgt_alignment_relative_MSE=float(np.linalg.norm(rg@ref.T-R@ref.T)**2/
                                        np.linalg.norm(rg@ref.T)**2),
       rotation_Frobenius=float(np.linalg.norm(R-rg)),
       P_entropy=float(-np.sum(P*np.log(np.maximum(P,1e-300)))),
       ADMM_pass=bool(out["criteria_satisfied"]),
       consensus=float(out["consensus_residual"]),
       rotation_delta=float(out["rotation_delta"]),
       iterations=len(out["history"]),
       P=P.tolist(),R=R.tolist())

def compare_author_like(seeds,args):
    out={}
    for raw in seeds:
        X,Y,LX,LY,Rgt,permutation=original_like_sample(raw)
        modes={"raw":(X,Y),"fitted_subspace_d2":(cluster_project(X,LX,2),cluster_project(Y,LY,2))}
        settings={
           "our_previous":dict(outer=500,tol=1e-9,mu=.05,inner=20,
                               sample=45,group=200),
           "author_fig1cd":dict(outer=300,tol=.1,mu=.005,inner=100,
                               sample=150,group=1000)}
        per_seed={}
        for setname,s in settings.items():
            for m,(xx,yy) in modes.items():
                cfg=SolverConfig(maxiter=s["outer"],tol=s["tol"],mu=s["mu"],
                      epsilon_group=.2,epsilon_sample=.1,
                      inner_maxiter=s["inner"],inner_tol=.01,
                      sinkhorn_maxiter=s["sample"],group_sinkhorn_maxiter=s["group"],
                      sinkhorn_tol=1e-8,stopping="legacy")
                t=time.perf_counter()
                fit=hierarchical_alignment(measurements(xx,LX),measurements(yy,LY),
                       seed=raw,config=cfg,scale=True,retain_plans=False)
                val=eval_run(fit,Rgt,X,permutation)
                val["sec"]=float(time.perf_counter()-t)
                per_seed[f"{setname}_{m}"]=val
                print(f"AUTHOR_LIKE seed={raw} setting={setname} "
                      f"mode={m}: Pmatch={val['true_correspondence_mass']:.4f} "
                      f"RgRMSE={val['Rgt_alignment_relative_MSE']:.4f} "
                      f"niter={val['iterations']},ADMM={val['ADMM_pass']},t={val['sec']:.2f}",
                      flush=True)
        out[str(raw)]=per_seed
        (args.output/"HIWA_AUTHOR_SOURCE_AUDIT_RESULTS.json").write_text(
            json.dumps({"matlab_one_step":one_iteration_parity(),
               "author_like_comparisons":out,
               "author_like_design":"D=6 ambient, d=2 per group, 5 independent true clusters, n35/group/domain, source and target independent Gaussian latent samples with common Rgt, within-cluster Gaussian + ambient noise, unknown cross-group permutation; uses true within-domain cluster partitions for author-known-groups positive control, P free",
               "critical_note":"This is author KNOWN within-domain clusters; it is not directly comparable to our strictly unsupervised independent KMeans DANDI4D benchmark. Real original Figure1cd uses same d2/D6/S5 subspace structure but varying sample n and no-noise in some experiments.",
               "diagnostic_not_groundtruth_fitted":True,
               "group_labels_used_for_within_domain_partition_author_semi_supervised":True,
               "ground_truth_R_only_postfit":True,
               "true_cross_domain_correspondence_only_postfit":True,
               "no_source_target_sample_pair_ids_exist":True},indent=2)+"\n",encoding="utf8")
    return out

def report_results(parity,res,out):
    lines=[
      "# Author source comparison and first fully reproducible correction audit",
      "",
      "**Finding:** We independently transcribed the MATLAB update rules and ran one identical random-number-stream outer iteration against the current Python solver. No incorrect matrix multiplication or q/P rotation temperature was found in the audited step.",
      "",
      "## One-outer-step literal MATLAB parity",
      "",
      "| Array | Maximum elementwise disagreement |",
      "|---|---:|"]
    for key,val in parity["maximum_elementwise_errors"].items():
        lines.append(f"| {key} | {val:.4g} |")
    lines += ["",
      "## ORIGINAL-LIKE synthetic experiment: low-dimensional local subspaces",
      "",
      "The author's Fig1cd used **five clusters, intrinsic dimension2 and ambient dimension6**. Our earlier 4-cluster full-dimensional 4D synthetic data was structurally very different. In this experiment we generate five 2D low-rank clusters in 6D, apply unknown true R*, independently draw target and source, and shuffle target cluster IDs (true correspondence only evaluated postfit). We disclose the **known within-domain group partitions** like original semi-supervised HiWA; P remains freely solved. Uses 35 samples/cluster/domain. Ambient noise 0.08 means learned rank2 subspace projection is an actual change.",
      "",
      "| Instance/solver seed | Parameters | Local projection | Correct P group mass↑ | True Rgt alignment rMSE↓ | Stopped after | Strict ADMM |",
      "|---|---|---|---:|---:|---:|---|"]
    for seed,entries in res.items():
        for name,val in entries.items():
            pname,_,sub=name.partition("_raw")
            setting="original DANDI" if name.startswith("our_previous") else "author Fig1cd"
            mode="local PCA2 projection" if name.endswith("fitted_subspace_d2") else "none"
            lines.append(f"| {seed} | {setting} | {mode} | {val['true_correspondence_mass']:.4f} | "
                         f"{val['Rgt_alignment_relative_MSE']:.4f} | {val['iterations']} | {val['ADMM_pass']} |")
    lines+=["",
      "## Identified differences from MATLAB",
      "",
      "1. **Local projection:** Author HiWA.m lines81-93 computes per-group A_i A_i^T X_i and B_j B_j^T Y_j before local OT. Current DANDI strict4D solver uses identity/full 4D (closer to HiWASSC.m raw-data path); this is a method difference, not a transpose bug. Explicit independent rank2 PCA-per-group is now tested on a low-rank reference.",
      "2. **Synthetic identifiability:** prior DANDI-positive control used K=4 clusters in ambient D=4, each with full-dimensional noise. Authors' Fig1cd varies intrinsic d2 and D6 with S5 group partitions. Four 4D centroids do not generically identify every O(4) degree of freedom.",
      "3. **Parameter budgets:** Original Fig1cd max outer300, tol0.1, mu0.005, local100, sample Sinkhorn150, group Sinkhorn1000; our DANDI4D max outer500, tol1e-9, mu0.05, local20, sample45, group200. All differences are independently labeled, and epsilon_group0.2/sample0.1 remain equal.",
      "4. **Stopping**: original only tests global rotation change; stringent final ADMM tolerances used in our archive are additional diagnostics, not a proof of a globally correct solution.",
      "5. **Preprocessing**: the author's Python neural demo whitened each domain and used Isomap-based low-d projections, not PCA4+full 4D unprojected data.",
      "",
      "## Next stage: fix/verify without rewriting old evidence",
      "",
      "- Do not overwrite archived original Hard/Soft results.",
      "- Add compatible optional initial_R and per-cluster projection interfaces to NEW experimental variant, preserving legacy defaults and regression tests.",
      "- On synthetic known-group/subspace benchmarks, increase independent generated instances and optimizer seeds. Only add proposed initialization changes after a train-only objective, not true Rgt, selects candidates.",
      "- For DANDI choose PCA/FA/subspace variants using train-internal validation. Keep P/Q/R, actual full hierarchical transport, physical position and velocity retrieval and convergence separate.",
      "",
      "[Original MATLAB HiWA](https://github.com/siplab-gt/hiwa-matlab/blob/master/code/toolbox/HiWA.m) | [Original MATLAB Figure1cd configuration](https://github.com/siplab-gt/hiwa-matlab/blob/master/code/figure1cd.m) | [Repo original numerics](../../code/zip_gcot/alignment.py) | [Audit source](../../code/audit_author_hiwa_parity_and_subspaces.py)",
      ""]
    (out/"HIWA_AUTHOR_SOURCE_AUDIT_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf8")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--instances",nargs="+",type=int,default=[91,92,93])
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    parity=one_iteration_parity()
    print("LITERAL_MATLAB_PARITY",parity,flush=True)
    results=compare_author_like(args.instances,args)
    report_results(parity,results,args.output)
    print("AUTHOR_AUDIT_COMPLETE",flush=True)

if __name__=="__main__":
    main()
