"""Single-solver source-fixed Hard/Soft ablation, DANDI:000138, 4D.

Same unmodified ZIP GCOT hierarchical_alignment, frozen source KMeans
and target membership inherited by the matching trial/time-window index.
Only group membership changes: onehot, distance-softmax tau=.20,.40,.80.
Reports UNREGULARIZED total heldout hierarchical P-Q transport cost
separately from raw entropy-regularized train objective.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.special import softmax,xlogy
from zip_gcot.alignment import hierarchical_alignment,SolverConfig
from zip_gcot.numerics import sinkhorn
from run_hard_hiwa import sample_groups
from run_soft_fixedgroup import evaluate,plot_comparison

TEMP=[None,.2,.4,.8]
SEEDS=[42,43,44,45,46]
K=4


def soft_members(x,centers,labels,tau):
    if tau is None:return np.eye(4)[labels]
    sq=((x[:,None,:]-centers[None,:,:])**2).sum(-1)
    z=softmax(-sq/(2*tau*tau),axis=1)
    if not np.array_equal(z.argmax(1),labels):
        raise ValueError("Source group assignments changed with temperature")
    return z


def build_measures(x,a,hard):
    result=[]
    for j in range(4):
        if hard:
            sel=a[:,j]>.5
            u=x[sel];w=np.full(len(u),1/len(u))
        else:
            u=x;w=np.maximum(a[:,j],np.finfo(float).tiny)
            w=w/w.sum()
        if not np.all(w>0):raise ValueError("Zero weighted marginal")
        result.append((u,w))
    return result


def heldout_nested(x,y,A,B,P,R,epsilon,maxiter,hard):
    """Exact ZIP Sinkhorn Q at fitted P on heldout, R=I or fitted R.

    Geometry is measured on 4D normalized coordinates; ZIP solver scale=True
    uses x/sqrt(4), so re-multiply the squared geometry by 4.
    """
    mx=build_measures(x,A,hard)
    my=build_measures(y,B,hard)
    total=0.;entr=0.;kl=0.;maxerr=0.
    C=np.zeros((4,4),float)
    for i,(xx,a) in enumerate(mx):
        xx=xx@R.T/2.
        for j,(yy,b) in enumerate(my):
            D=cdist(xx,yy/2.,"sqeuclidean")
            Q=sinkhorn(a,b,D,epsilon/max(P[i,j],1e-300),
                       maxiter=maxiter,tol=1e-9)
            if not np.isfinite(Q).all():raise FloatingPointError("Nonfinite Q")
            maxerr=max(maxerr,float(np.max(np.abs(Q.sum(0)-b))),
                       float(np.max(np.abs(Q.sum(1)-a))))
            cc=float(np.sum(Q*D))
            C[i,j]=cc*4
            total+=P[i,j]*cc
            entr+=float(np.sum(xlogy(Q,Q)))
            ref=np.outer(a,b)
            kl+=float(np.sum(xlogy(Q,np.maximum(Q,1e-300)/np.maximum(ref,1e-300))))
    if maxerr>1e-5:raise ArithmeticError(f"Test marginal error {maxerr:.3g}")
    pent=float(np.sum(xlogy(P,P)))
    return {"total_transport_W2sq":float(total*4),
            "raw_entropic_score_scaled":float(total+epsilon*entr+.2*pent),
            "KL_shifted_score_scaled":float(total+epsilon*kl+.2*np.sum(xlogy(P,P/.0625))),
            "max_Q_marginal_error":maxerr,
            "P_diagonal_mass":float(np.trace(P)),
            "cost_matrix":C.tolist()}


def fit_components(fit,x,y,A,B,eps,hard):
    """Recompute the FULL learned P-Q training objective at global R."""
    P=np.asarray(fit["P"]);Rg=np.asarray(fit["R"])
    rloc=np.asarray(fit["local_R"])
    sx=build_measures(x,A,hard);sy=build_measures(y,B,hard)
    total=0.;local=0.;entropy=0.;err=0.
    for i,(xx,a) in enumerate(sx):
        for j,(yy,b) in enumerate(sy):
            Q=np.asarray(fit[f"Q_{i}_{j}"])
            err=max(err,float(np.max(abs(Q.sum(0)-b))),
                    float(np.max(abs(Q.sum(1)-a))))
            total+=P[i,j]*float(np.sum(Q*cdist(xx@Rg.T/2.,yy/2.,"sqeuclidean")))
            local+=P[i,j]*float(np.sum(Q*cdist(xx@rloc[i,j].T/2.,yy/2.,"sqeuclidean")))
            entropy+=float(np.sum(xlogy(Q,Q)))
    value=float(total+eps*entropy+.2*np.sum(xlogy(P,P)))
    if abs(value-float(fit["objective"]))>1e-6:
        raise ArithmeticError("Did not reproduce archive's entire objective")
    return {"global_geometry_total_W2sq":float(4*total),
            "local_geometry_total_W2sq":float(4*local),
            "ZIP_original_entropy_objective_scaled":value,
            "sample_entropy_term_scaled":float(eps*entropy),
            "max_sample_Q_marginal_error":err}


def plot_summary(table,folder):
    methods=["hard","tau0.20","tau0.40","tau0.80"]
    met=[("total_W2sq","Total heldout P-Q transport (squared)"),
         ("same_group_W2sq","Same inherited group W2 squared"),
         ("global_W2sq","Unconstrained global W2 squared"),
         ("R2","Heldout paired R2")]
    fig,ax=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for a,(field,title) in zip(ax.flat,met):
        for k,m in enumerate(methods):
            rr=table["runs"][m]
            nums=np.array([r["heldout_after"]["total_transport_W2sq"]
                        if field=="total_W2sq" else
                        r["metrics"]["same_group_W2_squared"]
                        if field=="same_group_W2sq" else
                        r["metrics"]["global_W2_squared"]
                        if field=="global_W2sq" else r["metrics"]["R2"]
                        for r in rr])
            a.scatter(np.full(len(nums),k)+np.linspace(-.11,.11,len(nums)),nums,s=22)
            a.errorbar(k,nums.mean(),yerr=nums.std(ddof=1),capsize=4,fmt="D")
        a.set(title=title,xticks=range(4),
              xticklabels=["Hard","Soft .20","Soft .40","Soft .80"])
        a.grid(axis="y",alpha=.15)
    fig.suptitle("Controlled experiment: SAME GCOT solver, frozen source groups, 5 seeds")
    fig.savefig(folder/"controlled_soft_hard_all_OT_errors.png",dpi=195,bbox_inches="tight")
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--fixed",required=True,type=Path)
    ap.add_argument("--hard",required=True,type=Path)
    ap.add_argument("--hard-ot",required=True,type=Path)
    ap.add_argument("--output",required=True,type=Path)
    ap.add_argument("--seeds",type=int,nargs="+",default=SEEDS)
    ap.add_argument("--outer",type=int,default=500)
    ap.add_argument("--inner",type=int,default=20)
    ap.add_argument("--sample-sinkhorn",type=int,default=45)
    ap.add_argument("--group-sinkhorn",type=int,default=200)
    ap.add_argument("--eval-Q-iters",type=int,default=120)
    ap.add_argument("--eval-per-group",type=int,default=32)
    ap.add_argument("--tau-grid",type=float,nargs="+",default=[.2,.4,.8])
    ap.add_argument("--seed-diagnostics",action="store_true")
    ap.add_argument("--mu",type=float,default=.05)
    ap.add_argument("--sample-gamma",type=float,default=.1)
    ap.add_argument("--group-gamma",type=float,default=.2)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.group_gamma!=.2:raise ValueError("Fix group gamma .2 for this experiment")
    h=json.loads(args.hard.read_text())
    ho=json.loads(args.hard_ot.read_text())
    assert hashlib.sha256(args.fixed.read_bytes()).hexdigest()==h["data_sha256"]
    with np.load(args.fixed,allow_pickle=False) as z:
        X=np.array(z["neural4"],float);Y=np.array(z["hand4"],float)
        train=np.array(z["train_mask"],bool);test=np.array(z["test_mask"],bool)
        labels=np.array(z["shared_labels"],int)
        centers=np.array(z["source_centers"],float)
        assert np.array_equal(labels,z["neural_groups"])
        assert np.array_equal(labels,z["hand_groups"])
    assert X.shape==Y.shape==(17500,4) and train.sum()==14000 and test.sum()==3500
    x=(X-np.array(h["source_mean"]))/h["source_radius"]
    y=(Y-np.array(h["target_mean"]))/h["target_radius"]
    centers=(centers-np.array(h["source_mean"]))/h["source_radius"]
    test_x=x[test];test_y=y[test];test_labels=labels[test]
    variants={"hard":soft_members(x,centers,labels,None)}
    variants.update({f"tau{tau:.2f}":soft_members(x,centers,labels,tau)
                     for tau in args.tau_grid})
    entropy={}
    for key,a in variants.items():
        entropy[key]=float(-np.mean(np.sum(xlogy(a[train],a[train]),axis=1))/np.log(4))
        assert np.array_equal(a.argmax(axis=1),labels)
        assert np.all(np.isfinite(a))
    rng=np.random.default_rng(20261117)
    z=int(h["evaluation_unpaired_subset_n"])
    ixg=rng.choice(len(test_x),z,replace=False)
    iyg=rng.choice(len(test_y),z,replace=False)
    sr=np.random.default_rng(2026111701);tr=np.random.default_rng(2026111702)
    sx=[];sy=[];idx=[];idy=[]
    mass=np.bincount(test_labels,minlength=4)/len(test_labels)
    for k in range(4):
        zall=np.flatnonzero(test_labels==k)
        qx=sr.choice(zall,128,replace=False)
        qy=tr.choice(zall,128,replace=False)
        sx.append(test_x[qx]);sy.append(test_y[qy])
        idx.extend(qx[:args.eval_per_group]);idy.extend(qy[:args.eval_per_group])
    idx=np.asarray(idx,int);idy=np.asarray(idy,int)
    before=evaluate(np.eye(4),test_x,test_y,test_labels,
                    test_x[ixg],test_y[iyg],sx,sy,mass)
    assert abs(before["global_W2_squared"]-
        ho["baseline"]["unrestricted_W2_squared"])<1e-9
    assert abs(before["same_group_W2_squared"]-
        ho["baseline"]["same_group_W2_squared_empirical_masses"])<1e-9
    cfg=SolverConfig(maxiter=args.outer,tol=1e-9,mu=args.mu,
             epsilon_group=args.group_gamma,epsilon_sample=args.sample_gamma,
             inner_maxiter=args.inner,inner_tol=.01,
             sinkhorn_maxiter=args.sample_sinkhorn,
             group_sinkhorn_maxiter=args.group_sinkhorn,
             sinkhorn_tol=1e-8,stopping="legacy")
    full={"data":"DANDI000138, source-only KMeans, 4D",
       "same_solver":"user ZIP gcot.alignment.hierarchical_alignment (unchanged)",
       "same_training_indices_per_seed":True,
       "same_initialization_per_seed":True,
       "target_membership_copied_by_true_pairing":True,
       "independent_target_clustering":False,
       "relabeled_after_transport":False,
       "four_dimension_used":True,
       "evaluation":"Same test splits, same independently sampled OT test indices",
       "source_input_sha256":hashlib.sha256(args.fixed.read_bytes()).hexdigest(),
       "original_hard_hiwa_reference":ho["mean_sd"],
       "preselected_tau_grid":args.tau_grid,
       "normalized_assignment_entropy":entropy,
       "numerical_settings":{k:v for k,v in vars(args).items() if not isinstance(v,Path)},
       "baseline":before,"runs":{name:[] for name in variants}}
    train_idx=np.flatnonzero(train)
    for seed in args.seeds:
        rg=np.random.default_rng(seed)
        aidx=sample_groups(train_idx,labels,40,rg)
        bidx=sample_groups(train_idx,labels,40,rg)
        for name,Aall in variants.items():
            hard=(name=="hard")
            A=Aall[aidx];B=Aall[bidx]
            st=time.perf_counter()
            fitted=hierarchical_alignment(
                build_measures(x[aidx],A,hard),build_measures(y[bidx],B,hard),
                seed=seed,config=cfg,scale=True,retain_plans=True)
            duration=time.perf_counter()-st
            R=np.asarray(fitted["R"]);P=np.asarray(fitted["P"])
            assert np.linalg.norm(R.T@R-np.eye(4),"fro")<1e-7
            assert np.max(abs(P.sum(0)-.25))<1e-5
            assert np.max(abs(P.sum(1)-.25))<1e-5
            change=float(fitted["rotation_delta"])
            consensus=float(fitted["consensus_residual"])
            passed=bool(change<.01 and consensus<.04)
            trainobj=fit_components(fitted,x[aidx],y[bidx],A,B,
                                    args.sample_gamma,hard)
            metrics=evaluate(R,test_x,test_y,test_labels,
                        test_x[ixg],test_y[iyg],sx,sy,mass)
            matx=Aall[test][idx];maty=Aall[test][idy]
            evalb=heldout_nested(test_x[idx],test_y[idy],
                  matx,maty,P,np.eye(4),args.sample_gamma,args.eval_Q_iters,hard)
            evala=heldout_nested(test_x[idx],test_y[idy],
                  matx,maty,P,R,args.sample_gamma,args.eval_Q_iters,hard)
            record={"seed":seed,"method":name,"passed":passed,"change":change,
                "consensus":consensus,"fit_iterations":len(fitted["history"]),
                "duration_sec":duration,
                "R":R.tolist(),"P":P.tolist(),
                "training_total":trainobj,"heldout_before":evalb,
                "heldout_after":evala,"metrics":metrics}
            full["runs"][name].append(record)
            (args.output/f"{name}_seed{seed}.json").write_text(
                json.dumps(record,ensure_ascii=False,indent=2)+"\n")
            print(f"seed={seed}, method={name}, niter={len(fitted['history'])}, "
                  f"audit={passed}, totalPQ={evala['total_transport_W2sq']:.5f}, "
                  f"groupOT={metrics['same_group_W2_squared']:.5f}, "
                  f"globalOT={metrics['global_W2_squared']:.5f}, "
                  f"R2={metrics['R2']:.4f}, run={duration:.1f}s",flush=True)
    result={}
    fields={"total_PQ_OT_after":("heldout_after","total_transport_W2sq"),
            "total_PQ_OT_before":("heldout_before","total_transport_W2sq"),
            "total_PQ_KL_score_after":("heldout_after","KL_shifted_score_scaled"),
            "train_total_PQ_OT":("training_total","global_geometry_total_W2sq"),
            "train_raw_entropy_obj":("training_total","ZIP_original_entropy_objective_scaled"),
            "fixed_group_W2sq":("metrics","same_group_W2_squared"),
            "global_W2sq":("metrics","global_W2_squared"),
            "MAE":("metrics","MAE"),"R2":("metrics","R2"),
            "1NN":("metrics","one_nn_shared_group")}
    for name,records in full["runs"].items():
        result[name]={}
        for key,(section,value) in fields.items():
            vv=np.array([r[section][value] for r in records])
            result[name][key]={"mean":float(vv.mean()),
                               "sd":float(vv.std(ddof=1)) if len(vv)>1 else 0.}
        result[name]["numerical_audit_passed"]=sum(r["passed"] for r in records)
    full["summary"]=result
    (args.output/"CONTROLLED_ABLATION.json").write_text(
        json.dumps(full,ensure_ascii=False,indent=2)+"\n")
    plot_summary(full,args.output)
    if "tau0.40" in full["runs"] and 42 in args.seeds:
        rh=np.asarray(next(r for r in full["runs"]["hard"] if r["seed"]==42)["R"])
        rs=np.asarray(next(r for r in full["runs"]["tau0.40"] if r["seed"]==42)["R"])
        plot_comparison(test_x,test_y,test_labels,rh,rs,args.output)
    rows=["# DANDI 000138: Same GCOT solver, source-only Hard/Soft controlled ablation","",
          "Same source KMeans prototypes and one family of labels across neural/hand data;",
          "target inherits (hard or soft) membership from paired source window.",
          "This protocol uses known pairwise information, not fully unpaired soft grouping.",
          "Five common seeds, same sampled training points, same GCOT solver, equal numerical budgets.",
          "",
          "## HELDOUT TOTAL hierarchical P-Q transport cost",
          "The trained group plan P is frozen on test; local sample Q is reoptimized",
          "using the original ZIP Sinkhorn with temperature epsilon_sample/Pij.",
          "Primary cost is sum_ij Pij <Qij, 4D squared distance>; raw entropy",
          "is excluded, because support-dependent negative entropy constants",
          "would artificially favor larger (soft) support.",
          "",
          "| Membership | Total P-Q OT before | Total P-Q OT after | Same-group W2sq | Global W2sq | R2 | MAE | 1NN | Audit |",
          "|---|---:|---:|---:|---:|---:|---:|---:|:---:|"]
    for name,rr in result.items():
        def fmt(k):return f"{rr[k]['mean']:.5f} ± {rr[k]['sd']:.5f}"
        rows.append(f"| {name} | "+ " | ".join(fmt(k) for k in [
          "total_PQ_OT_before","total_PQ_OT_after","fixed_group_W2sq",
          "global_W2sq","R2","MAE","1NN"])+
          f" | {rr['numerical_audit_passed']}/{len(full['runs'][name])} |")
    rows += ["","## Soft membership entropy (normalized by log(4))"]
    rows.extend(f"- {name}: {val:.6f}" for name,val in entropy.items())
    rows += ["","## Full per-seed data"]
    for name,rs in full["runs"].items():
        for r in rs:
            rows.append(f"- {name}, seed={r['seed']}: total P-Q OT={r['heldout_after']['total_transport_W2sq']:.5f}, "
                        f"same-group OT={r['metrics']['same_group_W2_squared']:.5f}, "
                        f"global OT={r['metrics']['global_W2_squared']:.5f}, "
                        f"R2={r['metrics']['R2']:.5f}, audit={r['passed']}")
    rows += ["","## Important caveats",
       "- P may transport across different source-group IDs; the fixed same-group OT score forbids cross-group transport.",
       "- Train raw entropy-regularized objective is saved but cannot directly rank supports with different entropy baselines.",
       "- The relative KL-shifted score is a diagnostic, not the author's original objective.",
       "- All tau values are predeclared exploratory sensitivity points, not selected using an independent validation split.",
       "- Numeric residual audit is not a proof of global optimality or significance.",
       "- The 4D neural representation may not permit a true motion decoder via orthogonal mapping.",
       "",
       "![Measured OT transport and errors](controlled_soft_hard_all_OT_errors.png)",
       "",
       "![Hard/soft/target six projections](hard_soft_target_six_pairs.png)",
       ""]
    (args.output/"CONTROLLED_ABLATION.md").write_text("\n".join(rows))
    print("FINAL CONTROLLED SUMMARY",json.dumps({"statistics":result,"entropy":entropy},indent=2),flush=True)


if __name__=="__main__":
    main()
