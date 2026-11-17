"""Controlled OT-guided membership / hierarchical-alignment alternating test.

DANDI 000138 frozen source-only KMeans 4D features. Do NOT refit source
KMeans or cluster target. Target memberships always inherit source-window
memberships by synchronized index; hard labels never change.

Block 1: unmodified archived GCOT solver updates (R,P,Q) for fixed A.
Block 2: minimize a KL-corrected full P-Q transport objective over four
source-group softness gate parameters theta with R,P fixed, Q reoptimized
at each candidate theta. Regularize toward predeclared theta0.
Repeat twice, then refit R,P,Q at final memberships.

This is PARAMETERIZED, block-coordinate-inspired joint optimization;
the ZIP solver is nonconvex and restarted from identical seed each block.
Neither global optimality nor monotonicity of the whole ADMM sequence is
guaranteed. We require verified descent for the theta block.

No test samples involved in membership updates. The original train
representation was fitted on all 400 train trials; the 320/80 validation
is conditional on that prefit representation, not fully nested preprocessing.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import softmax,xlogy
from zip_gcot.alignment import hierarchical_alignment,SolverConfig
from zip_gcot.numerics import sinkhorn
from run_hard_hiwa import sample_groups
from run_controlled_soft_hard_ablation import build_measures
from run_boundary_adaptive_ablation import eval_model,summarize
from run_soft_fixedgroup import plot_comparison


def make_gate_features(x,centers,labels,inner_fit,tau=.40,q=.30):
    D=np.sum((x[:,None,:]-centers[None,:,:])**2,axis=2)
    order=np.sort(D,axis=1)
    delta=order[:,1]-order[:,0]
    h=float(np.quantile(delta[inner_fit],q))
    if h<=0:raise ValueError("Invalid source KMeans Voronoi margin width")
    anchor=softmax(-D/(2*tau*tau),axis=1)
    base_alpha=np.exp(-delta/h)
    assert np.array_equal(np.argmax(anchor,axis=1),labels)
    return anchor,base_alpha,h,delta


def gate_memberships(labels,anchor,alpha,theta):
    # max weight allowed to leave hard group is .49, hence fixed
    # original KMeans group is the strict argmax for every point.
    w=.49*alpha*theta[labels]
    A=(1-w[:,None])*np.eye(4)[labels]+w[:,None]*anchor
    A=np.maximum(A,np.finfo(float).tiny)
    A=A/A.sum(axis=1,keepdims=True)
    assert np.array_equal(np.argmax(A,axis=1),labels)
    assert np.max(np.abs(A.sum(1)-1))<1e-10
    return A


def exact_test_eval(testmask,x,y,labels):
    """Reconstruct historic GCOT evaluation indices without randomness drift."""
    ids=np.flatnonzero(testmask)
    xx=x[ids];yy=y[ids];ll=labels[ids]
    rng=np.random.default_rng(20261117)
    gi=rng.choice(len(xx),200,replace=False)
    gj=rng.choice(len(yy),200,replace=False)
    sr=np.random.default_rng(2026111701)
    tr=np.random.default_rng(2026111702)
    groups_x=[];groups_y=[];ei=[];ej=[]
    for k in range(4):
        members=np.flatnonzero(ll==k)
        if len(members)<128:raise ValueError("Evaluation needs >=128 points per group")
        ix=sr.choice(members,128,replace=False)
        iy=tr.choice(members,128,replace=False)
        groups_x.append(xx[ix]);groups_y.append(yy[iy])
        ei.extend(ix[:32]);ej.extend(iy[:32])
    return dict(x=xx,y=yy,labels=ll,idx=ids,global_x=xx[gi],
      global_y=yy[gj],groups_x=groups_x,groups_y=groups_y,
      masses=np.bincount(ll,minlength=4)/len(ll),
      ix=np.asarray(ei,int),iy=np.asarray(ej,int))


def subset_eval(mask,x,y,labels):
    ids=np.flatnonzero(mask)
    xx=x[ids];yy=y[ids];ll=labels[ids]
    rg=np.random.default_rng(30611217)
    gi=rg.choice(len(xx),200,replace=False)
    gj=rg.choice(len(yy),200,replace=False)
    r1=np.random.default_rng(30611218)
    r2=np.random.default_rng(30611219)
    gx=[];gy=[];ei=[];ej=[]
    for k in range(4):
        sel=np.flatnonzero(ll==k)
        ix=r1.choice(sel,128,replace=False)
        iy=r2.choice(sel,128,replace=False)
        gx.append(xx[ix]);gy.append(yy[iy])
        ei.extend(ix[:32]);ej.extend(iy[:32])
    return dict(x=xx,y=yy,labels=ll,idx=ids,global_x=xx[gi],
      global_y=yy[gj],groups_x=gx,groups_y=gy,
      masses=np.bincount(ll,minlength=4)/len(ll),
      ix=np.asarray(ei,int),iy=np.asarray(ej,int))


def fixed_RP_PQ_KL(theta,ids_x,ids_y,source_x,source_y,labels,anchor,alpha,
                    C,P,epsilon,qiters,anchor_lambda,anchor_theta):
    """Genuine fixed-(R,P) entropic OT + KL-corrected score, Q optimized.

    All four source and target conditional measures supported on 160 train
    windows; target group memberships are evaluated on SOURCE FEATURES at
    the corresponding y-window index, not on y itself.

    Objective is Σ P<C,Q> + eps Σ KL(Q||a⊗b) + λ mean((θ−θ0)^2).
    Optimizing θ is a well-defined differentiable-subproblem surrogate.
    It differs from ZIP's original negative-entropy objective by
    membership-dependent marginal entropy constants.
    """
    ax=gate_memberships(labels[ids_x],anchor[ids_x],alpha[ids_x],theta)
    by=gate_memberships(labels[ids_y],anchor[ids_y],alpha[ids_y],theta)
    am=ax/ax.sum(0,keepdims=True)
    bm=by/by.sum(0,keepdims=True)
    geometry=0.;KL=0.;maxerr=0.
    for i in range(4):
        a=am[:,i]
        for j in range(4):
            b=bm[:,j]
            p=max(P[i,j],1e-300)
            Q=sinkhorn(a,b,C,epsilon/p,maxiter=qiters,tol=1e-9)
            mass=a[:,None]*b[None,:]
            KL+=float(np.sum(xlogy(Q,np.maximum(Q,1e-300)/np.maximum(mass,1e-300))))
            geometry+=float(P[i,j]*np.sum(Q*C))
            maxerr=max(maxerr,float(np.max(np.abs(Q.sum(0)-b))),
                       float(np.max(np.abs(Q.sum(1)-a))))
    if maxerr>5e-4:raise ArithmeticError("OT membership optimizer Q marginal inaccurate")
    regularizer=anchor_lambda*float(np.mean((theta-anchor_theta)**2))
    return float(geometry+epsilon*KL+regularizer),float(geometry),float(KL),regularizer


def run_seed(seed,trainmask,evaldata,x,y,labels,anchor,alpha,cfg,
             rounds,theta0,lambda_anchor,opt_iter,q_iters,eval_q_iters):
    rng=np.random.default_rng(seed)
    idx=np.flatnonzero(trainmask)
    sx=sample_groups(idx,labels,40,rng)
    sy=sample_groups(idx,labels,40,rng)
    theta=np.full(4,theta0,float)
    traj=[]
    start=time.perf_counter()
    C_last=None
    fit_last=None
    for cycle in range(rounds):
        A=gate_memberships(labels,anchor,alpha,theta)
        fit=hierarchical_alignment(
            build_measures(x[sx],A[sx],False),
            build_measures(y[sy],A[sy],False),
            seed=seed,config=cfg,scale=True,retain_plans=False)
        R=np.asarray(fit["R"]);P=np.asarray(fit["P"])
        C=cdist(x[sx]@R.T/2,y[sy]/2,"sqeuclidean")
        orig=theta.copy()
        def fn(q):
            return fixed_RP_PQ_KL(np.asarray(q),sx,sy,x,y,labels,anchor,alpha,C,P,
                    cfg.epsilon_sample,q_iters,lambda_anchor,theta0)[0]
        previous=fn(theta)
        opt=minimize(fn,theta,method="L-BFGS-B",
           bounds=[(.02,1.)]*4,
           options={"maxiter":opt_iter,"maxfun":opt_iter*8,
                    "ftol":1e-7,"eps":.001})
        candidate=np.clip(opt.x,.02,1.)
        current=fn(candidate)
        accepted=bool(np.isfinite(current) and current<=previous-1e-8)
        if accepted:theta=candidate
        traj.append(dict(cycle=cycle+1,
          theta_before=orig.tolist(),theta_after=theta.tolist(),
          proxy_before=float(previous),proxy_after=float(current if accepted else previous),
          theta_step_accepted=accepted,
          optimizer_success=bool(opt.success),
          optimizer_evaluations=int(opt.nfev),
          solver_R_delta=float(fit["rotation_delta"]),
          solver_consensus=float(fit["consensus_residual"])))
        print(f"JOINT train seed={seed} cycle={cycle+1}/{rounds} "
              f"gateKL={previous:.6f}->{(current if accepted else previous):.6f}, "
              f"theta={theta.round(4).tolist()} accepted={accepted}",flush=True)
    A=gate_memberships(labels,anchor,alpha,theta)
    fit=hierarchical_alignment(
         build_measures(x[sx],A[sx],False),
         build_measures(y[sy],A[sy],False),
         seed=seed,config=cfg,scale=True,retain_plans=True)
    R=np.asarray(fit["R"]);P=np.asarray(fit["P"])
    metrics, before, after=eval_model(R,P,A,evaldata,eval_q_iters,cfg.epsilon_sample)
    marginal=float(max(np.max(abs(P.sum(0)-.25)),np.max(abs(P.sum(1)-.25))))
    orth=float(np.linalg.norm(R.T@R-np.eye(4),"fro"))
    assert marginal<1e-5 and orth<1e-7
    entropy=float(-np.mean(np.sum(xlogy(A[trainmask],A[trainmask]),axis=1))/np.log(4))
    fit_audit=bool(float(fit["rotation_delta"])<=.01
       and float(fit["consensus_residual"])<=.04)
    return dict(seed=int(seed),theta=theta.tolist(),membership_entropy=entropy,
        initial_theta=float(theta0),train_steps=traj,
        fit_iterations=len(fit["history"]),
        fit_audit_passed=fit_audit,fit_rotation_delta=float(fit["rotation_delta"]),
        fit_consensus=float(fit["consensus_residual"]),
        R=R.tolist(),P=P.tolist(),metrics=metrics,
        hierarchical_before=before,hierarchical_after=after,
        runtime_sec=float(time.perf_counter()-start))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fixed",type=Path,required=True)
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--boundary",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--rounds",type=int,default=2)
    p.add_argument("--theta0",type=float,default=.65)
    p.add_argument("--anchor-lambda",type=float,default=.02)
    p.add_argument("--inner-q-iters",type=int,default=70)
    p.add_argument("--opt-iter",type=int,default=10)
    p.add_argument("--outer",type=int,default=500)
    p.add_argument("--inner",type=int,default=20)
    p.add_argument("--local-sinkhorn",type=int,default=45)
    p.add_argument("--group-sinkhorn",type=int,default=200)
    p.add_argument("--test-q-iters",type=int,default=120)
    p.add_argument("--val-seeds",nargs="+",type=int,default=[42,43])
    p.add_argument("--final-seeds",nargs="+",type=int,default=[42,43,44,45,46])
    p.add_argument("--pilot",action="store_true")
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    if args.rounds<1 or not (0<args.theta0<1) or args.anchor_lambda<0:
        raise ValueError("invalid joint hyperparameters")
    old=json.loads(args.archive.read_text(encoding="utf8"))
    boundary=json.loads(args.boundary.read_text(encoding="utf8"))
    assert hashlib.sha256(args.fixed.read_bytes()).hexdigest()==old["source_input_sha256"]
    root=args.fixed.parents[2]
    hiwa=json.loads((root/"results/source_only_fixed_groups_outer500/hard_hiwa_results.json").read_text())
    with np.load(args.fixed,allow_pickle=False) as z:
        X=np.asarray(z["neural4"],float);Y=np.asarray(z["hand4"],float)
        tr=np.asarray(z["train_mask"],bool);te=np.asarray(z["test_mask"],bool)
        labels=np.asarray(z["shared_labels"],int)
        trials=np.asarray(z["trial_of_window"],int)
        centers=np.asarray(z["source_centers"],float)
        assert np.array_equal(labels,z["neural_groups"])
        assert np.array_equal(labels,z["hand_groups"])
    assert X.shape==Y.shape==(17500,4)
    x=(X-np.asarray(hiwa["source_mean"]))/hiwa["source_radius"]
    y=(Y-np.asarray(hiwa["target_mean"]))/hiwa["target_radius"]
    centers=(centers-np.asarray(hiwa["source_mean"]))/hiwa["source_radius"]
    training_trials=np.unique(trials[tr])
    perm=np.random.default_rng(20261117).permutation(training_trials)
    vf=tr & np.isin(trials,perm[80:])
    vv=tr & np.isin(trials,perm[:80])
    assert vf.sum()==320*35 and vv.sum()==80*35
    assert np.all(~(vv&te)) and np.all(~(vf&te))
    # Train-time boundary width and temperature fixed before running new test.
    anchor,alpha,h,delta=make_gate_features(x,centers,labels,vf,tau=.4,q=.3)
    print(f"GATING h={h:.6f} tau=.4; no label changes, 4 learned group gates",flush=True)
    cfg=SolverConfig(maxiter=args.outer,tol=1e-9,mu=.05,
          epsilon_group=.2,epsilon_sample=.1,
          inner_maxiter=args.inner,inner_tol=.01,
          sinkhorn_maxiter=args.local_sinkhorn,
          group_sinkhorn_maxiter=args.group_sinkhorn,
          sinkhorn_tol=1e-8,stopping="legacy")
    valdata=subset_eval(vv,x,y,labels)
    testdata=exact_test_eval(te,x,y,labels)
    identity,_before,_after=eval_model(np.eye(4),np.full((4,4),1/16),
        gate_memberships(labels,anchor,alpha,np.full(4,args.theta0)),
        testdata,args.test_q_iters,cfg.epsilon_sample)
    assert abs(identity["global_W2_squared"]-old["baseline"]["global_W2_squared"])<1e-9
    assert abs(identity["same_group_W2_squared"]-old["baseline"]["same_group_W2_squared"])<1e-9
    seeds_val=args.val_seeds[:1] if args.pilot else args.val_seeds
    seeds_final=args.final_seeds[:1] if args.pilot else args.final_seeds
    val_runs=[]
    for seed in seeds_val:
        r=run_seed(seed,vf,valdata,x,y,labels,anchor,alpha,cfg,
           args.rounds,args.theta0,args.anchor_lambda,args.opt_iter,
           args.inner_q_iters,args.test_q_iters)
        val_runs.append(r)
        print(f"INNER validation seed={seed}: sameW2={r['metrics']['same_group_W2_squared']:.6f}, "
             f"total={r['hierarchical_after']['total_transport_W2sq']:.6f}",flush=True)
    final_runs=[]
    for seed in seeds_final:
        r=run_seed(seed,tr,testdata,x,y,labels,anchor,alpha,cfg,
           args.rounds,args.theta0,args.anchor_lambda,args.opt_iter,
           args.inner_q_iters,args.test_q_iters)
        final_runs.append(r)
        (args.output/f"joint_seed{seed}.json").write_text(
              json.dumps(r,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
        print(f"OUTER test seed={seed}: sameW2={r['metrics']['same_group_W2_squared']:.6f}, "
              f"total={r['hierarchical_after']['total_transport_W2sq']:.6f}, "
              f"R2={r['metrics']['R2']:.5f}, audit={r['fit_audit_passed']}",flush=True)
    lookup={
       "total_PQ_OT_after":("hierarchical_after","total_transport_W2sq"),
       "same_group_W2sq":("metrics","same_group_W2_squared"),
       "global_W2sq":("metrics","global_W2_squared"),
       "MAE":("metrics","MAE"),"R2":("metrics","R2"),
       "1NN":("metrics","one_nn_shared_group")}
    summary={}
    for k,(section,field) in lookup.items():
        vals=np.asarray([r[section][field] for r in final_runs])
        summary[k]=dict(mean=float(vals.mean()),
           sd_sample=float(vals.std(ddof=1)) if len(vals)>1 else 0.,
           by_seed=vals.tolist())
    controls={}
    for method in ("hard","tau0.20"):
        controls[method]={}
        for k in lookup:
            v=old["summary"][method][k]
            controls[method][k]=dict(mean=v["mean"],sd_sample=v["sd"])
    controls["boundary"]={}
    for k,alt in [("total_PQ_OT_after","hierarchical_PQ_OT"),
                  ("same_group_W2sq","same_group_W2sq"),
                  ("global_W2sq","global_W2sq"),("MAE","MAE"),
                  ("R2","R2"),("1NN","1NN")]:
        controls["boundary"][k]=boundary["test_summary"][alt]
    result={
       "research_question":"Can a joint, OT-guided source-group membership update improve external DANDI alignment?",
       "mechanism":"Alternate original ZIP GCOT R/P/Q solver with finite-dimensional four-gate theta optimization of fixed-(R,P) KL-corrected P-Q OT",
       "not_full_unrestricted_joint_optimizer":True,
       "geometry_per_group_uses_source_only_KMeans":True,
       "all_target_weights_copied_from_synchronized_source_index":True,
       "no_target_clustering":True,
       "frozen_KMeans_group_argmax":True,
       "outer_test_not_used_for_training_or_hyperparameter_selection":True,
       "val_fit_trials":320,"val_trials":80,"test_trials":100,
       "validation_limitation":"Frozen original prefit PCA/KMeans across all 400 training trials; inner validation not fully nested preprocessor",
       "h":h,"tau":.4,"num_group_gates":4,
       "gate_formula":"A=(1-0.49 exp(-Delta/h) theta_{hard_group}) onehot + 0.49 exp(-Delta/h) theta_{hard_group} softmax(-distance^2/(2 tau^2))",
       "theta_bounds":[.02,1.],
       "theta0":args.theta0,"lambda_anchor":args.anchor_lambda,
       "block_iters":args.rounds,
       "optimizer_criterion":"train frozen R/P: geometry_scaled + eps*sum KL(Q||conditional marginals product) + lambda*mean((theta-theta0)^2), with Q reoptimized at theta",
       "four_dimensional_normalized_data":True,
       "fixed_sha256":hashlib.sha256(args.fixed.read_bytes()).hexdigest(),
       "baseline":identity,
       "validation_runs":val_runs,"test_runs":final_runs,
       "test_summary":summary,"old_archived_controls":controls,
       "settings":{k:v for k,v in vars(args).items() if not isinstance(v,Path)}}
    (args.output/"JOINT_OT_RESULTS.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    if not args.pilot and 42 in seeds_final:
        hard42=next(s for s in old["runs"]["hard"] if s["seed"]==42)
        joint42=next(s for s in final_runs if s["seed"]==42)
        plot_comparison(testdata["x"],testdata["y"],testdata["labels"],
             np.asarray(hard42["R"]),np.asarray(joint42["R"]),args.output)
        fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
        for ax,(key,name) in zip(axs.flat,[
             ("total_PQ_OT_after","Full P-Q transport"),
             ("same_group_W2sq","Fixed-group exact W2 squared"),
             ("global_W2sq","Free global W2 squared"),("R2","Paired R2")]):
            methods=["Hard","Fixed soft τ=.2","Boundary soft","Joint OT gates"]
            values=[controls["hard"][key]["mean"],controls["tau0.20"][key]["mean"],
                    controls["boundary"][key]["mean"],summary[key]["mean"]]
            sds=[controls["hard"][key]["sd_sample"],controls["tau0.20"][key]["sd_sample"],
                 controls["boundary"][key]["sd_sample"],summary[key]["sd_sample"]]
            ax.bar(methods,values,yerr=sds,capsize=4)
            ax.set(title=name)
            ax.tick_params(axis="x",rotation=12)
            ax.grid(axis="y",alpha=.14)
        fig.savefig(args.output/"joint_vs_baselines_four_errors.png",
            dpi=205,bbox_inches="tight")
        plt.close(fig)
    lines=["# 联合优化试验：GCOT / 学习四个源簇边界软化强度","",
       "**本试验是受约束的参数化交替优化，不是无约束最优软聚类，也没有证明全局最优。**",
       "仍然只使用原源域一次 KMeans 标签，目标组权重从同步源点复制。",
       "",
       "模型 A=(1−w)onehot+w softmax(τ=.4)，"
       "w=0.49 exp(−Δ/h) θ_{原硬组}，h 用320内部训练试次决定（30%分位），"
       "四个θ由 OT 引导更新，保证原来的硬分组始终是 A 的 argmax。",
       "",
       "交替步骤：(1) 用不改动的 ZIP GCOT 求解 R,P,Q；"
       "(2) 冻结 R,P，在相同160个训练样本上重求 Q，"
       "以 P-Q 传输几何代价、相对边际 KL 和θ锚定惩罚优化 θ；"
       "重复，再用最终θ重拟合 GCOT。",
       "",
       "**这里的 KL(Q||a⊗b) 修正了改变组支持引起的原始负熵常数，"
       "不能把原始含熵目标的变负误当作几何传输改善。**",
       "",
       f"固定预先指定：τ=.4，h={h:.6f}，θ初始={args.theta0:.2f}，"
       f"锚定λ={args.anchor_lambda:.5f}，交替轮数={args.rounds}。",
       "内训练320/内验证80试次，仅诊断，不以测试集挑参数；"
       "外训练400试次、独立测试100试次，五种随机种子。",
       "",
       "## 与历史基线在相同外测试集上的对照（5 seeds）",
       "",
       "| 方法 | 完整 P-Q 几何 OT↓ | 同组 W2²↓ | 全域 W2²↓ | MAE↓ | R²↑ | 1NN↑ |",
       "|---|---:|---:|---:|---:|---:|---:|"]
    for name,s in [(name,controls[key]) for key,name in
                  [("hard","Hard onehot"),("tau0.20","固定 Soft τ=.20"),("boundary","边界自适应")]]+[
                  ("OT-guided joint",summary)]:
        vals=[s[k]["mean"] for k in lookup]
        lines.append("| "+name+" | "+" | ".join(f"{s[k]['mean']:.5f}±{s[k]['sd_sample']:.5f}"
                 for k in lookup)+" |")
    lines+=["","## 学得的每组 θ、OT 代理目标与数值审计","",
       "| seed | θ（group 0/1/2/3） | 每次组权重块的 KL 目标下降 | 完整测试 P-Q OT | 同组 W2² | R² | GCOT审计 |",
       "|---|---|---|---:|---:|---:|:---:|"]
    for r in final_runs:
        change=", ".join(f"{t['proxy_before']:.5f}→{t['proxy_after']:.5f}"
                    for t in r["train_steps"])
        lines.append(f"| {r['seed']} | {np.round(r['theta'],3).tolist()} | "
            f"{change} | {r['hierarchical_after']['total_transport_W2sq']:.5f} | "
            f"{r['metrics']['same_group_W2_squared']:.5f} | {r['metrics']['R2']:.5f} | "
            f"{r['fit_audit_passed']} |")
    lines+=["","## 解释限制",
       "",
       "- 组原型和 PCA 最初用全400训练试次预处理，内部验证不是完全嵌套表示学习验证。",
       "- θ 在训练样本上更新、可推广到测试窗；没有直接用测试 Y 学习分组。",
       "- 实验仍借助真实神经/运动时窗对应传播目标组隶属度，非完全无配对。",
       "- GCOT 非凸，每块都从同种子重新运行而非严格暖启动；下降只保证固定(R,P)的 θ 子问题。",
       "- 固定同组 W2²、全域 W2² 是共同分布上的跨方法对照；完整 P-Q 代价使用不同组条件测度。",
       "- 五个随机种子不是独立动物/受试者重复，不构成显著性证据。",
       "", "[四项指标比较图](joint_vs_baselines_four_errors.png)",
       "", "[Hard/Joint/Target散点](hard_soft_target_xy.png)",
       ""]
    (args.output/"JOINT_OT_REPORT.md").write_text("\n".join(lines),encoding="utf8")
    print("JOINT FINAL METRICS",json.dumps({"learned_theta":[x["theta"] for x in final_runs],
      "summary":summary,"baseline_hard":controls["hard"],
      "fit_audit":[x["fit_audit_passed"] for x in final_runs]},indent=2),flush=True)


if __name__=="__main__":
    main()
