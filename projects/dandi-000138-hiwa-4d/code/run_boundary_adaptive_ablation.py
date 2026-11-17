"""Boundary-adaptive Soft-GCOT controlled study on DANDI 000138.

Design (predeclared before viewing outer test):
- Keep one original source-only KMeans(K=4) and synchronized target group labels.
- Inner trial-wise split of original 400 train trials: 320 fit / 80 validation.
  Frozen PCA, scales and KMeans centers came from the original 400 training
  trials, so validation is conditional on prefit unsupervised representation
  (NOT fully nested preprocessing).  The original 100 test trials remain sealed.
- Choose boundary width h from Q_0.10, Q_0.30, Q_0.60 of squared-distance
  Voronoi margin on INNER FIT windows, tau0 in (0.20, 0.40).
- A = (1-alpha) onehot + alpha softmax(-distance_squared/(2*tau0**2)),
  alpha=exp(-Delta/h), Delta=second_smallest_dist2-smallest_dist2.
  Target A is copied from synchronized source window, and no labels change.
- Select h/tau by mean INNER VALIDATION fixed-group W2 squared across seeds
  42/43/44.  Compare predeclared Hard and fixed Soft tau=.20 controls.
- Refit SELECTED adaptive candidate on original 400 train trials with 5 seeds
  42..46. Test ONCE, on original 100 heldout trials, against archived same-GCOT
  hard/soft controls and same OT evaluation samples. No test hyperparam tuning.

Same 4D, same ZIP GCOT hierarchical_alignment unchanged, same solver budget:
outer 500, inner 20, Sinkhorn local 45, group 200, sampled 40/group,
epsilon_sample .1, epsilon_group .2, mu .05.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import softmax,xlogy
from zip_gcot.alignment import hierarchical_alignment,SolverConfig
from run_hard_hiwa import sample_groups
from run_soft_fixedgroup import evaluate,plot_comparison
from run_controlled_soft_hard_ablation import build_measures,heldout_nested,fit_components


K=4
def make_adaptive(x,centers,labels,h,tau):
    dist=((x[:,None,:]-centers[None,:,:])**2).sum(axis=-1)
    order=np.sort(dist,axis=1)
    delta=order[:,1]-order[:,0]
    if np.min(delta)<-1e-12 or h<=0 or tau<=0:
        raise ValueError("Bad Voronoi gap/h/tau")
    alpha=np.exp(-np.maximum(delta,0.)/h)
    q=softmax(-dist/(2*tau*tau),axis=1)
    A=(1-alpha[:,None])*np.eye(K)[labels]+alpha[:,None]*q
    A/=A.sum(axis=1,keepdims=True)
    if not np.array_equal(A.argmax(1),labels):
        raise ValueError("Original source hard labels were changed")
    if not np.isfinite(A).all():raise FloatingPointError("Bad memberships")
    return A,delta,alpha


def make_uniform(x,centers,labels,tau):
    dist=((x[:,None,:]-centers[None,:,:])**2).sum(-1)
    A=softmax(-dist/(2*tau*tau),axis=1)
    assert np.array_equal(A.argmax(1),labels)
    return A


def sample_eval(evalmask,x,y,labels,seed,global_n=200,group_n=128,nested_n=32):
    indices=np.flatnonzero(evalmask)
    xx=x[indices];yy=y[indices];lab=labels[indices]
    src=np.random.default_rng(seed)
    gx=src.choice(len(xx),global_n,replace=False)
    gy=src.choice(len(yy),global_n,replace=False)
    g1=np.random.default_rng(seed+1)
    g2=np.random.default_rng(seed+2)
    byx=[];byy=[];selx=[];sely=[]
    for k in range(K):
        ids=np.flatnonzero(lab==k)
        if len(ids)<group_n:raise ValueError(f"Too small eval group {k} with {len(ids)}")
        ix=g1.choice(ids,group_n,replace=False)
        iy=g2.choice(ids,group_n,replace=False)
        byx.append(xx[ix]);byy.append(yy[iy])
        selx.extend(ix[:nested_n]);sely.extend(iy[:nested_n])
    mass=np.bincount(lab,minlength=K)/len(lab)
    return dict(global_x=xx[gx],global_y=yy[gy],groups_x=byx,groups_y=byy,
        masses=mass,ix=np.asarray(selx,dtype=int),iy=np.asarray(sely,dtype=int),
        x=xx,y=yy,labels=lab,idx=indices)


def eval_model(R,P,Aall,evaldata,nested_iters,epsilon):
    ev=evaldata
    m=evaluate(R,ev["x"],ev["y"],ev["labels"],
        ev["global_x"],ev["global_y"],
        ev["groups_x"],ev["groups_y"],ev["masses"])
    aa=Aall[ev["idx"]][ev["ix"]]
    bb=Aall[ev["idx"]][ev["iy"]]
    before=heldout_nested(ev["x"][ev["ix"]],ev["y"][ev["iy"]],
        aa,bb,P,np.eye(4),epsilon,nested_iters,hard=False)
    after=heldout_nested(ev["x"][ev["ix"]],ev["y"][ev["iy"]],
        aa,bb,P,R,epsilon,nested_iters,hard=False)
    return m,before,after


def fit_seed(seed,fitmask,labels,x,y,Aall,cfg,evaldata,eval_iters,store_train=False):
    rng=np.random.default_rng(seed)
    fitidx=np.flatnonzero(fitmask)
    ix=sample_groups(fitidx,labels,40,rng)
    iy=sample_groups(fitidx,labels,40,rng)
    A=Aall[ix];B=Aall[iy]
    t0=time.perf_counter()
    out=hierarchical_alignment(
        build_measures(x[ix],A,False),build_measures(y[iy],B,False),
        seed=seed,config=cfg,scale=True,retain_plans=store_train)
    sec=time.perf_counter()-t0
    R=np.asarray(out["R"]);P=np.asarray(out["P"])
    change=float(out["rotation_delta"]);consensus=float(out["consensus_residual"])
    marginal=max(float(abs(P.sum(0)-.25).max()),float(abs(P.sum(1)-.25).max()))
    orth=float(np.linalg.norm(R.T@R-np.eye(4),"fro"))
    assert orth<1e-7 and marginal<1e-5
    m,before,after=eval_model(R,P,Aall,evaldata,eval_iters,cfg.epsilon_sample)
    audit=bool(change<=.01 and consensus<=.04 and marginal<1e-5 and orth<1e-7)
    result=dict(seed=int(seed),R=R.tolist(),P=P.tolist(),
        elapsed_seconds=float(sec),
        iterations=len(out["history"]),
        strict_audit_passed=audit,rotation_delta=change,consensus_fro=consensus,
        group_marginal_error=marginal,orthogonality_error=orth,
        evaluation=m,OT_before=before,OT_after=after)
    if store_train:
        result["train_components"]=fit_components(
            out,x[ix],y[iy],A,B,cfg.epsilon_sample,False)
    return result


def summarize(records):
    keys={"hierarchical_PQ_OT":"OT_after.total_transport_W2sq",
          "hierarchical_PQ_before":"OT_before.total_transport_W2sq",
          "same_group_W2sq":"evaluation.same_group_W2_squared",
          "global_W2sq":"evaluation.global_W2_squared",
          "MAE":"evaluation.MAE","R2":"evaluation.R2",
          "1NN":"evaluation.one_nn_shared_group"}
    values={}
    for name,route in keys.items():
        section,k=route.split(".")
        v=np.array([s[section][k] for s in records],float)
        values[name]={"mean":float(v.mean()),
                      "sd_sample":float(v.std(ddof=1)) if len(v)>1 else 0.,
                      "by_seed":v.tolist()}
    return values


def make_plots(d,x,y,labels,testmask,output):
    testx=x[testmask];testy=y[testmask];l=labels[testmask]
    a=next(s for s in d["test_runs"] if s["seed"]==42)
    h=next(s for s in d["archived_control_runs"]["hard"] if s["seed"]==42)
    plot_comparison(testx,testy,l,np.asarray(h["R"]),np.asarray(a["R"]),output)
    # Plot six-projection with source before, adaptive source after, target.
    from itertools import combinations
    clouds=[("Source BEFORE",testx),
            ("Boundary soft AFTER",testx@np.asarray(a["R"]).T),
            ("Target, inherited groups",testy)]
    fig,axs=plt.subplots(6,3,figsize=(13.5,21),constrained_layout=True)
    cmap=plt.get_cmap("tab10")
    for row,(i,j) in enumerate(combinations(range(4),2)):
        lo=min(float(v[:,i].min()) for _,v in clouds)
        hi=max(float(v[:,i].max()) for _,v in clouds)
        l2=min(float(v[:,j].min()) for _,v in clouds)
        h2=max(float(v[:,j].max()) for _,v in clouds)
        pd=.05*(hi-lo);pd2=.05*(h2-l2)
        for col,(name,v) in enumerate(clouds):
            ax=axs[row,col]
            for k in range(K):
                mask=l==k
                ax.scatter(v[mask,i],v[mask,j],s=3,alpha=.45,
                           color=cmap(k),rasterized=True)
            ax.set(title=f"{name} ({i+1},{j+1})",xlabel=f"dim {i+1}",
                   ylabel=f"dim {j+1}",xlim=(lo-pd,hi+pd),ylim=(l2-pd2,h2+pd2))
    fig.suptitle("Identical 4D source KMeans labels | source before / adaptive after / target")
    fig.savefig(output/"boundary_before_after_target_six_views.png",
                dpi=180,bbox_inches="tight")
    plt.close(fig)

    # Chart: candidate validation metric, plus test vs archived hard and fixed tau
    val=d["validation_candidates"]
    vnames=[z["name"] for z in val]
    vmeans=[z["summary"]["same_group_W2sq"]["mean"] for z in val]
    vsds=[z["summary"]["same_group_W2sq"]["sd_sample"] for z in val]
    fig,ax=plt.subplots(figsize=(11,4.6),constrained_layout=True)
    ax.errorbar(np.arange(len(val)),vmeans,yerr=vsds,fmt="o",capsize=4)
    ax.axhline(d["validation_controls"]["hard"]["summary"]["same_group_W2sq"]["mean"],
        linestyle="--",color="gray",label="Hard validation")
    ax.set(xticks=np.arange(len(val)),xticklabels=vnames,
           xlabel="Training-only validation candidate",ylabel="Fixed-group W2 squared")
    ax.tick_params(axis="x",rotation=25)
    ax.legend()
    ax.grid(alpha=.17)
    fig.savefig(output/"boundary_validation_grid.png",dpi=200)
    plt.close(fig)

    final=d["test_summary"]
    hard=d["archived_control_summaries"]["hard"]
    fixed=d["archived_control_summaries"]["tau0.20"]
    figs=[("hierarchical_PQ_OT","Full hierarchical P-Q OT"),
          ("same_group_W2sq","Fixed same-group W2 squared"),
          ("global_W2sq","Unrestricted W2 squared"),
          ("R2","Paired R2")]
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,(key,title) in zip(axes.flat,figs):
        vals=[hard[key]["mean"],fixed[key]["mean"],final[key]["mean"]]
        std=[hard[key]["sd_sample"],fixed[key]["sd_sample"],
             final[key]["sd_sample"]]
        ax.bar(["Hard","Soft τ=.20","Boundary soft"],vals,yerr=std,capsize=4)
        ax.set(title=title)
        ax.grid(axis="y",alpha=.14)
    fig.suptitle("Frozen official test: same solver and common seeds")
    fig.savefig(output/"boundary_vs_hard_soft_all_errors.png",dpi=205)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fixed",required=True,type=Path)
    p.add_argument("--archive",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--outer",type=int,default=500)
    p.add_argument("--inner",type=int,default=20)
    p.add_argument("--local-sinkhorn",type=int,default=45)
    p.add_argument("--group-sinkhorn",type=int,default=200)
    p.add_argument("--val-seeds",type=int,nargs="+",default=[42,43,44])
    p.add_argument("--final-seeds",type=int,nargs="+",default=[42,43,44,45,46])
    p.add_argument("--val-trials",type=int,default=80)
    p.add_argument("--h-quantiles",type=float,nargs="+",default=[.10,.30,.60])
    p.add_argument("--taus",type=float,nargs="+",default=[.2,.4])
    p.add_argument("--eval-Q-iters",type=int,default=120)
    p.add_argument("--pilot",action="store_true")
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)

    archived=json.loads(args.archive.read_text(encoding="utf-8"))
    with np.load(args.fixed,allow_pickle=False) as z:
        X=np.asarray(z["neural4"],float);Y=np.asarray(z["hand4"],float)
        train=np.asarray(z["train_mask"],bool);test=np.asarray(z["test_mask"],bool)
        labels=np.asarray(z["shared_labels"],int)
        assert np.array_equal(labels,z["neural_groups"])
        assert np.array_equal(labels,z["hand_groups"])
        trials=np.asarray(z["trial_of_window"],int)
        centers=np.asarray(z["source_centers"],float)
    assert X.shape==Y.shape==(17500,4)
    assert int(train.sum())==14000 and int(test.sum())==3500
    assert len(np.unique(trials[train]))==400 and len(np.unique(trials[test]))==100
    assert set(trials[train]).isdisjoint(set(trials[test]))
    # EXACT original per-domain 400-trial RMS normalization from prior experiment,
    # all learned before observing the outer 100-trial test.
    baseline_sha=hashlib.sha256(args.fixed.read_bytes()).hexdigest()
    assert baseline_sha==archived["source_input_sha256"]
    old_hard=json.loads((args.fixed.parent.parent/
                         "source_only_fixed_groups_outer500"/"hard_hiwa_results.json").read_text())
    x=(X-np.asarray(old_hard["source_mean"]))/float(old_hard["source_radius"])
    y=(Y-np.asarray(old_hard["target_mean"]))/float(old_hard["target_radius"])
    centers=(centers-np.asarray(old_hard["source_mean"]))/float(old_hard["source_radius"])

    train_trials=np.unique(trials[train])
    rng=np.random.default_rng(20261117)
    perm=rng.permutation(train_trials)
    if args.val_trials>=len(perm):raise ValueError("Not enough fit trials")
    inner_v=np.isin(trials,perm[:args.val_trials])&train
    inner_f=np.isin(trials,perm[args.val_trials:])&train
    assert inner_v.sum()==35*args.val_trials
    assert inner_f.sum()==35*(400-args.val_trials)
    assert not np.any(inner_v&test) and not np.any(inner_f&test)

    dx=((x[:,None,:]-centers[None,:,:])**2).sum(-1)
    rank=np.sort(dx,axis=1)
    delta=rank[:,1]-rank[:,0]
    h_values=np.quantile(delta[inner_f],args.h_quantiles)
    if np.min(h_values)<=0:raise ValueError("Voronoi margin widths must be positive")
    if args.pilot:
        h_values=h_values[:1]
        args.taus=args.taus[:1]
        args.val_seeds=args.val_seeds[:1]
        args.final_seeds=args.final_seeds[:1]
    A_hard=np.eye(K)[labels]
    A_fixed=make_uniform(x,centers,labels,.2)
    candidates=[]
    for qi,h in zip(args.h_quantiles,h_values):
        for tau in args.taus:
            A,computed_delta,alpha=make_adaptive(x,centers,labels,float(h),float(tau))
            assert np.array_equal(computed_delta,delta)
            entropy=float(-np.mean(np.sum(xlogy(A[inner_f],A[inner_f]),axis=1))/np.log(K))
            boundary=float(np.mean(alpha[inner_f]))
            candidates.append(dict(name=f"hQ{int(qi*100):02d}_tau{tau:.2f}",
                    quantile=float(qi),h=float(h),tau=float(tau),
                    entropy=entropy,mean_alpha=boundary,A=A))
    if not candidates:raise ValueError("Grid empty")

    inner_eval=sample_eval(inner_v,x,y,labels,seed=30611217)
    full_eval=sample_eval(test,x,y,labels,seed=20261117)
    # The historical test benchmark's global picks are seed 20261117, group
    # sampling seed 2026111701/02.  Reconstruct its group sample exactly.
    def old_test_eval():
        z=np.flatnonzero(test);xt=x[z];yt=y[z];lt=labels[z]
        rng=np.random.default_rng(20261117)
        gi=rng.choice(len(xt),200,replace=False)
        gj=rng.choice(len(yt),200,replace=False)
        r1=np.random.default_rng(2026111701)
        r2=np.random.default_rng(2026111702)
        sx=[];sy=[];ii=[];jj=[]
        for k in range(K):
            group=np.flatnonzero(lt==k)
            i=r1.choice(group,128,replace=False)
            j=r2.choice(group,128,replace=False)
            sx.append(xt[i]);sy.append(yt[j])
            ii.extend(i[:32]);jj.extend(j[:32])
        return dict(x=xt,y=yt,labels=lt,idx=z,global_x=xt[gi],
                    global_y=yt[gj],groups_x=sx,groups_y=sy,
                    ix=np.asarray(ii),iy=np.asarray(jj),
                    masses=np.bincount(lt,minlength=4)/len(lt))
    full_eval=old_test_eval()
    ident=evaluate(np.eye(4),full_eval["x"],full_eval["y"],full_eval["labels"],
      full_eval["global_x"],full_eval["global_y"],full_eval["groups_x"],
      full_eval["groups_y"],full_eval["masses"])
    assert abs(ident["global_W2_squared"]-
               archived["baseline"]["global_W2_squared"])<1e-9
    assert abs(ident["same_group_W2_squared"]-
               archived["baseline"]["same_group_W2_squared"])<1e-9

    cfg=SolverConfig(maxiter=args.outer,tol=1e-9,mu=.05,
      epsilon_group=.2,epsilon_sample=.1,inner_maxiter=args.inner,
      inner_tol=.01,sinkhorn_maxiter=args.local_sinkhorn,
      group_sinkhorn_maxiter=args.group_sinkhorn,sinkhorn_tol=1e-8,stopping="legacy")

    controls={}
    for name,A in [("hard",A_hard),("soft_tau0.20",A_fixed)]:
        records=[]
        for seed in args.val_seeds:
            r=fit_seed(seed,inner_f,labels,x,y,A,cfg,inner_eval,
                       args.eval_Q_iters,store_train=False)
            records.append(r)
            print(f"VALIDATION baseline={name} seed={seed}, "
                  f"sameW2={r['evaluation']['same_group_W2_squared']:.5f}",flush=True)
        controls[name]={"runs":records,"summary":summarize(records)}
    grid=[]
    for item in candidates:
        rec=[]
        for seed in args.val_seeds:
            r=fit_seed(seed,inner_f,labels,x,y,item["A"],cfg,inner_eval,
                       args.eval_Q_iters,store_train=False)
            rec.append(r)
            print(f"VALIDATION candidate={item['name']} seed={seed}, "
                  f"sameW2={r['evaluation']['same_group_W2_squared']:.5f} "
                  f"totalPQ={r['OT_after']['total_transport_W2sq']:.5f}",flush=True)
        grid.append({k:v for k,v in item.items() if k!="A"} |
                   {"runs":rec,"summary":summarize(rec)})
    # Predeclared selection target is a FIXED-DISTRIBUTION group W2 score
    # rather than adaptive measure-dependent nested P-Q OT.
    good=[v for v in grid if all(r["strict_audit_passed"] for r in v["runs"])]
    selected=min(good or grid,key=lambda v:v["summary"]["same_group_W2sq"]["mean"])
    print("SELECTED BY INNER VALIDATION ONLY:",selected["name"],
          selected["summary"]["same_group_W2sq"]["mean"],flush=True)
    Astar=next(v["A"] for v in candidates if v["name"]==selected["name"])
    full=[]
    for seed in args.final_seeds:
        record=fit_seed(seed,train,labels,x,y,Astar,cfg,full_eval,
                        args.eval_Q_iters,store_train=True)
        full.append(record)
        print(f"FINAL OUTER TEST seed={seed} "
            f"totalPQ={record['OT_after']['total_transport_W2sq']:.6f}, "
            f"groupOT={record['evaluation']['same_group_W2_squared']:.6f}, "
            f"R2={record['evaluation']['R2']:.6f}, "
            f"audit={record['strict_audit_passed']}",flush=True)
        (args.output/f"adaptive_seed{seed}.json").write_text(
            json.dumps(record,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    fullstat=summarize(full)
    old_s=archived["summary"]
    controlstat={}
    for method in ("hard","tau0.20"):
        zz=old_s[method]
        controlstat[method]={}
        for newkey,oldkey in [
          ("hierarchical_PQ_OT","total_PQ_OT_after"),
          ("hierarchical_PQ_before","total_PQ_OT_before"),
          ("same_group_W2sq","fixed_group_W2sq"),
          ("global_W2sq","global_W2sq"),
          ("MAE","MAE"),("R2","R2"),("1NN","1NN")]:
            controlstat[method][newkey]={"mean":zz[oldkey]["mean"],
                         "sd_sample":zz[oldkey]["sd"]}

    result=dict(
      question="Does boundary adaptive softness beat fixed-soft/hard on DANDI 000138?",
      source_data_sha256=baseline_sha,archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
      neural_and_hand_four_dimensions=True,source_one_KMeans_only=True,
      target_membership_inherited_by_true_paired_window=True,
      original_hard_labels_unchanged=True,same_GCOT_solver_as_archived_ablation=True,
      preprocessing_and_KMeans_pre_fitted_on_all_400_training_trials=True,
      inner_validation_not_fully_nested_representation_learning=True,
      validation_method="trial-wise 320 fit / 80 inner validation from ORIGINAL 400 train; select by mean fixed-group W2 squared only; outer 100-trial test not used",
      validation_train_trials=400-args.val_trials,
      validation_trials=args.val_trials,final_test_trials=100,
      val_seed_list=args.val_seeds,final_seed_list=args.final_seeds,
      h_quantile_grid=args.h_quantiles,taus=args.taus,h_realization=h_values.tolist(),
      validation_controls=controls,validation_candidates=grid,
      selected_h_tau={k:selected[k] for k in ("name","quantile","h","tau","entropy","mean_alpha")},
      selection_metric="mean inner-val exact same-inherited-group W2^2, fixed hard groups and sample subset",
      test_runs=full,test_summary=fullstat,
      archived_control_summaries=controlstat,
      archived_control_runs={
          key:[{"seed":r["seed"],"R":r["R"]} for r in archived["runs"][key]]
          for key in ("hard","tau0.20")},
      test_identity_metrics=ident,
      provenance="DANDI 000138 frozen source-only KMeans, original ZIP GCOT solver unchanged",
      evaluation="Same original heldout 200 global OT, 4x128 same-group exact OT, and 4x32 nested P-Q sinkhorn OT",
      entropy_warning="Test hierarchical P-Q geometrical cost excludes entropy but uses entropic Q and membership-dependent conditional measures",
      configs={k:v for k,v in vars(args).items() if not isinstance(v,Path)})
    (args.output/"BOUNDARY_ADAPTIVE_RESULTS.json").write_text(
      json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if not args.pilot and len(args.final_seeds)==5:
        make_plots(result,x,y,labels,test,args.output)
    lines=[
      "# DANDI 000138 | 自适应边界软分组对照实验",
      "",
      "**目标：**测试在不改变唯一源域 K-means 硬标签的前提下，"
      "仅对簇边界附近的数据分配明显软权重，能否改善 OT 对齐。",
      "",
      "## 模型与边界定义",
      "",
      "在归一化四维神经域上对固定源 K-means 中心计算 Δ=第二近距离平方−最近距离平方。",
      "α=exp(-Δ/h)，A=(1−α)×onehot(KMeans标签)+α×distance-softmax(τ)。",
      "目标域组权重通过同步行索引直接复制源域，绝无目标 K-means、无对齐后重新分组。",
      "",
      "## 严格区分内部验证与外部测试",
      "",
      f"在原400训练试次内部按试次划分 {400-args.val_trials}/{args.val_trials} "
      f"训练/验证；温度候选 {args.taus}、h 基于内部训练 Voronoi 间隔分位数"
      f" {args.h_quantiles}，真实 h={h_values.tolist()}。",
      "选参指标**仅**为内部验证集的固定同编号组精确 W2²，而非最终测试集的任何数值。",
      f"选中：{selected['name']}（h={selected['h']:.6f},τ={selected['tau']:.3f}），"
      f"验证均值 W2²={selected['summary']['same_group_W2sq']['mean']:.6f}。",
      "",
      "**限制：**本实验沿用以前已在原400训练试次上拟合的 PCA、统计归一化与单一源域 K-means；"
      "内部验证在这些预先拟合的无监督表示上进行，因此不属于完全重嵌套的特征学习验证。",
      "原100测试试次从未用于 h/τ 的选择。",
      "",
      "## 验证候选",
      "",
      "| 候选 | h | τ | 平均分配熵 | 内部验证固定同组 OT W2² ±SD |",
      "|---|---:|---:|---:|---:|"
    ]
    for row in grid:
        ss=row["summary"]["same_group_W2sq"]
        lines.append(f"| {row['name']} | {row['h']:.5f} | {row['tau']:.2f} | "
            f"{row['entropy']:.4f} | {ss['mean']:.5f} ± {ss['sd_sample']:.5f} |")
    for name,row in controls.items():
        ss=row["summary"]["same_group_W2sq"]
        lines.append(f"| 验证控制：{name} | — | — | — | "
            f"{ss['mean']:.5f} ± {ss['sd_sample']:.5f} |")
    lines += ["","## 100 个完整独立测试试次（5 seeds）",
      "",
      "| 方法 | 测试 P-Q 总几何 OT↓ | 固定同编号组 OT↓ | 全域 OT↓ | MAE↓ | R²↑ | 1NN↑ |",
      "|---|---:|---:|---:|---:|---:|---:|"]
    def fmt(v):return f"{v['mean']:.5f}±{v['sd_sample']:.5f}"
    for name,stats in [("Hard同求解器基准",controlstat["hard"]),
                       ("固定 Soft τ=.20",controlstat["tau0.20"]),
                       (f"边界自适应 {selected['name']}",fullstat)]:
        lines.append("| "+name+" | "+" | ".join(fmt(stats[k]) for k in (
            "hierarchical_PQ_OT","same_group_W2sq","global_W2sq","MAE","R2","1NN"))+" |")
    lines += ["","## 模型审计与局限",
        "",
        f"新方法的最终残差审计通过次数：{sum(s['strict_audit_passed'] for s in full)}/{len(full)}。",
        "完整分层 P-Q 测试几何 OT 不含熵项，但组测度因软隶属度而变化，"
        "其绝对值不能当作相同固定测度上的唯一优劣证明；固定同组 W2² 与全域 W2² "
        "才是在固定分布上完全一致的外部质量评价。",
        "虽然 GCOT 数值求解器相同，非凸交替算法并无全局最优保证。",
        "1NN 是继承标签邻域一致性，不是独立神经运动标签分类准确率。",
        "软分组使用真实同步对应标签，不属于完全无配对或 TPAMI 两域独立原型方法。",
        "独立测试试次只有100个，种子重复不是独立受试者/动物样本。",
        "", "[验证网格](boundary_validation_grid.png)",
        "", "[误差对比](boundary_vs_hard_soft_all_errors.png)",
        "", "[四维投影](boundary_before_after_target_six_views.png)",
        ""]
    (args.output/"BOUNDARY_ADAPTIVE_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    print("BOUNDARY VALIDATION AND OUTER TEST RESULTS",
          json.dumps({"selected":result["selected_h_tau"],
            "hard":controlstat["hard"],"soft_fixed":controlstat["tau0.20"],
            "adaptive":fullstat,"num_audit_passed":sum(s["strict_audit_passed"] for s in full)},
           indent=2,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
