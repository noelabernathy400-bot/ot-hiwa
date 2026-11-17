"""Soft-vs-hard experiment on the same four-dimensional DANDI 000138 cache.

This is a SOURCE-PROTOTYPE-CONSTRAINED variant of the ZIP GCOT soft solver,
NOT the author's original two-domain separately fitted prototypes:
  - Exactly one frozen source K-means, K=4, supplies the centers.
  - Soft memberships are a distance-softmax around those same centers.
  - Target membership is copied by known synchronized time-window index.
  - All memberships stay fixed throughout GCOT alignment.
  - Original ZIP gcot.alignment.hierarchical_alignment and weighted_measures
    are imported unchanged; all computations stay in the shared 4D space.
This is a paired-label comparison, not a fully unsupervised experiment.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.special import softmax
from zip_gcot.alignment import hierarchical_alignment,weighted_measures,SolverConfig
from run_hard_hiwa import sample_groups


def ot_sq(a,b):
    assert a.shape==b.shape and a.shape[1]==4
    D=cdist(a,b,"sqeuclidean")
    ii,jj=linear_sum_assignment(D)
    return float(np.mean(D[ii,jj]))


def evaluate(R,x,y,label,global_x,global_y,source_groups,target_groups,group_masses):
    pred=x@R.T
    diff=pred-y
    sse=float(np.sum(diff**2))
    sst=float(np.sum((y-y.mean(axis=0))**2))
    nearest=cKDTree(pred).query(y,k=1)[1]
    return dict(
        one_nn_shared_group=float(np.mean(label[nearest]==label)),
        MAE=float(np.abs(diff).mean()),
        R2=float(1-sse/sst),
        global_W2_squared=ot_sq(global_x@R.T,global_y),
        same_group_W2_squared=float(sum(
            group_masses[k]*ot_sq(source_groups[k]@R.T,target_groups[k])
            for k in range(4))),
    )


def plot_comparison(x,y,label,Rh,Rs,out):
    old=x@Rh.T;new=x@Rs.T
    clouds=(("HARD | seed42",old),("SOFT | seed42",new),("TARGET | shared labels",y))
    limits=[]
    for k in range(4):
        v=np.concatenate([cloud[:,k] for _,cloud in clouds])
        lo,hi=float(v.min()),float(v.max())
        gap=(hi-lo)*.045
        limits.append((lo-gap,hi+gap))
    fig,axs=plt.subplots(6,3,figsize=(13.5,20),constrained_layout=True)
    from itertools import combinations
    for row,(i,j) in enumerate(combinations(range(4),2)):
        for col,(title,cloud) in enumerate(clouds):
            ax=axs[row,col]
            for group in range(4):
                m=(label==group)
                ax.scatter(cloud[m,i],cloud[m,j],s=3,alpha=.48,
                           c=[plt.get_cmap("tab10")(group)],rasterized=True)
            ax.set(xlabel=f"coordinate {i+1}",ylabel=f"coordinate {j+1}",
                   title=f"{title}: ({i+1},{j+1})",xlim=limits[i],ylim=limits[j])
            ax.grid(alpha=.12)
    fig.suptitle("Identical inherited SOURCE labels and held-out samples | Hard vs Soft vs Target")
    fig.savefig(out/"hard_soft_target_six_pairs.png",dpi=180,bbox_inches="tight")
    plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(13.8,4.6),constrained_layout=True)
    for ax,(title,cloud) in zip(axs,clouds):
        for group in range(4):
            m=(label==group)
            ax.scatter(cloud[m,0],cloud[m,1],s=5,alpha=.5,
                       c=[plt.get_cmap("tab10")(group)],rasterized=True)
        ax.set(xlabel="coordinate 1",ylabel="coordinate 2",xlim=limits[0],
               ylim=limits[1],title=title)
    fig.savefig(out/"hard_soft_target_xy.png",dpi=220,bbox_inches="tight")
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--fixed",required=True,type=Path)
    p.add_argument("--hard",required=True,type=Path)
    p.add_argument("--hard-ot",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--seeds",nargs="+",type=int,default=[42,43,44,45,46])
    p.add_argument("--fit-per-group",type=int,default=40)
    p.add_argument("--outer",type=int,default=120)
    p.add_argument("--inner",type=int,default=8)
    p.add_argument("--sample-sinkhorn",type=int,default=35)
    p.add_argument("--group-sinkhorn",type=int,default=200)
    p.add_argument("--tau",type=float,default=.40)
    p.add_argument("--group-gamma",type=float,default=.2)
    p.add_argument("--sample-gamma",type=float,default=.1)
    p.add_argument("--mu",type=float,default=.05)
    p.add_argument("--stop-tol",type=float,default=1e-9)
    p.add_argument("--store-local-plans",action="store_true")
    args=p.parse_args()
    if args.tau<=0 or args.fit_per_group<4:raise ValueError("Bad softness or support")
    args.output.mkdir(exist_ok=True,parents=True)
    hard=json.loads(args.hard.read_text(encoding="utf8"))
    hardot=json.loads(args.hard_ot.read_text(encoding="utf8"))
    assert hashlib.sha256(args.fixed.read_bytes()).hexdigest()==hard["data_sha256"]
    with np.load(args.fixed,allow_pickle=False) as z:
        X=np.asarray(z["neural4"],float)
        Y=np.asarray(z["hand4"],float)
        tr=np.asarray(z["train_mask"],bool)
        te=np.asarray(z["test_mask"],bool)
        labels=np.asarray(z["shared_labels"],int)
        assert np.array_equal(labels,z["neural_groups"])
        assert np.array_equal(labels,z["hand_groups"])
        centers=np.asarray(z["source_centers"],float)
    assert X.shape==Y.shape==(17500,4)
    assert tr.sum()==14000 and te.sum()==3500
    assert labels.max()==3 and labels.min()==0

    mean_x=np.asarray(hard["source_mean"])
    mean_y=np.asarray(hard["target_mean"])
    rad_x=float(hard["source_radius"])
    rad_y=float(hard["target_radius"])
    x=(X-mean_x)/rad_x
    y=(Y-mean_y)/rad_y
    centers=(centers-mean_x)/rad_x
    # This is soft KMeans: exactly the SAME hard Voronoi groups at argmax,
    # with finite-temperature continuous membership across boundaries.
    delta=x[:,None,:]-centers[None,:,:]
    dist2=np.einsum("nkd,nkd->nk",delta,delta)
    soft=softmax(-dist2/(2*args.tau**2),axis=1)
    assert np.max(abs(soft.sum(axis=1)-1))<1e-12
    assert np.isfinite(soft).all()
    assert np.array_equal(np.argmax(soft,axis=1),labels), "Soft groups changed underlying KMeans identity"
    # Copy the SAME soft membership per paired window to target; no target fit.
    target_soft=soft.copy()
    assert np.array_equal(soft,target_soft)
    ent=-np.mean(np.sum(soft[tr]*np.log(np.clip(soft[tr],1e-300,1.)),axis=1))
    entropy_norm=ent/np.log(4)
    if not (.01<entropy_norm<.99):
        raise ValueError(f"Soft memberships too nearly hard/uniform: entropy={entropy_norm}")

    # Identical preselected held-out indices/groups to the existing hard OT audit.
    xx=x[te]; yy=y[te]; c=labels[te]
    rng=np.random.default_rng(20261117)
    m=int(hard["evaluation_unpaired_subset_n"])
    gi=rng.choice(len(xx),m,replace=False)
    gj=rng.choice(len(yy),m,replace=False)
    a_rng=np.random.default_rng(2026111701)
    b_rng=np.random.default_rng(2026111702)
    sample_n=int(hardot["group_sample_per_domain"])
    groups_x=[];groups_y=[]
    masses=np.bincount(c,minlength=4)/len(c)
    for k in range(4):
        eligible=np.flatnonzero(c==k)
        assert len(eligible)>=sample_n
        groups_x.append(xx[a_rng.choice(eligible,sample_n,replace=False)])
        groups_y.append(yy[b_rng.choice(eligible,sample_n,replace=False)])
    baseline=evaluate(np.eye(4),xx,yy,c,xx[gi],yy[gj],groups_x,groups_y,masses)
    assert abs(baseline["global_W2_squared"]-hardot["baseline"]["unrestricted_W2_squared"])<1e-10
    assert abs(baseline["same_group_W2_squared"]-hardot["baseline"]["same_group_W2_squared_empirical_masses"])<1e-10

    # Trainable parameter only global alignment, original ZIP numerical solver.
    # Both domains have the same fixed inherited soft group memberships.
    trainidx=np.flatnonzero(tr)
    outcomes=[]
    for seed in args.seeds:
        rgen=np.random.default_rng(seed)
        ix=sample_groups(trainidx,labels,args.fit_per_group,rgen)
        iy=sample_groups(trainidx,labels,args.fit_per_group,rgen)
        # Support pool exactly matches Hard HiWA sampling for this seed.
        A=soft[ix]
        B=target_soft[iy]
        assert A.shape==B.shape==(4*args.fit_per_group,4)
        assert np.max(np.abs(A.sum(1)-1))<1e-12
        assert np.max(np.abs(B.sum(1)-1))<1e-12
        cfg=SolverConfig(
            maxiter=args.outer,tol=args.stop_tol,mu=args.mu,
            epsilon_group=args.group_gamma,
            epsilon_sample=args.sample_gamma,
            inner_maxiter=args.inner,inner_tol=.01,
            sinkhorn_maxiter=args.sample_sinkhorn,
            group_sinkhorn_maxiter=args.group_sinkhorn,
            sinkhorn_tol=1e-8,stopping="legacy",
        )
        t0=time.perf_counter()
        result=hierarchical_alignment(
            weighted_measures(x[ix],A),
            weighted_measures(y[iy],B),
            seed=seed,config=cfg,scale=True,
            retain_plans=args.store_local_plans)
        elapsed=time.perf_counter()-t0
        R=np.asarray(result["R"])
        P=np.asarray(result["P"])
        assert np.linalg.norm(R.T@R-np.eye(4),"fro")<1e-7
        assert np.isfinite(P).all()
        train_stats=np.asarray(result["history"])
        assert train_stats.shape[1]==3
        change=float(result["rotation_delta"])
        consensus=float(result["consensus_residual"])
        group_error=float(max(np.max(abs(P.sum(0)-.25)),np.max(abs(P.sum(1)-.25))))
        strict=bool(change<=.01 and consensus<=.04 and group_error<1e-5)
        ev=evaluate(R,xx,yy,c,xx[gi],yy[gj],groups_x,groups_y,masses)
        rec=dict(seed=int(seed),sample_pool_n=int(len(ix)),runtime_s=float(elapsed),
                 effective_soft_entropy_train=float(entropy_norm),
                 max_iterations=int(args.outer),
                 actually_executed_iterations=len(train_stats),
                 stopped_early=bool(result["converged"]),
                 strict_audit_passed=strict,
                 global_change=change,
                 local_global_consensus_fro=consensus,
                 group_marginal_error=group_error,
                 max_local_ot_marginal_error=float(result["max_marginal_error"]),
                 orthogonality_error=float(result["orthogonality_error"]),
                 learned_P=P.tolist(),
                 R=R.tolist(),metrics=ev,
                 final_zip_soft_training_objective_proxy=float(train_stats[-1,2]),
                 stop_reason=result["stop_reason"])
        outcomes.append(rec)
        (args.output/f"soft_seed{seed}.json").write_text(
            json.dumps(rec,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
        print(f"seed {seed} | iter {len(train_stats)} | {elapsed:.2f}s | "
              f"converged={strict} | OT={ev['same_group_W2_squared']:.6f} "
              f"R²={ev['R2']:.5f} | {result['stop_reason']}",flush=True)

    # Freeze results before images; figures never affect training or metrics.
    for x1 in outcomes:
        assert np.all(np.isfinite(list(x1["metrics"].values())))
    keys=list(baseline.keys())
    soft_stats={}
    hard_stats={}
    for metric in keys:
        vals=np.array([z["metrics"][metric] for z in outcomes])
        soft_stats[metric]={"mean":float(vals.mean()),
                            "sample_sd":float(vals.std(ddof=1)) if len(vals)>1 else 0.}
        hard_key=("unrestricted_W2_squared" if metric=="global_W2_squared"
                  else "same_group_W2_squared_empirical_masses" if metric=="same_group_W2_squared"
                  else "one_nn_shared_group" if metric=="one_nn_shared_group" else metric)
        fromhard=(hardot["mean_sd"] if "W2_squared" in hard_key
                  else json.loads((args.hard.parent/"frozen_label_metrics.json").read_text())["statistics"])
        if hard_key in fromhard:
            hard_stats[metric]={"mean":float(fromhard[hard_key]["mean"]),
                                "sample_sd":float(fromhard[hard_key].get("sample_sd",fromhard[hard_key].get("sd_sample",0.)))}
    # Store source memberships for inspectability, but not redundant target B.
    np.savez_compressed(args.output/"soft_group_memberships.npz",
                        train_soft=soft[tr].astype(np.float32),
                        test_soft=soft[te].astype(np.float32),
                        centers_normalized=centers.astype(np.float32),
                        shared_hard_label_sha256=np.frombuffer(hashlib.sha256(labels.tobytes()).digest(),dtype=np.uint8))
    summary=dict(dataset="DANDI:000138/0.220113.0407",
         approach="Source-only fixed-center distance-softmax + paired target membership + archived GCOT hierarchical_alignment (scale=True)",
         archive_source_code="projects/dandi-000138-hiwa-4d/code/zip_gcot",
         not_original_TPAMI_independent_prototype_fitting=True,
         not_fully_unpaired_training=True,
         labels_remain_original_hard_kmeans=True,
         input_sha256=hashlib.sha256(args.fixed.read_bytes()).hexdigest(),
         hard_result_sha256=hashlib.sha256(args.hard.read_bytes()).hexdigest(),
         tau=args.tau,soft_train_entropy_normalized=float(entropy_norm),
         training_group_soft_mass=soft[tr].mean(0).tolist(),
         original_hard_group_sizes=np.bincount(labels[tr],minlength=4).tolist(),
         frozen_baseline=baseline,soft_seeds=outcomes,
         hard_mean_sd=hard_stats,soft_mean_sd=soft_stats,
         runtime_note="Full soft support on the 4*fit_per_group selected windows, unlike original Hard HiWA per-group hard 40-window support",
         same_test_subsets_used=True,
         original_training_algorithm_not_modified=True,
         config={k:v for k,v in vars(args).items() if not isinstance(v,Path)})
    (args.output/"SOFT_HARD_COMPARISON.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    if 42 in args.seeds:
        Rh=np.asarray(next(s for s in hard["seed_results"] if s["seed"]==42)["R"])
        Rs=np.asarray(next(s for s in outcomes if s["seed"]==42)["R"])
        plot_comparison(xx,yy,c,Rh,Rs,args.output)
    f,axs=plt.subplots(1,3,figsize=(13.5,4.4),constrained_layout=True)
    data_keys=["global_W2_squared","same_group_W2_squared","R2"]
    for ax,key in zip(axs,data_keys):
        hard_b=hard_stats[key]["mean"];soft_b=soft_stats[key]["mean"]
        ax.bar(["Identity","Hard","Soft"],[baseline[key],hard_b,soft_b],
               color=["#666666","#4C78A8","#F58518"])
        ax.set(title=key,ylabel="cost" if key.endswith("squared") else "R2")
        ax.grid(axis="y",alpha=.13)
    f.suptitle("One fixed source KMeans, identical held-out OT and prediction metrics")
    f.savefig(args.output/"soft_hard_global_groupOT_R2.png",dpi=205,bbox_inches="tight")
    plt.close(f)

    metrics_labels=[
      ("Global OT W2^2","global_W2_squared"),
      ("Fixed SAME-GROUP OT W2^2","same_group_W2_squared"),
      ("MAE","MAE"),("R2","R2"),("1NN inherited-label","one_nn_shared_group")]
    lines=["# 软分组与硬分组对照：DANDI 000138 同一四维同一标签体系","",
      "**严格协议：**源域固定一次 KMeans 得到四个原型，",
      "目标软隶属度直接继承同步源时间窗的软隶属度；",
      "所有可视化仍用原来相同的 argmax 硬标签着色，不存在目标域独立聚类。",
      "软求解使用压缩包的原始 GCOT 分层 OT 函数与归一化加权测度，",
      "额外关闭源/目标各自独立学习原型的步骤以满足既定协议。",
      "**因此这是一种 source-constrained Soft-GCOT 变体，而非 TPAMI 原作者的无配对配置。**",
      "",
      f"温度 tau={args.tau:g}，训练软分组平均归一化熵={entropy_norm:.4f}；",
      f"每次用于原版软算法的样本池={4*args.fit_per_group}，每个组在这些样本上都有正的概率质量。",
      f"外循环最多={args.outer}、局部交替最多={args.inner}、样本 Sinkhorn 最多={args.sample_sinkhorn}；"
      "若与 Hard 的 500/20/45 不一致，属计算预算不同的试验性对照，不能仅凭此声称 soft 更优。",
      "",
      "| 指标 | Identity | Hard mean | Soft mean ± sample SD |",
      "|---|---:|---:|---:|"]
    for name,key in metrics_labels:
        lines.append(f"| {name} | {baseline[key]:.6f} | "
                     f"{hard_stats[key]['mean']:.6f} | "
                     f"{soft_stats[key]['mean']:.6f} ± {soft_stats[key]['sample_sd']:.6f} |")
    lines.extend(["","## 每个软模型 seed","",
                  "| seed | 4D全域OT W2² | 固定组OT W2² | 1NN | MAE | R2 | 收敛审计 |",
                  "|---:|---:|---:|---:|---:|---:|:---:|"])
    for row in outcomes:
        z=row["metrics"]
        lines.append(f"| {row['seed']} | {z['global_W2_squared']:.6f} | "
                     f"{z['same_group_W2_squared']:.6f} | {z['one_nn_shared_group']:.4f} | "
                     f"{z['MAE']:.5f} | {z['R2']:.5f} | "
                     f"{'PASS' if row['strict_audit_passed'] else 'NO'} |")
    lines.extend(["", "## 限制","",
      "- 只有全部种子严格满足收敛与相同数值预算时才能给出强对照结论，"
      "未通过的结果必须如实列出。",
      "- Hard 使用原版 PyHiWA，Soft 使用同一上传压缩包中的 GCOT 求解器，"
      "除了隶属度，求解器数值实现也存在差异。因此当前是方法级对照，不是"
      "仅替换 membership 的纯消融。",
      "- 共享组标签来自真实样本配对；1NN 不是独立运动标签分类指标。",
      "- 模型最终仍输出全局正交旋转，OT 计划负责学习和优化而不是直接将样本搬成新点云。",
      "",
      "![Hard vs Soft OT/R2](soft_hard_global_groupOT_R2.png)",
      "",
      "![Fixed-group Hard/Soft/Target scatter comparison](hard_soft_target_six_pairs.png)",
      "",
      "## 可复现性","",
      f"源数据哈希：{summary['input_sha256']}",
      f"Hard 实验哈希：{summary['hard_result_sha256']}",
      ""])
    (args.output/"SOFT_HARD_COMPARISON.md").write_text("\n".join(lines),encoding="utf8")
    print("SOFT HARD SUMMARY:",json.dumps({"hard":hard_stats,"soft":soft_stats,
          "convergence":[(z["seed"],z["strict_audit_passed"],
                          z["actually_executed_iterations"]) for z in outcomes]},indent=2),flush=True)


if __name__=="__main__":
    main()
