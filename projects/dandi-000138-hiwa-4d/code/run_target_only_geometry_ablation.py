"""Train-only, target-domain-only geometric transforms for STRICT UNPAIRED DANDI138.

Original X unchanged. Original Y is normalized 4D hand [posxy,velxy].
Explore, PREDECLARED (no data-driven cross-domain pairing, no test tuning):
  identity: original unchanged.
  block_zca: shrinkage ZCA independently in pos2 and vel2 (lambda .15).
  full_zca: shrinkage ZCA on all 4 target coordinates (lambda .15).
  arcsinh: invertible coordinatewise asinh marginal tail compression;
          median and absolute-75%-quantile fitted on TRAIN Y alone.

After each mapping, centered target has train RMS radius 1. For each
variant the target learns its own independent Hard KMeans and Soft
prototype memberships. Source hard/soft and 4D X stay unchanged.
Full original same-solver hierarchy: 40 per domain-own hard group, same
seeds, original 500/20/45/200, group P fully optimized. Test synchronous
pairing is never available for fitting, training or transform selection.

Important: do NOT compare paired R2 computed in transformed target space
across variants; INVERSE target preprocessing for all predictions and
calculate R2/MAE in the same original Y-normalized coordinate system.
Transport cost in transformed target space is conditional on this metric;
also report paired inverse-space score and free OT in original hand space.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from itertools import combinations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import softmax,xlogy
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from zip_gcot.grouping import soft_groups
from zip_gcot.alignment import hierarchical_alignment,SolverConfig
from run_hard_hiwa import sample_groups
from run_strict_unsupervised_hard_soft import fixed_eval_indices,memberships_from_prototypes
from run_controlled_soft_hard_ablation import build_measures,heldout_nested

MODES=("identity","block_zca","full_zca","arcsinh")
COLORS=("#367fb9","#e4923c","#299873","#9b7dc4")

def invsqrt_cov(data,lam=.15):
    S=np.cov(data,rowvar=False)
    ev,V=np.linalg.eigh(S)
    ridge=lam*float(np.trace(S)/len(ev))
    return (V*(1/np.sqrt(np.maximum(ev,0)+ridge)))@V.T

def target_transform(y,train,mode):
    """Fit Y-alone, return transformed Y, inversion function and provenance."""
    if mode=="identity":
        z=y.copy()
        inverse=lambda a:np.asarray(a)
        settings={"type":"identity"}
    elif mode in ("block_zca","full_zca"):
        if mode=="full_zca":
            W=invsqrt_cov(y[train])
        else:
            W=np.zeros((4,4))
            for g in (slice(0,2),slice(2,4)):
                W[g,g]=invsqrt_cov(y[train,g])
        z=y@W
        WI=np.linalg.inv(W)
        inverse=lambda a:np.asarray(a)@WI
        settings={"type":mode,"ridge_cov_trace_fraction":.15,
                  "linear_map":W.tolist(),"inverse_linear_map":WI.tolist()}
    elif mode=="arcsinh":
        m=np.median(y[train],axis=0)
        sc=np.quantile(np.abs(y[train]-m),.75,axis=0)
        sc=np.maximum(sc,.08)
        z=np.arcsinh((y-m)/sc)
        inverse=lambda a:m+sc*np.sinh(np.asarray(a))
        settings={"type":"per_dimension_invertible_asinh","training_median":m.tolist(),
                  "training_absdev_75percent_scale":sc.tolist()}
    else:
        raise ValueError(mode)
    center=z[train].mean(axis=0)
    radius=float(np.sqrt(np.mean(np.sum((z[train]-center)**2,axis=1))))
    zz=(z-center)/radius
    def recover(a):
        return inverse(np.asarray(a)*radius+center)
    assert np.all(np.isfinite(zz))
    assert np.allclose(recover(zz[train[:len(zz)]]),y[train[:len(zz)]],
                       atol=1e-10,rtol=1e-9) if mode=="identity" else True
    assert np.max(np.abs(recover(zz[train])-y[train]))<1e-8
    settings.update({"post_transform_train_mean":center.tolist(),"post_transform_train_RMS":radius})
    return zz,recover,settings

def shapediagnostic(x,y,sel,seed=277):
    """Train-only ROTATION-INVARIANT covariance/euclidean-distance descriptor.
    No paired sample indices are used. These diagnostics are NOT objectives.
    """
    def eigen(a):
        D=np.linalg.eigvalsh(np.cov(a,rowvar=False))[::-1]
        D=np.maximum(D,0)
        return D/np.sum(D)
    xe=eigen(x[sel]);ye=eigen(y[sel])
    rng=np.random.default_rng(seed)
    r1=rng.integers(0,sel.sum(),size=(5000,2))
    r2=rng.integers(0,sel.sum(),size=(5000,2))
    xx=x[sel];yy=y[sel]
    dx=np.linalg.norm(xx[r1[:,0]]-xx[r1[:,1]],axis=1)
    dy=np.linalg.norm(yy[r2[:,0]]-yy[r2[:,1]],axis=1)
    q=np.array([.10,.25,.50,.75,.90])
    return dict(
       source_cov_eigs=xe.tolist(),target_cov_eigs=ye.tolist(),
       covariance_spectrum_L2=float(np.linalg.norm(xe-ye)),
       source_dist_quantiles=np.quantile(dx,q).tolist(),
       target_dist_quantiles=np.quantile(dy,q).tolist(),
       distance_quantile_L2=float(np.linalg.norm(np.quantile(dx,q)-np.quantile(dy,q))),
       quantile_levels=q.tolist())

def postfit_original_space(x,y_original,R,recover,ix,iy):
    """Shared 4D original target coordinates, never used to update fit/parameters."""
    z=recover(x@R.T)
    if not np.all(np.isfinite(z)):raise ValueError("Nonfinite inverse physical prediction")
    e=z-y_original
    C=cdist(z[ix],y_original[iy],"sqeuclidean")
    i,j=linear_sum_assignment(C)
    sse=float((e*e).sum())
    return dict(original_target_sync_R2=float(1-sse/np.sum((y_original-y_original.mean(0))**2)),
         original_target_MAE=float(np.abs(e).mean()),
         original_target_mean_euclidean=float(np.linalg.norm(e,axis=1).mean()),
         original_target_relative_sse=float(sse/np.sum(y_original*y_original)),
         original_target_free_W2sq=float(np.mean(C[i,j])))

def new_scope_figures(ysets,x,train,test,hard_source,archive,out):
    modes=list(ysets)
    if len(modes)<2:return
    pairs=list(combinations(range(4),2))
    for scope,sel in [("TRAIN_14000",train),("TEST_3500",test)]:
        fig,axes=plt.subplots(6,len(modes),figsize=(4.0*len(modes),20.0),constrained_layout=True)
        for col,mode in enumerate(modes):
            y=ysets[mode]
            labs=hard_source[mode]
            for row,(i,j) in enumerate(pairs):
                ax=axes[row,col] if len(modes)>1 else axes[row]
                for k in range(4):
                    mask=sel&(labs==k)
                    ax.scatter(y[mask,i],y[mask,j],
                        s=.8 if scope=="TRAIN_14000" else 2.1,
                        alpha=.21 if scope=="TRAIN_14000" else .44,
                        linewidths=0,rasterized=True,c=COLORS[k])
                ax.set(xlabel=f"dim{i+1}",ylabel=f"dim{j+1}",
                       title=f"{mode} | ({i+1},{j+1})")
                ax.grid(alpha=.09)
        fig.suptitle(f"Independent target-domain KMeans groups AFTER TRAIN-ONLY feature transform | {scope}\n"
                 "No target groups are copied from source; six 2D projections of unchanged 4D sample count",
                 y=1.02,fontsize=13)
        fig.savefig(out/f"TARGET_TRANSFORMS_{scope}_six_projections.png",
                    dpi=145,bbox_inches="tight")
        plt.close(fig)

    fig,axes=plt.subplots(2,len(modes),figsize=(4.1*len(modes),7.7),constrained_layout=True)
    for col,mode in enumerate(modes):
        target=ysets[mode]
        lx=hard_source[mode]
        for row,sel in enumerate([train,test]):
            ax=axes[row,col]
            for k in range(4):
                ix=sel&(lx==k)
                ax.scatter(target[ix,0],target[ix,1],s=1 if row==0 else 2,
                           alpha=.22 if row==0 else .47,linewidths=0,
                           c=COLORS[k],rasterized=True)
            ax.set(title=f"{mode}: {['14k train','3.5k test'][row]}",
                   xlabel="Target feature 1",ylabel="Target feature 2")
    fig.suptitle("Same original hand windows, different invertible TRAIN-ONLY target transformations")
    fig.savefig(out/"TARGET_TRANSFORMS_train_test_overview.png",dpi=190,bbox_inches="tight")
    plt.close(fig)

def compare_plots(results,out):
    var=list(results["variants"])
    methods=("hard","soft")
    fig,axs=plt.subplots(2,2,figsize=(14.2,9.4),constrained_layout=True)
    for ax,(route,title) in zip(axs.flat,[
         (("shared_original_eval","original_target_sync_R2"),"R2 postfit INVERTED to original target"),
         (("shared_original_eval","original_target_free_W2sq"),"Free W2² postfit original target"),
         (("transport","total_transport_W2sq"),"Nested P-Q cost IN TRANSFORMED target"),
         (("shared_original_eval","original_target_MAE"),"MAE postfit INVERTED to original target")]):
        for j,method in enumerate(methods):
            ys=[];ds=[]
            for item in var:
                rec=results["variants"][item]["runs"][method]
                v=np.array([r[route[0]][route[1]] for r in rec],float)
                ys.append(v.mean());ds.append(v.std(ddof=1) if len(v)>1 else 0.)
            offsets=(j-.5)*.15
            ax.errorbar(np.arange(len(var))+offsets,ys,yerr=ds,fmt="o",
                        capsize=3,label=method.upper())
        ax.set(xticks=np.arange(len(var)),xticklabels=var,title=title)
        ax.grid(alpha=.13)
    axs[0,0].legend()
    fig.suptitle("Target-only geometry tests | same original 4D raw target error metric | mean ± seed SD")
    fig.savefig(out/"TRANSFORMS_HARD_SOFT_OT_AND_ORIGINAL_SPACE_ERRORS.png",
                dpi=190,bbox_inches="tight")
    plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--clusters",type=Path,required=True)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--modes",nargs="+",default=list(MODES))
    p.add_argument("--seeds",nargs="+",type=int,default=[42,43,44,45,46])
    p.add_argument("--outer",type=int,default=500)
    p.add_argument("--inner",type=int,default=20)
    p.add_argument("--sample-sinkhorn",type=int,default=45)
    p.add_argument("--group-sinkhorn",type=int,default=200)
    p.add_argument("--eval-Q-iters",type=int,default=120)
    p.add_argument("--prototype-iters",type=int,default=300)
    p.add_argument("--pilot",action="store_true")
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    modes=[m for m in args.modes if m in MODES]
    if len(modes)!=len(args.modes) or len(set(modes))!=len(modes):
        raise ValueError("Unknown/duplicate modes")
    baseline=json.loads(args.baseline.read_text(encoding="utf8"))
    assert baseline["source_and_target_hard_kmeans_independent"]
    assert baseline["source_and_target_soft_prototypes_independent"]
    assert baseline["no_synchronized_pair_ids_used_in_fitting"]
    assert args.clusters.name=="clustered_4d.npz"
    assert hashlib.sha256(args.clusters.read_bytes()).hexdigest()==baseline["input_sha256"]
    with np.load(args.clusters,allow_pickle=False) as d:
        assert "shared_labels" not in d.files
        X=np.asarray(d["neural4"],float);Y=np.asarray(d["hand4"],float)
        train=np.asarray(d["train_mask"],bool);test=np.asarray(d["test_mask"],bool)
        lx=np.asarray(d["neural_groups"],int);old_ly=np.asarray(d["hand_groups"],int)
        trial=np.asarray(d["trial_of_window"],int)
    assert X.shape==Y.shape==(17500,4)
    assert train.sum()==14000 and test.sum()==3500
    assert len(np.unique(trial[train]))==400 and len(np.unique(trial[test]))==100
    x=(X-np.asarray(baseline["source_mean"]))/float(baseline["source_radius"])
    y0=(Y-np.asarray(baseline["target_mean"]))/float(baseline["target_radius"])
    assert not np.array_equal(lx,old_ly)
    # Original source SOFT prototypes fixed and independent of target Y.
    CX=np.asarray(baseline["soft_prototypes"]["source_centers"],float)
    AX=memberships_from_prototypes(x,CX)
    models={}
    yplot={};lplot={}
    for m in modes:
        yy,recover,detail=target_transform(y0,train,m)
        if m=="identity":
            assert np.max(np.abs(yy-y0))<1e-10
            target_labels=old_ly.copy()
            CY=np.asarray(baseline["soft_prototypes"]["target_centers"])
            BY=memberships_from_prototypes(yy,CY)
        else:
            km=KMeans(n_clusters=4,random_state=42,n_init=20).fit(yy[train])
            target_labels=km.predict(yy)
            by,CY,info=soft_groups(yy[train],4,seed=43,entropy_weight=1.,
                                  maxiter=args.prototype_iters)
            BY=memberships_from_prototypes(yy,CY)
            assert np.allclose(BY[train],by,atol=1e-8)
        assert len(np.unique(target_labels[train]))==4
        assert not np.array_equal(lx,target_labels)
        assert not np.array_equal(AX,BY)
        models[m]=(yy,recover,target_labels,BY,CY,detail)
        yplot[m]=yy;lplot[m]=target_labels
        print(f"TRAIN ONLY target transform {m}: "
              f"target group counts={np.bincount(target_labels[train],minlength=4).tolist()} "
              f"target PCA eigen ratio={shapediagnostic(x,yy,train)['target_cov_eigs']}",flush=True)
    new_scope_figures(yplot,x,train,test,lplot,baseline,args.output)

    cfg=SolverConfig(maxiter=args.outer,tol=1e-9,mu=.05,
          epsilon_group=.2,epsilon_sample=.1,
          inner_maxiter=args.inner,inner_tol=.01,
          sinkhorn_maxiter=args.sample_sinkhorn,
          group_sinkhorn_maxiter=args.group_sinkhorn,
          sinkhorn_tol=1e-8,stopping="legacy")
    records={"research_question":"Can TARGET-ONLY geometry transforms improve independent unpaired 4D OT alignment?",
       "fixed_dataset_sha256":baseline["input_sha256"],
       "same_original_source":True,"hard_X_groups_fixed":True,
       "target_groups_refit_from_TARGET_FEATURES_only":True,
       "soft_target_prototypes_independently_fitted":True,
       "true_synchronous_pairing_is_postfit_evaluation_only":True,
       "no_test_hyperparameter_selection":True,
       "modes_are_predeclared":True,
       "target_transform_shrinkage_lambda":.15,
       "training_windows":14000,"test_windows":3500,"OT_windows_each_domain":160,
       "settings":{k:v for k,v in vars(args).items() if not isinstance(v,Path)},
       "baseline_identity_mean":baseline["summary"],"variants":{}}
    tridx=np.flatnonzero(train)
    xt=x[test];yot=y0[test]
    gi,gj,pi,pj=fixed_eval_indices(int(test.sum()))
    for mode in modes:
        yy,recover,ly,BY,CY,detail=models[mode]
        runs={"hard":[],"soft":[]}
        diag=shapediagnostic(x,yy,train)
        # Unpaired train shape quantities are the ONLY diagnostics available
        # before training; neither optimize parameters nor select by test.
        counts=np.bincount(ly[train],minlength=4).tolist()
        for seed in args.seeds:
            rg=np.random.default_rng(seed)
            idsx=sample_groups(tridx,lx,40,rg)
            idsy=sample_groups(tridx,ly,40,rg)
            assert np.unique(idsx).size==np.unique(idsy).size==160
            for method in ("hard","soft"):
                hard=method=="hard"
                a=np.eye(4)[lx[idsx]] if hard else AX[idsx]
                b=np.eye(4)[ly[idsy]] if hard else BY[idsy]
                st=time.perf_counter()
                fit=hierarchical_alignment(
                    build_measures(x[idsx],a,hard),
                    build_measures(yy[idsy],b,hard),
                    seed=seed,config=cfg,scale=True,retain_plans=False)
                R=np.asarray(fit["R"]);P=np.asarray(fit["P"])
                assert np.linalg.norm(R.T@R-np.eye(4),"fro")<1e-7
                assert max(abs(P.sum(0)-.25).max(),abs(P.sum(1)-.25).max())<1e-5
                # INVERSE all transformed target estimates into SAME
                # original normalized 4D hand coordinates before comparison.
                ev=postfit_original_space(xt,yot,R,recover,gi,gj)
                transformed_test=yy[test]
                Atest=np.eye(4)[lx[test][pi]] if hard else AX[test][pi]
                Btest=np.eye(4)[ly[test][pj]] if hard else BY[test][pj]
                oth=heldout_nested(xt[pi],transformed_test[pj],
                                  Atest,Btest,P,R,.1,args.eval_Q_iters,hard)
                rec=dict(seed=int(seed),method=method,R=R.tolist(),P=P.tolist(),
                  original_source_sample_hash=hashlib.sha256(idsx.tobytes()).hexdigest(),
                  original_target_sample_hash=hashlib.sha256(idsy.tobytes()).hexdigest(),
                  iterations=len(fit["history"]),stop_reason=fit["stop_reason"],
                  strict_admm_pass=bool(fit["criteria_satisfied"]),
                  rotation_delta=float(fit["rotation_delta"]),
                  consensus_residual=float(fit["consensus_residual"]),
                  dual_residual=float(fit["dual_residual"]),
                  marginal_error=float(fit["marginal_error"]),
                  elapsed_sec=float(time.perf_counter()-st),
                  shared_original_eval=ev,
                  transport=oth)
                # Exact identity reproduction with original full-budgets.
                if (mode=="identity" and args.outer==500 and
                    args.inner==20 and args.sample_sinkhorn==45 and
                    args.group_sinkhorn==200 and args.prototype_iters==300):
                    old=next(t for t in baseline["runs"][method] if t["seed"]==seed)
                    assert np.max(np.abs(R-np.asarray(old["R"])))<1e-9
                    assert np.max(np.abs(P-np.asarray(old["P"])))<1e-9
                    assert abs(oth["total_transport_W2sq"]-
                       old["heldout_after"]["total_transport_W2sq"])<1e-8
                    assert abs(ev["original_target_sync_R2"]-
                       old["metrics"]["sync_R2"])<1e-9
                runs[method].append(rec)
                print(f"FINAL {mode}/{method}/seed{seed}: "
                     f"shapeL2={diag['covariance_spectrum_L2']:.4g} "
                     f"OT_trans={oth['total_transport_W2sq']:.5f} "
                     f"ORIGINAL_R2={ev['original_target_sync_R2']:.5f} "
                     f"ORIGINAL_freeW2={ev['original_target_free_W2sq']:.5f} "
                     f"strictADMM={rec['strict_admm_pass']}",flush=True)
        records["variants"][mode]=dict(
           transform=detail,geometry_shape_diagnostic_train_only=diag,
           independent_target_group_counts_train=counts,
           target_soft_prototypes=CY.tolist(),
           runs=runs)
        (args.output/"TARGET_TRANSFORM_RESULTS.json").write_text(
            json.dumps(records,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    compare_plots(records,args.output)
    lines=["# DANDI:000138 | 目标域单侧无监督几何处理实验",
           "",
           "**仅修改目标域的四维表示，神经域和原试次/窗口/求解器不变。**每一种目标变换仅使用14,000训练点的目标边际特征估计，禁止源目标同步索引和测试R²参与拟合或筛选。处理后Hard的目标KMeans和Soft目标原型各自重新训练；源域原分组固定。",
           "",
           "预声明的4种方案：identity（原始基线）、block_zca（位置和速度各自2D shrinkage ZCA，λ=.15）、full_zca（4D shrinkage ZCA，λ=.15）、arcsinh（逐维训练集median和75%绝对偏差尺度做可逆反双曲正弦，压缩长尾，**不是解码跨模态非线性**）。每种变换后只在训练集上居中，缩放至RMS半径1。",
           "",
           "**数据层级**：17500全部，14000训练，3500测试，求解器每域160点、40/组；K=4、seed 42–46、outer 500、inner 20、sample Sinkhorn45、group Sinkhorn200、µ=.05、eps_s=.1、eps_g=.2。",
           "",
           "## 无配对训练形状诊断（旋转不变量；不能证明点对关系）",
           "",
           "| 目标处理 | 源/目标协方差特征值比例 L2 差 ↓ | 两域点内距离分位数 L2 差 ↓ | 目标训练四组点数 |",
           "|---|---:|---:|---|"]
    for m in modes:
        entry=records["variants"][m]
        dia=entry["geometry_shape_diagnostic_train_only"]
        lines.append(f"| {m} | {dia['covariance_spectrum_L2']:.5f} | "
             f"{dia['distance_quantile_L2']:.5f} | "
             f"{entry['independent_target_group_counts_train']} |")
    lines+=["",
       "## 完全相同原始目标坐标下的逆变换后测试指标",
       "",
       "| 目标处理 | 方法 | 原始Y空间R²↑ | 原始Y空间 MAE↓ | 原始Y空间自由W2²↓ | 处理后分层P-Q代价↓（不同度量不可直接排名） | 严格ADMM通过次数 |",
       "|---|---|---:|---:|---:|---:|---:|"]
    for m in modes:
        for method in ("hard","soft"):
            runs=records["variants"][m]["runs"][method]
            def mstd(section,key):
                v=np.array([r[section][key] for r in runs])
                return f"{v.mean():.5f}±{v.std(ddof=1):.5f}" if len(v)>1 else f"{v.mean():.5f}"
            lines.append(f"| {m} | {method} | "
               f"{mstd('shared_original_eval','original_target_sync_R2')} | "
               f"{mstd('shared_original_eval','original_target_MAE')} | "
               f"{mstd('shared_original_eval','original_target_free_W2sq')} | "
               f"{mstd('transport','total_transport_W2sq')} | "
               f"{sum(s['strict_admm_pass'] for s in runs)}/{len(runs)} |")
    lines+=["",
      "## 图像：完整3500测试点和14000训练点（六维投影组合）",
      "",
      "![3500目标域单侧变换](TARGET_TRANSFORMS_TEST_3500_six_projections.png)",
      "",
      "![14000目标域单侧变换](TARGET_TRANSFORMS_TRAIN_14000_six_projections.png)",
      "",
      "![训练与测试结构](TARGET_TRANSFORMS_train_test_overview.png)",
      "",
      "![所有同求解器Hard/Soft模型测试误差](TRANSFORMS_HARD_SOFT_OT_AND_ORIGINAL_SPACE_ERRORS.png)",
      "",
      "## 阅读注意",
      "",
      "- 协方差谱/分位数只看单模态自身样本，与另一个域的单点真实对应关系无关；形状接近不等于存在准确点对映射。",
      "- 目标单侧协方差白化改变了OT使用的欧氏度量，**处理后P-Q运输代价不能直接跨方案认定优劣**；共同的比较在逆变换回原始Y后的自由OT、配对误差。",
      "- 同步R²严格在全部训练完成后才计算，**没有使用R²选模型/调变换**。三种变换是预定探索、没有超参搜索及选择。",
      "- 原GCOT旋转R只能作正交映射，任何单侧处理都不能保证‘消去’未知非线性；若运动信号和神经活动真实关系为非线性，则可能需要另立非线性跨模态映射模型。",
      "- Soft未必满足严格ADMM残差收敛；对任何未收敛的结果只解释为给定相同计算预算下的数值结果。",
      "",
      "[脚本](../../code/run_target_only_geometry_ablation.py) | [全部逐种子R/P、逆变换误差与预处理参数JSON](TARGET_TRANSFORM_RESULTS.json)",
      ""]
    (args.output/"TARGET_TRANSFORM_REPORT.md").write_text("\n".join(lines),encoding="utf8")
    print("TARGET TRANSFORM STUDY COMPLETE",flush=True)
if __name__=="__main__":
    main()
