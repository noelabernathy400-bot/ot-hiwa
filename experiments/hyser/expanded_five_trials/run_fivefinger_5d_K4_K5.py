#!/usr/bin/env python3
"""Controlled 5-fold Hyser 1250-window 5D five-finger Hyser Hard vs Soft, K=4 or K=5 pilot.

Five trial-disjoint folds (1000 training vs 250 testing windows in each domain).
All 1000 training points inform independent KMeans and independent quantization
of each domain; for tractability each 5D distribution is represented by 96
weighted KMeans anchors in the inner Q optimizer. This five-finger five-dimensional experiment uses the SAME frozen 5D data for both K choices; compared within the same K across Hard/Soft. This is a disclosed
approximation to the full-data hierarchical OT, NOT the original HiWA ADMM.

The same GLOBAL orthogonal R, two-level P/Q, and entropy objective are
optimized for Hard and Soft via alternating block coordinate descent.
No paired correspondence, target labels copied from source, physical labels,
or heldout test points are used during training.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import root
from scipy.special import softmax,xlogy
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"unsupervised_control"))
sys.path.insert(0,str(HERE.parent/"scripts"))
from solver import sinkhorn,polar

def precise_group_entropic_OT(cost,eps=.2):
    # Dual root for the SAME entropy-regularized KxK transportation problem.
    c=cost-cost.min(axis=1,keepdims=True)
    c=c-c.min(axis=0,keepdims=True)
    def plan(z):
        f=z[:K];g=np.r_[z[K:],0]
        return np.exp(np.clip((f[:,None]+g[None,:]-c)/eps,-700,700))
    def res(z):
        p=plan(z)
        return np.r_[p.sum(axis=1)-1/K,p.sum(axis=0)[:K-1]-1/K]
    def jac(z):
        p=plan(z);J=np.zeros((2*K-1,2*K-1))
        J[:K,:K]=np.diag(p.sum(axis=1)/eps)
        J[:K,K:]=p[:,:K-1]/eps
        J[K:,:K]=p[:,:K-1].T/eps
        J[K:,K:]=np.diag(p.sum(axis=0)[:K-1]/eps)
        return J
    sol=root(res,np.zeros(2*K-1),jac=jac,method='hybr',options={'xtol':1e-12,'maxfev':400})
    p=plan(sol.x)
    if np.max(np.abs(res(sol.x)))>5e-8 or not np.isfinite(p).all():
        p=sinkhorn(np.full(K,1/K),np.full(K,1/K),c,eps,100000,1e-10)
    err=max(np.abs(p.sum(0)-1/K).max(),np.abs(p.sum(1)-1/K).max())
    if err>1e-6: raise ArithmeticError(f'P infeasible {err}')
    return p



K=5
ES=.1
EG=.2
TAU=.5
DIV=np.sqrt(5)

def make_domain_memberships(x, method, center):
    if method=="hard":
        lab=np.argmin(cdist(x,center,metric="sqeuclidean"),axis=1)
        return np.eye(K)[lab]
    if method=="soft05":
        return softmax(-cdist(x,center,metric="sqeuclidean")/TAU,axis=1)
    raise ValueError(method)

def quantize_group_measure(x,A,n_anchors,seed):
    """Per-group conditional probabilities on the same unsupervised anchors.

    All original training samples contribute to mean locations AND masses.
    Shared anchor positions are used for both Hard and Soft; grouping differs
    only in original sample membership weights.
    """
    km=KMeans(n_clusters=n_anchors,n_init=5,random_state=seed,max_iter=120)
    assignments=km.fit_predict(x)
    centers=km.cluster_centers_
    mass=np.zeros((K,n_anchors))
    for k in range(K):
        mass[k]=np.bincount(assignments,weights=A[:,k],minlength=n_anchors)
        mass[k]/=mass[k].sum()
    distortion=np.mean(np.sum((x-centers[assignments])**2,axis=1))
    groups=[]
    for k in range(K):
        ids=np.where(mass[k] >1e-14)[0]
        prob=mass[k,ids].copy()
        prob/=prob.sum()
        groups.append((ids,prob))
    return centers,groups,float(distortion)

def one_sinkhorn(a,b,C,p,iterations=100):
    if p < 1e-6:
        q=np.outer(a,b)  # limiting Q as epsilon_sample/p -> +inf
    else:
        q=sinkhorn(a,b,C,ES/max(p,1e-15),iterations,1e-8)
    if not np.isfinite(q).all(): raise ArithmeticError("Non-finite inner Sinkhorn")
    return q

def local_plan_and_cost(xs,ys,gx,gy,R,P,iterations=90):
    D=cdist(xs@R.T,ys,metric="sqeuclidean")
    costs=np.zeros((K,K)); qplans={}
    max_err=0.
    for k,(ik,a) in enumerate(gx):
        for l,(il,b) in enumerate(gy):
            C=D[np.ix_(ik,il)]
            q=one_sinkhorn(a,b,C,P[k,l],iterations)
            qplans[k,l]=q
            costs[k,l]=np.sum(q*C)
            max_err=max(max_err,abs(q.sum(0)-b).max(),abs(q.sum(1)-a).max())
    return costs,qplans,float(max_err)

def accumulate_cross(xs,ys,gx,gy,P,qs):
    weighted=np.zeros((len(xs),len(ys)))
    for k,(ik,_) in enumerate(gx):
        for l,(il,_) in enumerate(gy):
            weighted[np.ix_(ik,il)]+=P[k,l]*qs[k,l]
    return ys.T @ weighted.T @ xs

def calculate_objective(xs,ys,gx,gy,R,P,inner_iter=130):
    C,qs,err=local_plan_and_cost(xs,ys,gx,gy,R,P,inner_iter)
    transport=float(np.sum(P*C))
    sample_H=float(ES*sum(float(np.sum(xlogy(q,q))) for q in qs.values()))
    group_H=float(EG*np.sum(xlogy(P,P)))
    return {"T":transport,"sample_entropy":sample_H,"group_entropy":group_H,
            "J":transport+sample_H+group_H,
            "marginal_max":float(max(err,np.abs(P.sum(0)-1/K).max(),np.abs(P.sum(1)-1/K).max())),
            "P":P.tolist(),"C":C.tolist(),
            "Q_shapes":{f"{k}-{l}":list(q.shape) for (k,l),q in qs.items()},
            "Q_entropy":{f"{k}-{l}":float(np.sum(xlogy(q,q))) for (k,l),q in qs.items()}}
 
def fit_orthogonal_hierarchical(xs,ys,gx,gy,seed,max_iter=50):
    # SAME mathematical objective for Hard and Soft; globally shared R.
    rng=np.random.default_rng(seed)
    R=polar(rng.normal(size=(5,5)))
    P=np.full((K,K),1/(K*K))
    history=[]
    for it in range(max_iter):
        C,qs,err=local_plan_and_cost(xs,ys,gx,gy,R,P,iterations=100)
        newP=precise_group_entropic_OT(C,EG)
        cross=accumulate_cross(xs,ys,gx,gy,newP,qs)
        newR=polar(cross)
        dr=float(np.linalg.norm(newR-R));dp=float(np.linalg.norm(newP-P))
        R,P=newR,newP
        history.append({"iter":it+1,"rotation_delta":dr,"P_delta":dp,"inner_marginal_error":err})
        if it>=8 and dr<.002 and dp<.002 and err<1e-6:
            break
    # Optimize P,Q on the FINAL R to ensure comparison at a common R.
    for _ in range(3):
        C,qs,err=local_plan_and_cost(xs,ys,gx,gy,R,P,iterations=130)
        P=precise_group_entropic_OT(C,EG)
    fit=calculate_objective(xs,ys,gx,gy,R,P)
    before=calculate_objective(xs,ys,gx,gy,np.eye(5),P)
    status={
        "num_iterations":len(history),"last_step":history[-1],
        "simple_numeric_convergence":bool(
            len(history)<max_iter and history[-1]["rotation_delta"]<.002
            and history[-1]["P_delta"]<.002 and fit["marginal_max"]<1e-6),
        "max_rotation_orthogonality_error":float(np.linalg.norm(R.T@R-np.eye(5))),
        "training_result_at_final_global_R":fit,
        "training_before_identity_same_P":before,
        "optimizer_caveat":"Alternating block optimization of GLOBAL-R constrained hierarchical OT, not original local-R ADMM or global optimality certificate."}
    return R,P,status,history

def test_shared_reference(Xtr,Ytr,Xte,Yte,R,reference_centers_x,reference_centers_y):
    # Kmeans centers trained on training trials only. Test cluster labels
    # predicted separately; no use of synchronized pair correspondence.
    A=make_domain_memberships(Xte,"hard",reference_centers_x)
    B=make_domain_memberships(Yte,"hard",reference_centers_y)
    mx=[];my=[]
    for sample, membership, out in [(Xte,A,mx),(Yte,B,my)]:
        for k in range(K):
            ids=np.flatnonzero(membership[:,k]>.5)
            if len(ids)==0:
                return {"error":f"Empty heldout cluster {k}"}
            out.append((sample[ids],np.ones(len(ids))/len(ids)))
    def one(r):
        P=np.full((K,K),1/(K*K))
        for _ in range(5):
            C=np.zeros((K,K));errs=0.
            for k,(x,a) in enumerate(mx):
                for l,(y,b) in enumerate(my):
                    cost=cdist(x@r.T,y,metric="sqeuclidean")
                    q=one_sinkhorn(a,b,cost,P[k,l],100)
                    C[k,l]=np.sum(q*cost)
                    errs=max(errs,abs(q.sum(0)-b).max(),abs(q.sum(1)-a).max())
            P=precise_group_entropic_OT(C,EG)
        # At final P, account for Q entropy terms
        total=0.;sample_h=0.
        for k,(x,a) in enumerate(mx):
            for l,(y,b) in enumerate(my):
                c=cdist(x@r.T,y,metric="sqeuclidean")
                q=one_sinkhorn(a,b,c,P[k,l],120)
                total+=P[k,l]*np.sum(q*c)
                sample_h+=ES*np.sum(xlogy(q,q))
        gh=EG*np.sum(xlogy(P,P))
        return {"T":float(total),"J":float(total+sample_h+gh),"sample_H":float(sample_h),
                "group_H":float(gh),"P":P.tolist(),
                "row_col_error":float(max(abs(P.sum(0)-1/K).max(),abs(P.sum(1)-1/K).max()))}
    return {"before_identity":one(np.eye(5)),"after":one(R)}

def w2(x,y,r):
    C=cdist(x@r.T,y,metric="sqeuclidean")
    ii,jj=linear_sum_assignment(C)
    return float(C[ii,jj].mean())

def main():
    global K
    args=argparse.ArgumentParser()
    args.add_argument("--data-dir",type=Path,required=True)
    args.add_argument("--out",type=Path,required=True)
    args.add_argument("--seeds",nargs="+",type=int,default=[42,43])
    args.add_argument("--folds",nargs="+",type=int,default=[1,2,3,4,5])
    args.add_argument("--anchors",type=int,default=96)
    args.add_argument("--outer",type=int,default=50)
    args.add_argument("--k",type=int,choices=[4,5],required=True)
    z=args.parse_args()
    K=z.k
    z.out.mkdir(parents=True,exist_ok=True)
    results=[]
    for fold in z.folds:
        data=z.data_dir/f"trial_heldout_{fold}_5d.npz"
        with np.load(data) as a:
            Xtr=np.asarray(a["X_train"],float)/DIV
            Ytr=np.asarray(a["Y_train"],float)/DIV
            Xte=np.asarray(a["X_test"],float)/DIV
            Yte=np.asarray(a["Y_test"],float)/DIV
        assert Xtr.shape==Ytr.shape==(1000,5) and Xte.shape==Yte.shape==(250,5)
        # Same independent KMeans centers for both methods; only memberships differ.
        fitcx=KMeans(n_clusters=K,random_state=42,n_init=30).fit(Xtr)
        fitcy=KMeans(n_clusters=K,random_state=43,n_init=30).fit(Ytr)
        cx,cy=fitcx.cluster_centers_,fitcy.cluster_centers_
        before_global=w2(Xte,Yte,np.eye(5))
        for seed in z.seeds:
            # Same unsupervised domain anchors used by both methods,
            # learned from all 1000 training points per domain.
            Xanchors_cache={}
            for method in ("hard","soft05"):
                Am=make_domain_memberships(Xtr,method,cx)
                Bm=make_domain_memberships(Ytr,method,cy)
                Xanc,gx,xd=quantize_group_measure(Xtr,Am,z.anchors,seed+101)
                Yanc,gy,yd=quantize_group_measure(Ytr,Bm,z.anchors,seed+202)
                # Anchor locations should be bit-identical across hard/soft
                if method=="hard":Xanchors_cache={"x":Xanc.copy(),"y":Yanc.copy()}
                else:
                    assert np.max(abs(Xanc-Xanchors_cache["x"]))<1e-10
                    assert np.max(abs(Yanc-Xanchors_cache["y"]))<1e-10
                R,P,train,history=fit_orthogonal_hierarchical(
                    Xanc,Yanc,gx,gy,seed,max_iter=z.outer)
                held=test_shared_reference(Xtr,Ytr,Xte,Yte,R,cx,cy)
                metrics={
                    "holdout_fold":int(fold),"seed":int(seed),"method":method,
                    "train_n_per_domain":1000,"heldout_n_per_domain":250,
                    "train_anchor_per_domain":z.anchors,
                    "anchor_distortion_X":xd,"anchor_distortion_Y":yd,
                    "K":K,"epsilon_group":EG,"epsilon_sample":ES,
                    "temperature":(TAU if method=="soft05" else None),
                    "KMeans_source_group_sizes":np.bincount(fitcx.labels_,minlength=K).tolist(),
                    "KMeans_target_group_sizes":np.bincount(fitcy.labels_,minlength=K).tolist(),
                    "mean_max_soft_assignment_X":float(Am.max(1).mean()),
                    "mean_max_soft_assignment_Y":float(Bm.max(1).mean()),
                    "ambiguous_0p8_X":float(np.mean(Am.max(1)<.8)),
                    "ambiguous_0p8_Y":float(np.mean(Bm.max(1)<.8)),
                    "R":R.tolist(),"P":P.tolist(),
                    "train":train,
                    "heldout_common_hard_K_reference":held,
                    "test_W2_identity":before_global,
                    "test_W2_after":w2(Xte,Yte,R),
                    "paired_holdout_R2_EVAL_ONLY":float(1-np.sum((Xte@R.T-Yte)**2)/np.sum((Yte-Yte.mean(0))**2)),
                    "pair_synchrony_never_used_in_training":True,
                }
                results.append(metrics)
                (z.out/"results_partial.json").write_text(
                    json.dumps({"status":"PARTIAL","model":"weighted-anchor global-R hierarchical OT pilot",
                                "runs":results},indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
                print("DONE",fold,seed,method,
                      "train_T",round(train["training_result_at_final_global_R"]["T"],5),
                      "test_T",round(held.get("after",{}).get("T",float("nan")),5),
                      "test_W2",round(metrics["test_W2_after"],5),
                      "iterations",train["num_iterations"],
                      "passed",train["simple_numeric_convergence"],flush=True)
    (z.out/"results_all.json").write_text(
        json.dumps({"model":"weighted 96-anchor numerical surrogate of global-R two-level entropic OT",
                    "is_original_HiWA_ADMM":False,
                    "metric_priority":"heldout common Hard KMeans reference T,J; heldout global W2 aux",
                    "no_paired_training":True,"holdout_folds":z.folds,"seeds":z.seeds,
                    "runs":results},indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    # summary across folds at same seeds, no seed cherry-picking
    summary=[]
    for m in ("hard","soft05"):
        a=[r for r in results if r["method"]==m]
        for key,get in [
            ("test_W2",lambda r:r["test_W2_after"]),
            ("heldout_shared_T",lambda r:r["heldout_common_hard_K_reference"].get("after",{}).get("T",float("nan"))),
            ("heldout_shared_J",lambda r:r["heldout_common_hard_K_reference"].get("after",{}).get("J",float("nan"))),
            ("anchor_train_T",lambda r:r["train"]["training_result_at_final_global_R"]["T"])]:
            v=np.array([get(r) for r in a],float)
            summary.append({"method":m,"metric":key,"n":len(v),
                            "mean":float(np.mean(v)),"std":float(np.std(v,ddof=1)) if len(v)>1 else None,
                            "median":float(np.median(v))})
    (z.out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    # Per-fold average results, not a pooled independent iid variance estimate
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
    for axis,(metric,extract) in enumerate([
        ("test_W2_after",lambda r:r["test_W2_after"]),
        ("shared_hierarchical_T",lambda r:r["heldout_common_hard_K_reference"].get("after",{}).get("T",float("nan")))]):
        ax=axes[axis]
        hh=[];ss=[]
        for fold in z.folds:
            for arr,m in [(hh,"hard"),(ss,"soft05")]:
                arr.append(np.mean([extract(r) for r in results if r["method"]==m and r["holdout_fold"]==fold]))
        x=np.arange(len(z.folds))
        ax.plot(x,hh,"o-",label="Hard",linewidth=2)
        ax.plot(x,ss,"s-",label="Soft tau=0.5",linewidth=2)
        ax.set(xticks=x,xticklabels=[str(f) for f in z.folds],xlabel="heldout trial",
               ylabel=("global W2 squared" if axis==0 else "shared reference T"),
               title=("Full test empirical W2" if axis==0 else "Shared KMeans hierarchical OT"))
        ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.suptitle("Hyser: trial-disjoint Hard vs Soft pilot; no paired fitting")
    fig.savefig(z.out/"heldout_hard_soft_comparison.png",dpi=170,bbox_inches="tight")
    plt.close(fig)
    # Full explanation for correct scientific interpretation
    import statistics
    lines=["# Hyser 1250点 Hard vs Soft-KMeans 层级 OT：数值先导对照","",
        "**特别说明：这是加权锚点近似 + 全局正交R约束的二层OT，不是以前完整局部Rkl的HiWA ADMM运行。**",
        f"本研究保持同一个P/Q层级目标与K={K}/εs=.1/εg=.2，但为了对五折的1000训练点可计算，将每域各压缩为96个加权锚点；全部1000点都参与了独立K-means、锚点与权重估计。不能称为1000点全精度Q优化。",
        "训练只使用各自的Xtr/Ytr；源域组和目标域组独立训练；P自由学习；不借用时间同步对应、动作代理标签或测试数据。",
        "每折训练4次试次，每域1000点；测试完全留出1个试次，每域250点。预处理通过先前数据发布脚本仅在训练试次拟合。",
        "Hard/Soft 使用相同K-means中心、相同锚点、相同优化器和随机种子。Soft固定τ=0.5，无结果筛选。",
        "评价以同一训练簇中心预测的留出样本**共享Hard K簇参考**重求P、Q的T/J为主，全部250个测试样本整体W2²为辅。",
        "自家组测度不同，不能直接比原生训练T；本实验在同一个共享参考上评价其R。","",
        "## 关键数值（按折和初始化展开）",
        "",
        "| Heldout | Seed | Hard T | Soft T | Hard W2² | Soft W2² |",
        "|---:|---:|---:|---:|---:|---:|"]
    for fold in z.folds:
        for seed in z.seeds:
            h=next(r for r in results if r["holdout_fold"]==fold and r["seed"]==seed and r["method"]=="hard")
            s=next(r for r in results if r["holdout_fold"]==fold and r["seed"]==seed and r["method"]=="soft05")
            lines.append(
                f"|{fold}|{seed}|{h['heldout_common_hard_K_reference'].get('after',{}).get('T',float('nan')):.5f}|"
                f"{s['heldout_common_hard_K_reference'].get('after',{}).get('T',float('nan')):.5f}|"
                f"{h['test_W2_after']:.5f}|{s['test_W2_after']:.5f}|")
    lines += ["", "## 尚存局限",
        "该优化器是同一全局正交R下的交替P/Q/R优化：不是原始局部Rkl+共识ADMM，因此只能先作为受控数值方法对照。",
        "随机种子之间不是独立试次证据；仅5次试次来自同一位被试同一天的记录。",
        "收敛标志是简单数值阈值，并非KKT/全局最优证书；个别未达阈值的运行不得隐去。",
        "训练中两个域的配对关系完全不使用；paired R²仅训练结束后计算，绝不参与参数或最佳种子的选择。",
        "物理力量代理标签的单独审查见../results_1250/physical_meaning，与模型训练完全隔离。"]
    (z.out/"REPORT_ZH.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"status":"success","n_runs":len(results),"summary":summary},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
