#!/usr/bin/env python3
"""Postfit common-reference hierarchical OT (full heldout 250 points/domain).

Same domain-separately TRAIN-fitted KMeans K=3 and K=4 hard references
applied to the test clouds for every final R; both methods evaluated on
identical group partitions and entropic P/Q objective. No paired labels.
This is a TRANSductive *evaluation metric* (target test cloud available);
it is NOT an inductive prediction or a refitting of the learned R.

Do NOT conflate this reevaluated J/T with last training objective,
which cannot be recovered without archived full Q matrices.
"""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import xlogy
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rectangular_group_ot import group_entropic_transport

K_VALUES=(3,4)
ES=.1
EG=.2
REPS=("PCA3_PCA3","FA3_FA3","PCA5_RAW5","FA5_RAW5")
SEEDS=(42,43)
MAX_P_ITERS=6
MAX_Q_ITERS=90

def sinkhorn_pair(a,b,C,epsilon):
    # C is a squared Euclidean cost matrix, a and b sum to 1;
    # Q is *conditional*, with normalized marginals, NOT P weighted.
    a=np.asarray(a,float);b=np.asarray(b,float)
    z=np.exp(-np.minimum((C-C.min())/epsilon,700.))
    z=np.maximum(z,1e-300)
    v=np.ones(len(b))
    for i in range(MAX_Q_ITERS):
        u=a/np.maximum(z@v,1e-300)
        v=b/np.maximum(z.T@u,1e-300)
        if i%12==11 or i==MAX_Q_ITERS-1:
            Q=(u[:,None]*z)*v[None,:]
            err=max(float(np.max(abs(Q.sum(axis=1)-a))),
                    float(np.max(abs(Q.sum(axis=0)-b))))
            if err<2e-8:break
    Q=(u[:,None]*z)*v[None,:]
    err=max(float(np.max(abs(Q.sum(axis=1)-a))),
            float(np.max(abs(Q.sum(axis=0)-b))))
    if not np.isfinite(Q).all() or err>2e-6:
        raise ArithmeticError("Q marginal convergence failure "+str(err))
    return Q,err

def cluster_reference(Xtr,Ytr,Xte,Yte,K):
    # TRAIN-fit both sets of centres independently; no correspondence, no
    # physical test labels. Then apply centres to heldout samples.
    kmx=KMeans(n_clusters=K,n_init=30,random_state=42).fit(Xtr)
    kmy=KMeans(n_clusters=K,n_init=30,random_state=43).fit(Ytr)
    lx=kmx.predict(Xte);ly=kmy.predict(Yte)
    ix=[np.flatnonzero(lx==k) for k in range(K)]
    iy=[np.flatnonzero(ly==k) for k in range(K)]
    counts=[[len(s) for s in ix],[len(s) for s in iy]]
    return ix,iy,counts

def evaluate_shared(Xtest,Ytest,R,groupsX,groupsY):
    K=len(groupsX)
    assert len(groupsY)==K
    if any(len(a)==0 for a in groupsX+groupsY):
        raise ValueError("Empty heldout reference group -- do not silently reassign")
    D=cdist(Xtest@R.T,Ytest,metric="sqeuclidean")
    PP=np.full((K,K),1./(K*K))
    iterations=[]
    last_qinfo=[]
    for outer in range(MAX_P_ITERS+1):
        C=np.empty((K,K))
        H=np.empty((K,K))
        errs=[]
        qinfo=[]
        for k in range(K):
            ii=groupsX[k];pa=np.full(len(ii),1/len(ii))
            for l in range(K):
                jj=groupsY[l];pb=np.full(len(jj),1/len(jj))
                cost=D[np.ix_(ii,jj)]
                gamma=ES/max(PP[k,l],1e-15)
                Q,err=sinkhorn_pair(pa,pb,cost,gamma)
                C[k,l]=float(np.sum(Q*cost))
                H[k,l]=float(np.sum(xlogy(Q,Q)))
                qinfo.append({"source_group":k,"target_group":l,
                              "shape":[len(ii),len(jj)],
                              "conditional_mass":float(Q.sum()),
                              "conditional_entropy_negative":float(H[k,l]),
                              "marginal_max_error":err})
                errs.append(err)
        if outer==MAX_P_ITERS:
            last_qinfo=qinfo
            break
        newP=group_entropic_transport(
            C,np.full(K,1./K),np.full(K,1./K),eps=EG,tol=1e-9)
        change=float(np.linalg.norm(newP-PP))
        PP=newP
        iterations.append(change)
        # Fixed run count ensures identical postfit evaluation for every R,
        # irrespective of the method's native internal stopping rule.
    emarg=max(float(np.max(abs(PP.sum(axis=0)-1./K))),
              float(np.max(abs(PP.sum(axis=1)-1./K))))
    if emarg>1e-6:raise ArithmeticError("P marginal constraints not met "+str(emarg))
    TT=float(np.sum(PP*C))
    hs=float(ES*np.sum(H))
    hg=float(EG*np.sum(xlogy(PP,PP)))
    JJ=TT+hs+hg
    return {"T_hier_shared":TT,"J_hier_shared":JJ,
            "sample_entropy_term":hs,"group_entropy_term":hg,
            "P":PP.tolist(),"C":C.tolist(),
            "P_marginal_max_error":emarg,
            "Q_marginal_max_error":max(errs),
            "P_termination_diff":iterations[-1],
            "P_termination_history":iterations,
            "Q_block_diagnostics":last_qinfo,
            "Q_saved_full":False,
            "Q_all_recomputed_under_fixed_reference":True}

def load_frozen(base,rep,fold,seed):
    sub="results_formal_full1000_3D" if rep.endswith(("PCA3","FA3")) else "results_formal_full1000_Qfixed"
    stub=("hyser-threeD-" if "3D" in sub else "hyser-qfixed-")+rep+f"-f{fold}-s{seed}"
    file=base/sub/"individual_runs"/stub/"local_ADMM_results.json"
    o=json.loads(file.read_text())
    vals={x["method"]:x for x in o["runs"]}
    assert set(vals)=={"hard","soft05"}
    for mode,x in vals.items():
        assert x["n_OT_train_source"]==1000 and x["n_OT_train_target"]==1000
        assert x["Q_constraints_passed"] and x["numeric_convergence_ADMM_primal_and_rotation"]
    return vals

def savecsv(path,rows):
    fields=list(rows[0])
    with open(path,"w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--features",required=True,type=Path)
    ap.add_argument("--frozen",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--folds",nargs="+",type=int,default=list(range(1,6)))
    ap.add_argument("--reps",nargs="+",default=list(REPS),choices=REPS)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    rows=[];full=[];before={}
    for rep in args.reps:
        d=3 if rep.endswith(("PCA3","FA3")) else 5
        for fold in args.folds:
            fn=f"trial_heldout_{fold}"+("_5d" if d==5 else "")+".npz"
            with np.load(args.features/rep/fn) as z:
                Xtr=np.asarray(z["X_train"],float)/np.sqrt(d)
                Ytr=np.asarray(z["Y_train"],float)/np.sqrt(d)
                Xte=np.asarray(z["X_test"],float)/np.sqrt(d)
                Yte=np.asarray(z["Y_test"],float)/np.sqrt(d)
            assert Xtr.shape==Ytr.shape==(1000,d) and Xte.shape==Yte.shape==(250,d)
            for K in K_VALUES:
                gx,gy,counts=cluster_reference(Xtr,Ytr,Xte,Yte,K)
                refkey=(rep,fold,K)
                if any(0 in c for c in counts):
                    # Retain failures without deleting bad folds.
                    for seed in SEEDS:
                        for method in ("hard","soft05"):
                            rows.append({"representation":rep,"fold":fold,"seed":seed,
                             "method":method,"reference_K":K,"valid":False,
                             "failure":"Empty heldout hard reference group",
                             "source_group_counts":str(counts[0]),
                             "target_group_counts":str(counts[1]),
                             "T_hier_shared":None,"J_hier_shared":None,
                             "T_before_identity":None,"delta_T_vs_identity":None,
                             "P_marginal_max_error":None,"Q_marginal_max_error":None})
                    print("SKIP empty group",refkey,counts,flush=True)
                    continue
                identity=evaluate_shared(Xte,Yte,np.eye(d),gx,gy)
                before[refkey]=identity
                for seed in SEEDS:
                    model=load_frozen(args.frozen,rep,fold,seed)
                    for method in ("hard","soft05"):
                        r=model[method]
                        R=np.asarray(r["R"],float)
                        assert R.shape==(d,d) and np.linalg.norm(R.T@R-np.eye(d))<1e-6
                        try:
                            test=evaluate_shared(Xte,Yte,R,gx,gy)
                            row={"representation":rep,"fold":fold,"seed":seed,"method":method,
                                 "reference_K":K,"valid":True,"failure":"",
                                 "source_group_counts":str(counts[0]),
                                 "target_group_counts":str(counts[1]),
                                 "T_hier_shared":test["T_hier_shared"],
                                 "J_hier_shared":test["J_hier_shared"],
                                 "T_before_identity":identity["T_hier_shared"],
                                 "delta_T_vs_identity":test["T_hier_shared"]-identity["T_hier_shared"],
                                 "P_marginal_max_error":test["P_marginal_max_error"],
                                 "Q_marginal_max_error":test["Q_marginal_max_error"]}
                            full.append({"key":{"representation":rep,"fold":fold,"seed":seed,
                                                "method":method,"K":K},
                                         "numerical_qualifications":True,
                                         "reference_method":"KMeans TRAIN fitted independent both domains",
                                         "reference_group_counts":counts,
                                         "objective_after":test,
                                         "objective_at_identity":identity})
                        except Exception as ex:
                            row={"representation":rep,"fold":fold,"seed":seed,"method":method,
                                 "reference_K":K,"valid":False,"failure":str(ex),
                                 "source_group_counts":str(counts[0]),
                                 "target_group_counts":str(counts[1]),
                                 "T_hier_shared":None,"J_hier_shared":None,
                                 "T_before_identity":identity["T_hier_shared"],
                                 "delta_T_vs_identity":None,
                                 "P_marginal_max_error":None,"Q_marginal_max_error":None}
                        rows.append(row)
                        print(rep,fold,K,seed,method,row["valid"],
                              row["T_hier_shared"],flush=True)
    savecsv(args.out/"shared_reference_full250_all_runs.csv",rows)
    (args.out/"shared_reference_P_C_entropy_diagnostics.json").write_text(
        json.dumps({"formula":"J=T+0.1*sum(Q*logQ)+0.2*sum(P*logP)",
                   "P_row_col": "uniform 1/K per group",
                   "conditional_Q_each_block":"uniform 1/count within its reference test group",
                   "Q_epsilon":"0.1/P[k,l]",
                   "gamma_group":EG,
                   "disclosure":"newly reoptimized P/Q at frozen R; not archived original last training P/Q",
                   "references":"train-fitted independently for source and target, applied to test clouds",
                   "not_inductive":"target 250 heldout points participate in evaluation P/Q, never model fitting",
                   "records":full},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    summary=[]
    for rep in args.reps:
        for K in K_VALUES:
            for mode in ("hard","soft05"):
                r=[x for x in rows if x["representation"]==rep and x["reference_K"]==K and x["method"]==mode]
                q=[x for x in r if x["valid"]]
                summary.append({"representation":rep,"reference_K":K,"method":mode,
                                "n_observed":len(r),"n_valid":len(q),
                                "mean_T":float(np.mean([x["T_hier_shared"] for x in q])) if q else None,
                                "mean_J":float(np.mean([x["J_hier_shared"] for x in q])) if q else None,
                                "mean_delta_T_from_identity":float(np.mean([x["delta_T_vs_identity"] for x in q])) if q else None})
    savecsv(args.out/"shared_reference_summary.csv",summary)
    paired=[]
    for rep in args.reps:
        for K in K_VALUES:
            for fold in args.folds:
                for seed in SEEDS:
                    a=next(x for x in rows if x["representation"]==rep and x["reference_K"]==K and x["fold"]==fold and x["seed"]==seed and x["method"]=="hard")
                    b=next(x for x in rows if x["representation"]==rep and x["reference_K"]==K and x["fold"]==fold and x["seed"]==seed and x["method"]=="soft05")
                    paired.append({"representation":rep,"reference_K":K,"fold":fold,"seed":seed,
                                   "both_valid":bool(a["valid"] and b["valid"]),
                                   "soft_minus_hard_T":b["T_hier_shared"]-a["T_hier_shared"] if a["valid"] and b["valid"] else None,
                                   "soft_minus_hard_J":b["J_hier_shared"]-a["J_hier_shared"] if a["valid"] and b["valid"] else None})
    savecsv(args.out/"paired_shared_reference_effects.csv",paired)
    fig,axes=plt.subplots(1,2,figsize=(12,5.6),constrained_layout=True)
    for ax,K in zip(axes,K_VALUES):
        subset=[x for x in paired if x["reference_K"]==K and x["both_valid"]]
        labs=list(args.reps)
        vals=[np.mean([z["soft_minus_hard_T"] for z in subset if z["representation"]==rep])
              if any(z["representation"]==rep for z in subset) else np.nan for rep in labs]
        ax.barh(np.arange(len(labs)),vals)
        ax.set_yticks(np.arange(len(labs)),labels=labs)
        ax.axvline(0,color="gray",ls="--")
        ax.set(xlabel="Soft − Hard shared T (negative favors Soft)",
               title=f"TRAIN reference K={K}, full 250 test windows")
    fig.savefig(args.out/"shared_hierarchical_T_hard_soft.png",dpi=175)
    plt.close(fig)
    report=["# Hyser：冻结80次HiWA模型的两层层级OT共享参考评价",
        "","**评价合同：** 参考簇分别在两域四次训练试次拟合；完整250个留出点只用于重新计算评价的P/Q，**不是**训练时学习的P/Q，也没有真实时间配对标签。",
        "固定共同Hard KMeans参考K=3、K=4；目标是比较最终R在相同层级概率测度上的T与J，不代表Soft分组的所有表达能力。",
        "",
        "在每个共同参考中，按 \(Q^{kl}\) 行列和分别为组内均匀条件质量、\(P\) 行列和为 \(1/K\) 的约束迭代求解。纯运输T与包含两级熵的J分开报告。",
        "","|表示|共享参考K|方法|有效/总数|T↓|J↓|T−identity↓|",
        "|---|---:|---|---:|---:|---:|---:|"]
    for r in summary:
        ff=lambda x:"NA" if x is None else f"{x:.6f}"
        report.append(f"|{r['representation']}|{r['reference_K']}|{r['method']}|{r['n_valid']}/{r['n_observed']}|{ff(r['mean_T'])}|{ff(r['mean_J'])}|{ff(r['mean_delta_T_from_identity'])}|")
    report+=["","## 严格限制",
        "- 共享参考是**事后重新优化的评价 P/Q**，不是训练末轮的原生 P/Q；原生完整训练末轮 J 因缺少已序列化的 Q 矩阵尚不能证实。",
        "- 在两种表示、两种K比较时不能跨特征空间比较绝对T/J，仅同表示和相同参考下比较Hard/Soft。",
        "- 本研究的测试集250目标点参与共同参考的传输评价，所以这一项是无真实对应标签的*转导式分布评价*；单点物理预测仍是归纳式目标训练库检索。",
        "- 两种子在同一个试次上不是独立样本；对每折先取两种子平均，不能声称跨人群显著。",
        "- K3与K4参考结果若不一致，不得挑选对Soft有利的参考冒称稳定优势。",
        "","![共享层级运输Hard与Soft差](shared_hierarchical_T_hard_soft.png)"]
    (args.out/"SHARED_HIERARCHICAL_EVALUATION_ZH.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(json.dumps({"finished":True,"n_records":len(rows),"valid":sum(x["valid"] for x in rows),
                      "summary":summary},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
