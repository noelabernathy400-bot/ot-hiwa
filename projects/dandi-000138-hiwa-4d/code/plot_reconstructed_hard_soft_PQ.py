"""Rebuild REAL archived independent-group HiWA/GCOT P and local Q and plot Γ.

NO GROUP PAIRING LEAKAGE: training input is the original independently clustered
4D DANDI cache. Reproduce archived seed 42 hard and soft *exactly*, compare
train sample-index hashes, R/P, convergence and stored scores. Draw all plots
from actual re-run local Q plans, NOT made-up block matrices or synthetic data.

Global sample-level plan Γ[a,b] = Σ_{k,l} P[k,l] Q_{k,l}[a,b],
with original-source/original-target sampled point index matching.
Hard: Q on each disjoint group pair, embed to exact global 160x160 indices.
Soft: every local Q is already 160x160 (group-dependent marginal weights).
This is the LAST LOCAL Q of the archived solver's final iteration, weighted
by its final outer P; note final P is computed AFTER local Q each iteration.
Thus do not claim Q is a newly reoptimized optimum at final P/R.

The shared heatmaps use identical point ID order in both methods, sorted by
each domain's own independent HARD training KMeans labels, then by first
normalized coordinate. Colors on matrix edges denote those independent
training groups, not the posthoc paired sample coloring.

A point-to-point static explanatory panel highlights SAME source sample and
the largest Γ destination per method, with true original row IDs, P-Q weight,
row-normalized conditional mass. Training uses NO true paired label or id.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch, Rectangle
from scipy.special import softmax
from zip_gcot.alignment import hierarchical_alignment, SolverConfig
from run_hard_hiwa import sample_groups
from run_controlled_soft_hard_ablation import build_measures


PALETTE = ["#3479b9","#e79b3b","#319c7c","#9a78c2"]
BG="#f4f5f8"


def infer_soft(x,centers):
    a=softmax(x@centers.T,axis=1)
    a=np.maximum(a,np.finfo(float).tiny)
    return a/a.sum(1,keepdims=True)


def check_weights(M,P,sweights,tweights,groupx,groupy,hard):
    assert M.shape==(160,160) and np.isfinite(M).all()
    assert (M>=-1e-12).all()
    assert abs(M.sum()-1)<1e-6
    assert np.allclose(P.sum(0),.25,atol=1e-5)
    assert np.allclose(P.sum(1),.25,atol=1e-5)
    if hard:
        a=np.repeat(1/160,160)
        b=np.repeat(1/160,160)
    else:
        a=sum(.25*sweights[:,k]/sweights[:,k].sum() for k in range(4))
        b=sum(.25*tweights[:,k]/tweights[:,k].sum() for k in range(4))
    errs=[float(np.max(abs(M.sum(axis=1)-a))),
          float(np.max(abs(M.sum(axis=0)-b)))]
    if max(errs)>1e-5:raise ArithmeticError(f"Gamma marginal mismatch {errs}")
    return dict(gamma_mass=float(M.sum()),
                gamma_row_error=errs[0],
                gamma_col_error=errs[1],
                gamma_effective_edges_gt1e_minus_5=int((M>1e-5).sum()),
                gamma_entropy_nats=float(-np.sum(M[M>0]*np.log(M[M>0]))))


def compose_plans(fit,train_labels_x,train_labels_y,AX,BY,hard):
    P=np.asarray(fit["P"],float)
    M=np.zeros((160,160),float)
    Qs={}
    for k in range(4):
        ix=np.flatnonzero(train_labels_x==k)
        for l in range(4):
            iy=np.flatnonzero(train_labels_y==l)
            Q=np.asarray(fit[f"Q_{k}_{l}"],float)
            if hard:
                assert Q.shape==(len(ix),len(iy))==(40,40)
                M[np.ix_(ix,iy)]+=P[k,l]*Q
            else:
                assert Q.shape==(160,160)
                M+=P[k,l]*Q
            Qs[f"Q_{k}_{l}"]=Q
    audit=check_weights(M,P,AX,BY,train_labels_x,train_labels_y,hard)
    return M,Qs,audit


def display_heatmaps(hard,soft,sidx,tidx,xlab,ylab,out):
    ix=np.lexsort((hard["source_points"][:,0],xlab))
    iy=np.lexsort((hard["target_points"][:,0],ylab))
    assert np.array_equal(hard["source_indices"],soft["source_indices"])
    assert np.array_equal(hard["target_indices"],soft["target_indices"])
    mats=[hard["Gamma"][np.ix_(ix,iy)],soft["Gamma"][np.ix_(ix,iy)]]
    # common display colormap, log amplification, same scale
    positive=np.concatenate([v[v>0].ravel() for v in mats])
    vmax=float(max(np.max(m) for m in mats))
    vmin=float(max(np.quantile(positive,.02),vmax*1e-4))
    fig,axs=plt.subplots(1,2,figsize=(14.7,8.8),constrained_layout=True)
    cmap=plt.get_cmap("magma").copy()
    cmap.set_under("#fafafa")
    cmap.set_bad("#fafafa")
    color=colors.LogNorm(vmin=vmin,vmax=vmax,clip=False)
    hard_ordered_x=xlab[ix];hard_ordered_y=ylab[iy]
    boundaryX=np.cumsum(np.bincount(hard_ordered_x,minlength=4))[:-1]
    boundaryY=np.cumsum(np.bincount(hard_ordered_y,minlength=4))[:-1]
    # To emphasize soft main groups without altering ordering, add an outer
    # thin stripe with soft argmax (independent Y features) as a second track.
    sx=np.argmax(soft["source_soft"],axis=1)[ix]
    sy=np.argmax(soft["target_soft"],axis=1)[iy]
    for ax,M,method in zip(axs,mats,["Hard: independent X/Y K-means",
                                      "Soft: independent X/Y prototypes"]):
        im=ax.imshow(np.maximum(M,1e-300),origin="upper",aspect="equal",
                cmap=cmap,norm=color,interpolation="nearest")
        for q in boundaryX:ax.axhline(q-.5,color="white",lw=1.4,alpha=.85)
        for q in boundaryY:ax.axvline(q-.5,color="white",lw=1.4,alpha=.85)
        ax.set(xlabel="Independent target sample (column)",
               ylabel="Independent source sample (row)",
               title=method,
               xticks=[0,39,79,119,159],yticks=[0,39,79,119,159])
        for k in range(160):
            ax.add_patch(Rectangle((-5.1,k-.5),2.1,1.0,facecolor=PALETTE[hard_ordered_x[k]],
                edgecolor="none",clip_on=False))
            ax.add_patch(Rectangle((k-.5,-5.1),1,2.1,facecolor=PALETTE[hard_ordered_y[k]],
                edgecolor="none",clip_on=False))
        # smaller extra soft-argmax track on outer margin, both panels
        for k in range(160):
            ax.add_patch(Rectangle((-8,k-.5),1.7,1.0,facecolor=PALETTE[sx[k]],
                                 edgecolor="none",clip_on=False,alpha=.82))
            ax.add_patch(Rectangle((k-.5,-8),1,1.7,facecolor=PALETTE[sy[k]],
                                 edgecolor="none",clip_on=False,alpha=.82))
        ax.set_xlim(-9.5,159.5);ax.set_ylim(159.5,-9.5)
    cbar=fig.colorbar(im,ax=list(axs),shrink=.69,pad=.025)
    cbar.set_label("Γ[a,b] = total transport MASS (identical log color scale)")
    fig.suptitle("Actual trained local Q aggregated with learned free P | seed 42 | same 160 × 160 sample IDs",
                 fontsize=14)
    fig.savefig(out/"hard_soft_full_160x160_transport_heatmaps.png",
                dpi=200,bbox_inches="tight")
    plt.close(fig)

    # a separately readable HARD vs SOFT common matrix difference, not
    # suitable as primary transport mass because signs denote differences.
    d=mats[1]-mats[0]
    lim=float(np.quantile(abs(d),.999))
    lim=max(lim,1e-10)
    fig,ax=plt.subplots(figsize=(8,7.4),constrained_layout=True)
    im=ax.imshow(d,origin="upper",cmap="RdBu_r",
              vmin=-lim,vmax=lim,interpolation="nearest")
    for v in boundaryX:ax.axhline(v-.5,color="black",lw=.8,alpha=.5)
    for v in boundaryY:ax.axvline(v-.5,color="black",lw=.8,alpha=.5)
    ax.set(title="Soft − Hard: same source/target point pairs",
        xlabel="Target point (independent Hard group order)",
        ylabel="Source point (independent Hard group order)")
    fig.colorbar(im,ax=ax,label="Signed transport mass change Γ_soft − Γ_hard")
    fig.savefig(out/"soft_minus_hard_transport_change.png",
                dpi=200,bbox_inches="tight")
    plt.close(fig)
    return dict(source_display_positions=ix.tolist(),
        target_display_positions=iy.tolist(),
        visual_color_scale_log=[vmin,vmax],
        source_group_boundaries=boundaryX.tolist(),
        target_group_boundaries=boundaryY.tolist())


def ribbon(ax,x0,x1,y0a,y0b,y1a,y1b,rgba):
    t=(x1-x0)*.43
    vertices=[(x0,y0a),(x0+t,y0a),(x1-t,y1a),(x1,y1a),
              (x1,y1b),(x1-t,y1b),(x0+t,y0b),(x0,y0b),(x0,y0a)]
    path=MPath(vertices,[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,
                    MPath.LINETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4,MPath.CLOSEPOLY])
    ax.add_patch(PathPatch(path,facecolor=rgba,edgecolor="none"))


def alluvial_panel(ax,P,title):
    x0=.16;x1=.82
    pitch=.225;bar_height=.18;bottom=.105
    source_base=[bottom+(3-k)*pitch for k in range(4)]
    target_base=[bottom+(3-l)*pitch for l in range(4)]
    # row and column sum are .25 each, each bar has height .18.
    source_cursor=source_base.copy()
    target_cursor=target_base.copy()
    for k in range(4):
        for l in np.argsort(-P[k]):
            mass=float(P[k,l]);wid=bar_height*mass/.25
            ya=source_cursor[k];yb=target_cursor[l]
            ribbon(ax,x0+.025,x1-.025,ya,ya+wid,yb,yb+wid,
                   colors.to_rgba(PALETTE[k],alpha=.34))
            source_cursor[k]+=wid;target_cursor[l]+=wid
    for k in range(4):
        ya=source_base[k]
        ax.add_patch(Rectangle((x0,ya),.026,bar_height,
            facecolor=PALETTE[k],edgecolor="white",lw=.4,zorder=5))
        ax.text(x0-.024,ya+bar_height/2,f"X{k}",ha="right",va="center",fontsize=12,
            color=PALETTE[k],weight="bold")
        yb=target_base[k]
        ax.add_patch(Rectangle((x1-.026,yb),.026,bar_height,
            facecolor=PALETTE[k],edgecolor="white",lw=.4,zorder=5))
        ax.text(x1+.018,yb+bar_height/2,f"Y{k}",ha="left",va="center",fontsize=12,
            color=PALETTE[k],weight="bold")
    ax.set(xlim=(0,1),ylim=(0,1),title=title)
    ax.axis("off")


def plot_P_alluvial(hard,soft,out):
    fig,axs=plt.subplots(1,2,figsize=(14,6.1),constrained_layout=True)
    alluvial_panel(axs[0],hard["P"],"Hard — learned P (no fixed matching)")
    alluvial_panel(axs[1],soft["P"],"Soft — learned P (no fixed matching)")
    fig.suptitle("Actual hierarchical GROUP-LEVEL transport | ribbon width ∝ P[k,l] | each source & target group has mass 0.25",
                 fontsize=13.5)
    fig.savefig(out/"hard_soft_group_P_alluvial_flow.png",dpi=200,bbox_inches="tight")
    plt.close(fig)


def plot_point_to_matrix(hard,soft,out,ordering):
    # Fixed SAME source point, median among independent-Hard group 0,
    # no paired oracle / test data involved. Different destinations are
    # selected by each actual Γ row maximum.
    ids=ordering["source_display_positions"]
    xg=hard["source_independent_group"]
    one=np.array(ids)[np.array(xg)[ids]==0]
    a=int(one[len(one)//2])
    cmap=plt.get_cmap("tab10")
    fig=plt.figure(figsize=(16.5,11),constrained_layout=True)
    gs=fig.add_gridspec(2,3,width_ratios=[1.3,1,1],wspace=.13,hspace=.22)
    row_out=[]
    for index,data in enumerate([hard,soft]):
        name=["Hard","Soft"][index]
        y_best=int(np.argmax(data["Gamma"][a]))
        g=float(data["Gamma"][a,y_best])
        rowp=float(g/data["Gamma"][a].sum())
        m=data["Gamma"][np.ix_(ordering["source_display_positions"],
                                ordering["target_display_positions"])]
        ai=int(np.where(np.asarray(ordering["source_display_positions"])==a)[0][0])
        bi=int(np.where(np.asarray(ordering["target_display_positions"])==y_best)[0][0])
        ax=fig.add_subplot(gs[index,0])
        ax.imshow(np.maximum(m,1e-300),cmap="magma",
                  norm=colors.LogNorm(*ordering["visual_color_scale_log"]),
                  interpolation="nearest",aspect="equal")
        for v in ordering["source_group_boundaries"]:
            ax.axhline(v-.5,color="white",lw=1.0,alpha=.65)
        for v in ordering["target_group_boundaries"]:
            ax.axvline(v-.5,color="white",lw=1.0,alpha=.65)
        ax.axhline(ai,color="#54d5ca",lw=1,alpha=.75)
        ax.axvline(bi,color="#54d5ca",lw=1,alpha=.75)
        ax.scatter([bi],[ai],s=140,marker="o",
                   facecolor="none",edgecolor="#56f0e2",lw=2.5)
        ax.set(xlabel="Target sample, independent group order",
               ylabel="Source sample, independent group order",
               title=f"{name} Γ   (source row {ai}, target column {bi})")
        axx=fig.add_subplot(gs[index,1])
        axy=fig.add_subplot(gs[index,2])
        xx=data["source_points"]@data["R"].T
        yy=data["target_points"]
        for k in range(4):
            axx.scatter(xx[xg==k,0],xx[xg==k,1],
                        color=PALETTE[k],alpha=.53,s=18)
            tgt=data["target_independent_group"]
            axy.scatter(yy[tgt==k,0],yy[tgt==k,1],color=PALETTE[k],alpha=.53,s=18)
        axx.scatter([xx[a,0]],[xx[a,1]],s=200,facecolor="none",
                    edgecolor="#eb354d",lw=3,zorder=5)
        axy.scatter([yy[y_best,0]],[yy[y_best,1]],s=200,facecolor="none",
                    edgecolor="#eb354d",lw=3,zorder=5)
        axx.set(title=f"Source X[a]@Rᵀ | train row {int(data['source_indices'][a])}",
            xlabel="4D coordinate 1",ylabel="4D coordinate 2")
        axy.set(title=f"Target Y[b] | train row {int(data['target_indices'][y_best])}",
            xlabel="4D coordinate 1",ylabel="4D coordinate 2")
        fig.text(.51,.55 if index==0 else .075,
                 f"{name}: Γ[a,b]={g:.3g}; row-conditional share={rowp:.1%}; "
                 f"source index={int(data['source_indices'][a])}, target index={int(data['target_indices'][y_best])}",
                 ha="center",fontsize=9)
        row_out.append(dict(method=name,source_position=a,
             source_original_row=int(data["source_indices"][a]),
             target_position=y_best,target_original_row=int(data["target_indices"][y_best]),
             Gamma_weight=g,conditional_row_mass=rowp,heatmap_row=ai,heatmap_col=bi))
    fig.suptitle("One heatmap cell = (real source point a, real target point b): "
                 "highlight source and its largest Γ destination | 2D projection",
                 fontsize=13)
    fig.savefig(out/"matrix_pixel_to_real_points.png",dpi=195,bbox_inches="tight")
    plt.close(fig)
    return row_out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--clusters",type=Path,required=True)
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--outer",type=int,default=500)
    p.add_argument("--inner",type=int,default=20)
    p.add_argument("--sample-sinkhorn",type=int,default=45)
    p.add_argument("--group-sinkhorn",type=int,default=200)
    p.add_argument("--mu",type=float,default=.05)
    p.add_argument("--epsilon-group",type=float,default=.2)
    p.add_argument("--epsilon-sample",type=float,default=.1)
    args=p.parse_args()
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    if args.clusters.name!="clustered_4d.npz":
        raise ValueError("Must use independently grouped input, never paired label copy")
    saved=json.loads(args.archive.read_text(encoding="utf-8"))
    assert saved["source_and_target_hard_kmeans_independent"]
    assert saved["source_and_target_soft_prototypes_independent"]
    assert saved["no_target_membership_copy"]
    assert saved["no_synchronized_pair_ids_used_in_fitting"]
    assert hashlib.sha256(args.clusters.read_bytes()).hexdigest()==saved["input_sha256"]
    with np.load(args.clusters,allow_pickle=False) as z:
        assert "shared_labels" not in z.files
        X=np.asarray(z["neural4"],float)
        Y=np.asarray(z["hand4"],float)
        train=np.asarray(z["train_mask"],bool)
        hard_x=np.asarray(z["neural_groups"],int)
        hard_y=np.asarray(z["hand_groups"],int)
        assert not np.array_equal(hard_x,hard_y)
        assert train.sum()==14000
    x=(X-np.asarray(saved["source_mean"]))/saved["source_radius"]
    y=(Y-np.asarray(saved["target_mean"]))/saved["target_radius"]
    CX=np.asarray(saved["soft_prototypes"]["source_centers"])
    CY=np.asarray(saved["soft_prototypes"]["target_centers"])
    ax_all=infer_soft(x,CX)
    by_all=infer_soft(y,CY)
    assert not np.array_equal(ax_all,by_all)
    rg=np.random.default_rng(args.seed)
    tidx=np.flatnonzero(train)
    idsx=sample_groups(tidx,hard_x,40,rg)
    idsy=sample_groups(tidx,hard_y,40,rg)
    assert len(idsx)==len(idsy)==160
    archive_runs={}
    for method in ("hard","soft"):
        run=next(v for v in saved["runs"][method] if v["seed"]==args.seed)
        archive_runs[method]=run
        assert hashlib.sha256(idsx.tobytes()).hexdigest()==run["sample_source_rows_sha256"]
        assert hashlib.sha256(idsy.tobytes()).hexdigest()==run["sample_target_rows_sha256"]
    cfg=SolverConfig(maxiter=args.outer,tol=1e-9,mu=args.mu,
          epsilon_group=args.epsilon_group,epsilon_sample=args.epsilon_sample,
          inner_maxiter=args.inner,inner_tol=.01,
          sinkhorn_maxiter=args.sample_sinkhorn,
          group_sinkhorn_maxiter=args.group_sinkhorn,
          sinkhorn_tol=1e-8,stopping="legacy")
    results={}
    arrays=dict(source_global_train_indices=idsx.astype(np.int32),
                target_global_train_indices=idsy.astype(np.int32),
                source_training_hard_labels=hard_x[idsx].astype(np.int8),
                target_training_hard_labels=hard_y[idsy].astype(np.int8),
                source_training_soft_memberships=ax_all[idsx],
                target_training_soft_memberships=by_all[idsy],
                normalized_source_train_points=x[idsx],
                normalized_target_train_points=y[idsy])
    for method in ("hard","soft"):
        hard=(method=="hard")
        a=np.eye(4)[hard_x[idsx]] if hard else ax_all[idsx]
        b=np.eye(4)[hard_y[idsy]] if hard else by_all[idsy]
        fit=hierarchical_alignment(build_measures(x[idsx],a,hard),
              build_measures(y[idsy],b,hard),
              seed=args.seed,config=cfg,scale=True,retain_plans=True)
        old=archive_runs[method]
        newP=np.asarray(fit["P"])
        newR=np.asarray(fit["R"])
        diffP=float(np.max(abs(newP-np.asarray(old["P"]))))
        diffR=float(np.max(abs(newR-np.asarray(old["R"]))))
        print(f"REPRODUCE {method}: max|P-P_saved|={diffP:.3e}, max|R-R_saved|={diffR:.3e}",flush=True)
        if diffP>1e-9 or diffR>1e-9:
            raise ArithmeticError("Reproduced model differs from archived actual experiment")
        assert len(fit["history"])==old["iterations"]
        M,qs,audit=compose_plans(fit,hard_x[idsx],hard_y[idsy],a,b,hard)
        results[method]=dict(
             method=method,seed=args.seed,
             source_indices=idsx,target_indices=idsy,
             source_points=x[idsx],target_points=y[idsy],
             source_independent_group=hard_x[idsx],
             target_independent_group=hard_y[idsy],
             source_soft=ax_all[idsx],target_soft=by_all[idsy],
             R=newR,P=newP,Gamma=M,checks=audit,
             solver_iterations=len(fit["history"]),
             strict_admm_criteria=bool(fit["criteria_satisfied"]),
             archived_strict_admm_criteria=bool(old["admm_criteria_passed"]),
             comparison_max_P_error=diffP,comparison_max_R_error=diffR)
        arrays[method+"_P"]=newP
        arrays[method+"_R"]=newR
        arrays[method+"_Gamma"]=M
        for key,val in qs.items():
            arrays[method+"_"+key]=val
        print(f"RECONSTRUCT {method}: Γ.sum={audit['gamma_mass']:.12g}, "
            f"rowerr={audit['gamma_row_error']:.2g}, "
            f"colerr={audit['gamma_col_error']:.2g}; ADMM={bool(fit['criteria_satisfied'])}",flush=True)
    np.savez_compressed(out/"actual_sample_level_P_Q_and_indices.npz",**arrays)
    ordering=display_heatmaps(results["hard"],results["soft"],
          idsx,idsy,hard_x[idsx],hard_y[idsy],out)
    plot_P_alluvial(results["hard"],results["soft"],out)
    links=plot_point_to_matrix(results["hard"],results["soft"],out,ordering)
    for method in ("hard","soft"):
        r=results[method]
        for i,l in enumerate(links):
            if l["method"].lower()==method:
                assert abs(r["Gamma"][l["source_position"],l["target_position"]]-
                    l["Gamma_weight"])<1e-15
    metadata={
      "actual_data":True,"original_solver_code_unchanged":True,
      "dataset":"original strict unpaired DANDI000138 4D trial 400train/100test",
      "source_training_labels_independent_of_target":True,
      "source_soft_prototypes_independent_of_target":True,
      "source_and_target_training_indices_are_independently_sampled":True,
      "true_paired_sample_ids_used_for_training":False,
      "same_original_train_sample_ids_in_hard_soft":True,
      "seed":args.seed,"source_N":160,"target_N":160,
      "heatmap_quantity":"Gamma[a,b]=sum_kl trained final P[k,l] times trained final-iteration local Q_kl(a,b)",
      "solver_internals_note":"Original archived ZIP solver computes local Q in inner loop BEFORE updating outer P and global R; combined Gamma uses Q from final local update and FINAL P. It is a faithful visualization of archived solver output, not a newly optimized Q at fixed final P/R.",
      "soft_block_overlap_note":"Hard 40x40 Q blocks are embedded at independent cluster sample identities; Soft 160x160 Q blocks overlap on physical samples and must be summed, never spatially tiled.",
      "color_note":"Edge bars show independently learned hard training labels; outer thinner bars show source/target independently learned soft-prototype argmax; identical group number across domains NOT assumed correct correspondence.",
      "transport_mass_units":"P sums to1, each Q has unit total conditional mass",
      "reproduction_audit":{k:{p:v for p,v in r.items()
          if p in ("checks","solver_iterations","strict_admm_criteria",
                   "archived_strict_admm_criteria","comparison_max_P_error","comparison_max_R_error")}
          for k,r in results.items()},
      "source_original_train_indices_sha256":hashlib.sha256(idsx.tobytes()).hexdigest(),
      "target_original_train_indices_sha256":hashlib.sha256(idsy.tobytes()).hexdigest(),
      "heatmap_axis_ordering":ordering,
      "selected_actual_point_pairs":links,
      "visualization_outputs":[
        "hard_soft_full_160x160_transport_heatmaps.png",
        "hard_soft_group_P_alluvial_flow.png",
        "matrix_pixel_to_real_points.png",
        "soft_minus_hard_transport_change.png",
        "actual_sample_level_P_Q_and_indices.npz"],
    }
    (out/"VISUALIZATION_AUDIT.json").write_text(
         json.dumps(metadata,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    text=[
      "# 真实无监督 Hard/Soft 的分层 P-Q 传输可视化（seed 42）",
      "",
      "## 核心图像",
      "",
      "![160x160 Hard/Soft 样本级真实传输热力图](hard_soft_full_160x160_transport_heatmaps.png)",
      "",
      "![Hard/Soft learned P 的簇间质量流向 Sankey/alluvial 图](hard_soft_group_P_alluvial_flow.png)",
      "",
      "![热力图像素与真实源点/目标点样本索引的对应示例](matrix_pixel_to_real_points.png)",
      "",
      "![同一元素的 Soft 减 Hard 运量变化（有正有负）](soft_minus_hard_transport_change.png)",
      "",
      "## 数学含义与严谨性",
      "",
      "每种方法在同一批160个源域和160个目标域训练样本上，求得自己的源/目标**独立组**和真实学得的 P、Q。使用真实已拟合的 42号初始化重放原 ZIP-GCOT，逐元素检验 P/R 与正式归档完全一致（最大误差1e-9以下）；同一批点索引 SHA256 与归档一致。",
      "",
      "构建样本级总传输：Γ[a,b] = Σ_{k,l} P[k,l] Q_{k,l}(a,b)。Hard 每个 Q[k,l] 是40x40，按该簇对实际样本索引嵌入160x160矩阵。Soft 每个 Q[k,l] 已是160x160，逐元素加权相加，**绝不可把同一批物理样本重复拼成640x640**。",
      "",
      "大型热力图每个像素都对应原始 sampled row ID 的一个源点a和一个目标点b；行列排序在两种方法完全一致。行/列边缘粗色条是两域分别训练的 Hard K-means 组，细条是两域各自学得 Soft 原型的最大隶属组；**目标域的颜色不是由同步源样本标签复制**。",
      "",
      "第3张图使用同一个源点（取独立Hard组0中间点），分别挑出Hard和Soft下它运量最大的一条真实源→目标边；矩阵位置、高亮散点、原始行索引和行内条件概率一一对应。点云只展示四维坐标中的前两维投影，二维距离不能冒充原始四维运输代价。",
      "",
      "**IMPORTANT**：这16个局部 Q 是原始 GCOT 求解器最后一次局部迭代得到的 Q，而最后的 P 是该轮局部Q之后更新所得。两者与正式归档模型数值一致，但这里展示的不是保持最终P/R固定时重新求得的测试Q；也不是声称每个Q都在最终全局R下最优。",
      "",
      "与旧监督实验完全隔离。源目标组的真实配对仅在拟合以后可用于对照上色，没参与此图的P/Q重建。Soft正式求解在seed42达到原定500外循环上限但未通过严格ADMM判据；图像忠实展示这一有数值限制的结果。",
      "",
      "原始P/Q、两域160个真实训练样本行索引、4D坐标和组标签全部保存在 [actual_sample_level_P_Q_and_indices.npz](actual_sample_level_P_Q_and_indices.npz)，每个方法包含各16块局部Q和160x160总Γ。溯源与边际残差见 [VISUALIZATION_AUDIT.json](VISUALIZATION_AUDIT.json)。",
      ""]
    (out/"READ_ME_P_Q_VISUALS.md").write_text("\n".join(text),encoding="utf8")
    print("FIGURE OUTPUTS COMPLETE",out,flush=True)
if __name__=="__main__":
    main()
