#!/usr/bin/env python3
"""Aggregate SA/WA/modified DAD-like real 1000-point Hyser controls.

20 fold-by-representation TRAIN Fisher montages show ALL 1000 source
and ALL 1000 target points for EVERY model panel. Full-dimensional
numeric evaluation is never performed in the Fisher 2D plot plane.
"""
from __future__ import annotations
import argparse,csv,json,math,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from best_fisher_2d import fisher2
from additional_unsupervised_baselines import solve_icp,solve_single_ot

REPS=("PCA3_PCA3","FA3_FA3","PCA5_RAW5","FA5_RAW5")
METHODS=("Identity","ICP","SingleOT","SA","WA","DAD_like_3D","Hard HiWA","Soft HiWA")
COLORS=("#2677AA","#DA9051","#4A9376","#A06CB3")

def csvout(path,rows):
    rows=list(rows)
    with path.open("w",newline="",encoding="utf8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

def frozen_run(root,rep,fold,seed):
    dim3=rep.endswith(("PCA3","FA3"))
    folder="results_formal_full1000_3D" if dim3 else "results_formal_full1000_Qfixed"
    prefix="hyser-threeD-" if dim3 else "hyser-qfixed-"
    f=root/folder/"individual_runs"/(f"{prefix}{rep}-f{fold}-s{seed}")/"local_ADMM_results.json"
    z=json.loads(f.read_text())["runs"]
    ret={r["method"]:r for r in z}
    assert set(ret)=={"hard","soft05"}
    assert all(r["n_OT_train_source"]==r["n_OT_train_target"]==1000
               and r["Q_constraints_passed"] for r in z)
    return ret

def features(root,rep,fold):
    dim3=rep.endswith(("PCA3","FA3"))
    d=3 if dim3 else 5
    f=root/rep/(f"trial_heldout_{fold}"+("" if dim3 else "_5d")+".npz")
    with np.load(f) as a:
        x=np.asarray(a["X_train"],float)/np.sqrt(d)
        y=np.asarray(a["Y_train"],float)/np.sqrt(d)
    assert x.shape==y.shape==(1000,d)
    return x,y

def method_rotations(rep,fold,x,y,full,frozen):
    m={"Identity":np.eye(x.shape[1]),
       "ICP":solve_icp(x,y,42)[0],
       "SingleOT":solve_single_ot(x,y,42)[0]}
    for method in ("SA","WA","DAD_like_3D"):
        z=[r for r in full if r["representation"]==rep and r["fold"]==fold
           and r["method"]==method and r["seed"]==42 and r["qualified"]]
        if z:
            assert len(z)==1
            m[method]=np.asarray(z[0]["R"],float)
    h=frozen_run(frozen,rep,fold,42)
    m["Hard HiWA"]=np.asarray(h["hard"]["R"],float)
    m["Soft HiWA"]=np.asarray(h["soft05"]["R"],float)
    assert set(m)==set(METHODS)-({"DAD_like_3D"} if x.shape[1]==5 else set())
    assert all(np.linalg.norm(R.T@R-np.eye(x.shape[1]))<1e-6 for R in m.values())
    return m

def colored_2d(ax, coords, labels, title, xlim=None, ylim=None, legend=False):
    """One cloud per panel. Source and target KMeans colors are INDEPENDENT."""
    assert coords.shape == (1000, 2) and labels.shape == (1000,)
    for k in range(4):
        mask = (labels == k)
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=7, alpha=.60, c=COLORS[k],
                   linewidths=0, rasterized=True,
                   label=f"Own cluster {k + 1}")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set(xlabel="Fisher 1", ylabel="Fisher 2")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=.12)
    if legend:
        ax.legend(loc="best", fontsize=7, frameon=False, ncol=2)


def shared_2d_limits(target, raw_source_in_target, transformed):
    """Exactly the same x/y ranges for target and ALL method output plots."""
    clouds = np.vstack((target, raw_source_in_target, *transformed.values()))
    assert clouds.shape[1] == 2 and np.all(np.isfinite(clouds))
    lo, hi = clouds.min(axis=0), clouds.max(axis=0)
    pad = np.maximum((hi - lo) * .055, .08)
    return ((lo[0] - pad[0], hi[0] + pad[0]),
            (lo[1] - pad[1], hi[1] + pad[1]))


def shared_native3d_limits(target, raw_source, rotations):
    """Original 3D source & target coordinate system, equal axis units."""
    clouds = np.vstack((target, raw_source,
                        *(raw_source @ R.T for R in rotations.values())))
    assert clouds.shape[1] == 3 and np.all(np.isfinite(clouds))
    lo, hi = clouds.min(axis=0), clouds.max(axis=0)
    center = .5 * (lo + hi)
    half_span = .55 * np.max(hi - lo)
    assert half_span > 0
    return tuple((float(c - half_span), float(c + half_span))
                 for c in center)


def colored_native3d(ax, coords, labels, title, limits):
    """No PCA/Fisher projection: scatter genuine native 3D train features."""
    assert coords.shape == (1000, 3) and labels.shape == (1000,)
    for k in range(4):
        ids = labels == k
        ax.scatter(coords[ids, 0], coords[ids, 1], coords[ids, 2],
                   s=6, alpha=.62, c=COLORS[k], linewidths=0,
                   rasterized=True, depthshade=False)
    ax.set_xlim(*limits[0])
    ax.set_ylim(*limits[1])
    ax.set_zlim(*limits[2])
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=24, azim=-62)
    ax.set(xlabel="Coordinate 1", ylabel="Coordinate 2", zlabel="Coordinate 3")
    ax.set_title(title, fontsize=9, pad=7)


def save_separate_2d_montage(path, rep, fold, source_own, target,
                             source_labels, target_labels, methods, limits,
                             source_basis_is_independent=True):
    """Every method row is Target ONLY | Transformed Source ONLY."""
    xlim, ylim = limits
    nrows = 1 + len(methods)
    fig, axes = plt.subplots(nrows, 2, figsize=(12.7, 3.55 * nrows),
                             constrained_layout=True)
    colored_2d(axes[0, 0], source_own, source_labels,
               "Original EMG source · own TRAIN Fisher")
    colored_2d(axes[0, 1], target, target_labels,
               "Original force target · target TRAIN Fisher",
               xlim=xlim, ylim=ylim)
    for row, (method, transformed) in enumerate(methods.items(), start=1):
        colored_2d(axes[row, 0], target, target_labels,
                   f"TARGET ONLY · {method} · n=1000",
                   xlim=xlim, ylim=ylim)
        colored_2d(axes[row, 1], transformed, source_labels,
                   f"{method} · ALIGNED SOURCE ONLY · n=1000",
                   xlim=xlim, ylim=ylim)
    fig.suptitle(
        f"HySeR {rep}, fold {fold} · full1000 train / modality · seed42\n"
        "Each row: TARGET alone | ALIGNED SOURCE alone; common axes for ALL methods.\n"
        "Both domains' cluster colors are learned independently; Fisher 2D only for display.",
        fontsize=11)
    fig.savefig(path, dpi=143)
    plt.close(fig)


def save_separate_3d_montage(path, rep, fold, source, target, source_labels,
                             target_labels, rotations, limits):
    """Both raw point clouds and every transformed one are TRUE original 3D."""
    nrows = 1 + len(rotations)
    fig = plt.figure(figsize=(13.3, 3.65 * nrows), constrained_layout=True)
    axes = [fig.add_subplot(nrows, 2, j + 1, projection="3d")
            for j in range(nrows * 2)]
    colored_native3d(axes[0], source, source_labels,
                     "Original source 3D · 1000", limits)
    colored_native3d(axes[1], target, target_labels,
                     "Original target 3D · 1000", limits)
    for row, (method, R) in enumerate(rotations.items(), start=1):
        colored_native3d(axes[2 * row], target, target_labels,
                         f"TARGET ONLY · {method} · 1000", limits)
        colored_native3d(axes[2 * row + 1], source @ R.T, source_labels,
                         f"{method} · SOURCE AFTER · 1000", limits)
    fig.suptitle(
        f"HySeR {rep}, fold {fold} · NATIVE 3D (NO 2D PROJECTION)\n"
        "Target and aligned source are separate panels, same XYZ ranges, equal axes and camera.",
        fontsize=11)
    fig.savefig(path, dpi=133)
    plt.close(fig)


def all_scatter(out, full, frozen, feat):
    p = out / "figures"
    p.mkdir(exist_ok=True)
    individual2d = p / "individual_fold1"
    individual2d.mkdir(exist_ok=True)
    individual3d = p / "native3D_individual_allfolds"
    individual3d.mkdir(exist_ok=True)
    montages3d = p / "native3D_allmethods"
    montages3d.mkdir(exist_ok=True)
    audit = []
    total3d_individual = 0
    total3d_montage = 0

    for rep in REPS:
        for fold in range(1, 6):
            x, y = features(feat, rep, fold)
            m = method_rotations(rep, fold, x, y, full, frozen)
            kmx = KMeans(4, n_init=30, random_state=42).fit(x)
            kmy = KMeans(4, n_init=30, random_state=43).fit(y)
            source_labels, target_labels = kmx.labels_, kmy.labels_
            fx = fisher2(x, source_labels)
            fy = fisher2(y, target_labels)

            source_own = (x - fx["mean"]) @ fx["basis"]
            target_fisher = (y - fy["mean"]) @ fy["basis"]
            aligned_fisher = {
                method: (x @ R.T - fy["mean"]) @ fy["basis"]
                for method, R in m.items()
            }
            before = (x - fy["mean"]) @ fy["basis"]
            limits2d = shared_2d_limits(target_fisher, before, aligned_fisher)
            name2d = f"{rep}_FOLD{fold}_ALL1000_Fisher_METHODS.png"
            save_separate_2d_montage(
                p / name2d, rep, fold, source_own, target_fisher,
                source_labels, target_labels, aligned_fisher, limits2d
            )
            assert (p / name2d).stat().st_size > 30_000

            # The pre-existing 30 paths are deliberately REPLACED with clear
            # target-only and aligned-source-only side-by-side plots.
            if fold == 1:
                for method, transformed in aligned_fisher.items():
                    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.35),
                                             constrained_layout=True)
                    colored_2d(axes[0], target_fisher, target_labels,
                               "TARGET ONLY · 1000 train points",
                               xlim=limits2d[0], ylim=limits2d[1], legend=True)
                    colored_2d(axes[1], transformed, source_labels,
                               f"{method} · ALIGNED SOURCE ONLY · 1000 train points",
                               xlim=limits2d[0], ylim=limits2d[1], legend=True)
                    fig.suptitle(
                        f"{rep} · fold1 · {method} · same target-trained Fisher axes"
                        " and identical limits; separate source/target panels",
                        fontsize=10)
                    outname = (f"{rep}_FOLD1_{method.replace(' ', '_')}"
                               "_ALL1000_Fisher.png")
                    fig.savefig(individual2d / outname, dpi=169)
                    plt.close(fig)

            entry = {
                "rep": rep, "fold": fold, "file": name2d,
                "source_points_each_method": 1000,
                "target_points_each_method": 1000,
                "method_names": list(m),
                "source_cluster_counts": np.bincount(source_labels, minlength=4).tolist(),
                "target_cluster_counts": np.bincount(target_labels, minlength=4).tolist(),
                "test_not_used_to_fit_fisher": True,
                "source_target_labels_independent": True,
                "target_and_source_in_separate_2d_panels": True,
                "exact_same_2d_axes_across_target_and_methods": True,
                "common_2d_xlimits": list(limits2d[0]),
                "common_2d_ylimits": list(limits2d[1]),
            }
            if x.shape[1] == 3:
                # Genuine raw three-coordinate point clouds, NOT a
                # two-dimensional Fisher rendering or '3D-looking' picture.
                lim3d = shared_native3d_limits(y, x, m)
                name3d = f"{rep}_FOLD{fold}_ALL1000_NATIVE3D_SEPARATE_METHODS.png"
                save_separate_3d_montage(
                    montages3d / name3d, rep, fold, x, y,
                    source_labels, target_labels, m, lim3d
                )
                assert (montages3d / name3d).stat().st_size > 40_000
                total3d_montage += 1

                # Each method/each fold has its own 2-panel original 3D
                # scatter with all 1000+1000 genuine train points.
                for method, R in m.items():
                    fig = plt.figure(figsize=(12.5, 5.8), constrained_layout=True)
                    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
                    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
                    colored_native3d(ax1, y, target_labels,
                                     "TARGET ONLY · original 3D · 1000", lim3d)
                    colored_native3d(ax2, x @ R.T, source_labels,
                                     f"{method} · SOURCE AFTER · original 3D · 1000",
                                     lim3d)
                    fig.suptitle(
                        f"HySer {rep} · fold {fold} · {method} | common true 3D axes",
                        fontsize=10)
                    outname = (f"{rep}_FOLD{fold}_{method.replace(' ', '_')}"
                               "_NATIVE3D_SEPARATE1000.png")
                    fig.savefig(individual3d / outname, dpi=166)
                    plt.close(fig)
                    total3d_individual += 1

                entry.update({
                    "native3d_montage": name3d,
                    "native3d_separate_scatter_original_feature_coordinates": True,
                    "exact_same_xyz_limits_all_methods": True,
                    "native3d_xlim_ylim_zlim": [list(v) for v in lim3d],
                    "native3d_identical_view_and_equal_box_aspect": True,
                })
            audit.append(entry)
            print("FULL1000_SEPARATE_FISHER_NATIVE3D", rep, fold,
                  "methods", len(m), flush=True)

    assert len(audit) == 20 and total3d_montage == 10
    assert total3d_individual == 80
    (out / "scatter_count_audit.json").write_text(
        json.dumps({
            "figures": audit,
            "2D_representation":"TRAIN-fitted target Fisher 2D for after-alignment panels",
            "3D_representation":"NATIVE original full 3-dimensional model coordinates",
            "target_and_aligned_source_never_overlaid": True,
            "target_and_source_separate_panels": True,
            "exact_same_axes_within_each_target_source_comparison": True,
            "each_method_draws_all_1000_source_and_all_1000_target": True,
            "n_fold_gallery_2d": 20,
            "n_fold_gallery_3d": total3d_montage,
            "n_single_method_3d_allfold": total3d_individual,
            "n_single_method_2d_fold1": 30,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--baselines-dir",required=True,type=Path)
    ap.add_argument("--features",required=True,type=Path)
    ap.add_argument("--frozen",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    new=[]
    for rep in REPS:
        obj=json.loads((a.baselines_dir/rep/"runs_bundle.json").read_text())
        rs=obj["all_results_including_failures"]
        assert obj["representation"]==rep and obj["full1000_train_data"]
        assert len(rs)==(20 if rep.endswith(("PCA3","FA3")) else 15)
        assert all(r["source_train_n"]==r["target_train_n"]==1000 for r in rs)
        new.extend(rs)
    assert len(new)==70 and len({(r["representation"],r["fold"],r["seed"],r["method"]) for r in new})==70
    old={}
    for d in ("3D","5D"):
        p=a.frozen/"results_3D_5D_unified"/d/"full_physical_distance_curves.json"
        obj=json.loads(p.read_text())
        assert obj["n_runs"]==40
        for z in obj["records"]:
            k=(z["config"],z["fold"],z["seed"],z["method"])
            assert k not in old
            old[k]=z
    assert len(old)==80
    (a.out/"all_SA_WA_DAD_run_records.json").write_text(json.dumps({
        "DANDI_algorithm_source_commit":"38fb630001e11f5045a2a1055b2aa2686e7bfd34",
        "full_1000_training_pool":True,
        "WA_stochastic_batch_size":250,
        "DAD_is_modified_3D_not_official":True,
        "results":new
    },ensure_ascii=False,indent=2)+"\n",encoding="utf8")

    summary=[];perfold=[];curve=[]
    grouped={}
    for rep in REPS:
        methods=("Hard HiWA","Soft HiWA","SA","WA")+(() if rep.endswith("RAW5") else ("DAD_like_3D",))
        for method in methods:
            if method.startswith(("Hard","Soft")):
                tag="hard" if method.startswith("Hard") else "soft05"
                rows=[{"fold":f,"seed":seed,"qualified":True,
                       "hit5":old[rep,f,seed,tag]["results"]["physical_hit"]["0.05"],
                       "hit20":old[rep,f,seed,tag]["results"]["physical_hit"]["0.2"],
                       "L2":old[rep,f,seed,tag]["results"]["physical_error_mean"],
                       "errors":old[rep,f,seed,tag]["results"]["physical_errors_250"]}
                      for f in range(1,6) for seed in (42,43)]
            else:
                rows=[{"fold":z["fold"],"seed":z["seed"],"qualified":z["qualified"],
                       "hit5":z.get("hit_q05"),"hit20":z.get("hit_q20"),
                       "L2":z.get("mean_physical_l2"),
                       "errors":z.get("physical_errors_250")}
                      for z in new if z["representation"]==rep and z["method"]==method]
            good=[z for z in rows if z["qualified"]]
            grouped[(rep,method)]=good
            summary.append({"representation":rep,"method":method,"n_runs":len(rows),
                "n_qualified":len(good),"n_valid_folds":len(set(z["fold"] for z in good)),
                "hit5":float(np.mean([z["hit5"] for z in good])) if good else None,
                "hit20":float(np.mean([z["hit20"] for z in good])) if good else None,
                "physical_mean_L2":float(np.mean([z["L2"] for z in good])) if good else None})
            for fold in range(1,6):
                r=[z for z in good if z["fold"]==fold]
                perfold.append({"representation":rep,"method":method,"fold":fold,
                    "n_qualified":len(r),"n_attempted":len([z for z in rows if z["fold"]==fold]),
                    "hit5":float(np.mean([z["hit5"] for z in r])) if r else None,
                    "hit20":float(np.mean([z["hit20"] for z in r])) if r else None,
                    "physical_L2":float(np.mean([z["L2"] for z in r])) if r else None})
            if good:
                errors=np.concatenate([np.asarray(z["errors"],float) for z in good])
                assert len(errors)==250*len(good)
                for eps in np.arange(0,.601,.01):
                    curve.append({"representation":rep,"method":method,
                                  "epsilon":round(float(eps),4),
                                  "hit":float(np.mean(errors<=eps))})
    csvout(a.out/"summary_SA_WA_DAD_HiWA.csv",summary)
    csvout(a.out/"all_fivefold_methods.csv",perfold)
    csvout(a.out/"dense_physical_tolerance_all_methods.csv",curve)
    assert all(z["n_qualified"]==z["n_runs"] for z in summary),summary

    all_scatter(a.out,new,a.frozen,a.features)
    graphs=a.out/"figures"
    for rep in REPS:
        subset=[z for z in summary if z["representation"]==rep]
        fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
        ax.barh(np.arange(len(subset)),[z["hit5"]*100 for z in subset])
        ax.set(yticks=np.arange(len(subset)),yticklabels=[z["method"] for z in subset],
               xlabel="True 5-finger Hit@train 5%-quantile (%)",
               title=f"{rep}: full1000 unpaired SA WA DAD-like versus HiWA")
        ax.invert_yaxis()
        fig.savefig(graphs/f"{rep}_Hit5_all_methods.png",dpi=170)
        plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,5),constrained_layout=True)
        for method in [z["method"] for z in subset]:
            part=[z for z in curve if z["representation"]==rep and z["method"]==method]
            ax.plot([z["epsilon"] for z in part],[100*z["hit"] for z in part],
                    label=method,lw=1.7)
        ax.axhline(70,ls=":",color="gray")
        ax.set(xlim=(0,.6),ylim=(0,100),
               xlabel="Raw 5-finger force L2 threshold (not Newton)",
               ylabel="Physical hit (%)",
               title=f"{rep}: matching-distance tolerance curves")
        ax.legend(fontsize=8);ax.grid(alpha=.14)
        fig.savefig(graphs/f"{rep}_eps_curves_all_methods.png",dpi=170)
        plt.close(fig)

    lines=["# HySeR：SA、WA、DAD-like 与 Hard/Soft HiWA 真实1000点实验",
      "","## 已实际运行的对照及限制",
      "正式对照以单一受试者 HySeR S01/S1 Random Task 五个试次做留一训练验证，四次训练试次各域1000点，剩余一次各域250测试点；源和目标无配对拟合，单点检索只访问1000目标训练点，原始五指真值只用于事后评分。",
      "",
      "- SA 是 DANDI 实现的全基 PCA Subspace Alignment 转写，每折确定性运行一次。",
      "- WA 为 DANDI 的退火 Sinkhorn/正交 Procrustes（200外迭代、温度100至0.1），在1000训练池采用250/轮随机小批量：每4轮完整覆盖1000点、重复50轮。这是1000池上的随机 WA 变体，并非原DANDI 160点全批次的逐步重现。",
      "- DAD-like 仅对3D实现 DANDI 的 k=5 对称密度KL代理、128网格加Powell局部优化，明确不是KordingLab原版DAD；不编造5D DAD。",
      "- Hard/Soft HiWA使用先前冻结的每折全1000训练点局部P/Q/Rkl模型。",
      "- 匹配成功是五指原始物理力量距离低于各外折训练内非对应力距离的5%或20%分位。Fisher二维只用于可视化，不影响评价。",
      "- 每个方法各有独立的目标域和变换后源域散点图，左右统一轴范围，每张图包含真正1000点；PCA3/FA3另绘制完整三维（不是Fisher二维）散点，不进行稀疏采样。",
      "",
      "## 全五折物理评价（越高越好；L2越低越好）",
      "",
      "|特征表示|方法|合格/运行|Hit@5%|Hit@20%|平均五指力量L2|",
      "|---|---|---:|---:|---:|---:|"]
    for z in summary:
        def fnum(x,fmt):return "NA" if x is None else format(x,fmt)
        lines.append(f"|{z['representation']}|{z['method']}|{z['n_qualified']}/{z['n_runs']}|"+
          f"{fnum(z['hit5'],'.2%')}|{fnum(z['hit20'],'.2%')}|{fnum(z['physical_mean_L2'],'.5f')}|")
    lines+=["","## 五折1000点分离散点图（二维 Fisher；三维完整坐标）",
       "源域和目标域分别由训练数据拟合独立KMeans与Fisher；对齐后源域统一投到目标训练Fisher平面。**绝不重叠绘制两个域**：每个方法左图仅目标1000点，右图仅变换后的源1000点，二者和其它方法统一使用相同xlim/ylim及等轴尺度。目标与源各自独立的四簇着色只表达本域伪簇，不隐含真实簇对应关系。对于PCA3/FA3，另有原始全三维1000点左右并排视图，使用同一XYZ范围和相同相机视角。所有数值评价仍在完整d维和原始五指力量中。",
       ""]
    for rep in REPS:
        lines.append(f"### {rep}：五折每个方法全部1000点")
        for fold in range(1,6):
            fn=f"figures/{rep}_FOLD{fold}_ALL1000_Fisher_METHODS.png"
            lines.append(f"- [留出试次 {fold}：全部方法 Fisher 左右分离、同尺度1000+1000对照]({fn})")
        if rep.endswith(("PCA3","FA3")):
            for fold in range(1,6):
                fn=(f"figures/native3D_allmethods/"
                    f"{rep}_FOLD{fold}_ALL1000_NATIVE3D_SEPARATE_METHODS.png")
                lines.append(
                    f"- [留出试次 {fold}：完整三维1000点，所有方法左右分离同XYZ尺度]({fn})"
                )
        lines.append(f"- [该表示多方法严格Hit@5%图](figures/{rep}_Hit5_all_methods.png)")
        lines.append(f"- [该表示真实距离阈值曲线](figures/{rep}_eps_curves_all_methods.png)")
        lines.append("")
    lines+=["## 二维单独双面板散点图（Fold1，每个域各1000点，绝不重叠）",""]
    for rep in REPS:
        for method in METHODS:
            if method=="DAD_like_3D" and rep.endswith("RAW5"):continue
            fn=f"figures/individual_fold1/{rep}_FOLD1_{method.replace(' ','_')}_ALL1000_Fisher.png"
            lines.append(f"- [{rep} / {method} / 1000源+1000目标]({fn})")
    lines+=["","## 额外完整三维分离图：PCA3/FA3 每个方法、每一折",""]
    for rep in ("PCA3_PCA3","FA3_FA3"):
        for fold in range(1,6):
            for method in METHODS:
                fn=(f"figures/native3D_individual_allfolds/"
                    f"{rep}_FOLD{fold}_{method.replace(' ','_')}"
                    "_NATIVE3D_SEPARATE1000.png")
                lines.append(f"- [{rep} / trial {fold} / {method}：3D目标1000与源1000独立同尺度]({fn})")
    lines+=["","## 数值存档和严谨性",
        "全部新模型的正交R、每条测试样本真实五指误差、训练配置和数值状态都在下列JSON。上表按相同试次折比较，不能把同一被试的两种子当独立受试者；也不能将事后阈值70%作为预注册泛化准确率。",
        "",
        "- [所有SA/WA/DAD-like逐折逐种子原始数据](all_SA_WA_DAD_run_records.json)",
        "- [逐试次完整对照CSV](all_fivefold_methods.csv)",
        "- [汇总表CSV](summary_SA_WA_DAD_HiWA.csv)",
        "- [实际物理误差阈值曲线CSV](dense_physical_tolerance_all_methods.csv)",
        "- [1000点图像数目、XYZ轴一致性及独立分簇审计](scatter_count_audit.json)",
        "- [此前额外Identity/ICP/单层OT](../additional_baselines/ADDITIONAL_UNSUPERVISED_BASELINES_ZH.md)",
        "- [此前80次HiWA共享P/Q层级目标](../shared_hierarchical_full250/SHARED_HIERARCHICAL_EVALUATION_ZH.md)",
        "- [DANDI原迁移代码源分支](https://github.com/noelabernathy400-bot/ot-hiwa/blob/experiment/dandi-pca3-fa3-vs-pca4-fa4-20261119/projects/dandi-000138-hiwa-4d/code/closeout_dandi_sa_wa_dad_baselines.py)",
        "",
        "这是一组特定参数的实际基线复核，不是作者原SA/WA/DAD所有超参的最优性或统计跨人群显著性证明。"]
    (a.out/"SA_WA_DAD_HYSER_1000_FINAL_REPORT_ZH.md").write_text(
        "\n".join(lines)+"\n",encoding="utf8")
    print("HYSER_SAWAD_1000_DONE",len(new),sum(z["qualified"] for z in new),
          "n_figures",len(list(graphs.rglob("*.png"))),
          "SUMMARY",json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
