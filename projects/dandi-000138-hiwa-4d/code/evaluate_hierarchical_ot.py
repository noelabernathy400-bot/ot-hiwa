"""Posthoc OT evaluations of frozen 500-iteration, one-source-KMeans HiWA.

The saved model provides global orthogonal R and group P, but NOT final local
R_ij / Q_ij / C_ij. P-weighted global-R group costs are therefore TEST-TIME
proxies, not the exact original hierarchical training objective.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ot_sq(a,b):
    assert a.shape==b.shape and a.ndim==2 and a.shape[1]==4
    cost=cdist(a,b,"sqeuclidean")
    i,j=linear_sum_assignment(cost)
    return float(cost[i,j].mean())


def grouped_cost(a,b,R):
    C=np.zeros((4,4),float)
    for i in range(4):
        for j in range(4):
            C[i,j]=ot_sq(a[i]@R.T,b[j])
    return C


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--clusters",type=Path,required=True)
    p.add_argument("--alignment",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--per-group",type=int,default=128)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    h=json.loads(args.alignment.read_text(encoding="utf8"))
    sig=hashlib.sha256(args.clusters.read_bytes()).hexdigest()
    assert sig==h["data_sha256"],"Data do not match archived HiWA run"
    with np.load(args.clusters,allow_pickle=False) as z:
        X=np.asarray(z["neural4"],float)
        Y=np.asarray(z["hand4"],float)
        tr=np.asarray(z["train_mask"],bool)
        te=np.asarray(z["test_mask"],bool)
        lx=np.asarray(z["neural_groups"],int)
        ly=np.asarray(z["hand_groups"],int)
        lc=np.asarray(z["shared_labels"],int)
    assert np.array_equal(lx,ly) and np.array_equal(lx,lc)
    assert X.shape==Y.shape==(17500,4) and tr.sum()==14000 and te.sum()==3500
    X=(X-np.asarray(h["source_mean"]))/float(h["source_radius"])
    Y=(Y-np.asarray(h["target_mean"]))/float(h["target_radius"])
    x=X[te];y=Y[te];c=lc[te]
    masses=np.bincount(c,minlength=4)/len(c)
    # Source and target points independently drawn per inherited-label group.
    rngx=np.random.default_rng(2026111701)
    rngy=np.random.default_rng(2026111702)
    a=[];b=[]
    for k in range(4):
        eligible=np.flatnonzero(c==k)
        assert len(eligible)>=args.per_group
        a.append(x[rngx.choice(eligible,args.per_group,replace=False)])
        b.append(y[rngy.choice(eligible,args.per_group,replace=False)])
    # Original 200+200 independent test points, exactly the same RNG as runner.
    rng=np.random.default_rng(20261117)
    n=int(h["evaluation_unpaired_subset_n"])
    ix=rng.choice(len(x),n,replace=False)
    iy=rng.choice(len(y),n,replace=False)
    v0=ot_sq(x[ix],y[iy])
    assert abs(np.sqrt(v0)-h["baseline_identity"]["ot_rmse"])<1e-9
    C0=grouped_cost(a,b,np.eye(4))
    wd=float(masses@np.diag(C0))
    ud=float(np.trace(C0)/4)
    results=[]
    C42=None
    for row in h["seed_results"]:
        R=np.asarray(row["R"],float)
        P=np.asarray(row["P"],float)
        assert R.shape==(4,4) and P.shape==(4,4)
        assert np.linalg.norm(R.T@R-np.eye(4),"fro")<1e-7
        assert np.max(np.abs(P.sum(axis=0)-.25))<1e-5
        assert np.max(np.abs(P.sum(axis=1)-.25))<1e-5
        vc=ot_sq(x[ix]@R.T,y[iy])
        assert abs(np.sqrt(vc)-row["metrics"]["ot_rmse"])<1e-9
        C=grouped_cost(a,b,R)
        result={
          "seed":int(row["seed"]),
          "converged":bool(row["strict_converged"]),
          "unrestricted_W2_squared":vc,
          "unrestricted_W2":float(np.sqrt(vc)),
          "same_group_W2_squared_empirical_masses":float(masses@np.diag(C)),
          "same_group_W2_squared_uniform_masses":float(np.trace(C)/4),
          "P_weighted_W2_squared_after":float(np.sum(P*C)),
          "P_weighted_W2_squared_before_same_P":float(np.sum(P*C0)),
          "P_diagonal_mass":float(np.trace(P)),
          "group_cost_matrix":C.tolist(),
          "learned_P":P.tolist(),
        }
        results.append(result)
        if row["seed"]==42:C42=C
    assert len(results)==5 and all(z["converged"] for z in results)
    fields=["unrestricted_W2_squared","same_group_W2_squared_empirical_masses",
            "same_group_W2_squared_uniform_masses","P_weighted_W2_squared_after",
            "P_weighted_W2_squared_before_same_P","P_diagonal_mass"]
    stats={}
    for field in fields:
        values=np.array([z[field] for z in results])
        stats[field]={"mean":float(values.mean()),"sample_sd":float(values.std(ddof=1))}
    data={
       "dataset":"DANDI:000138, v0.220113.0407",
       "input_sha256":sig,
       "alignment_sha256":hashlib.sha256(args.alignment.read_bytes()).hexdigest(),
       "no_refitting":True,
       "one_source_kmeans":True,"target_labels_inherited_by_paired_index":True,
       "no_reclustering":True,
       "metric":"mean squared normalized 4D Euclidean OT distance W2 squared",
       "unrestricted_method":"Exact 200-by-200 balanced assignment OT on independently chosen heldout subsets, identical to original HiWA evaluation sample.",
       "same_label_method":"Exact 128-by-128 balanced OT for each of the four inherited-label groups, with independent fixed source/target samples; empirical test-mass weighting.",
       "hierarchical_P_method":"Posthoc sum_ij P_ij W2sq(R_global source_group_i, target_group_j), same saved learned P before/after. Uses global R and no entropy. NOT original local-R and local-Q nested training loss.",
       "group_sample_per_domain":args.per_group,
       "test_group_masses":masses.tolist(),
       "baseline":{"unrestricted_W2_squared":v0,
                   "unrestricted_W2":float(np.sqrt(v0)),
                   "same_group_W2_squared_empirical_masses":wd,
                   "same_group_W2_squared_uniform_masses":ud},
       "by_seed":results,"mean_sd":stats
    }
    (args.output/"OT_HELDOUT_METRICS.json").write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    # Plot 3 noninterchangeable transport questions with matched baseline per seed.
    f,axs=plt.subplots(1,3,figsize=(13.5,4.4),constrained_layout=True)
    cases=[
     ("All points: free OT","unrestricted_W2_squared",v0,None),
     ("Same-label groups: OT","same_group_W2_squared_empirical_masses",wd,None),
     ("Learned group P: OT","P_weighted_W2_squared_after",None,"P_weighted_W2_squared_before_same_P")]
    for ax,(name,field,base,bfield) in zip(axs,cases):
        for i,r in enumerate(results):
            before=r[bfield] if bfield else base
            after=r[field]
            ax.plot([i-.15,i+.15],[before,after],color="0.5",lw=1)
            ax.scatter(i-.15,before,marker="s",color="0.2",label="Before" if i==0 else "")
            ax.scatter(i+.15,after,color=plt.get_cmap("tab10")(i),label="HiWA" if i==0 else "")
        ax.set(title=name,xlabel="seed",ylabel="Squared 4D OT cost",
               xticks=np.arange(5),xticklabels=[str(r["seed"]) for r in results])
        ax.grid(alpha=.16)
        ax.legend(fontsize=8)
    f.suptitle("Held-out OT costs before/after | Same frozen four clusters",fontsize=14)
    f.savefig(args.output/"ot_three_costs_before_after.png",dpi=190,bbox_inches="tight")
    plt.close(f)
    # Group plan is generally NONDIAGONAL: visible diagnostic.
    P42=np.asarray(results[0]["learned_P"])
    f,axs=plt.subplots(1,3,figsize=(12.5,4.1),constrained_layout=True)
    maxc=max(float(C0.max()),float(C42.max()))
    for ax,m,title,vmax in zip(axs,[C0,C42,P42],
        ["Group OT: before","Group OT: after (seed 42)","Learned HiWA P (seed 42)"],
        [maxc,maxc,.25]):
        im=ax.imshow(m,cmap="viridis",vmin=0,vmax=vmax)
        ax.set(title=title,xlabel="Target inherited group",ylabel="Source KMeans group",
               xticks=range(4),yticks=range(4))
        for i in range(4):
            for j in range(4):
                ax.text(j,i,f"{m[i,j]:.2f}",ha="center",va="center",
                        fontsize=8,color="white" if m[i,j]<vmax*.42 else "black")
        f.colorbar(im,ax=ax,shrink=.78)
    f.suptitle("Free cross-group P vs user-defined inherited same-group identity")
    f.savefig(args.output/"ot_group_cost_and_P_seed42.png",dpi=190,bbox_inches="tight")
    plt.close(f)

    def fmt(x):return f"{x:.5f}"
    lines=[
      "# 固定组 Hard HiWA：留出测试集最优传输代价",
      "",
      "本轮只评估此前已经保存的五组 HiWA 全局旋转 R 和 4x4 簇间传输矩阵 P，"
      "完全没有重新训练、重新聚类或修改样本标签。",
      "所有下列 W2² 均为归一化四维欧氏距离的平方，单位无量纲。",
      "",
      "## 三种不同的 OT 问题",
      "",
      "- 全局自由 OT：两个域的测试样本各抽 200 个，最小代价一对一匹配，"
      "允许跨组传输，与此前 hard_hiwa_results.json 中的 ot_rmse 完全一致。",
      "- 同编号组 OT：目标继承神经源域的组号，只允许组 k 对组 k，"
      "每组在两域各独立抽取 128 个测试点，并按测试集经验组质量加权。",
      "- 原版求解器学到的 P 加权组间代价：固定每个 seed 原来的 P，"
      "比较 I 和全局 R 下 16 个组对的精确测试 OT 代价加权平均。"
      "这里允许非对角传输，仅为测试代理，不是原版嵌套目标的精确损失："
      "原版局部 R_ij 和 Q_ij 未保存，熵正则项也没有纳入。",
      "",
      f"测试组质量：{np.round(masses,6).tolist()}。",
      "",
      "## 均值（五个随机种子）",
      "",
      "| 代价指标（越低越好） | 对齐前 | 对齐后均值 ± 样本标准差 | 相对改善 |",
      "|---|---:|---:|---:|"
    ]
    entries=[
       ("全局自由 OT W2²","unrestricted_W2_squared",v0),
       ("共享标签同组 OT W2²","same_group_W2_squared_empirical_masses",wd),
       ("等组质量同组 OT W2²","same_group_W2_squared_uniform_masses",ud),
    ]
    for title,key,prior in entries:
        ss=stats[key];rel=100*(prior-ss["mean"])/prior
        lines.append(f"| {title} | {fmt(prior)} | {fmt(ss['mean'])} ± {fmt(ss['sample_sd'])} | {rel:+.2f}% |")
    bef=stats["P_weighted_W2_squared_before_same_P"]["mean"]
    aft=stats["P_weighted_W2_squared_after"]["mean"]
    lines.append(f"| 原版 P 加权组间代价（P 固定） | {fmt(bef)} | {fmt(aft)} ± {fmt(stats['P_weighted_W2_squared_after']['sample_sd'])} | {100*(bef-aft)/bef:+.2f}% |")
    lines.extend(["","## 每个 seed 的 OT 代价",
                  "| seed | 全局自由 W2² | 同组 W2² | P 加权 before | P 加权 after | P 对角总质量 |",
                  "|---:|---:|---:|---:|---:|---:|"])
    for row in results:
        lines.append(f"| {row['seed']} | {fmt(row['unrestricted_W2_squared'])} | {fmt(row['same_group_W2_squared_empirical_masses'])} | {fmt(row['P_weighted_W2_squared_before_same_P'])} | {fmt(row['P_weighted_W2_squared_after'])} | {fmt(row['P_diagonal_mass'])} |")
    lines+=["","![Three OT errors before and after](ot_three_costs_before_after.png)",
            "","![Four-by-four costs and learned P](ot_group_cost_and_P_seed42.png)",
            "","## 如何解释",
            "OT 分布误差小，不等于同一时刻的神经活动可以准确预测手部运动。"
            "两类误差评价目标不同。特别注意：用户要求的固定标签不代表"
            "原版 HiWA 的 P 被限制为对角阵；如果要只允许同组传输，"
            "应看上面的同编号组 OT，而不是直接把原版 P 当作组身份映射。",
            "","研究复现：模型和训练标签哈希已核对；"
            "对全局自由 OT 的每一轮计算均断言等于原有结果文件中的 ot_rmse。",
            ""]
    (args.output/"OT_EVALUATION.md").write_text("\n".join(lines),encoding="utf-8")
    print("SAVED OT RESULTS",json.dumps({"baseline":data["baseline"],"mean_sd":stats,"diagonal_P":[r["P_diagonal_mass"] for r in results]},indent=2),flush=True)


if __name__=="__main__":
    main()
