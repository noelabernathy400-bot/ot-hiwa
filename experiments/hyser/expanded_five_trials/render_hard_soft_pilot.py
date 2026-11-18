#!/usr/bin/env python3
"""Render publication-readable independent point clouds for the heldout Hyser pilot.

Predetermined example = fold 1 seed 42 (not selected by score).
Paired target coloring is visualization ONLY and isolated from training.
"""
import argparse,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from mpl_toolkits.mplot3d import Axes3D

COLORS=["#2776AE","#DC9456","#4F9C83","#9062B0"]
PAIRS=[(0,1),(0,2),(1,2)]
DIV=np.sqrt(3)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    allrun=json.loads(a.results.read_text())["runs"]
    selected={(r["method"],r["holdout_fold"],r["seed"]):r for r in allrun}
    fold=1;seed=42
    with np.load(a.data_dir/f"trial_heldout_{fold}.npz") as z:
        Xtr,Ytr=np.array(z["X_train"])/DIV,np.array(z["Y_train"])/DIV
        Xte,Yte=np.array(z["X_test"])/DIV,np.array(z["Y_test"])/DIV
    cx=KMeans(4,n_init=30,random_state=42).fit(Xtr)
    cy=KMeans(4,n_init=30,random_state=43).fit(Ytr)
    labs_x=cx.predict(Xte)
    labs_y=cy.predict(Yte)
    rh=np.array(selected["hard",fold,seed]["R"])
    rs=np.array(selected["soft05",fold,seed]["R"])
    clouds=[
        ("Source EMG, original",Xte,labs_x),
        ("Target force, posthoc source-color",Yte,labs_x),
        ("Hard: transformed source",Xte@rh.T,labs_x),
        ("Soft tau=0.5: transformed source",Xte@rs.T,labs_x),
        ("Target force, independent K=4",Yte,labs_y)
    ]
    bound=max(np.max(abs(x)) for _,x,_ in clouds)
    bound=math.ceil((bound+.15)*10)/10
    fig,axs=plt.subplots(len(clouds),3,figsize=(14,16),constrained_layout=True)
    for row,(title,z,labs) in enumerate(clouds):
        for col,(u,v) in enumerate(PAIRS):
            ax=axs[row,col]
            for k,c in enumerate(COLORS):
                mask=labs==k
                ax.scatter(z[mask,u],z[mask,v],s=21,c=c,alpha=.78,
                           edgecolors="white",linewidth=.12)
            ax.set(xlim=(-bound,bound),ylim=(-bound,bound),xlabel=f"dim {u+1}",ylabel=f"dim {v+1}")
            ax.set_aspect("equal",adjustable="box");ax.grid(alpha=.18)
            if row==0:ax.set_title(f"projection {u+1}-{v+1}")
            if col==0:
                ax.text(-.32,.5,title,transform=ax.transAxes,rotation=90,
                        va="center",ha="center",fontsize=11)
    fig.suptitle("Hyser first heldout trial (250 points), seed 42; source/target NEVER overlaid",fontsize=16)
    fig.savefig(a.out/"heldout_example_2D_separate.png",dpi=180,bbox_inches="tight")
    plt.close(fig)
    fig=plt.figure(figsize=(17,10))
    for i,(title,z,labs) in enumerate(clouds,1):
        ax=fig.add_subplot(2,3,i,projection="3d")
        for k,c in enumerate(COLORS):
            mask=labs==k
            ax.scatter(z[mask,0],z[mask,1],z[mask,2],
                       s=18,c=c,alpha=.8,edgecolors="white",linewidth=.12,
                       depthshade=False)
        ax.set(xlim=(-bound,bound),ylim=(-bound,bound),zlim=(-bound,bound),
               xlabel="dim 1",ylabel="dim 2",zlabel="dim 3")
        ax.set_box_aspect((1,1,1));ax.view_init(elev=22,azim=-59)
        ax.set_title(title,pad=10)
    fig.suptitle("Hyser heldout trial 1 / seed 42 / independent source and target clouds",fontsize=16)
    fig.savefig(a.out/"heldout_example_3D_separate.png",dpi=180,bbox_inches="tight")
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4.4),constrained_layout=True)
    for ax,(label,R) in zip(axs,[("Hard",selected["hard",fold,seed]),("Soft tau=.5",selected["soft05",fold,seed])]):
        P=np.array(R["P"])
        im=ax.imshow(P,vmin=0,vmax=.25,cmap="Blues")
        ax.set(xticks=range(4),yticks=range(4),xticklabels=range(1,5),yticklabels=range(1,5),
               xlabel="target cluster",ylabel="source cluster",title=label)
        for (i,j),v in np.ndenumerate(P):
            ax.text(j,i,f"{v:.4f}" if v>=.0001 else f"{v:.1e}",
                    color="white" if v>.13 else "#1e3344",ha="center",va="center",fontsize=8)
    fig.colorbar(im,ax=axs,shrink=.78)
    fig.suptitle("Trained group-level P: original numerical values")
    fig.savefig(a.out/"trained_P_comparison.png",dpi=185,bbox_inches="tight")
    plt.close(fig)
    groups=[]
    for fold in range(1,6):
        x=[]
        for seed in [42,43]:
            h=selected["hard",fold,seed];s=selected["soft05",fold,seed]
            x.append({"fold":fold,"seed":seed,
                      "T":s["heldout_common_hard_K4_reference"]["after"]["T"]-
                          h["heldout_common_hard_K4_reference"]["after"]["T"],
                      "W2":s["test_W2_after"]-h["test_W2_after"]})
        groups.extend(x)
    fig,axs=plt.subplots(1,2,figsize=(12.5,4.5),constrained_layout=True)
    for ax,(k,title) in zip(axs,[("T","shared hierarchical T difference (Soft - Hard)"),
                                  ("W2","global W2 squared difference (Soft - Hard)")]):
        ax.axhline(0,color="#6f7377",linewidth=1,linestyle="--")
        for seed,marker,color in [(42,"o","#3585b1"),(43,"s","#c17a40")]:
            data=[x for x in groups if x["seed"]==seed]
            ax.scatter([x["fold"]+(-.07 if seed==42 else .07) for x in data],
                       [x[k] for x in data],s=55,marker=marker,c=color,
                       label=f"seed {seed}",zorder=3)
        ax.set(xlabel="heldout trial",ylabel="Soft minus Hard (negative = Soft better)",
               xticks=range(1,6),title=title)
        ax.grid(alpha=.15);ax.legend(frameon=False)
    fig.suptitle("Fivefold, two-seed paired differences; no selected best seed")
    fig.savefig(a.out/"paired_difference_by_fold.png",dpi=185,bbox_inches="tight")
    plt.close(fig)
    note=[
        "# Hyser五折近似层级OT的补充散点图",
        "",
        "- 示例固定为第1折、随机种子42（事前固定，不是择优挑选）。",
        "- 2D / 3D 每幅图一个域，不允许两个域点云叠加。",
        "- 前四组图的颜色由测试源域通过训练源域中心预测的4簇标签决定；",
        "  目标域第二组使用同一时窗源域的参考颜色，仅在训练完成后绘图。",
        "- 最后一组由目标域自己的训练中心预测标签独立着色；相同色彩并不意味着跨域簇匹配。",
        "- 实验计算、聚类、P/Q/R学习以及评价不使用第二组可视化标签。",
        "- P图展示原训练簇级矩阵。差异图负值表示Soft更好。",
    ]
    (a.out/"PLOTTING_NOTE_ZH.md").write_text("\n".join(note)+"\n",encoding="utf-8")
    print("rendered",(len(list(a.out.glob("*.png")))),"plots")

if __name__=="__main__":
    main()
