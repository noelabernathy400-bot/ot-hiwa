"""Deterministic summary and diagnostic plots for the frozen 4D Hard HiWA run."""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).resolve().parents[1]
    folder=root/"results/hard_hiwa"
    raw=json.loads((folder/"hard_hiwa_results.json").read_text())
    seeds=raw["seed_results"]
    baseline=raw["baseline_identity"]
    oracle=raw["supervised_train_pairing_oracle_diagnostic"]
    stats={}
    for k in ("paired_nrmse","ot_nrmse"):
        a=np.array([s["metrics"][k] for s in seeds])
        stats[k]={"identity":baseline[k],"mean":float(a.mean()),
                  "sd_sample":float(np.std(a,ddof=1)),"min":float(a.min()),
                  "max":float(a.max()),"oracle_diagnostic":oracle[k],
                  "seeds":{str(s["seed"]):s["metrics"][k] for s in seeds}}
    stats["converged_seeds"]=raw["strict_converged"]
    stats["total_seeds"]=len(seeds)
    (folder/"summary_statistics.json").write_text(json.dumps(stats,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
    for ax,k,label in zip(axes,("paired_nrmse","ot_nrmse"),
        ("Synchronized test RMSE / target RMS","Unpaired test OT RMSE / target RMS")):
        ax.scatter([s["seed"] for s in seeds],
                   [s["metrics"][k] for s in seeds],s=50,label="Hard HiWA seeds")
        ax.axhline(baseline[k],linestyle="--",label="Identity before")
        ax.axhline(oracle[k],linestyle=":",label="Supervised training-pair diagnostic")
        ax.set(title=label,xlabel="Initialization seed",ylabel="Dimensionless NRMSE")
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(folder/"heldout_error_comparison.png",dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4.3),constrained_layout=True)
    for s in seeds:
        axes[0].semilogy(range(1,len(s["global_trace"])+1),s["global_trace"],
                        label=f'seed {s["seed"]}')
    axes[0].axhline(raw["parameters"]["audit_tol"],linestyle="--",color="black",
                    label="audit tolerance")
    axes[0].set(title="Global rotation change",xlabel="Outer iteration",
                ylabel="Frobenius norm")
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    first=seeds[0];P=np.asarray(first["P"])
    im=axes[1].imshow(P,cmap="viridis")
    for i in range(4):
        for j in range(4):
            axes[1].text(j,i,f"{P[i,j]:.3f}",ha="center",va="center",
                         color="white" if P[i,j]<P.max()*.65 else "black")
    axes[1].set(title=f'Cluster transport P, seed {first["seed"]}',
                xlabel="Hand cluster",ylabel="Neural cluster",
                xticks=range(4),yticks=range(4))
    fig.colorbar(im,ax=axes[1])
    fig.savefig(folder/"admm_and_group_transport.png",dpi=180);plt.close(fig)

    a=np.load(root/"results/clustering/clustered_4d.npz",allow_pickle=False)
    X=np.asarray(a["neural4"][a["test_mask"]],dtype=float)
    Y=np.asarray(a["hand4"][a["test_mask"]],dtype=float)
    X=(X-np.asarray(raw["source_mean"]))/raw["source_radius"]
    Y=(Y-np.asarray(raw["target_mean"]))/raw["target_radius"]
    R=np.asarray(first["R"]);rng=np.random.default_rng(19)
    idx=rng.choice(len(X),min(1000,len(X)),replace=False)
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    for ax,(v,title) in zip(axes,[(X[idx],"Neural (before)"),
        (X[idx]@R.T,"Hard HiWA (after)"),(Y[idx],"Hand (target)")]):
        ax.scatter(v[:,0],v[:,1],s=4,alpha=.35)
        ax.set(title=title,xlabel="coordinate 1",ylabel="coordinate 2",
               xlim=(-2,2),ylim=(-2,2),aspect="equal")
    fig.suptitle("Held-out distribution view, fixed normalized axes; not paired trajectories")
    fig.savefig(folder/"hard_hiwa_alignment_xy.png",dpi=180);plt.close(fig)
    print(json.dumps(stats,indent=2))


if __name__=="__main__":
    main()
