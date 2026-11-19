"""Real DANDI 3D / 4D scatter gallery and TRAIN-ONLY Fisher subspaces.

All figures use original DANDI NWB, trial-disjoint internal 320 training
versus 80 validation trials. Entirely postfit figures. Never use original
100 trial heldout or physical paired information for clustering/Fisher.
R is fixed (archived seed42) and not retrained/chosen for these figures.

Source and target Fisher bases are learned INDEPENDENTLY from unsupervised
KMeans groups. Native-space figures are NOT a common coordinate system.
True overlaid comparisons project BOTH Rx and y on the SAME target Fisher
basis, so each overlay is in a valid common coordinate system.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax
from zip_gcot.grouping import soft_groups

from run_hard_clustering import load_trial_windows

PAL=["#3A78B8","#D98B38","#299B83","#995BB6"]
C_SRC="#316AA2"
C_TGT="#D47B38"
plt.rcParams.update({
    "font.family":"DejaVu Sans",
    "font.size":10,
    "axes.titlesize":11,
    "axes.labelsize":9,
    "axes.grid":False
})


def axes_equal_3d(ax,arr):
    m=np.min(arr,axis=0); M=np.max(arr,axis=0)
    c=(m+M)/2.; half=max((M-m).max()*.54,0.001)
    ax.set_xlim(c[0]-half,c[0]+half)
    ax.set_ylim(c[1]-half,c[1]+half)
    ax.set_zlim(c[2]-half,c[2]+half)
    ax.set_box_aspect((1,1,1))
    ax.view_init(elev=22,azim=-60)


def pad_lims_2d(ax,arr):
    q=np.quantile(arr,[.005,.995],axis=0)
    lo,hi=q
    dist=np.maximum(hi-lo,1e-3)
    ax.set_xlim(lo[0]-.08*dist[0],hi[0]+.08*dist[0])
    ax.set_ylim(lo[1]-.08*dist[1],hi[1]+.08*dist[1])
    ax.set_aspect("equal",adjustable="box")
    ax.grid(alpha=.13,linewidth=.5)


def fisher(train_x,train_labels,d_out):
    """S_b v=lambda S_w v; weights and bases use train independent KMeans only."""
    d=train_x.shape[1]
    k=len(np.unique(train_labels))
    assert d_out <= min(d,k-1)
    mu=train_x.mean(axis=0)
    sb=np.zeros((d,d));sw=np.zeros((d,d))
    for label in np.unique(train_labels):
        pts=train_x[train_labels==label]
        group_mu=pts.mean(axis=0)
        v=group_mu-mu
        sb+=len(pts)*np.outer(v,v)
        delta=pts-group_mu
        sw+=delta.T@delta
    reg=1e-5*float(np.trace(sw)/d)
    eigenvalues,basis=eigh(sb,sw+reg*np.eye(d))
    take=np.argsort(eigenvalues)[::-1][:d_out]
    w=basis[:,take]
    # Sort and fix orientation (sign arbitrary, but deterministic for plotting).
    for j in range(d_out):
        p=int(np.argmax(np.abs(w[:,j])))
        if w[p,j]<0:w[:,j]*=-1
    # Cosmetic axis normalization from TRAIN variation only; same target basis
    # & target-axis scale applied to both target and Rx for overlay.
    train_projection=(train_x-mu)@w
    sigma=np.std(train_projection,axis=0,ddof=1)
    assert min(sigma)>0.
    w=w/sigma[None,:]
    return w,mu,eigenvalues[take]


def scatter3d(ax,v,labels,subtitle):
    for idx in range(4):
        mask=labels==idx
        ax.scatter(v[mask,0],v[mask,1],v[mask,2],
                   s=3.1,alpha=.32,color=PAL[idx],rasterized=True)
    axes_equal_3d(ax,v)
    ax.set_xlabel("z1");ax.set_ylabel("z2");ax.set_zlabel("z3")
    ax.set_title(subtitle,pad=13)


def scatter_overlay_3d(ax,x,y,group_labels,subtitle):
    """Gray target background + FOUR SOURCE cluster colors after fixed R.

    Source cluster IDs are independent of target cluster IDs. For Soft, the
    argmax of train-fitted source membership is a visualization-only label.
    """
    ax.scatter(y[:,0],y[:,1],y[:,2],s=2.2,alpha=.18,c="#60666e",
               label="Hand target (gray)",rasterized=True)
    for idx in range(4):
        mask=group_labels==idx
        ax.scatter(x[mask,0],x[mask,1],x[mask,2],s=4.3,alpha=.43,
                   color=PAL[idx],label=f"Neural group {idx+1}",
                   rasterized=True)
    axes_equal_3d(ax,np.concatenate([x,y]))
    ax.set_xlabel("z1");ax.set_ylabel("z2");ax.set_zlabel("z3")
    ax.set_title(subtitle,pad=13)


def scatter2d(ax,v,labels,subtitle):
    for idx in range(4):
        mask=labels==idx
        ax.scatter(v[mask,0],v[mask,1],s=4,alpha=.32,
                   color=PAL[idx],rasterized=True)
    pad_lims_2d(ax,v)
    ax.set_xlabel("Fisher 1");ax.set_ylabel("Fisher 2")
    ax.set_title(subtitle)


def scatter_overlay_2d(ax,x,y,group_labels,subtitle):
    ax.scatter(y[:,0],y[:,1],s=2.6,alpha=.14,c="#60666e",
               label="Hand target (gray)",rasterized=True)
    for idx in range(4):
        mask=group_labels==idx
        ax.scatter(x[mask,0],x[mask,1],s=4.5,alpha=.38,
                   color=PAL[idx],label=f"Neural group {idx+1}",
                   rasterized=True)
    pad_lims_2d(ax,np.concatenate([x,y]))
    ax.set_xlabel("Target Fisher 1");ax.set_ylabel("Target Fisher 2")
    ax.set_title(subtitle)


def make_panels(src,tr_x,va_x,tr_y,va_y,lx,ly,sx,sy,
                Rhard,Rsoft,Wxs,Mxs,Wyt,Myt,dout,use_3d,output):
    """2x3: Hard row and Soft row, each 4 group colors throughout.

    Top left: source hard-KMeans; top middle: independent target hard-KMeans;
    top right: aligned source HARD labels in four colors + target gray.
    Bottom: analogous SOFT source/target hard argmax of soft membership,
    and aligned soft source four colors + target gray.
    Soft colors are visualization argmax only; the actual OT fit used all
    soft membership weights. Fisher itself is fit on train Hard KMeans,
    so both methods share a fixed unsupervised reference subspace.
    """
    if use_3d:
        fig=plt.figure(figsize=(18,11),layout="constrained")
        axs=[fig.add_subplot(2,3,j+1,projection="3d") for j in range(6)]
    else:
        fig,axes=plt.subplots(2,3,figsize=(18,10),layout="constrained")
        axs=list(axes.flat)
    if Wxs is None:
        source=va_x;target=va_y
        aligned_hard=va_x@Rhard.T
        aligned_soft=va_x@Rsoft.T
    else:
        source=(va_x-Mxs)@Wxs
        target=(va_y-Myt)@Wyt
        aligned_hard=(va_x@Rhard.T-Myt)@Wyt
        aligned_soft=(va_x@Rsoft.T-Myt)@Wyt
    if use_3d:
        scatter3d(axs[0],source,lx,"Hard source: 4 KMeans groups")
        scatter3d(axs[1],target,ly,"Hard target: independent groups")
        scatter_overlay_3d(axs[2],aligned_hard,target,lx,
                           "Hard aligned SOURCE groups + gray target")
        scatter3d(axs[3],source,sx,"Soft source: 4 membership argmax groups")
        scatter3d(axs[4],target,sy,"Soft target: independent argmax groups")
        scatter_overlay_3d(axs[5],aligned_soft,target,sx,
                           "Soft aligned SOURCE argmax + gray target")
    else:
        scatter2d(axs[0],source,lx,"Hard source: 4 KMeans groups")
        scatter2d(axs[1],target,ly,"Hard target: independent groups")
        scatter_overlay_2d(axs[2],aligned_hard,target,lx,
                           "Hard aligned SOURCE groups + gray target")
        scatter2d(axs[3],source,sx,"Soft source: 4 argmax groups")
        scatter2d(axs[4],target,sy,"Soft target: independent argmax groups")
        scatter_overlay_2d(axs[5],aligned_soft,target,sx,
                           "Soft aligned SOURCE argmax + gray target")
    for idx in (2,5):
        axs[idx].legend(loc="upper left",fontsize=7,
                        markerscale=2.2,framealpha=.9)
    fig.suptitle(
        f"REAL DANDI:000138 | {src}, 80 validation trials (2800 windows) | "+
        ("native 3D / Fisher3" if use_3d else "train-only Fisher2")+
        " | FOUR cluster colors in all panels",
        fontsize=15)
    fig.text(.5,.007,
        "Top row HARD, bottom row SOFT. All four source group colors are retained AFTER alignment; "
        "GRAY is target background in overlays. Source/target cluster ID colors are independent "
        "(not known matches). Soft labels are argmax for display ONLY; training remains weighted. "
        "Overlay uses TARGET Fisher basis, fixed seed42.",
        ha="center",va="bottom",fontsize=8,wrap=True)
    fig.savefig(output,dpi=185,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    return output


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--nwb",type=Path,required=True)
    p.add_argument("--clusters",type=Path,required=True)
    p.add_argument("--four-d-results",type=Path,required=True)
    p.add_argument("--three-d-results",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    with np.load(args.clusters,allow_pickle=False) as z:
        trainmask=np.array(z["train_mask"],bool)
        testmask=np.array(z["test_mask"],bool)
        trial=np.array(z["trial_of_window"],int)
    ref4=json.loads(args.four_d_results.read_text())
    ref3=json.loads(args.three_d_results.read_text())
    inner_train=np.array(ref4["inner_train_trials"],int)
    inner_val=np.array(ref4["inner_validation_trials"],int)
    trmask=trainmask&np.isin(trial,inner_train)
    vamask=trainmask&np.isin(trial,inner_val)
    assert trmask.sum()==11200 and vamask.sum()==2800
    assert not np.any((trmask|vamask)&testmask)
    nr,br,valid,meta=load_trial_windows(args.nwb)
    assert meta["sha256"]==ref3["raw_nwb_sha256"]==ref4["file_sha256"]
    neural=gaussian_filter1d(np.sqrt(np.maximum(nr,0)),sigma=2,
                             axis=1,mode="nearest").reshape(-1,162)
    hand=br.reshape(-1,4).astype(float)
    neural_scaled=StandardScaler().fit(neural[trmask]).transform(neural)
    hand_scaler=StandardScaler().fit(hand[trmask])
    hand_std=hand_scaler.transform(hand)
    # Training-only hand PCA3; exactly same as earlier 3D OT run.
    handpca=PCA(n_components=3,svd_solver="full",random_state=42).fit(hand_std[trmask])
    assert np.max(abs(handpca.components_ - np.asarray(
        ref3["target_PCA3_components"])))<1e-6
    full_loadings=np.linalg.svd(handpca.components_,full_matrices=True)[2][-1]
    if full_loadings[np.argmax(abs(full_loadings))]<0:
        full_loadings=-full_loadings
    hand_rec_std=handpca.inverse_transform(handpca.transform(hand_std[vamask]))
    hand_rec=hand_scaler.inverse_transform(hand_rec_std)
    hand_true=hand[vamask]
    per_coor_mse=np.mean((hand_rec-hand_true)**2,axis=0)
    per_coor_var=np.var(hand_true,axis=0)
    r2=[float(1-v/w) for v,w in zip(per_coor_mse,per_coor_var)]
    recon={
        "interpretation":"PCA3 axes are measured linear combinations of original physical4D quantities; no axis is a standalone measured physical coordinate. Source PCA/FA axes are neuronal latent variables.",
        "original_physical_channels":["hand_pos_x","hand_pos_y","hand_vel_x","hand_vel_y"],
        "target_PCA3_train_component_loadings":handpca.components_.tolist(),
        "target_train_explained_variance_ratio_each_pc":handpca.explained_variance_ratio_.tolist(),
        "target_explained_variance_ratio_sum":float(
            handpca.explained_variance_ratio_.sum()),
        "discarded_fourth_axis_unit_vector_in_standardized4D":full_loadings.tolist(),
        "validation_reconstruction_original4D_R2_per_coor":r2,
        "validation_reconstruction_original4D_RMSE_per_coor":[
            float(np.sqrt(v)) for v in per_coor_mse],
        "validation_reconstruction_mean_pos_vector_error":float(
            np.mean(np.linalg.norm((hand_rec-hand_true)[:,:2],axis=1))),
        "validation_reconstruction_mean_vel_vector_error":float(
            np.mean(np.linalg.norm((hand_rec-hand_true)[:,2:],axis=1))),
        "all_figures_validation_trials":80,"validation_windows":2800,
        "fisher_basis_and_KMeans_fit":"separate independently, only on internal 320 training trials",
        "overlays_common_coordinate_basis":"TARGET Fisher basis only; never overlay separate Fisher-basis coordinate systems",
        "R_seed":42,"R_selection":"fixed for illustration only, no physical matching cherry-pick",
        "no_old_test_used":True,
        "separate_source_target_cluster_colors_not_same_labels":True,
        "all_postfit_overlays_preserve_four_source_colors":True,
        "postfit_target_overlay_gray_background_only":True,
        "soft_argmax_colors_visual_only_full_probabilities_train":True,
        "plots":{},
        "fisher_diagnostics":{}
    }
    plt.figure(figsize=(8,4))
    plt.imshow(handpca.components_,cmap="coolwarm",vmin=-.8,vmax=.8,aspect="auto")
    plt.colorbar(label="Weight on TRAIN-standardized physical variable")
    plt.xticks(range(4),["pos x","pos y","vel x","vel y"])
    plt.yticks(range(3),["PC1","PC2","PC3"])
    for i in range(3):
        for j in range(4):
            plt.text(j,i,f"{handpca.components_[i,j]:+.2f}",ha="center",
                     va="center",fontsize=10,color="#1c1c1c")
    plt.title("Hand 4D -> PCA3: real train-only loadings; 83.61% retained")
    plt.tight_layout()
    plt.savefig(out/"HAND_TARGET_PCA3_PHYSICAL_LOADINGS.png",dpi=190)
    plt.close()
    for feature in ("PCA3","FA3","PCA4","FA4"):
        dim=int(feature[-1])
        if feature.startswith("PCA"):
            model=PCA(n_components=dim,svd_solver="full",random_state=42)
        else:
            model=FactorAnalysis(n_components=dim,svd_method="randomized",
                random_state=42,max_iter=120,tol=1e-3,iterated_power=4)
        model.fit(neural_scaled[trmask])
        embedded=model.transform(neural_scaled)
        xm=embedded[trmask].mean(0)
        radius=np.sqrt(np.mean(np.sum((embedded[trmask]-xm)**2,axis=1)))
        X=(embedded-xm)/radius
        if dim==3:
            ey=handpca.transform(hand_std)
            ym=ey[trmask].mean(0)
            ry=np.sqrt(np.mean(np.sum((ey[trmask]-ym)**2,axis=1)))
            Y=(ey-ym)/ry
            arch=ref3["methods"][feature]
        else:
            ey=hand_std
            ym=ey[trmask].mean(0)
            ry=np.sqrt(np.mean(np.sum((ey[trmask]-ym)**2,axis=1)))
            Y=(ey-ym)/ry
            arch=ref4["methods"][feature]
        kx=KMeans(n_clusters=4,n_init=20,random_state=42).fit(X[trmask])
        ky=KMeans(n_clusters=4,n_init=20,random_state=42).fit(Y[trmask])
        lx=kx.predict(X[vamask]);ly=ky.predict(Y[vamask])
        # Actual independently fitted soft KMeans/prototype memberships.
        # Visualization uses argmax only; OT has already been trained using
        # original continuous weights, no cross-domain matching labels.
        _,prototype_x,_=soft_groups(X[trmask],4,seed=42,
                                    entropy_weight=1.,maxiter=300)
        _,prototype_y,_=soft_groups(Y[trmask],4,seed=43,
                                    entropy_weight=1.,maxiter=300)
        sx=softmax(X[vamask]@prototype_x.T,axis=1).argmax(1)
        sy=softmax(Y[vamask]@prototype_y.T,axis=1).argmax(1)
        assert set(sx)==set(sy)=={0,1,2,3}
        tr_x=X[trmask];tr_y=Y[trmask]
        va_x=X[vamask];va_y=Y[vamask]
        hard=np.asarray([r for r in arch["Hard"] if r["seed"]==42][0]["R"])
        soft=np.asarray([r for r in arch["Soft"] if r["seed"]==42][0]["R"])
        assert hard.shape==soft.shape==(dim,dim)
        assert np.linalg.norm(hard.T@hard-np.eye(dim))<1e-7
        for fisher_dim in (2,3):
            if fisher_dim>dim or fisher_dim>3:continue
            ws,ms,evs=fisher(tr_x,kx.labels_,fisher_dim)
            wt,mt,evt=fisher(tr_y,ky.labels_,fisher_dim)
            stem=f"{feature}_FISHER{fisher_dim}"
            outfile=out/f"{stem}_SPLIT_AND_ALIGNED.png"
            make_panels(feature,tr_x,va_x,tr_y,va_y,lx,ly,sx,sy,
                       hard,soft,ws,ms,wt,mt,fisher_dim,fisher_dim==3,outfile)
            recon["plots"][stem]=outfile.name
            recon["fisher_diagnostics"][stem]={
                "source_train_fisher_eigenvalues":evs.tolist(),
                "target_train_fisher_eigenvalues":evt.tolist(),
                "source_basis_train_only":ws.tolist(),
                "target_basis_train_only":wt.tolist()
            }
        if dim==3:
            outfile=out/f"{feature}_NATIVE3D_SPLIT_AND_ALIGNED.png"
            make_panels(feature,tr_x,va_x,tr_y,va_y,lx,ly,sx,sy,
                       hard,soft,None,None,None,None,3,True,outfile)
            recon["plots"][f"{feature}_NATIVE3D"]=outfile.name
        print("PLOT_DONE",feature,"base",va_x.shape,flush=True)
    (out/"PHYSICAL_SEMANTICS_AND_FISHER.json").write_text(
        json.dumps(recon,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    rows=["# Original DANDI 4D physical semantics versus PCA3 and Fisher plots","",
          "Only the original 320 training trials fit PCA, FA, KMeans and Fisher; "
          "the displayed 2800 windows are internal heldout 80 validation trials. "
          "Original 100 test trials entirely untouched.","",
          "## Physical meaning of the 3D target coordinates","",
          "Every PCA3 target axis is a MIX of physical hand position x/y and "
          "velocity x/y (after training-only standardization). This is still a "
          "three-dimensional linear physical-kinematics **subspace**, but these "
          "are not three independently measured physical quantities. "
          "Dimension reduction trades away one physical combination, "
          "and the reconstruction diagnostics below quantify that tradeoff.","",
          f"Train PC variance shares = {list(map(lambda v:round(float(v),5),handpca.explained_variance_ratio_))}. "
          f"Sum={handpca.explained_variance_ratio_.sum():.5f}. "
          f"Discarded share={1-handpca.explained_variance_ratio_.sum():.5f}.","",
          "| Original 4D measured variable | PC1 loading | PC2 loading | PC3 loading | Validation reconstruction R2 |",
          "|---|---:|---:|---:|---:|"]
    for j,name in enumerate(("Hand position X","Hand position Y","Hand velocity X","Hand velocity Y")):
        rows.append(f"| {name} | "+ " | ".join(
            [f"{handpca.components_[k,j]:+.4f}" for k in range(3)])+
            f" | {r2[j]:.4f} |")
    rows+=["","## Figure directory",""]
    for name,file in recon["plots"].items():
        rows.append(f"- [{name}]({file}) — Real 2800 validation points, "
                    "HARD row KMeans four colors; SOFT row membership argmax four colors; "
                    "aligned source stays four-color over gray target background in target Fisher basis")
    rows+=["",
           "Do not use plot-enhanced Fisher separation as a model scoring metric. "
           "The Fisher basis is fitted to train HARD KMeans pseudo-labels by maximizing "
           "between/within scatter, so it is explicitly biased towards "
           "making the discovered clusters look separable. "
           "Soft four-color figures use argmax of independent source/target memberships FOR VISUALIZATION ONLY, not hard groups in fitting. "
           "Neural-source Fisher axes and target-hand Fisher axes differ. "
           "Only third and fourth panels overlay on the same target Fisher axes.",
           "Absolute latent OT scores are not comparable across 3D and 4D; "
           "the original physical 4D matching metrics are directly comparable.",
           ""]
    (out/"PHYSICAL_SEMANTICS_AND_FISHER_REPORT.md").write_text(
        "\n".join(rows),encoding="utf8")
    print("FIGURE_GALLERY_COMPLETE",len(recon["plots"]),flush=True)


if __name__=="__main__":
    main()
