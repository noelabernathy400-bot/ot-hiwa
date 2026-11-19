"""DANDI138 read-only preflight of cluster semantics and frozen-model OT retrieval.

Uses strictly independently learned X/Y groups and the archived 5-seed
Hard/Soft-GCOT R. NEITHER fit nor tune any model. True synchronized hand
location/velocity is read strictly after applying the frozen R and selecting
the nearest-neighbor target index. Train behavior is used as postfit group
interpretation and to preregister physically meaningful thresholds.

Pilot is exploratory: heldout 100 trials have been inspected in prior
studies, so downstream feature choice requires internal train-only validation
and fresh final confirmation.
"""
from __future__ import annotations
import argparse,hashlib,json
from itertools import permutations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.special import softmax

PAL=("#3178b8","#e38933","#29a57f","#9b70bd")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                     "axes.titlesize":10,"figure.titlesize":13})

def metric_mi(c,l,k=4):
    """Mutual information fraction of H(l); requires contiguous nonnegative labels."""
    c=np.asarray(c,int);l=np.asarray(l,int)
    assert len(c)==len(l) and np.min(l)>=0
    nlabel=int(l.max())+1
    z=np.bincount(c*nlabel+l,minlength=k*nlabel).reshape(k,nlabel).astype(float)
    z/=z.sum()
    pr=z.sum(axis=1,keepdims=True);pc=z.sum(axis=0,keepdims=True)
    nz=z>0
    mi=np.sum(z[nz]*np.log(z[nz]/(pr@pc)[nz]))
    H=-np.sum(pc[pc>0]*np.log(pc[pc>0]))
    return float(mi/max(H,1e-12))

def trial_shuffle_null(clusters,labs,valid,train_indices,trial_id,n_trials,n_perm=120):
    rng=np.random.default_rng(20261119)
    assert train_indices.size==n_trials*35
    # Full trial windows consecutive in original source, train heldout trial-disjoint.
    c=clusters[train_indices].reshape(n_trials,35)
    v=labs[train_indices].reshape(n_trials,35)
    eligible=valid[train_indices].reshape(n_trials,35)
    obs=metric_mi(c[eligible],v[eligible])
    null=[]
    for _ in range(n_perm):
        choice=rng.permutation(n_trials)
        vp=v[choice]
        mask=eligible & valid[train_indices][choice.repeat(35)*0:choice.repeat(35)*0+0].reshape(0,35) if False else eligible
        # Eligibility is defined by original SOURCE true speed, so for the
        # correct shuffled-label null, combine the permuted target eligibility.
        perm_valid=valid[train_indices].reshape(n_trials,35)[choice]
        mask=eligible&perm_valid
        null.append(metric_mi(c[mask],vp[mask]))
    n=np.array(null)
    return dict(observed=obs,trial_shuffle_null_mean=float(n.mean()),
                trial_shuffle_null_95percent=float(np.quantile(n,.95)),
                above_null_95=bool(obs>np.quantile(n,.95)),permutations=n_perm)

def cnt_table(cluster,category,valid,num_cat=8):
    ix=valid & (category>=0)
    a=np.bincount(cluster[ix]*num_cat+category[ix],minlength=4*num_cat).reshape(4,num_cat)
    return a

def cluster_geometry(x,y,source_labels,target_labels,train):
    a=np.array([x[train&(source_labels==k)].mean(0) for k in range(4)])
    b=np.array([y[train&(target_labels==k)].mean(0) for k in range(4)])
    da=cdist(a,a);db=cdist(b,b)
    da/=da[np.triu_indices(4,1)].mean()
    db/=db[np.triu_indices(4,1)].mean()
    pp=list(permutations(range(4)))
    errs=[np.linalg.norm(da-db[np.ix_(p,p)],'fro')/np.linalg.norm(da,'fro') for p in pp]
    j=int(np.argmin(errs))
    return dict(source_normalized_cluster_centroid_distances=da.tolist(),
          target_normalized_cluster_centroid_distances=db.tolist(),
          best_unsupervised_permutation=list(pp[j]),
          best_relative_distance_matrix_discrepancy=float(errs[j]),
          mean_permutation_discrepancy=float(np.mean(errs)),
          second_best_discrepancy=float(sorted(errs)[1]),
          caution="Best centroid geometry alone is NOT proof of physical cluster correspondence or a learned P.")


def categories_from_raw(raw,phase,train,low):
    vel=raw[:,2:]
    speed=np.linalg.norm(vel,axis=1)
    theta=(np.arctan2(vel[:,1],vel[:,0])+2*np.pi)%(2*np.pi)
    cat8=np.floor(theta/(2*np.pi)*8).astype(int).clip(0,7)
    valid=speed>=low
    pos=raw[:,:2];med=np.median(pos[train],axis=0)
    position_quadrant=(pos[:,0]>=med[0]).astype(int)+2*(pos[:,1]>=med[1]).astype(int)
    edges=np.quantile(speed[train],[1/3,2/3])
    velocity_tertile=np.digitize(speed,edges).astype(int)
    return dict(direction8=cat8,valid_direction=valid,position_quadrant=position_quadrant,
         speed_tertile=velocity_tertile,phase=phase),dict(
         position_median_train=med.tolist(),speed_tertile_train_edges=edges.tolist(),
         speed_minimum_for_direction=float(low))


def retrieval_metrics(match,raw_test,low,pos_scale):
    n=len(raw_test)
    true=raw_test
    pred=raw_test[match]
    epos=np.linalg.norm(pred[:,:2]-true[:,:2],axis=1)
    evel=np.linalg.norm(pred[:,2:]-true[:,2:],axis=1)
    tvel=true[:,2:];pvel=pred[:,2:]
    nt=np.linalg.norm(tvel,axis=1);npred=np.linalg.norm(pvel,axis=1)
    valid=nt>=low
    both=valid&(npred>=low)
    angle=np.full(n,np.pi)
    cosine=np.sum(tvel[both]*pvel[both],axis=1)/(nt[both]*npred[both])
    angle[both]=np.arccos(np.clip(cosine,-1,1))
    truedir=np.floor(((np.arctan2(tvel[:,1],tvel[:,0])+2*np.pi)%(2*np.pi))/(2*np.pi)*8).astype(int)
    preddir=np.floor(((np.arctan2(pvel[:,1],pvel[:,0])+2*np.pi)%(2*np.pi))/(2*np.pi)*8).astype(int)
    assert np.sum(valid)>0
    return dict(position_mean=float(epos.mean()),position_median=float(np.median(epos)),
       speed_vector_mean_error=float(evel.mean()),
       speed_valid_fraction=float(valid.mean()),
       matched_candidate_low_speed_fraction=float(np.mean(~both[valid])),
       pos_hit5pct=float(np.mean(epos<=.05*pos_scale)),
       pos_hit10pct=float(np.mean(epos<=.10*pos_scale)),
       pos_hit20pct=float(np.mean(epos<=.20*pos_scale)),
       direction_hit30deg=float(np.mean(angle[valid]<=np.deg2rad(30))),
       direction_hit45deg=float(np.mean(angle[valid]<=np.deg2rad(45))),
       dir8_accuracy=float(np.mean((truedir[valid]==preddir[valid])&both[valid])),
       median_angle_deg=float(np.rad2deg(np.median(angle[valid]))),
       exact_true_index_recall1=float(np.mean(match==np.arange(n))))


def cross_trial_nearest(x,y,trial):
    tree=cKDTree(y)
    n=len(x)
    dst,neighbors=tree.query(x,k=80,workers=1)
    if neighbors.ndim==1:neighbors=neighbors[:,None]
    match=np.empty(n,int)
    for idx in range(n):
        q=neighbors[idx]
        ok=q[trial[q]!=trial[idx]]
        if len(ok):match[idx]=ok[0]
        else:
            dd=np.sum((y-x[idx])**2,axis=1)
            dd[trial==trial[idx]]=np.inf
            match[idx]=int(np.argmin(dd))
    assert not np.any(trial[match]==trial)
    return match


def visualize_semantics(sem,out):
    fig,axs=plt.subplots(2,2,figsize=(13.5,10.5),constrained_layout=True)
    for i,domain in enumerate(["source_hard","target_hard"]):
        cm=np.array(sem[domain]["train_direction8_counts"])
        ax=axs[0,i]
        a=cm/cm.sum(1,keepdims=True)
        im=ax.imshow(a,cmap="Blues",vmin=0,vmax=max(.4,a.max()),aspect="auto")
        ax.set(xticks=list(range(8)),xticklabels=[f"D{u}" for u in range(8)],
               yticks=list(range(4)),yticklabels=[f"group{k}" for k in range(4)],
               xlabel="True instantaneous speed direction (45-degree sectors)",
               ylabel="Independently found cluster",title=f"{domain}: train direction purity")
        fig.colorbar(im,ax=ax,fraction=.045)
    for col,variable in enumerate(["direction8","position_quadrant"]):
        ax=axs[1,col]
        vals=[sem[g]["train"][variable]["observed"] for g in ["source_hard","target_hard","source_soft","target_soft"]]
        nul=[sem[g]["train"][variable]["trial_shuffle_null_95percent"] for g in ["source_hard","target_hard","source_soft","target_soft"]]
        q=np.arange(4)
        ax.bar(q-.15,vals,width=.3,label="Observed MI/H(label)")
        ax.bar(q+.15,nul,width=.3,label="Trial-permutation null 95%")
        ax.set(xticks=q,xticklabels=["Source H","Target H","Source S","Target S"],
               ylabel="Fraction of physical-label entropy explained",
               title=f"Train-only physical association: {variable}")
        ax.legend(fontsize=8)
        ax.grid(axis="y",alpha=.1)
    fig.suptitle("PRECONDITION: do independently learned neural/hand groups carry behavioral structure?\n"
                 "Labels and synchronized behavior used POSTFIT for auditing only",y=1.02)
    fig.savefig(out/"PREFLIGHT_CLUSTER_SEMANTICS.png",dpi=175,bbox_inches="tight")
    plt.close(fig)


def visualize_retrieval(rec,out):
    fig,axes=plt.subplots(2,2,figsize=(13,8.8),constrained_layout=True)
    config=[("full_pool","pos_hit10pct","Position within 10% TRAIN median path length"),
            ("cross_trial_pool","pos_hit10pct","Cross-trial candidate: position hit"),
            ("full_pool","direction_hit30deg","Speed direction within 30 degrees"),
            ("cross_trial_pool","direction_hit30deg","Cross-trial candidate: direction hit")]
    methods=["random","identity","hard","soft"]
    for ax,(pool,key,title) in zip(axes.flat,config):
        vals=[];sd=[]
        for m in methods:
            if m in ("random","identity"):
                v=[rec["baseline"][pool][m][key]]
            else:
                v=[r[pool][key] for r in rec["methods"][m]]
            vals.append(float(np.mean(v)));sd.append(float(np.std(v,ddof=1)) if len(v)>1 else 0)
        for j,v in enumerate(vals):
            ax.bar(j,v,color=["#aaaaaa","#454545","#377db7","#d48a3d"][j])
            if sd[j]>0:ax.errorbar(j,v,yerr=sd[j],fmt="none",ecolor="black",capsize=3)
        ax.set(xticks=np.arange(4),xticklabels=methods,ylabel="Hit rate",ylim=(0,max(vals)*1.3+.02),title=title)
        ax.grid(axis="y",alpha=.13)
    fig.suptitle("Frozen five-seed Hard/Soft R only: PHYSICAL movement associated with retrieved target, not embedding distance\n"
                 "The heldout 100 trials were previously examined; this is exploratory, not independent model selection",
                 y=1.025,fontsize=11.5)
    fig.savefig(out/"PREFLIGHT_POSITION_SPEED_RETRIEVAL.png",dpi=190,bbox_inches="tight")
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--clusters",type=Path,required=True)
    ap.add_argument("--archive",type=Path,required=True)
    ap.add_argument("--clustering-report",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    record=json.loads(a.archive.read_text(encoding="utf8"))
    audit=json.loads(a.clustering_report.read_text(encoding="utf8"))
    assert record["input_sha256"]==hashlib.sha256(a.clusters.read_bytes()).hexdigest()
    assert record["no_synchronized_pair_ids_used_in_fitting"]
    with np.load(a.clusters,allow_pickle=False) as d:
        X=np.asarray(d["neural4"],float)
        Y=np.asarray(d["hand4"],float)
        train=np.asarray(d["train_mask"],bool)
        test=np.asarray(d["test_mask"],bool)
        src_h=np.asarray(d["neural_groups"],int)
        dst_h=np.asarray(d["hand_groups"],int)
        trial=np.asarray(d["trial_of_window"],int)
        assert "shared_labels" not in d.files
    assert X.shape==Y.shape==(17500,4) and train.sum()==14000 and test.sum()==3500
    assert len(np.unique(trial[train]))==400 and len(np.unique(trial[test]))==100
    assert np.all(np.bincount(trial)==35) and len(np.unique(trial))==500
    assert not np.array_equal(src_h,dst_h)
    x=(X-np.asarray(record["source_mean"]))/record["source_radius"]
    y=(Y-np.asarray(record["target_mean"]))/record["target_radius"]
    physical=Y*np.array(audit["scalers"]["behavior_scale"])+np.array(audit["scalers"]["behavior_mean"])
    assert np.all(np.isfinite(physical))
    idxtrain=np.flatnonzero(train)
    idxtest=np.flatnonzero(test)
    assert np.array_equal(trial,np.repeat(np.arange(500),35))
    phase=np.tile(np.select([np.arange(35)<5,np.arange(35)<20],[0,1],default=2),500)
    trpos=physical[train,:2].reshape(400,35,2)
    path_length=np.sum(np.linalg.norm(np.diff(trpos,axis=1),axis=2),axis=1)
    path_ref=float(np.median(path_length))
    assert path_ref>0 and np.isfinite(path_ref)
    low=float(np.quantile(np.linalg.norm(physical[train,2:],axis=1),.20))
    behavior,rule=categories_from_raw(physical,phase,train,low)
    CX=np.asarray(record["soft_prototypes"]["source_centers"],float)
    CY=np.asarray(record["soft_prototypes"]["target_centers"],float)
    src_soft=softmax(x@CX.T,axis=1).argmax(1)
    dst_soft=softmax(y@CY.T,axis=1).argmax(1)
    named={"source_hard":src_h,"target_hard":dst_h,
           "source_soft":src_soft,"target_soft":dst_soft}
    semantic={}
    # all physical labels are evaluation-only; group assignments are frozen.
    for name,labels in named.items():
        trinfo={};teinfo={}
        for key in ["direction8","position_quadrant","speed_tertile","phase"]:
            vals=behavior[key]
            valid=behavior["valid_direction"] if key=="direction8" else np.ones(len(X),bool)
            trinfo[key]=trial_shuffle_null(labels,vals,valid,idxtrain,trial,400)
            take=test&valid
            teinfo[key]={"test_mi_fraction":metric_mi(labels[take],vals[take])}
        if name=="source_hard" or name=="target_hard":
            cm=cnt_table(labels[train],behavior["direction8"][train],
                         behavior["valid_direction"][train],8)
        else:cm=None
        semantic[name]=dict(train=trinfo,test=teinfo,
                   train_direction8_counts=cm.tolist() if cm is not None else None,
                   train_group_counts=np.bincount(labels[train],minlength=4).tolist())
        print("SEMANTICS",name,"dir",trinfo["direction8"]["observed"],
               "block-null95",trinfo["direction8"]["trial_shuffle_null_95percent"],
               "test",teinfo["direction8"]["test_mi_fraction"],flush=True)
    geometry=cluster_geometry(x,y,src_h,dst_h,train)

    yy=y[test]
    xt=x[test]
    tt=trial[test]
    pt=physical[test]
    # random baseline from identical target candidate pool. Fixed once, same
    # all methods, with a cross-trial version to prevent within-trial lookup.
    g=np.random.default_rng(1945)
    rand=g.integers(len(tt),size=len(tt))
    rand_cross=rand.copy()
    same=(tt[rand_cross]==tt)
    rand_cross[same]=(rand_cross[same]+35)%len(tt)
    assert np.all(tt[rand_cross]!=tt)
    from scipy.spatial import cKDTree
    tree=cKDTree(yy)
    ident=tree.query(xt,k=1)[1]
    ident_cross=cross_trial_nearest(xt,yy,tt)
    baseline={"full_pool":{
               "random":retrieval_metrics(rand,pt,low,path_ref),
               "identity":retrieval_metrics(ident,pt,low,path_ref)},
              "cross_trial_pool":{
               "random":retrieval_metrics(rand_cross,pt,low,path_ref),
               "identity":retrieval_metrics(ident_cross,pt,low,path_ref)}}
    methods={"hard":[],"soft":[]}
    for method in methods:
        for run in record["runs"][method]:
            seed=run["seed"]
            R=np.asarray(run["R"],float)
            assert np.linalg.norm(R.T@R-np.eye(4))<1e-7
            source=xt@R.T
            best=tree.query(source,k=1)[1]
            cross=cross_trial_nearest(source,yy,tt)
            item=dict(seed=seed,strict_admm_pass=bool(run["admm_criteria_passed"]),
                full_pool=retrieval_metrics(best,pt,low,path_ref),
                cross_trial_pool=retrieval_metrics(cross,pt,low,path_ref))
            methods[method].append(item)
            print("RETRIEVAL",method,seed,
               "full-pos10",item["full_pool"]["pos_hit10pct"],
               "cross-pos10",item["cross_trial_pool"]["pos_hit10pct"],
               "cross-dir30",item["cross_trial_pool"]["direction_hit30deg"],flush=True)
    # Predeclared feasibility flags are diagnostics, never claims of statistical
    # significance. Existing test data has been inspected by other studies.
    source_mi=semantic["source_hard"]["train"]["direction8"]
    sem_prereq=bool(source_mi["above_null_95"] and source_mi["observed"]>=.05)
    checks=[]
    for meth in methods:
        for r in methods[meth]:
            pool="cross_trial_pool"
            val=r[pool]
            base=baseline[pool]
            posgain=(val["pos_hit10pct"]-base["identity"]["pos_hit10pct"])
            dirgain=(val["direction_hit30deg"]-base["identity"]["direction_hit30deg"])
            matches=(posgain>=.05 and val["pos_hit10pct"]-base["random"]["pos_hit10pct"]>=.05) or (
                     dirgain>=.05 and val["direction_hit30deg"]-base["random"]["direction_hit30deg"]>=.05)
            checks.append(dict(method=meth,seed=r["seed"],passes=matches,
                               position_gain_vs_identity=posgain,direction_gain_vs_identity=dirgain))
    by_meth={m:sum(q["passes"] for q in checks if q["method"]==m) for m in methods}
    quality={m:sum(q["strict_admm_pass"] for q in methods[m]) for m in methods}
    if sem_prereq and max(by_meth.values())>=3:
        verdict="CONDITIONAL_GO_FOR_REPRESENTATION_VALIDATION"
    elif not sem_prereq and max(by_meth.values())==0:
        verdict="RED_FOR_CURRENT_PCA4_GROUPING_AND_1NN; PRIORITIZE_FEATURE_REDESIGN"
    else:
        verdict="AMBER_MIXED_PRECONDITIONS; DO_NOT_LAUNCH_FULL_SOFT_ABLATION_YET"
    outcome=dict(run_type="read_only_preflight_existing_five_seed_models",
       model_retraining_occurred=False,test_labels_used_in_fitting_or_matching=False,
       test_previously_inspected=True,
       dataset_input_sha256=record["input_sha256"],
       data_layers=dict(all=17500,train=14000,heldout=3500,OT=160,train_trials=400,test_trials=100),
       feature_description="Train z-score 162 neural channels sqrt-smoothed PCA4, target zscore hand_pos_xy+hand_vel_xy",
       physical_scoring_note="Hand4 transformed back using archived StandardScaler; original NWB coordinate unit unverified",
       preregistered_scoring_rules=dict(
           train_median_700ms_path_length_native_units=path_ref,
           position_hit_thresholds_fraction_of_path=[.05,.10,.20],
           speed_minimum_train_20percent_quantile_native_units=low,
           angle_tolerances_degrees=[30,45],
           true_slow_velocity_samples_excluded_from_angle=True,
           retrieved_low_velocity_counts_as_angle_miss=True,
           movement_phase_time_offsets_ms="center -90 to -10 before, 10 to 290 early, 310 to 590 late"),
       labels_rule=rule,
       semantic=semantic,independent_cluster_centroid_geometry=geometry,
       retrieval=dict(baseline=baseline,methods=methods),
       feasibility_gate=dict(
          rule="source Hard direction MI/H>0.05 and above 95pct trial-block permutation null; plus >=3/5 frozen model seeds outperform identity by >=5pp and random by >=5pp on cross-trial pos10 OR dir30",
          semantic_gate=sem_prereq,matching_seed_pass_counts=by_meth,
          admm_pass_counts=quality,
          per_seed_matches=checks,
          recommendation=verdict,
          warning="Heuristic exploratory gate; NOT a statistical guarantee or permission to test-tune FA4/Isomap on the already examined heldout 100 trials."),
       caveat="K=4 cluster association might reflect movement epoch or trial autocorrelation. Trial-block null preserves temporal structure. Same-trial nearest candidates can inflate apparent success; cross-trial matching is also mandatory.")
    (a.output/"PREFLIGHT_DATA_AND_EVALUATION_AUDIT.json").write_text(
        json.dumps(outcome,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    visualize_semantics(semantic,a.output)
    visualize_retrieval(outcome["retrieval"],a.output)
    # Main report includes complete physical evidence and no unsupported
    # claims of full machine learning cross-session heldout generalization.
    lines=["# DANDI000138 | Phase-I preflight: does our EXISTING 4D data have recoverable physical cluster geometry?",
      "",
      "**This is a completed read-only preflight on the existing five-seed strict unpaired model. No R/P/Q/refitting changed; sample levels remain 17500/14000/3500/160.**",
      "",
      "## Predeclared evaluation scales (from TRAIN hand trajectories only)",
      "",
      f"- Median total path length per 35-bin training trial: {path_ref:.6g} native hand position units; position tolerances at 5%/10%/20% of this reference. NWB position unit metadata has **not** been verified.",
      f"- Speed threshold: 20th train speed percentile {low:.6g} native velocity units; low true velocity excluded from angular accuracy; matched slow velocity counted as an angular failure.",
      "- Actual train directions: instantaneous velocity angle quantized into 8 sectors; this is **NOT** a reach target condition label.",
      "- Physical labels/synchronized true samples used **only AFTER original models were trained**, for diagnostic and evaluation; never used to choose matching points or model parameters.",
      "- Candidate matches: all 3500 test target points, and separate cross-trial-only candidate pool; compare Identity, uniform random target, HardR, SoftR. True match ID not supplied to search.",
      "",
      "## Train-only cluster semantics (fraction of motion-direction entropy explained)",
      "",
      "| Independently learned groups | Train direction MI/H | Trial-block shuffled 95% null | Test MI/H (exploratory) | Train motion position MI/H |",
      "|---|---:|---:|---:|---:|"]
    for g2,sem in semantic.items():
        di=sem["train"]["direction8"]
        lines.append(f"| {g2} | {di['observed']:.5f} | {di['trial_shuffle_null_95percent']:.5f} | "
             f"{sem['test']['direction8']['test_mi_fraction']:.5f} | "
             f"{sem['train']['position_quadrant']['observed']:.5f} |")
    lines.extend(["","![Unsupervised group semantics](PREFLIGHT_CLUSTER_SEMANTICS.png)","",
      "## Original physical position and instantaneous movement angle retrieval",
      "",
      "Do not confuse low feature nearest-neighbor distance with physical match correctness. Test true position/velocity is read only once nearest target indices are fixed.",
      "",
      "| Matching method | Pool | Position hit ≤10% train path | Speed angle ≤30° | Speed 8-sector accuracy | Position median error (native units) |",
      "|---|---|---:|---:|---:|---:|"])
    for pool in ["full_pool","cross_trial_pool"]:
        for m in ["random","identity","hard","soft"]:
            rs=[baseline[pool][m]] if m in ("random","identity") else [z[pool] for z in methods[m]]
            avg=lambda k:np.mean([t[k] for t in rs])
            lines.append(f"| {m} | {pool} | {avg('pos_hit10pct'):.4f} | {avg('direction_hit30deg'):.4f} | "
              f"{avg('dir8_accuracy'):.4f} | {avg('position_median'):.5g} |")
    lines+=["","![Physical position/speed hit rates](PREFLIGHT_POSITION_SPEED_RETRIEVAL.png)","",
        "## Independent unpaired centroid structure",
        "",
        f"- Best group-centroid distance discrepancy over 24 independent group permutations: {geometry['best_relative_distance_matrix_discrepancy']:.5f}. This alone cannot prove paired physical correspondence.",
        "",
        "## PREDECLARED GO / NO-GO exploratory heuristic",
        "",
        f"- Train neural hard-group direction semantic gate: **{sem_prereq}**.",
        f"- Cross-trial physical 1NN matching seed gate passes: hard **{by_meth['hard']}/5**, soft **{by_meth['soft']}/5**.",
        f"- Strict solver residual convergence: hard **{quality['hard']}/5**, soft **{quality['soft']}/5**.",
        f"- Result: **{verdict}**.",
        "",
        "This gate addresses current PCA4+KMeans with fixed R+1NN only; it DOES NOT establish failure/success of other latent representations or full Gamma-based hierarchical transport.",
        "",
        "## Critical limitations",
        "",
        "1. This experiment reuses heldout 100 trials previously inspected in research, so it is an **exploratory audit**. Future PCA4/FA4/Isomap decisions MUST use train-internal validation rather than test score tuning; a fresh independent confirmation set is desirable.",
        "2. Trial-block permutations preserve within-trial temporal structure and approximate epoch effects, but 500 trials are from one recording/session and are not independent subjects.",
        "3. Present target retrieval is **transductive** and uses R-only 1NN. It does not yet evaluate target-aware recalculated Q/Gamma, which must be conducted in a separate predeclared OT-matching phase.",
        "4. The hand physics is scored in its archived raw numeric coordinate system. NWB unit metadata not checked, so tolerance is tied to a preregistered training path-length reference rather than invented centimeters.",
        "",
        "[Detailed frozen-model/semantic/seed audit JSON](PREFLIGHT_DATA_AND_EVALUATION_AUDIT.json) | [Original strict unpaired training](../../code/run_strict_unsupervised_hard_soft.py) | [This test-only diagnostic code](../../code/preflight_geometry_semantics_retrieval.py) | [Next-stage full protocol](../../NEXT_STAGE_EXPERIMENT_PROTOCOL.md)",
        ""]
    (a.output/"PREFLIGHT_REPORT.md").write_text("\n".join(lines),encoding="utf8")
    print("PREFLIGHT_DONE",verdict,flush=True)

if __name__=="__main__":
    main()
