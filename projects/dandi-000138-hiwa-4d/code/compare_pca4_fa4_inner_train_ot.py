"""Real DANDI: rigorous PCA4 vs FA4 comparison from ORIGINAL 162 neural channels.

Uses original DANDI:000138 500 trials x 35 windows, raw spikes re-binned
and sqrt-smoothed identically to run_hard_clustering.py. Original 100 test
trials NEVER used. Inner 400 train trials --> 320 model-train + 80 validation.
Both neural methods fit only 320 train trial neural features; same target
4D hand location/velocity fit on 320 train trials. No physical labels or
synchronized indices used by PCA/FA/clustering/HiWA fit. No sample matching
given to model; GT hand only post-fit.
Core hierarchical HiWA-GCOT P,Q,R is UNMODIFIED (same old default random init).
Each method uses independent Hard KMeans or independent soft prototypes,
same 160 indices per domain for Hard+Soft for a given method and seed.
We additionally report MAE/physical 1NN/position direction/mutual-info.
All training budgets and OT marginal/ADMM diagnostics recorded, no summaries
based on physical validation selection or post-hoc favorable seed.
"""
from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.special import xlogy, softmax
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from zip_gcot.alignment import SolverConfig,hierarchical_alignment
from zip_gcot.grouping import soft_groups
from run_hard_hiwa import sample_groups
from run_controlled_soft_hard_ablation import build_measures,heldout_nested
from run_hard_clustering import load_trial_windows,preprocess
from preflight_geometry_semantics_retrieval import cross_trial_nearest,retrieval_metrics,metric_mi

def _to_native(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,(tuple,list)):return [_to_native(x) for x in v]
    if isinstance(v,dict):return {str(k):_to_native(x) for k,x in v.items()}
    return v

def score_global_w2(x,y,R,ix,iy):
    d=cdist(x[ix]@R.T,y[iy],"sqeuclidean")
    ia,ib=linear_sum_assignment(d)
    return float(d[ia,ib].mean())

def trial_permuted_direction_mi(labels,directions,valid,trial_ids,index,n_perm=99,seed=209):
    rng=np.random.default_rng(seed)
    ids=np.unique(trial_ids[index])
    assert len(ids)*35==len(index)
    order=np.argsort(trial_ids[index],kind="stable")
    ii=index[order]
    labs=labels[ii].reshape(-1,35)
    di=directions[ii].reshape(-1,35)
    ve=valid[ii].reshape(-1,35)
    got=metric_mi(labs[ve],di[ve],4)
    out=np.empty(n_perm)
    for k in range(n_perm):
        p=rng.permutation(len(ids))
        mask=ve&ve[p]
        out[k]=metric_mi(labs[mask],di[p][mask],4)
    return dict(observed=got,null95=float(np.quantile(out,.95)),
          null_mean=float(out.mean()),over_null95=bool(got>np.quantile(out,.95)),
          block_permutations=n_perm)

def fit_factors(neural_std,train,method,seed,fa_max_iter):
    if method=="PCA4":
        fitter=PCA(n_components=4,svd_solver="full",random_state=seed)
    else:
        fitter=FactorAnalysis(n_components=4,svd_method="randomized",
                      random_state=seed,max_iter=fa_max_iter,tol=1e-3,
                      iterated_power=4)
    st=time.perf_counter()
    fitter.fit(neural_std[train])
    xt=fitter.transform(neural_std)
    elapsed=float(time.perf_counter()-st)
    if method=="PCA4":
        explained=float(np.sum(fitter.explained_variance_ratio_))
        converged=True
    else:
        explained=None
        converged=bool(fitter.n_iter_<fa_max_iter)
    return xt,fitter,dict(
        fit_duration_sec=elapsed,
        train_fitting_iteration_count=int(getattr(fitter,"n_iter_",1)),
        reached_convergence_before_budget=converged,
        fraction_total_standardized_neural_variance_explained_by_four_PCs=explained,
        method=method)

def raw_behavior_metrics(yphysical,train,test,trial):
    # fixed position threshold based on train trial trajectory length ONLY.
    trids=np.unique(trial[train])
    path=[]
    for k in trids:
        pos=yphysical[(trial==k)&train,:2]
        assert len(pos)==35
        path.append(np.linalg.norm(np.diff(pos,axis=0),axis=1).sum())
    median_path=float(np.median(path))
    speed=np.linalg.norm(yphysical[:,2:],axis=1)
    low=float(np.quantile(speed[train],.2))
    theta=(np.arctan2(yphysical[:,3],yphysical[:,2])+2*np.pi)%(2*np.pi)
    d8=np.floor(8*theta/(2*np.pi)).astype(int)%8
    return median_path,low,d8,speed>=low

def kmeans_eval(x,training,valid,seed,labels_direction,dir_valid,trial):
    km=KMeans(n_clusters=4,n_init=20,random_state=seed)
    km.fit(x[training])
    all_labels=km.predict(x)
    tridx=np.flatnonzero(training)
    rng=np.random.default_rng(308)
    sub=rng.choice(tridx,min(2800,len(tridx)),replace=False)
    sil=float(silhouette_score(x[sub],all_labels[sub]))
    mi_tr=trial_permuted_direction_mi(all_labels,labels_direction,dir_valid,
                 trial,tridx,n_perm=99)
    valmask=valid&dir_valid
    valmi=float(metric_mi(all_labels[valmask],labels_direction[valmask],4))
    return km,all_labels,dict(
        train_silhouette=sil,train_cluster_counts=np.bincount(all_labels[training],minlength=4).tolist(),
        train_direction_association=mi_tr,validation_direction_association_postfit=valmi)

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--nwb",type=Path,required=True)
    a.add_argument("--original-clusters",type=Path,required=True)
    a.add_argument("--clustering-report",type=Path,required=True)
    a.add_argument("--output",type=Path,required=True)
    a.add_argument("--seeds",type=int,nargs="+",default=[42,43,44])
    a.add_argument("--outer",type=int,default=500)
    a.add_argument("--inner",type=int,default=20)
    a.add_argument("--fa-max-iter",type=int,default=120)
    args=a.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    meta_original=json.loads(args.clustering_report.read_text())
    nr,br,valid_trials,metadata=load_trial_windows(args.nwb)
    assert metadata["sha256"]==meta_original["sha256"],"NWB bytes differ from original project"
    assert nr.shape==(500,35,162) and br.shape==(500,35,4)
    with np.load(args.original_clusters,allow_pickle=False) as z:
        old_mask=np.asarray(z["train_mask"],bool)
        original_test=np.asarray(z["test_mask"],bool)
        trial=np.asarray(z["trial_of_window"],int)
        old_y=np.asarray(z["hand4"],float)
        orig_source=np.asarray(z["neural4"],float)
        oldtrials=np.asarray(z["train_trials"],int)
        oldtesttrials=np.asarray(z["test_trials"],int)
        assert "shared_labels" not in z.files
    assert old_mask.sum()==14000 and original_test.sum()==3500
    assert len(oldtrials)==400 and len(oldtesttrials)==100
    assert np.array_equal(trial,np.repeat(np.arange(500),35))
    # Reproduce original preprocessing for independent input hash and 4D cache.
    x_rec,y_rec,keep,tids,tr,te,pca0,scaler0=preprocess(
       nr,br,seed=42,smooth_ms=40.,bin_ms=20.)
    assert np.array_equal(trial,tids)
    assert np.array_equal(old_mask,old_mask&keep)
    assert np.array_equal(original_test,original_test&keep)
    assert np.allclose(y_rec,old_y,atol=1e-5)
    # PCA eigenvector signs may change between numerical backends: assess
    # subspace via pairwise Gram and explained variance, not exact sign.
    assert np.allclose(pca0.explained_variance_ratio_,
        meta_original["neural_pca_explained_variance_ratio"],atol=5e-4)
    original_pca_grameig=np.linalg.eigvalsh(np.cov(orig_source[old_mask],rowvar=False))
    rec_pca_grameig=np.linalg.eigvalsh(np.cov(x_rec[old_mask],rowvar=False))
    assert np.allclose(original_pca_grameig,rec_pca_grameig,rtol=.002)
    # old test is NEVER used for fitting, model choice, physical matching,
    # feature diagnostics or figures in this script.
    inner_train_trials,inner_val_trials=train_test_split(
        oldtrials,test_size=.2,random_state=20261119)
    trmask=old_mask&np.isin(trial,inner_train_trials)
    vamask=old_mask&np.isin(trial,inner_val_trials)
    assert trmask.sum()==11200 and vamask.sum()==2800 and not np.any(trmask&vamask)
    assert not np.any(trmask&original_test) and not np.any(vamask&original_test)
    neural=gaussian_filter1d(np.sqrt(np.maximum(nr,0)),
                             sigma=2.,axis=1,mode="nearest").reshape(-1,162)
    yphysical=br.reshape(-1,4).astype(float)
    neural_scaler=StandardScaler().fit(neural[trmask])
    neural_standard=neural_scaler.transform(neural)
    target_scaler=StandardScaler().fit(yphysical[trmask])
    yraw=target_scaler.transform(yphysical)
    my=yraw[trmask].mean(0)
    sy=float(np.sqrt(np.mean(np.sum((yraw[trmask]-my)**2,axis=1))))
    y=(yraw-my)/sy
    center_target,lab_y,target_info=kmeans_eval(y,trmask,vamask,42,
        *raw_behavior_metrics(yphysical,trmask,vamask,trial)[2:],trial)
    # physical movement label is used above in POSTFIT MI, not target fitting.
    median_path,low,dir8,valid_direction=raw_behavior_metrics(
                         yphysical,trmask,vamask,trial)
    # Save split and original physical scales as audit, not a new test fit.
    result=dict(dataset="DANDI:000138 original NWB",file_sha256=metadata["sha256"],
       original_data_hierarchy=dict(full_trials=500,full_windows=17500,
                 original_train_trials=400,original_train_windows=14000,
                 untouched_original_test_trials=100,untouched_original_test_windows=3500,
                 internal_fit_trials=320,internal_fit_windows=11200,
                 internal_validation_trials=80,internal_validation_windows=2800,
                 OT_sample_windows_per_domain=160),
       physical_ground_truth_in_training=False,
       true_cross_domain_sample_pairing_in_alignment=False,
       target_group_labels_computed_from_target_only=True,
       source_group_labels_computed_from_source_only=True,
       no_original_test_evaluated_or_optimized=True,
       split_seed=20261119,
       inner_train_trials=sorted(map(int,inner_train_trials)),
       inner_validation_trials=sorted(map(int,inner_val_trials)),
       original_test_trial_ids_hash=hashlib.sha256(np.sort(oldtesttrials).tobytes()).hexdigest(),
       original_pca_reproduction=dict(
         mean_original_four_pc_train_variance_ratio=meta_original["neural_pca_explained_variance_ratio"],
         old_pca_eigenvalues=original_pca_grameig.tolist(),
         reproduced_pca_eigenvalues=rec_pca_grameig.tolist(),
         old_behavior_4d_cache_max_abs=float(np.max(np.abs(y_rec-old_y)))),
       physical_scoring_rules=dict(
         position_tolerance_fraction_of_inner_train_path=.10,
         train_median_path_length_native_units=median_path,
         physical_angle_threshold_degrees=30,
         speed_minimum_train_20pct=low,
         true_low_speed_samples_excluded_from_angle=True),
       original_HiWA_PQR_config=dict(outer=args.outer,inner=args.inner,
           group_count=4,group_subsample=40,epsilon_sample=.1,
           epsilon_group=.2,mu=.05,sample_sinkhorn_iters=45,
           group_sinkhorn_iters=200,rotation_tolerance=1e-9,
           numeric_sinkhorn_tol=1e-8),
       target_independent_cluster_info=target_info,
       neural_pca_fa={},methods={},convergence_audit={})
    rng=np.random.default_rng(3026)
    valrows=np.flatnonzero(vamask)
    split_train=np.flatnonzero(trmask)
    selx=rng.choice(len(valrows),size=200,replace=False)
    sely=rng.choice(len(valrows),size=200,replace=False)
    smallx=rng.choice(len(valrows),size=128,replace=False)
    smally=rng.choice(len(valrows),size=128,replace=False)
    xt_target=y[vamask]
    yttrial=trial[vamask]
    actual_val=yphysical[vamask]
    # Fixed baselines for the SAME validation candidate and phys thresholds.
    rng0=np.random.default_rng(10817)
    rand=rng0.integers(len(valrows),size=len(valrows))
    same=yttrial[rand]==yttrial
    rand[same]=(rand[same]+35)%len(valrows)
    assert np.all(yttrial[rand]!=yttrial)
    random_real=retrieval_metrics(rand,actual_val,low,median_path)
    result["random_cross_trial_baseline"]=random_real
    for name in ("PCA4","FA4"):
        xembed,trained,feature_meta=fit_factors(
            neural_standard,trmask,name,seed=42,fa_max_iter=args.fa_max_iter)
        mx=xembed[trmask].mean(0)
        sx=float(np.sqrt(np.mean(np.sum((xembed[trmask]-mx)**2,axis=1))))
        x=(xembed-mx)/sx
        assert x.shape==(17500,4) and np.isfinite(x).all()
        reconstruction=trained.inverse_transform(xembed[vamask]) if hasattr(trained,'inverse_transform') \
            else xembed[vamask]@trained.components_+trained.mean_
        feature_meta.update(
          four_factor_validation_reconstruction_MSE_standardized_neuron_space=
            float(np.mean((neural_standard[vamask]-reconstruction)**2)),
          latent_train_component_variances=np.var(x[trmask],axis=0).tolist(),
          train_group_rms=sx)
        source_k,lab_x,source_info=kmeans_eval(x,trmask,vamask,42,
                                          dir8,valid_direction,trial)
        feature_meta["source_cluster"]=source_info
        result["neural_pca_fa"][name]=feature_meta
        # Independent per-modality soft prototypes fitted only inner training.
        A,prototype_x,source_sof=soft_groups(x[trmask],4,
                  seed=42,entropy_weight=1.,maxiter=300)
        B,prototype_y,target_sof=soft_groups(y[trmask],4,
                  seed=43,entropy_weight=1.,maxiter=300)
        ax=softmax(x@prototype_x.T,axis=1)
        by=softmax(y@prototype_y.T,axis=1)
        assert np.allclose(ax[trmask],A,atol=1e-8)
        assert np.allclose(by[trmask],B,atol=1e-8)
        feature_meta["soft_source_mean_entropy_nats"]=float(
           -np.mean(np.sum(xlogy(ax[trmask],ax[trmask]),axis=1)))
        feature_meta["soft_target_mean_entropy_nats"]=float(
           -np.mean(np.sum(xlogy(by[trmask],by[trmask]),axis=1)))
        feature_meta["target_soft_prototypes_train_only"]=True
        xt=x[vamask];yt=y[vamask]
        tree=cKDTree(yt)
        ident=tree.query(xt,k=1)[1]
        ident_cross=cross_trial_nearest(xt,yt,yttrial)
        result["methods"][name]={}
        result["methods"][name]["identity_control"]=dict(
          cross_trial=retrieval_metrics(ident_cross,actual_val,low,median_path),
          full_pool=retrieval_metrics(ident,actual_val,low,median_path),
          heldout_unpaired_W2sq=score_global_w2(xt,yt,np.eye(4),selx,sely))
        fitcfg=SolverConfig(maxiter=args.outer,tol=1e-9,mu=.05,
                    epsilon_group=.2,epsilon_sample=.1,
                    inner_maxiter=args.inner,inner_tol=.01,
                    sinkhorn_maxiter=45,group_sinkhorn_maxiter=200,
                    sinkhorn_tol=1e-8,stopping="legacy")
        for grouping in ("Hard","Soft"):
            results=[]
            for seed in args.seeds:
                rg=np.random.default_rng(seed)
                tx=sample_groups(split_train,lab_x,40,rg)
                ty=sample_groups(split_train,lab_y,40,rg)
                if grouping=="Hard":
                    aa=np.eye(4)[lab_x[tx]]
                    bb=np.eye(4)[lab_y[ty]]
                    va=np.eye(4)[lab_x[vamask][smallx]]
                    vb=np.eye(4)[lab_y[vamask][smally]]
                else:
                    aa=ax[tx];bb=by[ty]
                    va=ax[vamask][smallx];vb=by[vamask][smally]
                ishard=grouping=="Hard"
                tic=time.perf_counter()
                fit=hierarchical_alignment(
                    build_measures(x[tx],aa,ishard),build_measures(y[ty],bb,ishard),
                    seed=seed,config=fitcfg,scale=True,retain_plans=True)
                run_sec=float(time.perf_counter()-tic)
                R=fit["R"];P=fit["P"]
                assert np.linalg.norm(R.T@R-np.eye(4))<1e-7
                assert max(abs(P.sum(1)-.25).max(),abs(P.sum(0)-.25).max())<1e-5
                rtrain=tree.query(xt@R.T,k=1)[1]
                r_cross=cross_trial_nearest(xt@R.T,yt,yttrial)
                phys_full=retrieval_metrics(rtrain,actual_val,low,median_path)
                phys_cross=retrieval_metrics(r_cross,actual_val,low,median_path)
                w2=score_global_w2(xt,yt,R,selx,sely)
                nested=heldout_nested(xt[smallx],yt[smally],va,vb,P,R,.1,120,ishard)
                train_q_max=max(max(float(np.max(abs(fit[f"Q_{i}_{j}"].sum(0)-build_measures(y[ty],bb,ishard)[j][1]))),
                                    float(np.max(abs(fit[f"Q_{i}_{j}"].sum(1)-build_measures(x[tx],aa,ishard)[i][1]))))
                                for i in range(4) for j in range(4))
                entry=dict(seed=int(seed),feature=name,grouping=grouping,
                     original_real_trial_train_source_indices_sha256=hashlib.sha256(tx.tobytes()).hexdigest(),
                     original_real_trial_train_target_indices_sha256=hashlib.sha256(ty.tobytes()).hexdigest(),
                     validation_cross_trial_physical_retrieval=phys_cross,
                     validation_full_pool_physical_retrieval=phys_full,
                     validation_free_W2sq=w2,
                     validation_nested_PQ_transport_W2sq=nested["total_transport_W2sq"],
                     validation_nested_PQ_max_marginal_error=nested["max_Q_marginal_error"],
                     strict_admm_passed=bool(fit["criteria_satisfied"]),
                     legacy_stop_reason=fit["stop_reason"],
                     iterations=int(len(fit["history"])),
                     runtime_sec=run_sec,
                     consensus_residual=float(fit["consensus_residual"]),
                     dual_residual=float(fit["dual_residual"]),
                     rotation_delta=float(fit["rotation_delta"]),
                     P_delta=float(fit["coupling_delta"]),
                     train_Q_marginal_error=float(train_q_max),
                     P_entropy=float(-np.sum(xlogy(P,P))),
                     P=P.tolist(),R=R.tolist())
                assert entry["train_Q_marginal_error"]<1e-5
                assert entry["validation_nested_PQ_max_marginal_error"]<1e-5
                results.append(entry)
                print(f"FA_PCA_DANDI_REAL {name}/{grouping} seed{seed}: "
                      f"cross_pos10={phys_cross['pos_hit10pct']:.4f} "
                      f"cross_dir30={phys_cross['direction_hit30deg']:.4f} "
                      f"W2={w2:.5f} nestedPQ={entry['validation_nested_PQ_transport_W2sq']:.5f} "
                      f"ADMM={entry['strict_admm_passed']} iter={entry['iterations']} "
                      f"time={run_sec:.1f}s",flush=True)
                result["methods"][name][grouping]=results
                (out/"DANDI_PCA4_FA4_INNER_VALIDATION_RESULTS.json").write_text(
                     json.dumps(_to_native(result),indent=2,ensure_ascii=False)+"\n",encoding="utf8")
        print("FEATURE_DONE",name,feature_meta,flush=True)
    # Paired seeds for the same method; no cherry-picking.
    report=[
      "# DANDI:000138 | 真实神经PCA4 与 FA4无监督几何、完整HiWA P/Q/R 比较",
      "",
      "**本结果基于原始162通道NWB重新计算，非旧PCA4缓存上再做FA。** 500 trials /17500 windows，全域原训练400试次/14000窗口、原测试100试次/3500窗口**完全不使用**。只将原训练400试次划分为320试次/11200样本模型拟合与80试次/2800样本验证。所有特征、KMeans、Soft原型和P/Q/R都仅拟合320训练试次。",
      "",
      "目标域始终为手部位置XY+速度XY四维；每个域仅使用自身特征拟合缩放，均采用相同的原随机初始化、四组、每组40样本、500/20/45/200、mu=.05、epsilon_sample=.1、epsilon_group=.2。两域严格独立拟合硬或软组；运动真实标签与同步索引**仅用于事后评分**。",
      "",
      "## 第一问：降维/分组本身的比较",
      "",
      "| 指标（内部训练 / 内部验证） | PCA4 | FA4 |",
      "|---|---:|---:|"]
    def fmt(s):return f"{s:.4f}" if s is not None else "N/A"
    me=result["neural_pca_fa"]
    for label,field in [
       ("神经训练四个维度方差之和 / 实际总标准化方差 (仅PCA有此定义)","fraction_total_standardized_neural_variance_explained_by_four_PCs"),
       ("神经162维标准化空间验证重建 MSE","four_factor_validation_reconstruction_MSE_standardized_neuron_space"),
       ("训练KMeans silhouette","source_cluster.train_silhouette"),
       ("神经训练组-实际瞬时方向标准化互信息","source_cluster.train_direction_association.observed"),
       ("完整试次方向置换95%界限","source_cluster.train_direction_association.null95"),
       ("内部验证组-实际瞬时方向标准化互信息","source_cluster.validation_direction_association_postfit"),
       ("神经因子训练耗时秒","fit_duration_sec")]:
        def fetch(n):
            z=me[n]
            for part in field.split("."):z=z[part]
            return fmt(z)
        report.append(f"| {label} | {fetch('PCA4')} | {fetch('FA4')} |")
    report += ["",
       "FA4训练迭代数="+str(me["FA4"]["train_fitting_iteration_count"])+
       "，在预算内正常收敛="+str(me["FA4"]["reached_convergence_before_budget"])+
       "。FA的潜在因子不是PCA主成分，因此不能把PCA解释方差比例当成FA直接对等指标。",
       "",
       "## 第二问：两域完全无配对、完整P/Q/R后的性能",
       "",
       "其中位置命中阈值是**内部训练试次**手部行程中位数的10%，速度方向30度；候选匹配时禁止同trial目标邻居，防止相邻20ms窗口带来的伪成功。",
       "",
       "| 方法 | 验证自由W2²↓ | 验证嵌套P-Q运输成本↓ | 真实位置命中10%↑ | 速度方向命中30°↑ | 严格ADMM通过 |",
       "|---|---:|---:|---:|---:|---:|"]
    def avg(rows,field):return float(np.mean([r[field] for r in rows]))
    def disp(t):return f"{t:.4f}"
    for name in ("PCA4","FA4"):
        ic=result["methods"][name]["identity_control"]
        report.append(f"| {name}: Identity (no P/Q) | {disp(ic['heldout_unpaired_W2sq'])} | — | "
            f"{disp(ic['cross_trial']['pos_hit10pct'])} | {disp(ic['cross_trial']['direction_hit30deg'])} | — |")
        for group in ("Hard","Soft"):
            rows=result["methods"][name][group]
            report.append(f"| {name}: {group} ({len(rows)} seeds) | {disp(avg(rows,'validation_free_W2sq'))} | "
                 f"{disp(avg(rows,'validation_nested_PQ_transport_W2sq'))} | "
                 f"{disp(np.mean([r['validation_cross_trial_physical_retrieval']['pos_hit10pct'] for r in rows]))} | "
                 f"{disp(np.mean([r['validation_cross_trial_physical_retrieval']['direction_hit30deg'] for r in rows]))} | "
                 f"{sum(r['strict_admm_passed'] for r in rows)}/{len(rows)} |")
    report += ["",
       f"跨试次随机候选位置命中率={random_real['pos_hit10pct']:.4f}，方向30°命中率={random_real['direction_hit30deg']:.4f}；两种表示的候选池和真实物理标注相同。",
       "",
       "**判断边界：**原始100试次完全未用，但这是原400训练trial内部一次划分的探索性比较；独立trial内的相邻时间窗高度相关。不同特征所得嵌套P-Q成本虽在同一形式下计算，但两个表示改变了源域测度，不可仅按它判断原始物理对应。若Soft严格ADMM未收敛，不能将其成绩解释为软分组机制本身的成败。",
       "",
       "[逐种子完整P/R和数值审计](DANDI_PCA4_FA4_INNER_VALIDATION_RESULTS.json) | [原始重建与实际代码](../../code/compare_pca4_fa4_inner_train_ot.py)",
       ""]
    (out/"DANDI_PCA4_FA4_REPORT_ZH.md").write_text("\n".join(report)+"\n",encoding="utf8")
    fig,ax=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    groups=[("PCA4","Hard"),("FA4","Hard"),("PCA4","Soft"),("FA4","Soft")]
    style=["#387ba8","#53977b","#e0a154","#b26d80"]
    for a2,(key,title) in zip(ax,[("validation_free_W2sq","Val global free W2² ↓"),
               ("pos_hit10pct","Cross-trial physical POS hit ↑"),
               ("direction_hit30deg","Cross-trial physical DIRECTION hit ↑")]):
        vals=[];errs=[]
        for name,group in groups:
            samples=result["methods"][name][group]
            v=np.array([r[key] if key=="validation_free_W2sq" else
                 r["validation_cross_trial_physical_retrieval"][key] for r in samples],float)
            vals.append(float(v.mean()));errs.append(float(v.std(ddof=1)) if len(v)>1 else 0)
        a2.bar(range(4),vals,yerr=errs,capsize=3,color=style)
        a2.set_xticks(range(4),labels=[f"{x} {y}" for x,y in groups],rotation=18)
        a2.set_title(title)
        a2.grid(axis="y",alpha=.12)
    fig.suptitle("Original DANDI raw 162 neurons: PCA4 vs FA4 (320 trial TRAIN / 80 trial VALID only)\n"
                 "Strict unpaired Hard/Soft P/Q/R; frozen movement labels only postfit",fontsize=11.3)
    fig.savefig(out/"DANDI_PCA4_FA4_HARD_SOFT_PHYSICAL_OT.png",dpi=190,bbox_inches="tight")
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    for a2,field,title in zip(axes,[
          "source_cluster.train_silhouette","source_cluster.train_direction_association.observed"],
          ["Neural HARD Kmeans silhouette on TRAIN","Train behavior-direction mutual information / H(direction)"]):
        vals=[]
        for name in ("PCA4","FA4"):
            z=result["neural_pca_fa"][name]
            for part in field.split("."):z=z[part]
            vals.append(z)
        a2.bar([0,1],vals,color=["#367baa","#679c71"])
        a2.set_xticks([0,1],labels=["PCA4","FA4"])
        a2.set_title(title)
        a2.grid(axis="y",alpha=.12)
    fig.savefig(out/"DANDI_PCA4_FA4_FEATURE_CLUSTER_COMPARISON.png",dpi=190,bbox_inches="tight")
    plt.close(fig)
    print("DANDI_PCA4_FA4_DONE",json.dumps({"feature":{
        m:{"train_silhouette":v["source_cluster"]["train_silhouette"],
           "val_direction_MI":v["source_cluster"]["validation_direction_association_postfit"]}
        for m,v in result["neural_pca_fa"].items()},
        "all_methods":list(result["methods"])},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
