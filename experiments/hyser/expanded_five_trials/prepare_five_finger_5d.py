#!/usr/bin/env python3
"""Extract ALL five physically named Hyser finger force signals for all five trials.

No finger-to-cluster assumption; K stays a learned model-selection variable.
Makes trial-disjoint 5D EMG PCA and five-finger-force whitening folds.
Does not train an OT model.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import wfdb

FINGERS=("thumb","index","middle","ring","little")

def white(train,test):
    mean=train.mean(axis=0)
    cov=np.cov(train-mean,rowvar=False)
    vals,vecs=np.linalg.eigh(cov)
    if vals.min() < 1e-10: raise ValueError(f"Nearly singular covariance eigenvalues: {vals}")
    trans=(vecs*(vals**-0.5))@vecs.T
    return (train-mean)@trans,(test-mean)@trans,vals

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--raw-dir",type=Path,required=True)
    p.add_argument("--features-3d-archive-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    old=a.features_3d_archive_dir/"all_five_trials_UNSCALED_windows.npz"
    with np.load(old) as z:
        X256=np.array(z["X_emg_log1p_rms_256"],float)
        trial_ids=np.array(z["trial_id"],int)
    assert X256.shape==(1250,256) and trial_ids.shape==(1250,)
    allY=[];records=[]
    for trial in range(1,6):
        base=a.raw_dir/f"random_force_sample{trial}"
        f=wfdb.rdrecord(str(base))
        raw=np.array(f.p_signal,float)
        names=[name.lower().strip() for name in f.sig_name]
        if len(names)!=5 or set(names)!=set(FINGERS):
            raise ValueError(f"Unexpected Hyser finger channel names: {names}")
        idx=[names.index(name) for name in FINGERS]
        if float(f.fs)!=100 or raw.shape[0]!=2500:
            raise ValueError((trial,f.fs,raw.shape))
        ordered=raw[:,idx]
        win=ordered.reshape(250,10,5).mean(axis=1)
        assert np.isfinite(win).all()
        allY.append(win)
        records.append({"trial":trial,"original_channel_names":list(f.sig_name),
                        "original_channel_indices_in_named_order":idx,
                        "n_windows":250})
        print("force-trial",trial,"header-order",f.sig_name,"reordered-to",FINGERS,flush=True)
    Y5=np.concatenate(allY,axis=0)
    assert Y5.shape==(1250,5)
    np.savez_compressed(a.out/"all_five_trials_fivefinger_raw_and_emg.npz",
                        X_emg_log1p_rms_256=X256,Y_force_5_raw=Y5,trial_id=trial_ids,
                        sample_in_trial=np.tile(np.arange(250),5),
                        force_names=np.array(FINGERS))
    folds=[]
    for holdout in range(1,6):
        train_mask=trial_ids!=holdout
        scaler=StandardScaler().fit(X256[train_mask])
        pca=PCA(n_components=5,svd_solver="full").fit(scaler.transform(X256[train_mask]))
        xtr=pca.transform(scaler.transform(X256[train_mask]))
        xte=pca.transform(scaler.transform(X256[~train_mask]))
        Xt,Xs,xvals=white(xtr,xte)
        Yt,Ys,yvals=white(Y5[train_mask],Y5[~train_mask])
        np.savez_compressed(a.out/f"trial_heldout_{holdout}_5d.npz",
                            X_train=Xt,Y_train=Yt,X_test=Xs,Y_test=Ys,
                            train_trial_id=trial_ids[train_mask],
                            test_trial_id=trial_ids[~train_mask],
                            original_target_force_train=Y5[train_mask],
                            original_target_force_test=Y5[~train_mask],
                            force_names=np.array(FINGERS),
                            X_PCA5_variance_ratio=pca.explained_variance_ratio_,
                            X_PCA5_components=pca.components_,X_EMG_scaler_mean=scaler.mean_,
                            X_EMG_scaler_scale=scaler.scale_,X_eigenvalues=xvals,Y_eigenvalues=yvals)
        folds.append({"heldout":holdout,"PCA5_explained_variance_ratio":pca.explained_variance_ratio_.tolist(),
                      "X_whitening_eigenvalues":xvals.tolist(),
                      "Y_whitening_eigenvalues":yvals.tolist(),
                      "Y_whitening_condition_number":float(yvals.max()/yvals.min())})
    manifest={"task":"Hyser subject01/session1 random five trials",
              "source":"PhysioNet HD-SEMG 2.0.0",
              "force_header_validation":records,"five_finger_target_order":list(FINGERS),
              "full_arrays":{"X":list(X256.shape),"Y":list(Y5.shape)},
              "train_only_per_fold_PCA5_and_whitening":folds,
              "note":"5D is a NEW dataset representation; do not compare its absolute transport cost against old 3D experiment. Force channels are identified by header names, not assumed positions. No OT model fitted."}
    (a.out/"FIVE_FINGER_SOURCE_CHECK.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"success":True,"force_names":list(FINGERS),"n":1250,
                      "fold_conditions":[x["Y_whitening_condition_number"] for x in folds]},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
