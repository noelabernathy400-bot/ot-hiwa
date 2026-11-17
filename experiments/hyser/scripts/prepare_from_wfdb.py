#!/usr/bin/env python3
"""Real Hyser sample1 to full-precision paired X,Y; target never re-clustered.
Input: WFDB random_preprocess_sample1.{hea,dat}, random_force_sample1.{hea,dat}.
Run from repo root: python experiments/hyser/scripts/prepare_from_wfdb.py --raw-dir experiments/hyser/raw
"""
from pathlib import Path
import argparse
import numpy as np
from scipy.linalg import sqrtm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import wfdb

def white(z):
    zz=z-z.mean(0,keepdims=True)
    root=np.real_if_close(sqrtm(np.cov(zz,rowvar=False))).astype(float)
    return zz@np.linalg.pinv(root)

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--raw-dir',type=Path,required=True)
    a.add_argument('--output',type=Path,default=Path('experiments/hyser/data/paired_full_precision.npz'))
    args=a.parse_args()
    e=wfdb.rdrecord(str(args.raw_dir/'random_preprocess_sample1'))
    f=wfdb.rdrecord(str(args.raw_dir/'random_force_sample1'))
    assert e.p_signal.shape==(51200,256) and f.p_signal.shape==(2500,5)
    assert abs(len(e.p_signal)/e.fs-len(f.p_signal)/f.fs)<.01
    names=['thumb','index','middle','ring','little']
    force=f.p_signal[:,[f.sig_name.index(n) for n in names]]
    rms=[];fw=[]
    for k in range(250):
        ei=e.p_signal[round(k*.1*e.fs):round((k+1)*.1*e.fs)]
        fi=force[round(k*.1*f.fs):round((k+1)*.1*f.fs)]
        rms.append(np.sqrt(np.mean((ei-ei.mean(0))**2,axis=0)))
        fw.append(fi.mean(0))
    A=StandardScaler().fit_transform(np.log1p(np.asarray(rms)*1e5))
    X=white(PCA(n_components=3,random_state=42).fit_transform(A))
    fw=np.asarray(fw)
    selected=np.argsort(-fw.var(0))[:3]
    Y=white(fw[:,selected])
    labels=KMeans(n_clusters=4,n_init=30,random_state=42).fit_predict(X)
    assert np.bincount(labels,minlength=4).tolist()==[46,53,91,60]
    args.output.parent.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(args.output,X=X,Y=Y,labels=labels)
    print('Saved',args.output,'force channels:',[names[j] for j in selected])
    print('Means',X.mean(0),Y.mean(0))
    print('Covariances',np.cov(X,rowvar=False),np.cov(Y,rowvar=False))
    print('Target labels are inherited: target_label[i] = source_label[i]')
if __name__=='__main__':main()
