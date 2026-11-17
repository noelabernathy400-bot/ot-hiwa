"""K-means on four-dimensional neural and kinematic observations of DANDI:000138.

Data source is DANDI:000138 version 0.220113.0407 (MC_Maze_Large).
This stage DOES NOT run HiWA or use kinematic/neural temporal matches in fitting.
Neural spikes -> binned firing rates -> sqrt -> smoothing -> z-score -> PCA(4).
Behavior -> [hand position x,y, hand velocity x,y], with NO dimensionality reduction;
train-fit z-score is for comparable K-means feature scales only.
Clustering is performed in 4D, plots are 2D coordinate projections.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET_ID = '000138'
DATASET_VERSION = '0.220113.0407'


def _as_float(f, key):
    return np.asarray(f[key], dtype=np.float64).reshape(-1)


def _select_event(trials, n):
    # Event times come only from NWB trial annotations, not paired behavior.
    options = ('move_onset_time', 'movement_onset_time', 'go_cue_time')
    for key in options:
        if key in trials:
            value = np.asarray(trials[key], dtype=float).reshape(-1)
            if len(value) == n and np.sum(np.isfinite(value)) >= 8:
                return key, value
    raise ValueError('No valid movement-onset/go-cue event among ' + repr(options)
                     + f'; available trial fields: {list(trials.keys())}')


def _find_hand_ts(f, flavor):
    """Find HAND position/velocity regardless of NWB container layout.

    Official NLB NWB datasets may store behavioral TimeSeries directly inside
    a ProcessingModule rather than inside a Position/Velocity container.
    Never substitute cursor position or velocity for hand measurements.
    """
    assert flavor in ("pos", "vel")
    candidates=[]
    def visit(name,obj):
        if not isinstance(obj,h5py.Group) or 'data' not in obj:
            return
        key=name.lower().split('/')[-1]
        if 'hand' not in key:
            return
        if flavor == 'pos' and not any(k in key for k in ('pos','position')):
            return
        if flavor == 'vel' and not any(k in key for k in ('vel','velocity')):
            return
        ds=obj['data']
        if not isinstance(ds,h5py.Dataset) or ds.ndim!=2 or ds.shape[1]<2:
            return
        candidates.append((name,obj))
    for root_name in ('processing','acquisition'):
        if root_name in f:
            f[root_name].visititems(lambda name,obj:visit(root_name+'/'+name,obj))
    if len(candidates)!=1:
        raise ValueError(f'Expected one hand {flavor} time series; found {[v[0] for v in candidates]}. '
                         'No cursor substitution is permitted.')
    name,series=candidates[0]
    xy=np.asarray(series['data'][:, :2],dtype=float)
    if 'timestamps' in series:
        ts=np.asarray(series['timestamps'],dtype=float)
    elif 'starting_time' in series and 'rate' in series['starting_time'].attrs:
        start=float(series['starting_time'][()])
        rate=float(series['starting_time'].attrs['rate'])
        ts=start+np.arange(len(xy))/rate
    else:
        raise ValueError(f'No timestamps/rate on {name}')
    if len(ts)!=len(xy) or not np.all(np.diff(ts)>0):
        raise ValueError(f'Invalid timestamps for {name}')
    return name,ts,xy


def _extract_spikes(units):
    if 'spike_times' not in units or 'spike_times_index' not in units:
        raise ValueError('NWB units table missing ragged spike_times and spike_times_index')
    values=np.asarray(units['spike_times'],dtype=float)
    ends=np.asarray(units['spike_times_index'],dtype=np.int64)
    starts=np.r_[0,ends[:-1]]
    if np.any(ends<starts):raise ValueError('Invalid ragged spike index')
    return [values[a:b] for a,b in zip(starts,ends)]


def load_trial_windows(path, bin_ms=20., event_start_ms=-100., event_end_ms=600.):
    """Returns aligned windows only for sampling; pairing never enters fitting."""
    dt=bin_ms/1000.
    if event_end_ms<=event_start_ms or bin_ms<=0:raise ValueError('Invalid bin/window')
    edges=np.arange(event_start_ms/1000.,event_end_ms/1000.+dt/2,dt)
    if len(edges)<3:raise ValueError('Window too short')
    centers=(edges[:-1]+edges[1:])/2
    with h5py.File(path,'r') as f:
        trials=f['intervals/trials']
        n=len(trials['start_time'])
        event_name,events=_select_event(trials,n)
        trial_start=_as_float(f,'intervals/trials/start_time')
        trial_stop=_as_float(f,'intervals/trials/stop_time')
        units=_extract_spikes(f['units'])
        name_pos,t_pos,xy_pos=_find_hand_ts(f,'pos')
        name_vel,t_vel,xy_vel=_find_hand_ts(f,'vel')
        valid=np.flatnonzero(np.isfinite(events) &
            (events+edges[0]>=trial_start) & (events+edges[-1]<=trial_stop) &
            (events+edges[0]>=max(t_pos[0],t_vel[0])) &
            (events+edges[-1]<=min(t_pos[-1],t_vel[-1])))
        if len(valid)<10:raise ValueError(f'Too few complete trial windows ({len(valid)})')
        x=np.zeros((len(valid),len(centers),len(units)),dtype=np.float32)
        y=np.zeros((len(valid),len(centers),4),dtype=np.float32)
        for ii,idx in enumerate(valid):
            time=events[idx]+centers
            e=events[idx]+edges
            # Spike rate is obtained from spike times with disjoint bins.
            for u,sp in enumerate(units):x[ii,:,u]=np.histogram(sp,bins=e)[0]/dt
            y[ii,:,:2]=np.column_stack([np.interp(time,t_pos,xy_pos[:,j]) for j in range(2)])
            y[ii,:,2:]=np.column_stack([np.interp(time,t_vel,xy_vel[:,j]) for j in range(2)])
    metadata={'dataset_id':DATASET_ID,'dataset_version':DATASET_VERSION,
              'nwb_name':path.name,'sha256':hashlib.file_digest(open(path,'rb'),'sha256').hexdigest(),
              'trial_count':int(n),'usable_trials':int(len(valid)),'neuron_count':len(units),
              'samples_per_trial':len(centers),'bin_ms':bin_ms,
              'event':event_name,'event_window_ms':[event_start_ms,event_end_ms],
              'position_series':name_pos,'velocity_series':name_vel,
              'kinematics':['hand_pos_x','hand_pos_y','hand_vel_x','hand_vel_y']}
    return x,y,valid,metadata


def preprocess(neural,behavior,seed=42,smooth_ms=40.,bin_ms=20.):
    """Trial-disjoint train/test. PCA fit on training neural activity only."""
    if neural.ndim!=3 or behavior.shape[:2]!=neural.shape[:2] or behavior.shape[-1]!=4:
        raise ValueError('Expected (trials,time,units) and (trials,time,4)')
    ids=np.arange(len(neural));tr,te=train_test_split(ids,test_size=.2,random_state=seed)
    neural=gaussian_filter1d(np.sqrt(np.maximum(neural,0)),
                            sigma=smooth_ms/bin_ms,axis=1,mode='nearest')
    a=neural.reshape(-1,neural.shape[-1]);b=behavior.reshape(-1,4)
    time_len=neural.shape[1]
    mask=np.isfinite(a).all(axis=1)&np.isfinite(b).all(axis=1)
    train_mask=np.isin(np.repeat(ids,time_len),tr)&mask
    if train_mask.sum()<50:raise ValueError('Too few finite training observations')
    std_x=StandardScaler().fit(a[train_mask]);ax=std_x.transform(a)
    pca=PCA(n_components=4,svd_solver='full',random_state=seed).fit(ax[train_mask])
    x=pca.transform(ax)
    # This is feature-wise standardization, not PCA or projection of behavior.
    std_y=StandardScaler().fit(b[train_mask]);y=std_y.transform(b)
    mask &= np.isfinite(x).all(axis=1)&np.isfinite(y).all(axis=1)
    return x,y,mask,np.repeat(ids,time_len),tr,te,pca,{'neural_mean':std_x.mean_.tolist(),
             'neural_scale':std_x.scale_.tolist(),'behavior_mean':std_y.mean_.tolist(),
             'behavior_scale':std_y.scale_.tolist()}


def _plot_projections(x, labels, title, names, out):
    pairs=list(combinations(range(4),2))
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    palette=plt.get_cmap('tab10')
    for ax,(i,j) in zip(axes.flat,pairs):
        for cluster in np.unique(labels):
            idx=labels==cluster
            ax.scatter(x[idx,i],x[idx,j],s=3,alpha=.5,color=palette(int(cluster)),
                       label=f'Group {int(cluster)+1}',rasterized=True)
        ax.set(xlabel=names[i],ylabel=names[j]);ax.grid(alpha=.14)
    handles,leg=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,leg,ncol=len(handles),loc='lower center',bbox_to_anchor=(.5,-.02))
    fig.suptitle(title,fontsize=15)
    fig.savefig(out,dpi=170,bbox_inches='tight');plt.close(fig)


def _cluster(x,trial_of,trials,k,seed):
    keep=np.isin(trial_of,trials)
    model=KMeans(n_clusters=k,random_state=seed,n_init=20).fit(x[keep])
    labels=model.predict(x)
    rng=np.random.default_rng(seed)
    idx=np.flatnonzero(keep)
    if len(idx)>4000:idx=rng.choice(idx,4000,replace=False)
    score=silhouette_score(x[idx],labels[idx])
    return model,labels,float(score)


def run(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    path=Path(args.nwb)
    if not path.is_file():raise FileNotFoundError(f'DANDI 000138 NWB file absent: {path}')
    x_raw,y_raw,valid,meta=load_trial_windows(path,args.bin_ms,args.start_ms,args.end_ms)
    x,y,keep,trial_of,tr,te,pca,scalers=preprocess(x_raw,y_raw,args.seed,args.smooth_ms,args.bin_ms)
    train=np.isin(trial_of,tr)&keep
    if train.sum()<args.k*10:raise ValueError('Insufficient points for K-means')
    mx,lx,sx=_cluster(x,trial_of,tr,args.k,args.seed)
    my,ly,sy=_cluster(y,trial_of,tr,args.k,args.seed)
    for key,array,labels,names in [
        ('neural',x,lx,[f'Neural PC{i+1}' for i in range(4)]),
        ('hand',y,ly,['pos x (z)','pos y (z)','vel x (z)','vel y (z)'])]:
        _plot_projections(array[train],labels[train],f'{key}: 4D K-means, K={args.k} (train only)',names,
                          out/f'{key}_4d_pairwise_train.png')
    np.savez_compressed(out/'clustered_4d.npz',neural4=x,hand4=y,
         train_mask=train,test_mask=np.isin(trial_of,te)&keep,
         train_trials=tr,test_trials=te,trial_of_window=trial_of,
         original_trial_ids=valid,neural_groups=lx,hand_groups=ly,
         neural_centers=mx.cluster_centers_,hand_centers=my.cluster_centers_)
    with (out/'windows_4d.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['trial_id','train','valid',*[f'neural_pc{i+1}' for i in range(4)],
                           'hand_pos_x_standard','hand_pos_y_standard','hand_vel_x_standard','hand_vel_y_standard',
                           'neural_cluster','hand_cluster'])
        for j in np.flatnonzero(keep):
            writer.writerow([int(valid[trial_of[j]]),int(train[j]),1,*x[j].tolist(),*y[j].tolist(),int(lx[j]),int(ly[j])])
    report={**meta,'split_seed':args.seed,'train_trials':len(tr),'test_trials':len(te),
            'train_window_count':int(train.sum()),'test_window_count':int((np.isin(trial_of,te)&keep).sum()),
            'feature_dim':4,'group_count':args.k,'neural_pca_explained_variance_ratio':pca.explained_variance_ratio_.tolist(),
            'neural_silhouette_train':sx,'behavior_silhouette_train':sy,
            'neural_cluster_counts_train':np.bincount(lx[train],minlength=args.k).tolist(),
            'behavior_cluster_counts_train':np.bincount(ly[train],minlength=args.k).tolist(),
            'scalers':scalers,'fitting_uses_pairs':False,'fitting_uses_labels':False,
            'notes':'Cluster numbers/colors are independent across domains; identical ID != correspondence. No HiWA fit at this stage.'}
    (out/'clustering_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('usable_trials','train_window_count','feature_dim','group_count',
            'neural_silhouette_train','behavior_silhouette_train')},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nwb',required=True,help='local DANDI 000138 NWB path')
    p.add_argument('--output',default='results/dandi_000138/hard_clustering')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--k',type=int,default=4)
    p.add_argument('--bin-ms',type=float,default=20.)
    p.add_argument('--start-ms',type=float,default=-100.)
    p.add_argument('--end-ms',type=float,default=600.)
    p.add_argument('--smooth-ms',type=float,default=40.)
    run(p.parse_args())

if __name__=='__main__':main()
