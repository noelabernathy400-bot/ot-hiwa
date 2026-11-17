#!/usr/bin/env python3
"""Plot-only reconstruction of real Hyser Hard vs Soft group-supervised runs.

X/Y plot coordinates are quantized (~0.0067 per coordinate) in
data/visual_points_base36.txt. The global rotations, prototype centers and
numerical metrics come from results/latest.json and
results/soft_experiment_20261117.json, calculated on full precision data.
Plot data MUST NOT be used to re-compute published errors.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
(ROOT/'figures').mkdir(exist_ok=True,parents=True)
plt.rcParams.update({'font.family':'sans-serif',
 'font.sans-serif':['Noto Sans CJK SC','Noto Sans CJK JP','DejaVu Sans'],
 'axes.unicode_minus':False})

def points():
    records={}
    for line in (ROOT/'data/visual_points_base36.txt').read_text().splitlines():
        if not line.strip():continue
        name,packed=line.split(':',1)
        assert len(packed)==250*8,(name,len(packed))
        chunks=[packed[i:i+8] for i in range(0,len(packed),8)]
        xyz=np.array([[int(t[q:q+2],36)/150-4 for q in (0,2,4)]
                      for t in chunks])
        labs=np.array([int(t[6]) for t in chunks],dtype=int)
        records[name]=(xyz,labs)
    X,labels=records['EMG']
    Y=records['FORCE'][0]
    return X,Y,labels

X,Y,labels=points()
hard=json.loads((ROOT/'results/latest.json').read_text())
soft=json.loads((ROOT/'results/soft_experiment_20261117.json').read_text())
rhard=np.asarray(hard['fixed']['R'])
entry=next(x for x in soft['fixed_P_seed42']
           if x['name']=='soft_TPAMI_prototype_source_only_lambda1')
rsoft=np.asarray(entry['R'])
C=np.asarray(entry['prototypes'])
palette=['#347faa','#e3943a','#2b9b70','#ad65bc']
allpts=np.vstack([Y,X@rhard.T,X@rsoft.T])
mid=(allpts.max(0)+allpts.min(0))/2
width=np.ptp(allpts,axis=0).max()*.54
limits=[(t-width,t+width) for t in mid]

fig=plt.figure(figsize=(15.8,9),facecolor='white')
for j,(V,caption) in enumerate([
  (Y,'Target domain (force)'),
  (X@rhard.T,'Source after Hard HiWA'),
  (X@rsoft.T,'Source after Soft prototype HiWA')
]):
    ax=fig.add_subplot(2,3,j+1,projection='3d')
    for k in range(4):
        sub=V[labels==k]
        ax.scatter(sub[:,0],sub[:,1],sub[:,2],color=palette[k],
                   s=14,alpha=.75,depthshade=False)
    ax.set_xlim(*limits[0]);ax.set_ylim(*limits[1]);ax.set_zlim(*limits[2])
    ax.set_box_aspect((1,1,1))
    ax.set(xlabel='Coord 1',ylabel='Coord 2',zlabel='Coord 3')
    ax.tick_params(labelsize=7);ax.view_init(elev=22,azim=-60)
    ax.set_title(caption+' (3D)',fontsize=10)
    ax2=fig.add_subplot(2,3,j+4)
    for k in range(4):
        sub=V[labels==k]
        ax2.scatter(sub[:,0],sub[:,1],s=15,color=palette[k],alpha=.73)
    ax2.set(xlabel='Coord 1',ylabel='Coord 2')
    ax2.set_xlim(*limits[0]);ax2.set_ylim(*limits[1])
    ax2.set_aspect('equal',adjustable='box')
    ax2.grid(alpha=.15);ax2.set_title(caption+' (2D)',fontsize=10)
fig.suptitle('250 paired Hyser windows: inherited source groups, fixed P=I/4',
             fontsize=14,y=.99)
handles=[Line2D([0],[0],marker='o',linestyle='',color=palette[k],
                 label=f'Reference source group {k+1}') for k in range(4)]
fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,fontsize=9)
fig.subplots_adjust(left=.015,right=.985,top=.91,bottom=.13,hspace=.2,wspace=.1)
fig.savefig(ROOT/'figures/05_hard_vs_soft_paired_2d3d.png',dpi=165,bbox_inches='tight')
plt.close(fig)

z=X@C.T;z-=z.max(axis=1,keepdims=True)
A=np.exp(z);A/=A.sum(axis=1,keepdims=True)
entropy=-np.sum(np.where(A>0,A*np.log(A),0),axis=1)
fig,(ax,ax2)=plt.subplots(1,2,figsize=(13.2,5),
                         gridspec_kw={'width_ratios':[2.15,1]})
hm=ax.imshow(A.T,aspect='auto',cmap='viridis',
             vmin=0,vmax=1,interpolation='nearest',origin='lower')
ax.set(xlabel='Synchronized window index',ylabel='Soft source groups',
       title='Source-only TPAMI prototypes; target inherits same memberships')
ax.set_yticks(range(4),[f'Soft {i+1}' for i in range(4)])
fig.colorbar(hm,ax=ax,fraction=.028,pad=.02,label='Membership probability')
ax2.hist(entropy,bins=18,color='#2b819b',alpha=.8)
ax2.set(xlabel='Entropy (nats)',ylabel='Window count',
        title='Soft group assignment uncertainty')
ax2.grid(axis='y',alpha=.15)
fig.tight_layout()
fig.savefig(ROOT/'figures/06_soft_source_membership.png',dpi=170,bbox_inches='tight')
plt.close(fig)

names=['Hard fixed','Soft KMeans tau=.5','Soft TPAMI prototype','Oracle']
values=[hard['fixed']['R2'],
        next(x for x in soft['fixed_P_seed42'] if x['name']=='soft_kmeans_tau0.5')['r2'],
        entry['r2'],soft['oracle']['r2']]
fig,ax=plt.subplots(figsize=(8,3.5))
xx=np.arange(len(names))
ax.plot(xx,values,marker='o',color='#24728a',lw=2,ms=8)
ax.set_xticks(xx,names)
ax.set_ylabel('Paired R² (larger is better)')
ax.set_ylim(-.16,-.133)
ax.grid(axis='y',alpha=.2)
for i,v in enumerate(values):
    ax.annotate(f'{v:.5f}',(i,v),xytext=(0,7),textcoords='offset points',
                ha='center',fontsize=9)
fig.tight_layout()
fig.savefig(ROOT/'figures/07_paired_R2_comparison.png',dpi=170,bbox_inches='tight')
plt.close(fig)
print('Rendered authentic (plot-quantized) Hyser source-inherited soft figures.')
