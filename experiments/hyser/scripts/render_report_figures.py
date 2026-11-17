#!/usr/bin/env python3
"""Replot authentic Hyser hard clustering and aligned scatter for LaTeX.
Source numbers are archived figure-only quantizations; never used for fitting.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
root=Path(__file__).resolve().parents[1]
folder=root/'figures'
colors=['#267bb0','#2fa46d','#976cc0','#e17da8']

def decode(filename,n):
    records={}
    for line in (folder/filename).read_text().splitlines():
        if not line.strip():continue
        name,packed=line.split(':',1)
        width=7 if n==200 else 8
        assert len(packed)==n*width,(name,len(packed))
        chunks=[packed[i:i+width] for i in range(0,len(packed),width)]
        coords=np.asarray([[int(v[j:j+2],36)/150-4 for j in (0,2,4)] for v in chunks])
        labels=np.asarray([int(v[6]) for v in chunks])
        gmm=np.asarray([int(v[7]) for v in chunks]) if n==250 else labels
        records[name]=(coords,labels,gmm)
    return records

def initial():
    p=decode('original200_base36.txt',200)
    fig=plt.figure(figsize=(12.8,9.9))
    for col,name in enumerate(('EMG','FORCE')):
        x,l,_=p[name]
        ax=fig.add_subplot(2,2,col+1,projection='3d')
        centers=np.asarray([x[l==k].mean(axis=0) for k in range(4)])
        for k in range(4):
            pts=x[l==k]
            ax.scatter(*pts.T,s=12,alpha=.65,c=colors[k],label=f'Cluster {k+1}')
            ax.scatter(*centers[k],s=145,marker='X',c=colors[k],edgecolors='black',linewidths=.7)
        ax.set(xlabel='Feature 1',ylabel='Feature 2',zlabel='Feature 3',
               title=f'Hyser real {name} — independent hard K-means')
        ax.legend(fontsize=7)
        ax=fig.add_subplot(2,2,col+3)
        _,_,v=np.linalg.svd(x-x.mean(0),full_matrices=False)
        y=(x-x.mean(0))@v[:2].T
        means=(centers-x.mean(0))@v[:2].T
        for k in range(4):
            ax.scatter(*y[l==k].T,s=15,alpha=.68,c=colors[k],label=f'Cluster {k+1}')
            ax.scatter(*means[k],s=145,marker='X',c=colors[k],edgecolors='black',linewidths=.7)
        ax.set(xlabel='Projection PC1',ylabel='Projection PC2',
               title=f'Hyser real {name} — independent hard K-means — 2D PCA projection')
        ax.legend(fontsize=7)
    fig.tight_layout(pad=2)
    fig.savefig(folder/'initial200_hard_clusters_fourpanel.png',dpi=160)
    plt.close(fig)

def post():
    d=decode('clouds_base36.txt',250)
    x,cx,gx=d['EMG']; y,cy,gy=d['FORCE']
    K=np.array([[.22966914,-.90608859,-.35532457],[-.96818997,-.24994848,.01157283],[-.09929885,.34136377,-.93467134]])
    G=np.array([[-.15877815,-.16200661,-.97393191],[.48899079,-.86986552,.06497673],[.85771645,.46592685,-.21733538]])
    cloud=np.vstack((x,y,x@K.T,x@G.T))
    low=cloud.min(0)-.1;high=cloud.max(0)+.1
    def cfg(ax,title):
        ax.set_title(title,fontsize=10)
        ax.set(xlabel='Coordinate 1',ylabel='Coordinate 2',zlabel='Coordinate 3')
        ax.set_xlim(low[0],high[0]);ax.set_ylim(low[1],high[1]);ax.set_zlim(low[2],high[2])
        ax.view_init(elev=21,azim=-64);ax.tick_params(labelsize=7)
    fig=plt.figure(figsize=(12,5.2))
    for i,(cloud,l,title) in enumerate(((x,cx,'Whitened EMG / K-means'),(y,cy,'Whitened force / K-means')),1):
        ax=fig.add_subplot(1,2,i,projection='3d');cfg(ax,title)
        for k in range(4):ax.scatter(*cloud[l==k].T,s=10,c=colors[k],alpha=.72)
    fig.tight_layout()
    fig.savefig(folder/'all250_whitened_hard_clusters_3D.png',dpi=180);plt.close(fig)
    cases=[(np.eye(3),cx,'Before alignment'),(K,cx,'After K-means Hard HiWA'),(G,gx,'After GMM Hard HiWA')]
    handles=[Line2D([0],[0],marker='^',color='w',markerfacecolor='#454954',markersize=6,label='Force target')]
    handles += [Line2D([0],[0],marker='o',color='w',markerfacecolor=colors[k],markersize=6,label=f'EMG group {k+1}') for k in range(4)]
    fig=plt.figure(figsize=(16,5.3))
    for i,(R,l,title) in enumerate(cases,1):
        ax=fig.add_subplot(1,3,i,projection='3d');cfg(ax,title)
        ax.scatter(*y.T,s=9,c='#454954',marker='^',alpha=.26)
        v=x@R.T
        for k in range(4):ax.scatter(*v[l==k].T,s=12,c=colors[k],alpha=.75)
    fig.legend(handles=handles,ncol=5,loc='lower center',frameon=False,fontsize=8)
    fig.suptitle('Real Hyser 250 points: Hard HiWA alignment',fontsize=12)
    fig.subplots_adjust(left=.015,right=.985,top=.88,bottom=.15,wspace=.02)
    fig.savefig(folder/'all250_after_hard_hiwa_3D_overlay.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5),sharex=True,sharey=True)
    for ax,(R,l,title) in zip(axes,cases):
        moved=x@R.T
        ax.scatter(*y[:,:2].T,s=12,c='#454954',marker='^',alpha=.28)
        for k in range(4):ax.scatter(*moved[l==k,:2].T,s=12,c=colors[k],alpha=.7)
        ax.set_title(title,fontsize=10);ax.set_xlabel('Whitened coordinate 1');ax.grid(alpha=.14)
    axes[0].set_ylabel('Whitened coordinate 2')
    fig.legend(handles=handles,ncol=5,loc='lower center',frameon=False,fontsize=8)
    fig.suptitle('Same whitened axes (first two coordinates)',fontsize=12)
    fig.tight_layout(rect=(0,.07,1,.92))
    fig.savefig(folder/'all250_after_hard_hiwa_2D_overlay.png',dpi=180)
    plt.close(fig)
    print('Figure generation completed from authentic archived Hyser point coordinates.')

if __name__=='__main__':
    initial();post()
