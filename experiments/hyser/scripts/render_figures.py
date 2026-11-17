#!/usr/bin/env python3
"""Source-only K-means groups on paired REAL Hyser visual point records.
The coordinates in visual_points_base36.txt are quantized ONLY for plotting.
Never fit the model or report R2 from these quantized points.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
COLORS=['#3589c2','#ee9b43','#33a46f','#d881b7']
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Noto Sans CJK SC','Noto Sans CJK JP','DejaVu Sans'],'axes.unicode_minus':False})
lines=(ROOT/'data/visual_points_base36.txt').read_text().splitlines()
cloud={}
for line in lines:
    name,s=line.split(':',1)
    assert len(s)==250*8
    groups=[s[i:i+8] for i in range(0,len(s),8)]
    coords=np.array([[int(g[k:k+2],36)/150-4 for k in (0,2,4)] for g in groups])
    cloud[name]=(coords,np.array([int(g[6]) for g in groups],dtype=int))
X,labels=cloud['EMG']
Y,_old_independent_labels=cloud['FORCE']
# IMPORTANT: target Y inherits labels from X by synchronized index.
assert len(X)==len(Y)==len(labels)==250 and list(np.bincount(labels))==[46,53,91,60]
d=json.loads((ROOT/'results/latest.json').read_text())
Rf=np.array(d['free']['R']); Rm=np.array(d['fixed']['R'])
points=np.vstack((X,Y,X@Rf.T,X@Rm.T))
center=(points.max(0)+points.min(0))/2
size=np.ptp(points,axis=0).max()*0.54
limits=[(t-size,t+size) for t in center]
handles=[Line2D([0],[0],marker='o',linestyle='',color=COLORS[i],label=f'组{i+1}（{(labels==i).sum()}对）') for i in range(4)]

def panel(ax,Z,is3d,title):
    for k in range(4):
        sub=Z[labels==k];mu=sub.mean(0)
        if is3d:
            ax.scatter(*sub.T,color=COLORS[k],s=16,alpha=.74,depthshade=False,rasterized=True)
            ax.scatter(*mu,marker='X',color=COLORS[k],edgecolors='#222',s=95)
        else:
            ax.scatter(sub[:,0],sub[:,1],color=COLORS[k],s=18,alpha=.74,rasterized=True)
            ax.scatter(mu[0],mu[1],marker='X',color=COLORS[k],edgecolors='#222',s=105)
    ax.set_title(title,pad=12,fontsize=11)
    ax.set_xlim(*limits[0]);ax.set_ylim(*limits[1])
    ax.set_xlabel('坐标1');ax.set_ylabel('坐标2')
    if is3d:
        ax.set_zlim(*limits[2]);ax.set_zlabel('坐标3')
        ax.view_init(elev=23,azim=-61);ax.set_box_aspect((1,1,1))
    else:ax.set_aspect('equal',adjustable='box')
    ax.grid(alpha=.16)

def save_pair(filename, R, title, subtitle):
    fig=plt.figure(figsize=(13.1,10.1),facecolor='white')
    fig.suptitle(title,fontsize=16,y=.986,fontweight='bold')
    fig.text(.5,.955,'左：目标域（手指力量）    右：源域（肌电）    标签只由源域K-means给出',ha='center',fontsize=10)
    for col,(Z,word) in enumerate(((Y,'目标域，标签继承'),(X@R.T,'源域'))):
        panel(fig.add_subplot(2,2,col+1,projection='3d'),Z,True,word+' 3D')
        panel(fig.add_subplot(2,2,col+3),Z,False,word+' 2D')
    fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.038))
    fig.text(.5,.01,subtitle,ha='center',fontsize=9)
    fig.subplots_adjust(left=.035,right=.98,top=.88,bottom=.12,wspace=.13,hspace=.28)
    fig.savefig(ROOT/'figures'/filename,dpi=175);plt.close(fig)

save_pair('01_before_hiwa.png',np.eye(3),'第一阶段：源域唯一 K-means 与继承标签的目标域','全部250对同步窗口，尚未进行正交对齐')
save_pair('02_after_hiwa_free.png',Rf,'第二阶段：自由簇级 OT Hard HiWA 对齐后','R² -0.6741 → -0.1691；配对相对平方误差 1.6741 → 1.1691；仅旋转变化准则满足')
save_pair('03_after_hiwa_fixed.png',Rm,'第三阶段：固定已知簇对应的 Hard HiWA 对齐后','R² -0.6741 → -0.1539；配对相对平方误差 1.6741 → 1.1539')
fig=plt.figure(figsize=(18,7.9))
fig.suptitle('唯一源域 K-means：目标域 / 源域对齐前 / 自由簇OT / 固定簇对应',fontsize=15,y=.986)
for j,(Z,t) in enumerate(((Y,'目标域'),(X,'源域原始'),(X@Rf.T,'源域自由OT后'),(X@Rm.T,'源域固定组后'))):
    panel(fig.add_subplot(2,4,j+1,projection='3d'),Z,True,t+' 3D')
    panel(fig.add_subplot(2,4,j+5),Z,False,t+' 2D')
fig.legend(handles=handles,loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.044))
fig.subplots_adjust(left=.01,right=.99,top=.89,bottom=.12,wspace=.09,hspace=.25)
fig.savefig(ROOT/'figures/00_complete_comparison.png',dpi=175);plt.close(fig)
P=np.asarray(d['free']['P'])
fig,ax=plt.subplots(figsize=(5.7,4.7))
im=ax.imshow(P,cmap='Blues',vmin=0,vmax=.25)
for i in range(4):
    for j in range(4):
        ax.text(j,i,f'{P[i,j]:.4f}',ha='center',va='center',color='white' if P[i,j]>.1 else '#222',fontsize=8.5)
ax.set(xticks=range(4),yticks=range(4),xlabel='目标域继承组',ylabel='源域K-means组',title='自由簇级OT矩阵 P')
fig.colorbar(im,ax=ax,shrink=.8);fig.tight_layout()
fig.savefig(ROOT/'figures/04_free_group_transport.png',dpi=175);plt.close(fig)
print('Five updated HiWA figures generated (target always inherits source-only labels).')
