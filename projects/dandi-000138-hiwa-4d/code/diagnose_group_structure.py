"""Observe independent KMeans group imbalance, cross-domain ambiguity and HiWA nonidentifiability."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
 root=Path(__file__).resolve().parents[1];out=root/'results/diagnostics';out.mkdir(exist_ok=True)
 a=np.load(root/'results/clustering/clustered_4d.npz',allow_pickle=False);z=json.loads((root/'results/hard_hiwa/hard_hiwa_results.json').read_text())
 t=np.asarray(a['train_mask'],bool);a1=np.asarray(a['neural_groups'])[t];b1=np.asarray(a['hand_groups'])[t]
 hist=np.zeros((4,4),dtype=np.int64)
 for k,l in zip(a1,b1):hist[k,l]+=1
 p=hist/hist.sum()
 weights_x=hist.sum(1)/hist.sum(); weights_y=hist.sum(0)/hist.sum()
 perm=linear_sum_assignment(hist,maximize=True)
 matched=float(hist[perm].sum()/len(a1))
 per_seed=[]
 for s in z['seed_results']:
  P=np.asarray(s['P']);R=np.asarray(s['R']);u,v=linear_sum_assignment(P,maximize=True)
  selected=[(int(i),int(j),float(P[i,j])) for i,j in zip(u,v)]
  match_coocc=float(np.sum(p[u,v]))
  per_seed.append({'seed':s['seed'],'det':float(np.linalg.det(R)),
                   'best_transport_permutation':selected,'matched_paired_sample_fraction':match_coocc,
                   'transport_permutation_mass':float(P[u,v].sum()),
                   'relative_pairwise_R_diff':None})
 for k in range(len(per_seed)):
  rot=np.asarray(z['seed_results'][k]['R']);ref=np.asarray(z['seed_results'][0]['R'])
  per_seed[k]['relative_pairwise_R_diff']=float(np.linalg.norm(rot-ref,'fro'))
 # cross-domain pair distribution can have non-one-to-one structure
 target_optimum=hist[perm]
 stat={'n_train_windows':int(len(a1)), 'neural_cluster_sizes':hist.sum(1).tolist(),
       'hand_cluster_sizes':hist.sum(0).tolist(),
       'neural_cluster_empirical_masses':weights_x.tolist(),'hand_cluster_empirical_masses':weights_y.tolist(),
       'HiWA_assumed_cluster_mass_each':.25,
       'max_deviation_from_uniform_mass':{'neural':float(np.max(abs(weights_x-.25))),'hand':float(np.max(abs(weights_y-.25)))},
       'time_aligned_cluster_cooccurrence':hist.tolist(),
       'max_pair_fraction_any_permutation':matched,
       'trivial_modal_behavior_cluster_fraction':float(max(weights_y)),
       'per_seed_HiWA_transport_alignment_vs_true_time_cooccurrence':per_seed,
       'notes':'The time-paired contingency is diagnostic only. Unsupervised fitting uses no matched labels. Hungarian cluster correspondence is not a neural decoder and KMeans labels are arbitrary.'}
 (out/'group_geometry.json').write_text(json.dumps(stat,ensure_ascii=False,indent=2)+'\n')
 fig,axes=plt.subplots(1,2,figsize=(10,4.1),constrained_layout=True)
 im=axes[0].imshow(p,cmap='Blues');axes[0].set(title='Time-paired cluster co-occurrence',xlabel='Hand KMeans group',ylabel='Neural KMeans group',xticks=range(4),yticks=range(4))
 for i in range(4):
  for j in range(4):axes[0].text(j,i,f'{p[i,j]:.3f}',ha='center',va='center',fontsize=8)
 plt.colorbar(im,ax=axes[0],fraction=.05)
 b=np.arange(4);axes[1].bar(b-.18,weights_x,width=.35,label='Neural');axes[1].bar(b+.18,weights_y,width=.35,label='Hand')
 axes[1].axhline(.25,ls='--',color='black',label='Uniform HiWA cluster prior')
 axes[1].set(title='Empirical cluster probabilities differ',xlabel='Cluster index (within-domain)',ylabel='Fraction of train windows',xticks=b)
 axes[1].legend(fontsize=8)
 fig.savefig(out/'cluster_masses_and_cooccurrence.png',dpi=180,bbox_inches='tight');plt.close(fig)
 print(json.dumps({'weights_x':weights_x.tolist(),'weights_y':weights_y.tolist(),'max_match_frac':matched,'mode':float(max(weights_y)),
                   'seed_P_semantic_match':[(r['seed'],r['matched_paired_sample_fraction'],r['det'],r['relative_pairwise_R_diff']) for r in per_seed]},indent=2))

if __name__=='__main__':main()
