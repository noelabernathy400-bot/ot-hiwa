"""Fixed-outcome-window lag diagnostic: preserves time index across candidate lags."""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def r2(y,p):return float(1-np.sum((y-p)**2)/np.sum((y-y.mean(0))**2))

def main():
 root=Path(__file__).resolve().parents[1];out=root/'results/diagnostics';out.mkdir(exist_ok=True)
 z=np.load(root/'results/clustering/clustered_4d.npz',allow_pickle=False)
 h=json.loads((root/'results/hard_hiwa/hard_hiwa_results.json').read_text())
 x=(np.asarray(z['neural4'],float)-np.asarray(h['source_mean']))/h['source_radius']
 y=(np.asarray(z['hand4'],float)-np.asarray(h['target_mean']))/h['target_radius']
 x=x.reshape(500,35,4);y=y.reshape(500,35,4)
 tr=z['train_mask'].reshape(500,35)[:,0];te=z['test_mask'].reshape(500,35)[:,0]
 ids=np.flatnonzero(tr);fit,val=train_test_split(ids,test_size=.2,random_state=2026)
 fit=np.isin(np.arange(500),fit);val=np.isin(np.arange(500),val)
 # 0-index target [10..24] (110..390 ms) permits [-10,+10] neural-side offsets.
 target_idx=np.arange(10,25)
 candidates=list(range(-10,11))
 vals=[]
 for lag in candidates:
  neural_idx=target_idx-lag
  def gather(mask):return x[mask][:,neural_idx].reshape(-1,4),y[mask][:,target_idx].reshape(-1,4)
  xf,yf=gather(fit);xv,yv=gather(val)
  choice=max(( (r2(yv,Ridge(alpha=a).fit(xf,yf).predict(xv)),a) for a in [.001,.1,1,10,100]), key=lambda t:t[0]);score,alpha=choice
  xtr,ytr=gather(tr);xe,ye=gather(te)
  model=Ridge(alpha=alpha).fit(xtr,ytr)
  vals.append({'lag_bins':lag,'lag_ms':20*lag,'valid_R2':float(score),'alpha':float(alpha),'test_R2_exploratory':r2(ye,model.predict(xe))})
 selected=max(vals,key=lambda v:v['valid_R2'])
 print('Best fixed-target lag',selected)
 # permutation controls: trial permutation on test with unchanged time bins
 rng=np.random.default_rng(3027)
 sel=selected['lag_bins'];nx=target_idx-sel
 xr=x[tr][:,nx].reshape(-1,4);yr=y[tr][:,target_idx].reshape(-1,4)
 xt=x[te][:,nx].reshape(-1,4);yt=y[te][:,target_idx].reshape(-1,4)
 model=Ridge(alpha=selected['alpha']).fit(xr,yr)
 original=r2(yt,model.predict(xt))
 shuffled=[]
 for i in range(100):
  order=rng.permutation(te.sum());z=y[te][order][:,target_idx].reshape(-1,4)
  shuffled.append(r2(z,model.predict(xt)))
 summary={'design':'Fixed target time indices [10,24] (110ms..390ms), same outcomes for all lag candidates; lag 20ms * integer; chosen on 80 train-validation trials only',
          'best_validation_lag':selected,'heldout_R2_chosen':original,
          'test_trial_shuffled_R2_mean':float(np.mean(shuffled)),
          'test_trial_shuffled_R2_sd':float(np.std(shuffled,ddof=1)),
          'lag_curve':vals, 'warning':'Correlation is non-causal; task time and trial structure can confound delay interpretation.'}
 (out/'fixed_window_lag.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 fig,ax=plt.subplots(figsize=(8,4));ax.plot([v['lag_ms'] for v in vals],[v['valid_R2'] for v in vals],marker='o',label='Train-val R2 (constant target window)')
 ax.plot([v['lag_ms'] for v in vals],[v['test_R2_exploratory'] for v in vals],linestyle='--',label='Test (descriptive)')
 ax.axvline(selected['lag_ms'],ls=':',color='gray',label='Lag chosen on train-val')
 ax.set(xlabel='hand time - neural time (ms)',ylabel='R2',title='Fixed-window lag diagnostic');ax.legend(fontsize=8);ax.grid(alpha=.2)
 fig.savefig(out/'fixed_window_lag.png',dpi=180,bbox_inches='tight');plt.close(fig)
 print(json.dumps({k:v for k,v in summary.items() if k not in ('lag_curve',)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
