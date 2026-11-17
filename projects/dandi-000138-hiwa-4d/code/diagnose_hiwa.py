"""Research diagnostics on archived DANDI:000138 4D features, without altering HiWA.

Data and HIWA transforms are frozen. All model and lag choices use training trials;
held-out test is evaluated once for the selected approach. This script does not
claim any lagged result is a true HiWA outcome.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def r2(y: np.ndarray, z: np.ndarray) -> float:
    den = np.sum((y - y.mean(axis=0)) ** 2)
    return float(1 - np.sum((y-z)**2)/den)


def procrustes(x,y):
    u,s,vt = np.linalg.svd(y.T@x)
    R=u@vt
    return R


def canonical_cors(x,y):
    # Orthogonal CCA directions; shrinkage ridge to guard conditioning.
    x=x-x.mean(0); y=y-y.mean(0)
    xx=x.T@x/len(x); yy=y.T@y/len(y); xy=x.T@y/len(x)
    def inv_sqrt(cov):
        val,u=np.linalg.eigh(cov)
        return (u/np.sqrt(np.maximum(val,1e-9)))@u.T
    return np.linalg.svd(inv_sqrt(xx)@xy@inv_sqrt(yy))[1]


def paired_at_lag(a,b,trial_mask,lag):
    # positive lag => neural earlier than behavior
    xx=a[trial_mask]; yy=b[trial_mask]
    if lag > 0:
        xx=xx[:,:-lag];yy=yy[:,lag:]
    elif lag < 0:
        xx=xx[:,-lag:];yy=yy[:,:lag]
    return xx.reshape(-1,4),yy.reshape(-1,4)


def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--project',default=str(Path(__file__).resolve().parents[1]))
    pa.add_argument('--out',default=None)
    p=pa.parse_args();root=Path(p.project)
    out=Path(p.out) if p.out else root/'results/diagnostics';out.mkdir(parents=True,exist_ok=True)
    d=np.load(root/'results/clustering/clustered_4d.npz',allow_pickle=False)
    results=json.loads((root/'results/hard_hiwa/hard_hiwa_results.json').read_text())
    tr=np.asarray(d['train_mask'],bool); te=np.asarray(d['test_mask'],bool)
    idx=np.asarray(d['trial_of_window'],int); ntrial=idx.max()+1
    timebins=len(idx)//ntrial
    assert len(idx)==ntrial*timebins and np.array_equal(idx,np.repeat(np.arange(ntrial),timebins))
    x=np.asarray(d['neural4'],float);y=np.asarray(d['hand4'],float)
    x=(x-np.asarray(results['source_mean']))/results['source_radius']
    y=(y-np.asarray(results['target_mean']))/results['target_radius']
    a=x.reshape(ntrial,timebins,4);b=y.reshape(ntrial,timebins,4)
    trtrial=tr.reshape(ntrial,timebins)[:,0];tetrial=te.reshape(ntrial,timebins)[:,0]
    assert trtrial.sum()==400 and tetrial.sum()==100
    assert np.array_equal(tr,trtrial.repeat(timebins))
    assert np.array_equal(te,tetrial.repeat(timebins))
    alltrain=np.where(trtrial)[0];fit_trials,val_trials=train_test_split(alltrain,random_state=2026,test_size=.2)
    fitmask=np.isin(np.arange(ntrial),fit_trials);valmask=np.isin(np.arange(ntrial),val_trials)
    alpha_grid=[0.001,0.1,1.0,10.0,100.0]
    lag_grid=list(range(-15,16))
    selected=None; evaluations=[]
    for lag in lag_grid:
        xfit,yfit=paired_at_lag(a,b,fitmask,lag)
        xv,yv=paired_at_lag(a,b,valmask,lag)
        xt,yt=paired_at_lag(a,b,trtrial,lag)
        xe,ye=paired_at_lag(a,b,tetrial,lag)
        best=None
        for alpha in alpha_grid:
            dec=Ridge(alpha=alpha,fit_intercept=True)
            dec.fit(xfit,yfit)
            score=r2(yv,dec.predict(xv))
            if best is None or score>best[0]:best=(score,alpha)
        dc=Ridge(alpha=best[1]).fit(xt,yt)
        testscore=r2(ye,dc.predict(xe))
        ort=procrustes(xt,yt)
        ortscore=r2(ye,xe@ort.T)
        values={'lag_bins':lag,'lag_ms':lag*20,'validation_R2':best[0],
                'ridge_alpha':best[1], 'test_ridge_R2_exploratory':testscore,
                'test_orthogonal_R2_exploratory':ortscore}
        evaluations.append(values)
        if selected is None or best[0]>selected['validation_R2']:selected=values
    # exact same-time evaluation and rotation, tests only measure not tune
    xtr,ytr=paired_at_lag(a,b,trtrial,0)
    xtest,ytest=paired_at_lag(a,b,tetrial,0)
    rorth=procrustes(xtr,ytr)
    id_r2=r2(ytest,xtest)
    orth_r2=r2(ytest,xtest@rorth.T)
    hard=np.array([r2(ytest,xtest@np.asarray(row['R']).T) for row in results['seed_results']])
    linear=Ridge(alpha=1.,fit_intercept=True).fit(xtr,ytr)
    lin_r2=r2(ytest,linear.predict(xtest))
    # Independent training-only selection of lag and regularization
    lag=selected['lag_bins']; alpha=selected['ridge_alpha']
    xlag,ylag=paired_at_lag(a,b,trtrial,lag)
    xlagtest,ylagtest=paired_at_lag(a,b,tetrial,lag)
    best=Ridge(alpha=alpha).fit(xlag,ylag)
    best_test=r2(ylagtest,best.predict(xlagtest))
    # Canonical correlations have no relation to cause, just linear association
    cors=canonical_cors(xtr,ytr)
    cors_test=canonical_cors(xtest,ytest)
    crosscorr=np.corrcoef(xtr.T,ytr.T)[:4,4:]
    # Model slopes and residual-by-target-dimension
    pred=linear.predict(xtest)
    r2dims=[r2(ytest[:,j:j+1],pred[:,j:j+1]) for j in range(4)]
    # compare within-trial fluctuations, using trial centering (separate test operation)
    a0=a-a.mean(axis=1,keepdims=True);b0=b-b.mean(axis=1,keepdims=True)
    xztr,yztr=paired_at_lag(a0,b0,trtrial,0)
    xzte,yzte=paired_at_lag(a0,b0,tetrial,0)
    dz=Ridge(alpha=1.).fit(xztr,yztr)
    within_r2=r2(yzte,dz.predict(xzte))
    # geometry check: sorted train covariance spectra and domain-wise cross-correlation
    ex=np.linalg.eigvalsh(np.cov(xtr.T))[::-1]; ey=np.linalg.eigvalsh(np.cov(ytr.T))[::-1]
    rep={
        'dataset':'DANDI:000138','method':'frozen 4D features, no re-fit HiWA',
        'bin_ms':20,'lag_convention':'positive lag: neural(t) predicts hand(t+lag)',
        'fit_trials':int(fitmask.sum()),'validation_trials':int(valmask.sum()),'test_trials':int(tetrial.sum()),
        'baseline_identity_R2':id_r2,'orthogonal_train_supervised_R2':orth_r2,
        'hard_hiwa_R2_seeds':hard.tolist(),'hard_hiwa_R2_mean':float(hard.mean()),
        'full_linear_ridge_zero_lag_R2':lin_r2,'full_linear_ridge_zero_lag_per_target_dimension_R2':r2dims,
        'train_and_test_canonical_correlations':{'train':cors.tolist(),'test_descriptive':cors_test.tolist()},
        'neural_target_crosscorr_train':crosscorr.tolist(),
        'neural_eigen_spectrum_train':ex.tolist(),'hand_eigen_spectrum_train':ey.tolist(),
        'within_trial_demeaned_ridge_zero_lag_R2':within_r2,
        'lag_selection_train_validation_best':selected,
        'selected_lag_heldout_R2':best_test,
        'lag_curve_exploratory_with_test_scores':evaluations,
        'notes':['Only validation on training trials selects lag and Ridge alpha. Test-set per-lag curve is an exploratory diagnostic not a valid selection criterion.',
                 'Linear Ridge is a supervised upper benchmark for representational adequacy, not a run of HiWA.',
                 'R² is negative if held-out squared error exceeds constant held-out target mean.',
                 'Trial demeaned result removes between-trial offsets; per-trial centering in test is a diagnostic, not a causal decoder.']}
    (out/'diagnosis.json').write_text(json.dumps(rep,indent=2,ensure_ascii=False)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    xx=np.array([v['lag_ms'] for v in evaluations]);
    axes[0].plot(xx,[v['validation_R2'] for v in evaluations],marker='o',ms=3,label='Ridge validation, train split')
    axes[0].plot(xx,[v['test_ridge_R2_exploratory'] for v in evaluations],ls=':',label='Ridge test (descriptive)')
    axes[0].axvline(selected['lag_ms'],linestyle='--',color='gray',label='Chosen by validation')
    axes[0].set(xlabel='hand time - neural time (ms)',ylabel='R²',title='Cross-lag evaluation of 4D neural features')
    axes[0].legend(fontsize=7);axes[0].grid(alpha=.2)
    labs=['Identity','Hard HiWA avg','Orthogonal supervised','Full linear Ridge'];vals=[id_r2,float(hard.mean()),orth_r2,lin_r2]
    axes[1].bar(labs,vals)
    axes[1].set(title='Held-out R², same-time neural-to-hand',ylabel='Test R²')
    axes[1].tick_params(axis='x',labelrotation=24)
    axes[1].axhline(0,color='black',lw=.7);axes[1].grid(alpha=.2,axis='y')
    fig.savefig(out/'diagnostic_linear_vs_orthogonal.png',dpi=170,bbox_inches='tight');plt.close(fig)
    print(json.dumps({k:v for k,v in rep.items() if k in ['baseline_identity_R2','orthogonal_train_supervised_R2','hard_hiwa_R2_mean','full_linear_ridge_zero_lag_R2','within_trial_demeaned_ridge_zero_lag_R2','lag_selection_train_validation_best','selected_lag_heldout_R2','train_and_test_canonical_correlations']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
