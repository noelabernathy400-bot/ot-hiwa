"""Train-domain-only, independently computed projection utilities for HiWA.

Original MATLAB HiWA.m constructs each group-specific projected feature:
    X_proj[group i] = X_i @ (A_i A_i.T)
and same for Y. This port previously advertised a projection hook in
GCOT/TPAMIGCOT but omitted the module entirely (ModuleNotFoundError).
This module repairs that OPTIONAL path without changing archived defaults.

Do NOT infer a target projector or group identity from synchronized source
labels. Group projector training is always independent per domain.
"""
from __future__ import annotations

import itertools
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def _operator(value,d):
    a=np.asarray(value,dtype=float)
    if a.ndim!=2 or a.shape[0]!=d:
        raise ValueError(f"Projector/basis must have {d} rows.")
    if a.shape[1]==d and np.allclose(a,a.T,atol=1e-8) and np.allclose(a@a,a,atol=1e-8):
        return a
    # Supplied D x r basis (including a square basis) -> projection onto span.
    u,s,_=np.linalg.svd(a,full_matrices=False)
    rank=int(np.sum(s>max(s[0],1)*1e-11))
    if rank==0:raise ValueError("Empty projector span.")
    return u[:,:rank]@u[:,:rank].T

def project_for_alignment(x,y,projections):
    """Apply explicitly supplied, independent domain projectors.

    projections either (Px,Py) or {"source": Px, "target": Py},
    each a D x rank basis or D x D symmetric idempotent projector.
    It does not fit projectors or inspect true cross-domain group labels.
    Returns two N x D arrays in unchanged ambient dimensionality.
    """
    x=np.asarray(x,float);y=np.asarray(y,float)
    if x.ndim!=2 or y.ndim!=2 or x.shape[1]!=y.shape[1]:
        raise ValueError("Expect two sample-by-ambient-d arrays.")
    if isinstance(projections,dict):
        if set(projections)!={"source","target"}:
            raise ValueError("Projections dict needs exactly source and target")
        px,py=projections["source"],projections["target"]
    elif isinstance(projections,(tuple,list)) and len(projections)==2:
        px,py=projections
    else:
        raise TypeError("Projection argument must be two independently specified matrices")
    X=x@_operator(px,x.shape[1])
    Y=y@_operator(py,y.shape[1])
    if not (np.isfinite(X).all() and np.isfinite(Y).all()):
        raise FloatingPointError("Non-finite projected features")
    return X,Y

def independent_group_projectors(x,labels,rank,fit_mask=None):
    """Original groupwise A_i A_i.T low-rank geometry, separately on ONE domain.

    Subspaces pass through origin as in MATLAB HiWA.m. Group ids are used
    only within this domain. fit_mask fits each basis on TRAIN rows only;
    same basis projects its corresponding test rows after training.
    Returns projected data and the actual per-group projectors.
    """
    x=np.asarray(x,float)
    labels=np.asarray(labels)
    mask=np.ones(len(x),bool) if fit_mask is None else np.asarray(fit_mask,bool)
    if x.ndim!=2 or labels.shape!=(len(x),) or mask.shape!=(len(x),):
        raise ValueError("Mismatched group projection input sizes")
    if not (1<=rank<=x.shape[1]):
        raise ValueError("rank out of bounds")
    out=np.empty_like(x)
    projectors={}
    for k in np.unique(labels):
        rows=(labels==k)
        tr=rows&mask
        if tr.sum()<rank:
            raise ValueError(f"Too few training samples in group {k}")
        # NO centering: original MATLAB uses A_i A_i^T X_i
        _,_,vh=np.linalg.svd(x[tr],full_matrices=False)
        b=vh[:rank].T
        p=b@b.T
        out[rows]=x[rows]@p
        projectors[int(k)]=p
    assert np.isfinite(out).all()
    return out,projectors

def independent_covariance_sign_candidates(x,y,fit_mask_x=None,fit_mask_y=None):
    """Rotation candidates from each TRAIN distribution alone, no pairing.

    For D dimensions, enumerate 2**D covariance eigenbasis sign choices.
    Sign enumeration is only suitable for small D with well-resolved
    covariance eigenvalues; no guarantee if eigenvalues degenerate.
    """
    x=np.asarray(x,float);y=np.asarray(y,float)
    if x.ndim!=2 or y.ndim!=2 or x.shape[1]!=y.shape[1]:
        raise ValueError("Both domains need same ambient feature dim")
    xx=x[np.asarray(fit_mask_x,bool)] if fit_mask_x is not None else x
    yy=y[np.asarray(fit_mask_y,bool)] if fit_mask_y is not None else y
    _,Ux=np.linalg.eigh(np.cov(xx,rowvar=False))
    _,Uy=np.linalg.eigh(np.cov(yy,rowvar=False))
    Ux=Ux[:,::-1];Uy=Uy[:,::-1]
    return [(tuple(signs),Uy@np.diag(signs)@Ux.T)
            for signs in itertools.product((-1.,1.),repeat=x.shape[1])]

def select_spectral_candidate_train_ot(candidates,x,y,ix,iy):
    """Select by unpaired TRAIN empirical squared W2; no physical ground truth."""
    a=x[np.asarray(ix,int)];b=y[np.asarray(iy,int)]
    costs=[]
    for _,R in candidates:
        D=cdist(a@R.T,b,"sqeuclidean")
        i,j=linear_sum_assignment(D)
        costs.append(float(np.mean(D[i,j])))
    choice=int(np.argmin(costs))
    return candidates[choice][1],dict(
        chosen_signs=list(candidates[choice][0]),chosen_index=choice,
        training_unpaired_W2sq=costs[choice],all_train_costs=costs)
