#!/usr/bin/env python3
"""Rectangular Kx-by-Ky group-OT primitive for auto-selected clusters.

This supports independently inferred Kx != Ky without leaking source cluster
IDs into the target side. The complete HiWA ADMM/inner Q solver is NOT
implemented here; only independent group construction and group-level P.

No labels, synchronized rows, or physical force annotations enter the model.
"""
from __future__ import annotations
import argparse
import numpy as np
from scipy.optimize import root
from scipy.special import softmax, logsumexp
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def independent_memberships(X,Y,Kx,Ky,method="hard",tau=.5,seed=42):
    X=np.asarray(X,float);Y=np.asarray(Y,float)
    if X.ndim!=2 or Y.ndim!=2 or X.shape[1]!=Y.shape[1]:
        raise ValueError("Two domains must be same feature dimension")
    if not (2<=Kx<len(X) and 2<=Ky<len(Y)):
        raise ValueError("Invalid group counts")
    cx=KMeans(n_clusters=Kx,n_init=20,random_state=seed).fit(X)
    cy=KMeans(n_clusters=Ky,n_init=20,random_state=seed+1).fit(Y)
    if method=="hard":
        A=np.eye(Kx)[cx.labels_]
        B=np.eye(Ky)[cy.labels_]
    elif method=="soft":
        if tau<=0:raise ValueError("tau must be positive")
        A=softmax(-cdist(X,cx.cluster_centers_,metric="sqeuclidean")/tau,axis=1)
        B=softmax(-cdist(Y,cy.cluster_centers_,metric="sqeuclidean")/tau,axis=1)
    else:
        raise ValueError("method must be hard or soft")
    assert A.shape==(len(X),Kx) and B.shape==(len(Y),Ky)
    assert np.allclose(A.sum(1),1.) and np.allclose(B.sum(1),1.)
    return A,B,cx.cluster_centers_,cy.cluster_centers_


def group_marginals(A,B,mode="uniform_groups"):
    """Return group probabilities for P, not per-group normalized sample Q."""
    Kx=A.shape[1];Ky=B.shape[1]
    if mode=="uniform_groups":
        return np.ones(Kx)/Kx,np.ones(Ky)/Ky
    if mode=="empirical_point_mass":
        return A.mean(axis=0),B.mean(axis=0)
    raise ValueError("mode")


def group_entropic_transport(C,alpha,beta,eps=.2,tol=1e-7):
    """min_P <P,C> + eps*sum(P log P) with rectangular marginals.

    The 4x4 HiWA port used gauge-fixed root of the exact entropic dual;
    the same approach here works for rectangular 4x7 or 5x8 problems.
    """
    C=np.asarray(C,dtype=float)
    alpha=np.asarray(alpha,dtype=float)
    beta=np.asarray(beta,dtype=float)
    nx,ny=C.shape
    if C.ndim!=2 or alpha.shape!=(nx,) or beta.shape!=(ny,):
        raise ValueError("incompatible cost and marginals")
    if not np.isfinite(C).all() or np.any(alpha<=0) or np.any(beta<=0) or eps<=0:
        raise ValueError("invalid transport data")
    if abs(alpha.sum()-1)>1e-9 or abs(beta.sum()-1)>1e-9:
        raise ValueError("both marginals must sum to 1")
    # Row/column potentials can absorb these subtracted constants.
    c=C-C.min(axis=1,keepdims=True)
    c=c-c.min(axis=0,keepdims=True)

    def plan(z):
        f=z[:nx]
        g=np.r_[z[nx:],0.]
        return np.exp(np.clip((f[:,None]+g[None,:]-c)/eps,-700,700))

    def residual(z):
        P=plan(z)
        return np.r_[P.sum(axis=1)-alpha,P.sum(axis=0)[:ny-1]-beta[:ny-1]]

    def jacobian(z):
        P=plan(z);J=np.zeros((nx+ny-1,nx+ny-1))
        J[:nx,:nx]=np.diag(P.sum(axis=1)/eps)
        J[:nx,nx:]=P[:,:ny-1]/eps
        J[nx:,:nx]=P[:,:ny-1].T/eps
        J[nx:,nx:]=np.diag(P.sum(axis=0)[:ny-1]/eps)
        return J

    sol=root(residual,np.zeros(nx+ny-1),jac=jacobian,method="hybr",
             options={"xtol":1e-12,"maxfev":4000})
    P=plan(sol.x)
    err=max(np.max(abs(P.sum(1)-alpha)),np.max(abs(P.sum(0)-beta)))
    if err>tol:
        # Stable log-Sinkhorn for difficult costs.
        f=np.zeros(nx);g=np.zeros(ny)
        la=np.log(alpha);lb=np.log(beta)
        for it in range(30000):
            f=eps*(la-logsumexp((g[None,:]-c)/eps,axis=1))
            g=eps*(lb-logsumexp((f[:,None]-c)/eps,axis=0))
            if (it+1)%50==0:
                P=np.exp(np.clip((f[:,None]+g[None,:]-c)/eps,-700,700))
                err=max(np.max(abs(P.sum(1)-alpha)),np.max(abs(P.sum(0)-beta)))
                if err<tol:break
    if err>tol or not np.isfinite(P).all():
        raise ArithmeticError(f"Rectangular group-OT marginal error {err}")
    return P


def self_test():
    rng=np.random.default_rng(42)
    for nx,ny in ((4,4),(4,7),(5,8),(8,5)):
        C=rng.random((nx,ny))*3
        alpha=rng.dirichlet(np.ones(nx)*2)
        beta=rng.dirichlet(np.ones(ny)*2)
        for mode in ("uniform","empirical"):
            a,b=(np.full(nx,1/nx),np.full(ny,1/ny)) if mode=="uniform" else (alpha,beta)
            P=group_entropic_transport(C,a,b,eps=.2)
            err=max(abs(P.sum(1)-a).max(),abs(P.sum(0)-b).max())
            assert err<1e-7 and P.shape==(nx,ny)
            assert np.all(P>=0)
            print("PASS",mode,(nx,ny),"max_marginal_error",err)
    x=rng.normal(size=(200,3));y=rng.normal(size=(210,3))
    A,B,_,_=independent_memberships(x,y,4,7,"soft",tau=.5)
    alpha,beta=group_marginals(A,B,"empirical_point_mass")
    assert A.shape==(200,4) and B.shape==(210,7)
    P=group_entropic_transport(rng.random((4,7)),alpha,beta)
    assert abs(P.sum()-1)<1e-9
    print("All rectangular group P test cases passed (not a full HiWA solver)")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:self_test()
