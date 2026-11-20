"""First real-stock Citi Bike citywide fixed-R hierarchical OT pilot.

Source: CityBikes genuine June 2026 station inventory observations.
Targets: ASSUMED half-full station policy, NOT observed operational demand.
Balanced policy: deterministic proportional-served deficits; unmet demand is
explicitly counted. Identical source / served-target integer marginals used
for all methods (including exact min-cost flow global optimum).
Fixed geography, R=I. Great-circle km proxy, not real dispatch road cost.
Hard / Soft use same 8 training-independent spatial KMeans centers.
Soft has 2 positive memberships per station; Hard one-hot.
Hierarchical solution: for each k,l, solve conditional entropic OT Q_kl;
then solve group-level linear OT P on exact weighted conditional costs.
Gamma_ij=sum_kl P_kl Q_kl[i,j] with correct physical row/col marginals.
Both are approximate vs exact flat min-cost flow, never claim Soft beats
global min cost on the same objective.
Flat entropic Sinkhorn (= WA with R fixed to identity) is a baseline.
SA/DAD are not original dispatch methods and must not be misrepresented.
"""
from __future__ import annotations
import argparse, time, json, csv
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.special import softmax
from sklearn.cluster import KMeans
from ortools.graph.python import min_cost_flow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preflight_actual_inventory_snapshots import getdata, SNAPSHOTS


def dist_km(latx,lonx,laty,lony):
    earth=6371.0088
    x=np.deg2rad(latx)[:,None]
    y=np.deg2rad(laty)[None,:]
    dl=np.deg2rad(laty)[None,:]-x
    dlon=np.deg2rad(lony)[None,:]-np.deg2rad(lonx)[:,None]
    a=np.sin(dl/2)**2+np.cos(x)*np.cos(y)*np.sin(dlon/2)**2
    return 2*earth*np.arcsin(np.sqrt(np.clip(a,0.,1.)))


def balanced_service(demand,actual_supply):
    """Serve a proportional integer quota; unmet demand remains explicit."""
    count=int(actual_supply.sum())
    gross=int(demand.sum())
    if gross<count: raise ValueError("pilot assumes deficit>=supply")
    scaled=demand.astype(float)*count/gross
    served=np.floor(scaled).astype(np.int64)
    extra=count-int(served.sum())
    # Stable arg-sort, deterministic tie resolution by sorted station identifier.
    order=np.argsort(-(scaled-served),kind="stable")
    served[order[:extra]]+=1
    assert np.all(served<=demand) and np.all(served>=0)
    assert served.sum()==count
    return served


def finite_soft_kernel_transport(a,b,c,reg,maxiter=500,tol=1e-5):
    """Positive entropy-regularized independent coupling, normalized total=1.

    Exponent shift is global; support cost ranges may be tens of km, so use
    reg>=2 km for numerical robustness and reject non-convergent results.
    """
    assert abs(a.sum()-1)<1e-7 and abs(b.sum()-1)<1e-7
    shift=float(c.min())
    kernel=np.exp(-(c-shift)/reg)
    assert np.all(kernel>0),("regularization too low",float(c.max()-shift))
    v=np.ones(len(b))
    for it in range(maxiter):
        u=a/np.maximum(kernel@v,1e-300)
        v=b/np.maximum(kernel.T@u,1e-300)
        if (it+1)%25==0 or it==maxiter-1:
            q=(u[:,None]*kernel)*v[None,:]
            err=max(float(np.max(np.abs(q.sum(1)-a))),
                    float(np.max(np.abs(q.sum(0)-b))))
            if err<tol: return q,{"max_marginal_error":err,"iterations":it+1}
    raise RuntimeError(f"Sinkhorn nonconvergence at reg {reg}, err={err:.3g}, maxiter={maxiter}")


def exact_transport_cost_integer(a,b,c,return_edges=True):
    """OR-tools exact integer mincost flow on FULL bipartite geographic graph."""
    n,m=c.shape
    assert a.dtype.kind in "iu" and b.dtype.kind in "iu"
    assert a.sum()==b.sum()>0
    solver=min_cost_flow.SimpleMinCostFlow()
    unit=np.maximum(0,np.rint(c*1000).astype(np.int64))
    for i in range(n):
        ii=int(i)
        # all shipment options allowed, no k-nearest sparse shortcut
        for j in range(m):
            solver.add_arc_with_capacity_and_unit_cost(
                ii,int(n+j),int(min(a[i],b[j])),int(unit[i,j]))
    for i in range(n):solver.set_node_supply(int(i),int(a[i]))
    for j in range(m):solver.set_node_supply(int(n+j),int(-b[j]))
    start=time.monotonic()
    status=solver.solve()
    elapsed=time.monotonic()-start
    assert status==solver.OPTIMAL,("network simplex failed",status)
    result={"cost_integer_km":float(solver.optimal_cost()/1000.),
            "full_bipartite_arcs":n*m,"runtime_solver_s":elapsed}
    flow=np.zeros((n,m),dtype=np.int32)
    if return_edges:
        for k in range(solver.num_arcs()):
            value=solver.flow(k)
            if value>0:
                i=solver.tail(k);j=solver.head(k)-n
                flow[i,j]=value
        assert np.array_equal(flow.sum(1),a)
        assert np.array_equal(flow.sum(0),b)
        result["nonzero_dispatch_routes"]=int(np.count_nonzero(flow))
    return result,flow


def greedy_transport(a,b,c):
    """Feasible nearest-outstanding-demand greedy with fixed station order."""
    remain=b.copy()
    flow=np.zeros(c.shape,dtype=np.int32)
    for i in np.argsort(-a,kind="stable"):
        need=int(a[i])
        for j in np.argsort(c[i],kind="stable"):
            move=min(need,int(remain[j]))
            if move:
                flow[i,j]+=move;remain[j]-=move;need-=move
            if need==0:break
        assert need==0
    assert not np.any(remain)
    return flow


def group_assignments(xy,K):
    allpts=np.asarray(xy)
    fit=KMeans(n_clusters=K,random_state=42,n_init=15).fit(allpts)
    z=fit.transform(allpts)
    # KMeans distances squared in km units.
    squared=z**2
    zsorted=np.sort(squared,axis=1)
    tau=float(np.median(zsorted[:,1]-zsorted[:,0]))
    assert tau>0
    # Top two only: soft overlaps are measurable but local; exact zeros
    # make the dense conditional Q computations practical and interpretable.
    idx=np.argpartition(squared,2,axis=1)[:,:2]
    A=np.zeros((len(xy),K),dtype=float)
    rows=np.arange(len(xy))[:,None]
    A[rows,idx]=softmax(-squared[rows,idx]/tau,axis=1)
    H=np.eye(K)[fit.labels_]
    assert np.allclose(A.sum(1),1)
    return H,A,fit.cluster_centers_,tau


def hierarchical(a,b,c,Ax,By,regularization=2.5):
    K=Ax.shape[1]
    assert K==By.shape[1]
    source_grp=a[:,None]*Ax
    target_grp=b[:,None]*By
    alpha=source_grp.sum(0)
    beta=target_grp.sum(0)
    assert all(alpha>0) and all(beta>0)
    normalized_a=source_grp/alpha[None,:]
    normalized_b=target_grp/beta[None,:]
    # Each Q^(k,l) independent of group coupling P for pure-cost+conditional
    # entropic objective; its marginal constraints are locally checked.
    Q={}; C=np.empty((K,K)); iters=[]
    start=time.monotonic()
    for k in range(K):
        isrc=np.flatnonzero(normalized_a[:,k]>1e-14)
        ak=normalized_a[isrc,k]
        ak/=ak.sum()
        for l in range(K):
            itgt=np.flatnonzero(normalized_b[:,l]>1e-14)
            bl=normalized_b[itgt,l]
            bl/=bl.sum()
            q,meta=finite_soft_kernel_transport(
                ak,bl,c[np.ix_(isrc,itgt)],
                reg=regularization,maxiter=650,tol=3e-6)
            Q[k,l]=(isrc,itgt,q)
            C[k,l]=np.sum(q*c[np.ix_(isrc,itgt)])
            iters.append(meta)
    # Linear group transportation P, nonuniform actual group mass in bikes,
    # never the inadmissible old row/col = 1/K margins.
    from scipy.sparse import lil_matrix
    E=lil_matrix((2*K,K*K),dtype=float)
    for k in range(K):
        for l in range(K):
            E[k,k*K+l]=1.
            E[K+l,k*K+l]=1.
    res=linprog(C.ravel(),A_eq=E.tocsr(),b_eq=np.r_[alpha,beta],
                bounds=(0,None),method="highs",
                options={"primal_feasibility_tolerance":1e-8})
    assert res.success,res.message
    P=res.x.reshape(K,K)
    g=np.zeros((len(a),len(b)),dtype=float)
    for k in range(K):
        for l in range(K):
            ids,jds,q=Q[k,l]
            g[np.ix_(ids,jds)]+=P[k,l]*q
    merr=max(np.max(abs(g.sum(1)-a)),np.max(abs(g.sum(0)-b)))
    assert merr<.008,("hierarchical Gamma marginal error",merr)
    summary={"hier_transport_km":float(np.sum(g*c)),
             "P_shape":[K,K],
             "P_row_max_error":float(np.max(abs(P.sum(1)-alpha))),
             "P_col_max_error":float(np.max(abs(P.sum(0)-beta))),
             "Gamma_max_marginal_error_bikes":float(merr),
             "q_max_conditional_marginal_error":max(x["max_marginal_error"] for x in iters),
             "q_max_iters":max(x["iterations"] for x in iters),
             "q_mean_iters":float(np.mean([x["iterations"] for x in iters])),
             "computation_time_s":time.monotonic()-start,
             "P":P.tolist(),"C_conditional_km":C.tolist(),
             "group_source_bikes":alpha.tolist(),
             "group_target_bikes":beta.tolist(),
             "Q_conditioned_included_in_artifact":True,
             "Gamma_fractional_not_yet_executable_integer_routes":True}
    return g,summary,Q


def plot_scene(out,name,locations,sources,targets,c,items):
    lat=np.array([x["lat"] for x in locations])
    lon=np.array([x["lon"] for x in locations])
    src=np.flatnonzero(sources>0);tgt=np.flatnonzero(targets>0)
    fig,axs=plt.subplots(1,2,figsize=(15,7),layout="constrained")
    for ax in axs:
        ax.scatter(lon,lat,s=3,c="lightgray",alpha=.36,label="All stations")
        ax.scatter(lon[src],lat[src],s=7+sources[src]*1.7,
                   alpha=.52,c="#2a77b7",label="Supply stations")
        ax.scatter(lon[tgt],lat[tgt],s=7+targets[tgt]*1.7,
                   alpha=.50,c="#d67637",label="Served deficit stations")
        ax.set_xlabel("Longitude");ax.set_ylabel("Latitude")
        ax.set_aspect(1./np.cos(np.deg2rad(lat.mean())))
        ax.grid(alpha=.2)
    axs[0].set_title("Observed stock, assumed half-full policy: supply vs served deficit")
    axs[0].legend(fontsize=9)
    # Compare transportation arrows by the largest flow mass, do not plot
    # every entropic edge and imply its fractional dust is a vehicle shipment.
    for label,flow,colour in items:
        ff=flow.copy()
        ii,jj=np.unravel_index(np.argsort(ff.ravel())[-90:],ff.shape)
        for i,j in zip(ii,jj):
            if ff[i,j]<=0:continue
            xs=lon[src[i]];xt=lon[tgt[j]]
            ys=lat[src[i]];yt=lat[tgt[j]]
            axs[1].plot([xs,xt],[ys,yt],
                        color=colour,alpha=.17,lw=min(4.,.25+float(ff[i,j])*.055))
    axs[1].set_title("Largest 90 arrows of each transport plan (overlaid)")
    for label,flow,colour in items:
        axs[1].plot([],[],color=colour,lw=2,label=label)
    axs[1].legend(fontsize=9)
    fig.suptitle(f"REAL NYC Citi Bike historical stock {name} | all transport needs are POLICY assumptions",fontsize=13)
    fig.savefig(out/f"{name.replace(' ','_')}_SUPPLY_AND_TRANSPORT.png",dpi=165)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--snapshots",nargs="+",type=int,default=[0])
    parser.add_argument("--K",nargs="+",type=int,default=[4,8])
    parser.add_argument("--max-nodes",type=int,default=0,
        help="0=full citywide nonzero supply+served deficit, >0=balanced subpilot")
    parser.add_argument("--reg",type=float,default=2.5)
    args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    groups,meta=getdata(args.input)
    runs=[]
    for it in args.snapshots:
        time_start=time.monotonic()
        data=groups[it]
        loc=[data[x] for x in sorted(data)]
        B=np.array([r["bikes"] for r in loc],dtype=np.int64)
        T=np.array([r["slots"]//2 for r in loc],dtype=np.int64)
        surplus=np.maximum(B-T,0)
        deficit=np.maximum(T-B,0)
        if args.max_nodes:
            # Subpilot deterministic by largest nonzero masses, re-balance ONLY
            # this subproblem, not falsely claim full citywide optimality.
            sid=np.argsort(-surplus,kind="stable")[:args.max_nodes]
            did=np.argsort(-deficit,kind="stable")[:args.max_nodes]
            omit=np.ones(len(loc),bool);omit[np.union1d(sid,did)]=False
            surplus[omit]=0;deficit[omit]=0
        gross_supply=int(surplus.sum());gross_deficit=int(deficit.sum())
        if gross_supply<=0 or gross_deficit<=gross_supply:
            raise ValueError(f"Expected shortage for this scenario: {gross_supply},{gross_deficit}")
        served=balanced_service(deficit,gross_supply)
        si=np.flatnonzero(surplus>0)
        di=np.flatnonzero(served>0)
        a=surplus[si];b=served[di]
        assert a.sum()==b.sum()==gross_supply
        source=[loc[i] for i in si];target=[loc[j] for j in di]
        latx=np.array([v["lat"] for v in source])
        lonx=np.array([v["lon"] for v in source])
        laty=np.array([v["lat"] for v in target])
        lony=np.array([v["lon"] for v in target])
        c=dist_km(latx,lonx,laty,lony)
        assert c.min()>=0 and c.shape==(len(a),len(b))
        city=np.array([r["lat"] for r in loc]);lon=np.array([r["lon"] for r in loc])
        xy=np.column_stack([
           (lon-lon.mean())*111.32*np.cos(np.deg2rad(city.mean())),
           (city-city.mean())*111.32])
        # Greedy, exact flat integer optimum first.
        t0=time.monotonic()
        exact,flow_exact=exact_transport_cost_integer(a,b,c)
        exact["runtime_total_s"]=time.monotonic()-t0
        greedy=greedy_transport(a,b,c)
        greedy_cost=float(np.sum(greedy*c))
        assert greedy_cost>=exact["cost_integer_km"]-.05
        # Flat entropic WA-I with same epsilon
        t0=time.monotonic()
        flat_q,meta_flat=finite_soft_kernel_transport(
            a/a.sum(),b/b.sum(),c,reg=args.reg,maxiter=750,tol=2e-6)
        flat_g=flat_q*a.sum()
        flat_err=max(np.max(abs(flat_g.sum(1)-a)),np.max(abs(flat_g.sum(0)-b)))
        assert flat_err<.012
        flat_cost=float(np.sum(flat_g*c))
        assert flat_cost>=exact["cost_integer_km"]-.2
        snapshot={
            "raw_inventory_snapshot_timestamp":SNAPSHOTS[it],
            "timestamp_timezone":"Unknown in source, no local-time claim",
            "real_inventory_station_count":len(loc),
            "supply_station_count":len(si),
            "served_demand_station_count":len(di),
            "gross_policy_supply_bikes":gross_supply,
            "gross_policy_deficit_bikes":gross_deficit,
            "served_policy_deficit_bikes":int(served.sum()),
            "unserved_policy_deficit_bikes":int(deficit.sum()-served.sum()),
            "allocation_policy":"Proportional integer fair-share of each policy deficit, largest remainder; fixed across ALL algorithms",
            "station_target":"ASSUMED half dock capacity, NOT operator-defined target",
            "position":"Geographic lat-lon; no rotation; R=I",
            "cost":"Great-circle distance in km, NOT road driving time or emissions",
            "exact_flat_integer_min_cost_flow":exact,
            "greedy_cost_km":greedy_cost,
            "flat_sinkhorn_WA_identity":{"cost_km":flat_cost,
                     "reg_km":args.reg,"max_marginal_error_bikes":float(flat_err),
                     **meta_flat,
                     "methodology":"WA with fixed R=I is simply flat entropy-regularized OT"},
            "SA_status":"SA is geometry subspace alignment; NOT directly a city shipping-plan method",
            "DAD_status":"DAD is geometric density alignment; NOT directly a city shipping-plan method",
            "models":{},"elapsed_s":None}
        for K in args.K:
            hard,soft,centers,tau=group_assignments(xy,K)
            for method,A in [("Hard",hard),("Soft",soft)]:
                g,res,Q=hierarchical(a,b,c,A[si],A[di],regularization=args.reg)
                res["fractional_cost_gap_vs_exact_percent"]=100*(res["hier_transport_km"]-exact["cost_integer_km"])/exact["cost_integer_km"]
                res["K"]=K
                res["soft_temperature_sq_km"]=tau if method=="Soft" else None
                assert res["fractional_cost_gap_vs_exact_percent"]>=-.01
                key=f"{method}_K{K}"
                snapshot["models"][key]=res
                np.savez_compressed(out/f"{it}_{key}_P_GAMMA_Q.npz",
                    P=np.array(res["P"]),Gamma=g,
                    source_id=np.array([v["nuid"] for v in source]),
                    target_id=np.array([v["nuid"] for v in target]),
                    source_supply_bikes=a,target_served_demand_bikes=b,
                    source_lat=latx,source_lon=lonx,target_lat=laty,target_lon=lony,
                    conditional_C_km=np.array(res["C_conditional_km"]),
                    group_weights_source=A[si],group_weights_target=A[di],
                    **{f"Q_{k}_{l}":Q[k,l][2] for k in range(K) for l in range(K)},
                    **{f"src_ids_Q_{k}_{l}":Q[k,l][0] for k in range(K) for l in range(K)},
                    **{f"tgt_ids_Q_{k}_{l}":Q[k,l][1] for k in range(K) for l in range(K)})
                print("OT_RESULT",it,key,"cost",round(res["hier_transport_km"],2),
                      "gap_pct",round(res["fractional_cost_gap_vs_exact_percent"],3),
                      "rowerr",res["Gamma_max_marginal_error_bikes"],
                      "time",round(res["computation_time_s"],1),flush=True)
                (out/"CITIBIKE_FIRST_ACTUAL_OT_RESULTS.json").write_text(
                    json.dumps({"runs":runs+[snapshot],"in_progress":True},indent=2,ensure_ascii=False))
            if K==args.K[0]:
                plot_scene(out,f"June15_{SNAPSHOTS[it][-8:].replace(':','_')}",
                    loc,surplus,served,c,
                    [("Exact flat",flow_exact,"#777777"),
                     ("Hard",np.load(out/f"{it}_Hard_K{K}_P_GAMMA_Q.npz")["Gamma"],"#2a77b7"),
                     ("Soft",np.load(out/f"{it}_Soft_K{K}_P_GAMMA_Q.npz")["Gamma"],"#d77633")])
        snapshot["elapsed_s"]=time.monotonic()-time_start
        runs.append(snapshot)
        (out/"CITIBIKE_FIRST_ACTUAL_OT_RESULTS.json").write_text(
            json.dumps({"runs":runs,"in_progress":False,
                        "study_status":"first real-stock, assumed-target, fixed-deficit-service experimental PILOT",
                        "future_work":"validate timestamp, historical operational targets, road routing, and integerize hierarchical outputs; don't claim saved operational cost"},indent=2,ensure_ascii=False))
        print("COMPLETE_STOCK_SNAPSHOT",it,
              "exact",exact["cost_integer_km"],
              "greedy",round(greedy_cost,2),
              "flat_entropy",round(flat_cost,2),
              "elapsed",round(snapshot["elapsed_s"],1),flush=True)
    # finalize summary readable as a table
    lines=["# Citi Bike NYC: FIRST actual baseline vs Hard/Soft hierarchical OT pilot",
           "","**Real historical station inventory; 50% dock target is a POLICY ASSUMPTION, not measured demand or actual operator dispatch.**",
           "Source: CityBikes NYC 2026-06 status Parquet, exact file hash in input audit.",
           "No learned geometric R; fix R=I. Great-circle km are proxy logistics costs, not road distance.",
           "All algorithms share the same selected integer serviced deficits; remaining deficits are explicit, not silently normalized.",
           "Hard/Soft Q and Gamma are **fractional** entropy-regularized transport and need integerization before a physical vehicle dispatch.",
           "Flat exact mincost uses an unrestricted full bipartite network and integer flows; this is the lower-bound reference under these policy assumptions.",
           "WA-I is exactly the shown flat Sinkhorn; SA and DAD are not applicable without new explicitly adapted definitions.",
           "",
           "| Snapshot | Method | K | Pure transport cost (bike-km) | gap vs flat exact | Row/column valid |",
           "|---|---|---:|---:|---:|---|"]
    for run in runs:
        name=run["raw_inventory_snapshot_timestamp"]
        base=run["exact_flat_integer_min_cost_flow"]["cost_integer_km"]
        lines.append(f"| {name} | Exact flat min-cost flow | – | {base:.2f} | 0.00% | integer exact |")
        lines.append(f"| {name} | Greedy feasible | – | {run['greedy_cost_km']:.2f} | {100*(run['greedy_cost_km']/base-1):.2f}% | integer exact |")
        fl=run["flat_sinkhorn_WA_identity"]
        lines.append(f"| {name} | Flat Sinkhorn / WA-I | – | {fl['cost_km']:.2f} | {100*(fl['cost_km']/base-1):.2f}% | fractional approx |")
        for method,d in run["models"].items():
            lines.append(f"| {name} | {method.split('_')[0]} | {d['K']} | {d['hier_transport_km']:.2f} | {d['fractional_cost_gap_vs_exact_percent']:.2f}% | fractional, max marginal err {d['Gamma_max_marginal_error_bikes']:.4g} bike |")
    lines.extend(["",
                  "**Interpretation:** Flat exact must be <= any valid fractional plan on the same cost, to solver/roundoff tolerance. "
                  "Better Hard/Soft plan is a lower P/Q approximation gap, not better than the true flat optimum.",
                  "These are two snapshots of one day in a single month, not independent monthly generalization.",
                  "Source and target sites are never transformed in physical geography.",
                  "SA and DAD cannot be compared as original alignment algorithms on this dispatch cost.",
                  "",
                  "Raw transport arrays and 16 (or K²) conditional plans saved in separate NPZ run artifacts; JSON stores P and metrics.",
                  ""])
    (out/"CITIBIKE_FIRST_ACTUAL_OT_REPORT.md").write_text("\n".join(lines),encoding="utf8")
    print("FIRST_CITIBIKE_REAL_STOCK_OT_FINISHED",flush=True)

if __name__=="__main__":
    main()
