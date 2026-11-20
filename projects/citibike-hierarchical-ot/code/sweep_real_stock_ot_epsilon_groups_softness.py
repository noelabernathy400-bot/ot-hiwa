"""Predeclared, DATA-backed Citi Bike actual-stock hierarchical OT parameter sweep.

This is a second-stage direct followup to the 2026-11-20 fixed-R=I
pilot. Uses the SAME genuine CityBikes 2026-06 stock snapshots and
ASSUMED half-dock policy; same deterministic proportional served quotas
and unserved demand record. Original real stock data is NOT operator
recommended dispatch.

Runs exact flat min-cost flow and Greedy once/snapshot; flat Sinkhorn
(=WA-I) and weighted hard/soft hierarchical P/Q at several entropy scales,
K and soft grouping temperature. Shared true actual station geodesic costs.
No learned orthogonal transformation. Original SA/DAD not applicable.
DO NOT claim a hierarchy beats globally optimal unregularized flat OT.

Experimental contrasts:
  (1) K=4/8, epsilon=[0.25,.5,1,2.5] on both independently observed
      snapshots, equal entropic conditional Q solver, same group P cost.
  (2) K=8, epsilon=.5, soft temperature multipliers [.1,.25,.5,1,2,4],
      with near-hard limit as a correctness check.
  (3) exact mass-conserving INTEGER rounding of selected Gamma using
      floor-Gamma plus residual integer min-cost flow, not merely a
      matrix of fractional routes.
Convergence failures are RECORDED, not replaced with convenient values.
All results and plots are saved in the dedicated Citi Bike project only.
"""
from __future__ import annotations
import argparse, csv, json, math, time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.special import logsumexp, softmax
from scipy.sparse import lil_matrix
from sklearn.cluster import KMeans
from ortools.graph.python import min_cost_flow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preflight_actual_inventory_snapshots import getdata, SNAPSHOTS
from run_first_real_stock_hierarchical_ot import (
    balanced_service, dist_km, exact_transport_cost_integer, greedy_transport
)

def sinkhorn_with_audit(a,b,c,eps,maxiter=2100,tol=2.5e-9):
    """Stable scalar-K Sinkhorn, log-domain rescue for eps small.
    a,b are probabilities with strictly positive support.
    """
    assert eps>0 and a.min()>0 and b.min()>0
    assert abs(a.sum()-1)<1e-8 and abs(b.sum()-1)<1e-8
    C=c-c.min()
    K=np.exp(-C/eps)
    # Fast path; if exp underflows, use log path instead.
    if np.all(K>0):
        u=np.ones_like(a)
        v=np.ones_like(b)
        for it in range(maxiter):
            div=K@v
            if div.min()<1e-280:break
            u=a/div
            div=K.T@u
            if div.min()<1e-280:break
            v=b/div
            if (it+1)%25==0 or it+1==maxiter:
                q=u[:,None]*K*v[None,:]
                err=max(float(np.max(abs(q.sum(axis=0)-b))),
                        float(np.max(abs(q.sum(axis=1)-a))))
                if err<=tol:
                    return q,{"iterations":it+1,"max_prob_marginal_error":err,
                              "solver":"scaling","converged":True}
    f=np.zeros(len(a))
    g=np.zeros(len(b))
    loga=np.log(a); logb=np.log(b)
    L=-C/eps
    for it in range(maxiter*2):
        f=loga-logsumexp(L+g[None,:],axis=1)
        g=logb-logsumexp(L+f[:,None],axis=0)
        if (it+1)%25==0 or it+1==maxiter*2:
            q=np.exp(L+f[:,None]+g[None,:])
            err=max(float(np.max(abs(q.sum(axis=0)-b))),
                    float(np.max(abs(q.sum(axis=1)-a))))
            if err<=tol:
                return q,{"iterations":it+1,"max_prob_marginal_error":err,
                          "solver":"log-domain","converged":True}
    return None,{"iterations":maxiter*2,"max_prob_marginal_error":err,
                 "solver":"log-domain","converged":False}

def geo_groups(xy,K):
    km=KMeans(n_clusters=K,random_state=42,n_init=15).fit(xy)
    sq=km.transform(xy)**2
    sorted_sq=np.sort(sq,axis=1)
    tau_base=float(np.median(sorted_sq[:,1]-sorted_sq[:,0]))
    assert tau_base>0
    hard=np.eye(K)[km.labels_]
    top2=np.argpartition(sq,2,axis=1)[:,:2]
    return sq,top2,hard,tau_base

def softly_assigned(sq,top2,K,tau):
    row=np.arange(len(sq))[:,None]
    result=np.zeros((len(sq),K))
    result[row,top2]=softmax(-sq[row,top2]/tau,axis=1)
    assert np.allclose(result.sum(1),1,atol=1e-12)
    return result

def conditional_Q_pairwise(a,b,C,A,B,epsilon,save_plan=False):
    """Group masses in ACTUAL bike units, normalized Q per conditional pair."""
    k=A.shape[1]
    ga=a[:,None]*A
    gb=b[:,None]*B
    alpha=ga.sum(0);beta=gb.sum(0)
    if np.min(alpha)<.01 or np.min(beta)<.01:
        return None,{"converged":False,"reason":"group with near-zero supply/target mass"}
    q_cache=[]
    cost=np.zeros((k,k))
    q_iter=0
    max_prob_err=0.
    tstart=time.perf_counter()
    for i in range(k):
        ii=np.flatnonzero(ga[:,i]>1e-12)
        ai=ga[ii,i]/alpha[i]
        ai=ai/ai.sum()
        for j in range(k):
            jj=np.flatnonzero(gb[:,j]>1e-12)
            bj=gb[jj,j]/beta[j]
            bj=bj/bj.sum()
            subC=C[np.ix_(ii,jj)]
            q,md=sinkhorn_with_audit(ai,bj,subC,epsilon)
            if q is None:
                return None,{"converged":False,
                    "reason":f"conditional Q[{i},{j}] numerical failure",
                    "solver":md,"K":k,"epsilon":epsilon}
            cost[i,j]=float(np.sum(q*subC))
            q_iter=max(q_iter,md["iterations"])
            max_prob_err=max(max_prob_err,md["max_prob_marginal_error"])
            if save_plan:q_cache.append((i,j,ii,jj,q))
    E=lil_matrix((2*k,k*k),dtype=float)
    for i in range(k):
        for j in range(k):
            E[i,i*k+j]=1.
            E[k+j,i*k+j]=1.
    res=linprog(cost.ravel(),A_eq=E.tocsr(),b_eq=np.r_[alpha,beta],
        bounds=(0,None),method="highs")
    if not res.success:
        return None,{"converged":False,"reason":"Group LP P failed: "+str(res.message)}
    P=res.x.reshape(k,k)
    # Sum_kl P_kl * expected conditional distance:
    total=float((P*cost).sum())
    meta={"converged":True,"K":k,"epsilon":epsilon,"cost_bike_km":total,
          "max_conditional_prob_margin_err":max_prob_err,
          "max_Q_iterations":q_iter,
          "source_group_mass_bikes":alpha.tolist(),
          "target_group_mass_bikes":beta.tolist(),
          "P_max_row_col_error_bikes":float(max(
            np.max(abs(P.sum(1)-alpha)),np.max(abs(P.sum(0)-beta)))),
          "solver_time_s":float(time.perf_counter()-tstart),
          "P":P.tolist(),
          "expected_transport_cost_C_km":cost.tolist(),
          "Q_local_saved_in_memory":save_plan}
    if not save_plan:return None,meta
    gamma=np.zeros(C.shape)
    for i,j,ii,jj,q in q_cache:
        gamma[np.ix_(ii,jj)]+=P[i,j]*q
    maxerr=max(np.max(abs(gamma.sum(1)-a)),np.max(abs(gamma.sum(0)-b)))
    meta["Gamma_max_marginal_error_bikes"]=float(maxerr)
    meta["Gamma_direct_cost_bike_km"]=float((gamma*C).sum())
    assert maxerr<.01,(maxerr,"bad mass")
    assert abs(meta["Gamma_direct_cost_bike_km"]-total)<.005
    return gamma,meta

def round_integer_preserving_exact_marginals(gamma,a,b,C,objective="min_cost"):
    """Exact integer station marginals, on floor/ceil of the fractional plan.

    'min_cost' heavily REOPTIMIZES nearly all (initially fractional) bikes;
    it is NOT faithful to HiWA/GCOT and cannot establish a Soft advantage.
    'min_L1' prioritizes minimizing transport matrix L1 difference and
    breaks nearly equal deviations by geographic transport cost.
    Report total variation of the integerized plan vs input Gamma.
    """
    floors=np.floor(gamma+1e-10).astype(np.int32)
    ra=a-floors.sum(1)
    rb=b-floors.sum(0)
    assert np.min(ra)>=0 and np.min(rb)>=0 and ra.sum()==rb.sum()
    n,m=gamma.shape
    solver=min_cost_flow.SimpleMinCostFlow()
    frac=gamma-floors
    edges=[]
    nz=np.nonzero(frac>1e-9)
    for i,j in zip(*nz):
        if objective=="min_cost":
            unit=int(round(float(C[i,j])*1000))
        elif objective=="min_L1":
            # Each selected rounded-UP edge changes L1 by (1-2*frac),
            # and the number of selected edges is fixed.  Thus minimizing
            # sum(1-frac) is equivalent to minimizing total L1 deviation.
            # 1e6 fractional multiplier; 1e-2 geographic cost tie-break.
            unit=int(round((1.-float(frac[i,j]))*1_000_000+
                           float(C[i,j])*100))
        else:raise ValueError(f"unsupported integer rounding {objective}")
        arc=solver.add_arc_with_capacity_and_unit_cost(
            int(i),int(n+j),1,unit)
        edges.append((arc,int(i),int(j)))
    for i in range(n):solver.set_node_supply(int(i),int(ra[i]))
    for j in range(m):solver.set_node_supply(int(n+j),int(-rb[j]))
    t0=time.perf_counter()
    status=solver.solve()
    secs=time.perf_counter()-t0
    if status!=solver.OPTIMAL:
        raise RuntimeError(f"rounding mincost flow failed, status {status}")
    out=floors.copy()
    for arc,i,j in edges:
        if solver.flow(arc)>0:out[i,j]+=1
    assert np.array_equal(out.sum(1),a)
    assert np.array_equal(out.sum(0),b)
    assert np.max(abs(out-gamma))<1.001
    l1=float(np.abs(out-gamma).sum())
    return out,{"rounding_objective":objective,
                "rounding_cost_bike_km":float((out*C).sum()),
                "rounding_time_s":secs,"integer_routes":int(np.count_nonzero(out)),
                "total_variation_fraction_of_mass":l1/(2.*a.sum()),
                "L1_coupling_change_bikes":l1,
                "max_pairwise_rounding_error_bikes":float(np.max(abs(out-gamma))),
                "residual_vehicles_rounded":int(ra.sum())}

def snapshot_data(raw,ident):
    loc=[raw[x] for x in sorted(raw)]
    stock=np.array([r["bikes"] for r in loc],dtype=int)
    target=np.array([r["slots"]//2 for r in loc],dtype=int)
    a0=np.maximum(stock-target,0)
    b0=np.maximum(target-stock,0)
    serve=balanced_service(b0,int(a0.sum()))
    ii=np.flatnonzero(a0>0); jj=np.flatnonzero(serve>0)
    a=a0[ii];b=serve[jj]
    assert a.sum()==b.sum()
    latx=np.array([loc[z]["lat"] for z in ii])
    lonx=np.array([loc[z]["lon"] for z in ii])
    laty=np.array([loc[z]["lat"] for z in jj])
    lony=np.array([loc[z]["lon"] for z in jj])
    C=dist_km(latx,lonx,laty,lony)
    lat=np.array([r["lat"] for r in loc]);lon=np.array([r["lon"] for r in loc])
    xy=np.stack([(lon-lon.mean())*111.32*np.cos(np.deg2rad(lat.mean())),
                 (lat-lat.mean())*111.32],axis=1)
    return {"snapshot":SNAPSHOTS[ident],"total_stations":len(loc),"source_stations":len(a),
            "target_stations":len(b),"stock_supply_bikes":int(a.sum()),
            "total_deficit_half_dock_bikes":int(b0.sum()),
            "served_half_dock_deficit_bikes":int(serve.sum()),
            "unserved_half_dock_deficit_bikes":int(b0.sum()-serve.sum()),
            "a":a,"b":b,"cost":C,"xy":xy,"ii":ii,"jj":jj}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--snapshots",nargs="+",type=int,default=[0,1])
    args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    data,metadata=getdata(args.input)
    eps_grid=[.25,.5,1.,2.5]
    K_grid=[4,8]
    tau_grid=[.1,.25,.5,1.,2.,4.]
    allrows=[]
    results={"experiment":"Citi Bike real stock hierarchy, FIXED R=I, physical half-dock ASSUMED",
             "eps_grid_km":eps_grid,"K_grid":K_grid,"soft_tau_multipliers":tau_grid,
             "snapshots":[],"validation_split_status":"two snapshots same day, exploratory not independent holdout",
             "flat_cost_lower_bound":"exact min cost same fixed integer served-deficit marginals",
             "SA_DAD_not_run":"not direct transport solvers; WA-I=flat Sinkhorn",
             "real_source":"CityBikes June 2026 real historical stock parquet, source SHA archived",
             "no_inferred_actual_operator_demand":True}
    for ident in args.snapshots:
        d=snapshot_data(data[ident],ident)
        a,b,C=d["a"],d["b"],d["cost"]
        print("SNAPSHOT",ident,d["snapshot"],"source",len(a),"demand",len(b),"bikes",int(a.sum()),flush=True)
        exact,flow=exact_transport_cost_integer(a,b,C)
        exact_cost=exact["cost_integer_km"]
        greedy=greedy_transport(a,b,C)
        greedy_cost=float((greedy*C).sum())
        assert greedy_cost>=exact_cost-.2
        rowbase={"snapshot_id":ident,"snapshot":d["snapshot"],"source_nodes":len(a),
                 "target_nodes":len(b),"bikes":int(a.sum()),
                 "exact_flat_integer_cost_bike_km":exact_cost,
                 "greedy_integer_cost_bike_km":greedy_cost}
        result={"snapshot_id":ident,"snapshot":d["snapshot"],"source_nodes":len(a),
                "target_nodes":len(b),"bikes":int(a.sum()),
                "unserved_demand_half_dock_policy":d["unserved_half_dock_deficit_bikes"],
                "exact_flat_integer":exact,
                "greedy_cost_bike_km":greedy_cost,
                "parameter_runs":[]}
        group_basis={}
        for K in K_grid:
            group_basis[K]=geo_groups(d["xy"],K)
        for eps in eps_grid:
            # Fresh full-domain entropy OT = WA-I (no rotation).
            q,sm=sinkhorn_with_audit(a/a.sum(),b/b.sum(),C,eps,maxiter=2200)
            if q is None:
                row={**rowbase,"configuration":"flat_sinkhorn_WA_I","K":0,
                     "epsilon_km":eps,"tau_multiplier":"","converged":False,
                     "reason":str(sm),"cost_bike_km":None,"gap_percent":None}
            else:
                cost=float(np.sum(q*C)*a.sum())
                row={**rowbase,"configuration":"flat_sinkhorn_WA_I","K":0,
                     "epsilon_km":eps,"tau_multiplier":"","converged":True,
                     "cost_bike_km":cost,"gap_percent":100*(cost/exact_cost-1),
                     "sinkhorn_iterations":sm["iterations"],
                     "prob_marginal_max_error":sm["max_prob_marginal_error"]}
            result["parameter_runs"].append(row);allrows.append(row)
            print("EPS",ident,"flat",eps,"cost",row.get("cost_bike_km"),
                  "good",row["converged"],flush=True)
            for K in K_grid:
                sq,nearest,hard,tau0=group_basis[K]
                for mode in ["Hard","Soft"]:
                    A=hard if mode=="Hard" else softly_assigned(sq,nearest,K,tau0)
                    gamma,md=conditional_Q_pairwise(a,b,C,A[d["ii"]],A[d["jj"]],eps)
                    row={**rowbase,"configuration":mode,"K":K,
                         "epsilon_km":eps,"tau_multiplier":0 if mode=="Hard" else 1,
                         "converged":md["converged"],
                         "cost_bike_km":md.get("cost_bike_km"),
                         "gap_percent":None if not md["converged"] else 100*(md["cost_bike_km"]/exact_cost-1),
                         "Q_max_iterations":md.get("max_Q_iterations"),
                         "conditional_marginal_err":md.get("max_conditional_prob_margin_err"),
                         "group_P_marginal_error_bikes":md.get("P_max_row_col_error_bikes"),
                         "hierarchical_solver_time_s":md.get("solver_time_s"),
                         "error_reason":None if md["converged"] else md.get("reason")}
                    result["parameter_runs"].append(row);allrows.append(row)
                    print("EPS",ident,mode,"K",K,"eps",eps,"cost",row["cost_bike_km"],
                          "good",row["converged"],flush=True)
        # Narrow sweep only for Soft K8 eps .5, test tau->0 matches Hard.
        K=8; eps=.5
        sq,near,hard,tau0=group_basis[K]
        hard_reference=next(r["cost_bike_km"] for r in result["parameter_runs"] if
                    r["configuration"]=="Hard" and r["K"]==K and r["epsilon_km"]==eps)
        for multiplier in tau_grid:
            A=softly_assigned(sq,near,K,tau0*multiplier)
            _,md=conditional_Q_pairwise(a,b,C,A[d["ii"]],A[d["jj"]],eps)
            row={**rowbase,"configuration":"Soft_temperature_sweep","K":K,
                 "epsilon_km":eps,"tau_multiplier":multiplier,
                 "converged":md["converged"],"cost_bike_km":md.get("cost_bike_km"),
                 "gap_percent":100*(md["cost_bike_km"]/exact_cost-1) if md["converged"] else None,
                 "Q_max_iterations":md.get("max_Q_iterations"),
                 "hierarchical_solver_time_s":md.get("solver_time_s"),
                 "error_reason":None if md["converged"] else md.get("reason")}
            result["parameter_runs"].append(row);allrows.append(row)
            print("TAU",ident,"K8 eps0.5 tau_factor",multiplier,
                  "cost",row["cost_bike_km"],"hard_reference",hard_reference,flush=True)
        # Integerize selected epsilon=.5, K8 with all actual conditions.
        result["physical_integer_rounding_eps05_K8"]={}
        for mode in ("Hard","Soft"):
            A=hard if mode=="Hard" else softly_assigned(sq,near,K,tau0)
            gamma,md=conditional_Q_pairwise(a,b,C,A[d["ii"]],A[d["jj"]],eps,save_plan=True)
            if gamma is None:
                result["physical_integer_rounding_eps05_K8"][mode]={"converged":False,"reason":md.get("reason")}
                continue
            im={}
            payload={"Gamma":gamma.astype(np.float32),
                     "supply_bikes":a,"served_deficit_bikes":b,
                     "P":np.array(md["P"],dtype=float),
                     "source_indices":d["ii"],"target_indices":d["jj"]}
            for strategy in ("min_cost","min_L1"):
                integer,z=round_integer_preserving_exact_marginals(
                    gamma,a,b,C,objective=strategy)
                z["rounded_integer_gap_percent"]=100*(z["rounding_cost_bike_km"]/exact_cost-1)
                im[strategy]=z
                payload[f"integer_transport_{strategy}"]=integer
            im["fractional_cost_bike_km"]=md["cost_bike_km"]
            im["fractional_gap_percent"]=100*(md["cost_bike_km"]/exact_cost-1)
            result["physical_integer_rounding_eps05_K8"][mode]=im
            np.savez_compressed(out/f"snapshot{ident}_{mode}_K8_eps05_integer.npz",**payload)
            print("INTEGERIZED",ident,mode,im,flush=True)
        # Persist incremental results if any unanticipated numerical issue.
        results["snapshots"].append(result)
        (out/"CITIBIKE_OT_SWEEP_REAL_RESULTS.json").write_text(
            json.dumps(results,indent=2,ensure_ascii=False)+"\n")
        print("SNAPSHOT_SWEEP_FINISHED",ident,len(result["parameter_runs"]),flush=True)
    keys=["snapshot_id","snapshot","configuration","K","epsilon_km",
          "tau_multiplier","converged","cost_bike_km","gap_percent",
          "exact_flat_integer_cost_bike_km","bikes",
          "hierarchical_solver_time_s","Q_max_iterations","error_reason"]
    with (out/"CITIBIKE_REAL_OT_SWEEP.csv").open("w",encoding="utf8",newline="") as fp:
        writer=csv.DictWriter(fp,fieldnames=keys,extrasaction="ignore")
        writer.writeheader();writer.writerows(allrows)
    def charts():
        colours={"Hard":"#2a77ae","Soft":"#e07f35","flat_sinkhorn_WA_I":"#5b956f"}
        plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.18})
        f,axes=plt.subplots(1,2,figsize=(13,5.5),layout="constrained",sharey=True)
        for ax,result in zip(axes,results["snapshots"]):
            for K in K_grid:
                for mode in ("Hard","Soft"):
                    rows=[r for r in result["parameter_runs"] if r["configuration"]==mode and
                          r["K"]==K and r["converged"]]
                    if rows:
                        ax.plot([r["epsilon_km"] for r in rows],
                                [r["gap_percent"] for r in rows],
                                color=colours[mode],ls="-" if K==8 else "--",
                                marker="o",label=f"{mode} K={K}")
            rows=[r for r in result["parameter_runs"] if r["configuration"]=="flat_sinkhorn_WA_I" and r["converged"]]
            if rows:
                ax.plot([r["epsilon_km"] for r in rows],
                        [r["gap_percent"] for r in rows],marker="x",
                        color=colours["flat_sinkhorn_WA_I"],lw=1.8,
                        label="Flat Sinkhorn / WA-I")
            ax.axhline(0,color="black",ls=":",lw=1.1,label="Exact mincost (0%)")
            ax.set_xscale("log");ax.set_xticks(eps_grid,labels=[str(q) for q in eps_grid])
            ax.set_title(result["snapshot"])
            ax.set_xlabel("Sample-level entropic epsilon (km)")
            ax.set_ylabel("Pure geographic cost gap vs EXACT flat (%)")
            ax.legend(fontsize=7.5)
        f.suptitle("REAL NYC June15 stock, ASSUMED half dock target: entropy vs excess shipping cost")
        f.savefig(out/"FIG_EPSILON_HARD_SOFT_K4_K8_COST_GAP.png",dpi=180)
        plt.close(f)
        f,axes=plt.subplots(1,2,figsize=(13,5.5),layout="constrained",sharey=True)
        for ax,result in zip(axes,results["snapshots"]):
            subset=[r for r in result["parameter_runs"] if r["configuration"]=="Soft_temperature_sweep"
                    and r["converged"]]
            ax.plot([r["tau_multiplier"] for r in subset],
                    [r["gap_percent"] for r in subset],color=colours["Soft"],
                    marker="o",label="Soft K=8 eps=.5")
            hard=next((r for r in result["parameter_runs"] if r["configuration"]=="Hard"
                       and r["K"]==8 and r["epsilon_km"]==.5 and r["converged"]),None)
            if hard:ax.axhline(hard["gap_percent"],color=colours["Hard"],
                              ls="--",lw=1.5,label="Hard K=8 eps=.5")
            ax.set_xscale("log");ax.set_xticks(tau_grid,labels=[str(q) for q in tau_grid])
            ax.set_xlabel("Soft membership temperature / median gap")
            ax.set_ylabel("Pure geographic cost gap vs EXACT flat (%)")
            ax.set_title(result["snapshot"]);ax.legend()
        f.suptitle("REAL stock: group ambiguity sensitivity | same K8 and epsilon0.5")
        f.savefig(out/"FIG_SOFT_TEMPERATURE_GROUP_OVERLAP_COST_GAP.png",dpi=180)
        plt.close(f)
        names=["Exact flat","Greedy","Hard K8 frac","Hard K8 integer",
               "Soft K8 frac","Soft K8 integer"]
        x=np.arange(len(names));f,axes=plt.subplots(1,2,figsize=(14,5),layout="constrained",sharey=True)
        for ax,res in zip(axes,results["snapshots"]):
            exact=res["exact_flat_integer"]["cost_integer_km"]
            hard=next((r["cost_bike_km"] for r in res["parameter_runs"] if r["configuration"]=="Hard" and r["K"]==8 and r["epsilon_km"]==.5),None)
            soft=next((r["cost_bike_km"] for r in res["parameter_runs"] if r["configuration"]=="Soft" and r["K"]==8 and r["epsilon_km"]==.5),None)
            h=res["physical_integer_rounding_eps05_K8"]["Hard"].get("min_L1",{}).get("rounding_cost_bike_km")
            s=res["physical_integer_rounding_eps05_K8"]["Soft"].get("min_L1",{}).get("rounding_cost_bike_km")
            vals=[exact,res["greedy_cost_bike_km"],hard,h,soft,s]
            ax.bar(x,vals,color=["#575757","#97a0a9","#2a77ae","#7ca7cf","#de8034","#f2b57a"])
            ax.set_xticks(x,labels=names,rotation=25,ha="right")
            ax.set_title(res["snapshot"]);ax.set_ylabel("bike-km")
        f.suptitle("Minimal-L1 integer rounding retains marginals; changes to Gamma MUST be audited")
        f.savefig(out/"FIG_INTEGER_ROUNDING_HARD_SOFT_VS_EXACT.png",dpi=180)
        plt.close(f)
    charts()
    assert len(results["snapshots"])==len(args.snapshots)
    assert len(list(out.glob("FIG_*.png")))==3
    failures=sum(not z["converged"] for r in results["snapshots"] for z in r["parameter_runs"])
    report=["# Citi Bike real historical stock: parameter-sensitivity and integer transport",
            "","**Research status: actual 2026-06-15 historical inventories; half-dock targets ASSUMED, not actual operator preferences.**",
            "Fixed physical coordinates R=I, great-circle km proxy. All same-snapshot methods use the SAME integer supply and served-demand station marginals.",
            "The globally exact unregularized flat integer min-cost flow is the true benchmark; no feasible Hard/Soft scheme can have lower pure cost.",
            "","## Frozen sensitivity grids",
            "epsilon={0.25,0.5,1,2.5} km; K={4,8}; Soft temperature multiplier={0.1,0.25,0.5,1,2,4} for K8 eps.5.",
            "Independent group hard/soft assignments use same KMeans centers; Soft top-2 membership, HARD onehot.",
            "FA, SA, DAD are not defined as city transport without explicit adaptation; WA-I is the flat Sinkhorn curve.",
            "","## Numerical audit",""]
    report.append(f"Number of numerical nonconvergence cases (recorded, not silently removed): {failures}")
    report.append("Costs are pure <Gamma,C>, not regularized objective J; fractional Gamma requires exact row/col-preserving integerization.")
    report.extend(["","## Key 0.5 km K8 result and integerization","",
        "| Snapshot | Method | fractional bike-km | rounded executable bike-km | exact flat bike-km |"])
    report.append("|---|---|---:|---:|---:|")
    for res in results["snapshots"]:
        exact=res["exact_flat_integer"]["cost_integer_km"]
        for mode in ("Hard","Soft"):
            ff=next((r["cost_bike_km"] for r in res["parameter_runs"] if r["configuration"]==mode and r["K"]==8 and r["epsilon_km"]==.5),None)
            round_results=res["physical_integer_rounding_eps05_K8"][mode]
            rr=round_results.get("min_L1",{}).get("rounding_cost_bike_km")
            tv=round_results.get("min_L1",{}).get("total_variation_fraction_of_mass")
            report.append(f"| {res['snapshot']} | {mode} | {ff if ff is not None else 'N/A'} | {rr if rr is not None else 'FAILED'} (TV={tv:.3f}) | {exact:.3f} |")
    report.extend(["",
      "NOTE ON INTEGRAL DISPATCH: Exact residual min-cost rounding can nearly reoptimize the plan; "
      "its cheap routes are NOT evidence the original Soft hierarchy is more efficient. "
      "Min-L1 rounding follows the original fractional plan as closely as achievable on the fractional-edge support. "
      "Both disclose total variation fraction of total transported mass. "
      "Neither corresponds to an operator-approved dispatch.",
      "","## Figures",
                   "- [Epsilon cost-gap curves](FIG_EPSILON_HARD_SOFT_K4_K8_COST_GAP.png)",
                   "- [Soft temperature curves](FIG_SOFT_TEMPERATURE_GROUP_OVERLAP_COST_GAP.png)",
                   "- [True integer rounding](FIG_INTEGER_ROUNDING_HARD_SOFT_VS_EXACT.png)",
                   "","## Limitations",
                   "- Only two time points on the SAME date, not independent days. The full multi-month dataset is NOT used in OT yet.",
                   "- Geographic distance is not truck road routing or real operational cost; stock target is purely ASSUMED.",
                   "- Clusters spatial only; their algorithmic advantages are hypothesized, not validated by one soft temperature sweep.",
                   "- Direct SA and DAD comparisons require a genuinely defined city-transport adaptation; these methods are NOT scored here.",
                   "- Integer rounding solves a new residual problem after the fractional plan; it is not mathematically the original Q's optimum.",
                   "- This project is isolated from DANDI/Hyser; source data remains out of Git.",
                   ""])
    (out/"CITIBIKE_REAL_OT_PARAMETER_SWEEP_REPORT_ZH.md").write_text("\n".join(report),encoding="utf8")
    print("CITIBIKE_REAL_STOCK_PARAMETER_SWEEP_COMPLETE",
          "snapshots",len(results["snapshots"]),"n_rows",len(allrows),
          "failures",failures,flush=True)

if __name__=="__main__":
    main()
