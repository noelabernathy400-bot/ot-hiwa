"""Postfit deterministic analysis of already computed June2026 REAL stock results.

No model fitting and no raw downloads: transport sensitivity and 2-snapshot
comparison only, do NOT extrapolate to independent NYC dates or real
business savings. Figures are created from transparent archived JSON.
"""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    j=json.loads(a.source.read_text())
    dest=a.out;dest.mkdir(parents=True,exist_ok=True)
    assert len(j["snapshots"])==2 and j["eps_grid_km"]==[.25,.5,1.,2.5]
    cols={"Hard":"#2774a8","Soft":"#d68038"}
    plt.rcParams.update({"font.size":10,"axes.grid":True,"grid.alpha":.17})
    f,axs=plt.subplots(1,2,figsize=(13.6,5.6),layout="constrained",sharey=True)
    table=[]
    for ax,z in zip(axs,j["snapshots"]):
        for K in (4,8):
            x=[];y=[]
            for eps in j["eps_grid_km"]:
                h=next(r for r in z["parameter_runs"] if r["configuration"]=="Hard"
                       and r["K"]==K and r["epsilon_km"]==eps)
                s=next(r for r in z["parameter_runs"] if r["configuration"]=="Soft"
                       and r["K"]==K and r["epsilon_km"]==eps)
                assert h["converged"] and s["converged"]
                improvement=(h["cost_bike_km"]-s["cost_bike_km"])/h["cost_bike_km"]*100
                delta=s["cost_bike_km"]-h["cost_bike_km"]
                table.append({
                    "snapshot":z["snapshot"],"K":K,"epsilon_km":eps,
                    "hard_bike_km":round(h["cost_bike_km"],6),
                    "soft_bike_km":round(s["cost_bike_km"],6),
                    "soft_minus_hard_bike_km":round(delta,6),
                    "soft_reduces_cost_relative_to_hard_percent":round(improvement,6)})
                x.append(eps);y.append(improvement)
            ax.plot(x,y,"o-",lw=2.3,markersize=6,color="#2774a8" if K==4 else "#d68038",
                    label=f"K={K}")
        ax.axhline(0,color="#202020",ls="--",lw=1.2)
        ax.set_xticks(j["eps_grid_km"])
        ax.set_xlabel("Sample-level entropic ε (km)")
        ax.set_title(z["snapshot"]+" (literal timestamp)")
        ax.set_ylabel("Soft improvement relative to Hard (%)")
        ax.legend(loc="best")
        ax.text(.02,.04,"ABOVE 0 = Soft cheaper\nBELOW 0 = Hard cheaper",
                transform=ax.transAxes,va="bottom",fontsize=9,alpha=.8)
    f.suptitle("REAL Citi Bike inventory: soft grouping is NOT uniformly better; epsilon x K interaction")
    f.savefig(dest/"SOFT_VS_HARD_DIRECT_COST_GAIN_BY_EPSILON_K.png",dpi=180)
    plt.close(f)
    with (dest/"SOFT_MINUS_HARD_REAL_STOCK_EPS_K.csv").open("w",newline="",encoding="utf8") as h:
        w=csv.DictWriter(h,fieldnames=list(table[0]))
        w.writeheader();w.writerows(table)
    f,axs=plt.subplots(1,2,figsize=(13.8,5.8),layout="constrained",sharey=True)
    for ax,z in zip(axs,j["snapshots"]):
        data=[]
        for mode in ("Hard","Soft"):
            q=z["physical_integer_rounding_eps05_K8"][mode]
            for objective in ("min_cost","min_L1"):
                m=q[objective]
                data.append((mode,objective,m["total_variation_fraction_of_mass"]*100,
                             m["rounding_cost_bike_km"]))
        pos=np.arange(len(data))
        bars=ax.bar(pos,[q[2] for q in data],
                    color=[cols[q[0]] for q in data])
        for bar,row in zip(bars,data):
            bar.set_alpha(.55 if row[1]=="min_cost" else 1.)
        ax.set_xticks(pos,labels=[q[0]+"\n"+q[1] for q in data])
        ax.set_title(z["snapshot"])
        ax.set_ylabel("Total variation in transport mass after rounding (%)")
        for bar,row in zip(bars,data):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,
                    f"{row[2]:.1f}%",ha="center",fontsize=9)
        ax.set_ylim(0,100)
    f.suptitle("Rounding is NOT neutral: even min-L1 changes 65–75% of the mass in original Gamma")
    f.savefig(dest/"INTEGER_ROUNDING_CHANGE_IN_TRANSPORT_MASS.png",dpi=180)
    plt.close(f)
    lines=["# 2026-11-20 Citi Bike actual-stock OT parameter sweep: interpretive conclusions",
       "",
       "**This document analyses REAL computed results, not promises or synthetic data.** Original June15 historical inventory, fixed R=I, assumed HALF-FULL policy, 5,777/5,953 bikes. Only TWO same-day snapshots; not a cross-day study.",
       "",
       "## Important discovery 1: Soft x K x epsilon interaction",
       "",
       "Soft reduces pure transport cost relative to Hard at SMALL epsilon for K=4, but not consistently at K=8. This does not establish an intrinsic advantage in the general case.",
       "",
       "Here positive percentages mean Soft costs **LESS** than the matched Hard configuration, negative mean Soft costs more:",
       "",
       "| Snapshot | K | ε=.25 | ε=.5 | ε=1.0 | ε=2.5 |",
       "|---|---:|---:|---:|---:|---:|"]
    for z in j["snapshots"]:
        name=z["snapshot"].split(" ")[1]
        for K in (4,8):
            vals=[next(q["soft_reduces_cost_relative_to_hard_percent"]
                 for q in table if q["snapshot"]==z["snapshot"] and q["K"]==K and q["epsilon_km"]==eps)
                 for eps in j["eps_grid_km"]]
            lines.append(f"| {name} | {K} | "+" | ".join(f"{v:+.2f}%" for v in vals)+" |")
    lines+=["","![Real Soft relative improvement](SOFT_VS_HARD_DIRECT_COST_GAIN_BY_EPSILON_K.png)",
    "",
    "All 52 numerical settings on the TWO observed snapshots (20 fixed ε×K comparisons plus 12 τ scans per two snapshots), plus true exact mincost and integer b-matching audit, passed the pipeline. The exact integer baseline remains a lower bound on the same linear cost. All Q/P outputs are from separate fixed-R geographic hierarchical models; no arbitrary point-cloud rotation.",
    "",
    "## Important discovery 2: soft temperature sensitivity",
    "",
    "For K8 and ε=.5, reducing soft temperature multiplier toward 0.1–.25 draws Soft near Hard transport cost; making membership more diffuse (1–4) increases the cost in both snapshots. This is NOT a controlled real-world border ambiguity sweep: the station location features are fixed, only soft grouping temperature changes.",
    "",
    "## Important discovery 3: fractional OT is NOT automatically an executable integer dispatch",
    "",
    "The proposed floor+residual min-cost integerization legally conserves all station vehicle counts, but on THIS dense entropic plan, more than 90% of vehicles are processed by a NEW residual optimizer; this optimization can overwhelm the original algorithm's choice of Gamma. Cost-after-rounding is therefore not direct evidence about Hard vs Soft.",
    "",
    "Even the **minimal-L1** integerizing b-matching changes most pairwise transport:",
    "",
    "| Stored snapshot | Model | Min-L1 integer cost (bike-km) | Changed total transport mass (TV) | Min-cost rounding TV |",
    "|---|---|---:|---:|---:|"]
    for z in j["snapshots"]:
        for mode in ("Hard","Soft"):
            r=z["physical_integer_rounding_eps05_K8"][mode]
            lin=r["min_L1"];cost=r["min_cost"]
            lines.append(f"| {z['snapshot']} | {mode} | {lin['rounding_cost_bike_km']:.2f} | "
                         f"{100*lin['total_variation_fraction_of_mass']:.1f}% | "
                         f"{100*cost['total_variation_fraction_of_mass']:.1f}% |")
    lines+=["",
        "![Rounding transport mass changed](INTEGER_ROUNDING_CHANGE_IN_TRANSPORT_MASS.png)",
        "",
        "A lower integer cost can be entirely due to new min-cost rounding, not the unmodified hierarchical Gamma. The outcome is a diagnostic showing that dense Sinkhorn transport needs sparsification / integer feasibility IN the design, not a proof that the method yielded real dispatch savings.",
        "",
        "## Next independent scientific experiment",
        "",
        "1. Define structural why-K4-soft-helps: separate P-level routing from conditionally forced Q-level routing under fixed epsilon; compare a capacity-aware integer hierarchical flow against the flat exact mincost, holding all mass/route constraints constant.",
        "2. **Do not tune on July validation / August-September heldout** until the design and performance metrics are frozen; two same-day windows are not statistically independent.",
        "3. Use multiple genuinely independent dates; report confidence intervals on Soft−Hard effects and on matching flow mass / boundary station role; do not pick a visually attractive parameter post hoc.",
        "4. City-wide full network source count is per snapshot (hundreds of supply/deficit stations), not 18.87 million OT nodes. Real road routing and operator demand absent; current cost uses great-circle bike-km proxy.",
        "5. SA/DAD original versions are geometric alignment methods and **not** meaningful direct fixed-geography dispatch baselines. WA with fixed R=I is flat Sinkhorn. There is no SA/DAD score in this transport study.",
        ""]
    (dest/"REAL_OT_PARAMETER_SWEEP_SCIENTIFIC_INTERPRETATION_ZH.md").write_text("\n".join(lines),encoding="utf8")
    print("POSTFIT_CITIBIKE_PARAMETER_INTERPRETATION_COMPLETE",len(table),flush=True)

if __name__=="__main__":main()
