"""Build evidence-only DANDI closure curves/tables from archived real results.

Creates no model results and never touches experiments/hyser. Results from
different sample splits/dimensions are not silently pooled. The historical
tau sweep with paired group leakage is watermarked DIAGNOSTIC ONLY.
"""
from pathlib import Path
import json,csv,argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def data(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def mean(a):return float(np.mean(a))
def writecsv(f,header,rows):
    with f.open("w",newline="",encoding="utf8") as h:
        w=csv.writer(h);w.writerow(header);w.writerows(rows)

def main():
    p=argparse.ArgumentParser()
    for k in ("three-d","four-d","baselines","transforms","historical-tau","output"):
        p.add_argument("--"+k,required=True,type=Path)
    args=p.parse_args()
    a=data(args.three_d);b=data(args.four_d);c=data(args.baselines)
    t=data(args.transforms);s=data(args.historical_tau)
    assert a["original_100_test_trials_never_used"] and b["no_original_test_evaluated_or_optimized"]
    assert c["original_test100_not_used"]
    out=args.output;fig=out/"figures";tab=out/"tables"
    fig.mkdir(parents=True,exist_ok=True);tab.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":10,"figure.dpi":130})
    colors={"Hard":"#336ba2","Soft":"#d98135","SA":"#777777","WA":"#21856e","DAD_like_3D":"#9864a0"}
    tolrows=[]
    for dim in [3,4]:
        for family in ("PCA","FA"):
            src=f"{family}{dim}"
            for method in ("Hard","Soft"):
                rs=(a["methods"][src][method] if dim==3 else b["methods"][src][method])
                for point in (5,10,20):
                    val=mean([r["validation_cross_trial_physical_retrieval"][f"pos_hit{point}pct"] for r in rs])
                    tolrows.append([dim,src,method,"position",point,round(val,8)])
                for point in (30,45):
                    val=mean([r["validation_cross_trial_physical_retrieval"][f"direction_hit{point}deg"] for r in rs])
                    tolrows.append([dim,src,method,"direction",point,round(val,8)])
    random=b["random_cross_trial_baseline"]
    for point in (5,10,20):tolrows.append(["same", "random", "random", "position",point,round(random[f"pos_hit{point}pct"],8)])
    for point in (30,45):tolrows.append(["same", "random", "random", "direction",point,round(random[f"direction_hit{point}deg"],8)])
    writecsv(tab/"DANDI_3D4D_REAL_PHYSICAL_TOLERANCE_CURVES.csv",
             ["dimension","representation","method","metric","threshold","hit_fraction"],tolrows)
    for kind,points,ylabel,file in [
        ("position",[5,10,20],"Physical position matching hit rate","POSITION_TOLERANCE_3D_4D.png"),
        ("direction",[30,45],"Physical velocity direction hit rate","DIRECTION_TOLERANCE_3D_4D.png")]:
        f,axs=plt.subplots(1,2,figsize=(13,4.7),constrained_layout=True)
        for ax,dim in zip(axs,[3,4]):
            for family,style in (("PCA","--"),("FA","-")):
                for method in ("Hard","Soft"):
                    vals=[next(r[-1] for r in tolrows if r[0]==dim and r[1]==f"{family}{dim}" and r[2]==method and r[3]==kind and r[4]==pt) for pt in points]
                    ax.plot(points,np.array(vals)*100,marker="o",ls=style,
                            color=colors[method],label=f"{family}{dim} {method}",lw=1.65)
            rand=[next(r[-1] for r in tolrows if r[1]=="random" and r[3]==kind and r[4]==pt) for pt in points]
            ax.plot(points,np.array(rand)*100,marker="x",color="black",lw=1.5,ls=":",label="Random same validation")
            ax.set_title(f"{dim}D, same 80 internal validation trials");ax.set_xlabel("Position threshold as % of train median path" if kind=="position" else "Velocity angular tolerance (degrees)")
            ax.set_ylabel(ylabel+" (%)");ax.grid(alpha=.24);ax.legend(fontsize=8)
            ax.set_xticks(points)
        f.suptitle("Real DANDI post-fit evaluation threshold curves (not model tuning)")
        f.savefig(fig/file,dpi=175);plt.close(f)
    base_rows=[]
    for dim in (3,4):
        src=f"FA{dim}"
        for method in ("Hard","Soft"):
            rows=a["methods"][src][method] if dim==3 else b["methods"][src][method]
            base_rows.append([src,method,len(rows),
               mean([r["validation_free_W2sq"] for r in rows]),
               mean([r["validation_cross_trial_physical_retrieval"]["pos_hit10pct"] for r in rows]),
               mean([r["validation_cross_trial_physical_retrieval"]["direction_hit30deg"] for r in rows]),
               sum(r["strict_admm_passed"] for r in rows),"HiWA full hierarchical PQR"])
        for method,rows in c["results"][src].items():
            base_rows.append([src,method,len(rows),
               mean([r["validation_free_W2sq"] for r in rows]),
               mean([r["validation_physical_cross_trial"]["pos_hit10pct"] for r in rows]),
               mean([r["validation_physical_cross_trial"]["direction_hit30deg"] for r in rows]),
               "N/A","DAD-inspired, not official" if "DAD" in method else "ported baseline"])
    writecsv(tab/"DANDI_SAME_VALIDATION_METHOD_COMPARISON.csv",
             ["representation","method","run_count","W2sq_in_native_latent","pos_hit10","direction_hit30","admm_passed","status"],base_rows)
    for src in ("FA3","FA4"):
        rows=[r for r in base_rows if r[0]==src]
        names=[r[1] for r in rows];v=np.arange(len(rows));f,axes=plt.subplots(1,3,figsize=(15,4.2),constrained_layout=True)
        for ax,idx,title in zip(axes,[3,4,5],["Free W2 squared (same representation)","Original position hit <=10%","Original speed direction hit <=30 deg"]):
            vals=np.array([r[idx] for r in rows])
            if idx in (4,5):vals*=100
            ax.bar(v,vals,color=[colors[x] for x in names])
            ax.set_xticks(v,labels=names,rotation=30,ha="right")
            ax.set_title(title);ax.grid(axis="y",alpha=.2)
            if idx in (4,5):ax.set_ylabel("Percent")
        f.suptitle(src+" / 80-trial internal validation | SA=one fit, DAD=modified 3D; not official DAD")
        f.savefig(fig/f"DANDI_{src}_HARD_SOFT_SA_WA_DAD_BASELINES.png",dpi=170);plt.close(f)
    # Predeclared 4D TARGET-only changes are discrete conditions, not a continuous scan.
    rows_t=[]
    for m,entry in t["variants"].items():
        for met in ("hard","soft"):
            run=entry["runs"][met]
            rows_t.append([m,met,len(run),
              mean([z["shared_original_eval"]["original_target_free_W2sq"] for z in run]),
              mean([z["shared_original_eval"]["original_target_sync_R2"] for z in run]),
              sum(z["strict_admm_pass"] for z in run)])
    writecsv(tab/"DANDI_4D_TARGET_TRANSFORM_ABLATION.csv",
             ["target_transform","method","seeds","original_4d_free_W2sq","original_4d_sync_R2","strict_admm_passed"],rows_t)
    order=["identity","block_zca","full_zca","arcsinh"]
    fig2,ax=plt.subplots(1,2,figsize=(12,4.4),constrained_layout=True)
    for idx,title in ((3,"4D original-space W2 squared"),(4,"Original synchronized R2 (postfit, diagnostic)")):
        axs=ax[0] if idx==3 else ax[1]
        for m in ("hard","soft"):
            z=[next(r[idx] for r in rows_t if r[0]==k and r[1]==m) for k in order]
            axs.plot(order,z,marker="o",label=m,color=colors[m.capitalize()])
        axs.set_title(title);axs.grid(alpha=.2);axs.legend()
    fig2.suptitle("DANDI 4D TARGET-only transformations (categorical ablation, original 100 TEST already explored)")
    fig2.savefig(fig/"DANDI_4D_TARGET_TRANSFORM_ABLATION.png",dpi=165);plt.close(fig2)
    # Historical tau samples are paired-label GROUP-SUPERVISED, NEVER unpaired.
    taurows=[]
    for key,x in s["summary"].items():
        if not (key=="hard" or key.startswith("tau")):continue
        val=0.0 if key=="hard" else float(key.replace("tau",""))
        taurows.append([val,key,x["total_PQ_OT_after"]["mean"],
             x["global_W2sq"]["mean"],x["1NN"]["mean"],
             x["numerical_audit_passed"],"GROUP-SUPERVISED: target labels copied using time pairing"])
    taurows.sort()
    writecsv(tab/"LEGACY_LEAKED_GROUP_TAU_SWEEP_DIAGNOSTIC_ONLY.csv",
             ["tau","method","nested_transport","global_free_W2sq","legacy_1NN","numerical_audit_passed","WARNING"],taurows)
    f,axs=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for ax,col,title in zip(axs,[2,4],["Hierarchical pure transport cost","Legacy same-group 1NN measure"]):
        ax.plot([r[0] for r in taurows],[r[col] for r in taurows],
                marker="o",color="#aa3355")
        ax.set_xlabel("tau (0=Hard); INVALID FOR STRICT UNSUPERVISED")
        ax.set_title(title);ax.grid(alpha=.25)
        ax.text(.5,.56,"SUPERVISED GROUP LEAKAGE\nDIAGNOSTIC ONLY",
                color="#ba1730",fontsize=13,weight="bold",alpha=.53,
                ha="center",va="center",transform=ax.transAxes,rotation=20)
    f.suptitle("Historical DANDI tau sweep - DO NOT USE as Soft-vs-Hard evidence")
    f.savefig(fig/"LEGACY_TAU_SWEEP_SUPERVISED_WARNING.png",dpi=160);plt.close(f)
    meta={"source_files":[str(args.three_d),str(args.four_d),str(args.baselines),
                          str(args.transforms),str(args.historical_tau)],
          "plots":[p.name for p in fig.glob("*.png")],
          "tables":[p.name for p in tab.glob("*.csv")],
          "main_curves":"physical tolerance; method comparison; 4D target transform",
          "tau_curve_status":"historical supervision leakage; excluded from valid unpaired findings",
          "requested_overlap_rate_vs_matching_curve_status":"NOT RUN: proposed future experiment only",
          "figure_export_date":"2026-11-19"}
    (out/"FIGURES_EVIDENCE_MANIFEST.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False)+"\n")
    assert len(meta["plots"])==6 and len(meta["tables"])==4,(meta["plots"],meta["tables"])
    print("DANDI_CLOSEOUT_FIGURES_VALID",meta,flush=True)

if __name__=="__main__":main()
