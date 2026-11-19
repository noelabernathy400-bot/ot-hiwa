"""Genuine CityBikes historical NYC station INVENTORY preflight, no OT.

Read a full REAL CityBikes 2026-06 Parquet (12M rows), filter via Arrow
in bounded batches to two fixed June 15 snapshot times. Snapshot bikes/free
are real observed values. Supply/deficit colors are explicitly a POLICY
COUNTERFACTUAL targeting half of recorded station dock slots, not verified
operator demand. Never interpret this as observed relocation/savings.
"""
import argparse,json
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

SNAPSHOTS=["2026-06-15 13:00:00","2026-06-15 20:00:00"]
HALFWINDOW=timedelta(minutes=12)

def getdata(path):
    pf=pq.ParquetFile(path)
    req=["nuid","name","latitude","longitude","bikes","free","timestamp","extra"]
    assert all(n in pf.schema_arrow.names for n in req),pf.schema_arrow.names
    targets=[datetime.strptime(z,"%Y-%m-%d %H:%M:%S") for z in SNAPSHOTS]
    data=[{} for _ in targets]
    errors={"invalid_stock":0,"no_status":0,"invalid_coords":0,"missing_or_invalid_slots":0}
    nrows=0;nkept=0
    for batch in pf.iter_batches(batch_size=175000,columns=req,use_threads=True):
        stamps=batch.column(batch.schema.get_field_index("timestamp"))
        mask=None
        for target in targets:
            current=pc.and_(pc.greater_equal(stamps,pa.scalar(target-HALFWINDOW)),
                            pc.less_equal(stamps,pa.scalar(target+HALFWINDOW)))
            mask=current if mask is None else pc.or_(mask,current)
        filtered=pa.Table.from_batches([batch]).filter(mask)
        nrows+=batch.num_rows
        if filtered.num_rows==0:continue
        nkept+=filtered.num_rows
        for row in filtered.to_pylist():
            station=row["nuid"]
            if not station:continue
            try:
                lat=float(row["latitude"]);lon=float(row["longitude"])
                bikes=int(row["bikes"]);free=int(row["free"])
            except (TypeError,ValueError):
                errors["invalid_stock"]+=1;continue
            if not (40.45<=lat<=40.96 and -74.3<=lon<=-73.63):
                errors["invalid_coords"]+=1;continue
            if bikes<0 or free<0:
                errors["invalid_stock"]+=1;continue
            try: extra=json.loads(row["extra"] or "{}")
            except (TypeError,ValueError):extra={}
            slots=extra.get("slots")
            try:slots=int(slots) if slots is not None else None
            except (ValueError,TypeError):slots=None
            if slots is None or slots<=0:
                errors["missing_or_invalid_slots"]+=1;continue
            if bikes+free>slots+8:
                errors["invalid_stock"]+=1;continue
            if extra.get("virtual") is True:
                errors["no_status"]+=1;continue
            if not extra.get("renting",True) or not extra.get("returning",True):
                errors["no_status"]+=1;continue
            ts=row["timestamp"]
            for i,target in enumerate(targets):
                dt=abs((ts-target).total_seconds())
                if dt<=HALFWINDOW.total_seconds():
                    if station not in data[i] or dt<data[i][station]["age_s"]:
                        data[i][station]={
                            "nuid":station,"name":row.get("name") or "",
                            "lat":lat,"lon":lon,"bikes":bikes,"free":free,
                            "slots":slots,"age_s":dt}
    assert nrows==pf.metadata.num_rows,(nrows,pf.metadata.num_rows)
    assert nkept>5000,nkept
    for i,v in enumerate(data):
        assert len(v)>1000,(i,len(v))
    return data,{"total_parquet_rows_read":nrows,"rows_around_two_time_slices":nkept,
                 "row_groups":pf.metadata.num_row_groups,
                 "source_file_bytes":path.stat().st_size,
                 "screening_errors":errors,
                 "approx_max_age_seconds":HALFWINDOW.total_seconds()}

def draw(ax,arr,values,label,cm,vmin=None,vmax=None):
    lat=np.array([r["lat"] for r in arr]);lon=np.array([r["lon"] for r in arr])
    size=2.5+22*np.sqrt(np.array([r["slots"] for r in arr])/max(r["slots"] for r in arr))
    im=ax.scatter(lon,lat,s=size,c=values,cmap=cm,vmin=vmin,vmax=vmax,
                  edgecolors="none",alpha=.8,rasterized=True)
    ax.set_title(label)
    ax.set_xlabel("Longitude");ax.set_ylabel("Latitude")
    ax.set_aspect(1/np.cos(np.deg2rad(lat.mean())))
    ax.grid(alpha=.14)
    plt.colorbar(im,ax=ax,shrink=.78)
    return im

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    groups,meta=getdata(args.input)
    summary={
      "source":"https://data.citybik.es/dumps/by-network/2026/202606-citi-bike-nyc-stats.parquet",
      "source_license":"CityBikes ODbL, attribution required, underlying operator policy applies",
      "source": "CityBikes NYC historical station status, not monthly ride file",
      "source_details": "https://data.citybik.es/dumps/by-network/2026/202606-citi-bike-nyc-stats.parquet",
      "snapshots":SNAPSHOTS,
      "timestamp_note":"Parquet timestamp lacks timezone marker; report literal stored timestamp, NOT inferred local hour",
      "window_each": "Nearest true observation within +/- 12 minutes of fixed timestamp",
      "target_inventory_rule":"ILLUSTRATIVE policy target floor(slots/2), NOT real operator target and NOT a demand measurement",
      "no_hiwa_or_soft_model_fit":True,
      "true_observed_bikes_and_free_vs_hypothetical_target_distinguished":True,
      "no_historical_operator_relations_claim":True,
      "metadata":meta,"slices":[]
    }
    fig,axs=plt.subplots(2,3,figsize=(17.5,11.7),layout="constrained")
    for idx,g in enumerate(groups):
        arr=list(g.values())
        bike=np.array([r["bikes"] for r in arr])
        free=np.array([r["free"] for r in arr])
        slots=np.array([r["slots"] for r in arr])
        fill=np.minimum(1.,bike/np.maximum(1,bike+free))
        target=slots//2
        difference=bike-target
        source=np.clip(difference,0,None)
        demand=np.clip(-difference,0,None)
        count_bike=int(bike.sum());count_free=int(free.sum())
        rec={
          "timestamp_raw":SNAPSHOTS[idx],
          "real_station_statuses":len(arr),
          "observed_bikes_total":count_bike,
          "observed_free_docks_total":count_free,
          "observed_slots_from_metadata_total":int(slots.sum()),
          "observed_bikes_station_p10_p50_p90":np.quantile(bike,[.1,.5,.9]).tolist(),
          "observed_fill_ratio_station_p10_p50_p90":np.quantile(fill,[.1,.5,.9]).tolist(),
          "illustrative_policy_target_fraction_of_slots":.5,
          "policy_scenario_surplus_station_count":int(np.count_nonzero(source)),
          "policy_scenario_deficit_station_count":int(np.count_nonzero(demand)),
          "policy_scenario_surplus_bikes":int(source.sum()),
          "policy_scenario_deficit_bikes":int(demand.sum()),
          "mass_difference_not_silently_normalized":int(source.sum()-demand.sum()),
          "not_actual_bike_relocation_demand":True
        }
        summary["slices"].append(rec)
        draw(axs[idx,0],arr,bike,"Observed AVAILABLE BIKES at stations","viridis")
        draw(axs[idx,1],arr,free,"Observed FREE DOCKS at stations","cividis")
        limit=max(3,np.quantile(np.abs(difference),.95))
        draw(axs[idx,2],arr,difference,"COUNTERFACTUAL: B − floor(slots/2)","coolwarm",-limit,limit)
        axs[idx,0].text(.01,.01,SNAPSHOTS[idx]+" recorded time",transform=axs[idx,0].transAxes,
                        fontsize=8,va="bottom")
    fig.suptitle("REAL June 2026 CityBikes NYC inventory snapshots | target stock is a separately labeled ASSUMPTION",
                 fontsize=15)
    fig.savefig(args.output/"NYC_REAL_STATION_INVENTORY_AND_ILLUSTRATIVE_SURPLUS.png",
                dpi=155,bbox_inches="tight")
    plt.close(fig)
    # Adjacent stock for same station may change substantially during a single day.
    both=set(groups[0])&set(groups[1])
    if both:
        b0=np.array([groups[0][k]["bikes"] for k in sorted(both)])
        b1=np.array([groups[1][k]["bikes"] for k in sorted(both)])
        change=b1-b0
        summary["same_station_between_snapshot_count"]=len(both)
        summary["absolute_stock_change_median"]=float(np.median(np.abs(change)))
        summary["absolute_stock_change_p90"]=float(np.quantile(np.abs(change),.9))
        summary["total_abs_stock_change"]=int(np.sum(np.abs(change)))
    (args.output/"REAL_HISTORICAL_INVENTORY_PREFLIGHT.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("REAL_INVENTORY_PREFLIGHT_COMPLETE",json.dumps({
        "parquet_rows":meta["total_parquet_rows_read"],
        "stations":[x["real_station_statuses"] for x in summary["slices"]],
        "available_bikes":[x["observed_bikes_total"] for x in summary["slices"]],
        "policy_surpluses":[x["policy_scenario_surplus_bikes"] for x in summary["slices"]],
        "policy_deficits":[x["policy_scenario_deficit_bikes"] for x in summary["slices"]],
        "same_stations":summary.get("same_station_between_snapshot_count"),
        },ensure_ascii=False),flush=True)
if __name__=="__main__":main()
