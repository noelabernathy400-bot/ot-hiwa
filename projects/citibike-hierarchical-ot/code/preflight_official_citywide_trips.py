"""Preflight REAL official Citi Bike NYC trip ZIPs, citywide April-July 2026.

Never mistakes trip outflow-inflow for actual inventory surplus/deficit.
No optimal transport model is fitted here. Station groups are TRAIN-only
spatial KMeans/softmax visualization, not transferred label supervision.
Input validation July kept separate. Aug+Sep are RESERVED future holdout
and are not read. Source official https://citibikenyc.com/system-data.
No raw rider IDs or archives copied to GitHub.
"""
import argparse, collections, csv, hashlib, io, json, math, os, zipfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from sklearn.cluster import KMeans
from scipy.special import softmax

def increment(counter, key, value=1):
    counter[key]+=value

def parse_time(s):
    # 2026 Citi Bike timestamps are published without explicit local UTC offset.
    s=str(s or "").replace("T"," ").strip()
    try: return datetime.fromisoformat(s)
    except (ValueError,TypeError): return None

def valid_coord(lat,lon):
    return (40.45<=lat<=40.96 and -74.3<=lon<=-73.63)

def read_official_zip(path,month,store):
    h=hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b=f.read(8<<20)
            if not b:break
            h.update(b)
    manifest={"official_zip_filename":path.name,"sha256":h.hexdigest(),
              "archive_bytes":path.stat().st_size}
    qc=collections.Counter()
    seen_headers=[]
    with zipfile.ZipFile(path) as z:
        members=[n for n in z.namelist() if n.lower().endswith(".csv")
                 and not n.startswith("__MACOSX/")]
        assert members,(path,"no csv")
        for name in members:
            with z.open(name) as handle:
                rd=csv.DictReader(io.TextIOWrapper(handle,encoding="utf-8-sig",newline=""))
                required=["started_at","ended_at","start_station_id",
                  "end_station_id","start_lat","start_lng","end_lat","end_lng"]
                missing=[x for x in required if x not in (rd.fieldnames or [])]
                assert not missing,("missing expected official columns",name,missing)
                seen_headers.append(list(rd.fieldnames))
                for row in rd:
                    qc["raw_rides"]+=1
                    a=parse_time(row["started_at"])
                    b=parse_time(row["ended_at"])
                    if a is None or b is None or b<=a:
                        qc["invalid_time"]+=1;continue
                    # Cross-month boundary rides must not leak into another fold.
                    if a.strftime("%Y%m")!=month or b.strftime("%Y%m")!=month:
                        qc["outside_month_or_cross_month"]+=1;continue
                    s=row["start_station_id"].strip()
                    t=row["end_station_id"].strip()
                    if not s or not t:
                        qc["missing_station_id"]+=1;continue
                    try:
                        al,ao,bl,bo=map(float,[
                          row["start_lat"],row["start_lng"],row["end_lat"],row["end_lng"]])
                    except (ValueError,TypeError):
                        qc["invalid_location"]+=1;continue
                    if not (valid_coord(al,ao) and valid_coord(bl,bo)):
                        qc["outside_NYC_bbox"]+=1;continue
                    qc["accepted_rides"]+=1
                    if a.weekday()>=5:qc["weekend_rides"]+=1
                    train=month in {"202604","202605","202606"}
                    target=store["train"] if train else store["validation"]
                    day=a.date().isoformat()
                    target["daily_trip_count"][day]+=1
                    target["month_trip_count"][month]+=1
                    for id_,lat,lon,time,field in [
                        (s,al,ao,a,0),(t,bl,bo,b,1)
                    ]:
                        station=target["stations"].setdefault(
                            id_,{"lat_sum":0.,"lon_sum":0.,"coord_n":0,
                                 "departures":0,"arrivals":0,
                                 "weekhour":np.zeros((7,24,2),dtype=np.int64),
                                 "peak":np.zeros((4,2),dtype=np.int64),
                                 "name":""})
                        station["lat_sum"]+=lat
                        station["lon_sum"]+=lon
                        station["coord_n"]+=1
                        station["departures" if field==0 else "arrivals"]+=1
                        station["weekhour"][time.weekday(),time.hour,field]+=1
                        if time.weekday()<5:
                            if 7<=time.hour<10:
                                station["peak"][0,field]+=1
                            elif 17<=time.hour<20:
                                station["peak"][1,field]+=1
                        else:
                            if 7<=time.hour<10:station["peak"][2,field]+=1
                            elif 17<=time.hour<20:station["peak"][3,field]+=1
    manifest.update({"csv_members":members,"csv_count":len(members),
                     "headers":seen_headers,"quality_counts":dict(qc),
                     "validated_rides":qc["accepted_rides"]})
    return manifest

def visual(source,validation,out,manifest):
    rows=[]
    for id_,v in source["stations"].items():
        if v["coord_n"]<20:continue
        lat=v["lat_sum"]/v["coord_n"]
        lon=v["lon_sum"]/v["coord_n"]
        n=v["departures"]+v["arrivals"]
        if n<50:continue
        rows.append((id_,lat,lon,n,v))
    assert len(rows)>=500,("unusually small NYC station coverage",len(rows))
    labels=[r[0] for r in rows]
    lat=np.array([r[1] for r in rows])
    lon=np.array([r[2] for r in rows])
    activity=np.array([r[3] for r in rows])
    x=(lon-float(lon.mean()))*111.32*np.cos(np.deg2rad(lat.mean()))
    y=(lat-float(lat.mean()))*111.32
    xy=np.column_stack([x,y])
    K=12
    kmeans=KMeans(n_clusters=K,n_init=20,random_state=42).fit(xy)
    dist2=((xy[:,None,:]-kmeans.cluster_centers_[None,:,:])**2).sum(axis=2)
    ordered=np.sort(dist2,axis=1)
    gap=ordered[:,1]-ordered[:,0]
    tau=float(np.quantile(gap,.50))
    assert tau>0.
    probs=softmax(-dist2/tau,axis=1)
    norm_entropy=(-np.sum(probs*np.log(np.maximum(probs,1e-15)),axis=1)/np.log(K))
    morning=np.array([v["peak"][0,1]-v["peak"][0,0] for *_,v in rows])
    night=np.array([v["peak"][1,1]-v["peak"][1,0] for *_,v in rows])
    # These are ride-induced net arrivals, NOT actual excess vehicle inventory.
    def scatter(ax,z,caption,cm=None,limits=None):
        s=2.+30*np.sqrt(activity/activity.max())
        if z is None:
            ax.scatter(lon,lat,s=s,c="#2e7cb0",alpha=.57,lw=0,rasterized=True)
        else:
            opts={"c":z,"cmap":cm,"s":s,"alpha":.78,"lw":0,"rasterized":True}
            if limits is not None:opts.update(vmin=-limits,vmax=limits)
            im=ax.scatter(lon,lat,**opts)
            plt.colorbar(im,ax=ax,shrink=.76)
        ax.set_xlabel("Longitude");ax.set_ylabel("Latitude")
        ax.set_title(caption)
        ax.set_aspect(1./np.cos(np.deg2rad(lat.mean())))
        ax.grid(alpha=.16)
    fig,axs=plt.subplots(2,2,figsize=(13,12),layout="constrained")
    scatter(axs[0,0],None,"All active Citi Bike NYC stations: train Apr-Jun 2026")
    lim=float(max(1,np.quantile(np.abs(morning),.98)))
    scatter(axs[0,1],morning,"Morning net ride arrivals (not physical surplus)","coolwarm",lim)
    lim=float(max(1,np.quantile(np.abs(night),.98)))
    scatter(axs[1,0],night,"Evening net ride arrivals (not physical surplus)","coolwarm",lim)
    scatter(axs[1,1],norm_entropy,"Spatial soft grouping normalized entropy","viridis")
    fig.suptitle("REAL Citi Bike citywide official trip history: station structure BEFORE OT",fontsize=15)
    fig.savefig(out/"CITIBIKE_NYC_CITYWIDE_STATION_STRUCTURE.png",
                dpi=170,bbox_inches="tight")
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(14,6.5),layout="constrained")
    scatter(axs[0],kmeans.labels_,"K=12 train spatial HARD clusters","tab20")
    scatter(axs[1],probs.argmax(axis=1),"SOFT membership argmax (DISPLAY ONLY)","tab20")
    fig.suptitle("Identical 12 center locations, spatial Hard and Soft 12-group visualization only",fontsize=14)
    fig.savefig(out/"CITIBIKE_NYC_HARD_SOFT_12GROUPS.png",dpi=165,bbox_inches="tight")
    plt.close(fig)
    trains=sorted(source["daily_trip_count"].items())
    vals=sorted(validation["daily_trip_count"].items())
    fig,ax=plt.subplots(figsize=(13,4.5),layout="constrained")
    for label,rr,col in [
        ("Train Apr-Jun",trains,"#2b79aa"),
        ("Validation July",vals,"#db8a36")
    ]:
        dates=np.array([np.datetime64(t) for t,_ in rr])
        vals0=np.array([v for _,v in rr])
        ax.plot(dates,vals0,label=label,color=col,lw=1.3)
    ax.set_title("Actual official Citi Bike daily ride records (not actual inventories)")
    ax.set_ylabel("Clean NYC ride counts");ax.set_xlabel("Calendar date")
    ax.grid(alpha=.25);ax.legend()
    fig.savefig(out/"CITIBIKE_NYC_DAILY_TRIP_VOLUME.png",dpi=160)
    plt.close(fig)
    # Aggregate K region data. No individual rider trajectories or raw ZIP uploads.
    with (out/"CITIBIKE_12_GROUP_REGION_SUMMARY.csv").open("w",encoding="utf8",newline="") as f:
        w=csv.writer(f);w.writerow(
            ["group_id","n_stations","center_lon","center_lat","train_rides_departure",
             "train_rides_arrival","mean_soft_entropy","train_weekday_morning_net_arrivals",
             "train_weekday_evening_net_arrivals"])
        for k in range(K):
            r=[i for i in range(len(rows)) if kmeans.labels_[i]==k]
            center=kmeans.cluster_centers_[k]
            clat=y.mean()+center[1] # center y is 0-mean km; convert below
            centre_lat=lat.mean()+center[1]/111.32
            centre_lon=lon.mean()+center[0]/(111.32*np.cos(np.deg2rad(lat.mean())))
            w.writerow([k,len(r),round(float(centre_lon),7),round(float(centre_lat),7),
                sum(rows[i][4]["departures"] for i in r),
                sum(rows[i][4]["arrivals"] for i in r),
                round(float(norm_entropy[r].mean()),6),
                sum(int(morning[i]) for i in r),
                sum(int(night[i]) for i in r)])
    report={
        "dataset":"Citi Bike NYC official citywide monthly trip archives",
        "archive_source":"https://citibikenyc.com/system-data",
        "status":"preflight only; no OT fit; no real inventory claim",
        "months_train":["202604","202605","202606"],
        "months_validation":["202607"],
        "months_reserved_test_not_read":["202608","202609"],
        "official_input_archives":manifest,
        "trip_total_train":sum(source["month_trip_count"].values()),
        "trip_total_validation":sum(validation["month_trip_count"].values()),
        "train_all_station_ids":len(source["stations"]),
        "validation_all_station_ids":len(validation["stations"]),
        "stations_active_with_50_plus_train_events":len(rows),
        "group_K_for_visualization_only":K,
        "cluster_group_sizes":np.bincount(kmeans.labels_,minlength=K).tolist(),
        "soft_entropy_normalized_mean":float(norm_entropy.mean()),
        "soft_entropy_normalized_p10_p50_p90":
            np.quantile(norm_entropy,[.1,.5,.9]).tolist(),
        "soft_membership_temperature_for_visualization_only_km2":tau,
        "max_ride_induced_station_net_morning":int(morning.max()),
        "min_ride_induced_station_net_morning":int(morning.min()),
        "max_ride_induced_station_net_evening":int(night.max()),
        "min_ride_induced_station_net_evening":int(night.min()),
        "data_not_sufficient_for_actual_surplus_deficit":True,
        "real_inventory_data_needed":"historical timestamped GBFS station_status + capacity/target and service-move ambiguity",
        "plots":[
          "CITIBIKE_NYC_CITYWIDE_STATION_STRUCTURE.png",
          "CITIBIKE_NYC_HARD_SOFT_12GROUPS.png",
          "CITIBIKE_NYC_DAILY_TRIP_VOLUME.png"],
        "no_DANDI_or_Hyser_edits":True,
        "no_true_paired_label_transfer":True
    }
    (out/"REAL_CITYWIDE_DATA_PREFLIGHT.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input-dir",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    base=lambda:{"stations":{},"daily_trip_count":collections.Counter(),
                 "month_trip_count":collections.Counter()}
    data={"train":base(),"validation":base()}
    archives=[]
    for month in ["202604","202605","202606","202607"]:
        name=f"{month}-citibike-tripdata.zip"
        path=args.input_dir/name
        assert path.exists(),name
        print("READING_OFFICIAL_MONTH",month,path.stat().st_size,flush=True)
        entry=read_official_zip(path,month,data)
        archives.append(entry)
        print("DONE_MONTH",month,entry["validated_rides"],flush=True)
    res=visual(data["train"],data["validation"],args.output,archives)
    print("CITIBIKE_CITYWIDE_PREFLIGHT_COMPLETE",json.dumps({
        k:res[k] for k in ["trip_total_train","trip_total_validation",
                          "stations_active_with_50_plus_train_events",
                          "data_not_sufficient_for_actual_surplus_deficit"]}),flush=True)

if __name__=="__main__":
    main()
