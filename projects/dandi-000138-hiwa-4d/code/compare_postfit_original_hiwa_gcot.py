"""POST-FIT ONLY same-scale strict-unpaired 3-way evaluation.

DOES NOT TRAIN OR SELECT MODELS. Original PyHiWA independent X/Y KMeans
R/P are loaded after their 5-seed fits. Compare using EXACT same original
test windows, independent test OT indices, metrics as same-kernel hard/soft.
No test metric feeds the already trained R/P. Group-coupled P-Q nested cost
for original PyHiWA is NOT claimed because original local Q were not saved.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def eval_saved(x,y,R,ix,iy):
    out=x@R.T
    e=out-y
    d=cdist(out[ix],y[iy],metric="sqeuclidean")
    i,j=linear_sum_assignment(d)
    SSE=float(np.sum(e**2))
    return {
      "global_free_W2sq":float(np.mean(d[i,j])),
      "MAE":float(np.abs(e).mean()),
      "sync_mean_Euclidean":float(np.linalg.norm(e,axis=1).mean()),
      "sync_relative_MSE":float(SSE/np.sum(y**2)),
      "sync_R2":float(1-SSE/np.sum((y-y.mean(0))**2)),
    }

def stats(a):
    return {k:{"mean":float(np.mean([v[k] for v in a])),
                "sample_sd":float(np.std([v[k] for v in a],ddof=1)),
                "by_seed":[v[k] for v in a]} for k in a[0]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--same-solver",type=Path,required=True)
    p.add_argument("--clusters",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    h=json.loads(a.original.read_text(encoding="utf8"))
    c=json.loads(a.same_solver.read_text(encoding="utf8"))
    assert h["data_sha256"]==c["input_sha256"]==hashlib.sha256(a.clusters.read_bytes()).hexdigest()
    assert not h["group_supervision_diagnostic"]
    assert c["source_and_target_hard_kmeans_independent"]
    assert c["source_and_target_soft_prototypes_independent"]
    assert c["no_synchronized_pair_ids_used_in_fitting"]
    with np.load(a.clusters,allow_pickle=False) as z:
        X=np.asarray(z["neural4"],float)
        Y=np.asarray(z["hand4"],float)
        tr=np.asarray(z["train_mask"],bool)
        te=np.asarray(z["test_mask"],bool)
        assert not np.array_equal(z["neural_groups"],z["hand_groups"])
        assert "shared_labels" not in z.files
    assert np.allclose(h["source_mean"],c["source_mean"]) and np.allclose(h["target_mean"],c["target_mean"])
    assert abs(h["source_radius"]-c["source_radius"])<1e-12
    assert abs(h["target_radius"]-c["target_radius"])<1e-12
    x=((X-np.asarray(h["source_mean"]))/h["source_radius"])[te]
    y=((Y-np.asarray(h["target_mean"]))/h["target_radius"])[te]
    rg=np.random.default_rng(20261117)
    ix=rg.choice(len(x),200,replace=False)
    iy=rg.choice(len(y),200,replace=False)
    base=eval_saved(x,y,np.eye(4),ix,iy)
    assert abs(np.sqrt(base["global_free_W2sq"])-h["baseline_identity"]["ot_rmse"])<1e-10
    sets={
      "Original PyHiWA independent Hard":h["seed_results"],
      "ZIP GCOT kernel independent Hard":c["runs"]["hard"],
      "ZIP GCOT kernel independent Soft":c["runs"]["soft"],
    }
    values={}
    for name,seeds in sets.items():
        assert [int(q["seed"]) for q in seeds]==[42,43,44,45,46]
        v=[]
        for q in seeds:
            m=eval_saved(x,y,np.asarray(q["R"]),ix,iy)
            if name=="Original PyHiWA independent Hard":
                assert abs(np.sqrt(m["global_free_W2sq"])-q["metrics"]["ot_rmse"])<1e-10
            else:
                for k,other in [
                   ("global_free_W2sq","global_free_W2sq"),("MAE","MAE"),
                   ("sync_mean_Euclidean","mean_sync_euclidean"),
                   ("sync_relative_MSE","sync_relative_MSE"),("sync_R2","sync_R2")]:
                    assert abs(m[k]-q["metrics"][other])<1e-10,(name,k)
            v.append(m)
        values[name]={"stats":stats(v),"all_convergence_flags":[
           bool(q["strict_converged"] if name.startswith("Original")
                else q["admm_criteria_passed"]) for q in seeds]}
    result={
      "study":"Three-way postfit only, independently grouped 4D neural hand alignment",
      "all_methods_trained_without_true_pairing":True,
      "test_used_only_postfit":True,
      "same_test_indices_and_scale":True,
      "source_data_sha256":hashlib.sha256(a.clusters.read_bytes()).hexdigest(),
      "reference_note":"Original PyHiWA and ZIP GCOT core are different numerical solvers; comparison is reference only, not pure membership ablation.",
      "original_hiwa_nested_Q_note":"Original PyHiWA did not save final local Q, so no claimed exact training P-Q total cost; only common free W2sq and synchronous error here.",
      "test_baseline":base,"results":values,
    }
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/"THREE_WAY_POSTFIT.json").write_text(
       json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    label=[
      ("Global exact OT W2 squared ↓","global_free_W2sq"),
      ("Postfit relative paired MSE ↓","sync_relative_MSE"),
      ("Postfit mean Euclidean ↓","sync_mean_Euclidean"),
      ("Postfit MAE ↓","MAE"),("Postfit standard R2 ↑","sync_R2")]
    lines=[
      "# Strict unpaired 3-way comparable postfit scores",
      "",
      "All three models used original DANDI 000138 4D data and original independent KMeans labels as applicable; soft source and target prototypes were fit independently in their own 4D domains.",
      "The metric calculation begins ONLY after all fits; no test pairing selected hyperparameters or permuted group labels. The test sample W2 assignment uses the same independently drawn 200 source/200 target points in each model.",
      "",
      "Important: original uploaded PyHiWA and ZIP GCOT are different numerical implementations. **Only the latter two form the same-solver Hard/Soft controlled comparison.**",
      "",
      "| Metric | Identity | Original PyHiWA / independent Hard | Same GCOT / independent Hard | Same GCOT / independent Soft |",
      "|---|---:|---:|---:|---:|"]
    keys=list(sets)
    for title,k in label:
        zz=[values[name]["stats"][k] for name in keys]
        lines.append("| "+title+" | "+f"{base[k]:.6f}"+" | "+" | ".join(
         f"{z['mean']:.6f} ± {z['sample_sd']:.6f}" for z in zz)+" |")
    lines.extend(["",
      "Original PyHiWA strict numerical convergence flags (5 seeds): "+
      repr(values[keys[0]]["all_convergence_flags"]),
      "GCOT Hard ADMM convergence flags: "+
      repr(values[keys[1]]["all_convergence_flags"]),
      "GCOT Soft ADMM convergence flags: "+
      repr(values[keys[2]]["all_convergence_flags"]),
      "",
      "Original PyHiWA final local sample OT Q not saved, therefore original nested P-Q total training loss CANNOT be reconstructed accurately; use common W2 squared for this 3-way table and the separate test P-Q scores for same-solver hard/soft.",
      "",
      "For full P-Q geometry and raw entropic objective of original GCOT kernel see STRICT_UNSUPERVISED_REPORT.md.",
      ""]) 
    (a.output/"THREE_WAY_POSTFIT.md").write_text("\n".join(lines),encoding="utf8")
    print("POSTFIT 3-WAY VERIFIED SUMMARY",json.dumps({
        "baseline":base,"mean":{k:{met:obj["mean"] for met,obj in val["stats"].items()} for k,val in values.items()},
        "convergence":{k:v["all_convergence_flags"] for k,v in values.items()}
    },indent=2),flush=True)

if __name__=="__main__":
    main()
