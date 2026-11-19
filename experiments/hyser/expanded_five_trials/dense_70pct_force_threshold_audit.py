#!/usr/bin/env python3
"""Dense post-fit threshold curves from frozen real Hyser full1000 results.

No OT refitting, no cross-modal pair supervision during training, no model selection.
The q=70% crossing is an evaluation-only descriptive statistic (NOT a tuned threshold).
An exact chance baseline uses all 250 held-out forces x all 1000 train forces
for each of the five folds, so it does not depend on Monte Carlo randomness.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

METHODS = ("hard", "soft05")
REPS = ("PCA3_PCA3", "FA3_FA3", "PCA5_RAW5", "FA5_RAW5")
Q_ORIGINAL = (0.01, 0.025, 0.05, 0.10, 0.20)


def training_null_distances(force, ids):
    """Exactly reproduce the older training-only calibration RNG stream."""
    rng = np.random.default_rng(20261119)
    trial_values = np.unique(ids)
    assert len(trial_values) == 4
    assert all(np.sum(ids == i) == 250 for i in trial_values)
    members = {i: np.flatnonzero(ids == i) for i in trial_values}
    result = np.empty(6000, dtype=np.float64)
    for h in range(6000):
        ta, tb = rng.choice(trial_values, size=2, replace=False)
        ia = rng.choice(members[ta])
        ib = rng.choice(members[tb])
        result[h] = np.linalg.norm(force[ia] - force[ib])
    return np.sort(result)


def empirical_hit(errors, eps):
    return float(np.mean(np.asarray(errors) <= eps))


def writecsv(path, entries):
    entries = list(entries)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        wr = csv.DictWriter(fp, fieldnames=list(entries[0]))
        wr.writeheader()
        wr.writerows(entries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=Path,
                        help="Extracted release's per-representation 1000train/250test NPZs")
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    records, sources = [], {}
    for dim in ("3D", "5D"):
        path = args.results_root / dim / "full_physical_distance_curves.json"
        raw = path.read_bytes()
        sources[dim] = hashlib.sha256(raw).hexdigest()
        obj = json.loads(raw)
        assert obj["n_runs"] == 40 and len(obj["records"]) == 40
        records.extend(obj["records"])

    by_key = {}
    for rec in records:
        key = (rec["config"], rec["method"], int(rec["fold"]), int(rec["seed"]))
        assert key not in by_key, key
        assert len(rec["results"]["physical_errors_250"]) == 250
        by_key[key] = np.asarray(rec["results"]["physical_errors_250"], float)
    assert len(by_key) == 80

    cal, null = {}, {}
    for fold in range(1, 6):
        fp = args.features / "PCA5_RAW5" / f"trial_heldout_{fold}_5d.npz"
        with np.load(fp) as z:
            ytrain = np.asarray(z["original_target_force_train"], float)
            ytest = np.asarray(z["original_target_force_test"], float)
            trial_ids = np.asarray(z["train_trial_id"], int)
        assert ytrain.shape == (1000, 5) and ytest.shape == (250, 5)
        cal[fold] = training_null_distances(ytrain, trial_ids)
        # Exact uniform-random target-bank probability, on the real held-out trial.
        null[fold] = np.sort(cdist(ytest, ytrain).ravel())
        archived = next(x for x in records if x["fold"] == fold)
        for q in Q_ORIGINAL:
            reference = archived["train_only_thresholds"][str(q)]
            computed = np.quantile(cal[fold], q)
            assert abs(reference - computed) < 1e-10, (fold, q, reference, computed)

    egrid = np.round(np.arange(0.0, 0.801, 0.005), 6)
    qgrid = np.round(np.linspace(0.01, 0.90, 90), 6)
    curve_rows, q_rows, per_fold, threshold_rows = [], [], [], []

    for conf in REPS:
        for method in METHODS:
            hits = []
            for fold in range(1, 6):
                for seed in (42, 43):
                    hits.append((fold, by_key[conf, method, fold, seed]))
            all_errors = np.concatenate([x for _, x in hits])
            assert len(all_errors) == 2500
            # 'higher' ensures 70% or above, instead of an interpolated quantile.
            eps70 = float(np.quantile(all_errors, 0.70, method="higher"))
            eps50 = float(np.quantile(all_errors, 0.50, method="higher"))
            min_cdf = empirical_hit(all_errors, eps70)
            random70 = float(np.mean([empirical_hit(null[f], eps70) for f in range(1, 6)]))
            cal_percentile = float(np.mean([empirical_hit(cal[f], eps70) for f in range(1, 6)]))
            threshold_rows.append({
                "representation": conf, "method": method,
                "physical_epsilon_for_at_least_70pct": eps70,
                "actual_hit_at_epsilon": min_cdf,
                "random_bank_exact_hit_at_same_epsilon": random70,
                "hit_minus_random_at_same_epsilon": min_cdf - random70,
                "train_unrelated_distance_percentile_at_epsilon": cal_percentile,
                "epsilon_at_50pct_model_hit": eps50,
            })

            for fold in range(1, 6):
                errs = np.concatenate([x for f, x in hits if f == fold])
                per_fold.append({
                    "representation": conf, "method": method, "heldout_trial": fold,
                    "hit_at_raw_epsilon_0_325": empirical_hit(errs, 0.325),
                    "hit_at_pooled_epsilon_70": empirical_hit(errs, eps70),
                    "physical_error_mean": float(np.mean(errs)),
                    "foldwise_epsilon_70": float(np.quantile(errs, 0.70, method="higher")),
                    "train_nonmatch_median": float(np.median(cal[fold])),
                })

            for eps in egrid:
                curve_rows.append({
                    "representation": conf, "method": method,
                    "physical_epsilon": float(eps),
                    "hit": empirical_hit(all_errors, eps),
                    "random_bank_exact_hit": float(np.mean(
                        [empirical_hit(null[f], eps) for f in range(1, 6)])),
                })
            for q in qgrid:
                rate = float(np.mean([
                    empirical_hit(errs, float(np.quantile(cal[fold], q)))
                    for fold, errs in hits
                ]))
                chance = float(np.mean([
                    empirical_hit(null[fold], float(np.quantile(cal[fold], q)))
                    for fold in range(1, 6)
                ]))
                q_rows.append({
                    "representation": conf, "method": method,
                    "train_only_unrelated_pair_quantile": float(q),
                    "hit": rate, "random_bank_exact_hit": chance,
                    "hit_minus_random": rate - chance,
                })

    writecsv(args.out / "dense_physical_distance_curves.csv", curve_rows)
    writecsv(args.out / "dense_training_quantile_curves.csv", q_rows)
    writecsv(args.out / "epsilon_70_exact_null.csv", threshold_rows)
    writecsv(args.out / "per_trial_threshold_stability.csv", per_fold)
    (args.out / "audit_provenance.json").write_text(json.dumps({
        "source_SHA256": sources,
        "n_full1000_frozen_runs": len(by_key), "n_paired_fold_seed_points_per_model": 2500,
        "folds_are_one_subject_one_session": True,
        "random_baseline": "EXACT all heldout test target forces x all training target forces",
        "calibration": "6000 random pairs sampled from four distinct training trials, seed=20261119",
        "no_refitting_or_heldout_parameter_selection": True,
        "meaning_of_70pct": "descriptive test-set distance CDF crossing; not a performance-validated threshold",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7), sharex=True, sharey=True,
                            constrained_layout=True)
    for ax, dim, reps in zip(axes, (3, 5), (REPS[:2], REPS[2:])):
        ax.plot(egrid, [np.mean([empirical_hit(null[f], e) for f in range(1, 6)])
                             for e in egrid], "k--", lw=2, label="Uniform random train-bank")
        for conf in reps:
            for method in METHODS:
                ee = np.concatenate([by_key[conf, method, fold, seed]
                        for fold in range(1, 6) for seed in (42, 43)])
                ax.plot(egrid, [empirical_hit(ee, e) for e in egrid],
                        label=f"{conf.split('_')[0]} {method}", lw=1.8)
        ax.axhline(0.7, color="gray", ls=":", lw=1.5)
        ax.set(xlabel="Physical five-finger L2 error tolerance (original units)",
               ylabel="Hit rate", title=f"Hyser frozen {dim}D models")
        ax.set_xlim(0, 0.65)
        ax.set_ylim(0, 1.01)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc="lower right")
    fig.savefig(args.out / "dense_physical_threshold_with_exact_random.png", dpi=185)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7), sharex=True, sharey=True,
                            constrained_layout=True)
    for ax, dim, reps in zip(axes, (3, 5), (REPS[:2], REPS[2:])):
        ref = [x for x in q_rows if x["representation"] == reps[0] and x["method"] == "hard"]
        ax.plot(qgrid, [x["random_bank_exact_hit"] for x in ref],
                "k--", lw=2, label="Exact uniform random train-bank")
        for conf in reps:
            for method in METHODS:
                z = [x for x in q_rows if x["representation"] == conf and x["method"] == method]
                ax.plot(qgrid, [x["hit"] for x in z], lw=1.8,
                        label=f"{conf.split('_')[0]} {method}")
        ax.axhline(.7, color="gray", ls=":", lw=1.5)
        ax.set(xlabel="Quantile of training-only unrelated five-finger distances",
               ylabel="Hit rate", title=f"Hyser frozen {dim}D models")
        ax.set_xlim(0, .9)
        ax.set_ylim(0, 1.01)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc="lower right")
    fig.savefig(args.out / "training_quantile_threshold_with_exact_random.png", dpi=185)
    plt.close(fig)

    summary = [
        "# Hyser：70% 阈值真实测量及随机基线审计",
        "",
        "这是冻结模型上的**事后评价**，不是训练调参，不得把在外测试集上找到的70%阈值反过来用于模型选择。",
        "每模型5个留出试次×2个随机种子×250点，共2500条预测五指力量欧氏距离。",
        "随机基线为同一留出试次所有250个真实力量与训练库1000个目标力量的完整25万对距离；五折共125万对，**不是**训练中真实同步配对。",
        "训练阈值由四个训练试次之间6000次独立于外测试的随机非对应力量抽样估计。",
        "",
        "|表示|方法|达到70%所需ε|同ε模型Hit|同ε随机Hit|模型−随机|训练距离分位|",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in threshold_rows:
        summary.append(
            f"|{r['representation']}|{r['method']}|{r['physical_epsilon_for_at_least_70pct']:.5f}"
            f"|{r['actual_hit_at_epsilon']:.2%}|{r['random_bank_exact_hit_at_same_epsilon']:.2%}"
            f"|{r['hit_minus_random_at_same_epsilon']:+.2%}"
            f"|{r['train_unrelated_distance_percentile_at_epsilon']:.2%}|"
        )
    summary.extend([
        "",
        "## 关键限制",
        "",
        "1. 不同试次间的表现不稳定，必须同时检查 per_trial_threshold_stability.csv。",
        "2. 这只是在物理容忍度增大时观察到的CDF变化，70%不等于准确反演；应同时看同阈值随机基线及误差绝对量。",
        "3. q 分位阈值只使用训练力量校准，不能用测试误差选择最优 q 再声称泛化。",
        "4. 所有结果属于一个被试的一次会话；不能将两种子或五试次当作10位独立被试。",
        "",
        "## 图",
        "",
        "![距离阈值曲线](dense_physical_threshold_with_exact_random.png)",
        "",
        "![训练分位阈值曲线](training_quantile_threshold_with_exact_random.png)",
        "",
    ])
    (args.out / "DENSE_THRESHOLD_70PCT_REPORT_ZH.md").write_text(
        "\n".join(summary) + "\n", encoding="utf-8")

    assert len(threshold_rows) == 8 and len(per_fold) == 40
    print(json.dumps({"success": True, "n_models": 8, "n_runs": len(by_key),
                      "first_70": threshold_rows[0],
                      "outputs": [x.name for x in args.out.iterdir()]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
