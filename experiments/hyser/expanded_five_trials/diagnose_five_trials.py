#!/usr/bin/env python3
"""Pre-experiment structural diagnostics on all 5 real Hyser trials.

Exploratory use of all windows only. Never use these pooled transforms or
post-hoc cluster colors to fit a held-out OT model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.special import softmax
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score, silhouette_samples, silhouette_score,
    davies_bouldin_score, calinski_harabasz_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler

TRIAL_COLORS = ["#2776ae", "#dc9456", "#4f9c83", "#9062b0", "#d25171"]
CLUSTER_COLORS = ["#2776ae", "#dc9456", "#4f9c83", "#9062b0"]
PAIRS = [(0, 1), (0, 2), (1, 2)]


def whiten(z):
    z = np.asarray(z, dtype=float)
    mean = z.mean(0)
    centered = z - mean
    covariance = np.cov(centered, rowvar=False)
    val, vec = np.linalg.eigh(covariance)
    if val.min() <= 1e-10:
        raise ValueError("singular pooled covariance")
    W = (vec * (val ** (-0.5))) @ vec.T
    return centered @ W


def render_2d(name, data, labs, colors, title, names, output, limit):
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.55), constrained_layout=True)
    for ax, (i, j) in zip(axes, PAIRS):
        if labs is None:
            ax.scatter(data[:, i], data[:, j], c="#5b7080", s=14,
                       alpha=.65, edgecolors="none", rasterized=True)
        else:
            for k, color in enumerate(colors):
                mask = labs == k
                ax.scatter(data[mask, i], data[mask, j], c=color, s=15, alpha=.75,
                           edgecolors="none", label=f"{names[k]} (n={mask.sum()})",
                           rasterized=True)
        ax.set(xlabel=f"axis {i+1}", ylabel=f"axis {j+1}",
               xlim=(-limit, limit), ylim=(-limit, limit))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.12)
        ax.set_title(f"Projection {i+1}-{j+1}")
    if labs is not None:
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="outside lower center", ncol=len(colors),
                   frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=15)
    fig.savefig(output / f"{name}_2d.png", dpi=165, bbox_inches="tight")
    plt.close(fig)


def render_3d(name, data, labs, colors, title, names, output, limit):
    fig = plt.figure(figsize=(8.2, 7.8))
    ax = fig.add_subplot(111, projection="3d")
    if labs is None:
        ax.scatter(data[:, 0], data[:, 1], data[:, 2],
                   c="#5b7080", s=12, alpha=.62, depthshade=False)
    else:
        for k, color in enumerate(colors):
            mask = labs == k
            ax.scatter(data[mask, 0], data[mask, 1], data[mask, 2], c=color,
                       s=13, alpha=.77, edgecolors="none", depthshade=False,
                       label=f"{names[k]} (n={mask.sum()})")
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1.02), ncol=2,
                  frameon=False, fontsize=9)
    ax.set(xlabel="axis 1", ylabel="axis 2", zlabel="axis 3",
           xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-59)
    fig.suptitle(title, fontsize=14)
    fig.savefig(output / f"{name}_3d.png", dpi=165, bbox_inches="tight")
    plt.close(fig)


def save_csv(filename, rows):
    with filename.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def diagnose(name, Z, trials, out, limit):
    random_seeds = range(40, 50)
    models = {}
    kscan = []
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=30).fit(Z)
        models[k] = km
        kscan.append({
            "domain": name, "K": k,
            "silhouette": float(silhouette_score(Z, km.labels_)),
            "davies_bouldin": float(davies_bouldin_score(Z, km.labels_)),
            "calinski_harabasz": float(calinski_harabasz_score(Z, km.labels_)),
            "cluster_sizes": ",".join(map(str, np.bincount(km.labels_))),
        })
    km4 = models[4]
    label = km4.labels_
    sil = silhouette_samples(Z, label)
    center_d = cdist(Z, km4.cluster_centers_, metric="sqeuclidean")
    sorted_d = np.sort(center_d, axis=1)
    boundary_margin = (sorted_d[:, 1] - sorted_d[:, 0]) / np.maximum(sorted_d[:, 1], 1e-12)
    # A visual index of how near the best and runner-up centroid are.
    pct_boundary = float(np.mean(boundary_margin < 0.15))
    pct_low_sil = float(np.mean(sil < 0.2))
    pct_neg_sil = float(np.mean(sil < 0))
    seeds_labels = [
        KMeans(n_clusters=4, random_state=seed, n_init=20).fit_predict(Z)
        for seed in random_seeds
    ]
    aris = [adjusted_rand_score(seeds_labels[i], seeds_labels[j])
            for i in range(len(seeds_labels)) for j in range(i+1, len(seeds_labels))]
    tab = np.zeros((5, 4), dtype=int)
    for trial, group in zip(trials, label):
        tab[trial - 1, group] += 1
    chi2, _, _, _ = chi2_contingency(tab)
    cramers_v = float(np.sqrt(chi2 / (len(Z) * min(5-1, 4-1))))
    trial_nmi = float(normalized_mutual_info_score(trials, label))
    mix = []
    for tau in (0.25, 0.5, 1.0):
        weights = softmax(-center_d / tau, axis=1)
        max_w = weights.max(axis=1)
        ent = -np.sum(weights * np.log(np.maximum(weights, 1e-300)), axis=1)/np.log(4)
        mix.append({
            "domain": name, "temperature": tau, "ambiguous_max_lt_0p8_fraction": float(np.mean(max_w < 0.8)),
            "ambiguous_max_lt_0p6_fraction": float(np.mean(max_w < 0.6)),
            "mean_normalized_membership_entropy": float(np.mean(ent)),
        })
    summary = {
        "domain": name, "n_samples": int(len(Z)),
        "K": 4, "cluster_sizes": np.bincount(label).tolist(),
        "silhouette_mean": float(np.mean(sil)),
        "silhouette_median": float(np.median(sil)),
        "silhouette_p10": float(np.percentile(sil, 10)),
        "silhouette_p90": float(np.percentile(sil, 90)),
        "fraction_silhouette_lt_0p2": pct_low_sil,
        "fraction_silhouette_negative": pct_neg_sil,
        "fraction_near_boundary_margin_lt_0p15": pct_boundary,
        "ARI_10_KMeans_seeds_mean": float(np.mean(aris)),
        "ARI_10_KMeans_seeds_min": float(np.min(aris)),
        "NMI_trial_vs_cluster": trial_nmi,
        "Cramers_V_trial_vs_cluster": cramers_v,
        "trials_by_clusters": tab.tolist(),
        "k_scan": kscan, "membership_mixing": mix
    }
    render_2d(f"{name}_raw", Z, None, None,
              f"Hyser {name}: 1250 points (uncolored)", None, out, limit)
    render_3d(f"{name}_raw", Z, None, None,
              f"Hyser {name}: 1250 points (uncolored)", None, out, limit)
    render_2d(f"{name}_by_trial", Z, trials-1, TRIAL_COLORS,
              f"Hyser {name}: 5 trial labels (NOT learned clusters)",
              [f"trial {i}" for i in range(1, 6)], out, limit)
    render_3d(f"{name}_by_trial", Z, trials-1, TRIAL_COLORS,
              f"Hyser {name}: 5 trial labels", [f"trial {i}" for i in range(1, 6)], out, limit)
    render_2d(f"{name}_kmeans4", Z, label, CLUSTER_COLORS,
              f"Hyser {name}: independent K-means K=4",
              [f"cluster {i+1}" for i in range(4)], out, limit)
    render_3d(f"{name}_kmeans4", Z, label, CLUSTER_COLORS,
              f"Hyser {name}: independent K-means K=4",
              [f"cluster {i+1}" for i in range(4)], out, limit)
    # Single domain silhouette distribution, no target/source overlay
    fig, ax = plt.subplots(figsize=(7.7, 4.9))
    ax.hist(sil, bins=35, alpha=.87, color="#3c799d", edgecolor="white")
    ax.axvline(.2, color="#cc6358", linewidth=1.5, linestyle="--",
               label="silhouette = 0.2")
    ax.axvline(np.mean(sil), color="#254c86", linewidth=2.1,
               label=f"mean = {np.mean(sil):.3f}")
    ax.set(xlabel="per-point silhouette", ylabel="number of windows",
           title=f"Hyser {name}: K=4 individual point separation")
    ax.legend(frameon=False)
    fig.savefig(out / f"{name}_silhouette_hist.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    # Per-trial proportions: batch effects can cause an apparent cluster.
    fig, ax = plt.subplots(figsize=(8.6, 5))
    frac = tab / tab.sum(axis=1, keepdims=True)
    bottom = np.zeros(5)
    for j in range(4):
        ax.bar(range(1, 6), frac[:, j], bottom=bottom, color=CLUSTER_COLORS[j],
               label=f"cluster {j+1}")
        bottom += frac[:, j]
    ax.set(ylim=(0, 1), xticks=range(1, 6), xlabel="recording trial",
           ylabel="fraction of windows", title=f"Hyser {name}: cluster composition by recording trial")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=4, frameon=False)
    fig.savefig(out / f"{name}_trial_cluster_mix.png", dpi=165, bbox_inches="tight")
    plt.close(fig)
    save_csv(out / f"{name}_per_window_diagnostics.csv", [
        {"trial": int(trial), "window_index": int(idx % 250), "cluster": int(label[idx]+1),
         "point_silhouette": float(sil[idx]), "centroid_margin": float(boundary_margin[idx])}
        for idx, trial in enumerate(trials)
    ])
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with np.load(args.data) as z:
        X256 = np.asarray(z["X_emg_log1p_rms_256"], dtype=np.float64)
        Y3 = np.asarray(z["Y_force_thumb_middle_index_3"], dtype=np.float64)
        trials = np.asarray(z["trial_id"], dtype=int)
    assert X256.shape == (1250, 256), X256.shape
    assert Y3.shape == (1250, 3), Y3.shape
    assert np.array_equal(np.bincount(trials)[1:], np.full(5, 250))
    # Pooled transforms for one descriptive visualization only; NEVER use for
    # leave-one-trial-out model fitting. Those files are fold-specific.
    scaler = StandardScaler()
    pca = PCA(n_components=3, svd_solver="full")
    Xpca = pca.fit_transform(scaler.fit_transform(X256))
    X = whiten(Xpca) / np.sqrt(3)
    Y = whiten(Y3) / np.sqrt(3)
    limit = float(math.ceil((max(np.abs(X).max(), np.abs(Y).max()) + .2) * 10) / 10)
    style = {"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
             "figure.facecolor": "white", "savefig.facecolor": "white"}
    plt.rcParams.update(style)
    src = diagnose("source_EMG", X, trials, out, limit)
    tgt = diagnose("target_force", Y, trials, out, limit)
    summaries = {"provenance": "Original Hyser s01/s1 five official random-task trials",
                 "sample_count_per_domain": 1250, "trials": 5,
                 "num_samples_per_trial": 250,
                 "note": "pooled transform is EDA-only; held-out OT must use trial-specific transforms",
                 "pca3_explained_variance": pca.explained_variance_ratio_.tolist(),
                 "common_axis_extent": limit,
                 "source": src, "target": tgt}
    (out / "diagnostic_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_csv(out / "K2_K7_comparison.csv", src["k_scan"]+tgt["k_scan"])
    save_csv(out / "soft_membership_diagnostics.csv", src["membership_mixing"]+tgt["membership_mixing"])
    save_csv(out / "K4_summary.csv", [
        {k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
         for k,v in s.items() if k not in ("k_scan","membership_mixing","trials_by_clusters")}
        for s in [src,tgt]
    ])
    # Side by side comparison of cluster visibility, using separate panels and
    # independent labels. Not overlaying source with target.
    fig, axes = plt.subplots(1,2,figsize=(12,4.5), constrained_layout=True)
    for ax, summary, domain in zip(axes,[src,tgt],["Source EMG","Target force"]):
        data = summary["k_scan"]
        ax.plot([v["K"] for v in data], [v["silhouette"] for v in data],
                marker="o", linewidth=2)
        ax.axvline(4, color="#888", linestyle="--", alpha=.6)
        ax.set(xticks=range(2,8), xlabel="number of clusters K",
               ylabel="silhouette", title=domain)
        ax.grid(alpha=.2)
    fig.suptitle("Hyser five real trials (1250 samples): cluster count sensitivity")
    fig.savefig(out / "K_comparison_1250.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    report = [
        "# Hyser 1250-window structural check (five real trials)",
        "", "**This is a pre-training data diagnostic, not a Hard/Soft OT result.**",
        "The 3D pooled PCA and whitening transforms here are for exploratory figures only.",
        "Use the existing five trial-held-out 3D fold files for actual learning/evaluation.",
        "", "## Cluster structure, K=4",
        "",
        "| Domain | Silhouette ↑ | negative silhouettes | silhouette <0.2 | nearest centroid margin<0.15 | minimum ARI across 10 seeds | trial vs cluster NMI | Cramer's V |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in [src,tgt]:
        report.append(
            f"| {s['domain']} | {s['silhouette_mean']:.4f} | {s['fraction_silhouette_negative']:.1%}"
            f" | {s['fraction_silhouette_lt_0p2']:.1%} | {s['fraction_near_boundary_margin_lt_0p15']:.1%}"
            f" | {s['ARI_10_KMeans_seeds_min']:.3f} | {s['NMI_trial_vs_cluster']:.3f}"
            f" | {s['Cramers_V_trial_vs_cluster']:.3f} |"
        )
    report += [
        "", "## Independent plots",
        "",
        "For both EMG and force: raw 2D/3D, colored by five recording trials 2D/3D,",
        "colored by independently learned K-means K=4 groups 2D/3D,",
        "K=4 silhouette histograms and trial-versus-cluster stacked bars.",
        "Same spatial axes are used for all scatter panels. Source and target are never overlaid.",
        "Cluster colors are stable within a domain, but color names do not imply matched",
        "cross-domain groups. Trial colors identify recording session, not an OT training label.",
        "", "## Interpretation rules",
        "",
        "A prominent four-cluster arrangement in both domains would give little room",
        "for soft memberships to improve a good hard partition on this one criterion.",
        "But silhouette alone does not establish whether soft OT is valuable.",
        "A high trial-vs-cluster dependence could indicate session/batch effects.",
        "Even if these are small, the paired ordering must not enter training.",
        "",
        "## Numerical and reproducibility evidence",
        "",
        "See diagnostic_summary.json, K2_K7_comparison.csv,",
        "soft_membership_diagnostics.csv and per-window diagnostics.",
    ]
    (out / "RESULTS.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    print(json.dumps({"success": True, "source": src, "target": tgt}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
