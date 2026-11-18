# Hyser 1250-window structural check (five real trials)

**This is a pre-training data diagnostic, not a Hard/Soft OT result.**
The 3D pooled PCA and whitening transforms here are for exploratory figures only.
Use the existing five trial-held-out 3D fold files for actual learning/evaluation.

## Cluster structure, K=4

| Domain | Silhouette ↑ | negative silhouettes | silhouette <0.2 | nearest centroid margin<0.15 | minimum ARI across 10 seeds | trial vs cluster NMI | Cramer's V |
|---|---:|---:|---:|---:|---:|---:|---:|
| source_EMG | 0.4739 | 3.6% | 14.0% | 3.8% | 1.000 | 0.039 | 0.182 |
| target_force | 0.4494 | 2.9% | 16.9% | 3.0% | 0.997 | 0.066 | 0.256 |

## Independent plots

For both EMG and force: raw 2D/3D, colored by five recording trials 2D/3D,
colored by independently learned K-means K=4 groups 2D/3D,
K=4 silhouette histograms and trial-versus-cluster stacked bars.
Same spatial axes are used for all scatter panels. Source and target are never overlaid.
Cluster colors are stable within a domain, but color names do not imply matched
cross-domain groups. Trial colors identify recording session, not an OT training label.

## Interpretation rules

A prominent four-cluster arrangement in both domains would give little room
for soft memberships to improve a good hard partition on this one criterion.
But silhouette alone does not establish whether soft OT is valuable.
A high trial-vs-cluster dependence could indicate session/batch effects.
Even if these are small, the paired ordering must not enter training.

## Numerical and reproducibility evidence

See diagnostic_summary.json, K2_K7_comparison.csv,
soft_membership_diagnostics.csv and per-window diagnostics.
