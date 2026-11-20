# Citi Bike NYC: FIRST actual baseline vs Hard/Soft hierarchical OT pilot

**Real historical station inventory; 50% dock target is a POLICY ASSUMPTION, not measured demand or actual operator dispatch.**
Source: CityBikes NYC 2026-06 status Parquet, exact file hash in input audit.
No learned geometric R; fix R=I. Great-circle km are proxy logistics costs, not road distance.
All algorithms share the same selected integer serviced deficits; remaining deficits are explicit, not silently normalized.
Hard/Soft Q and Gamma are **fractional** entropy-regularized transport and need integerization before a physical vehicle dispatch.
Flat exact mincost uses an unrestricted full bipartite network and integer flows; this is the lower-bound reference under these policy assumptions.
WA-I is exactly the shown flat Sinkhorn; SA and DAD are not applicable without new explicitly adapted definitions.

| Snapshot | Method | K | Pure transport cost (bike-km) | gap vs flat exact | Row/column valid |
|---|---|---:|---:|---:|---|
| 2026-06-15 13:00:00 | Exact flat min-cost flow | – | 11506.90 | 0.00% | integer exact |
| 2026-06-15 13:00:00 | Greedy feasible | – | 16141.61 | 40.28% | integer exact |
| 2026-06-15 13:00:00 | Flat Sinkhorn / WA-I | – | 23671.94 | 105.72% | fractional approx |
| 2026-06-15 13:00:00 | Hard | 4 | 21985.82 | 91.07% | fractional, max marginal err 1.774e-11 bike |
| 2026-06-15 13:00:00 | Soft | 4 | 22500.29 | 95.54% | fractional, max marginal err 2.609e-10 bike |
| 2026-06-15 13:00:00 | Hard | 8 | 19384.23 | 68.46% | fractional, max marginal err 3.553e-14 bike |
| 2026-06-15 13:00:00 | Soft | 8 | 20998.04 | 82.48% | fractional, max marginal err 2.842e-14 bike |
| 2026-06-15 20:00:00 | Exact flat min-cost flow | – | 12893.45 | 0.00% | integer exact |
| 2026-06-15 20:00:00 | Greedy feasible | – | 17660.94 | 36.98% | integer exact |
| 2026-06-15 20:00:00 | Flat Sinkhorn / WA-I | – | 25054.77 | 94.32% | fractional approx |
| 2026-06-15 20:00:00 | Hard | 4 | 23107.79 | 79.22% | fractional, max marginal err 1.14e-11 bike |
| 2026-06-15 20:00:00 | Soft | 4 | 24859.56 | 92.81% | fractional, max marginal err 3.778e-10 bike |
| 2026-06-15 20:00:00 | Hard | 8 | 20235.32 | 56.94% | fractional, max marginal err 2.487e-14 bike |
| 2026-06-15 20:00:00 | Soft | 8 | 22238.20 | 72.48% | fractional, max marginal err 4.263e-14 bike |

**Interpretation:** Flat exact must be <= any valid fractional plan on the same cost, to solver/roundoff tolerance. Better Hard/Soft plan is a lower P/Q approximation gap, not better than the true flat optimum.
These are two snapshots of one day in a single month, not independent monthly generalization.
Source and target sites are never transformed in physical geography.
SA and DAD cannot be compared as original alignment algorithms on this dispatch cost.

Raw transport arrays and 16 (or K²) conditional plans saved in separate NPZ run artifacts; JSON stores P and metrics.
