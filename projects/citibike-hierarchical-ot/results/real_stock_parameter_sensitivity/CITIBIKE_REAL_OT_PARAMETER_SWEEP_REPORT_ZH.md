# Citi Bike real historical stock: parameter-sensitivity and integer transport

**Research status: actual 2026-06-15 historical inventories; half-dock targets ASSUMED, not actual operator preferences.**
Fixed physical coordinates R=I, great-circle km proxy. All same-snapshot methods use the SAME integer supply and served-demand station marginals.
The globally exact unregularized flat integer min-cost flow is the true benchmark; no feasible Hard/Soft scheme can have lower pure cost.

## Frozen sensitivity grids
epsilon={0.25,0.5,1,2.5} km; K={4,8}; Soft temperature multiplier={0.1,0.25,0.5,1,2,4} for K8 eps.5.
Independent group hard/soft assignments use same KMeans centers; Soft top-2 membership, HARD onehot.
FA, SA, DAD are not defined as city transport without explicit adaptation; WA-I is the flat Sinkhorn curve.

## Numerical audit

Number of numerical nonconvergence cases (recorded, not silently removed): 0
Costs are pure <Gamma,C>, not regularized objective J; fractional Gamma requires exact row/col-preserving integerization.

## Key 0.5 km K8 result and integerization

| Snapshot | Method | fractional bike-km | rounded executable bike-km | exact flat bike-km |
|---|---|---:|---:|---:|
| 2026-06-15 13:00:00 | Hard | 16059.262219856644 | 14548.329127632684 (TV=0.651) | 11506.901 |
| 2026-06-15 13:00:00 | Soft | 16343.769027085233 | 13562.442276413258 (TV=0.695) | 11506.901 |
| 2026-06-15 20:00:00 | Hard | 17108.493030763093 | 15577.146986658085 (TV=0.698) | 12893.451 |
| 2026-06-15 20:00:00 | Soft | 17758.989806651196 | 15580.16090181366 (TV=0.745) | 12893.451 |

NOTE ON INTEGRAL DISPATCH: Exact residual min-cost rounding can nearly reoptimize the plan; its cheap routes are NOT evidence the original Soft hierarchy is more efficient. Min-L1 rounding follows the original fractional plan as closely as achievable on the fractional-edge support. Both disclose total variation fraction of total transported mass. Neither corresponds to an operator-approved dispatch.

## Figures
- [Epsilon cost-gap curves](FIG_EPSILON_HARD_SOFT_K4_K8_COST_GAP.png)
- [Soft temperature curves](FIG_SOFT_TEMPERATURE_GROUP_OVERLAP_COST_GAP.png)
- [True integer rounding](FIG_INTEGER_ROUNDING_HARD_SOFT_VS_EXACT.png)

## Limitations
- Only two time points on the SAME date, not independent days. The full multi-month dataset is NOT used in OT yet.
- Geographic distance is not truck road routing or real operational cost; stock target is purely ASSUMED.
- Clusters spatial only; their algorithmic advantages are hypothesized, not validated by one soft temperature sweep.
- Direct SA and DAD comparisons require a genuinely defined city-transport adaptation; these methods are NOT scored here.
- Integer rounding solves a new residual problem after the fractional plan; it is not mathematically the original Q's optimum.
- This project is isolated from DANDI/Hyser; source data remains out of Git.
