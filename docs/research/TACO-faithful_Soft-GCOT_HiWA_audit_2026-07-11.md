# TACO-faithful Soft-GCOT HiWA transfer audit

Date: 2026-07-11  
Scope: MiHiA neural--movement baseline, not CC-HiWA/PBMC exploration.

## Verdict

The pre-existing `SoftHiWA` already contained the essential hybrid: TACO-style learned soft prototypes and weighted per-group measures inside the HiWA hierarchical OT/ADMM solver. It was **not yet a complete reference baseline**, because every run silently used top-membership sparse supports and the stopping rule ignored the ADMM primal consensus residual. This change adds an explicit full-support reference and an ADMM-aware stopping diagnostic. With all extension weights at zero, the resulting full-support configuration is the frozen TACO-faithful Soft-GCOT HiWA baseline.

“Faithful” means a TACO/GCOT transfer into the HiWA setting, not a claim that TACO itself used HiWA ADMM: TACO contributes the prototypes, soft measures, $P$, and $Q_{ij}$ structure; HiWA contributes $R_{ij}$, global $R$, and ADMM consensus.

## Formula-to-code audit

| Required element | Status | Evidence |
|---|---|---|
| Learnable soft prototypes | Present | `soft_groups.learn_soft_groups` optimizes reconstruction, negative assignment entropy, and small L2 regularization. |
| Soft assignments | Present | temperature-softmax prototype assignment; each row is checked to sum to one. |
| Soft weighted group measures | Present | normalized assignment columns are `Q_ij` marginals. |
| Group transport $P$ | Present | entropic group Sinkhorn in `SoftHiWA.fit`. |
| Group marginals | Faithful to TACO | $P\in\mathcal U(1/S,1/S)$ (Birkhoff/uniform group marginal); soft marginals belong to each $Q_{ij}$. |
| Independent $Q_{ij}$ | Present | one weighted Sinkhorn problem is solved for each group pair. |
| Local/global rotations | Present | `_weighted_subspace_alignment` returns $R_{ij}$; global polar/SVD update returns $R$. |
| ADMM consensus | Present and strengthened | multiplier update retained; primal residual now recorded and required for early stopping. |
| Entropic regularization | Present | separate group and within-pair Sinkhorn temperatures. |
| Full support | Added | `support_mode="full"` keeps every sample for every soft group. |
| Sparse support | Retained as approximation | `support_mode="sparse"` truncates and renormalizes high-membership samples; it changes every affected $Q_{ij}$ and is not mathematically identical. |

## Deliberately excluded from the pure baseline

`representative_guidance_weight`, `representative_rotation_weight`, `component_conditioning_weight`, and `rotation_anchor_weight` are all zero. Joint prototypes are not called. ROCA is not called. The unified runner serializes this contract into its JSON output.

`cc_hiwa.py` implements the one-shot compatibility construction $S=A P B^\top$ and a single global sample OT. It is a useful **exploratory CC-HiWA extension**, but it is not TACO GCOT's family of $Q_{ij}$ and is not used by the baseline. `joint_prototypes.py` is also exploratory.

## Minimal neural result

Run: `experiments/hiwa/run_taco_faithful_baseline.py --seeds 0 --tag 20260711_minimal`.

The run is a deterministic 96-by-96 neural smoke subproblem with the intentionally short `baseline-mini` iteration budget. It proves execution and diagnostic wiring, not convergence or a multi-seed performance claim.

| Method | Direction accuracy | Movement $R^2$ | Runtime (s) |
|---|---:|---:|---:|
| Hard HiWA | 0.5729 | 0.5587 | 0.535 |
| Soft-GCOT full | 0.5625 | 0.5307 | 0.763 |
| Soft-GCOT sparse | 0.3333 | 0.1973 | 0.478 |
| Soft-GCOT + ROCA | 0.2813 | -1.2193 | 0.651 |

Full support is close to Hard HiWA in this smoke run; sparse support is materially different. ROCA selected det$=-1$ from unlabeled representatives, but its high-assignment-entropy warning fired and it did not improve either evaluation metric. This is negative evidence, not a reason to tune with labels.

## Next question

Before extending algorithms, run a pre-registered multi-seed, adequately converged full-vs-sparse comparison. The most informative question is whether full-support Soft-GCOT has a stable performance/compute trade-off relative to Hard HiWA; only then should ROCA be reassessed on the stable full-support baseline.
