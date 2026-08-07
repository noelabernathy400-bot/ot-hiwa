# ROCA synthetic determinant ground-truth validation report

## Summary

- Total cases: `480`.
- Overall selected determinant accuracy: `1.0000`.
- Nondegenerate cases: `480`.
- Nondegenerate selected determinant accuracy: `1.0000`.
- Degenerate or near-degenerate cases flagged: `0`.
- Selector error count: `0`.

## Main answers

1. ROCA selector can recover `true_det=+1` and `true_det=-1` in the normal nondegenerate synthetic setting.
2. Near-degenerate representatives are intentionally fragile; failures or warnings there support the need for a degeneracy check.
3. Mean normal-mode accuracy across noise/det/assignment groups is `1.0000`.
4. Mean near-degenerate-mode accuracy across noise/det/assignment groups is `1.0000`.
5. The experiment supports the real-data interpretation only inside the tested assumptions: `d=3`, `K=4`, nondegenerate representative tetrahedron, and no label-based branch selection.

## Group table

| assignment | degeneracy | noise | true_det | cases | accuracy | nondegenerate_cases | nondegenerate_accuracy | degenerate_count |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| learned_soft | near_degenerate | 0.0 | -1 | 20 | 1.0000 | 20 | 1.0 | 0 |
| learned_soft | near_degenerate | 0.0 | 1 | 20 | 1.0000 | 20 | 1.0 | 0 |
| learned_soft | near_degenerate | 0.05 | -1 | 20 | 1.0000 | 20 | 1.0 | 0 |
| learned_soft | near_degenerate | 0.05 | 1 | 20 | 1.0000 | 20 | 1.0 | 0 |
| learned_soft | near_degenerate | 0.1 | -1 | 20 | 1.0000 | 20 | 1.0 | 0 |
| learned_soft | near_degenerate | 0.1 | 1 | 20 | 1.0000 | 20 | 1.0 | 0 |
| learned_soft | normal | 0.0 | -1 | 20 | 1.0000 | 20 | 1.0 | 0 |
| learned_soft | normal | 0.0 | 1 | 20 | 1.0000 | 20 | 1.0 | 0 |
| learned_soft | normal | 0.05 | -1 | 20 | 1.0000 | 20 | 1.0 | 0 |
| learned_soft | normal | 0.05 | 1 | 20 | 1.0000 | 20 | 1.0 | 0 |
| learned_soft | normal | 0.1 | -1 | 20 | 1.0000 | 20 | 1.0 | 0 |
| learned_soft | normal | 0.1 | 1 | 20 | 1.0000 | 20 | 1.0 | 0 |
| oracle_soft | near_degenerate | 0.0 | -1 | 20 | 1.0000 | 20 | 1.0 | 0 |
| oracle_soft | near_degenerate | 0.0 | 1 | 20 | 1.0000 | 20 | 1.0 | 0 |
| oracle_soft | near_degenerate | 0.05 | -1 | 20 | 1.0000 | 20 | 1.0 | 0 |
| oracle_soft | near_degenerate | 0.05 | 1 | 20 | 1.0000 | 20 | 1.0 | 0 |
| oracle_soft | near_degenerate | 0.1 | -1 | 20 | 1.0000 | 20 | 1.0 | 0 |
| oracle_soft | near_degenerate | 0.1 | 1 | 20 | 1.0000 | 20 | 1.0 | 0 |
| oracle_soft | normal | 0.0 | -1 | 20 | 1.0000 | 20 | 1.0 | 0 |
| oracle_soft | normal | 0.0 | 1 | 20 | 1.0000 | 20 | 1.0 | 0 |
| oracle_soft | normal | 0.05 | -1 | 20 | 1.0000 | 20 | 1.0 | 0 |
| oracle_soft | normal | 0.05 | 1 | 20 | 1.0000 | 20 | 1.0 | 0 |
| oracle_soft | normal | 0.1 | -1 | 20 | 1.0000 | 20 | 1.0 | 0 |
| oracle_soft | normal | 0.1 | 1 | 20 | 1.0000 | 20 | 1.0 | 0 |
