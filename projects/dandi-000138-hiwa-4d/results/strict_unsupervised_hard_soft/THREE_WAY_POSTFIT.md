# Strict unpaired 3-way comparable postfit scores

All three models used original DANDI 000138 4D data and original independent KMeans labels as applicable; soft source and target prototypes were fit independently in their own 4D domains.
The metric calculation begins ONLY after all fits; no test pairing selected hyperparameters or permuted group labels. The test sample W2 assignment uses the same independently drawn 200 source/200 target points in each model.

Important: original uploaded PyHiWA and ZIP GCOT are different numerical implementations. **Only the latter two form the same-solver Hard/Soft controlled comparison.**

| Metric | Identity | Original PyHiWA / independent Hard | Same GCOT / independent Hard | Same GCOT / independent Soft |
|---|---:|---:|---:|---:|
| Global exact OT W2 squared ↓ | 0.439478 | 0.400328 ± 0.021874 | 0.379350 ± 0.037392 | 0.380061 ± 0.027006 |
| Postfit relative paired MSE ↓ | 1.792656 | 1.890681 ± 0.224605 | 2.019661 ± 0.532772 | 2.454564 ± 0.351690 |
| Postfit mean Euclidean ↓ | 1.256710 | 1.292300 ± 0.100756 | 1.328862 ± 0.220298 | 1.508588 ± 0.135604 |
| Postfit MAE ↓ | 0.531486 | 0.551615 ± 0.050908 | 0.563081 ± 0.107866 | 0.636772 ± 0.053375 |
| Postfit standard R2 ↑ | -0.796486 | -0.894721 ± 0.225085 | -1.023976 ± 0.533911 | -1.459808 ± 0.352441 |

Original PyHiWA strict numerical convergence flags (5 seeds): [True, True, True, True, True]
GCOT Hard ADMM convergence flags: [True, True, True, True, True]
GCOT Soft ADMM convergence flags: [False, False, False, False, False]

Original PyHiWA final local sample OT Q not saved, therefore original nested P-Q total training loss CANNOT be reconstructed accurately; use common W2 squared for this 3-way table and the separate test P-Q scores for same-solver hard/soft.

For full P-Q geometry and raw entropic objective of original GCOT kernel see STRICT_UNSUPERVISED_REPORT.md.
