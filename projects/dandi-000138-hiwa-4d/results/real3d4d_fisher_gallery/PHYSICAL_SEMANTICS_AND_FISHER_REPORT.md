# Original DANDI 4D physical semantics versus PCA3 and Fisher plots

Only the original 320 training trials fit PCA, FA, KMeans and Fisher; the displayed 2800 windows are internal heldout 80 validation trials. Original 100 test trials entirely untouched.

## Physical meaning of the 3D target coordinates

Every PCA3 target axis is a MIX of physical hand position x/y and velocity x/y (after training-only standardization). This is still a three-dimensional linear physical-kinematics **subspace**, but these are not three independently measured physical quantities. Dimension reduction trades away one physical combination, and the reconstruction diagnostics below quantify that tradeoff.

Train PC variance shares = [0.36268, 0.29022, 0.18325]. Sum=0.83615. Discarded share=0.16385.

| Original 4D measured variable | PC1 loading | PC2 loading | PC3 loading | Validation reconstruction R2 |
|---|---:|---:|---:|---:|
| Hand position X | +0.5432 | +0.4637 | +0.1235 | 0.6668 |
| Hand position Y | -0.5154 | +0.4225 | +0.7455 | 0.9999 |
| Hand velocity X | +0.6356 | +0.1708 | +0.3320 | 0.6870 |
| Hand velocity Y | -0.1879 | +0.7598 | -0.5646 | 0.9535 |

## Figure directory

- [PCA3_FISHER2](PCA3_FISHER2_SPLIT_AND_ALIGNED.png) — Real 2800 validation points, separate-domain KMeans pseudocolors; overlay in target domain basis
- [PCA3_FISHER3](PCA3_FISHER3_SPLIT_AND_ALIGNED.png) — Real 2800 validation points, separate-domain KMeans pseudocolors; overlay in target domain basis
- [PCA3_NATIVE3D](PCA3_NATIVE3D_SPLIT_AND_ALIGNED.png) — Real 2800 validation points, separate-domain KMeans pseudocolors; overlay in target domain basis
- [FA3_FISHER2](FA3_FISHER2_SPLIT_AND_ALIGNED.png) — Real 2800 validation points, separate-domain KMeans pseudocolors; overlay in target domain basis
- [FA3_FISHER3](FA3_FISHER3_SPLIT_AND_ALIGNED.png) — Real 2800 validation points, separate-domain KMeans pseudocolors; overlay in target domain basis
- [FA3_NATIVE3D](FA3_NATIVE3D_SPLIT_AND_ALIGNED.png) — Real 2800 validation points, separate-domain KMeans pseudocolors; overlay in target domain basis
- [PCA4_FISHER2](PCA4_FISHER2_SPLIT_AND_ALIGNED.png) — Real 2800 validation points, separate-domain KMeans pseudocolors; overlay in target domain basis
- [PCA4_FISHER3](PCA4_FISHER3_SPLIT_AND_ALIGNED.png) — Real 2800 validation points, separate-domain KMeans pseudocolors; overlay in target domain basis
- [FA4_FISHER2](FA4_FISHER2_SPLIT_AND_ALIGNED.png) — Real 2800 validation points, separate-domain KMeans pseudocolors; overlay in target domain basis
- [FA4_FISHER3](FA4_FISHER3_SPLIT_AND_ALIGNED.png) — Real 2800 validation points, separate-domain KMeans pseudocolors; overlay in target domain basis

Do not use plot-enhanced Fisher separation as a model scoring metric. The Fisher basis is fitted to KMeans pseudo-labels by maximizing between/within scatter, so it is explicitly biased towards making the discovered clusters look separable. Neural-source Fisher axes and target-hand Fisher axes differ. Only third and fourth panels overlay on the same target Fisher axes.
Absolute latent OT scores are not comparable across 3D and 4D; the original physical 4D matching metrics are directly comparable.
