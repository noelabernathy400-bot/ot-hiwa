# CMU Panoptic three-group cross-camera experiment

The accepted slice is `171204_pose2`, person 0, frames 141–339. The same 19-joint
world-coordinate skeleton is expressed in HD camera `00_00` and `00_01`
coordinates using the official calibration rotations. Poses are centred at
COCO19 joint 2, so camera translation cancels.

The two domains are independently reduced to two PCA dimensions and RMS
normalized. Alignment methods receive only the two unordered point clouds.
Frame identity and the three visual groups are evaluation-only. The groups are
inferred by KMeans on the source t-SNE embedding and correspond to the three
temporally contiguous trajectory sections visible in the diagnostic figure.

Run from the repository root:

```powershell
python -m experiments.panoptic.run data/panoptic/171204_pose2 results/panoptic
```

Outputs:

- `pca_tsne_three_groups.{png,pdf}`: PCA/t-SNE source and target diagnostic.
- `figure2a_panoptic.{png,svg,pdf}`: HiWA neural Figure 2a-style comparison.
- `alignment_metrics.{png,pdf}` and `metrics.csv`: quantitative comparison.
