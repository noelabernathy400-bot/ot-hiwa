# DANDI 000138 — four-dimensional hard-clustering preparation

This new experiment entry is added alongside, not instead of, the bundled original code.

**Dataset:** [DANDI 000138 v0.220113.0407](https://dandiarchive.org/dandiset/000138/0.220113.0407), `MC_Maze_Large`.

**Protocol:** Spike times from each unit are binned at 20 ms, square-root transformed and Gaussian-smoothed **within each trial**. A `StandardScaler` fitted on training trials and a PCA fitted on those training trials yield 4-dimensional neural states. Hand position and hand velocity must be present as separate *hand* time series in the NWB file, each with two coordinates; they are concatenated as `(pos_x, pos_y, vel_x, vel_y)`. Hand movement is **not PCA-projected** or reduced. Training-only standardization changes feature scale but not dimension. The two domains are grouped independently with K-means (`K=4` by default) in four dimensions. For each domain, the six 2-D coordinate projections are visualized with the clustering colors. Paired time indices and labels are never supplied to K-means.

This is **clustering only**, not a HiWA fit. The original untouched Python HiWA source is under `reference/HiWA/PyHiWA/src/hiwa.py` in this package. Original GCOT sources reside in `GC-OT/gcot/`. The entire uploaded 330MB archive is not copied into Git because it also contains 39,000+ raw/video/result entries and unrelated data, including DANDI 000688. `SOURCE_PROVENANCE.json` records the per-file SHA-256 values of extracted code.

Example (run from `GC-OT`):

```bash
python -m pip install numpy scipy scikit-learn h5py matplotlib
# First download the original NWB file(s) from DANDI:000138 v0.220113.0407
python -m experiments.dandi_000138.run_hard_clustering --nwb /path/to/your/000138.nwb --k 4 --seed 42
```

Output: `neural_4d_pairwise_train.png`, `hand_4d_pairwise_train.png`, `clustered_4d.npz`, `windows_4d.csv`, `clustering_report.json`. The runner refuses to silently substitute cursor positions for hand positions, refuses to create results if the NWB is missing, and checks trial completeness.

**Important:** No MC_Maze_Large metrics or graphs are claimed until this program has completed on the official real NWB file. Coordinate index pairing is saved strictly for post-fit evaluation, not for this clustering step. The split is trial-disjoint, but this is not a proof of cross-session generalization.
