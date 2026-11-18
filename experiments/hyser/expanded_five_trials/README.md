# Hyser 5-trial dataset acquisition (Subject 01, Session 1, Random Task)

**Status:** acquisition/preparation automation only; the prior 250-point benchmark is unchanged.

Official source: https://physionet.org/content/hd-semg/2.0.0/random_dataset/subject01_session1/

## Exactly what is downloaded

For `sample1` through `sample5`, the script downloads four files per trial:

- `random_preprocess_sampleN.hea`
- `random_preprocess_sampleN.dat`
- `random_force_sampleN.hea`
- `random_force_sampleN.dat`

Total: 20 WFDB files; about 125 MB. No `random_raw_sampleN.*` files are necessary.

Original citation: Jiang, X., Dai, C., Liu, X., & Fan, J. (2023).
*Open Access Dataset and Toolbox of High-Density Surface Electromyogram Recordings*, v2.0.0,
PhysioNet. https://doi.org/10.13026/hxan-pe94.
Official file license: ODC-By v1.0.
See https://physionet.org/content/hd-semg/2.0.0/ for metadata and license details.

## Automated GitHub acquisition

GitHub workflow: `.github/workflows/hyser-download-five-trials.yml`.
On the workflow's first push or subsequent manual `workflow_dispatch`, it downloads
the official records, checks their sizes, makes 100 ms windows and trial-disjoint
train/test data, then uploads *both original waveform files and processed features*
to GitHub Releases and GitHub Actions run artifacts.

Release name/tag (when the Action succeeds): `hyser-s01-s1-random-5trials-v1`.

The original WFDB data are stored as **release assets** rather than committing
~125 MB of binary files into Git history.

The processed feature archive contains:
- `all_five_trials_UNSCALED_windows.npz` (1250x256 EMG log1p RMS, 1250x3 force)
- `trial_heldout_1.npz` to `trial_heldout_5.npz` (1000 train / 250 test each domain, 3D)
- `SOURCE_AND_VALIDATION.json` including SHA-256 provenance and per-trial validation.

Preprocessing: EMG 256ch, sample frequency 2048Hz -> 100ms RMS -> log1p;
force 100Hz -> mean thumb/middle/index (in this order). PCA3 and separate
whitening are fitted **on training trials only** in each holdout fold.

**Important:** The fold-specific 3D features are not interchangeable with
the earlier 250-point baseline. Do not compare historical method errors
directly against these new inputs; rerun every comparator with common
fold-specific preprocessing. The extraction preserves synchronous row
indices only for later post-training evaluation and never uses them for
unpaired OT training or selecting hyperparameters.
