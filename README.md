# Low‑Dimensional, Stable, and Moderately Discriminative Subspaces for Engine Sound Attributes

This repository investigates whether engine sounds sharing a fixed attribute (here: `engine_configuration`) live in low‑dimensional, stable, and moderately discriminative linear subspaces.

- Feature space: MFCC‑20 + Δ + ΔΔ → `D=60`
- Subspaces: per‑class PCA on TRAIN frames, uniform rank `r=5`
- Stability: bootstrap principal angles on TRAIN (B=10, p=0.70 of clips)
- Classifier: calibrated Nearest‑Subspace (NSC) with trimmed aggregation (q=0.40, K≥10)


## Quick Start
- Read the project report: `REPORT.md` (figures in `figures/`).
- Browse cross‑validation artifacts: `Results/cv/engine_configuration/...`.
- Paper draft: `Paper/main.pdf` (TeX in `Paper/main.tex`).


## Repository Layout
- `prepare_data.py` — Build balanced subset, resample/trim/normalize audio, select frames; writes `Data/`.
- `make_mfcc_frames.py` — Compute per‑clip MFCC‑20+Δ+ΔΔ (60‑D) aligned to frames; writes `Data/mfcc`.
- `cv_subspace_pipeline.py` — 5‑fold CV for subspace evidence + NSC; writes `Results/cv/<ATTRIBUTE>/...`.
- `nsc_calibrated.py`, `nsc_eval.py` — Standalone NSC utilities (non‑CV experiments).
- `split_pca_per_class.py`, `pairwise_subspace_angles.py`, `subspace_stability_bootstrap.py` — analysis helpers.
- `Download/` — input metadata/audio (e.g., `Download/engine_metadata_combined.parquet`, `Download/engine_downloads/`).
- `Data/` — processed audio/frames/features and analysis outputs.
- `Results/` — CV outputs (tables, per‑fold figures, summaries).
- `Paper/` — paper sources and compiled PDF.


## Environment
Python 3.9+ with the following packages:

- numpy, pandas, scikit‑learn, matplotlib
- librosa, soundfile
- pyarrow (for parquet I/O)

Install example:

```
pip install numpy pandas scikit-learn matplotlib librosa soundfile pyarrow
```


## Data Preparation (one‑time)
1) Prepare a balanced, preprocessed subset and frame index

```
python prepare_data.py
```

Defaults inside the script:
- Input metadata: `Download/engine_metadata_combined.parquet`
- Source audio: `Download/engine_downloads/`
- Output root: `Data/`
- Classes kept: `inline-4`, `V8`, `inline-6`, `V6`, `single-cylinder`
- Per‑class cap: 60 clips (adjust `MAX_PER_CLASS` if needed)
- Audio: mono 22.05 kHz; trim non‑silent (`top_db=40`), peak‑normalize to 0.99
- Frames: `frame_length=2048`, `hop_length=512`, target ≈50 frames/clip

Outputs (key):
- `Data/engine_downloads/` — processed WAVs
- `Data/frames/` — per‑clip frames (NPZ)
- `Data/metadata.parquet` — processed clip metadata (with `filename`, `engine_configuration`)
- `Data/frames_index.parquet` — per‑frame index

2) Compute MFCC‑60 per clip (aligned to the frame grid)

```
python make_mfcc_frames.py
```

Outputs:
- `Data/mfcc/<clip>.npz` with key `mfcc` shaped `(n_frames, 60)`
- `Data/mfcc_index.parquet` — per‑frame to MFCC mapping


## Cross‑Validation Pipeline (5 folds)
Run subspace modeling + stability + calibrated NSC for a chosen attribute (primary: `engine_configuration`).

```
python cv_subspace_pipeline.py \
  --attribute engine_configuration \
  --metadata Data/metadata.parquet \
  --mode npz --npz_dir Data/mfcc --npz_key mfcc \
  --results_root Results/cv --plots
```

Key defaults (see top of `cv_subspace_pipeline.py`):
- `D=60`, `R_UNIFORM=5`, `Q_TRIM=0.40`, `K_MIN=10`, `B_BOOT=10`, `BOOT_P=0.70`
- Seeds: CV split `RANDOM_SEED_CV=0`; numeric routines `RANDOM_SEED_NUM=42`

Outputs:
- Per fold: `fold_*/scree/*.png`, `nsc_accuracy.json`, `per_class_report.csv`, `stability_summary.csv`, `between_class_angles.csv`, confusion matrices.
- Summary: `summary/table_A_lowdim.csv`, `table_B_nsc.csv`, `table_C_stability.csv`, `perm_test.json`, and representative figures in `summary/figures/`.


## Results at a Glance
- Low‑dimensional: with `r=5`, cumulative EVR ≈ 94–96% across classes.
- Stable: bootstrapped largest principal angles are modest for most classes (≈12–18° median); inline‑6 is weaker.
- Discriminative: calibrated NSC achieves ≈0.256 ± 0.020 overall accuracy vs. 0.20 chance (5 classes).

See `REPORT.md` for tables, figures, and interpretation. Representative figures are mirrored in `figures/` for convenience.


## Tips & Troubleshooting
- Paths are configurable at the top of the scripts; adjust to your environment before running.
- If parquet I/O fails, verify `pyarrow` is installed.
- If you already have `Results/` populated, you can skip running the pipeline and read the artifacts directly (as done in `REPORT.md`).


## Acknowledgments
This codebase builds on standard signal processing and subspace modeling practices using open‑source Python libraries.

