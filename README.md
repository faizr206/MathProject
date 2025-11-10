# Vroom2Vec: Learning Low-Dimensional Representations of Engine Sounds

This repository investigates whether engine sounds sharing a fixed attribute (here: `engine_configuration`) live in low‑dimensional, stable, and moderately discriminative linear subspaces.

- Feature space: MFCC‑20 + Δ + ΔΔ → `D=60`
- Subspaces: per‑class PCA on TRAIN frames, uniform rank `r=5`
- Stability: bootstrap principal angles on TRAIN (`B` bootstraps, `p=0.70` of clips)
- Classifier: calibrated Nearest‑Subspace (NSC) with trimmed aggregation (`q=0.40`, `K≥10`)

## Engine Sound Data Source

https://research.google.com/audioset//dataset/engine.html


## Quick Start
- Open the paper: `Paper/main.pdf` (sources in `Paper/main.tex`).
- Browse CV artifacts: `Results/cv/engine_configuration/...`.
- Browse inversion overlays: `Results/inversion/engine_configuration/...`.


## What’s Used in the Paper
Required scripts for reproducing reported results/figures:
- `prepare_data.py`
- `make_mfcc_frames.py`
- `cv_subspace_pipeline.py`
- `compare_variance_mfcc_vs_fft.py`
- `compute_median_frames_per_class.py`
- `invert_subspace_to_audio.py`
- `plot_mfcc.py`

Optional utilities (not required for this paper):
- `nsc_calibrated.py`, `nsc_eval.py` — standalone NSC experiments (non‑CV).
- `split_pca_per_class.py`, `pairwise_subspace_angles.py`, `subspace_stability_bootstrap.py` — one‑off analyses superseded by the CV pipeline for the paper.
- `plot_overlay_mean_spectra.py` — alternate overlay from audio; paper uses overlays from `invert_subspace_to_audio.py`.


## Environment
Python 3.9+ with the following packages:

- numpy, pandas, scikit‑learn, matplotlib
- librosa, soundfile
- pyarrow (for parquet I/O)

Install example:

```
pip install numpy pandas scikit-learn matplotlib librosa soundfile pyarrow
```


## Step‑by‑Step Reproduction
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
- Frames: `frame_length=2048`, `hop_length=512`

Outputs (key):
- `Data/engine_downloads/` — processed WAVs
- `Data/frames/` — per‑clip frames (NPZ)
- `Data/metadata.parquet` — processed clip metadata (with `filename`, `engine_configuration`)
- `Data/frames_index.parquet` — per‑frame index

2) Compute MFCC‑60 per clip (aligned, 25 ms window / 10 ms hop)

```
python make_mfcc_frames.py
```

Outputs:
- `Data/mfcc/<clip>.npz` with key `mfcc` shaped `(n_frames, 60)`
- `Data/mfcc_index.parquet` — per‑frame to MFCC mapping

3) Cross‑validation pipeline (5 folds): low‑dimensionality, stability, geometry, NSC

```
python cv_subspace_pipeline.py \
  --attribute engine_configuration \
  --metadata Data/metadata.parquet \
  --mode npz --npz_dir Data/mfcc --npz_key mfcc \
  --results_root Results/cv --plots
```

Outputs used in the paper:
- `Results/cv/engine_configuration/fold_*/…`: per‑fold confusion matrices, stability summaries, between‑class angles, scree curves.
- `Results/cv/engine_configuration/summary/`: `table_A_lowdim.csv`, `table_B_nsc.csv`, `table_C_stability.csv`, `perm_across_folds.json`.
- `Results/cv/engine_configuration/summary/figures/`: representative scree, angles heatmap, confusions.

4) MFCC vs FFT variance comparison (Figure/Table in Representation section)

```
python compare_variance_mfcc_vs_fft.py \
  --audio_dir Data/engine_downloads \
  --sr 22050 --win_ms 25 --hop_ms 10 \
  --n_mels 64 --n_mfcc 20 --include_deltas \
  --out_dir Results
```

Outputs:
- `Results/scree_mfcc_vs_fft.png`
- `Results/variance_comparison.csv` (EVR@k and components for 90/95/99%)

5) Frames/clip per class summary (Data table cited in paper)

```
python compute_median_frames_per_class.py \
  --frames-parquet Data/mfcc_index.parquet \
  --attribute-col engine_configuration \
  --filename-col clip_filename \
  --out-csv Results/summary/frames_per_clip_by_class.csv
```

6) Subspace‑to‑spectrum inversion + cross‑class overlays (Interpretability section)

```
python invert_subspace_to_audio.py \
  --attribute engine_configuration \
  --metadata Data/metadata.parquet \
  --mode npz --npz_dir Data/mfcc --npz_key mfcc \
  --min_clips 30 --results_root Results/inversion \
  --top_r 5 --k_sigma 2.0 --t_seconds 2.0 --gl_iters 48 \
  --n_mfcc 20 --n_mels 64 --sr 22050 --n_fft 2048 --hop_length 512 \
  --overlay_curve pca_filtered_mean --normalize refhz --ref_hz 200
```

Per‑class outputs: under `Results/inversion/engine_configuration/<class>/` — model/data/PCA‑filtered spectra, PC± excursions, `mean_envelope.wav`.

Cross‑class overlays (used in paper):
- `Results/inversion/engine_configuration/cross_class_overlay_pca_filtered_mean.png`
- `Results/inversion/engine_configuration/cross_class_descriptors_pca_filtered_mean.csv`

7) MFCC heatmap example (Figure in paper)

```
python plot_mfcc.py \
  --npz Data/mfcc/<some_clip>.npz \
  --out figures/mfcc_example.png --mode zscore --max_frames 1000
```


## Script Reference (one‑liners)
- `prepare_data.py` — preprocess audio, select frames → `Data/`.
- `make_mfcc_frames.py` — compute MFCC‑60 per clip → `Data/mfcc`, `Data/mfcc_index.parquet`.
- `cv_subspace_pipeline.py` — full 5‑fold CV (EVR, stability, angles, NSC) → `Results/cv/...`.
- `compare_variance_mfcc_vs_fft.py` — MFCC vs FFT EVR comparison → `Results/scree_mfcc_vs_fft.png`.
- `compute_median_frames_per_class.py` — frames/clip stats → `Results/summary/frames_per_clip_by_class.csv`.
- `invert_subspace_to_audio.py` — inversion + overlays → `Results/inversion/...`.
- `plot_mfcc.py` — MFCC heatmap → `figures/mfcc_example.png`.


## Tips & Troubleshooting
- Adjust paths at the top of scripts if your directory layout differs.
- If parquet I/O fails, verify `pyarrow` is installed.
- Seeds are fixed in code to help reproducibility (CV splits and numeric routines).


## Acknowledgments
This codebase builds on standard signal processing and subspace modeling practices using open‑source Python libraries.
