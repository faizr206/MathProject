Analysis Extras

This folder contains lightweight, scriptable analyses that complement the main notebook. They let you recompute numeric summaries and robustness checks without re‑running the heavy feature extraction cells.

Contents
- posthoc_stats.py: Recomputes subspace CV, angles, and bootstrap stability from saved feature parquet. Writes compact numeric summaries.
- window_stability.py: Within‑clip stability prototype by splitting a clip into short windows and comparing subspaces across windows. Also compares to between‑clip baselines.
- attribute_subspace_scan.py: Runs the subspace analysis across multiple metadata attributes (e.g., engine_configuration, engine_state, turbo_supercharged) using the saved features.

Typical usage
- Posthoc summaries (use the same OUTDIR that the notebook wrote):
  - `python analysis_extras/posthoc_stats.py --outdir analysis_outputs3 --label-col turbo_supercharged`

- Within‑clip stability (sampled clips per class):
  - `python analysis_extras/window_stability.py --outdir analysis_outputs3 --audio-dir Download/engine_downloads --label-col turbo_supercharged --n-per-class 30`

- Attribute scan:
  - `python analysis_extras/attribute_subspace_scan.py --outdir analysis_outputs3 --attrs engine_configuration engine_state turbo_supercharged`

Notes
- All scripts operate on the saved features parquet in the specified `--outdir` (e.g., `analysis_outputs3/clip_features.parquet`).
- They re‑run PCA/subspace math on the cached feature vectors only. They do not re‑extract audio features.
- If your label column is different (e.g., `is_car`), pass it via `--label-col`.
