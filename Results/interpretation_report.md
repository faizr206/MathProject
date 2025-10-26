**Executive Summary**
- MFCC+Δ+ΔΔ features are highly low‑dimensional: 5 PCs explain ~92–94% per class and ~91.9% overall. Raw FFT representations require tens to hundreds of PCs for comparable coverage.
- Inverting class‑conditional PCs yields interpretable spectral controls: PC1 consistently modulates spectral brightness/centroid; PC2 and PC3 shift spectral tilt and band emphasis with class‑dependent orientation; audio/spectra confirm coherent changes.
- Subspace stability is generally good (median 12–16°) except for inline‑6, which is markedly less stable (median ~28° with wide IQR), suggesting greater intra‑class variability.
- Between‑class geometry shows meaningful separation: V‑type vs inline classes often exhibit large angles (≈50–65°); some inline classes are closer (~23–35°). Patterns vary by fold but are consistent in broad clusters.
- Discriminativeness via nearest‑subspace classification is modest but above chance (mean overall ≈0.25 vs 0.20 baseline; permutation test supports significance on the representative fold). Per‑class recall varies by fold.

**Data Overview**
- Balanced coverage across classes with near‑uniform frames/clip; medians ~985–998 frames per clip.
- Reference: `Results/summary/frames_per_clip_by_class.csv:1`.

**Low‑Dimensionality**
- Cross‑representation comparison (global): `Results/variance_comparison.csv:1`
  - `MFCC+Δ+ΔΔ (D=60)`: `EVR_at_5 = 0.9187`, `EVR_at_10 = 0.9654`, `k_for_90 = 5`, `k_for_95 = 8`, `k_for_99 = 17`.
  - `FFT_logmag (D=513)`: `EVR_at_5 = 0.6626`, `EVR_at_10 = 0.7244`, `k_for_90 = 74`, `k_for_95 = 128`, `k_for_99 = 249`.
  - `FFT_mag (D=513)`: `EVR_at_5 = 0.6545`, `EVR_at_10 = 0.7659`, `k_for_90 = 32`, `k_for_95 = 60`, `k_for_99 = 153`.
- Per‑class (CV summary): `Results/cv/engine_configuration/summary/table_A_lowdim.csv:1`
  - `EVR5_mean` ranges ≈0.923–0.943 across classes with small `EVR5_sd` (≈0.002–0.006).
  - Test reconstruction MSEs are consistent across classes (≈1080–1270), supporting good out‑of‑sample coverage by r=5 subspaces.
- Visual scree corroboration: `Results/scree_mfcc_vs_fft.png` and representative class scree in `Results/cv/engine_configuration/summary/figures/`.
- Takeaway: MFCCs concentrate variance in very few directions, enabling compact, class‑specific subspaces without severe reconstruction loss.

**Inversion Interpretability**
- Procedure: For each class, invert ±2σ along the first 5 PCs using the class mean and PCA basis; inspect audio (`pcX_±.wav`) and spectra/metrics (centroid, peak).
- Class means (centroid/peak) are comparable across classes (centroids ~1976–2156 Hz; peaks often at low/mid bins). Reference: `Results/inversion/engine_configuration/*/summary.json` (e.g., single‑cylinder centroid 2155.6 Hz, peak 247.6 Hz).
- PC1 (spectral brightness axis):
  - Minus direction: low spectral centroid (≈590–1190 Hz), strong low‑frequency peak (≈53–248 Hz).
  - Plus direction: much higher centroid (≈3693–6003 Hz) and peaks shift to mid/high bands (≈667–5556 Hz, class‑dependent).
  - References: `.../pc1_minus.json` and `.../pc1_plus.json` across classes (e.g., inline‑6 plus centroid 5006.8 Hz vs minus 718.7 Hz).
- PC2 (tilt/brightness with class‑dependent orientation):
  - Inline classes and single‑cylinder: plus increases brightness (centroid ≈4640–6744 Hz), minus decreases (~1801–1843 Hz).
  - V‑types (V6/V8): plus lowers centroid (~1628–1663 Hz), minus raises (~5536–5776 Hz).
  - References: `.../pc2_plus.json` and `.../pc2_minus.json` for each class.
- PC3 (band emphasis/harmonic distribution):
  - Moderate centroid shifts; peaks move between low and mid bands depending on direction (e.g., inline‑4 minus centroid ~2989 Hz with mid‑band peak; plus ~621 Hz).
  - References: `.../pc3_plus.json`, `.../pc3_minus.json`.
- Implication: The leading PCs correspond to understandable manipulations of spectral shape (brightness/tilt/harmonic focus). Inversions yield coherent, audible changes; directions vary by class sign convention but semantics are consistent.

**Stability (Bootstrap Subspace Angles)**
- Summary (median principal angle across resamples): `Results/cv/engine_configuration/summary/table_C_stability.csv:1`
  - V6: 11.86° (IQR 10.13–18.07°)
  - V8: 13.76° (IQR 10.24–17.91°)
  - inline‑4: 12.54° (IQR 8.55–17.68°)
  - inline‑6: 28.10° (IQR 17.07–36.97°)
  - single‑cylinder: 14.61° (IQR 11.38–16.93°)
- Fold details show inline‑6 instability spikes (e.g., fold 2 median 41.77°, fold 3 median 36.08°). References: `Results/cv/engine_configuration/fold_*/stability_summary.csv`.
- Interpretation: Most classes have stable 5‑D subspaces; inline‑6 exhibits greater intra‑class variability or multi‑modality.

**Between‑Class Geometry (Principal Angles)**
- Angles between class subspaces indicate separation and clustering; see representative heatmap: `Results/cv/engine_configuration/summary/figures/rep_angles_heatmap.png`.
- Example (fold 1): `Results/cv/engine_configuration/fold_1/between_class_angles.csv:1`
  - Large: V6 vs inline‑4 (52.2°), V6 vs single‑cylinder (53.2°).
  - Moderate: V8 vs inline‑4 (24.0°), inline‑6 vs single‑cylinder (23.5°).
- Across folds, V‑type vs inline pairs often show larger angles (≈50–65°), while some inline pairs are closer (≈23–37°). Fold 3 shows very strong separations (many >60°).
- Interpretation: Subspace geometry encodes meaningful class structure; separation is strongest across engine families (V vs inline) and weaker within families.

**Discriminativeness (Nearest‑Subspace Classification)**
- 5‑fold CV accuracy per fold: `Results/cv/engine_configuration/summary/table_B_nsc.csv:1`
  - Overall: 0.136, 0.271, 0.237, 0.328, 0.276 (mean ≈0.25).
  - Macro averages are similar, indicating class balance.
- Permutation test (representative fold): `Results/cv/engine_configuration/summary/perm_test.json:1`
  - Observed overall 0.271 vs permuted mean 0.185 ± 0.050; baseline 0.20.
  - Conclusion: Above chance with modest effect size; subspaces alone provide limited clip‑level discriminative power.
- Per‑class variability (examples): `Results/cv/engine_configuration/fold_*/per_class_report.csv:1`
  - Fold 3: V6 recall 0.636; others ~0.20–0.33.
  - Fold 4: single‑cylinder recall 0.571; others ~0.10–0.33.
- Interpretation: Despite clear geometric structure and interpretability, decision boundaries remain fuzzy under nearest‑subspace; likely need temporal aggregation or discriminative modeling.

**Key Takeaways**
- Representation: MFCC+Δ+ΔΔ is compact and captures >90% variance in ≤8 PCs per class; FFT features are far less compact.
- Interpretability: Leading PCs map to intuitive spectral controls (brightness/tilt/harmonic emphasis); inversions are coherent and class‑consistent up to sign.
- Stability: Most classes are stable; inline‑6 shows notably higher instability.
- Geometry: Clear separation across engine families; some within‑family proximity.
- Classification: Above chance but modest; per‑class performance is uneven.

**Recommendations**
- Use subspace distances/angles as features in a discriminative classifier (e.g., SVM, logistic) instead of nearest‑subspace alone.
- Aggregate frame‑level evidence (vote or HMM/CRF) to form clip‑level decisions; explore temporal dynamics.
- Consider discriminative dimensionality reduction (LDA/PLDA) or subspace alignment to enhance between‑class separation.
- Investigate inline‑6 heterogeneity (data quality, subclasses) driving instability.
- Explore r>5 where beneficial (trade‑off vs stability) and feature augmentation (e.g., spectral contrast, tonnetz) if consistent with domain.

**File References**
- Variance comparison: `Results/variance_comparison.csv:1`
- Per‑class low‑dim and recon: `Results/cv/engine_configuration/summary/table_A_lowdim.csv:1`
- Stability summary: `Results/cv/engine_configuration/summary/table_C_stability.csv:1`
- Angles (example): `Results/cv/engine_configuration/fold_1/between_class_angles.csv:1`
- CV accuracies: `Results/cv/engine_configuration/summary/table_B_nsc.csv:1`
- Permutation test: `Results/cv/engine_configuration/summary/perm_test.json:1`
- Inversions: `Results/inversion/engine_configuration/<class>/pcX_±.*`
