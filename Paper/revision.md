Here’s expert, actionable feedback—prioritized so you can make the biggest improvements quickly.

# High-impact revisions (do these first)

1. **Quantify, don’t hint.**

* Replace qualitative claims (“moderately discriminative”, “weaker”) with **numbers + uncertainty**. Report **95% CIs** over clips for overall/macro accuracy (bootstrap over clips), and **per-class** accuracies with CIs.
* Extend the **permutation test across all folds** and report a **p-value** (fraction of permutations ≥ observed), not just one fold’s mean±SD.

2. **Fill missing dataset details.**

* Table 1 currently shows `median frames/clip = n/a`. Replace with **actual per-class medians and IQRs**, plus total **#clips/#frames per class**.
* Add a sentence stating **how many clips/folds/classes survived filtering**.

3. **Prevent and document leakage explicitly.**

* In **Methods**, state plainly that you use **StratifiedGroupKFold with groups = clip filename** so *no frame from a test clip appears in train*. Right now this is implied by the project but not spelled out here.

4. **Baselines and ablations—show, don’t tell.**

* You list baselines (SVM, kNN, CCR) and ablations (rank sweep, trims, calibration) but don’t show results. Include a **small table** summarizing each with mean±SD (or CIs). Even a subset (e.g., SVM, CCR, r∈{2,5,10}) strengthens the story.

5. **Relate geometry to errors with a statistic.**

* You say confusions align with angles; add **Spearman ρ between pairwise angle and confusion rate** (per fold, then average). A simple figure or one-line statistic tightens this claim.

6. **Explain inline-6.**

* Inline-6 has the weakest stability. Diagnose: fewer clips? broader state variation? lower SNR? Add a **short paragraph** with evidence (counts/SNR proxy) so it doesn’t look like a dangling outlier.

# Medium-impact clarity/polish

7. **Define the math you use.**

* Add 2–3 lines defining **principal angles** and the **NSC score**. Example snippet you can drop in:

  * *Principal angles:* “Given orthonormal bases (U,V\in\mathbb{R}^{D\times r}), the cosines of the principal angles are the singular values of (U^\top V); we report (\theta_{\max}=\arccos(\sigma_{\min})).”
  * *NSC residual:* “Score for class (c): (s_c=\operatorname{trimmean}_{q}{\lVert (I-U_cU_c^\top)x_t\rVert_2^2}_t), then per-class (z)-score calibration on TRAIN.”

8. **MFCC vs FFT: add classification evidence.**

* You compare EVR curves; also report **NSC accuracy using FFT** (and, if possible, log-mag FFT). A small 2-row table (MFCC vs FFT) will substantiate “FFT is noisier”.

9. **Rank choice justification.**

* You use uniform (r=5). Add a **mini figure or table**: EVR@r and NSC@r for r∈{2,5,10}. If class-wise EVR≥95% suggests different r, mention that per-class (r(\tau)) gave similar/better accuracy (or not).

10. **Confidence and calibration.**

* Since you calibrate scores, add a one-panel **reliability plot** or ECE (even coarse). If not feasible, briefly justify calibration with a sentence.

11. **Figures: make them self-contained.**

* Add **colorbars** and **units** to the angle heatmap; ensure tick labels are legible.
* In the stability violin, mark **medians** and give the **sample size** (B×folds).
* In the inversion figure, label axes (Hz) and note whether it’s **log-mel or linear frequency** after inversion.

12. **Write tighter.**

* Trim **Motivation** by ~20% (it’s strong but a bit wordy). Move “Downstream utility” to **Discussion**.
* Replace hedgy adjectives with data: e.g., “NSC achieves **25.0% [21.3, 29.1]** overall vs 20% chance.”

# LaTeX & reproducibility hygiene

13. **Replace `[H]` except when essential.**

* You’ve loaded `float`; good. Prefer `[!htbp]` to avoid float errors and page breaks. Keep `\FloatBarrier` sparingly.

14. **Simplify `\graphicspath`.**

* Consider copying figures to `Paper/figures` and using just `{./figures/}` to avoid fragile relative paths.

15. **Numerics consistency.**

* Use `\SI{25.0 \pm 7.1}{\percent}` style everywhere; harmonize decimals (e.g., EVR to 0.1%).
* Standardize class labels (e.g., `inline-6` with a non-breaking hyphen).

16. **Fill Table formats.**

* For tables with percentages/angles, use `siunitx` columns (`S[table-format=2.1]`) so decimals align.

# Optional but valuable additions

17. **Related work (short).**

* A 4–6 sentence paragraph connecting to *subspace methods in audio/speaker recognition (i-vectors, PLDA)* and *MFCCs for timbre/ASR* will anchor the paper. (No need for a full survey.)

18. **Threats to validity.**

* Add a small subsection: domain shift (recorders, exhaust mods), label noise, SNR variation, and the limitation of **B=10 bootstraps** (suggest B=100 in future).

19. **Pipeline figure.**

* One schematic showing **audio → MFCC → class PCA → residuals → trim → calibration → predict** helps readers.

# Drop-in text you can paste

**Threats to validity (short):**
*We observe variability in recording conditions (device, distance, environment) and engine states, which may blur class boundaries. Labels sourced from metadata may contain noise. Although GroupKFold prevents frame-level leakage, hyperparameter choices (rank, trim, calibration) were tuned on TRAIN and may overfit folds. Bootstrap size (B=10) yields wide uncertainty for stability; future work will increase (B) and report BCa CIs.*

**Abstract tightening (optional):**
*We test whether engine-configuration audio occupies low-dimensional, stable subspaces. With MFCC-60 features and 5-fold GroupKFold, class-conditional PCA with (r=5) explains (\approx)92–94% variance, bootstrap largest-angle medians are 12–15° (inline-6 higher), and a calibrated nearest-subspace classifier reaches (25.0%) [CI] vs 20% chance. EVR and error patterns align with between-class angles. These results support compact, interpretable indexing for engine audio and motivate state-conditioned and mixture-of-subspaces extensions.*

---

If you want, I can turn the key adds (angle–confusion correlation table, rank-sweep summary, short definitions block) into ready-to-paste LaTeX snippets.
