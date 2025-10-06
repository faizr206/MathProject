Awesome project. Here’s a clean, end-to-end plan to *demonstrate* (not just claim) that engine sounds sharing a fixed attribute live in low-dimensional, stable, and discriminative linear subspaces of a frequency-feature space.

I’ll keep it practical: concrete steps, what to code, what to plot, and why each piece matters.

---

# 0) Hypothesis → measurable claims

For a chosen attribute (A) (e.g., **engine_configuration** or **engine_state**):

1. **Low-dimensionality:** Frames from clips with the same (A=a) are well-approximated by a rank-(r) linear subspace in a frequency-feature space (small reconstruction error; high explained variance for small (r)).

2. **Stability:** The learned subspace for (A=a) barely changes across resamples of clips (small principal angles between subspaces fitted on different subsets).

3. **Discriminativeness:** A nearest-subspace classifier (project and compare residuals) predicts (A) accurately on held-out clips; between-class subspaces have large principal angles.

---

# 1) Data prep (quick & robust)

**Why:** Reduce noise and keep compute tractable.

* **Resample & mono:** 22.05 kHz mono for all files.
* **Trim silence / non-engine:** `librosa.effects.split(y, top_db=40)` then keep high-energy chunks only.
* **Standardize loudness:** Peak or RMS normalize each clip after trimming.
* **Frame selection:** Work in *frames* (not whole clips) to get many samples per class. To keep it light, sample ~**50 frames/clip** uniformly from voiced segments.

**Code sketch**

```python
import librosa, numpy as np

SR = 22050
N_MELS = 128
HOP = 512
N_FFT = 2048

def load_logmel_frames(path, max_frames_per_clip=50):
    y, _ = librosa.load(path, sr=SR, mono=True)
    # trim silence
    intervals = librosa.effects.split(y, top_db=40)
    y = np.concatenate([y[s:e] for s, e in intervals]) if len(intervals) else y
    if len(y) == 0:
        return None  # skip empty after trimming

    S = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP,
                                       n_mels=N_MELS, fmin=20, fmax=SR//2)
    X = librosa.power_to_db(S, ref=np.max)  # log-mel, shape (N_MELS, T)
    # z-normalize per-frame to reduce loudness bias
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)

    T = X.shape[1]
    if T <= max_frames_per_clip:
        idx = np.arange(T)
    else:
        idx = np.linspace(0, T-1, max_frames_per_clip).astype(int)
    return X[:, idx].T  # (frames, N_MELS)
```

---

# 2) Build labeled frame dataset

**Why:** We’ll fit subspaces *per attribute value* from pooled frames.

* Pick one attribute to start (e.g., `engine_configuration` or `engine_state`).
* Keep only attribute values with enough clips (e.g., **≥ 30 clips** each).
* Split train/test **by clip** (never by frames) to avoid leakage.

```python
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

def collect_frames(df, base_dir, attr, max_frames_per_clip=50, min_clips=30):
    # df columns: filename, <your attributes> ...
    # group by attr and filter rare classes
    counts = df.groupby(attr)['filename'].nunique()
    keep_vals = set(counts[counts >= min_clips].index)

    rows = df[df[attr].isin(keep_vals)].copy()
    X_frames, y_frames, clip_ids = [], [], []
    for _, r in rows.iterrows():
        F = load_logmel_frames(f"{base_dir}/{r['filename']}", max_frames_per_clip)
        if F is None: 
            continue
        X_frames.append(F)                # (k, N_MELS)
        y_frames += [r[attr]] * len(F)
        clip_ids += [r['filename']] * len(F)
    if not X_frames:
        raise RuntimeError("No features extracted.")
    X = np.vstack(X_frames)               # (N, N_MELS)
    y = np.array(y_frames)
    groups = np.array(clip_ids)          # group=clip for CV
    return X, y, groups, list(keep_vals)
```

---

# 3) Learn subspaces with PCA (per class)

**Why:** PCA is the standard linear subspace estimator.

* For each attribute value (a), pool **train** frames with (A=a).
* Fit PCA, choose rank (r_a) by an explained-variance threshold, e.g. **90%** (cap at, say, **r ≤ 20**).
* Record: basis (U_a), mean (\mu_a), explained variance curve.

```python
from sklearn.decomposition import PCA

def fit_class_subspace(X_class, evr_thresh=0.90, r_cap=20):
    pca = PCA(svd_solver='randomized', random_state=0).fit(X_class)
    cum = np.cumsum(pca.explained_variance_ratio_)
    r = min( np.searchsorted(cum, evr_thresh) + 1, r_cap )
    U = pca.components_[:r].T            # (d, r)
    mu = pca.mean_                        # (d,)
    evr = cum[r-1]
    return dict(U=U, mu=mu, r=r, evr=evr, pca=pca)
```

**Evidence of low-dimensionality**
For each class, report:

* chosen (r_a)
* cumulative explained variance at (r_a) (aim ≥ 0.9)
* **reconstruction MSE** on held-out frames (see §5)

Make a **scree plot** (variance vs components) per class.

---

# 4) Subspace stability via bootstrapping

**Why:** Show “the same” subspace shows up from different samples.

* For each class (a):

  * Repeat (B) times (e.g., **B=20**): sample 70% of the **train clips** for that class, collect their frames, fit a subspace with the same (r_a).
  * Compute **principal angles** between each bootstrap subspace and the full-train subspace.
  * Summarize with median & IQR of the **largest** principal angle (worst-case).

```python
from scipy.linalg import subspace_angles
rng = np.random.default_rng(0)

def principal_angle_deg(U1, U2):
    # U1, U2 are (d, r) with orthonormal columns
    th = subspace_angles(U1, U2)  # radians, increasing
    return np.degrees(th[-1])     # largest angle

def bootstrap_stability(class_frames_by_clip, U_ref, mu_ref, r, B=20):
    angles = []
    clips = list(class_frames_by_clip.keys())
    for _ in range(B):
        k = max(1, int(0.7*len(clips)))
        subset = rng.choice(clips, size=k, replace=False)
        Xb = np.vstack([class_frames_by_clip[c] for c in subset])
        # center by mean, fit PCA, take first r
        pca = PCA(svd_solver='randomized', random_state=0).fit(Xb)
        Ub = pca.components_[:r].T
        # Orthonormality is guaranteed by PCA
        angles.append(principal_angle_deg(U_ref, Ub))
    return np.array(angles)
```

**Evidence of stability**
Small angles (e.g., median < **10–15°**) indicate stable subspaces.

---

# 5) Discriminativeness via nearest-subspace classifier (NSC)

**Why:** If classes truly occupy different subspaces, projection residuals will separate them.

* On **test** frames (grouped by clip):

  * For each class subspace ( (U_a, \mu_a) ), compute residual
    ( r_a(x) = |x - \mu_a - U_a U_a^\top (x - \mu_a)|_2 ).
  * Predict ( \hat{a} = \arg\min_a r_a(x) ).
* Aggregate per **clip** (e.g., majority vote across that clip’s frames).

```python
import numpy as np
from collections import defaultdict

def residual_norm(x, U, mu):
    v = x - mu
    proj = U @ (U.T @ v)
    return np.linalg.norm(v - proj)

def predict_clip(frames, subspaces):
    # subspaces: {a: dict(U, mu)}
    residuals = {a: np.median([residual_norm(f, S['U'], S['mu']) for f in frames]) 
                 for a, S in subspaces.items()}
    return min(residuals, key=residuals.get)

def evaluate_nsc(test_clip_frames, test_clip_labels, subspaces):
    y_true, y_pred = [], []
    for clip, frames in test_clip_frames.items():
        y_true.append(test_clip_labels[clip])
        y_pred.append(predict_clip(frames, subspaces))
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = (y_true == y_pred).mean()
    return acc, y_true, y_pred
```

**Evidence of discriminativeness**

* Report **clip-level accuracy** (macro-avg) with **StratifiedGroupKFold** (e.g., 5-fold).
* Plot a **confusion matrix**.
* Optional: show **residual distributions** (own-class vs nearest other).

---

# 6) Between-class subspace angles (global separation)

**Why:** Quantify geometric separation independent of classification.

* Compute pairwise principal angles between **class subspaces** learned on full train.
* Visualize a **heatmap of largest angles** (bigger = more separated).

```python
import itertools

def pairwise_subspace_angle_deg(subspaces):
    labels = list(subspaces.keys())
    A = np.zeros((len(labels), len(labels)))
    for i, j in itertools.product(range(len(labels)), repeat=2):
        Ui, Uj = subspaces[labels[i]]['U'], subspaces[labels[j]]['U']
        A[i, j] = principal_angle_deg(Ui, Uj)
    return labels, A
```

---

# 7) Cross-validation protocol (putting it all together)

**Why:** Turn the above into a reproducible pipeline.

1. Choose attribute (A) (start with `engine_configuration`; repeat for `engine_state`).
2. Split with `StratifiedGroupKFold(n_splits=5)`, stratify by (A), groups=clip filename.
3. For each fold:

   * Extract **train** frames & **test** frames.
   * Fit per-class PCA subspaces (choose (r_a) by 90% EVR).
   * **Low-dim evidence:** scree + reconstruction MSE on test frames for own class.
   * **Stability:** bootstrap angles on train set (per class).
   * **Discriminativeness:** NSC accuracy on test clips; confusion matrix.
4. **Between-class geometry:** fit on full train + heatmap of principal angles.

Keep simple tables:

* Table A: class → `#clips`, chosen `r`, EVR@r (%, mean±sd over folds)
* Table B: NSC accuracy per fold (macro-avg), mean±sd
* Table C: Stability (median largest angle per class), median±IQR
* Figure 1: Scree plots per class
* Figure 2: Heatmap of between-class largest angles
* Figure 3: Confusion matrix
* Figure 4: Own-class vs other-class residuals (violin/box)

---

# 8) Practical tips & guardrails

* **Outliers & label noise:**

  * Iterative trimming: after fitting class subspace, drop top **5%** frames by own-class residual, refit once.
  * Clip-level audit: if a clip is a severe outlier in its class, exclude it (likely wrong metadata).
* **Balance:** If class imbalance is large, cap clips per class (e.g., random 80 clips max) for fairness.
* **Feature tweaks:** If log-mel alone is noisy, append simple spectral stats per frame (centroid, bandwidth, rolloff) or MFCC (13 + deltas). Keep linear PCA space dimension moderate (e.g., 128–160).
* **Rank caps:** Engine acoustics are rich; reasonable caps: **r ∈ [5, 20]**.
* **Reproducibility:** fix `random_state`, store features to **Parquet** to avoid recomputation.

---

# 9) Minimal runnable scaffold (glue code)

Below is a compact skeleton showing the CV loop and the three claims. You can paste your metadata paths and run.

```python
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix
from collections import defaultdict, Counter

ATTR = "engine_configuration"  # or "engine_state"
BASE_DIR = "engine_downloads"  # your wav folder

# 1) Load metadata (must have columns: filename, engine_configuration, engine_state, ...)
meta = pd.read_parquet("engine_metadata_combined.parquet")  # or CSV
# ensure 'filename' column exists and points to wavs in BASE_DIR

# 2) Build dataset
X, y, groups, keep_vals = collect_frames(meta, BASE_DIR, attr=ATTR, max_frames_per_clip=50, min_clips=30)

# map frames back to clips for evaluation
# (modify collect_frames to also return per-clip frames if you prefer)
# Here we'll reconstruct per-clip frames for test split only.

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
fold_metrics = []

for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
    Xtr, ytr, gtr = X[tr], y[tr], groups[tr]
    Xte, yte, gte = X[te], y[te], groups[te]

    # 3) Fit subspaces per class on TRAIN
    subspaces = {}
    for a in keep_vals:
        Xa = Xtr[ytr == a]
        # optional: iterative trimming of top 5% residuals once
        S0 = fit_class_subspace(Xa, evr_thresh=0.90, r_cap=20)
        resids = np.linalg.norm(Xa - S0['mu'] - (S0['U'] @ (S0['U'].T @ (Xa - S0['mu']).T)).T, axis=1)
        keep = resids <= np.quantile(resids, 0.95)
        S = fit_class_subspace(Xa[keep], evr_thresh=0.90, r_cap=20)
        subspaces[a] = S

    # 4) Low-dimensionality: reconstruction MSE on TEST own-class frames
    recon = []
    for a in keep_vals:
        Xt = Xte[yte == a]
        S = subspaces[a]
        V = Xt - S['mu']
        Vproj = (S['U'] @ (S['U'].T @ V.T)).T
        mse = np.mean(np.sum((V - Vproj)**2, axis=1))
        recon.append((a, S['r'], S['evr'], mse))
    recon_df = pd.DataFrame(recon, columns=["class", "rank_r", "evr_at_r", "test_recon_mse"])

    # 5) Discriminativeness: NSC clip-level
    # gather frames by clip for TEST
    test_clip_frames = defaultdict(list)
    test_clip_label = {}
    for xi, yi, gi in zip(Xte, yte, gte):
        test_clip_frames[gi].append(xi)
        test_clip_label[gi] = yi
    # convert lists to arrays
    for k in list(test_clip_frames.keys()):
        test_clip_frames[k] = np.stack(test_clip_frames[k], axis=0)

    acc, y_true, y_pred = evaluate_nsc(test_clip_frames, test_clip_label, subspaces)
    C = confusion_matrix(y_true, y_pred, labels=list(keep_vals))
    fold_metrics.append(dict(acc=acc, recon=recon_df, cm=C))

    print(f"Fold {fold+1}: NSC clip-accuracy = {acc:.3f}")
```

You can add:

* **Scree plots:** `subspaces[a]['pca'].explained_variance_ratio_`
* **Angles heatmap:** `pairwise_subspace_angle_deg(subspaces)`
* **Stability:** cache frames per clip per class (train side), then call `bootstrap_stability`.

---

# 10) What your final write-up can claim

* **Low-dimensionality:** “For configuration=inline-4 (N=___ clips), a rank-(r=8) subspace explains (92%\pm3%) of variance across 5 folds; test reconstruction MSE median ___.” (Include scree + MSE.)

* **Stability:** “Bootstrap (B=20) largest principal angle to the reference subspace: median (7.4^\circ) [IQR 5.9–9.8] for inline-4, (6.2^\circ) [4.8–8.1] for V8.” (Include violin/box.)

* **Discriminativeness:** “Nearest-subspace classifier achieves **78%** clip-level accuracy (macro-avg) across 5 folds over {inline-4, V6, V8, boxer-4}, far above chance (25%).” (Include confusion matrix + residual plots.)

If you want one dial to make it “more mathy,” add a simple **permutation test**: shuffle class labels at the clip level, retrain, and show NSC accuracy collapses to chance—evidence the structure isn’t random.

---

## Optional upgrades (only if you want)

* **Subspace distances:** report **chordal** or **Grassmann geodesic** distances (they’re monotone in principal angles).
* **K-subspaces** clustering to discover subspaces without labels (then check alignment with attributes).
* **Robust PCA** (outlier-resistant) if label noise is gnarly.

---

If you want, I can adapt this scaffold to your exact folder/column names and print the plots. Or we can run it twice—once for `engine_configuration`, once for `engine_state`—and compare which attribute yields cleaner subspaces.
