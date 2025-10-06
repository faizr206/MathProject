# cv_subspace_pipeline.py
# Requirements:
#   pip install numpy pandas scikit-learn matplotlib pyarrow

import os
import re
import json
import math
import shutil
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score

# ---------------------- Defaults & constants ----------------------
RANDOM_SEED_CV = 0             # for fold split reproducibility
RANDOM_SEED_NUM = 42           # for numeric/procedural reproducibility
D = 60                         # feature dimension (MFCC-20+Δ+ΔΔ)
R_UNIFORM = 5                  # class subspace rank (fixed across folds)
Q_TRIM = 0.40                  # keep lowest q% frames within-clip
K_MIN = 10                     # min frames kept; else fallback to median
B_BOOT = 10                    # bootstraps per class
BOOT_P = 0.70                  # % of TRAIN clips per bootstrap
EPS = 1e-8                     # for calibration std

# ---------------------- Small utilities ----------------------
np.random.seed(RANDOM_SEED_NUM)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def canonicalize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-]+", "_", str(name))

def plot_confmat(cm: np.ndarray, classes: List[str], out_path: str, normalize: bool = False):
    plt.figure(figsize=(6, 5))
    if normalize:
        C = cm.astype(np.float64)
        row_sums = C.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        C_disp = C / row_sums
        fmt = ".2f"
    else:
        C_disp = cm.astype(np.float64)
        fmt = ".0f"
    im = plt.imshow(C_disp, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45, ha='right')
    plt.yticks(ticks, classes)
    thresh = (np.nanmax(C_disp) + np.nanmin(C_disp)) / 2.0
    for i in range(C_disp.shape[0]):
        for j in range(C_disp.shape[1]):
            plt.text(j, i, format(C_disp[i, j], fmt),
                     ha="center", va="center",
                     color="white" if C_disp[i, j] > thresh else "black", fontsize=9)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix (normalized)" if normalize else "Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_scree(cum: np.ndarray, out_path: str, title: str):
    k = len(cum)
    xs = np.arange(1, k+1)
    plt.figure(figsize=(6,4))
    plt.plot(xs, cum, marker="o")
    plt.axhline(0.90, linestyle="--")
    plt.axvline(R_UNIFORM, linestyle="--")
    plt.ylim(0, 1.01)
    plt.xlabel("Components")
    plt.ylabel("Cumulative EVR")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def principal_angles_rad(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    if U1.size == 0 or U2.size == 0:
        return np.array([], dtype=np.float32)
    r = min(U1.shape[1], U2.shape[1])
    if r < 1:
        return np.array([], dtype=np.float32)
    Q1, _ = np.linalg.qr(U1[:, :r])
    Q2, _ = np.linalg.qr(U2[:, :r])
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    th = np.arccos(s)
    return np.sort(th).astype(np.float32)

def largest_principal_angle_deg(U1: np.ndarray, U2: np.ndarray) -> float:
    th = principal_angles_rad(U1, U2)
    if th.size == 0:
        return float("nan")
    return float(np.degrees(th.max()))

def residuals_to_subspace(X: np.ndarray, mu: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    r(x) = ||(I - UU^T)(x - mu)||_2 per frame
    Efficient via energy identity.
    """
    Xc = X - mu[None, :]
    tot = np.sum(Xc * Xc, axis=1)
    if U.size == 0:
        resid2 = tot
    else:
        coeff = Xc @ U
        proj = np.sum(coeff * coeff, axis=1)
        resid2 = np.maximum(tot - proj, 0.0)
    return np.sqrt(resid2, dtype=np.float32)

def trimmed_mean_best(resids: np.ndarray, q: float = Q_TRIM, min_k: int = K_MIN) -> float:
    n = resids.size
    if n == 0:
        return float("inf")
    if n < min_k:
        return float(np.median(resids))
    k = int(round(q * n))
    k = max(min_k, min(k, n))
    part = np.partition(resids, k-1)[:k]
    return float(np.mean(part))

# ---------------------- Feature source abstraction ----------------------
class FrameSource:
    """
    Loader for per-clip frame features:
      - mode 'npz': one npz per clip in a directory; key = args.npz_key (fallback: 'mfcc' then 'X')
      - mode 'parquet': one big parquet with f0..f59 and 'filename' (clip id)
    """
    def __init__(self, mode: str, npz_dir: Optional[str], npz_key: Optional[str],
                 frames_parquet: Optional[str]):
        self.mode = mode
        self.npz_dir = npz_dir
        self.npz_key = npz_key
        self.frames_df: Optional[pd.DataFrame] = None
        if mode == "parquet":
            self.frames_df = pd.read_parquet(frames_parquet)
            needed = {"filename"} | {f"f{i}" for i in range(D)}
            miss = needed - set(self.frames_df.columns)
            if miss:
                raise KeyError(f"frames parquet missing columns: {sorted(miss)}")
        elif mode == "npz":
            ensure_dir(npz_dir)
        else:
            raise ValueError("mode must be 'npz' or 'parquet'")

        self.cache: Dict[str, np.ndarray] = {}

    def _npz_path_for(self, filename: str) -> str:
        base = filename
        if base.lower().endswith(".wav") or base.lower().endswith(".mp3"):
            base = os.path.splitext(base)[0] + ".npz"
        elif not base.lower().endswith(".npz"):
            base = base + ".npz"
        return os.path.join(self.npz_dir, base)

    def load_clip(self, filename: str) -> np.ndarray:
        if filename in self.cache:
            return self.cache[filename]
        if self.mode == "npz":
            path = self._npz_path_for(filename)
            if not os.path.exists(path):
                return np.empty((0, D), dtype=np.float32)
            z = np.load(path)
            key = self.npz_key
            X = None
            if key in z:
                X = z[key]
            elif "mfcc" in z:
                X = z["mfcc"]
            elif "X" in z:
                X = z["X"]
            else:
                return np.empty((0, D), dtype=np.float32)
            X = np.asarray(X, dtype=np.float32)
            if X.ndim != 2 or X.shape[1] != D:
                return np.empty((0, D), dtype=np.float32)
            self.cache[filename] = X
            return X
        else:
            g = self.frames_df[self.frames_df["filename"] == filename]
            if g.empty:
                return np.empty((0, D), dtype=np.float32)
            Fcols = [f"f{i}" for i in range(D)]
            X = g[Fcols].to_numpy(dtype=np.float32, copy=False)
            self.cache[filename] = X
            return X

# ---------------------- Subspace fitting ----------------------
def fit_class_subspaces(train_clips_by_class: Dict[str, List[str]],
                        fs: FrameSource,
                        outlier_trim: bool = False) -> Dict[str, Dict]:
    """
    Returns dict:
      cls -> { 'U': D×r, 'mu': D, 'evr': (n_comp,), 'cum_evr': (n_comp,), 'n_train_clips': int }
    With optional one-pass outlier trimming (top 10% residual frames) before refit.
    """
    out: Dict[str, Dict] = {}
    for cls, clip_list in train_clips_by_class.items():
        parts = []
        for fname in clip_list:
            X = fs.load_clip(fname)
            if X.size:
                parts.append(X)
        if not parts:
            out[cls] = {"U": np.empty((D,0), dtype=np.float32),
                        "mu": np.zeros(D, dtype=np.float32),
                        "evr": np.array([], dtype=np.float32),
                        "cum_evr": np.array([], dtype=np.float32),
                        "n_train_clips": 0}
            continue
        Xall = np.vstack(parts).astype(np.float32)  # (N,D)
        mu = Xall.mean(axis=0)
        Xc = Xall - mu[None, :]

        if outlier_trim and Xc.shape[0] >= 50:
            # initial PCA to compute residuals
            n_comp0 = max(1, min(R_UNIFORM, Xc.shape[0]-1, D))
            p0 = PCA(n_components=n_comp0, svd_solver="full", random_state=RANDOM_SEED_NUM).fit(Xc)
            U0 = p0.components_.T.astype(np.float32)
            r0 = U0.shape[1]
            # residuals to initial subspace
            res = residuals_to_subspace(Xall, mu, U0)
            thr = np.quantile(res, 0.90)
            keep = res <= thr
            Xc_trim = (Xall[keep] - mu[None, :]).astype(np.float32)
            Xc = Xc_trim if Xc_trim.shape[0] >= 10 else Xc  # safety fallback

        n_comp = max(1, min(R_UNIFORM, Xc.shape[0]-1, D))
        pca = PCA(n_components=n_comp, svd_solver="full", random_state=RANDOM_SEED_NUM).fit(Xc)
        U = pca.components_.T.astype(np.float32)  # D×r
        evr = pca.explained_variance_ratio_.astype(np.float32)
        cum = np.cumsum(evr)
        # If r < R_UNIFORM due to data, pad cum to length R_UNIFORM for plotting consistency
        out[cls] = {"U": U, "mu": mu.astype(np.float32), "evr": evr, "cum_evr": cum,
                    "n_train_clips": len(clip_list)}
    return out

# ---------------------- Split helpers ----------------------
def make_cv_splits(meta: pd.DataFrame, attribute: str, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    meta has columns: filename, attribute
    Returns list of (train_idx, test_idx) over the *clip-level* rows.
    """
    clips = meta[["filename", attribute]].drop_duplicates().reset_index(drop=True)
    y = clips[attribute].to_numpy()
    groups = clips["filename"].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED_CV)
    return list(sgkf.split(X=np.zeros(len(clips)), y=y, groups=groups)), clips

# ---------------------- Per-fold procedures ----------------------
def reconstruction_mse_for_class(test_clips: List[str], cls: str, U: np.ndarray, mu: np.ndarray,
                                 fs: FrameSource) -> float:
    parts = []
    for fname in test_clips:
        X = fs.load_clip(fname)
        if X.size:
            parts.append(X)
    if not parts:
        return float("nan")
    Xall = np.vstack(parts).astype(np.float32)
    res = residuals_to_subspace(Xall, mu, U)
    return float(np.mean(res * res))

def stability_bootstrap(train_clips: List[str], U_ref: np.ndarray, mu_ref: np.ndarray,
                        fs: FrameSource, B: int = B_BOOT, p: float = BOOT_P) -> List[float]:
    rng = np.random.default_rng(RANDOM_SEED_NUM)
    angles = []
    n = len(train_clips)
    if n < 2:
        return angles
    k = max(1, int(round(p * n)))
    for _ in range(B):
        sample = rng.choice(train_clips, size=k, replace=False).tolist()
        # pool frames
        parts = []
        for fname in sample:
            X = fs.load_clip(fname)
            if X.size:
                parts.append(X)
        if not parts:
            continue
        Xb = np.vstack(parts).astype(np.float32)
        mub = Xb.mean(axis=0)
        Xbc = Xb - mub[None, :]
        nb = max(1, min(R_UNIFORM, Xbc.shape[0]-1, D))
        pca_b = PCA(n_components=nb, svd_solver="full", random_state=RANDOM_SEED_NUM).fit(Xbc)
        Ub = pca_b.components_.T.astype(np.float32)
        # compare at common rank
        r_used = min(U_ref.shape[1], Ub.shape[1])
        if r_used < 1:
            continue
        theta = largest_principal_angle_deg(U_ref[:, :r_used], Ub[:, :r_used])
        if not np.isnan(theta):
            angles.append(float(theta))
    return angles

def calibrated_nsc(test_clips: List[str],
                   classes: List[str],
                   subspaces: Dict[str, Dict],
                   train_clips_by_class: Dict[str, List[str]],
                   fs: FrameSource,
                   q: float = Q_TRIM,
                   K: int = K_MIN,
                   do_plots: bool = False,
                   out_dir: Optional[str] = None) -> Dict:
    """
    Returns metrics, confusion matrices, per-class report, and writes figures if requested.
    """
    # ---- calibration on TRAIN (true-class residuals)
    calib = {}
    for cls in classes:
        vals = []
        mu = subspaces[cls]["mu"]; U = subspaces[cls]["U"]
        for fname in train_clips_by_class.get(cls, []):
            X = fs.load_clip(fname)
            if not X.size:
                continue
            r = residuals_to_subspace(X, mu, U)
            Rtrim = trimmed_mean_best(r, q=q, min_k=K)
            vals.append(Rtrim)
        arr = np.array(vals, dtype=np.float64)
        if arr.size == 0:
            calib[cls] = {"mu": np.nan, "sigma": np.nan}
        else:
            calib[cls] = {"mu": float(arr.mean()), "sigma": float(arr.std() + EPS)}

    # ---- score TEST clips
    rows = []
    clip_preds = []
    for fname, true_cls in test_clips:
        Sa = {}
        Za = {}
        for cls in classes:
            mu = subspaces[cls]["mu"]; U = subspaces[cls]["U"]
            X = fs.load_clip(fname)
            if not X.size:
                Sa[cls] = np.inf
                Za[cls] = np.inf
                continue
            r = residuals_to_subspace(X, mu, U)
            Rtrim = trimmed_mean_best(r, q=q, min_k=K)
            Sa[cls] = Rtrim
            mu_t = calib[cls]["mu"]; sg_t = calib[cls]["sigma"]
            if np.isnan(mu_t) or np.isnan(sg_t) or sg_t <= 0:
                Za[cls] = np.inf
            else:
                Za[cls] = (Rtrim - mu_t) / sg_t

        ranked = sorted([(Za[c], Sa[c], subspaces[c]["U"].shape[1], c) for c in classes])
        pred = ranked[0][3]
        clip_preds.append((fname, true_cls, pred))
        for c in classes:
            rows.append({"filename": fname, "true": true_cls, "class": c, "Rtrim": float(Sa[c]), "Z": float(Za[c])})

    # ---- metrics
    df_long = pd.DataFrame(rows)
    per_clip = df_long[["filename","true"]].drop_duplicates().merge(
        pd.DataFrame(clip_preds, columns=["filename","true","pred"]),
        on=["filename","true"], how="left"
    )
    y_true = per_clip["true"].tolist()
    y_pred = per_clip["pred"].tolist()
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    overall = float(np.mean(np.array(y_true) == np.array(y_pred)))
    macro = float(balanced_accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)

    if do_plots and out_dir:
        plot_confmat(cm, classes, os.path.join(out_dir, "confusion_raw.png"), normalize=False)
        plot_confmat(cm, classes, os.path.join(out_dir, "confusion_norm.png"), normalize=True)

    return {
        "cm": cm, "overall": overall, "macro": macro, "report": report,
        "scores_long": df_long, "per_clip": per_clip
    }

# ---------------------- Main CV driver ----------------------
def run_cv(attribute: str,
           meta_path: str,
           mode: str,
           npz_dir: Optional[str],
           npz_key: Optional[str],
           frames_parquet: Optional[str],
           min_clips: int,
           results_root: str,
           plots: bool,
           outlier_trim: bool):
    # ---- Load metadata and filter classes with enough clips
    meta = pd.read_parquet(meta_path) if meta_path.lower().endswith(".parquet") else pd.read_csv(meta_path)
    need = {"filename", attribute}
    if not need.issubset(set(meta.columns)):
        raise KeyError(f"Metadata must include columns: {sorted(need)}")
    # keep one row per clip for split
    clip_df = meta[["filename", attribute]].drop_duplicates()
    # filter classes with >= min_clips
    counts = clip_df[attribute].value_counts()
    keep_classes = counts[counts >= min_clips].index.tolist()
    clip_df = clip_df[clip_df[attribute].isin(keep_classes)].reset_index(drop=True)
    if clip_df.empty or len(keep_classes) < 2:
        raise RuntimeError("Not enough classes/clips for CV after filtering.")

    classes = sorted(keep_classes)
    # Build folds
    folds, clip_table = make_cv_splits(clip_df, attribute, n_splits=5)

    # Prepare feature source
    fs = FrameSource(mode=mode, npz_dir=npz_dir, npz_key=npz_key, frames_parquet=frames_parquet)

    # Paths
    base_dir = os.path.join(results_root, attribute)
    ensure_dir(base_dir)

    # Aggregation stores
    agg_lowdim = []     # per fold & class: EVR@5, recon MSE, n_train_clips
    agg_nsc = []        # per fold: overall & macro
    agg_stab_raw = {}   # class -> list of angles across folds
    agg_stab_medians = []  # for representative selection maybe

    # Per-fold loop
    for k, (train_idx, test_idx) in enumerate(folds):
        fold_dir = os.path.join(base_dir, f"fold_{k}")
        scree_dir = os.path.join(fold_dir, "scree")
        ensure_dir(fold_dir); ensure_dir(scree_dir)

        # split clip lists
        train_clips_tbl = clip_table.iloc[train_idx].reset_index(drop=True)
        test_clips_tbl  = clip_table.iloc[test_idx].reset_index(drop=True)
        # coverage log
        train_counts = train_clips_tbl[attribute].value_counts().reindex(classes).fillna(0).astype(int).to_dict()
        test_counts  = test_clips_tbl[attribute].value_counts().reindex(classes).fillna(0).astype(int).to_dict()
        with open(os.path.join(fold_dir, "coverage.json"), "w") as f:
            json.dump({"train": train_counts, "test": test_counts}, f, indent=2)

        # group clips by class
        train_by_class = {c: train_clips_tbl[train_clips_tbl[attribute]==c]["filename"].tolist() for c in classes}
        test_by_class  = {c: test_clips_tbl[test_clips_tbl[attribute]==c]["filename"].tolist() for c in classes}

        # ---- Fit TRAIN subspaces (uniform r=5), optional outlier trim
        subspaces = fit_class_subspaces(train_by_class, fs, outlier_trim=outlier_trim)

        # ---- Low-dimensionality: scree & EVR@5, and reconstruction MSE on TEST (same class)
        rec_rows = []
        for c in classes:
            evr = subspaces[c]["evr"]
            cum = subspaces[c]["cum_evr"]
            # scree fig
            if cum.size > 0:
                plot_scree(cum, os.path.join(scree_dir, f"scree_{canonicalize(c)}.png"),
                           f"Scree — {c} (Fold {k})")
            # EVR@5 (if fewer comps, take last cum)
            if cum.size == 0:
                evr5 = np.nan
            else:
                idx = min(R_UNIFORM, len(cum)) - 1
                evr5 = float(cum[idx])
            # reconstruction MSE on TEST frames for same class
            mse = reconstruction_mse_for_class(test_by_class[c], c, subspaces[c]["U"], subspaces[c]["mu"], fs)
            rec_rows.append({"class": c, "EVR_at_5": evr5, "test_recon_MSE": mse,
                             "n_train_clips": subspaces[c]["n_train_clips"]})
            agg_lowdim.append({"fold": k, "class": c, "EVR_at_5": evr5, "test_recon_MSE": mse,
                               "n_train_clips": subspaces[c]["n_train_clips"]})
        pd.DataFrame(rec_rows).to_csv(os.path.join(fold_dir, "reconstruction_mse.csv"), index=False)

        # ---- Stability (TRAIN-only) bootstraps
        stab_rows = []
        raw_angles_rows = []
        for c in classes:
            Uref = subspaces[c]["U"]; mu = subspaces[c]["mu"]
            angles = stability_bootstrap(train_by_class[c], Uref, mu, fs, B=B_BOOT, p=BOOT_P)
            if c not in agg_stab_raw: agg_stab_raw[c] = []
            agg_stab_raw[c].extend(angles)
            if len(angles) == 0:
                med = p25 = p75 = np.nan
            else:
                med = float(np.median(angles))
                p25 = float(np.percentile(angles, 25))
                p75 = float(np.percentile(angles, 75))
            stab_rows.append({"class": c, "r": min(Uref.shape[1], R_UNIFORM),
                              "median_deg": med, "p25_deg": p25, "p75_deg": p75,
                              "n_train_clips": len(train_by_class[c])})
            for a in angles:
                raw_angles_rows.append({"class": c, "theta_max_deg": float(a)})
        pd.DataFrame(stab_rows).to_csv(os.path.join(fold_dir, "stability_summary.csv"), index=False)
        pd.DataFrame(raw_angles_rows).to_csv(os.path.join(fold_dir, "stability_raw.csv"), index=False)

        # ---- Discriminativeness (TEST): calibrated NSC
        test_clip_list = list(zip(test_clips_tbl["filename"].tolist(),
                                  test_clips_tbl[attribute].tolist()))
        nsc = calibrated_nsc(test_clip_list, classes, subspaces, train_by_class, fs,
                             q=Q_TRIM, K=K_MIN, do_plots=plots, out_dir=fold_dir)
        # save metrics
        with open(os.path.join(fold_dir, "nsc_accuracy.json"), "w") as f:
            json.dump({"overall": nsc["overall"], "macro": nsc["macro"]}, f, indent=2)
        # per-class report
        pd.DataFrame(nsc["report"]).transpose().to_csv(os.path.join(fold_dir, "per_class_report.csv"))
        agg_nsc.append({"fold": k, "overall": nsc["overall"], "macro": nsc["macro"]})
        # save clip scores (long)
        nsc["scores_long"].to_csv(os.path.join(fold_dir, "clip_scores.csv"), index=False)

        # ---- Between-class geometry (TRAIN): pairwise largest angles
        mat = np.full((len(classes), len(classes)), np.nan, dtype=np.float32)
        for i, ci in enumerate(classes):
            Ui = subspaces[ci]["U"]
            for j, cj in enumerate(classes):
                if j < i: 
                    mat[i, j] = mat[j, i]
                    continue
                Uj = subspaces[cj]["U"]
                if i == j:
                    val = 0.0
                else:
                    r_used = min(Ui.shape[1], Uj.shape[1])
                    if r_used < 1:
                        val = np.nan
                    else:
                        val = largest_principal_angle_deg(Ui[:, :r_used], Uj[:, :r_used])
                mat[i, j] = mat[j, i] = val
        angles_df = pd.DataFrame(mat, index=classes, columns=classes)
        angles_df.to_csv(os.path.join(fold_dir, "between_class_angles.csv"))
        # heatmap
        plt.figure(figsize=(6,5))
        im = plt.imshow(mat, interpolation="nearest")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        ticks = np.arange(len(classes))
        plt.xticks(ticks, classes, rotation=45, ha='right')
        plt.yticks(ticks, classes)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                txt = "nan" if not np.isfinite(v) else f"{v:.1f}"
                plt.text(j, i, txt, ha="center", va="center",
                         color="white" if (np.isfinite(v) and v > (np.nanmax(mat)+np.nanmin(mat))/2.0) else "black",
                         fontsize=9)
        plt.title(f"Between-class largest angle (°) — Fold {k}")
        plt.tight_layout()
        plt.savefig(os.path.join(fold_dir, "angles_heatmap.png"), dpi=150)
        plt.close()

    # ---------------------- Aggregate across folds ----------------------
    summary_dir = os.path.join(base_dir, "summary")
    figs_dir = os.path.join(summary_dir, "figures")
    ensure_dir(summary_dir); ensure_dir(figs_dir)

    # Table A (low-dim): mean±sd EVR@5, test_recon_MSE, avg #train clips
    df_low = pd.DataFrame(agg_lowdim)
    tableA_rows = []
    for c in classes:
        g = df_low[df_low["class"] == c]
        evr_mean = float(g["EVR_at_5"].mean())
        evr_sd = float(g["EVR_at_5"].std())
        mse_mean = float(g["test_recon_MSE"].mean())
        mse_sd = float(g["test_recon_MSE"].std())
        ntrain_avg = float(g["n_train_clips"].mean())
        tableA_rows.append({
            "class": c,
            "EVR5_mean": evr_mean, "EVR5_sd": evr_sd,
            "test_recon_MSE_mean": mse_mean, "test_recon_MSE_sd": mse_sd,
            "avg_train_clips": ntrain_avg
        })
    pd.DataFrame(tableA_rows).to_csv(os.path.join(summary_dir, "table_A_lowdim.csv"), index=False)

    # Table B (NSC): accuracy per fold & mean±sd
    df_nsc = pd.DataFrame(agg_nsc).sort_values("fold")
    df_nsc.to_csv(os.path.join(summary_dir, "table_B_nsc.csv"), index=False)

    # Table C (stability): per class median of medians & pooled IQR using all raw angles
    tableC_rows = []
    # compute per-fold medians first (for completeness)
    for c in classes:
        pooled = np.array(agg_stab_raw.get(c, []), dtype=np.float64)
        if pooled.size == 0:
            med_of_meds = np.nan; iqr25 = np.nan; iqr75 = np.nan
        else:
            iqr25 = float(np.percentile(pooled, 25))
            iqr75 = float(np.percentile(pooled, 75))
            med_of_meds = float(np.median(pooled))
        tableC_rows.append({"class": c, "median_deg": med_of_meds,
                            "iqr25_deg": iqr25, "iqr75_deg": iqr75})
    pd.DataFrame(tableC_rows).to_csv(os.path.join(summary_dir, "table_C_stability.csv"), index=False)

    # Representative fold = median overall accuracy
    med_fold = int(df_nsc.sort_values("overall")["fold"].iloc[len(df_nsc)//2])
    rep_fold_dir = os.path.join(base_dir, f"fold_{med_fold}")
    # copy figures (all scree, both confusions, angle heatmap)
    # scree
    sdir = os.path.join(rep_fold_dir, "scree")
    if os.path.isdir(sdir):
        for fn in os.listdir(sdir):
            if fn.endswith(".png"):
                shutil.copy2(os.path.join(sdir, fn), os.path.join(figs_dir, f"rep_{fn}"))
    for fn in ["confusion_raw.png", "confusion_norm.png", "angles_heatmap.png"]:
        src = os.path.join(rep_fold_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(figs_dir, f"rep_{fn}"))

    # Permutation test (once) on the representative fold
    # Shuffle TEST labels by clip, recompute accuracy (using the already saved per-clip predictions)
    rep_scores = pd.read_csv(os.path.join(rep_fold_dir, "clip_scores.csv"))
    per_clip = rep_scores[["filename","true"]].drop_duplicates()
    # fake "pred" by reusing the same preds from calibrated NSC (we didn't save them directly, so compute via argmin on Z)
    # Build prediction per clip (min Z across classes)
    z_cols = [c for c in rep_scores.columns if c in ["Z"]]  # long format; we need to reconstruct
    # Reconstruct preds:
    preds = (rep_scores.loc[rep_scores.groupby("filename")["Z"].idxmin()][["filename","class"]]
             .rename(columns={"class":"pred"}))
    per_clip = per_clip.merge(preds, on="filename", how="left")
    y_true = per_clip["true"].to_numpy()
    y_pred = per_clip["pred"].to_numpy()
    rng = np.random.default_rng(RANDOM_SEED_NUM)
    chance = 1.0 / len(classes)
    perm_accs = []
    for _ in range(200):
        y_perm = rng.permutation(y_true)
        perm_accs.append(float(np.mean(y_perm == y_pred)))
    perm_summary = {
        "representative_fold": med_fold,
        "observed_overall": float(np.mean(y_true == y_pred)),
        "perm_mean_acc": float(np.mean(perm_accs)),
        "perm_std_acc": float(np.std(perm_accs)),
        "chance_baseline": chance,
        "perm_runs": 200
    }
    with open(os.path.join(summary_dir, "perm_test.json"), "w") as f:
        json.dump(perm_summary, f, indent=2)

    print("=== CV complete ===")
    print("Summary tables written to:", summary_dir)
    print("Representative figures in:", figs_dir)

# ---------------------- CLI ----------------------
def main():
    ap = argparse.ArgumentParser(description="5-Fold CV: Subspace Evidence + Classification")
    ap.add_argument("--attribute", required=True, type=str,
                    help="Attribute column name to classify (e.g., engine_configuration or engine_state).")
    ap.add_argument("--metadata", required=True, type=str, help="Path to metadata parquet/csv with 'filename' and attribute.")
    ap.add_argument("--mode", choices=["npz","parquet"], default="npz",
                    help="Feature source: per-clip NPZs or a frames parquet with f0..f59.")
    ap.add_argument("--npz_dir", type=str, default="Data/mfcc",
                    help="Directory with per-clip NPZs (when --mode npz).")
    ap.add_argument("--npz_key", type=str, default="mfcc",
                    help="NPZ key to read per-clip frames (fallbacks to 'mfcc' then 'X').")
    ap.add_argument("--frames_parquet", type=str, default=None,
                    help="Path to frames parquet with columns f0..f59 and filename (when --mode parquet).")
    ap.add_argument("--min_clips", type=int, default=30,
                    help="Only include classes with at least this many clips.")
    ap.add_argument("--results_root", type=str, default="Results/cv",
                    help="Root directory for outputs.")
    ap.add_argument("--plots", action="store_true", help="Save figures (scree, confusions, heatmap).")
    ap.add_argument("--trim_train_outliers", action="store_true",
                    help="Optional: on TRAIN only, drop top 10%% frames by own-class residual, refit PCA (still r=5).")
    args = ap.parse_args()

    run_cv(attribute=args.attribute,
           meta_path=args.metadata,
           mode=args.mode,
           npz_dir=args.npz_dir,
           npz_key=args.npz_key,
           frames_parquet=args.frames_parquet,
           min_clips=args.min_clips,
           results_root=args.results_root,
           plots=args.plots,
           outlier_trim=args.trim_train_outliers)

if __name__ == "__main__":
    main()
