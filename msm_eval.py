# msm_eval.py
# Requirements:
#   pip install numpy pandas scikit-learn matplotlib pyarrow

import os
import re
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score

# ---------- Defaults ----------
DATA_DIR = "Data"
SPLITS_DIR = "splits"
OUT_DIR = "msm"

TRAIN_FRAMES_PARQUET = "train_frames.parquet"
TEST_FRAMES_PARQUET  = "test_frames.parquet"

R_CLASS = 5        # class subspace rank r
EPS = 1e-8
RANDOM_SEED = 42
# ------------------------------

np.random.seed(RANDOM_SEED)

# ------------- Utils -------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def canonicalize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-]+", "_", str(name))

def load_clip_mfcc(base: str, npz_rel: str) -> np.ndarray:
    return np.load(os.path.join(base, npz_rel))["mfcc"].astype(np.float32, copy=False)

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

# ------------- Subspace math -------------
def orthonormal_columns(M: np.ndarray) -> np.ndarray:
    """QR for safety; returns D×k with orthonormal columns."""
    Q, _ = np.linalg.qr(M)
    return Q

def fit_pca_subspace(X: np.ndarray, k: int) -> np.ndarray:
    """
    X: (N,D) rows = frames. Mean-center across frames, fit PCA, return D×k basis.
    """
    if X.ndim != 2 or X.shape[0] < 2:
        return np.empty((X.shape[1], 0), dtype=np.float32) if X.ndim == 2 else np.empty((0, 0), dtype=np.float32)
    Xc = X - X.mean(axis=0, keepdims=True)
    n_comp = int(min(k, Xc.shape[0]-1, Xc.shape[1]))
    if n_comp < 1:
        return np.empty((X.shape[1], 0), dtype=np.float32)
    pca = PCA(n_components=n_comp, svd_solver="full", random_state=RANDOM_SEED)
    pca.fit(Xc)
    U = pca.components_.T.astype(np.float32)  # D×n_comp, columns orthonormal
    return U

def canonical_correlations(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    U: D×r, V: D×s with (approx) orthonormal columns.
    Returns singular values of U^T V, sorted descending (these are cos(theta_i)).
    """
    if U.size == 0 or V.size == 0:
        return np.array([], dtype=np.float32)
    r = min(U.shape[1], V.shape[1])
    if r < 1:
        return np.array([], dtype=np.float32)
    # safety orthonormalization
    Uo = orthonormal_columns(U[:, :r])
    Vo = orthonormal_columns(V[:, :r])
    s = np.linalg.svd(Uo.T @ Vo, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)  # numerical
    return s  # descending

def msm_score(Ua: np.ndarray, Vc: np.ndarray, metric: str = "sumcos2") -> float:
    """
    Ua: class subspace D×r ; Vc: clip subspace D×s
    metric: 'sumcos2' (∑ s_i^2) or 'minangle' (max s_i^2)
    """
    s = canonical_correlations(Ua, Vc)
    if s.size == 0:
        return float(0.0 if metric == "sumcos2" else 0.0)
    if metric == "minangle":
        return float(np.max(s)**2)
    # default sumcos2
    return float(np.sum(s**2))

# ------------- Data pooling -------------
def pool_frames_for_clip(df_frames: pd.DataFrame,
                         clip: str,
                         cache: Dict[str, np.ndarray],
                         base: str) -> np.ndarray:
    """
    Returns (T,D) MFCC frames for the clip using indices in df_frames.
    """
    g = df_frames[df_frames["clip_filename"] == clip]
    if g.empty:
        return np.empty((0, 0), dtype=np.float32)
    npz_rel = g["mfcc_npz_path"].iloc[0]
    if clip not in cache:
        try:
            cache[clip] = load_clip_mfcc(base, npz_rel)
        except Exception:
            return np.empty((0, 0), dtype=np.float32)
    mf = cache[clip]
    idx = g["mfcc_idx"].astype(int).to_numpy()
    idx = idx[(idx >= 0) & (idx < mf.shape[0])]
    if idx.size == 0:
        return np.empty((0, mf.shape[1]), dtype=np.float32)
    return mf[idx, :]

def fit_class_subspaces(train_df: pd.DataFrame,
                        base: str,
                        r_class: int) -> Dict[str, np.ndarray]:
    """
    Fit class subspaces U_a (D×r) by pooling TRAIN frames per class and PCA with k=r_class.
    """
    cache: Dict[str, np.ndarray] = {}
    subspaces: Dict[str, np.ndarray] = {}
    for cls, g in train_df.groupby("engine_configuration", sort=False):
        # pool all frames for this class
        parts: List[np.ndarray] = []
        for clip in g["clip_filename"].unique():
            Xc = pool_frames_for_clip(g, clip, cache, base)
            if Xc.size:
                parts.append(Xc)
        if not parts:
            subspaces[cls] = np.empty((0, 0), dtype=np.float32)
            continue
        X = np.vstack(parts).astype(np.float32)
        U = fit_pca_subspace(X, k=r_class)
        subspaces[cls] = U
    return subspaces

# ------------- Calibration -------------
def build_msm_calibration(train_df: pd.DataFrame,
                          subspaces: Dict[str, np.ndarray],
                          base: str,
                          s_clip: int,
                          metric: str,
                          robust: bool) -> Dict[str, Dict[str, float]]:
    """
    For each class a: compute scores S_a(c) for TRAIN clips c with true label a,
    then compute mu/sigma (or median/MAD) for z-scoring at test time.
    """
    classes = sorted(subspaces.keys())
    by_clip = train_df.groupby(["clip_filename", "engine_configuration"], sort=False)
    cache: Dict[str, np.ndarray] = {}
    per_class_vals: Dict[str, List[float]] = {c: [] for c in classes}

    for (clip, true_cls), g in by_clip:
        if true_cls not in classes:
            continue
        X = pool_frames_for_clip(train_df, clip, cache, base)
        if X.size == 0:
            continue
        Vc = fit_pca_subspace(X, k=s_clip)
        if Vc.size == 0 or subspaces[true_cls].size == 0:
            continue
        S = msm_score(subspaces[true_cls], Vc, metric=metric)
        per_class_vals[true_cls].append(S)

    stats: Dict[str, Dict[str, float]] = {}
    for cls, vals in per_class_vals.items():
        arr = np.array(vals, dtype=np.float64)
        if arr.size == 0:
            stats[cls] = {"mu": np.nan, "sigma": np.nan, "n_clips": 0}
            continue
        if robust:
            mu = float(np.median(arr))
            mad = float(np.median(np.abs(arr - mu)))
            sigma = float(1.4826 * mad)
            if sigma <= 0:
                sigma = float(np.std(arr))
        else:
            mu = float(np.mean(arr))
            sigma = float(np.std(arr))
        sigma = float(sigma + EPS)
        stats[cls] = {"mu": mu, "sigma": sigma, "n_clips": int(arr.size)}
    return stats

# ------------- Scoring / Prediction -------------
def evaluate_msm(base: str,
                 s_clip: int,
                 metric: str,
                 robust: bool,
                 plots: bool,
                 label_suffix: str) -> Dict:
    ensure_dir(os.path.join(base, OUT_DIR))

    # Load splits
    train_df = pd.read_parquet(os.path.join(base, SPLITS_DIR, TRAIN_FRAMES_PARQUET))
    test_df  = pd.read_parquet(os.path.join(base, SPLITS_DIR, TEST_FRAMES_PARQUET))

    classes = sorted(train_df["engine_configuration"].drop_duplicates().tolist())

    # Fit class subspaces U_a with rank r=5 on TRAIN
    subspaces = fit_class_subspaces(train_df, base, r_class=R_CLASS)

    # Calibration from TRAIN clips (true-class scores)
    calib = build_msm_calibration(train_df, subspaces, base, s_clip=s_clip, metric=metric, robust=robust)

    # Score TEST clips
    by_clip = test_df.groupby(["clip_filename", "engine_configuration"], sort=False)
    cache: Dict[str, np.ndarray] = {}
    rows = []
    preds = []
    for (clip, true_cls), g in by_clip:
        X = pool_frames_for_clip(test_df, clip, cache, base)
        if X.size == 0:
            continue
        Vc = fit_pca_subspace(X, k=s_clip)
        if Vc.size == 0:
            continue

        Sa = {}
        Za = {}
        for cls in classes:
            Ua = subspaces.get(cls, np.empty((0,0), dtype=np.float32))
            if Ua.size == 0:
                Sa[cls] = 0.0
                Za[cls] = -np.inf
                continue
            s_val = msm_score(Ua, Vc, metric=metric)
            Sa[cls] = s_val
            mu = calib.get(cls, {}).get("mu", np.nan)
            sg = calib.get(cls, {}).get("sigma", np.nan)
            if np.isnan(mu) or np.isnan(sg):
                Za[cls] = -np.inf
            else:
                Za[cls] = (s_val - mu) / sg

        ranked_cal = sorted([(Za[c], Sa[c], -R_CLASS, c) for c in classes], reverse=True)  # argmax; tie by raw S then rank
        pred_cal = ranked_cal[0][3]
        preds.append((clip, true_cls, pred_cal))

        for cls in classes:
            rows.append({
                "clip_filename": clip,
                "true": true_cls,
                "class": cls,
                "S": float(Sa[cls]),
                "mu_train": float(calib.get(cls, {}).get("mu", np.nan)),
                "sigma_train": float(calib.get(cls, {}).get("sigma", np.nan)),
                "Z": float(Za[cls]),
                "pred_calibrated": pred_cal,
            })

    long_df = pd.DataFrame(rows)
    score_csv = os.path.join(base, OUT_DIR, f"msm_scores_{label_suffix}.csv")
    long_df.to_csv(score_csv, index=False)

    # Metrics
    by_clip_df = long_df[['clip_filename','true','pred_calibrated']].drop_duplicates(subset=['clip_filename'])
    y_true = by_clip_df["true"].tolist()
    y_pred = by_clip_df["pred_calibrated"].tolist()

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    overall = float(np.mean(np.array(y_true) == np.array(y_pred)))
    macro = float(balanced_accuracy_score(y_true, y_pred))

    report = classification_report(y_true, y_pred, labels=classes, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(os.path.join(base, OUT_DIR, f"msm_report_{label_suffix}.csv"))

    if plots:
        plot_confmat(cm, classes, os.path.join(base, OUT_DIR, f"msm_cm_{label_suffix}.png"), normalize=False)
        plot_confmat(cm, classes, os.path.join(base, OUT_DIR, f"msm_cm_norm_{label_suffix}.png"), normalize=True)

        # True-class score distribution (S vs Z) for insight
        true_rows = long_df[long_df["class"] == long_df["true"]].copy()
        if not true_rows.empty:
            plt.figure(figsize=(8,4))
            plt.subplot(1,2,1)
            plt.violinplot(true_rows["S"].values, showmeans=True, showmedians=True)
            plt.xticks([1], ["S_true"])
            plt.title("True-class MSM scores (raw)")
            plt.subplot(1,2,2)
            plt.violinplot(true_rows["Z"].values, showmeans=True, showmedians=True)
            plt.xticks([1], ["Z_true"])
            plt.title("True-class MSM scores (z-scored)")
            plt.tight_layout()
            plt.savefig(os.path.join(base, OUT_DIR, f"msm_true_violin_{label_suffix}.png"), dpi=150)
            plt.close()

    summary = {
        "classes": classes,
        "r_class": R_CLASS,
        "s_clip": s_clip,
        "metric": metric,
        "robust": robust,
        "overall_accuracy": overall,
        "macro_accuracy": macro,
        "n_test_clips": int(by_clip_df.shape[0]),
    }
    with open(os.path.join(base, OUT_DIR, f"msm_metrics_{label_suffix}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("=== MSM evaluation complete ===")
    print(json.dumps(summary, indent=2))
    print("Scores CSV:", score_csv)
    return summary

# ------------- CLI / Ablation -------------
def main():
    ap = argparse.ArgumentParser(description="Mutual Subspace Method (MSM) clip classification with calibration.")
    ap.add_argument("--data_dir", default=DATA_DIR, type=str)
    ap.add_argument("--s", default=2, type=int, help="Clip subspace rank s (2 or 3).")
    ap.add_argument("--metric", default="sumcos2", choices=["sumcos2", "minangle"],
                    help="MSM score: sum of squared canonical correlations, or min-angle (max cos^2).")
    ap.add_argument("--robust", action="store_true", help="Use median/MAD calibration instead of mean/std.")
    ap.add_argument("--plots", action="store_true", help="Save confusion matrices and score distributions.")
    ap.add_argument("--ablate", action="store_true", help="Run ablation over s_list and metric_list.")
    ap.add_argument("--s_list", default="2,3", type=str, help="Comma-separated list of s values for ablation.")
    ap.add_argument("--metric_list", default="sumcos2,minangle", type=str,
                    help="Comma-separated list of metrics for ablation.")
    args = ap.parse_args()

    base = args.data_dir
    ensure_dir(os.path.join(base, OUT_DIR))

    if not args.ablate:
        label = f"s{args.s}_{args.metric}{'_rob' if args.robust else ''}"
        evaluate_msm(base, s_clip=args.s, metric=args.metric, robust=args.robust, plots=args.plots, label_suffix=label)
        return

    # Ablation loop
    s_vals = [int(x) for x in args.s_list.split(",") if x.strip()]
    met_vals = [m.strip() for m in args.metric_list.split(",") if m.strip()]
    rows = []
    for s_clip in s_vals:
        for met in met_vals:
            label = f"s{s_clip}_{met}{'_rob' if args.robust else ''}"
            try:
                res = evaluate_msm(base, s_clip=s_clip, metric=met, robust=args.robust, plots=args.plots, label_suffix=label)
                rows.append(res)
            except Exception as e:
                rows.append({"s_clip": s_clip, "metric": met, "error": str(e)})
    ablate_path = os.path.join(base, OUT_DIR, f"msm_ablation_{'_'.join([str(x) for x in s_vals])}__{'_'.join(met_vals)}{'_rob' if args.robust else ''}.csv")
    pd.DataFrame(rows).to_csv(ablate_path, index=False)
    print("Ablation saved to:", ablate_path)

if __name__ == "__main__":
    main()
