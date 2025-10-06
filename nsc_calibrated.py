# nsc_calibrated.py
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
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score

# ---------------- Defaults ----------------
DATA_DIR = "Data"
SPLITS_DIR = "splits"
PCA_DIR = "pca"
PCA_SUMMARY_CSV = "pca_summary.csv"

OUT_DIR = "nsc_calibrated"
TRAIN_FRAMES_PARQUET = "train_frames.parquet"
TEST_FRAMES_PARQUET  = "test_frames.parquet"

RANDOM_SEED = 42
EPS = 1e-8
# ------------------------------------------

np.random.seed(RANDOM_SEED)

# ------------ Utils ------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def canonicalize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-]+", "_", str(name))

def load_models_from_summary(base: str) -> Dict[str, Dict]:
    """Load per-class PCA models (mean, components, rank) saved by split_and_pca_per_class.py"""
    summ_path = os.path.join(base, PCA_DIR, PCA_SUMMARY_CSV)
    models = {}
    if os.path.exists(summ_path):
        df = pd.read_csv(summ_path)
        for _, row in df.iterrows():
            if row.get("status", "ok") != "ok":
                continue
            cls = row["engine_configuration"]
            rel = row.get("pca_npz", "")
            if isinstance(rel, str) and rel:
                npz_path = os.path.join(base, rel)
            else:
                npz_path = os.path.join(base, PCA_DIR, f"{canonicalize(cls)}_pca.npz")
            if not os.path.exists(npz_path):
                continue
            z = np.load(npz_path)
            U = z["components"].T.astype(np.float32)  # (D,r)
            mu = z["mean"].astype(np.float32)
            r  = int(z["rank"][0]) if "rank" in z else U.shape[1]
            models[cls] = {"U": U, "mu": mu, "rank": r}
    else:
        # Fallback: discover *_pca.npz
        pdir = os.path.join(base, PCA_DIR)
        for fn in os.listdir(pdir):
            if fn.endswith("_pca.npz"):
                z = np.load(os.path.join(pdir, fn))
                U = z["components"].T.astype(np.float32)
                mu = z["mean"].astype(np.float32)
                r  = int(z["rank"][0]) if "rank" in z else U.shape[1]
                cls = fn.replace("_pca.npz", "")
                models[cls] = {"U": U, "mu": mu, "rank": r}
    if not models:
        raise FileNotFoundError("No PCA models found. Run split_and_pca_per_class.py first.")
    return models

def maybe_truncate_rank(models: Dict[str, Dict], uniform_rank: int | None) -> Dict[str, Dict]:
    """Optionally enforce uniform rank across classes by truncating bases."""
    if uniform_rank is None:
        return models
    out = {}
    for cls, m in models.items():
        r_use = int(min(uniform_rank, m["U"].shape[1]))
        out[cls] = {
            "U": m["U"][:, :r_use].copy(),
            "mu": m["mu"].copy(),
            "rank": r_use
        }
    return out

def load_clip_mfcc(base: str, npz_rel: str) -> np.ndarray:
    return np.load(os.path.join(base, npz_rel))["mfcc"].astype(np.float32, copy=False)

def frame_residuals(X: np.ndarray, mu: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Residuals to a subspace: r(x)=|| (I-UU^T)(x-mu) ||_2
    """
    Xc = X - mu[None, :]
    tot = np.sum(Xc * Xc, axis=1)
    if U.size == 0:
        resid2 = tot
    else:
        coeff = Xc @ U
        proj  = np.sum(coeff * coeff, axis=1)
        resid2 = np.maximum(tot - proj, 0.0)
    return np.sqrt(resid2, dtype=np.float32)

def trimmed_mean_best(resids: np.ndarray, q: float = 0.35, min_k: int = 10) -> float:
    """
    Keep the lowest q% residuals, K >= min_k. If N < min_k, fall back to median.
    """
    n = resids.size
    if n == 0:
        return float("inf")
    if n < min_k:
        return float(np.median(resids))
    k = int(round(q * n))
    k = max(min_k, min(k, n))
    # partition to get K smallest without full sort
    part = np.partition(resids, k-1)[:k]
    return float(np.mean(part))

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

# --------------- Core ---------------

def build_calibration_stats(train_df: pd.DataFrame,
                            models: Dict[str, Dict],
                            base: str,
                            q: float,
                            min_k: int,
                            robust: bool) -> Dict[str, Dict[str, float]]:
    """
    For each class a, compute mu_train & sigma_train over TRAIN clips of class a.
    Returns dict[class] -> {"mu": float, "sigma": float, "n_clips": int}
    """
    # group train by (clip, label)
    groups = train_df.groupby(["clip_filename", "engine_configuration"], sort=False)
    # cache MFCC per clip
    cache: Dict[str, np.ndarray] = {}

    # collect per-class list of R_trim for true-class residuals
    per_class_vals: Dict[str, List[float]] = {c: [] for c in models.keys()}

    for (clip, true_cls), g in groups:
        if true_cls not in models:
            continue
        npz_rel = g["mfcc_npz_path"].iloc[0]
        if clip not in cache:
            try:
                cache[clip] = load_clip_mfcc(base, npz_rel)
            except Exception:
                continue
        mf = cache[clip]
        idx = g["mfcc_idx"].astype(int).to_numpy()
        idx = idx[(idx >= 0) & (idx < mf.shape[0])]
        if idx.size == 0:
            continue
        X = mf[idx, :]

        # residuals against its own class subspace
        m = models[true_cls]
        res = frame_residuals(X, m["mu"], m["U"])
        R_trim = trimmed_mean_best(res, q=q, min_k=min_k)
        per_class_vals[true_cls].append(R_trim)

    # compute calibration stats
    stats: Dict[str, Dict[str, float]] = {}
    for cls, vals in per_class_vals.items():
        arr = np.array(vals, dtype=np.float64)
        if arr.size == 0:
            stats[cls] = {"mu": np.nan, "sigma": np.nan, "n_clips": 0}
            continue
        if robust:
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)))
            sigma = 1.4826 * mad
            if sigma <= 0:
                sigma = float(np.std(arr))
        else:
            med = float(np.mean(arr))
            sigma = float(np.std(arr))
        sigma = float(sigma + EPS)
        stats[cls] = {"mu": med, "sigma": sigma, "n_clips": int(arr.size)}
    return stats

def score_test_clips(test_df: pd.DataFrame,
                     models: Dict[str, Dict],
                     calib: Dict[str, Dict[str, float]],
                     base: str,
                     q: float,
                     min_k: int) -> Tuple[pd.DataFrame, Dict]:
    """
    For each TEST clip and each class, compute R_trim and calibrated Z.
    Return long dataframe (rows = clip × class) and summary dict.
    """
    classes = sorted(models.keys())
    groups = test_df.groupby(["clip_filename", "engine_configuration"], sort=False)
    cache: Dict[str, np.ndarray] = {}

    rows = []
    # We’ll also compute uncalibrated argmin over R_trim for comparison.
    clip_preds_cal = []
    clip_preds_raw = []

    for (clip, true_cls), g in groups:
        if true_cls not in classes:
            continue
        npz_rel = g["mfcc_npz_path"].iloc[0]
        if clip not in cache:
            try:
                cache[clip] = load_clip_mfcc(base, npz_rel)
            except Exception:
                continue
        mf = cache[clip]
        idx = g["mfcc_idx"].astype(int).to_numpy()
        idx = idx[(idx >= 0) & (idx < mf.shape[0])]
        if idx.size == 0:
            continue
        X = mf[idx, :]

        # compute per-class R_trim and Z
        Ra = {}
        Za = {}
        for cls in classes:
            m = models[cls]
            res = frame_residuals(X, m["mu"], m["U"])
            r_trim = trimmed_mean_best(res, q=q, min_k=min_k)
            Ra[cls] = r_trim
            mu_tr = calib.get(cls, {}).get("mu", np.nan)
            sg_tr = calib.get(cls, {}).get("sigma", np.nan)
            if np.isnan(mu_tr) or np.isnan(sg_tr):
                Za[cls] = np.inf
            else:
                Za[cls] = (r_trim - mu_tr) / sg_tr

        # predictions (calibrated & raw)
        ranked_cal = sorted([(Za[c], Ra[c], models[c]["rank"], c) for c in classes])
        ranked_raw = sorted([(Ra[c], models[c]["rank"], c) for c in classes])

        pred_cal = ranked_cal[0][3]
        pred_raw = ranked_raw[0][2]
        clip_preds_cal.append((clip, true_cls, pred_cal))
        clip_preds_raw.append((clip, true_cls, pred_raw))

        for cls in classes:
            rows.append({
                "clip_filename": clip,
                "true": true_cls,
                "class": cls,
                "R_trim": float(Ra[cls]),
                "mu_train": float(calib.get(cls, {}).get("mu", np.nan)),
                "sigma_train": float(calib.get(cls, {}).get("sigma", np.nan)),
                "Z": float(Za[cls]),
                "pred_calibrated": pred_cal,
                "pred_raw": pred_raw,
            })

    long_df = pd.DataFrame(rows)
    # quick provenance
    summary = {"n_test_clips": int(len(set([c for c,_,_ in clip_preds_cal]))), "classes": classes}
    return long_df, summary

def metrics_and_plots(long_df: pd.DataFrame,
                      classes: List[str],
                      out_dir: str,
                      label: str):
    """
    Compute metrics for calibrated (Z) and raw (R_trim) predictions and save CMs + reports.
    `label` is a suffix for filenames (e.g., 'q035_r5' or 'q035').
    """
    # per-clip predictions are duplicated across classes; pick one row per clip by any class (e.g., first)
    # Extract via groupby on clip_filename
    by_clip = (long_df[['clip_filename','true','pred_calibrated','pred_raw']]
               .drop_duplicates(subset=['clip_filename']))

    y_true = by_clip["true"].tolist()
    y_pred_cal = by_clip["pred_calibrated"].tolist()
    y_pred_raw = by_clip["pred_raw"].tolist()

    # ---- Calibrated ----
    cm_cal = confusion_matrix(y_true, y_pred_cal, labels=classes)
    overall_cal = float(np.mean(np.array(y_true) == np.array(y_pred_cal)))
    macro_cal = float(balanced_accuracy_score(y_true, y_pred_cal))

    report_cal = classification_report(y_true, y_pred_cal, labels=classes, output_dict=True, zero_division=0)
    pd.DataFrame(report_cal).transpose().to_csv(os.path.join(out_dir, f"report_cal_{label}.csv"))

    plot_confmat(cm_cal, classes, os.path.join(out_dir, f"cm_cal_{label}.png"), normalize=False)
    plot_confmat(cm_cal, classes, os.path.join(out_dir, f"cm_cal_norm_{label}.png"), normalize=True)

    # ---- Raw (uncalibrated) ----
    cm_raw = confusion_matrix(y_true, y_pred_raw, labels=classes)
    overall_raw = float(np.mean(np.array(y_true) == np.array(y_pred_raw)))
    macro_raw = float(balanced_accuracy_score(y_true, y_pred_raw))

    report_raw = classification_report(y_true, y_pred_raw, labels=classes, output_dict=True, zero_division=0)
    pd.DataFrame(report_raw).transpose().to_csv(os.path.join(out_dir, f"report_raw_{label}.csv"))

    plot_confmat(cm_raw, classes, os.path.join(out_dir, f"cm_raw_{label}.png"), normalize=False)
    plot_confmat(cm_raw, classes, os.path.join(out_dir, f"cm_raw_norm_{label}.png"), normalize=True)

    # ---- Residual distribution plot (true class): R_trim vs Z ----
    # For each clip, grab the row where class==true
    true_rows = long_df[long_df["class"] == long_df["true"]].copy()
    if not true_rows.empty:
        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.violinplot(true_rows["R_trim"].values, showmeans=True, showmedians=True)
        plt.xticks([1], ["R_true_trim"])
        plt.ylabel("Residual")
        plt.title("Raw residuals (true class)")
        plt.subplot(1,2,2)
        plt.violinplot(true_rows["Z"].values, showmeans=True, showmedians=True)
        plt.xticks([1], ["Z_true"])
        plt.title("Calibrated z-scores (true class)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"residual_violin_true_{label}.png"), dpi=150)
        plt.close()

    # ---- Return metrics
    return {
        "overall_acc_cal": overall_cal,
        "macro_acc_cal": macro_cal,
        "overall_acc_raw": overall_raw,
        "macro_acc_raw": macro_raw,
    }

# --------------- Driver ---------------

def run_single(base: str,
               q: float,
               min_k: int,
               uniform_rank: int | None,
               robust: bool,
               plots: bool,
               label_extra: str) -> Dict:
    ensure_dir(os.path.join(base, OUT_DIR))

    # Load models & truncate rank if requested
    models0 = load_models_from_summary(base)
    models = maybe_truncate_rank(models0, uniform_rank)

    # Filter frames to only classes with models
    train_df = pd.read_parquet(os.path.join(base, SPLITS_DIR, TRAIN_FRAMES_PARQUET))
    test_df  = pd.read_parquet(os.path.join(base, SPLITS_DIR, TEST_FRAMES_PARQUET))
    classes = sorted(models.keys())
    train_df = train_df[train_df["engine_configuration"].isin(classes)].copy()
    test_df  = test_df[test_df["engine_configuration"].isin(classes)].copy()
    if train_df.empty or test_df.empty:
        raise RuntimeError("Empty TRAIN/TEST after filtering to classes with PCA models.")

    # Calibration from TRAIN (true-class residuals)
    calib = build_calibration_stats(train_df, models, base, q=q, min_k=min_k, robust=robust)

    # Score TEST
    long_df, info = score_test_clips(test_df, models, calib, base, q=q, min_k=min_k)

    # Save clip×class scores
    label = f"q{int(round(q*100)):02d}{label_extra}"
    out_dir = os.path.join(base, OUT_DIR)
    long_df.to_csv(os.path.join(out_dir, f"clip_scores_{label}.csv"), index=False)

    # Metrics & plots
    metrics = metrics_and_plots(long_df, classes, out_dir, label=label if plots else label)
    # Save metrics JSON
    summary = {
        "q": q,
        "min_k": min_k,
        "uniform_rank": uniform_rank,
        "robust": robust,
        "classes": classes,
        "pca_ranks": {c: int(models[c]["rank"]) for c in classes},
        **metrics
    }
    with open(os.path.join(out_dir, f"metrics_{label}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary

def main():
    ap = argparse.ArgumentParser(description="NSC with Calibrated Residuals + Trimmed-Frame Aggregation")
    ap.add_argument("--data_dir", default=DATA_DIR, type=str)
    ap.add_argument("--q", default=0.35, type=float, help="Trim fraction (keep lowest q%% of frames, default 0.35)")
    ap.add_argument("--min_k", default=10, type=int, help="Minimum kept frames for trimmed mean (fallback to median if fewer).")
    ap.add_argument("--uniform_rank", default=None, type=lambda s: None if s in ["", "none", "None"] else int(s),
                    help="Optional uniform rank across classes (e.g., 4 or 5).")
    ap.add_argument("--robust", action="store_true", help="Use robust calibration (median + 1.4826*MAD).")
    ap.add_argument("--plots", action="store_true", help="Save confusion matrices and residual plots.")
    ap.add_argument("--ablate", action="store_true", help="Run an ablation over q_list (and current uniform_rank).")
    ap.add_argument("--q_list", default="0.30,0.35,0.40", type=str, help="Comma-separated q values for ablation.")
    args = ap.parse_args()

    base = args.data_dir
    ensure_dir(os.path.join(base, OUT_DIR))

    if not args.ablate:
        label_extra = f"_r{args.uniform_rank}" if args.uniform_rank is not None else ""
        summary = run_single(base, args.q, args.min_k, args.uniform_rank, args.robust, args.plots, label_extra)
        print("=== NSC Calibrated (single run) ===")
        print(json.dumps(summary, indent=2))
        return

    # Ablation over q_list
    q_vals = [float(s) for s in args.q_list.split(",") if s.strip()]
    rows = []
    for q in q_vals:
        label_extra = f"_r{args.uniform_rank}" if args.uniform_rank is not None else ""
        try:
            summary = run_single(base, q, args.min_k, args.uniform_rank, args.robust, args.plots, label_extra)
            rows.append(summary)
        except Exception as e:
            rows.append({
                "q": q, "uniform_rank": args.uniform_rank, "error": str(e)
            })
    ablate_df = pd.DataFrame(rows)
    ablate_path = os.path.join(base, OUT_DIR, f"ablation_q{'_'.join([str(int(round(100*x))) for x in q_vals])}"
                                              f"{'_r'+str(args.uniform_rank) if args.uniform_rank is not None else ''}.csv")
    ablate_df.to_csv(ablate_path, index=False)
    print("=== Ablation complete ===")
    print(ablate_df)
    print("Saved:", ablate_path)

if __name__ == "__main__":
    main()
