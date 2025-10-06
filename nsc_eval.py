# nsc_eval.py
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

# ---------- Defaults ----------
DATA_DIR = "Data"
SPLITS_DIR = "splits"
PCA_DIR = "pca"
NSC_DIR = "nsc"

TEST_FRAMES_PARQUET = "test_frames.parquet"
PCA_SUMMARY_CSV = "pca_summary.csv"

RANDOM_SEED = 42
# ------------------------------

np.random.seed(RANDOM_SEED)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def canonicalize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-]+", "_", str(name))

def load_models_from_summary(data_dir: str, pca_dir: str, summary_csv: str):
    """
    Load per-class PCA models (mean, components, rank) from pca_summary.csv.
    Fallback to Data/pca/<safe>_pca.npz if pca_npz path is missing.
    Returns dict: cls -> {"U": (D,r), "mu": (D,), "rank": int}
    """
    summ_path = os.path.join(data_dir, pca_dir, summary_csv)
    if not os.path.exists(summ_path):
        # backwards-compat fallback: list *_pca.npz
        model_paths = []
        for fn in os.listdir(os.path.join(data_dir, pca_dir)):
            if fn.endswith("_pca.npz"):
                model_paths.append((fn.replace("_pca.npz", ""), os.path.join(data_dir, pca_dir, fn)))
        if not model_paths:
            raise FileNotFoundError("No PCA models found. Run split_and_pca_per_class.py first.")
        models = {}
        for safe, path in model_paths:
            z = np.load(path)
            U = z["components"].T  # (D,r)
            mu = z["mean"]
            r = int(z["rank"][0]) if "rank" in z else U.shape[1]
            cls = safe  # safe name; if you prefer pretty labels, keep a mapping
            models[cls] = {"U": U.astype(np.float32), "mu": mu.astype(np.float32), "rank": r}
        return models

    df = pd.read_csv(summ_path)
    models = {}
    for _, row in df.iterrows():
        if row.get("status", "ok") != "ok":
            continue
        cls = row["engine_configuration"]
        # primary path from csv; fallback to default naming
        rel = row.get("pca_npz", "")
        if isinstance(rel, str) and len(rel):
            npz_path = os.path.join(data_dir, rel)
        else:
            safe = canonicalize(cls)
            npz_path = os.path.join(data_dir, pca_dir, f"{safe}_pca.npz")
        if not os.path.exists(npz_path):
            continue
        z = np.load(npz_path)
        U = z["components"].T  # (D,r)
        mu = z["mean"]
        r = int(z["rank"][0]) if "rank" in z else U.shape[1]
        models[cls] = {"U": U.astype(np.float32), "mu": mu.astype(np.float32), "rank": r}
    if not models:
        raise RuntimeError("Found pca_summary.csv but could not load any PCA npz files.")
    return models

def load_clip_mfcc(data_dir: str, npz_rel: str) -> np.ndarray:
    return np.load(os.path.join(data_dir, npz_rel))["mfcc"].astype(np.float32, copy=False)

def frame_residuals(X: np.ndarray, mu: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    X: (N,D) MFCC frames for one clip
    mu: (D,)
    U: (D,r) with orthonormal columns
    Returns r(x_i) = ||(I-UU^T)(x_i - mu)||_2 per frame (length N)
    Uses energy identity for speed: ||x||^2 - ||U^T x||^2.
    """
    Xc = X - mu[None, :]
    tot = np.sum(Xc * Xc, axis=1)
    if U.size == 0:
        resid2 = tot
    else:
        coeff = Xc @ U     # (N,r)
        proj = np.sum(coeff * coeff, axis=1)
        resid2 = np.maximum(tot - proj, 0.0)
    return np.sqrt(resid2, dtype=np.float32)

def aggregate_residual(resids: np.ndarray, trim: float = 0.0) -> float:
    """
    Median by default. If trim>0, use symmetric trimmed mean with fraction trim (e.g., 0.1).
    """
    if resids.size == 0:
        return float("inf")
    if trim <= 0:
        return float(np.median(resids))
    trim = float(np.clip(trim, 0.0, 0.49))
    k = resids.size
    if k < 2:
        return float(resids[0])
    lo = int(np.floor(trim * k))
    hi = int(np.ceil((1.0 - trim) * k))
    sel = np.sort(resids)[lo:hi]
    if sel.size == 0:
        return float(np.median(resids))
    return float(np.mean(sel))

def predict_clip(X: np.ndarray, models: Dict[str, Dict], trim: float = 0.0) -> Tuple[str, Dict[str, float]]:
    """
    Returns predicted class and dict of R_a per class.
    Tie-break: smaller R, then smaller rank, then class name.
    """
    Ra = {}
    for cls, m in models.items():
        r = frame_residuals(X, m["mu"], m["U"])
        Ra[cls] = aggregate_residual(r, trim=trim)
    # tie-break
    ranked = sorted([(Ra[c], models[c]["rank"], c) for c in Ra])
    return ranked[0][2], Ra

# Replace your plot_confmat() with this version
def plot_confmat(cm: np.ndarray, classes: List[str], out_path: str, normalize: bool = False):
    plt.figure(figsize=(6, 5))

    if normalize:
        C = cm.astype(np.float64)
        row_sums = C.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # avoid div-by-zero for empty classes
        C_disp = C / row_sums
        fmt = ".2f"
    else:
        # keep counts but display as integer-rounded floats
        C_disp = cm.astype(np.float64)
        fmt = ".0f"

    im = plt.imshow(C_disp, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45, ha='right')
    plt.yticks(ticks, classes)

    thresh = (C_disp.max() + C_disp.min()) / 2.0
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

def main():
    ap = argparse.ArgumentParser(description="Nearest-Subspace Classification on TEST clips.")
    ap.add_argument("--data_dir", default=DATA_DIR, type=str)
    ap.add_argument("--trim", default=0.0, type=float, help="Trim fraction for trimmed mean (0 => median).")
    ap.add_argument("--plots", action="store_true", help="Write confusion matrix and residual plots.")
    ap.add_argument("--perm", default=0, type=int, help="Permutation shuffles for sanity check (default 0).")
    args = ap.parse_args()

    base = args.data_dir
    splits_dir = os.path.join(base, SPLITS_DIR)
    pca_dir = os.path.join(base, PCA_DIR)
    out_dir = os.path.join(base, NSC_DIR)
    ensure_dir(out_dir)

    # --- Load PCA models (TRAIN-fitted) ---
    models = load_models_from_summary(base, PCA_DIR, PCA_SUMMARY_CSV)
    classes = sorted(models.keys())  # fixed class order
    A = len(classes)
    chance = 1.0 / max(A, 1)

    # --- Load TEST frames index ---
    test_frames_path = os.path.join(splits_dir, TEST_FRAMES_PARQUET)
    if not os.path.exists(test_frames_path):
        raise FileNotFoundError(f"Missing {test_frames_path}. Run split_and_pca_per_class.py first.")
    fi = pd.read_parquet(test_frames_path)

    need = {"clip_filename", "engine_configuration", "mfcc_npz_path", "mfcc_idx"}
    missing = need - set(fi.columns)
    if missing:
        raise KeyError(f"{TEST_FRAMES_PARQUET} missing columns: {sorted(missing)}")

    # Keep only clips whose labels are among loaded models (if some classes were skipped in PCA)
    fi = fi[fi["engine_configuration"].isin(classes)].copy()
    if fi.empty:
        raise RuntimeError("No TEST frames left after filtering to classes with PCA models.")

    # --- Count TRAIN/TEST clips per class for provenance ---
    split_df = pd.read_parquet(os.path.join(splits_dir, "clip_split.parquet"))
    train_counts = split_df[split_df["split"] == "train"]["engine_configuration"].value_counts().reindex(classes).fillna(0).astype(int).to_dict()
    test_counts = split_df[split_df["split"] == "test"]["engine_configuration"].value_counts().reindex(classes).fillna(0).astype(int).to_dict()

    # --- Group TEST by clip ---
    groups = fi.groupby(["clip_filename", "engine_configuration"], sort=False)

    # Cache MFCC per clip
    mfcc_cache: Dict[str, np.ndarray] = {}

    # --- Predict clip labels ---
    rows = []
    for (clip, true_cls), g in groups:
        npz_rel = g["mfcc_npz_path"].iloc[0]
        if clip not in mfcc_cache:
            try:
                mfcc_cache[clip] = load_clip_mfcc(base, npz_rel)
            except Exception:
                continue
        mf = mfcc_cache[clip]
        idx = g["mfcc_idx"].astype(int).to_numpy()
        idx = idx[(idx >= 0) & (idx < mf.shape[0])]
        if idx.size == 0:
            continue
        X = mf[idx, :]  # (T_c, D)

        pred, Ra = predict_clip(X, models, trim=args.trim)
        R_true = float(Ra.get(true_cls, np.inf))
        R_min_other = float(min(v for k, v in Ra.items() if k != true_cls)) if len(Ra) > 1 else float("nan")
        rows.append({
            "clip_filename": clip,
            "true": true_cls,
            "pred": pred,
            "R_true": R_true,
            "R_min_other": R_min_other,
            **{f"R[{k}]": float(v) for k, v in Ra.items()}
        })

    if not rows:
        raise RuntimeError("No TEST predictions were produced. Check inputs.")
    preds_df = pd.DataFrame(rows)
    preds_csv = os.path.join(out_dir, "nsc_clip_results.csv")
    preds_df.to_csv(preds_csv, index=False)

    # --- Metrics ---
    y_true = preds_df["true"].to_list()
    y_pred = preds_df["pred"].to_list()

    # Fixed class order for CM
    label_order = classes
    cm = confusion_matrix(y_true, y_pred, labels=label_order)
    overall_acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    macro_acc = float(balanced_accuracy_score(y_true, y_pred))  # mean per-class recall

    # Per-class precision/recall/F1
    report = classification_report(y_true, y_pred, labels=label_order, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_csv = os.path.join(out_dir, "per_class_report.csv")
    report_df.to_csv(report_csv)

    # Save confusion matrices
    if args.plots:
        plot_confmat(cm, label_order, os.path.join(out_dir, "confusion_matrix.png"), normalize=False)
        plot_confmat(cm, label_order, os.path.join(out_dir, "confusion_matrix_norm.png"), normalize=True)

        # Residual distributions: R_true vs min R_other
        plt.figure(figsize=(6,4))
        data = [preds_df["R_true"].values, preds_df["R_min_other"].values]
        plt.violinplot(data, showmeans=True, showmedians=True)
        plt.xticks([1,2], ["R_true", "min R_other"])
        plt.ylabel("Clip residual (median over frames)")
        plt.title("Residual distributions on TEST clips")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "residual_violin.png"), dpi=150)
        plt.close()

    # Provenance & summary
    summary = {
        "classes": label_order,
        "chance_baseline": chance,
        "overall_accuracy": overall_acc,
        "macro_accuracy": macro_acc,
        "n_test_clips_total": int(len(preds_df)),
        "train_clip_counts": train_counts,
        "test_clip_counts": test_counts,
        "pca_ranks": {c: int(models[c]["rank"]) for c in label_order},
        "aggregator": ("median" if args.trim <= 0 else f"trimmed_mean_{args.trim:.2f}"),
    }

    # --- Optional permutation test ---
    perm_results = []
    if args.perm and args.perm > 0:
        rng = np.random.default_rng(RANDOM_SEED)
        y_true_arr = np.array(y_true)
        for t in range(int(args.perm)):
            y_perm = rng.permutation(y_true_arr)
            acc_perm = float(np.mean(y_perm == np.array(y_pred)))
            perm_results.append(acc_perm)
        perm_mean = float(np.mean(perm_results))
        perm_std = float(np.std(perm_results))
        # one-sided p-value: how often permuted acc >= observed acc
        greater = int(np.sum(np.array(perm_results) >= overall_acc))
        p_val = (greater + 1) / (len(perm_results) + 1)
        summary.update({
            "perm_mean_acc": perm_mean,
            "perm_std_acc": perm_std,
            "perm_p_value": float(p_val),
            "perm_runs": int(len(perm_results)),
        })
        # Save histogram
        if args.plots:
            plt.figure(figsize=(6,4))
            plt.hist(perm_results, bins=20)
            plt.axvline(overall_acc, linestyle="--")
            plt.xlabel("Permutation accuracy")
            plt.ylabel("Count")
            plt.title("Permutation test for NSC accuracy")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "perm_hist.png"), dpi=150)
            plt.close()
        # Save raw perm CSV
        pd.DataFrame({"perm_acc": perm_results}).to_csv(os.path.join(out_dir, "perm_results.csv"), index=False)

    # Write metrics
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Console report
    print("=== NSC evaluation complete ===")
    print(f"Chance baseline:   {chance:.3f}")
    print(f"Overall accuracy:  {overall_acc:.3f}")
    print(f"Macro accuracy:    {macro_acc:.3f}")
    print(f"Results CSV:       {preds_csv}")
    print(f"Report CSV:        {report_csv}")
    if args.plots:
        print(f"Confusion matrices: {os.path.join(out_dir,'confusion_matrix.png')} (raw), "
              f"{os.path.join(out_dir,'confusion_matrix_norm.png')} (normalized)")
        print(f"Residual violin:   {os.path.join(out_dir,'residual_violin.png')}")
    print("Train clip counts:", train_counts)
    print("Test clip counts: ", test_counts)
    print("PCA ranks:        ", {c: models[c]['rank'] for c in classes})
    if args.perm and args.perm > 0:
        print(f"Permutation mean acc: {summary['perm_mean_acc']:.3f} "
              f"(std {summary['perm_std_acc']:.3f}), p={summary['perm_p_value']:.4f}")

if __name__ == "__main__":
    main()
