# pairwise_subspace_angles.py
# Requirements:
#   pip install numpy pandas matplotlib pyarrow

import os
import re
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------- Defaults --------
DATA_DIR = "Data"
PCA_DIR = "pca"
PCA_SUMMARY_CSV = "pca_summary.csv"
OUT_DIR = "pairwise"
RANDOM_SEED = 42
# --------------------------

np.random.seed(RANDOM_SEED)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def canonicalize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-]+", "_", str(name))

def load_models_from_summary(data_dir: str, pca_dirname: str, summary_csv: str):
    """
    Load per-class PCA (mean, components, rank) from pca_summary.csv (and npz files).
    Returns dict: cls -> {"U": (D,r), "mu": (D,), "rank": int}
    """
    summ_path = os.path.join(data_dir, pca_dirname, summary_csv)
    models = {}

    if os.path.exists(summ_path):
        df = pd.read_csv(summ_path)
        for _, row in df.iterrows():
            if row.get("status", "ok") != "ok":
                continue
            cls = row["engine_configuration"]
            rel = row.get("pca_npz", "")
            if isinstance(rel, str) and rel:
                npz_path = os.path.join(data_dir, rel)
            else:
                npz_path = os.path.join(data_dir, pca_dirname, f"{canonicalize(cls)}_pca.npz")
            if not os.path.exists(npz_path):
                continue
            z = np.load(npz_path)
            U = z["components"].T.astype(np.float32)  # (D,r)
            mu = z["mean"].astype(np.float32)
            r = int(z["rank"][0]) if "rank" in z else U.shape[1]
            models[cls] = {"U": U, "mu": mu, "rank": r}
    else:
        # Fallback: discover *_pca.npz if summary is missing
        pdir = os.path.join(data_dir, pca_dirname)
        for fn in os.listdir(pdir):
            if fn.endswith("_pca.npz"):
                z = np.load(os.path.join(pdir, fn))
                U = z["components"].T.astype(np.float32)
                mu = z["mean"].astype(np.float32)
                r = int(z["rank"][0]) if "rank" in z else U.shape[1]
                cls = fn.replace("_pca.npz", "")
                models[cls] = {"U": U, "mu": mu, "rank": r}

    if not models:
        raise FileNotFoundError("No PCA models found. Run split_and_pca_per_class.py first.")
    return models

def principal_angles_rad(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    """
    Return principal angles (in radians), sorted ascending, length r=min(r1,r2).
    U1: (D,r1), U2: (D,r2), columns are (approx) orthonormal.
    """
    if U1.size == 0 or U2.size == 0:
        return np.array([], dtype=np.float32)
    r = min(U1.shape[1], U2.shape[1])
    if r < 1:
        return np.array([], dtype=np.float32)
    # Orthonormalize (safety)
    Q1, _ = np.linalg.qr(U1[:, :r])
    Q2, _ = np.linalg.qr(U2[:, :r])
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)  # numerical safety
    # Sorted descending singular values -> ascending angles when applying arccos
    thetas = np.arccos(s)
    # ensure non-decreasing
    thetas = np.sort(thetas)
    return thetas.astype(np.float32)

def aggregate_angles_deg(thetas_rad: np.ndarray, metric: str) -> float:
    if thetas_rad.size == 0:
        return float("nan")
    vals = np.degrees(thetas_rad)
    metric = metric.lower()
    if metric == "theta_min":
        return float(np.min(vals))
    if metric == "theta_mean":
        return float(np.mean(vals))
    if metric == "theta_median":
        return float(np.median(vals))
    # default: theta_max
    return float(np.max(vals))

def build_pairwise(models: Dict[str, Dict], metric: str):
    classes = sorted(models.keys())
    n = len(classes)
    mat = np.full((n, n), np.nan, dtype=np.float32)
    rows = []
    for i, ci in enumerate(classes):
        Ui = models[ci]["U"]
        ri = models[ci]["rank"]
        for j, cj in enumerate(classes):
            if j < i:
                continue  # fill symmetric later
            Uj = models[cj]["U"]
            rj = models[cj]["rank"]
            r_common = min(ri, rj)
            th = principal_angles_rad(Ui, Uj)
            val = aggregate_angles_deg(th, metric=metric) if th.size else float("nan")
            mat[i, j] = val
            mat[j, i] = val
            # store pairwise row (i<=j to avoid duplicates)
            rows.append({
                "class_i": ci, "class_j": cj,
                "r_i": int(ri), "r_j": int(rj),
                "r_common": int(r_common),
                "theta_min_deg": float(np.degrees(th).min()) if th.size else np.nan,
                "theta_mean_deg": float(np.degrees(th).mean()) if th.size else np.nan,
                "theta_median_deg": float(np.median(np.degrees(th))) if th.size else np.nan,
                "theta_max_deg": float(np.degrees(th).max()) if th.size else np.nan,
                "metric_value_deg": val,
            })
        mat[i, i] = 0.0  # by convention
    return classes, mat, pd.DataFrame(rows)

def plot_heatmap(mat: np.ndarray, classes: List[str], out_path: str, title: str):
    plt.figure(figsize=(6, 5))
    im = plt.imshow(mat, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45, ha='right')
    plt.yticks(ticks, classes)
    # annotate cells
    vmax = np.nanmax(mat[np.isfinite(mat)])
    vmin = np.nanmin(mat[np.isfinite(mat)])
    thresh = (vmax + vmin) / 2.0 if np.isfinite(vmax) and np.isfinite(vmin) else 0.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isfinite(v):
                txt = "nan"
            else:
                txt = f"{v:.1f}"
            plt.text(j, i, txt,
                     ha="center", va="center",
                     color="white" if (np.isfinite(v) and v > thresh) else "black",
                     fontsize=9)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Class")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Between-class principal angles from TRAIN subspaces.")
    ap.add_argument("--data_dir", default=DATA_DIR, type=str)
    ap.add_argument("--metric", default="theta_max",
                    choices=["theta_max", "theta_min", "theta_mean", "theta_median"],
                    help="Aggregation of principal angles to fill the matrix.")
    args = ap.parse_args()

    base = args.data_dir
    out_dir = os.path.join(base, OUT_DIR)
    ensure_dir(out_dir)

    # Load PCA models
    models = load_models_from_summary(base, PCA_DIR, PCA_SUMMARY_CSV)

    # Build pairwise matrices
    classes, mat, pairs_df = build_pairwise(models, metric=args.metric)

    # Save pairwise table
    pairs_csv = os.path.join(out_dir, "angles_pairs.csv")
    pairs_df.to_csv(pairs_csv, index=False)

    # Save square matrix CSV
    mat_df = pd.DataFrame(mat, index=classes, columns=classes)
    mat_csv = os.path.join(out_dir, f"angles_matrix_{args.metric}.csv")
    mat_df.to_csv(mat_csv)

    # Plot heatmap (“confusion matrix” style)
    title = f"Between-class subspace angles ({args.metric}, deg)"
    fig_path = os.path.join(out_dir, f"angles_confusion_{args.metric}.png")
    plot_heatmap(mat, classes, fig_path, title=title)

    print("=== Between-class principal angles complete ===")
    print(f"Pairwise table:  {pairs_csv}")
    print(f"Matrix CSV:      {mat_csv}")
    print(f"Heatmap image:   {fig_path}")
    print("Classes & ranks: ", {c: int(models[c]['rank']) for c in classes})

if __name__ == "__main__":
    main()
