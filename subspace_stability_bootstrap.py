# subspace_stability_bootstrap.py
# Requirements:
#   pip install pandas numpy scikit-learn matplotlib pyarrow

import os
import re
import math
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# -------------------- Defaults --------------------
DATA_DIR = "Data"
SPLITS_DIR = "splits"
TRAIN_FRAMES_PARQUET = "train_frames.parquet"   # produced by split_and_pca_per_class.py
STABILITY_DIR = "stability"                      # outputs under Data/stability
MFCC_DIRNAME = "mfcc"                            # Data/mfcc/<clip>.npz

EVR_THRESHOLD = 0.90
MAX_RANK = 20
RANDOM_SEED = 42
# --------------------------------------------------

np.random.seed(RANDOM_SEED)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def canonicalize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-]+", "_", str(name))

def load_clip_mfcc(npz_rel: str) -> np.ndarray:
    """Load per-clip MFCC (n_frames, D)."""
    path = os.path.join(DATA_DIR, npz_rel)
    return np.load(path)["mfcc"].astype(np.float32, copy=False)

def fit_pca_and_rank(X: np.ndarray, evr_threshold: float, max_rank: int) -> Tuple[PCA, int]:
    """
    Fit PCA on X (N,D), choose rank r: smallest k with cumsum(EVR) >= threshold,
    capped by max_rank and by data limits (<= min(N,D)).
    """
    N, D = X.shape
    n_comp = int(min(max_rank, N, D))
    if n_comp < 1:
        raise ValueError(f"Not enough data for PCA: N={N}, D={D}")
    pca = PCA(n_components=n_comp, svd_solver="full", random_state=RANDOM_SEED)
    pca.fit(X)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    r = int(np.argmax(cum >= evr_threshold) + 1) if np.any(cum >= evr_threshold) else n_comp
    r = max(1, min(r, n_comp, max_rank))
    return pca, r

def largest_principal_angle_deg(U1: np.ndarray, U2: np.ndarray) -> float:
    """
    U1: (D, r1), U2: (D, r2) with column vectors spanning the subspaces.
    Returns largest principal angle in degrees.
    """
    r = min(U1.shape[1], U2.shape[1])
    if r < 1:
        return float("nan")
    # Orthonormalize columns
    Q1, _ = np.linalg.qr(U1[:, :r])
    Q2, _ = np.linalg.qr(U2[:, :r])
    # Singular values are cos(theta_i), sorted desc
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    # largest angle = arccos(min singular value)
    s_clipped = np.clip(s, -1.0, 1.0)
    theta_max = np.arccos(s_clipped.min())
    return float(np.degrees(theta_max))

def pool_frames_for_clips(df_frames: pd.DataFrame,
                          clip_list: List[str],
                          cache: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Given a frame-level DF (with mfcc_idx & mfcc_npz_path), pool MFCC rows for the given clip list.
    Returns array shape (N_frames, D). Skips clips with missing files.
    """
    parts: List[np.ndarray] = []
    for clip in clip_list:
        g = df_frames[df_frames["clip_filename"] == clip]
        if g.empty:
            continue
        npz_rel = g["mfcc_npz_path"].iloc[0]
        if clip not in cache:
            try:
                cache[clip] = load_clip_mfcc(npz_rel)
            except Exception:
                continue
        mf = cache[clip]  # (n_frames, D)
        idx = g["mfcc_idx"].astype(int).to_numpy()
        idx = idx[(idx >= 0) & (idx < mf.shape[0])]
        if idx.size == 0:
            continue
        parts.append(mf[idx, :])
    if not parts:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(parts).astype(np.float32, copy=False)

def main():
    parser = argparse.ArgumentParser(description="Estimate subspace stability via bootstrap on TRAIN split.")
    parser.add_argument("--data_dir", default=DATA_DIR, type=str, help="Base data directory (default: Data)")
    parser.add_argument("--B", default=100, type=int, help="Number of bootstraps per class (default: 100)")
    parser.add_argument("--p", default=0.70, type=float, help="Fraction of TRAIN clips sampled per bootstrap (default: 0.70)")
    parser.add_argument("--evr", default=EVR_THRESHOLD, type=float, help="Explained-variance threshold (default: 0.90)")
    parser.add_argument("--max_rank", default=MAX_RANK, type=int, help="Max PCA rank (default: 20)")
    parser.add_argument("--plots", action="store_true", help="If set, write box/violin plots.")
    args = parser.parse_args()

    base = args.data_dir
    splits_dir = os.path.join(base, SPLITS_DIR)
    train_frames_path = os.path.join(splits_dir, TRAIN_FRAMES_PARQUET)
    out_dir = os.path.join(base, STABILITY_DIR)
    ensure_dir(out_dir)

    # Load TRAIN frame index (one row per frame)
    if not os.path.exists(train_frames_path):
        raise FileNotFoundError(f"Missing {train_frames_path}. Run split_and_pca_per_class.py first.")
    df = pd.read_parquet(train_frames_path)

    # Sanity
    need = {"clip_filename", "engine_configuration", "mfcc_npz_path", "mfcc_idx"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"train_frames.parquet missing columns: {sorted(missing)}")

    # Build per-class TRAIN clip lists
    class_to_clips: Dict[str, List[str]] = {}
    for cls, g in df.groupby("engine_configuration", sort=False):
        class_to_clips[cls] = sorted(g["clip_filename"].unique().tolist())

    # Cache MFCC per clip to avoid repeated I/O
    mfcc_cache: Dict[str, np.ndarray] = {}

    # --- Reference subspaces U_a from ALL TRAIN frames per class
    ref_info: Dict[str, Dict] = {}
    for cls, clips in class_to_clips.items():
        # pool all frames from TRAIN clips (class)
        df_cls = df[df["engine_configuration"] == cls]
        X_ref = pool_frames_for_clips(df_cls, clips, mfcc_cache)
        if X_ref.size == 0 or X_ref.shape[0] < 2:
            ref_info[cls] = {"status": "skipped", "reason": "insufficient frames", "rank": 0}
            continue
        pca_ref, r_ref = fit_pca_and_rank(X_ref, args.evr, args.max_rank)
        U_ref = pca_ref.components_[:r_ref, :].T  # (D, r_ref)
        ref_info[cls] = {
            "status": "ok",
            "rank": int(r_ref),
            "N_frames": int(X_ref.shape[0]),
            "D": int(X_ref.shape[1]),
            "U_ref": U_ref,
            "evr": pca_ref.explained_variance_ratio_,
            "cum_evr": np.cumsum(pca_ref.explained_variance_ratio_),
        }
        # Save reference PCA payload
        safe = canonicalize(cls)
        np.savez_compressed(
            os.path.join(out_dir, f"ref_pca_{safe}.npz"),
            components=pca_ref.components_[:r_ref, :],
            mean=pca_ref.mean_,
            evr=pca_ref.explained_variance_ratio_,
            cum_evr=np.cumsum(pca_ref.explained_variance_ratio_),
            rank=np.array([r_ref], dtype=np.int32),
            N_frames=np.array([X_ref.shape[0]], dtype=np.int32),
            D=np.array([X_ref.shape[1]], dtype=np.int32),
            evr_threshold=np.array([args.evr], dtype=np.float32),
            max_rank=np.array([args.max_rank], dtype=np.int32),
        )

    # --- Bootstraps
    raw_rows: List[Dict] = []
    rng = np.random.default_rng(RANDOM_SEED)

    for cls, clips in class_to_clips.items():
        info = ref_info.get(cls, {})
        if info.get("status") != "ok":
            continue
        U_ref = info["U_ref"]
        r_ref = info["rank"]

        n_clips = len(clips)
        if n_clips == 0:
            continue

        # Pre-filter class frames to speed selection
        df_cls = df[df["engine_configuration"] == cls].copy()

        for b in range(args.B):
            # Sample p fraction of TRAIN clips (at least 1, at most n_clips-0)
            k = max(1, int(round(args.p * n_clips)))
            k = min(k, n_clips)
            sample = rng.choice(clips, size=k, replace=False).tolist()

            # Pool frames for sampled clips
            Xb = pool_frames_for_clips(df_cls, sample, mfcc_cache)
            if Xb.size == 0 or Xb.shape[0] < 2:
                # not enough data; skip this bootstrap
                continue

            # Fit PCA to sampled frames
            try:
                pca_b, r_b_full = fit_pca_and_rank(Xb, args.evr, args.max_rank)
            except Exception:
                continue

            # Use top r_ref components, but clamp to available
            r_used = int(min(r_ref, r_b_full))
            if r_used < 1:
                continue
            U_b = pca_b.components_[:r_used, :].T  # (D, r_used)

            # Align reference to r_used
            U_ref_used = U_ref[:, :r_used]

            # Largest principal angle
            theta_max_deg = largest_principal_angle_deg(U_ref_used, U_b)

            raw_rows.append({
                "engine_configuration": cls,
                "bootstrap": int(b),
                "theta_max_deg": float(theta_max_deg),
                "r_ref": int(r_ref),
                "r_used": int(r_used),
                "n_train_clips": int(n_clips),
                "n_sampled_clips": int(k),
                "n_frames_sampled": int(Xb.shape[0]),
            })

    # Save raw results
    raw_df = pd.DataFrame(raw_rows)
    raw_csv = os.path.join(out_dir, "stability_raw.csv")
    if raw_df.empty:
        raise RuntimeError("No bootstrap angles computed. Check that TRAIN split and MFCCs exist.")
    raw_df.to_csv(raw_csv, index=False)

    # Summaries per class
    def iqr_bounds(x: np.ndarray) -> Tuple[float, float]:
        q25 = float(np.percentile(x, 25))
        q75 = float(np.percentile(x, 75))
        return q25, q75

    sum_rows: List[Dict] = []
    for cls, g in raw_df.groupby("engine_configuration", sort=False):
        thetas = g["theta_max_deg"].to_numpy()
        med = float(np.median(thetas))
        q25, q75 = iqr_bounds(thetas)
        r_ref_vals = ref_info[cls]["rank"] if cls in ref_info else 0
        n_train = int(g["n_train_clips"].iloc[0]) if not g.empty else 0
        sum_rows.append({
            "engine_configuration": cls,
            "median_theta_max_deg": med,
            "iqr25_deg": q25,
            "iqr75_deg": q75,
            "r_ref": int(r_ref_vals),
            "B_effective": int(len(g)),
            "train_clips": n_train,
        })

    summary_df = pd.DataFrame(sum_rows).sort_values("engine_configuration")
    summary_csv = os.path.join(out_dir, "stability_summary.csv")
    summary_df.to_csv(summary_csv, index=False)

    # Optional plots
    if args.plots:
        # Box plot
        plt.figure(figsize=(8, 4.5))
        order = summary_df["engine_configuration"].tolist()
        data = [raw_df[raw_df["engine_configuration"] == cls]["theta_max_deg"].to_numpy() for cls in order]
        plt.boxplot(data, labels=order, showmeans=True)
        plt.ylabel(r"largest principal angle $\theta_{\max}$ (deg)")
        plt.title("Subspace stability (boxplot) — TRAIN bootstraps")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "boxplot_theta_max.png"), dpi=150)
        plt.close()

        # Violin plot
        plt.figure(figsize=(8, 4.5))
        parts = plt.violinplot(data, showmeans=True, showmedians=True)
        plt.xticks(np.arange(1, len(order) + 1), order, rotation=0)
        plt.ylabel(r"largest principal angle $\theta_{\max}$ (deg)")
        plt.title("Subspace stability (violin) — TRAIN bootstraps")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "violin_theta_max.png"), dpi=150)
        plt.close()

    # Console report
    print("=== Subspace stability (TRAIN) complete ===")
    print(f"Raw angles:     {raw_csv}")
    print(f"Summary:        {summary_csv}")
    if args.plots:
        print(f"Plots written to {out_dir}")
    print("\nPer-class summary:")
    for _, row in summary_df.iterrows():
        print(f"  {row['engine_configuration']:16s}  rank={int(row['r_ref']):2d}  "
              f"median={row['median_theta_max_deg']:.2f}°  IQR=[{row['iqr25_deg']:.2f}°, {row['iqr75_deg']:.2f}°]  "
              f"B_eff={int(row['B_effective'])}")

if __name__ == "__main__":
    main()
