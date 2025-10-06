# split_and_pca_per_class.py
# Requirements:
#   pip install pandas numpy scikit-learn matplotlib pyarrow

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ----------------- Config -----------------
DATA_DIR = "Data"
MFCC_INDEX = "mfcc_index.parquet"          # produced by make_mfcc_frames.py
MFCC_DIRNAME = "mfcc"                      # folder with per-clip .npz
SPLITS_DIR = "splits"                      # -> Data/splits
PCA_DIR = "pca"                            # -> Data/pca

TRAIN_RATIO = 0.70
RANDOM_SEED = 42

EVR_THRESHOLD = 0.95
MAX_RANK = 20
# ------------------------------------------

np.random.seed(RANDOM_SEED)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def canonicalize_class(name: str) -> str:
    # safe for filenames
    return re.sub(r"[^A-Za-z0-9\-]+", "_", str(name))

def stratified_clip_split(clip_df: pd.DataFrame, train_ratio: float) -> pd.DataFrame:
    """
    clip_df columns: clip_filename, engine_configuration
    Returns same columns + 'split' in {'train','test'} with per-class 70/30 split.
    Guarantees at least 1 clip in each split when possible.
    """
    rows = []
    rng = np.random.default_rng(RANDOM_SEED)
    for cls, g in clip_df.groupby("engine_configuration", sort=False):
        clips = g["clip_filename"].unique().tolist()
        n = len(clips)
        if n <= 1:
            # too few to split; put lone clip in train
            train_clips = clips
            test_clips = []
        elif n == 2:
            # 1/1 split
            rng.shuffle(clips)
            train_clips, test_clips = [clips[0]], [clips[1]]
        else:
            k_train = int(round(train_ratio * n))
            k_train = max(1, min(k_train, n - 1))  # keep both sides non-empty
            rng.shuffle(clips)
            train_clips = clips[:k_train]
            test_clips = clips[k_train:]
        for c in train_clips:
            rows.append({"clip_filename": c, "engine_configuration": cls, "split": "train"})
        for c in test_clips:
            rows.append({"clip_filename": c, "engine_configuration": cls, "split": "test"})
    return pd.DataFrame(rows)

def load_clip_mfcc(npz_path: str) -> np.ndarray:
    data = np.load(npz_path)
    return data["mfcc"]  # shape: (n_frames, n_coeff)

def pool_frames(fi: pd.DataFrame, split_tag: str) -> Dict[str, np.ndarray]:
    """
    Returns dict[class_label] -> 2D array (N_frames, D) pooled from the requested split.
    Loads each clip .npz once; selects rows by 'mfcc_idx' present in index.
    """
    out: Dict[str, List[np.ndarray]] = {}
    # Preload per-clip MFCC to avoid multiple IOs
    cache: Dict[str, np.ndarray] = {}

    for (clip, cls), g in fi[fi["split"] == split_tag].groupby(["clip_filename", "engine_configuration"]):
        npz_rel = g["mfcc_npz_path"].iloc[0]  # same for all rows of a clip
        npz_path = os.path.join(DATA_DIR, npz_rel)
        if not os.path.exists(npz_path):
            # skip silently; upstream should have created them
            continue
        if clip not in cache:
            try:
                cache[clip] = load_clip_mfcc(npz_path)
            except Exception:
                continue
        mf = cache[clip]  # (n_frames, D)
        idx = g["mfcc_idx"].astype(int).to_numpy()
        idx = idx[(idx >= 0) & (idx < mf.shape[0])]
        if idx.size == 0:
            continue
        X = mf[idx, :]
        out.setdefault(cls, []).append(X)

    # Concatenate lists
    pooled: Dict[str, np.ndarray] = {}
    for cls, parts in out.items():
        pooled[cls] = np.vstack(parts) if len(parts) > 1 else parts[0]
    return pooled

def fit_pca_per_class(pooled: Dict[str, np.ndarray],
                      evr_threshold: float,
                      max_rank: int) -> List[Dict]:
    """
    For each class matrix X (N, D), fit PCA with up to max_rank components.
    Choose r = min(first k s.t. cumulative EVR>=threshold, max_rank).
    Returns list of dicts with PCA payload + metadata.
    """
    results = []
    for cls, X in pooled.items():
        N, D = X.shape
        if N < 2 or D < 1:
            results.append({
                "engine_configuration": cls,
                "status": "skipped",
                "reason": f"Insufficient data (N={N}, D={D})"
            })
            continue

        n_comp = min(max_rank, D, N)
        pca = PCA(n_components=n_comp, svd_solver="full", random_state=RANDOM_SEED)
        pca.fit(X)  # centers by default

        evr = pca.explained_variance_ratio_  # (n_comp,)
        cum = np.cumsum(evr)
        # first index where cum >= threshold
        r = int(np.argmax(cum >= evr_threshold) + 1) if np.any(cum >= evr_threshold) else n_comp
        r = max(1, min(r, n_comp, max_rank))

        results.append({
            "engine_configuration": cls,
            "status": "ok",
            "N_frames": int(N),
            "D_features": int(D),
            "rank": int(r),
            "components": pca.components_[:r, :],            # (r, D)
            "mean": pca.mean_.copy(),                        # (D,)
            "evr": evr,                                      # full EVR curve (len = n_comp)
            "cum_evr": cum,                                  # cumulative EVR
        })
    return results

def save_pca_outputs(results: List[Dict], out_dir: str):
    ensure_dir(out_dir)
    summary_rows = []
    for res in results:
        cls = res["engine_configuration"]
        safe = canonicalize_class(cls)
        if res.get("status") != "ok":
            summary_rows.append({
                "engine_configuration": cls,
                "status": res.get("status"),
                "reason": res.get("reason", ""),
                "N_frames": res.get("N_frames", 0),
                "D_features": res.get("D_features", 0),
                "rank": 0,
                "cum_evr_at_20": np.nan,
            })
            continue

        npz_path = os.path.join(out_dir, f"{safe}_pca.npz")
        np.savez_compressed(
            npz_path,
            components=res["components"],
            mean=res["mean"],
            evr=res["evr"],
            cum_evr=res["cum_evr"],
            rank=np.array([res["rank"]], dtype=np.int32),
            N_frames=np.array([res["N_frames"]], dtype=np.int32),
            D_features=np.array([res["D_features"]], dtype=np.int32),
            evr_threshold=np.array([EVR_THRESHOLD], dtype=np.float32),
            max_rank=np.array([MAX_RANK], dtype=np.int32),
        )

        # scree plot
        fig = plt.figure(figsize=(6, 4))
        k = len(res["evr"])
        xs = np.arange(1, k + 1)
        # bars for EVR
        plt.bar(xs, res["evr"])
        # cumulative line
        plt.plot(xs, res["cum_evr"], marker="o")
        # threshold line
        plt.axhline(EVR_THRESHOLD, linestyle="--")
        # selected rank marker
        plt.axvline(res["rank"], linestyle="--")
        plt.title(f"Scree — {cls} (rank={res['rank']}, N={res['N_frames']})")
        plt.xlabel("Component")
        plt.ylabel("Explained variance ratio")
        plt.tight_layout()
        fig_path = os.path.join(out_dir, f"scree_{safe}.png")
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)

        summary_rows.append({
            "engine_configuration": cls,
            "status": "ok",
            "N_frames": res["N_frames"],
            "D_features": res["D_features"],
            "rank": res["rank"],
            "cum_evr_at_20": float(res["cum_evr"][min(len(res["cum_evr"]), MAX_RANK) - 1]),
            "pca_npz": os.path.relpath(npz_path, DATA_DIR),
            "scree_png": os.path.relpath(fig_path, DATA_DIR),
        })

    # write summary
    pd.DataFrame(summary_rows).sort_values("engine_configuration").to_csv(
        os.path.join(out_dir, "pca_summary.csv"), index=False
    )

def main():
    # ---- setup
    mfcc_index_path = os.path.join(DATA_DIR, MFCC_INDEX)
    splits_dir = os.path.join(DATA_DIR, SPLITS_DIR)
    pca_dir = os.path.join(DATA_DIR, PCA_DIR)
    ensure_dir(splits_dir)
    ensure_dir(pca_dir)

    # ---- load frame index (one row per selected frame)
    fi = pd.read_parquet(mfcc_index_path)
    # required columns
    need = {"clip_filename", "engine_configuration", "mfcc_npz_path", "mfcc_idx"}
    missing = need - set(fi.columns)
    if missing:
        raise KeyError(f"{MFCC_INDEX} missing columns: {sorted(missing)}")

    # ---- build clip list (one row per clip)
    clips = (
        fi[["clip_filename", "engine_configuration"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # ---- per-class 70/30 split at clip level
    clip_split = stratified_clip_split(clips, TRAIN_RATIO)
    clip_split_path = os.path.join(splits_dir, "clip_split.parquet")
    clip_split.to_parquet(clip_split_path, index=False)

    # ---- attach split back to frame index
    fi = fi.merge(clip_split, on=["clip_filename", "engine_configuration"], how="left")

    # ---- save frame-level split manifests (handy for training later)
    train_frames = fi[fi["split"] == "train"].reset_index(drop=True)
    test_frames = fi[fi["split"] == "test"].reset_index(drop=True)
    train_frames.to_parquet(os.path.join(splits_dir, "train_frames.parquet"), index=False)
    test_frames.to_parquet(os.path.join(splits_dir, "test_frames.parquet"), index=False)

    # ---- pool training frames per class
    pooled = pool_frames(fi, split_tag="train")

    # ---- PCA per class
    results = fit_pca_per_class(pooled, EVR_THRESHOLD, MAX_RANK)

    # ---- save PCA payloads + scree
    save_pca_outputs(results, pca_dir)

    # ---- small console report
    print("=== Split & PCA complete ===")
    print(f"Clip split:        {clip_split_path}")
    print(f"Train frames:      {os.path.join(splits_dir, 'train_frames.parquet')}")
    print(f"Test frames:       {os.path.join(splits_dir, 'test_frames.parquet')}")
    print(f"PCA dir:           {pca_dir}")
    for r in results:
        cls = r["engine_configuration"]
        if r.get("status") == "ok":
            print(f"  {cls:16s} -> rank={r['rank']}, N={r['N_frames']}, D={r['D_features']}")
        else:
            print(f"  {cls:16s} -> SKIPPED ({r.get('reason','')})")

if __name__ == "__main__":
    main()
