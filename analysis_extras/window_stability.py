#!/usr/bin/env python3
import os
import argparse
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
import librosa
from sklearn.decomposition import PCA


def logmel_stats(y: np.ndarray, sr: int = 22050, n_fft: int = 2048, hop: int = 512, n_mels: int = 64) -> np.ndarray:
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))**2
    mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=n_mels)
    logmel = librosa.power_to_db(mel + 1e-12)
    return np.concatenate([
        logmel.mean(axis=1), logmel.std(axis=1),
        np.percentile(logmel, 10, axis=1), np.percentile(logmel, 90, axis=1)
    ])


def window_features(path: str, window_sec: float, hop_sec: float, sr: int = 22050) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    win = int(window_sec * sr)
    hop = int(hop_sec * sr)
    if len(y) < win:
        return np.empty((0, 4 * 64), dtype=float)
    feats = []
    for s in range(0, len(y) - win + 1, hop):
        seg = y[s:s + win]
        feats.append(logmel_stats(seg, sr=sr))
    return np.array(feats)


def subspace_from_windows(F: np.ndarray, d: int) -> np.ndarray:
    if F.shape[0] < 2:
        return np.zeros((F.shape[1], 0))
    p = PCA(svd_solver="full").fit(F - F.mean(axis=0))
    d_eff = int(min(d, p.components_.shape[0]))
    return p.components_[:d_eff].T


def subspace_affinity(U: np.ndarray, V: np.ndarray) -> float:
    from numpy.linalg import svd
    if U.size == 0 or V.size == 0:
        return 0.0
    s = svd(U.T @ V, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return float(np.mean(s**2))


def main():
    ap = argparse.ArgumentParser(description="Within-clip window-level stability vs between-clip baseline")
    ap.add_argument("--outdir", default="analysis_outputs3")
    ap.add_argument("--audio-dir", default="Download/engine_downloads")
    ap.add_argument("--parquet", default=None, help="Optional explicit path to features parquet; defaults to OUTDIR/clip_features.parquet")
    ap.add_argument("--label-col", default="turbo_supercharged")
    ap.add_argument("--n-per-class", type=int, default=30)
    ap.add_argument("--window-sec", type=float, default=2.5)
    ap.add_argument("--hop-sec", type=float, default=1.25)
    ap.add_argument("--subspace-d", type=int, default=5)
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    feat_path = args.parquet or os.path.join(args.outdir, "clip_features.parquet")
    df = pd.read_parquet(feat_path)

    if args.label_col not in df.columns:
        raise ValueError(f"Label column '{args.label_col}' not found in {feat_path}")

    # Construct full audio path
    def to_path(row) -> str:
        if "path" in row and isinstance(row["path"], str) and os.path.exists(row["path"]):
            return row["path"]
        fname = row.get("filename") or row.get("id") or ""
        if not str(fname).lower().endswith(".wav"):
            fname = f"{fname}.wav"
        return os.path.join(args.audio_dir, str(fname))

    df["_path"] = df.apply(to_path, axis=1)
    df = df[(df["_path"].apply(os.path.exists)) & (df[args.label_col] != "none")]

    # Sample clips per class
    samples = []
    for c, g in df.groupby(args.label_col):
        g = g.sample(n=min(args.n_per_class, len(g)), random_state=args.seed) if len(g) > args.n_per_class else g
        for _, r in g.iterrows():
            samples.append((str(c), r["_path"]))

    # Compute within-clip stability: split windows into two halves, compare subspaces
    within_scores = []
    by_class_within = {}
    for c, path in samples:
        F = window_features(path, args.window_sec, args.hop_sec, sr=args.sr)
        if len(F) < 4:
            continue
        mid = len(F) // 2
        U1 = subspace_from_windows(F[:mid], args.subspace_d)
        U2 = subspace_from_windows(F[mid:], args.subspace_d)
        s = subspace_affinity(U1, U2)
        within_scores.append((c, s))
        by_class_within.setdefault(c, []).append(s)

    # Between-clip baseline: pair random clips of the same class
    between_scores = []
    by_class_between = {}
    by_class_paths = {}
    for c, path in samples:
        by_class_paths.setdefault(c, []).append(path)
    for c, paths in by_class_paths.items():
        if len(paths) < 2:
            continue
        k = min(len(paths) // 2, 20)
        idx = rng.choice(len(paths), size=2 * k, replace=False)
        for i in range(k):
            p1, p2 = paths[idx[2 * i]], paths[idx[2 * i + 1]]
            F1 = window_features(p1, args.window_sec, args.hop_sec, sr=args.sr)
            F2 = window_features(p2, args.window_sec, args.hop_sec, sr=args.sr)
            U1 = subspace_from_windows(F1, args.subspace_d)
            U2 = subspace_from_windows(F2, args.subspace_d)
            s = subspace_affinity(U1, U2)
            between_scores.append((c, s))
            by_class_between.setdefault(c, []).append(s)

    # Summaries
    def summarize(d: dict) -> str:
        lines = []
        for c, vals in d.items():
            if len(vals):
                lines.append(f"{c}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}, n={len(vals)}")
            else:
                lines.append(f"{c}: insufficient data")
        return "\n".join(lines)

    out_path = os.path.join(args.outdir, "window_stability_summary.txt")
    with open(out_path, "w") as f:
        f.write("== Within-clip stability (subspace affinity) ==\n")
        f.write(summarize(by_class_within) + "\n\n")
        f.write("== Between-clip baseline (same-class pairs) ==\n")
        f.write(summarize(by_class_between) + "\n")

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

