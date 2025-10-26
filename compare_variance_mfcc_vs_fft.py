#!/usr/bin/env python3
import os
import glob
import math
import argparse
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def power_of_two_ge(n: int) -> int:
    return 1 << (n - 1).bit_length()


def stft_mag(y: np.ndarray, sr: int, win_ms: float, hop_ms: float) -> np.ndarray:
    """Return |STFT| matrix with shape (freq, frames)."""
    win_length = int(round(win_ms * 1e-3 * sr))
    hop_length = int(round(hop_ms * 1e-3 * sr))
    n_fft = power_of_two_ge(win_length)
    window = np.hanning(win_length)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=True)
    mag = np.abs(S).astype(np.float32)
    return mag  # (freq, frames)


def extract_fft_mag(y: np.ndarray, sr: int, win_ms: float, hop_ms: float) -> np.ndarray:
    """Return frames x D_fft of linear magnitude spectrum (no log)."""
    mag = stft_mag(y, sr, win_ms, hop_ms)
    return mag.T  # (frames, D_fft)


def extract_fft_logmag(y: np.ndarray, sr: int, win_ms: float, hop_ms: float) -> np.ndarray:
    """Return frames x D_fft of log(1+|STFT|)."""
    mag = stft_mag(y, sr, win_ms, hop_ms)
    return np.log1p(mag).T  # (frames, D_fft)


def extract_mfcc(y: np.ndarray, sr: int, win_ms: float, hop_ms: float,
                 n_mels: int, n_mfcc: int, fmin: float, fmax: float, include_deltas: bool) -> np.ndarray:
    """Return frames x D_mfcc (20 or 60 with deltas)."""
    win_length = int(round(win_ms * 1e-3 * sr))
    hop_length = int(round(hop_ms * 1e-3 * sr))
    n_fft = power_of_two_ge(win_length)
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels, fmin=fmin, fmax=fmax,
        n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann', center=True
    ).T  # (frames, n_mfcc)
    if include_deltas:
        d1 = librosa.feature.delta(mfcc.T, order=1, mode='nearest').T
        d2 = librosa.feature.delta(mfcc.T, order=2, mode='nearest').T
        X = np.concatenate([mfcc, d1, d2], axis=1).astype(np.float32)
    else:
        X = mfcc.astype(np.float32)
    return X


def pca_evr(X: np.ndarray, max_components: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return (evr, cum_evr) for centered X."""
    if X.size == 0 or X.shape[0] < 2:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    Xc = X - X.mean(axis=0, keepdims=True)
    n_comp = min(max_components or X.shape[1], Xc.shape[0] - 1, Xc.shape[1])
    if n_comp < 1:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    p = PCA(n_components=n_comp, svd_solver="full", random_state=0).fit(Xc)
    evr = p.explained_variance_ratio_.astype(np.float32)
    cum = np.cumsum(evr)
    return evr, cum


def comps_for_threshold(cum: np.ndarray, thr: float) -> int:
    if cum.size == 0:
        return 0
    idx = np.searchsorted(cum, thr, side="left")
    return int(idx + 1) if idx < len(cum) else len(cum)


def summarize_feature_variances(X: np.ndarray, name: str) -> pd.DataFrame:
    if X.size == 0:
        return pd.DataFrame(columns=["rep", "feature_index", "var"])
    v = np.var(X, axis=0, ddof=1)
    df = pd.DataFrame({"rep": name, "feature_index": np.arange(len(v)), "var": v})
    return df


def main():
    ap = argparse.ArgumentParser(description="Compare variance/PCA between MFCC and FFT features (mag & logmag).")
    ap.add_argument("--audio_dir", type=str, required=True, help="Directory of WAV files (searched recursively).")
    ap.add_argument("--pattern", type=str, default="**/*.wav", help="Glob pattern under audio_dir.")
    ap.add_argument("--sr", type=int, default=22050, help="Target sampling rate (audio will be resampled).")
    ap.add_argument("--win_ms", type=float, default=25.0, help="Frame window length in ms.")
    ap.add_argument("--hop_ms", type=float, default=10.0, help="Frame hop in ms.")
    ap.add_argument("--n_mels", type=int, default=64, help="Mel bins for MFCC.")
    ap.add_argument("--n_mfcc", type=int, default=20, help="Number of MFCC coefficients.")
    ap.add_argument("--fmin", type=float, default=20.0, help="Min frequency for mel/MFCC.")
    ap.add_argument("--fmax_frac_nyq", type=float, default=1.0, help="fmax as a fraction of Nyquist (0-1].")
    ap.add_argument("--include_deltas", action="store_true", help="If set, stack Δ and ΔΔ (60-D total).")
    ap.add_argument("--max_seconds", type=float, default=10.0, help="Cap per-file duration to this many seconds.")
    ap.add_argument("--max_frames_per_file", type=int, default=4000, help="Randomly sample at most this many frames per file.")
    ap.add_argument("--limit_files", type=int, default=0, help="If >0, only use the first N files after globbing.")
    ap.add_argument("--out_dir", type=str, default="results", help="Output directory for tables/figures.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fmax = (args.sr * 0.5) * float(args.fmax_frac_nyq)

    # Collect files
    all_files = sorted(glob.glob(os.path.join(args.audio_dir, args.pattern), recursive=True))
    if args.limit_files and args.limit_files > 0:
        all_files = all_files[: args.limit_files]
    if not all_files:
        raise SystemExit("No WAV files found. Check --audio_dir and --pattern.")

    rng = np.random.default_rng(0)
    X_fft_log_list: List[np.ndarray] = []
    X_fft_lin_list: List[np.ndarray] = []
    X_mfcc_list: List[np.ndarray] = []

    for i, fp in enumerate(all_files):
        try:
            y, sr = sf.read(fp, always_2d=False)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            if sr != args.sr:
                y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=args.sr)
                sr = args.sr
            if args.max_seconds is not None and args.max_seconds > 0:
                y = y[: int(args.max_seconds * sr)]
            y = y.astype(np.float32)

            X_fft_lin = extract_fft_mag(y, sr, args.win_ms, args.hop_ms)
            X_fft_log = extract_fft_logmag(y, sr, args.win_ms, args.hop_ms)
            X_mfcc = extract_mfcc(y, sr, args.win_ms, args.hop_ms,
                                  n_mels=args.n_mels, n_mfcc=args.n_mfcc,
                                  fmin=args.fmin, fmax=fmax,
                                  include_deltas=args.include_deltas)

            # Randomly downsample frames per file to control memory
            def sample_frames(X: np.ndarray) -> np.ndarray:
                if X.shape[0] <= args.max_frames_per_file:
                    return X
                idx = rng.choice(X.shape[0], size=args.max_frames_per_file, replace=False)
                return X[idx]

            if X_fft_lin.size:
                X_fft_lin_list.append(sample_frames(X_fft_lin))
            if X_fft_log.size:
                X_fft_log_list.append(sample_frames(X_fft_log))
            if X_mfcc.size:
                X_mfcc_list.append(sample_frames(X_mfcc))
        except Exception as e:
            print(f"[WARN] Skipping {fp}: {e}")

    if not X_fft_lin_list or not X_fft_log_list or not X_mfcc_list:
        raise SystemExit("Insufficient features extracted.")

    X_fft_lin_all = np.vstack(X_fft_lin_list).astype(np.float32)
    X_fft_log_all = np.vstack(X_fft_log_list).astype(np.float32)
    X_mfcc_all = np.vstack(X_mfcc_list).astype(np.float32)

    # Raw feature-wise variance (before PCA)
    var_mfcc = summarize_feature_variances(X_mfcc_all, "MFCC" + ("+Δ+ΔΔ" if args.include_deltas else ""))
    var_fft_log = summarize_feature_variances(X_fft_log_all, "FFT_logmag")
    var_fft_lin = summarize_feature_variances(X_fft_lin_all, "FFT_mag")
    var_df = pd.concat([var_mfcc, var_fft_log, var_fft_lin], axis=0, ignore_index=True)
    var_df.to_csv(os.path.join(args.out_dir, "feature_var_summary.csv"), index=False)

    # PCA EVR
    evr_mfcc, cum_mfcc = pca_evr(X_mfcc_all)
    evr_fft_log, cum_fft_log = pca_evr(X_fft_log_all)
    evr_fft_lin, cum_fft_lin = pca_evr(X_fft_lin_all)

    # Metrics table
    targets = [0.90, 0.95, 0.99]
    rows = []
    name_mfcc = "MFCC" + ("+Δ+ΔΔ" if args.include_deltas else "")
    rows.append({
        "rep": name_mfcc,
        "D": X_mfcc_all.shape[1],
        "EVR_at_5": float(cum_mfcc[min(4, len(cum_mfcc)-1)]) if len(cum_mfcc) else np.nan,
        "EVR_at_10": float(cum_mfcc[min(9, len(cum_mfcc)-1)]) if len(cum_mfcc) else np.nan,
        "k_for_90": comps_for_threshold(cum_mfcc, targets[0]),
        "k_for_95": comps_for_threshold(cum_mfcc, targets[1]),
        "k_for_99": comps_for_threshold(cum_mfcc, targets[2]),
    })
    rows.append({
        "rep": "FFT_logmag",
        "D": X_fft_log_all.shape[1],
        "EVR_at_5": float(cum_fft_log[min(4, len(cum_fft_log)-1)]) if len(cum_fft_log) else np.nan,
        "EVR_at_10": float(cum_fft_log[min(9, len(cum_fft_log)-1)]) if len(cum_fft_log) else np.nan,
        "k_for_90": comps_for_threshold(cum_fft_log, targets[0]),
        "k_for_95": comps_for_threshold(cum_fft_log, targets[1]),
        "k_for_99": comps_for_threshold(cum_fft_log, targets[2]),
    })
    rows.append({
        "rep": "FFT_mag",
        "D": X_fft_lin_all.shape[1],
        "EVR_at_5": float(cum_fft_lin[min(4, len(cum_fft_lin)-1)]) if len(cum_fft_lin) else np.nan,
        "EVR_at_10": float(cum_fft_lin[min(9, len(cum_fft_lin)-1)]) if len(cum_fft_lin) else np.nan,
        "k_for_90": comps_for_threshold(cum_fft_lin, targets[0]),
        "k_for_95": comps_for_threshold(cum_fft_lin, targets[1]),
        "k_for_99": comps_for_threshold(cum_fft_lin, targets[2]),
    })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(args.out_dir, "variance_comparison.csv"), index=False)

    # Scree plot: overlay cumulative EVR
    plt.figure(figsize=(8, 5))
    if len(cum_mfcc):
        xs_m = np.arange(1, len(cum_mfcc) + 1)
        plt.plot(xs_m, cum_mfcc, marker='o', label=name_mfcc)
    if len(cum_fft_log):
        xs_flog = np.arange(1, len(cum_fft_log) + 1)
        plt.plot(xs_flog, cum_fft_log, marker='o', linestyle='--', label='FFT_logmag')
    if len(cum_fft_lin):
        xs_flin = np.arange(1, len(cum_fft_lin) + 1)
        plt.plot(xs_flin, cum_fft_lin, marker='o', linestyle='-.', label='FFT_mag')
    plt.axhline(0.90, linestyle=':', label='90%')
    plt.axhline(0.95, linestyle=':')
    plt.axhline(0.99, linestyle=':')
    plt.xlabel("Components")
    plt.ylabel("Cumulative explained variance")
    plt.title("Cumulative EVR: MFCC vs FFT (mag & logmag)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "scree_mfcc_vs_fft.png"), dpi=160)
    plt.close()

    # Console summary
    print("=== Variance / PCA comparison ===")
    print(summary.to_string(index=False))
    print(f"Frames pooled — MFCC: {X_mfcc_all.shape}, FFT_logmag: {X_fft_log_all.shape}, FFT_mag: {X_fft_lin_all.shape}")
    print(f"Outputs written to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
