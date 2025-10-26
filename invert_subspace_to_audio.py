# python invert_subspace_to_audio.py \
#   --attribute engine_configuration \
#   --metadata Data/metadata.parquet \
#   --mode npz --npz_dir Data/mfcc --npz_key mfcc \
#   --min_clips 30 --results_root Results/inversion \
#   --top_r 5 --k_sigma 2.0 --t_seconds 2.0 --gl_iters 64 \
#   --n_mfcc 20 --n_mels 64 --sr 22050 --n_fft 2048 --hop_length 512 \
#   --overlay_curve pca_filtered_mean --normalize refhz --ref_hz 200

# invert_subspace_to_audio.py
# Requirements:
#   pip install numpy pandas librosa soundfile scikit-learn matplotlib pyarrow tqdm

import os
import re
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
from tqdm import tqdm

from sklearn.decomposition import PCA

import librosa
import librosa.display
from librosa.feature import inverse as lbinv

# ---------------------- Defaults ----------------------
RANDOM_SEED = 42

# Feature layout defaults (must match your MFCC computation)
D_TOTAL = 60          # 20 + 20Δ + 20ΔΔ
N_MFCC = 20           # static MFCC count at front of the 60-D vector
N_MELS = 64           # mel bins used when features were computed
FMIN = 20.0
FMAX_FRAC_OF_NYQ = 1.0

# Inversion / audio defaults
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = None
GL_ITERS = 48
T_SECONDS = 2.0
TOP_R = 5
K_SIGMA = 2.0

# Practical cap for PCA-filtered average (compute control)
PCA_FILTERED_FRAMES_LIMIT = 20000

# Cross-class overlay defaults
OVERLAY_CURVE = "pca_filtered_mean"  # or: model_mean, data_mean
NORMALIZE = "refhz"                   # none | refhz | l2 | area
REF_HZ = 200.0
BANDS = [(50.0, 200.0), (200.0, 800.0), (800.0, 3000.0)]  # low/mid/high bands

# ---------------------- Utils ----------------------
np.random.seed(RANDOM_SEED)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def canonicalize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9\-]+", "_", str(name))

def info(msg: str):
    print(f"[invert] {msg}")

# ---------------------- FrameSource ----------------------
class FrameSource:
    """
    Loader for per-clip frame features:
      - mode 'npz': one npz per clip in a directory; keys tried: npz_key, 'mfcc', 'X'
      - mode 'parquet': one big parquet with f0..f59 and 'filename'
    """
    def __init__(self, mode: str, npz_dir: Optional[str], npz_key: Optional[str],
                 frames_parquet: Optional[str], D: int):
        self.mode = mode
        self.npz_dir = npz_dir
        self.npz_key = npz_key
        self.D = D
        self.frames_df: Optional[pd.DataFrame] = None
        if mode == "parquet":
            if frames_parquet is None:
                raise ValueError("frames_parquet must be provided for mode='parquet'")
            self.frames_df = pd.read_parquet(frames_parquet)
            needed = {"filename"} | {f"f{i}" for i in range(D)}
            miss = needed - set(self.frames_df.columns)
            if miss:
                raise KeyError(f"frames parquet missing columns: {sorted(miss)}")
        elif mode == "npz":
            if npz_dir is None:
                raise ValueError("npz_dir must be provided for mode='npz'")
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
                self.cache[filename] = np.empty((0, self.D), dtype=np.float32)
                return self.cache[filename]
            z = np.load(path)
            key = self.npz_key
            X = None
            if key and key in z:
                X = z[key]
            elif "mfcc" in z:
                X = z["mfcc"]
            elif "X" in z:
                X = z["X"]
            else:
                self.cache[filename] = np.empty((0, self.D), dtype=np.float32)
                return self.cache[filename]
            X = np.asarray(X, dtype=np.float32)
            if X.ndim != 2 or X.shape[1] != self.D:
                self.cache[filename] = np.empty((0, self.D), dtype=np.float32)
                return self.cache[filename]
            self.cache[filename] = X
            return X
        else:
            g = self.frames_df[self.frames_df["filename"] == filename]
            if g.empty:
                return np.empty((0, self.D), dtype=np.float32)
            Fcols = [f"f{i}" for i in range(self.D)]
            X = g[Fcols].to_numpy(dtype=np.float32, copy=False)
            self.cache[filename] = X
            return X

# ---------------------- Inversion helpers ----------------------
def mfcc_frames_to_audio(mfcc_frames: np.ndarray,
                         sr: int = SR,
                         n_mels: int = N_MELS,
                         n_fft: int = N_FFT,
                         hop_length: int = HOP_LENGTH,
                         gl_iters: int = GL_ITERS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """mfcc_frames: (n_mfcc, T) -> (y, mel[power], S_mag)"""
    mel = lbinv.mfcc_to_mel(mfcc_frames, n_mels=n_mels)
    S_mag = lbinv.mel_to_stft(mel, sr=sr, n_fft=n_fft, power=2.0)
    y = librosa.griffinlim(S_mag, n_iter=gl_iters, hop_length=hop_length, win_length=WIN_LENGTH)
    return y, mel, S_mag

def save_spectrum_plot(freqs: np.ndarray, spec: np.ndarray, out_path: str, title: str):
    plt.figure(figsize=(7,3))
    plt.plot(freqs, spec)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (a.u.)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_mel(mel: np.ndarray, sr: int, hop_length: int, out_path: str, fmin: float, fmax: float):
    plt.figure(figsize=(7, 3))
    librosa.display.specshow(librosa.power_to_db(mel, ref=np.max),
                             sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='mel', fmin=fmin, fmax=fmax)
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel spectrogram (dB)")
    plt.xlabel("Time (s)")
    plt.ylabel("Mel frequency (mel)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_spectrum(S_mag: np.ndarray, sr: int, out_path: str, title: str = "Median magnitude spectrum"):
    """Plot median-over-time spectrum; return (spec, freqs, centroid_hz, peak_hz)."""
    spec = np.median(S_mag, axis=1)
    n_fft_est = (S_mag.shape[0] - 1) * 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft_est)
    plt.figure(figsize=(7,3))
    plt.plot(freqs, spec)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (a.u.)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    centroid = float(librosa.feature.spectral_centroid(S=S_mag, sr=sr).mean())
    peak_idx = int(np.argmax(spec))
    peak_hz = float(freqs[peak_idx])
    return spec, freqs, centroid, peak_hz

def tile_static_mfcc(mu: np.ndarray, t_seconds: float, hop_length: int, sr: int) -> np.ndarray:
    T = max(1, int(np.ceil(t_seconds * sr / hop_length)))
    return np.repeat(mu[:, None], T, axis=1)

# ----- Data mean (empirical, no PCA) -----
def accumulate_mean_mel_from_mfcc_frames(clip_list: List[str], fs: "FrameSource",
                                         n_mfcc: int, n_mels: int) -> np.ndarray:
    """Return mean mel power vector (n_mels,)."""
    sum_mel = np.zeros((n_mels,), dtype=np.float64)
    total_T = 0
    for fname in clip_list:
        X = fs.load_clip(fname)
        if X.size == 0:
            continue
        M = X[:, :n_mfcc]
        mel = lbinv.mfcc_to_mel(M.T, n_mels=n_mels)  # (n_mels, T)
        sum_mel += mel.sum(axis=1)
        total_T += mel.shape[1]
    if total_T == 0:
        return np.zeros((n_mels,), dtype=np.float32)
    return (sum_mel / float(total_T)).astype(np.float32)

# ----- PCA-filtered mean (project frame -> invert -> average) -----
def pca_filtered_mean_spectrum(clip_list: List[str], fs: "FrameSource",
                               mu: np.ndarray, U: np.ndarray,
                               n_mfcc: int, n_mels: int, sr: int, n_fft: int,
                               frames_limit: int) -> np.ndarray:
    P = U @ U.T  # projection matrix
    F = 1 + n_fft // 2
    sum_spec = np.zeros((F,), dtype=np.float64)
    total_T = 0
    for fname in clip_list:
        X = fs.load_clip(fname)
        if X.size == 0:
            continue
        M = X[:, :n_mfcc]
        # optional compute cap
        if frames_limit is not None and total_T + M.shape[0] > frames_limit:
            M = M[:max(0, frames_limit - total_T), :]
        if M.size == 0:
            continue
        Xc = M - mu[None, :]
        M_hat = mu[None, :] + Xc @ P
        mel = lbinv.mfcc_to_mel(M_hat.T, n_mels=n_mels)
        S_mag = lbinv.mel_to_stft(mel, sr=sr, n_fft=n_fft, power=2.0)
        sum_spec += S_mag.sum(axis=1)
        total_T += S_mag.shape[1]
        if frames_limit is not None and total_T >= frames_limit:
            break
    if total_T == 0:
        return np.zeros((F,), dtype=np.float32)
    return (sum_spec / float(total_T)).astype(np.float32)

# ---------- Overlay helpers ----------
def plot_overlay_spectra(freqs: np.ndarray, spec_a: np.ndarray, spec_b: np.ndarray,
                         labels: Tuple[str, str], out_path: str,
                         title: str = "Mean-of-spectra vs spectrum-of-mean"):
    plt.figure(figsize=(7,3))
    plt.plot(freqs, spec_a, label=labels[0])
    plt.plot(freqs, spec_b, label=labels[1])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (a.u.)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_inversion_bundle_overlay(freqs: np.ndarray, curves: Dict[str, np.ndarray],
                                  out_path: str, title: str):
    plt.figure(figsize=(8,4))
    for name, spec in curves.items():
        plt.plot(freqs, spec, label=name)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (a.u.)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ---------------------- Main per-class routine ----------------------
def process_class(class_name: str,
                  clip_list: List[str],
                  fs: FrameSource,
                  out_dir: str,
                  n_mfcc: int = N_MFCC,
                  top_r: int = TOP_R,
                  k_sigma: float = K_SIGMA,
                  sr: int = SR,
                  n_mels: int = N_MELS,
                  n_fft: int = N_FFT,
                  hop_length: int = HOP_LENGTH,
                  gl_iters: int = GL_ITERS,
                  t_seconds: float = T_SECONDS,
                  fmin: float = FMIN,
                  fmax_frac_of_nyq: float = FMAX_FRAC_OF_NYQ,
                  pca_filtered_frames_limit: int = PCA_FILTERED_FRAMES_LIMIT):
    ensure_dir(out_dir)

    # Pool frames (only static MFCCs: first n_mfcc of the 60-D vector)
    parts = []
    for fname in clip_list:
        X = fs.load_clip(fname)
        if X.size == 0:
            continue
        parts.append(X[:, :n_mfcc])
    if not parts:
        info(f"{class_name}: no frames found")
        return

    Xall = np.vstack(parts).astype(np.float32)
    mu = Xall.mean(axis=0)
    Xc = Xall - mu[None, :]

    # Fit PCA
    r = min(top_r, n_mfcc, Xc.shape[0] - 1) if Xc.shape[0] >= 2 else 1
    pca = PCA(n_components=r, svd_solver="full", random_state=RANDOM_SEED).fit(Xc)
    U = pca.components_.T.astype(np.float32)
    lambdas = pca.explained_variance_.astype(np.float32)

    # Save PCA summary
    with open(os.path.join(out_dir, "pca_summary.json"), "w") as f:
        json.dump({
            "r": int(r),
            "explained_variance": [float(x) for x in lambdas],
            "explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        }, f, indent=2)

    # ---------- MODEL INVERSION: class mean ----------
    mfcc_mean_frames = tile_static_mfcc(mu, t_seconds, hop_length, sr)
    y_mean, mel_mean, S_mean = mfcc_frames_to_audio(mfcc_mean_frames, sr=sr,
                                                    n_mels=n_mels, n_fft=n_fft,
                                                    hop_length=hop_length, gl_iters=gl_iters)
    fmax = (sr * 0.5) * fmax_frac_of_nyq
    plot_mel(mel_mean, sr, hop_length, os.path.join(out_dir, "mean_mel.png"), fmin=fmin, fmax=fmax)
    spec_inv, freqs, centroid_hz, peak_hz = plot_spectrum(
        S_mean, sr, os.path.join(out_dir, "mean_spectrum_hz.png"),
        title="Model inversion: median spectrum of mean-MFCC audio"
    )
    sf.write(os.path.join(out_dir, "mean_envelope.wav"), y_mean, sr)

    # ---------- DATA MEAN (no PCA) ----------
    mean_mel = accumulate_mean_mel_from_mfcc_frames(clip_list, fs, n_mfcc=n_mfcc, n_mels=n_mels)
    S_emp = lbinv.mel_to_stft(mean_mel[:, None], sr=sr, n_fft=n_fft, power=2.0)
    spec_emp = S_emp[:, 0]
    save_spectrum_plot(freqs, spec_emp,
                       os.path.join(out_dir, "data_mean_spectrum_hz.png"),
                       title="Data average: mean spectrum per class (mel-power mean → STFT)")

    # ---------- PCA-FILTERED MEAN ----------
    spec_pca = pca_filtered_mean_spectrum(
        clip_list, fs, mu=mu, U=U, n_mfcc=n_mfcc, n_mels=n_mels,
        sr=sr, n_fft=n_fft, frames_limit=pca_filtered_frames_limit
    )
    save_spectrum_plot(freqs, spec_pca,
                       os.path.join(out_dir, "pca_filtered_mean_spectrum_hz.png"),
                       title="PCA-filtered mean: project→invert→average (uses r PCs)")

    # ---------- OVERLAYS (per-class) ----------
    plot_overlay_spectra(freqs, spec_emp, spec_inv,
                         labels=("Data mean of spectra", "Model spectrum of mean MFCC"),
                         out_path=os.path.join(out_dir, "compare_mean_vs_inversion.png"),
                         title="Mean-of-spectra vs Spectrum-of-mean")

    curves = {
        "data_mean": spec_emp.astype(float),
        "model_mean": spec_inv.astype(float),
        "pca_filtered_mean": spec_pca.astype(float)
    }
    pd.DataFrame({"freq_hz": freqs, **{k: v for k, v in curves.items()}}).to_csv(
        os.path.join(out_dir, "inversion_bundle.csv"), index=False
    )
    plot_inversion_bundle_overlay(
        freqs, curves,
        out_path=os.path.join(out_dir, "compare_inversion_bundle.png"),
        title="Data mean vs Model mean vs PCA-filtered mean (+ optional ±PCs)"
    )

    # ---------- PC excursions (± along first r PCs) ----------
    max_pcs_for_bundle = min(2, r)  # keep figure readable
    for j in range(r):
        u = U[:, j]
        sigma = float(np.sqrt(max(lambdas[j], 0.0)))
        for sign, tag in [(+1.0, "plus"), (-1.0, "minus")]:
            mu_j = mu + (sign * K_SIGMA * sigma) * u
            frames = tile_static_mfcc(mu_j, t_seconds, hop_length, sr)
            y_j, mel_j, S_j = mfcc_frames_to_audio(frames, sr=sr,
                                                   n_mels=n_mels, n_fft=n_fft,
                                                   hop_length=hop_length, gl_iters=gl_iters)
            base = f"pc{j+1}_{tag}"
            plot_mel(mel_j, sr, hop_length, os.path.join(out_dir, f"{base}_mel.png"),
                     fmin=fmin, fmax=fmax)
            plot_spectrum(S_j, sr, os.path.join(out_dir, f"{base}_spectrum_hz.png"),
                          title=f"Model inversion: median spectrum (PC{j+1} {tag})")
            sf.write(os.path.join(out_dir, f"{base}.wav"), y_j, sr)

# ---------------------- Cross-class overlays ----------------------
def _nearest_idx(freqs: np.ndarray, ref_hz: float) -> int:
    return int(np.argmin(np.abs(freqs - ref_hz)))

def _normalize_curve(freqs: np.ndarray, spec: np.ndarray, mode: str, ref_hz: float):
    if mode == "none":
        return spec, 1.0
    if mode == "refhz":
        idx = _nearest_idx(freqs, ref_hz)
        ref = float(spec[idx]) if idx is not None else 1.0
        scale = (ref if ref > 0 else 1.0)
        return spec / scale, scale
    if mode == "l2":
        norm = float(np.sqrt(np.sum(spec**2)))
        return (spec / norm if norm > 0 else spec), (norm if norm > 0 else 1.0)
    if mode == "area":
        area = float(np.trapz(spec, freqs))
        return (spec / area if area > 0 else spec), (area if area > 0 else 1.0)
    # default fallback
    return spec, 1.0

def _band_energy_pct(freqs: np.ndarray, spec: np.ndarray, band: Tuple[float,float]) -> float:
    mask = (freqs >= band[0]) & (freqs < band[1])
    band_area = float(np.trapz(spec[mask], freqs[mask])) if np.any(mask) else 0.0
    total_area = float(np.trapz(spec, freqs)) if np.any(spec) else 0.0
    return (100.0 * band_area / total_area) if total_area > 0 else 0.0

def make_cross_class_overlay(base_dir: str, classes: List[str],
                             curve_key: str = OVERLAY_CURVE,
                             normalize: str = NORMALIZE,
                             ref_hz: float = REF_HZ,
                             bands: List[Tuple[float,float]] = BANDS):
    """
    Read each class's inversion_bundle.csv and overlay the requested curve across classes.
    Saves:
      - cross_class_overlay_<curve_key>.png
      - cross_class_overlay_<curve_key>.csv
      - cross_class_descriptors_<curve_key>.csv
    """
    rows = []
    desc_rows = []
    freqs_ref = None
    for cls in classes:
        cls_dir = os.path.join(base_dir, canonicalize(cls))
        csv_path = os.path.join(cls_dir, "inversion_bundle.csv")
        if not os.path.exists(csv_path):
            info(f"skip {cls}: missing inversion_bundle.csv")
            continue
        df = pd.read_csv(csv_path)
        if curve_key not in df.columns:
            info(f"skip {cls}: missing column '{curve_key}' in {csv_path}")
            continue
        freqs = df["freq_hz"].to_numpy()
        spec = df[curve_key].to_numpy(dtype=float)
        if freqs_ref is None:
            freqs_ref = freqs
        else:
            if len(freqs) != len(freqs_ref) or np.max(np.abs(freqs - freqs_ref)) > 1e-6:
                info(f"skip {cls}: frequency grid mismatch")
                continue
        spec_norm, scale = _normalize_curve(freqs_ref, spec, normalize, ref_hz)
        rows.append((cls, spec_norm))

        # descriptors
        peak_hz = float(freqs_ref[int(np.argmax(spec))])
        centroid_hz = float(np.sum(freqs_ref * spec) / np.sum(spec)) if np.sum(spec) > 0 else 0.0
        low_pct  = _band_energy_pct(freqs_ref, spec, bands[0])
        mid_pct  = _band_energy_pct(freqs_ref, spec, bands[1])
        high_pct = _band_energy_pct(freqs_ref, spec, bands[2])
        desc_rows.append({
            "class": cls,
            "norm_mode": normalize,
            "norm_factor": scale,
            "centroid_hz": centroid_hz,
            "peak_hz": peak_hz,
            f"band_low_{int(bands[0][0])}-{int(bands[0][1])}_pct": low_pct,
            f"band_mid_{int(bands[1][0])}-{int(bands[1][1])}_pct": mid_pct,
            f"band_high_{int(bands[2][0])}-{int(bands[2][1])}_pct": high_pct
        })

    if not rows or freqs_ref is None:
        info("No curves found for cross-class overlay.")
        return

    # wide CSV
    wide = pd.DataFrame({"freq_hz": freqs_ref})
    for cls, spec in rows:
        wide[canonicalize(cls)] = spec
    out_csv = os.path.join(base_dir, f"cross_class_overlay_{curve_key}.csv")
    wide.to_csv(out_csv, index=False)

    # descriptors CSV
    out_desc = os.path.join(base_dir, f"cross_class_descriptors_{curve_key}.csv")
    pd.DataFrame(desc_rows).to_csv(out_desc, index=False)

    # plot
    plt.figure(figsize=(9,4))
    for cls, spec in rows:
        plt.plot(freqs_ref, spec, label=str(cls))
    plt.xlabel("Frequency (Hz)")
    ylabel = "Magnitude (a.u.)" if normalize == "none" else "Magnitude (normalized, a.u.)"
    plt.ylabel(ylabel)
    title = f"Cross-class overlay: {curve_key.replace('_',' ')} ({normalize} normalization)"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_png = os.path.join(base_dir, f"cross_class_overlay_{curve_key}.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    info(f"Wrote overlay: {out_png}")
    info(f"Wrote overlay CSV: {out_csv}")
    info(f"Wrote descriptors CSV: {out_desc}")

# ---------------------- Top-level driver ----------------------
def run(attribute: str,
        meta_path: str,
        mode: str,
        npz_dir: Optional[str],
        npz_key: Optional[str],
        frames_parquet: Optional[str],
        min_clips: int,
        results_root: str,
        classes_include: Optional[List[str]] = None,
        n_mfcc: int = N_MFCC,
        n_mels: int = N_MELS,
        sr: int = SR,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        gl_iters: int = GL_ITERS,
        t_seconds: float = T_SECONDS,
        top_r: int = TOP_R,
        k_sigma: float = K_SIGMA,
        fmin: float = FMIN,
        fmax_frac_of_nyq: float = FMAX_FRAC_OF_NYQ,
        pca_filtered_frames_limit: int = PCA_FILTERED_FRAMES_LIMIT,
        overlay_curve: str = OVERLAY_CURVE,
        normalize: str = NORMALIZE,
        ref_hz: float = REF_HZ,
        bands: List[Tuple[float,float]] = BANDS):

    # Load metadata
    meta = pd.read_parquet(meta_path) if meta_path.lower().endswith(".parquet") else pd.read_csv(meta_path)
    need = {"filename", attribute}
    if not need.issubset(set(meta.columns)):
        raise KeyError(f"Metadata must include columns: {sorted(list(need))}")

    clips = meta[["filename", attribute]].drop_duplicates().reset_index(drop=True)
    if classes_include is not None and len(classes_include) > 0:
        clips = clips[attribute].isin(classes_include)
        clips = meta[["filename", attribute]][clips].drop_duplicates().reset_index(drop=True)

    counts = meta[attribute].value_counts()
    keep_classes = counts[counts >= min_clips].index.tolist()
    clips = clips[clips[attribute].isin(keep_classes)].reset_index(drop=True)
    classes = sorted(keep_classes)
    if len(classes) < 1:
        raise RuntimeError("No classes after filtering by min_clips")

    info(f"Attribute: {attribute}; Classes: {classes}")
    info(f"Counts: {counts[counts.index.isin(classes)].to_dict()}")

    fs = FrameSource(mode=mode, npz_dir=npz_dir, npz_key=npz_key,
                     frames_parquet=frames_parquet, D=D_TOTAL)

    base_dir = os.path.join(results_root, attribute)
    ensure_dir(base_dir)

    with open(os.path.join(base_dir, "inversion_config.json"), "w") as f:
        json.dump({
            "attribute": attribute,
            "n_mfcc": n_mfcc,
            "n_mels": n_mels,
            "sr": sr,
            "n_fft": n_fft,
            "hop_length": hop_length,
            "griffin_lim_iters": gl_iters,
            "duration_seconds": t_seconds,
            "top_r": top_r,
            "k_sigma": k_sigma,
            "fmin": fmin,
            "fmax_fraction_of_nyquist": fmax_frac_of_nyq,
            "feature_D_total": D_TOTAL,
            "pca_filtered_frames_limit": pca_filtered_frames_limit,
            "overlay_curve": overlay_curve,
            "normalize": normalize,
            "ref_hz": ref_hz,
            "bands": bands,
        }, f, indent=2)

    # Per-class processing
    for cls in tqdm(classes, desc="Classes"):
        clip_list = meta.loc[meta[attribute] == cls, "filename"].drop_duplicates().tolist()
        out_dir = os.path.join(base_dir, canonicalize(cls))
        process_class(cls, clip_list, fs, out_dir,
                      n_mfcc=n_mfcc, top_r=top_r, k_sigma=k_sigma,
                      sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
                      gl_iters=gl_iters, t_seconds=t_seconds,
                      fmin=fmin, fmax_frac_of_nyq=fmax_frac_of_nyq,
                      pca_filtered_frames_limit=pca_filtered_frames_limit)

    # Cross-class overlay (the part you asked for)
    make_cross_class_overlay(base_dir, classes,
                             curve_key=overlay_curve,
                             normalize=normalize,
                             ref_hz=ref_hz,
                             bands=bands)

    info(f"Done. Results written to: {base_dir}")

# ---------------------- CLI ----------------------
def _parse_band(s: str) -> Tuple[float, float]:
    a, b = s.split(",")
    return float(a), float(b)

def main():
    ap = argparse.ArgumentParser(description="Invert MFCC subspaces, compute mean spectra, PCA-filtered means, and cross-class overlays for interpretability.")
    ap.add_argument("--attribute", required=True, type=str, help="Attribute column (e.g., engine_configuration).")
    ap.add_argument("--metadata", required=True, type=str, help="Path to metadata parquet/csv with 'filename' and the attribute.")
    ap.add_argument("--mode", choices=["npz","parquet"], default="npz", help="Feature source: per-clip NPZs or a frames parquet with f0..f59.")
    ap.add_argument("--npz_dir", type=str, default="Data/mfcc", help="Directory with per-clip NPZs when --mode npz.")
    ap.add_argument("--npz_key", type=str, default="mfcc", help="NPZ key for per-clip frames (fallback: 'mfcc' then 'X').")
    ap.add_argument("--frames_parquet", type=str, default=None, help="Path to frames parquet with columns f0..f59 and filename (when --mode parquet).")
    ap.add_argument("--min_clips", type=int, default=30, help="Only include classes with at least this many clips.")
    ap.add_argument("--results_root", type=str, default="Results/inversion", help="Root directory for outputs.")
    ap.add_argument("--classes", type=str, default=None, help="Comma-separated subset of classes to include (optional).")
    # Inversion/audio params
    ap.add_argument("--n_mfcc", type=int, default=N_MFCC)
    ap.add_argument("--n_mels", type=int, default=N_MELS)
    ap.add_argument("--sr", type=int, default=SR)
    ap.add_argument("--n_fft", type=int, default=N_FFT)
    ap.add_argument("--hop_length", type=int, default=HOP_LENGTH)
    ap.add_argument("--gl_iters", type=int, default=GL_ITERS)
    ap.add_argument("--t_seconds", type=float, default=T_SECONDS)
    ap.add_argument("--top_r", type=int, default=TOP_R)
    ap.add_argument("--k_sigma", type=float, default=K_SIGMA)
    ap.add_argument("--fmin", type=float, default=FMIN)
    ap.add_argument("--fmax_frac_of_nyq", type=float, default=FMAX_FRAC_OF_NYQ)
    ap.add_argument("--pca_filtered_frames_limit", type=int, default=PCA_FILTERED_FRAMES_LIMIT)
    # Cross-class overlay controls
    ap.add_argument("--overlay_curve", choices=["pca_filtered_mean","model_mean","data_mean"],
                    default=OVERLAY_CURVE, help="Which curve to overlay across classes.")
    ap.add_argument("--normalize", choices=["none","refhz","l2","area"], default=NORMALIZE,
                    help="How to normalize curves before overlay.")
    ap.add_argument("--ref_hz", type=float, default=REF_HZ, help="Reference Hz for refhz normalization.")
    ap.add_argument("--band_low", type=_parse_band, default=f"{BANDS[0][0]},{BANDS[0][1]}",
                    help="Low band as 'lo,hi' in Hz, e.g. '50,200'.")
    ap.add_argument("--band_mid", type=_parse_band, default=f"{BANDS[1][0]},{BANDS[1][1]}",
                    help="Mid band as 'lo,hi' in Hz, e.g. '200,800'.")
    ap.add_argument("--band_high", type=_parse_band, default=f"{BANDS[2][0]},{BANDS[2][1]}",
                    help="High band as 'lo,hi' in Hz, e.g. '800,3000'.")

    args = ap.parse_args()
    classes = [s.strip() for s in args.classes.split(",")] if args.classes else None
    bands = [args.band_low if isinstance(args.band_low, tuple) else _parse_band(args.band_low),
             args.band_mid if isinstance(args.band_mid, tuple) else _parse_band(args.band_mid),
             args.band_high if isinstance(args.band_high, tuple) else _parse_band(args.band_high)]

    run(attribute=args.attribute,
        meta_path=args.metadata,
        mode=args.mode,
        npz_dir=args.npz_dir,
        npz_key=args.npz_key,
        frames_parquet=args.frames_parquet,
        min_clips=args.min_clips,
        results_root=args.results_root,
        classes_include=classes,
        n_mfcc=args.n_mfcc,
        n_mels=args.n_mels,
        sr=args.sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        gl_iters=args.gl_iters,
        t_seconds=args.t_seconds,
        top_r=args.top_r,
        k_sigma=args.k_sigma,
        fmin=args.fmin,
        fmax_frac_of_nyq=args.fmax_frac_of_nyq,
        pca_filtered_frames_limit=args.pca_filtered_frames_limit,
        overlay_curve=args.overlay_curve,
        normalize=args.normalize,
        ref_hz=args.ref_hz,
        bands=bands)

if __name__ == "__main__":
    main()
