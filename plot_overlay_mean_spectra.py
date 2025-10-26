# plot_overlay_mean_spectra.py
# Overlay mean spectra (Hz vs dB) for all classes under Results/inversion/<attribute>/*
#
# Usage (example):
#   python plot_overlay_mean_spectra.py --attribute engine_configuration \
#       --results_root Results/inversion --sr 22050 --n_fft 2048 --hop_length 512 \
#       --normalize max --fmin 20 --fmax 12000 --smoothing 9
#
# Outputs:
#   Results/inversion/<attribute>/overlay_mean_spectra.png
#   Results/inversion/<attribute>/overlay_mean_spectra.pdf
#   Results/inversion/<attribute>/overlay_mean_spectra_db.csv  (frequency + one column per class)

import os
import argparse
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def list_class_dirs(root: str):
    if not os.path.isdir(root):
        return []
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w is None or w <= 1:
        return x
    w = int(w)
    if w % 2 == 0:  # make odd
        w += 1
    pad = w // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=np.float64) / w
    return np.convolve(xpad, kernel, mode="valid")

def compute_median_spectrum_from_wav(wav_path: str, sr: int, n_fft: int, hop_length: int):
    y, sr_file = librosa.load(wav_path, sr=sr, mono=True)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    spec = np.median(S, axis=1)  # (1 + n_fft//2,)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return freqs, spec

def normalize_spec(spec: np.ndarray, mode: str):
    eps = 1e-12
    if mode == "none":
        return spec
    if mode == "max":
        m = np.max(spec) + eps
        return spec / m
    if mode == "area":
        s = np.sum(spec) + eps
        return spec / s
    raise ValueError("normalize must be one of {'none','max','area'}")

def main():
    ap = argparse.ArgumentParser(description="Overlay mean spectra (Hz vs dB) per class.")
    ap.add_argument("--attribute", required=True, type=str, help="Attribute name, e.g. engine_configuration")
    ap.add_argument("--results_root", type=str, default="Results/inversion", help="Root directory used by inversion script")
    ap.add_argument("--sr", type=int, default=22050, help="Sample rate (match inversion)")
    ap.add_argument("--n_fft", type=int, default=2048, help="FFT size (match inversion)")
    ap.add_argument("--hop_length", type=int, default=512, help="Hop length (match inversion)")
    ap.add_argument("--normalize", type=str, default="max", choices=["none","max","area"],
                    help="Per-curve normalization before dB (recommended: 'max')")
    ap.add_argument("--fmin", type=float, default=20.0, help="Min frequency to show")
    ap.add_argument("--fmax", type=float, default=None, help="Max frequency to show (default: Nyquist)")
    ap.add_argument("--smoothing", type=int, default=9, help="Moving-average window (odd). Set 1 to disable.")
    args = ap.parse_args()

    base = os.path.join(args.results_root, args.attribute)
    out_png = os.path.join(base, "overlay_mean_spectra.png")
    out_pdf = os.path.join(base, "overlay_mean_spectra.pdf")
    out_csv = os.path.join(base, "overlay_mean_spectra_db.csv")
    os.makedirs(base, exist_ok=True)

    class_dirs = list_class_dirs(base)
    if not class_dirs:
        raise SystemExit(f"No class folders found under: {base}")

    curves = {}
    freqs_ref = None

    for cls in class_dirs:
        wav_path = os.path.join(base, cls, "mean_envelope.wav")
        if not os.path.exists(wav_path):
            # skip silently if class has no wav (e.g., empty or filtered)
            continue
        freqs, spec = compute_median_spectrum_from_wav(wav_path, args.sr, args.n_fft, args.hop_length)
        if freqs_ref is None:
            freqs_ref = freqs
        else:
            if len(freqs) != len(freqs_ref) or not np.allclose(freqs, freqs_ref):
                raise RuntimeError("Frequency axis mismatch across classes. Ensure same sr/n_fft for all.")
        # frequency cropping
        keep = np.ones_like(freqs, dtype=bool)
        if args.fmin is not None:
            keep &= freqs >= args.fmin
        if args.fmax is not None:
            keep &= freqs <= args.fmax
        freqs_use = freqs[keep]
        spec_use = spec[keep]

        # normalization (on amplitude), smoothing, then dB
        spec_norm = normalize_spec(spec_use, args.normalize)
        spec_smooth = moving_average(spec_norm, args.smoothing)
        spec_db = librosa.amplitude_to_db(spec_smooth, ref=np.max(spec_smooth) if args.normalize!="none" else np.max(spec_smooth))

        curves[cls] = (freqs_use, spec_db)

    if not curves:
        raise SystemExit("No curves assembled (no mean_envelope.wav files found).")

    # Plot
    plt.figure(figsize=(9,5))
    for cls, (f_hz, db_vals) in curves.items():
        plt.plot(f_hz, db_vals, label=cls)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(f"Overlay: Mean Spectrum per Class — {args.attribute}")
    plt.grid(True, alpha=0.3)
    plt.legend(ncols=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.savefig(out_pdf)
    plt.close()

    # Save CSV (aligned by frequency). Use the first curve's frequency as index.
    # Reindex others to that axis (they should already match).
    classes_sorted = sorted(curves.keys())
    f0 = curves[classes_sorted[0]][0]
    df = pd.DataFrame({"frequency_hz": f0})
    for cls in classes_sorted:
        f_hz, db_vals = curves[cls]
        if len(f_hz) != len(f0) or not np.allclose(f_hz, f0):
            raise RuntimeError("Frequency axis mismatch when building CSV.")
        df[cls] = db_vals
    df.to_csv(out_csv, index=False)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    main()
