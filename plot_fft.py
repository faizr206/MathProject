#!/usr/bin/env python3
"""
Usage:
  python plot_fft.py Data/engine_downloads/--Dxk606LRQ_30_40.wav --out figures/fft_example.png --seconds 5

Requires: numpy, matplotlib, librosa
Install:  pip install numpy matplotlib librosa
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa

def main():
    parser = argparse.ArgumentParser(description="Plot FFT of an audio file.")
    parser.add_argument("audio", help="Path to the audio file (wav, mp3, flac, etc.)")
    parser.add_argument("--out", default="figures/fft_example.png",
                        help="Output image path (default: figures/fft_example.png)")
    parser.add_argument("--sr", type=int, default=None,
                        help="Target sample rate. Default: None (keep native).")
    parser.add_argument("--mono", action="store_true",
                        help="Force mono mixdown (recommended).")
    parser.add_argument("--seconds", type=float, default=None,
                        help="If set, only analyze the first N seconds.")
    parser.add_argument("--logx", action="store_true",
                        help="Plot frequency axis on log scale.")
    args = parser.parse_args()

    # Load audio (librosa supports many formats through soundfile/audioread)
    y, sr = librosa.load(args.audio, sr=args.sr, mono=args.mono or True, duration=args.seconds)
    if y.size == 0:
        raise RuntimeError("Loaded audio is empty.")

    # Apply a Hann window to reduce spectral leakage
    window = np.hanning(len(y))
    y_win = y * window

    # Real FFT (one-sided)
    Y = np.fft.rfft(y_win)
    freqs = np.fft.rfftfreq(len(y_win), d=1.0/sr)

    # Magnitude in dB (add small epsilon for numerical safety)
    mag = np.abs(Y)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))

    # Prepare output directory
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Plot
    plt.figure(figsize=(10, 5))
    if args.logx:
        plt.semilogx(freqs, mag_db)
    else:
        plt.plot(freqs, mag_db)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(f"FFT Magnitude Spectrum\n{os.path.basename(args.audio)} (sr={sr} Hz)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    plt.close()

    print(f"Saved FFT plot to: {args.out}")

if __name__ == "__main__":
    main()
