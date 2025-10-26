#!/usr/bin/env python3
"""
Plot a 60-D MFCC (+Δ +ΔΔ) matrix from Data/mfcc/<clip>.npz as a heatmap and save to an image.

Usage examples:
  python plot_mfcc.py --npz Data/mfcc/00hDsNZOL-M_30_40.npz --out figures/mfcc_example.png
  python plot_mfcc.py  # auto-picks the first NPZ under Data/mfcc

Options:
  --mode {zscore,raw,minmax}  # visualization scaling (default: zscore)
  --cmap <matplotlib colormap> (default: magma)
  --max_frames N  # optionally crop to first N frames for readability
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def find_any_npz(root: str) -> str | None:
    if not os.path.isdir(root):
        return None
    for fn in sorted(os.listdir(root)):
        if fn.lower().endswith('.npz'):
            return os.path.join(root, fn)
    return None


def scale_for_viz(X: np.ndarray, mode: str) -> np.ndarray:
    if mode == 'raw':
        return X
    if mode == 'minmax':
        x = X.astype(np.float32, copy=False)
        mn = np.nanmin(x)
        mx = np.nanmax(x)
        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
            return x
        return (x - mn) / (mx - mn)
    # default: zscore per-coefficient for contrast
    x = X.astype(np.float32, copy=False)
    mu = np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, keepdims=True) + 1e-8
    return (x - mu) / sd


def plot_mfcc(mfcc: np.ndarray, title: str, out_path: str, cmap: str = 'magma') -> None:
    # mfcc shape: (T, 60). Show coefficients on y-axis.
    T, D = mfcc.shape
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mfcc.T, aspect='auto', origin='lower', cmap=cmap,
                   extent=[0, T, 1, D])
    # Mark MFCC / Δ / ΔΔ boundaries at 20 and 40
    for y in (20, 40):
        ax.axhline(y, color='white', linewidth=0.8, linestyle='--', alpha=0.8)
    ax.set_ylabel('Coefficient (1..60)')
    ax.set_xlabel('Frame index')
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Value (scaled)')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Visualize a 60-D MFCC (+Δ +ΔΔ) NPZ as heatmap.')
    ap.add_argument('--npz', type=str, default=None, help='Path to Data/mfcc/<clip>.npz')
    ap.add_argument('--out', type=str, default='mfcc_plot.png', help='Output image path')
    ap.add_argument('--mode', type=str, default='zscore', choices=['zscore', 'raw', 'minmax'],
                    help='Scaling mode for visualization')
    ap.add_argument('--cmap', type=str, default='magma', help='Matplotlib colormap')
    ap.add_argument('--max_frames', type=int, default=None, help='Crop to first N frames for readability')
    args = ap.parse_args()

    npz_path = args.npz
    if not npz_path:
        npz_path = find_any_npz('Data/mfcc')
        if not npz_path:
            raise SystemExit('No NPZ found under Data/mfcc and no --npz provided.')

    if not os.path.exists(npz_path):
        raise SystemExit(f'NPZ not found: {npz_path}')

    z = np.load(npz_path)
    key = 'mfcc' if 'mfcc' in z else ('X' if 'X' in z else None)
    if key is None:
        raise SystemExit(f"NPZ missing expected keys ('mfcc' or 'X'): {npz_path}")
    X = np.asarray(z[key])
    if X.ndim != 2 or X.shape[1] != 60:
        raise SystemExit(f'Expected shape (T,60), got {X.shape} from {npz_path}')

    if args.max_frames is not None and args.max_frames > 0:
        X = X[:args.max_frames]

    Xv = scale_for_viz(X, args.mode)
    title = f"MFCC 60-D ({os.path.basename(npz_path)})"
    plot_mfcc(Xv, title=title, out_path=args.out, cmap=args.cmap)
    print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()

