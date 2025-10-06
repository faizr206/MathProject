# make_mfcc_frames.py
# Requirements:
#   pip install pandas numpy librosa soundfile pyarrow

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from typing import Dict, List

# --------------- Config (adjust if needed) ----------------
DATA_DIR = "Data"
AUDIO_DIRNAME = "engine_downloads"       # processed WAVs from prior step
FRAMES_INDEX_PARQUET = "frames_index.parquet"
MFCC_DIRNAME = "mfcc"                    # -> Data/mfcc
MFCC_INDEX_PARQUET = "mfcc_index.parquet"

# MFCC settings (60-D total = 20 + Δ20 + ΔΔ20)
N_MFCC = 20
N_MELS = 64
FMIN = 20.0
FMAX_FRACTION_OF_NYQUIST = 1.0  # 1.0 => sr/2; 0.9 => 90% of Nyquist
INCLUDE_DELTAS = True           # <-- always include Δ and ΔΔ
DELTA_WIDTH = 9                 # must be odd; 9 is standard
# ----------------------------------------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_audio(path: str, sr: int) -> np.ndarray:
    # Files are already standardized by your preprocessing pipeline; enforce sr for safety.
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y

def compute_mfcc_matrix(y: np.ndarray, sr: int, n_fft: int, hop_length: int,
                        n_mfcc: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        center=False,            # align with your frame grid
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0
    )
    mfcc = librosa.feature.mfcc(
        S=librosa.power_to_db(S, ref=np.max),
        n_mfcc=n_mfcc
    )  # shape: (n_mfcc, T)
    return mfcc

def main():
    frames_index_path = os.path.join(DATA_DIR, FRAMES_INDEX_PARQUET)
    mfcc_dir = os.path.join(DATA_DIR, MFCC_DIRNAME)
    ensure_dir(mfcc_dir)

    # Load frame index (made by your previous script)
    fi = pd.read_parquet(frames_index_path)

    # Sanity columns
    required_cols = {"clip_filename", "start_sample", "sr", "frame_length", "hop_length"}
    missing = required_cols - set(fi.columns)
    if missing:
        raise KeyError(f"frames_index.parquet is missing required columns: {sorted(missing)}")

    # Group frames by clip
    groups = fi.groupby("clip_filename", sort=False)

    mfcc_index_rows: List[Dict] = []

    for clip, g in groups:
        g = g.sort_values("start_sample")
        sr = int(g["sr"].iloc[0])
        frame_length = int(g["frame_length"].iloc[0])
        hop_length = int(g["hop_length"].iloc[0])

        # MFCC frequency range
        nyq = sr / 2.0
        fmax = min(nyq, nyq * FMAX_FRACTION_OF_NYQUIST)

        # Load processed audio
        wav_path = os.path.join(DATA_DIR, AUDIO_DIRNAME, clip)
        if not os.path.exists(wav_path):
            # Skip clip if the WAV is missing
            continue
        y = load_audio(wav_path, sr=sr)
        if y.size < frame_length:
            continue

        # Compute MFCC matrix for the full clip on the same grid as frames_index
        mfcc = compute_mfcc_matrix(
            y=y, sr=sr, n_fft=frame_length, hop_length=hop_length,
            n_mfcc=N_MFCC, n_mels=N_MELS, fmin=FMIN, fmax=fmax
        )  # (n_mfcc, T)
        T = mfcc.shape[1]

        # Always add Δ and ΔΔ (so total dims = 60)
        if INCLUDE_DELTAS:
            d1 = librosa.feature.delta(mfcc, width=DELTA_WIDTH, order=1)
            d2 = librosa.feature.delta(mfcc, width=DELTA_WIDTH, order=2)
            mfcc_full = np.concatenate([mfcc, d1, d2], axis=0)  # (n_mfcc*3, T)
        else:
            mfcc_full = mfcc

        # Select MFCC vectors at the exact frame starts saved earlier
        starts = g["start_sample"].astype(int).to_numpy()
        cols = starts // hop_length  # center=False grid ensures integer alignment
        cols = np.clip(cols, 0, T - 1)  # guard against tail

        mfcc_sel = mfcc_full[:, cols].T.astype(np.float32)  # (n_frames, n_coeff) -> expected 60 columns

        # Save per-clip MFCCs
        stem, _ = os.path.splitext(clip)
        mfcc_rel = os.path.join(MFCC_DIRNAME, f"{stem}.npz")
        mfcc_path = os.path.join(DATA_DIR, mfcc_rel)
        np.savez_compressed(
            mfcc_path,
            mfcc=mfcc_sel,                 # (n_frames, 60)
            sr=np.array([sr], dtype=np.int32),
            n_mfcc=np.array([N_MFCC], dtype=np.int32),
            n_mels=np.array([N_MELS], dtype=np.int32),
            frame_length=np.array([frame_length], dtype=np.int32),
            hop_length=np.array([hop_length], dtype=np.int32),
            fmin=np.array([FMIN], dtype=np.float32),
            fmax=np.array([fmax], dtype=np.float32),
            include_deltas=np.array([INCLUDE_DELTAS]),
            n_coeff=np.array([mfcc_sel.shape[1]], dtype=np.int32),  # should be 60
        )

        # Build per-frame index rows
        for local_idx, (_, row) in enumerate(g.iterrows()):
            mfcc_index_rows.append({
                "clip_filename": clip,
                "engine_configuration": row.get("engine_configuration", None),
                "frame_idx": int(row.get("frame_idx", local_idx)),
                "start_sample": int(row["start_sample"]),
                "end_sample": int(row["end_sample"]),
                "start_time": float(row["start_time"]),
                "end_time": float(row["end_time"]),
                "sr": sr,
                "frame_length": frame_length,
                "hop_length": hop_length,
                "mfcc_npz_path": mfcc_rel,
                "mfcc_idx": local_idx,                 # row index into mfcc_sel
                "n_coeff": int(mfcc_sel.shape[1]),     # 60
                "has_deltas": bool(INCLUDE_DELTAS),    # True
            })

    # Write the MFCC index parquet
    out_index_path = os.path.join(DATA_DIR, MFCC_INDEX_PARQUET)
    if mfcc_index_rows:
        pd.DataFrame(mfcc_index_rows).to_parquet(out_index_path, index=False)
    else:
        raise RuntimeError("No MFCC frames were generated. Check inputs and parameters.")

    print("=== MFCC(20)+Δ+ΔΔ frame extraction complete (60-D) ===")
    print(f"Input frames index: {frames_index_path}")
    print(f"Processed audio dir: {os.path.join(DATA_DIR, AUDIO_DIRNAME)}")
    print(f"MFCC npz dir:       {mfcc_dir}")
    print(f"MFCC index parquet: {out_index_path}")
    # Quick peek
    counts = pd.DataFrame(mfcc_index_rows)["engine_configuration"].value_counts()
    print("\nPer-class frame counts (from MFCC index):")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print("\nExample row:", mfcc_index_rows[0])
    print("\nYou can load a clip's MFCCs like:")
    print("  data = np.load('Data/mfcc/<clip_stem>.npz'); data['mfcc'].shape  # (n_frames, 60)")

if __name__ == "__main__":
    main()
