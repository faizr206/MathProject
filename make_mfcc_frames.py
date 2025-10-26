# make_mfcc_frames.py
# Requirements:
#   pip install pandas numpy librosa soundfile pyarrow

import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from typing import Dict, List
from math import ceil, log2

# ================= Recommended setup for 10 s engine clips =================
# ~1k frames/clip: 25 ms window, 10 ms hop. Consistent across all clips.
SR = 22050                 # stable and wide enough (Nyquist ~11.025 kHz)
WIN_MS = 25                # analysis window ~25 ms
HOP_MS = 10                # hop ~10 ms  -> ~100 fps
N_MFCC = 20                # 20 static + deltas + delta-deltas = 60 total
N_MELS = 80                # denser mel filterbank helps timbre detail
FMIN = 30.0                # ignore very low rumble below ~30 Hz
FMAX_FRAC_NYQ = 0.95       # cap near Nyquist to avoid edge artifacts
INCLUDE_DELTAS = True
DELTA_WIDTH = 9            # odd; 9 is standard/safe
CENTER = False             # keep grid aligned to exact frame starts (important!)
VAD_DROP_SILENT = False    # optional: set True to drop very low-energy frames
VAD_POWER_DB = -60.0       # threshold if VAD is used
# ===========================================================================

# --------------- I/O ----------------
DATA_DIR = "Data"
AUDIO_DIRNAME = "engine_downloads"       # processed WAVs
FRAMES_INDEX_PARQUET = "frames_index.parquet"  # must exist (your earlier step)
MFCC_DIRNAME = "mfcc"                    # -> Data/mfcc
MFCC_INDEX_PARQUET = "mfcc_index.parquet"
# ------------------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def next_pow2(n: int) -> int:
    return 1 << ceil(log2(max(1, n)))

def load_audio(path: str, sr: int) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y

def compute_mfcc_matrix(y: np.ndarray, sr: int, win_length: int, hop_length: int,
                        n_mfcc: int, n_mels: int, fmin: float, fmax: float,
                        center: bool) -> np.ndarray:
    n_fft = next_pow2(win_length)  # efficient FFT while preserving 25 ms analysis
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
        center=center,            # keep False to align with frame grid
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
        htk=False,                # psychoacoustically standard; set True if you prefer HTK
        norm="slaney"
    )
    mfcc = librosa.feature.mfcc(
        S=librosa.power_to_db(S, ref=np.max),
        n_mfcc=n_mfcc
    )  # (n_mfcc, T)
    return mfcc

def main():
    # Derived timing (samples)
    win_length = int(round(SR * WIN_MS / 1000.0))
    hop_length = int(round(SR * HOP_MS / 1000.0))

    frames_index_path = os.path.join(DATA_DIR, FRAMES_INDEX_PARQUET)
    mfcc_dir = os.path.join(DATA_DIR, MFCC_DIRNAME)
    ensure_dir(mfcc_dir)

    fi = pd.read_parquet(frames_index_path)

    # Require these columns from your prior framing step
    required_cols = {"clip_filename", "start_sample", "end_sample", "start_time", "end_time"}
    missing = required_cols - set(fi.columns)
    if missing:
        raise KeyError(f"frames_index.parquet is missing required columns: {sorted(missing)}")

    # If your earlier step stored a different sr/win/hop, ignore them and use the recommended ones.
    # We will *re-grid* MFCCs with (SR, WIN_MS, HOP_MS) and then select columns that match the
    # start_sample grid implied by these settings.
    # To do that, we regenerate a clean frame grid per clip below.

    groups = fi.groupby("clip_filename", sort=False)
    mfcc_index_rows: List[Dict] = []

    nyq = SR / 2.0
    fmax = min(nyq, nyq * FMAX_FRAC_NYQ)

    expected_frames_per_10s = (SR * 10 - win_length) // hop_length + 1  # ~1000
    print(f"[INFO] Target framing: SR={SR}, win={WIN_MS} ms, hop={HOP_MS} ms "
          f"-> ~{expected_frames_per_10s} frames per 10 s clip")

    for clip, g in groups:
        wav_path = os.path.join(DATA_DIR, AUDIO_DIRNAME, clip)
        if not os.path.exists(wav_path):
            print(f"[WARN] Missing audio for {clip}, skipping.")
            continue

        y = load_audio(wav_path, sr=SR)
        if y.size < win_length:
            print(f"[WARN] Too short audio for {clip}, skipping.")
            continue

        # Compute MFCCs for the *full clip* on the recommended grid
        mfcc = compute_mfcc_matrix(
            y=y, sr=SR, win_length=win_length, hop_length=hop_length,
            n_mfcc=N_MFCC, n_mels=N_MELS, fmin=FMIN, fmax=fmax, center=CENTER
        )  # (n_mfcc, T_mfcc)
        T_mfcc = mfcc.shape[1]

        # Add Δ and ΔΔ to get 60-D per frame
        if INCLUDE_DELTAS:
            d1 = librosa.feature.delta(mfcc, width=DELTA_WIDTH, order=1)
            d2 = librosa.feature.delta(mfcc, width=DELTA_WIDTH, order=2)
            mfcc_full = np.concatenate([mfcc, d1, d2], axis=0)  # (60, T_mfcc)
        else:
            mfcc_full = mfcc  # (20, T_mfcc)

        # Build a clean frame-start grid for this clip with our recommended hop
        # start_sample_k = k * hop_length  (since center=False)
        # Align to 0 … ensure we don't exceed len(y)-win_length
        max_k = (len(y) - win_length) // hop_length
        starts_clean = (np.arange(max_k + 1) * hop_length).astype(int)  # (T_clean,)
        T_clean = len(starts_clean)

        # Optional: simple VAD to drop very quiet frames (based on mel power)
        if VAD_DROP_SILENT:
            mel_power_db = librosa.power_to_db(
                librosa.feature.melspectrogram(
                    y=y, sr=SR, n_fft=next_pow2(win_length), hop_length=hop_length,
                    win_length=win_length, center=CENTER, n_mels=N_MELS,
                    fmin=FMIN, fmax=fmax, power=2.0, norm="slaney"
                ),
                ref=np.max
            )
            frame_energy_db = mel_power_db.mean(axis=0)  # per-frame average dB
            keep_mask = frame_energy_db > VAD_POWER_DB
            mfcc_full = mfcc_full[:, keep_mask]
            starts_clean = starts_clean[keep_mask]
            T_clean = len(starts_clean)

        # Final per-clip MFCC array (T_clean, n_coeff)
        mfcc_sel = mfcc_full.T.astype(np.float32)

        # Save per-clip MFCCs
        stem, _ = os.path.splitext(clip)
        mfcc_rel = os.path.join(MFCC_DIRNAME, f"{stem}.npz")
        mfcc_path = os.path.join(DATA_DIR, mfcc_rel)
        np.savez_compressed(
            mfcc_path,
            mfcc=mfcc_sel,                 # (T_clean, 60)
            sr=np.array([SR], dtype=np.int32),
            n_mfcc=np.array([N_MFCC], dtype=np.int32),
            n_mels=np.array([N_MELS], dtype=np.int32),
            win_length=np.array([win_length], dtype=np.int32),
            hop_length=np.array([hop_length], dtype=np.int32),
            fmin=np.array([FMIN], dtype=np.float32),
            fmax=np.array([fmax], dtype=np.float32),
            include_deltas=np.array([INCLUDE_DELTAS]),
            n_coeff=np.array([mfcc_sel.shape[1]], dtype=np.int32),  # 60
            starts_clean=starts_clean,       # for traceability
        )

        # Build MFCC index rows aligned to our clean grid (not the legacy one)
        for local_idx, start_samp in enumerate(starts_clean):
            start_time = start_samp / SR
            end_samp = start_samp + win_length
            end_time = end_samp / SR
            # Try to carry class label if present in original index (fallback None)
            # (Use the modal/first label per clip if available)
            maybe_label = g.get("engine_configuration")
            eng_cfg = (maybe_label.mode().iloc[0] if hasattr(maybe_label, "mode") and not maybe_label.empty
                       else maybe_label.iloc[0] if "engine_configuration" in g and len(g) > 0
                       else None)

            mfcc_index_rows.append({
                "clip_filename": clip,
                "engine_configuration": eng_cfg,
                "frame_idx": local_idx,
                "start_sample": int(start_samp),
                "end_sample": int(end_samp),
                "start_time": float(start_time),
                "end_time": float(end_time),
                "sr": SR,
                "win_length": win_length,
                "hop_length": hop_length,
                "mfcc_npz_path": mfcc_rel,
                "mfcc_idx": local_idx,
                "n_coeff": int(mfcc_sel.shape[1]),   # 60
                "has_deltas": bool(INCLUDE_DELTAS),
            })

        # Quick sanity for this clip
        if T_clean < 300:
            print(f"[WARN] Very few frames for {clip}: {T_clean}. "
                  f"Expected around {expected_frames_per_10s}. Check hop/window/VAD settings.")
        elif abs(T_clean - expected_frames_per_10s) / expected_frames_per_10s > 0.1:
            print(f"[NOTE] {clip}: frames={T_clean}, expected~{expected_frames_per_10s} "
                  f"(duration mismatch or trimming?)")

    # Write the MFCC index parquet
    out_index_path = os.path.join(DATA_DIR, MFCC_INDEX_PARQUET)
    if mfcc_index_rows:
        pd.DataFrame(mfcc_index_rows).to_parquet(out_index_path, index=False)
    else:
        raise RuntimeError("No MFCC frames were generated. Check inputs and parameters.")

    print("=== MFCC(20)+Δ+ΔΔ extraction complete (60-D) ===")
    print(f"Processed audio dir: {os.path.join(DATA_DIR, AUDIO_DIRNAME)}")
    print(f"MFCC npz dir:       {mfcc_dir}")
    print(f"MFCC index parquet: {out_index_path}")

    # Quick per-class counts if labels exist
    df_idx = pd.DataFrame(mfcc_index_rows)
    if "engine_configuration" in df_idx.columns:
        counts = df_idx["engine_configuration"].value_counts(dropna=False)
        print("\nPer-class frame counts (balanced hop):")
        for k, v in counts.items():
            print(f"  {k}: {v}")

    # Example
    print("\nYou can load a clip's MFCCs like:")
    print("  data = np.load('Data/mfcc/<clip_stem>.npz'); data['mfcc'].shape  # (T, 60)")

if __name__ == "__main__":
    main()
