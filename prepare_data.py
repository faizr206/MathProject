# prepare_engine_subset_max60_preproc.py
# Requirements:
#   pip install pandas numpy librosa soundfile pyarrow

import os
import re
import math
import shutil
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import soundfile as sf
import librosa

# ----------------- Config (change if needed) -----------------
METADATA_PATH = "Download/engine_metadata_combined.parquet"
AUDIO_SRC_DIR = "Download/engine_downloads"

DEST_DIR = "Data"
DEST_AUDIO_DIRNAME = "engine_downloads"   # -> Data/engine_downloads
FRAMES_DIRNAME = "frames"                 # -> Data/frames
OUTPUT_PARQUET = "metadata.parquet"
FRAMES_INDEX_PARQUET = "frames_index.parquet"

# Class cap
MAX_PER_CLASS = 60

# Audio preprocessing
TARGET_SR = 22050
TRIM_TOP_DB = 40

# Normalization: choose "peak" or "rms"
NORM_METHOD = "peak"
PEAK_TARGET = 0.99                 # for peak norm
RMS_TARGET_DBFS = -20.0            # for RMS norm (≈0.1 linear)

# Frame selection
FRAME_LENGTH = 2048                # samples
HOP_LENGTH = 512                   # samples
FRAMES_PER_CLIP = 50               # aim; if fewer available, take all

# Reproducibility
RANDOM_SEED = 42
# -------------------------------------------------------------

# Canonical categories we keep
KEEP_CATEGORIES = {
    "inline-4": "inline-4",
    "v8": "V8",
    "inline-6": "inline-6",
    "single-cylinder": "single-cylinder",
    "v6": "V6",
}

# Common variant normalization to canonical keys above
NORMALIZE_MAP = {
    # inline-4
    "inline-4": "inline-4", "inline4": "inline-4", "straight-4": "inline-4",
    "i4": "inline-4", "l4": "inline-4", "4-inline": "inline-4",
    # inline-6
    "inline-6": "inline-6", "inline6": "inline-6", "straight-6": "inline-6",
    "i6": "inline-6", "l6": "inline-6", "6-inline": "inline-6",
    # V8
    "v8": "v8", "v-8": "v8",
    # V6
    "v6": "v6", "v-6": "v6",
    # single-cylinder
    "single-cylinder": "single-cylinder", "single": "single-cylinder",
    "singlecylinder": "single-cylinder", "1-cylinder": "single-cylinder",
    "one-cylinder": "single-cylinder",
}

def slugify_engine_config(x: str) -> str:
    """Lowercase, collapse non-alnum to '-', trim, then map to known key."""
    if not isinstance(x, str) or not x.strip():
        return ""
    s = re.sub(r"[^a-z0-9]+", "-", x.lower()).strip("-")
    return NORMALIZE_MAP.get(s, s)

def to_canonical(label_key: str) -> str:
    """Map normalized key to final canonical label (display form)."""
    return KEEP_CATEGORIES.get(label_key, "")

def find_filename_column(df: pd.DataFrame) -> str:
    for cand in ["filename", "file_name", "file", "filepath", "path"]:
        if cand in df.columns:
            return cand
    raise KeyError("Could not find a filename-like column in metadata. "
                   "Tried: filename, file_name, file, filepath, path")

def build_case_insensitive_file_index(root_dir: str):
    """Map lowercased basenames to relative paths under root_dir."""
    index = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            key = fn.lower()
            rel = os.path.relpath(os.path.join(dirpath, fn), root_dir)
            if key not in index:
                index[key] = rel
    return index

def cap_sample_per_class(df: pd.DataFrame, label_col: str, max_per_class: int, seed: int) -> pd.DataFrame:
    """For each class: if count > cap, sample cap; else take all."""
    rng = np.random.default_rng(seed)
    parts = []
    for c, grp in df.groupby(label_col, dropna=False):
        take = min(len(grp), max_per_class)
        if take == len(grp):
            parts.append(grp)
        else:
            parts.append(grp.sample(n=take, random_state=int(rng.integers(0, 1_000_000))))
    out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out

def ensure_unique_basename(dest_dir: str, base: str) -> str:
    """Avoid overwriting if two different source files share a basename."""
    target = os.path.join(dest_dir, base)
    if not os.path.exists(target):
        return base
    stem, ext = os.path.splitext(base)
    i = 1
    while True:
        candidate = f"{stem}_{i}{ext}"
        if not os.path.exists(os.path.join(dest_dir, candidate)):
            return candidate
        i += 1

def load_resample_mono(path: str, target_sr: int) -> np.ndarray:
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y

def trim_non_silent(y: np.ndarray, top_db: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    if y.size == 0:
        return y, []
    intervals = librosa.effects.split(y, top_db=top_db)
    if intervals.shape[0] == 0:
        return np.array([], dtype=y.dtype), []
    # Concatenate voiced/high-energy chunks
    parts = [y[start:end] for (start, end) in intervals]
    y_trim = np.concatenate(parts) if parts else np.array([], dtype=y.dtype)
    return y_trim, [(int(s), int(e)) for s, e in intervals]

def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))

def normalize_audio(y: np.ndarray, method: str = "peak",
                    peak_target: float = 0.99,
                    rms_target_dbfs: float = -20.0) -> np.ndarray:
    if y.size == 0:
        return y
    y = y.astype(np.float32, copy=False)
    if method.lower() == "rms":
        target_lin = 10.0 ** (rms_target_dbfs / 20.0)
        cur = rms(y)
        if cur > 0:
            y = y * (target_lin / cur)
    else:  # peak
        peak = float(np.max(np.abs(y)))
        if peak > 0:
            y = y * (peak_target / peak)
    # Safety clip
    y = np.clip(y, -1.0, 1.0)
    return y

def select_uniform_frames(y: np.ndarray, n_frames: int,
                          frame_length: int, hop_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      frames_sel: shape (n_sel, frame_length)
      starts_sel: shape (n_sel,) sample indices
    """
    if y.size < frame_length:
        return np.empty((0, frame_length), dtype=np.float32), np.empty((0,), dtype=np.int64)

    # Build all valid frame starts (center=False)
    n_total = 1 + (len(y) - frame_length) // hop_length
    if n_total <= 0:
        return np.empty((0, frame_length), dtype=np.float32), np.empty((0,), dtype=np.int64)

    # Uniformly spaced indices across [0, n_total-1]
    k = min(n_frames, n_total)
    idxs = np.linspace(0, n_total - 1, num=k, dtype=int)
    starts = idxs * hop_length
    frames = np.stack([y[s:s + frame_length] for s in starts], axis=0).astype(np.float32)
    return frames, starts

def write_wav(path: str, y: np.ndarray, sr: int):
    # 16-bit PCM for compactness/compatibility
    sf.write(path, y, sr, subtype="PCM_16")

def main():
    # Prepare output dirs
    os.makedirs(DEST_DIR, exist_ok=True)
    dest_audio_dir = os.path.join(DEST_DIR, DEST_AUDIO_DIRNAME)
    frames_dir = os.path.join(DEST_DIR, FRAMES_DIRNAME)
    os.makedirs(dest_audio_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    # Read metadata
    df = pd.read_parquet(METADATA_PATH)
    fname_col = find_filename_column(df)

    # Normalize & filter engine_configuration
    if "engine_configuration" not in df.columns:
        raise KeyError("Metadata lacks 'engine_configuration' column.")

    norm_key = df["engine_configuration"].apply(slugify_engine_config)
    df["engine_configuration"] = norm_key.apply(to_canonical)

    # Keep only the requested 5 categories
    df = df[df["engine_configuration"].isin(KEEP_CATEGORIES.values())].copy()

    # Drop rows without a filename value
    df = df[df[fname_col].astype(str).str.strip().ne("")].copy()
    if df.empty:
        raise ValueError("No rows left after filtering for the 5 engine configurations.")

    # Source audio index
    file_index = build_case_insensitive_file_index(AUDIO_SRC_DIR)

    # Keep rows whose audio exists; record missing
    missing = []
    resolved_paths = []
    for fn in df[fname_col].astype(str):
        key = fn.lower()
        if key in file_index:
            resolved_paths.append(file_index[key])
        else:
            missing.append(fn)
            resolved_paths.append(None)
    df["src_relpath"] = resolved_paths

    if missing:
        with open(os.path.join(DEST_DIR, "missing_files.txt"), "w", encoding="utf-8") as f:
            for m in sorted(set(missing)):
                f.write(m + "\n")
        df = df[df["src_relpath"].notna()].copy()

    if df.empty:
        raise ValueError("All filtered rows are missing audio files in the source directory.")

    # Cap per class (≤ MAX_PER_CLASS each)
    before_counts = df["engine_configuration"].value_counts().to_dict()
    df_capped = cap_sample_per_class(df, "engine_configuration", MAX_PER_CLASS, RANDOM_SEED)
    after_counts = df_capped["engine_configuration"].value_counts().to_dict()

    # Process each audio: resample->trim->normalize->save; also extract frames
    out_basenames = []
    out_relpaths = []
    frame_rows = []  # rows for frames_index.parquet

    processed_empty = []

    for _, row in df_capped.iterrows():
        src_rel = row["src_relpath"]
        src_path = os.path.join(AUDIO_SRC_DIR, src_rel)
        base = os.path.basename(src_rel)
        unique_base = ensure_unique_basename(dest_audio_dir, base)
        dst_audio_path = os.path.join(dest_audio_dir, unique_base)

        # Load/mono/resample
        try:
            y = load_resample_mono(src_path, TARGET_SR)
        except Exception as e:
            processed_empty.append((base, f"load_error: {e}"))
            continue

        # Trim non-silent parts (keep high-energy chunks)
        y_trim, intervals = trim_non_silent(y, TRIM_TOP_DB)
        if y_trim.size == 0:
            processed_empty.append((base, "no_voiced_audio_after_trim"))
            continue

        # Normalize loudness
        y_norm = normalize_audio(y_trim, method=NORM_METHOD, peak_target=PEAK_TARGET, rms_target_dbfs=RMS_TARGET_DBFS)

        # Save processed WAV (PCM_16)
        try:
            write_wav(dst_audio_path, y_norm, TARGET_SR)
        except Exception as e:
            processed_empty.append((base, f"write_error: {e}"))
            continue

        # Select frames uniformly from voiced audio
        frames_sel, starts_sel = select_uniform_frames(
            y_norm, FRAMES_PER_CLIP, FRAME_LENGTH, HOP_LENGTH
        )

        # Save frames as one NPZ per clip (reduces tiny-file explosion)
        clip_stem, _ = os.path.splitext(unique_base)
        frames_npz_rel = os.path.join(FRAMES_DIRNAME, f"{clip_stem}.npz")
        frames_npz_path = os.path.join(DEST_DIR, frames_npz_rel)

        # Even if 0 frames (super short), still write an empty NPZ to keep metadata consistent
        np.savez_compressed(
            frames_npz_path,
            frames=frames_sel.astype(np.float32),
            start_samples=starts_sel.astype(np.int64),
            sr=np.array([TARGET_SR], dtype=np.int32),
            frame_length=np.array([FRAME_LENGTH], dtype=np.int32),
            hop_length=np.array([HOP_LENGTH], dtype=np.int32),
            norm_method=np.array([NORM_METHOD])
        )

        # Record outputs
        out_basenames.append(unique_base)
        out_relpaths.append(os.path.join(DEST_AUDIO_DIRNAME, unique_base))

        # Add rows for each frame
        for idx, (samp_start) in enumerate(starts_sel):
            samp_end = int(samp_start + FRAME_LENGTH)
            frame_rows.append({
                "clip_filename": unique_base,
                "engine_configuration": row["engine_configuration"],
                "frame_idx": int(idx),
                "start_sample": int(samp_start),
                "end_sample": int(samp_end),
                "start_time": float(samp_start) / TARGET_SR,
                "end_time": float(samp_end) / TARGET_SR,
                "npz_path": frames_npz_rel,
                "sr": TARGET_SR,
                "frame_length": FRAME_LENGTH,
                "hop_length": HOP_LENGTH,
            })

    # Update metadata for processed clips (only those successfully written)
    keep_mask = pd.Series(out_basenames).isin(out_basenames)
    # Build mapping from original df_capped order to successful outputs
    success_map = {}
    for base, rel in zip(out_basenames, out_relpaths):
        success_map[base] = rel

    # Keep only rows whose processed file exists
    processed_set = set(out_basenames)
    rows_out = []
    for _, row in df_capped.iterrows():
        src_rel = row["src_relpath"]
        base = os.path.basename(src_rel)
        # We used possibly a uniquified basename; find it by checking presence in processed_set
        # A direct mapping from original base->unique_base is not tracked per-row, so we infer by existence in Data dir.
        # Instead, regenerate the unique name deterministically again:
        # We'll pick the first match in out_basenames that shares the stem; safer approach: check file existence:
        # Simplify: if any processed file starts with original stem, accept. Otherwise, skip.
        orig_stem = os.path.splitext(base)[0]
        # Find best candidate among processed basenames
        candidates = [b for b in out_basenames if b == base or b.startswith(orig_stem + "_")]
        if not candidates:
            continue
        # Choose the earliest candidate by order of creation
        chosen = candidates[0]
        relpath = os.path.join(DEST_AUDIO_DIRNAME, chosen)
        new_row = row.copy()
        new_row["relative_path"] = relpath
        new_row[find_filename_column(df)] = chosen
        rows_out.append(new_row)

    if not rows_out:
        raise RuntimeError("No clips were successfully processed. Check logs and input paths.")

    df_out = pd.DataFrame(rows_out)
    df_out = df_out.drop(columns=["src_relpath"])
    out_parquet_path = os.path.join(DEST_DIR, OUTPUT_PARQUET)
    df_out.to_parquet(out_parquet_path, index=False)

    # Write frames index parquet
    frames_index_path = os.path.join(DEST_DIR, FRAMES_INDEX_PARQUET)
    pd.DataFrame(frame_rows).to_parquet(frames_index_path, index=False)

    # Print summary
    print("=== Engine subset (per-class cap + preprocessing) complete ===")
    print(f"Source metadata: {METADATA_PATH}")
    print(f"Audio source dir: {AUDIO_SRC_DIR}")
    print(f"Destination dir:  {DEST_DIR}")
    print(f"Processed audio:  {os.path.join(DEST_DIR, DEST_AUDIO_DIRNAME)}")
    print(f"Frames NPZ dir:   {os.path.join(DEST_DIR, FRAMES_DIRNAME)}")
    print(f"Wrote metadata:   {out_parquet_path}")
    print(f"Wrote frames idx: {frames_index_path}")
    if os.path.exists(os.path.join(DEST_DIR, "missing_files.txt")):
        print("Some source files were missing. See Data/missing_files.txt")
    if processed_empty:
        print("\nSkipped/empty after trim:")
        for base, reason in processed_empty[:10]:
            print(f"  {base}: {reason}")
        if len(processed_empty) > 10:
            print(f"  ... and {len(processed_empty) - 10} more")

    print("\nCounts before cap (kept classes only):")
    for k in sorted(before_counts):
        print(f"  {k}: {before_counts[k]}")

    print(f"\nCounts after cap (MAX_PER_CLASS = {MAX_PER_CLASS}):")
    for k, v in sorted(after_counts.items()):
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
