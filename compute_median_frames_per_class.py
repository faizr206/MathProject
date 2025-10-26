# compute_median_frames_per_class.py
# Requires: pandas, pyarrow (for parquet), numpy
# Usage (example):
#   python compute_median_frames_per_class.py \
#       --frames-parquet Data/mfcc_index.parquet \
#       --attribute-col engine_configuration \
#       --filename-col filename \
#       --out-csv Results/summary/frames_per_clip_by_class.csv

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def compute_from_frames_parquet(parquet_path: Path, attribute_col: str, filename_col: str):
    df = pd.read_parquet(parquet_path)

    missing = [c for c in (attribute_col, filename_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Parquet missing required columns: {missing}. "
                         f"Found columns: {list(df.columns)[:15]}...")

    # Count frames per (class, clip)
    frames_per_clip = (
        df.groupby([attribute_col, filename_col], observed=True)
          .size()
          .rename("frames")
          .reset_index()
    )

    # Aggregate per class
    by_class = (
        frames_per_clip.groupby(attribute_col, observed=True)["frames"]
        .agg(
            clips="count",
            total_frames="sum",
            median_frames="median",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
            mean_frames="mean",
            std_frames="std",
        )
        .reset_index()
        .rename(columns={attribute_col: "class"})
    )
    by_class["IQR"] = by_class["q75"] - by_class["q25"]

    # Order columns nicely
    cols = ["class", "clips", "total_frames", "median_frames", "IQR", "q25", "q75", "mean_frames", "std_frames"]
    by_class = by_class[cols]

    # Overall (across all classes), in case you want it
    overall = (
        frames_per_clip["frames"].agg(
            clips="count",
            total_frames="sum",
            median_frames="median",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
            mean_frames="mean",
            std_frames="std",
        )
    )
    overall["IQR"] = overall["q75"] - overall["q25"]
    overall = pd.DataFrame([overall])[["clips","total_frames","median_frames","IQR","q25","q75","mean_frames","std_frames"]]

    return frames_per_clip, by_class.sort_values("class").reset_index(drop=True), overall

def main():
    ap = argparse.ArgumentParser(description="Compute median frames/clip per class from a frames parquet.")
    ap.add_argument("--frames-parquet", type=Path, required=True,
                    help="Path to frames or MFCC index parquet with at least [filename, attribute] columns.")
    ap.add_argument("--attribute-col", type=str, default="engine_configuration",
                    help="Name of the attribute/class column (default: engine_configuration).")
    ap.add_argument("--filename-col", type=str, default="filename",
                    help="Name of the clip filename/id column (default: filename).")
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="Optional path to save the per-class summary CSV.")
    args = ap.parse_args()

    try:
        _, by_class, overall = compute_from_frames_parquet(
            args.frames_parquet, args.attribute_col, args.filename_col
        )
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # Pretty print
    pd.set_option("display.max_rows", None)
    print("\nPer-class frames/clip summary:")
    print(by_class.to_string(index=False, formatters={
        "median_frames": "{:.1f}".format, "IQR": "{:.1f}".format,
        "q25": "{:.1f}".format, "q75": "{:.1f}".format,
        "mean_frames": "{:.2f}".format, "std_frames": "{:.2f}".format
    }))
    print("\nOverall frames/clip summary (all classes pooled over clips):")
    print(overall.to_string(index=False, formatters={
        "median_frames": "{:.1f}".format, "IQR": "{:.1f}".format,
        "q25": "{:.1f}".format, "q75": "{:.1f}".format,
        "mean_frames": "{:.2f}".format, "std_frames": "{:.2f}".format
    }))

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        by_class.to_csv(args.out_csv, index=False)
        print(f"\nSaved per-class summary to: {args.out_csv}")

if __name__ == "__main__":
    main()
