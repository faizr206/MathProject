import pandas as pd
import yt_dlp
import os
import subprocess

# Load CSV
df = pd.read_csv(
    "unbalanced_train_segments.csv",
    comment="#",  # ignore the comment lines
    header=None,
    names=["YTID", "start_seconds", "end_seconds", "positive_labels"],
    skipinitialspace=True
)

print(f"Loaded {len(df)} segments from CSV")

# Filter for only the desired label
engine_class_id = "/m/02mk9"
df = df[df['positive_labels'].str.contains(engine_class_id, na=False)]
df = df.reset_index(drop=True)  # reset to ensure 0..N index
print(f"{len(df)} segments after filtering for {engine_class_id}")

# Output folder
os.makedirs("engine_downloads", exist_ok=True)

# 🔎 Find the last successfully downloaded row
last_done = -1
for idx in reversed(df.index):
    video_id = df.loc[idx, "YTID"]
    start = int(df.loc[idx, "start_seconds"])
    end = int(df.loc[idx, "end_seconds"])
    segment_file = f"engine_downloads/{video_id}_{start}_{end}.wav"
    if os.path.exists(segment_file):
        last_done = idx
        break

print(f"✅ Last completed row: {last_done}")
print(f"▶️ Resuming from row {last_done + 1}")

# yt-dlp options
ydl_opts = {
    "format": "bestaudio/best",
    "quiet": True,
    "outtmpl": "engine_downloads/%(id)s.%(ext)s",
    "keepvideo": True,
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for idx, row in df.iloc[last_done + 1:].iterrows():
        video_id = row["YTID"]
        start = float(row["start_seconds"])
        end = float(row["end_seconds"])
        segment_file = f"engine_downloads/{video_id}_{int(start)}_{int(end)}.wav"

        # Skip if already exists
        if os.path.exists(segment_file):
            print(f"[{idx+1}/{len(df)}] Skipping {segment_file} (already exists)")
            continue

        url = f"https://www.youtube.com/watch?v={video_id}"
        print(f"[{idx+1}/{len(df)}] Downloading {video_id} ({start}-{end}) → {segment_file}")

        try:
            # Download full audio
            ydl.download([url])

            # Find downloaded file
            downloaded_file = None
            for f in os.listdir("engine_downloads"):
                if f.startswith(video_id) and not f.endswith(".wav"):
                    downloaded_file = os.path.join("engine_downloads", f)
                    break

            if downloaded_file is None:
                print(f"❌ Could not find downloaded file for {video_id}")
                continue

            # Trim & convert
            subprocess.run([
                "ffmpeg",
                "-hide_banner", "-loglevel", "error",
                "-y",
                "-i", downloaded_file,
                "-ss", str(start),
                "-to", str(end),
                "-ar", "44100",
                segment_file
            ])

            # Remove original file
            if os.path.exists(downloaded_file):
                os.remove(downloaded_file)

        except Exception as e:
            print(f"❌ Failed to process {video_id}: {e}")
