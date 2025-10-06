import os
import pandas as pd
import yt_dlp
import time
import json
import re
from google import genai
from google.genai import types

# Initialize Gemini client (API key from GEMINI_API_KEY)
client = genai.Client()

# Load CSV with all metadata
df = pd.read_csv(
    "unbalanced_train_segments.csv",
    comment="#",
    header=None,
    names=["YTID", "start_seconds", "end_seconds", "positive_labels"],
    skipinitialspace=True
)

# Keep only engine label rows
engine_class_id = "/m/02mk9"
df = df[df['positive_labels'].str.contains(engine_class_id, na=False)]

# Ensure output folders exist
os.makedirs("engine_downloads", exist_ok=True)
os.makedirs("engine_metadata_parquet", exist_ok=True)

# Checkpoint file
checkpoint_file = "checkpoint.txt"
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "r") as f:
        last_processed_index = int(f.read().strip())
else:
    last_processed_index = -1  # Start from the beginning

print(f"Starting from row index {last_processed_index + 1}")

# yt-dlp options
ydl_opts = {"quiet": True, "skip_download": True}

# Rate limit config
requests_per_minute = 10
sleep_time = 60 / requests_per_minute

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for idx, row in df.iterrows():
        if idx <= last_processed_index:
            continue  # Skip already processed rows

        video_id = row["YTID"]
        start = int(row["start_seconds"])
        end = int(row["end_seconds"])
        filename = f"{video_id}_{start}_{end}.wav"
        filepath = os.path.join("engine_downloads", filename)

        if not os.path.exists(filepath):
            continue

        url = f"https://www.youtube.com/watch?v={video_id}"

        # Extract YouTube metadata
        try:
            info = ydl.extract_info(url, download=False)
            title = info.get("title", "Unknown Title")
            description = info.get("description", "")
        except Exception as e:
            print(f"❌ Failed to get metadata for {video_id}: {e}")
            title = "Unknown Title"
            description = ""

        # Gemini prompt
        prompt = f"""
You are an expert in engines and vehicles. Extract structured metadata from the YouTube video title and description.
You may also use your own knowledge about engines, vehicles, and common configurations to infer missing information.

Title: {title}
Description: {description}

Output JSON with the following fields:
- brand: the vehicle manufacturer (e.g., Toyota, Ford). Write "none" if unknown.
- model: the vehicle model (e.g., 4D56-T, Civic Type R). Write "none" if unknown.
- engine_type: specific engine model or code (e.g., 2JZ-GTE, LS3). Write "none" if unknown.
- engine_configuration: engine cylinder layout and count (e.g., V8, V6, inline-4, flat-6, W16). Write "none" if unknown.
- vehicle_type: type of vehicle (e.g., sedan, SUV, truck, JDM, muscle car). Write "none" if unknown.
- fuel_type: gasoline, diesel, electric, hybrid, etc. Write "none" if unknown.
- engine_state: must be exactly one of ["running", "idle", "off", "revving", "accelerating", "none"].
- turbo_supercharged: must be exactly "yes", "no", or "none".
- is_car: must be exactly "yes", "no", or "none", indicating if the vehicle is a car.

Make sure to output **valid JSON only**, and do not include extra explanations or text outside the JSON.
"""

        # Call Gemini
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                ),
            )
            engine_info_text = response.text
        except Exception as e:
            print(f"❌ Gemini failed: {e}")
            engine_info_text = "{}"

        # Clean and parse JSON
        cleaned_text = re.sub(r"^['\"`\s]*json['\"`\s]*", "", engine_info_text.strip(), flags=re.IGNORECASE)
        cleaned_text = cleaned_text.strip("'\"`\n ")
        match = re.search(r"\{.*\}", cleaned_text, flags=re.DOTALL)
        if match:
            cleaned_text = match.group(0)

        try:
            engine_info = json.loads(cleaned_text)
        except json.JSONDecodeError:
            engine_info = {}

        # Build metadata row
        row_data = {
            "filename": filename,
            "title": title,
            "description": description,
            "brand": engine_info.get("brand", ""),
            "model": engine_info.get("model", ""),
            "engine_type": engine_info.get("engine_type", ""),
            "engine_configuration": engine_info.get("engine_configuration", ""),
            "vehicle_type": engine_info.get("vehicle_type", ""),
            "fuel_type": engine_info.get("fuel_type", ""),
            "engine_state": engine_info.get("engine_state", ""),
            "turbo_supercharged": engine_info.get("turbo_supercharged", ""),
            "is_car": engine_info.get("is_car", "")
        }

        # Print all metadata nicely
        print(f"Metadata for {filename}:")
        for key, value in row_data.items():
            print(f"  {key}: {value}")
        print("-" * 80)  # Separator for readability


        # Save to Parquet (one file per row)
        try:
            single_df = pd.DataFrame([row_data])
            single_df.to_parquet(
                os.path.join("engine_metadata_parquet", f"{filename}.parquet"),
                engine='pyarrow',
                index=False
            )
        except Exception as e:
            print(f"❌ Failed to save Parquet for {filename}: {e}")

        # Update checkpoint
        with open(checkpoint_file, "w") as f:
            f.write(str(idx))

        # Respect rate limit
        time.sleep(sleep_time)

print("✅ Finished processing. Individual Parquet files saved in 'engine_metadata_parquet/'")
