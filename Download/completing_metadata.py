"""
Re-fill missing engine metadata and combine all parquet rows.

- Scans engine_metadata_parquet/ for *.parquet files
- If brand or model is empty/"none", calls Gemini again using the YouTube title/description
- Saves updated row back to its parquet
- Finally concatenates all rows into engine_metadata_combined.parquet and .csv

Requires:
  pip install pandas pyarrow yt-dlp google-genai
Env:
  GEMINI_API_KEY must be set for google.genai client
"""

import os
import re
import json
import time
import glob
import pandas as pd
from google import genai
from google.genai import types

# ----------------------------- Config -----------------------------
PARQUET_DIR = "engine_metadata_parquet"
COMBINED_PARQUET = "engine_metadata_combined.parquet"
COMBINED_CSV = "engine_metadata_combined.csv"

# LLM rate limit
REQUESTS_PER_MIN = 10
SLEEP_SECS = 60.0 / REQUESTS_PER_MIN

# What counts as "missing"
MISSING_TOKENS = {""}; #, "none", "unknown", "n/a", "na", "null"}

# Initialize Gemini client (GEMINI_API_KEY must be set)
client = genai.Client()


# -------------------------- Helper funcs --------------------------
def is_missing(x) -> bool:
    if x is None:
        return True
    s = str(x).strip().lower()
    return s in MISSING_TOKENS


def build_prompt(title: str, description: str) -> str:
    return f"""
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

Make sure to output valid JSON only, and do not include extra explanations or text outside the JSON.
""".strip()


def clean_and_parse_json(text: str) -> dict:
    """Robustly find the first {...} JSON object in text and parse it."""
    cleaned = re.sub(r"^['\"`\s]*json['\"`\s]*", "", (text or "").strip(), flags=re.IGNORECASE)
    cleaned = cleaned.strip("'\"`\n ")
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        return json.loads(cleaned)
    except Exception:
        return {}


def regenerate_from_llm(title: str, description: str) -> dict:
    """Call Gemini and return a dict with the expected keys (may be empty)."""
    prompt = build_prompt(title or "", description or "")
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            ),
        )
        return clean_and_parse_json(resp.text)
    except Exception as e:
        print(f"❌ Gemini failed: {e}")
        return {}


def update_row_with_llm(row: pd.Series) -> pd.Series:
    """If brand or model is missing, call LLM and update fields in row."""
    if not (is_missing(row.get("brand")) or is_missing(row.get("model"))):
        return row  # Nothing to do

    print(f"🔁 Re-querying LLM for {row.get('filename', '<unknown file>')} (brand/model missing)")
    meta = regenerate_from_llm(row.get("title", ""), row.get("description", ""))

    # Only update known keys if present (fallback to original otherwise)
    for key in [
        "brand",
        "model",
        "engine_type",
        "engine_configuration",
        "vehicle_type",
        "fuel_type",
        "engine_state",
        "turbo_supercharged",
        "is_car",
    ]:
        if key in meta and str(meta[key]).strip() != "":
            row[key] = meta[key]

    time.sleep(SLEEP_SECS)
    return row


def process_single_parquet(path: str) -> pd.DataFrame:
    """Load a parquet (expected 1 row), refresh if needed, and save back."""
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"❌ Failed to read {path}: {e}")
        return pd.DataFrame()

    if df.empty:
        print(f"⚠️  {os.path.basename(path)} is empty; skipping.")
        return df

    # Apply update per row (support multi-row files just in case)
    df = df.apply(update_row_with_llm, axis=1)

    # Save back
    try:
        df.to_parquet(path, engine="pyarrow", index=False)
        print(f"💾 Saved updated {os.path.basename(path)}")
    except Exception as e:
        print(f"❌ Failed to save updated {path}: {e}")

    return df


# ------------------------------ Main ------------------------------
def main():
    if not os.path.isdir(PARQUET_DIR):
        raise SystemExit(f"Directory not found: {PARQUET_DIR}")

    files = sorted(glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))
    if not files:
        raise SystemExit(f"No parquet files found in {PARQUET_DIR}/")

    combined = []
    refreshed_count = 0
    total = len(files)

    print(f"📦 Found {total} parquet file(s) in {PARQUET_DIR}/")
    for i, path in enumerate(files, 1):
        print(f"[{i}/{total}] Processing {os.path.basename(path)}")
        before = None
        try:
            tmp = pd.read_parquet(path)
            if not tmp.empty:
                # Snapshot to see if brand/model are missing before updates
                before = (
                    is_missing(tmp.iloc[0].get("brand"))
                    or is_missing(tmp.iloc[0].get("model"))
                )
        except Exception:
            pass

        df = process_single_parquet(path)
        if before:
            refreshed_count += 1
        if not df.empty:
            combined.append(df)

    # Combine all into one DataFrame
    if combined:
        big = pd.concat(combined, ignore_index=True)
        # Keep latest per filename if duplicates happen to exist
        big = big.drop_duplicates(subset=["filename"], keep="last")

        try:
            big.to_parquet(COMBINED_PARQUET, engine="pyarrow", index=False)
            big.to_csv(COMBINED_CSV, index=False)
            print(f"✅ Combined dataset saved: {COMBINED_PARQUET}, {COMBINED_CSV}")
        except Exception as e:
            print(f"❌ Failed to save combined outputs: {e}")

        # Quick summary
        missing_brand = big["brand"].apply(is_missing).sum() if "brand" in big.columns else 0
        missing_model = big["model"].apply(is_missing).sum() if "model" in big.columns else 0
        print(
            f"Summary: refreshed={refreshed_count}, total_rows={len(big)}, "
            f"missing_brand={missing_brand}, missing_model={missing_model}"
        )
    else:
        print("⚠️ No data combined (all files empty or unreadable).")


if __name__ == "__main__":
    main()
