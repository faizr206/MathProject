import pandas as pd
import os

def check_parquet_empty(idx, df):
    row = df.iloc[idx]  # pick the row by index
    video_id = row["YTID"]
    start = int(row["start_seconds"])
    end = int(row["end_seconds"])
    filename = f"{video_id}_{start}_{end}.wav"
    parquet_path = os.path.join("engine_metadata_parquet", f"{filename}.parquet")

    if not os.path.exists(parquet_path):
        print(f"❌ No parquet for {idx} ({filename})")
        return True  # treat missing as "empty"

    try:
        temp_df = pd.read_parquet(parquet_path)
        if temp_df.empty:
            print(f"⚠️ Empty parquet at {idx} ({filename})")
            return True
        else:
            print(f"✅ Parquet OK at {idx} ({filename}), {len(temp_df)} rows")
            print(temp_df.to_string(index=False))  # 👈 print contents
            return False
    except Exception as e:
        print(f"❌ Failed to read parquet at {idx} ({filename}): {e}")
        return True



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

original_idx = 167874

# Check if it's still in the filtered df
if original_idx in df.index:
    pos = df.index.get_loc(original_idx)  # get its position (0-based)
    print(f"✅ Original index {original_idx} is at row {pos} in the filtered df")
else:
    print(f"❌ Original index {original_idx} is not in the filtered df")

for i in range(200):
    print(df.index[pos])
    print(check_parquet_empty(pos, df))
    pos -= 1