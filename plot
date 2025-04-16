import pandas as pd
import glob
import os

# ✅ Step 1: Set the base folder path
parent_dir = r"C:\Users\2179048\Downloads\Y21, Y22, Y23 - Sem1 to Sem6  2024 (1)"

# ✅ Step 2: Use glob to find all CSVs recursively
csv_files = glob.glob(os.path.join(parent_dir, "**", "*.csv"), recursive=True)
print(f"📁 Total CSV files found: {len(csv_files)}")

all_data = []

# ✅ Step 3: Read all CSVs and store DataFrames
for file_path in csv_files:
    print(f"📄 Reading file: {file_path}")
    try:
        df = pd.read_csv(file_path)
        # Add source folder (optional)
        df['SourceFolder'] = os.path.basename(os.path.dirname(file_path))
        all_data.append(df)
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")

# ✅ Step 4: Combine and export
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    display(final_df.head())  # Show first 5 rows
    final_df.to_csv("merged_data.csv", index=False)
    print("✅ Data merged and saved as 'merged_data.csv'")
else:
    print("⚠️ No CSVs were successfully read.")
