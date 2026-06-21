"""
SPX Option Data Combination Script (2019-2025)
----------------------------------------------
Combines SPX call option data from three different source formats
into a single final dataset with a uniform column schema.

Source data formats:
    1. .txt files (2019-2023): column names wrapped in square brackets.
    2. .csv file  (2024)      : lowercase column names.
    3. .csv file  (2025)      : CamelCase column names.

Output:
    options_dataset_final.csv with columns:
    QUOTE_DATE, UNDERLYING_LAST, EXPIRE_DATE, C_LAST, C_BID, C_ASK, STRIKE
"""

import os
import glob
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Project directories (relative to the repository root).
RAW_DIR       = os.path.join("data", "raw")        # external downloads (not committed)
PROCESSED_DIR = os.path.join("data", "processed")  # intermediate CSVs (not committed)

FILE_2024   = os.path.join(RAW_DIR, "SPX_2024.csv")
FILE_2025   = os.path.join(RAW_DIR, "SPX_2025.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "options_dataset_final.csv")

# Final column schema to retain.
KEEP_COLS = [
    "QUOTE_DATE",
    "UNDERLYING_LAST",
    "EXPIRE_DATE",
    "C_LAST",
    "C_BID",
    "C_ASK",
    "STRIKE",
]

# Source-column to standard-schema mapping (2024 format).
MAPPING_2024 = {
    "quotedate":       "QUOTE_DATE",
    "underlying_last": "UNDERLYING_LAST",
    "expiration":      "EXPIRE_DATE",
    "last":            "C_LAST",
    "bid":             "C_BID",
    "ask":             "C_ASK",
    "strike":          "STRIKE",
}

# Source-column to standard-schema mapping (2025 format).
MAPPING_2025 = {
    "DataDate":        "QUOTE_DATE",
    "UnderlyingPrice": "UNDERLYING_LAST",
    "Expiration":      "EXPIRE_DATE",
    "Last":            "C_LAST",
    "Bid":             "C_BID",
    "Ask":             "C_ASK",
    "Strike":          "STRIKE",
}

# Numeric columns whose types are validated at the end of the process.
NUMERIC_COLS = ["UNDERLYING_LAST", "C_LAST", "C_BID", "C_ASK", "STRIKE"]


# ============================================================
# 2. DATA-READING FUNCTIONS
# ============================================================

def load_txt_files(folder_path):
    """
    Load all .txt files (2019-2023) from the source folder.
    Column names are stripped of square brackets and excess whitespace,
    then projected onto the KEEP_COLS schema.
    """
    frames = []
    files = sorted(glob.glob(os.path.join(folder_path, "*.txt")))
    print(f"Number of .txt files found: {len(files)}")

    for file in files:
        try:
            df = pd.read_csv(file, skipinitialspace=True)

            # Clean column names: remove [ ] and whitespace.
            df.columns = (
                df.columns
                  .str.replace("[", "", regex=False)
                  .str.replace("]", "", regex=False)
                  .str.strip()
            )

            df = df[KEEP_COLS]
            frames.append(df)
            print(f"  Loaded: {os.path.basename(file)} | rows = {df.shape[0]:,}")

        except KeyError as e:
            print(f"  Incomplete columns in {os.path.basename(file)}: {e}")
        except Exception as e:
            print(f"  Failed to read {os.path.basename(file)}: {e}")

    return frames


def load_csv_mapped(file_path, column_mapping, type_col, date_format=None):
    """
    Load a .csv file in 2024 or 2025 format:
    - Filter rows of type 'call' (case-insensitive).
    - Apply the column-name mapping to the standard schema.
    - Standardize the date format to 'YYYY-MM-DD'.

    Parameters
    ----------
    file_path     : str         Path to the csv file.
    column_mapping: dict        Source-column to standard-schema mapping.
    type_col      : str         Name of the column holding the option type (call/put).
    date_format   : str | None  Explicit date format; None = automatic inference.

    Returns
    -------
    pd.DataFrame | None
    """
    if not os.path.exists(file_path):
        print(f"  Warning: {file_path} not found.")
        return None

    try:
        usecols = list(column_mapping.keys()) + [type_col]
        df = pd.read_csv(file_path, usecols=usecols)

        # Keep only call options (robust to capitalization differences).
        mask_call = df[type_col].astype(str).str.strip().str.lower() == "call"
        df = df.loc[mask_call].copy()

        # Standardize column names.
        df = df.drop(columns=[type_col]).rename(columns=column_mapping)

        # Standardize date format -> 'YYYY-MM-DD'.
        if date_format:
            df["QUOTE_DATE"]  = pd.to_datetime(df["QUOTE_DATE"],  format=date_format)
            df["EXPIRE_DATE"] = pd.to_datetime(df["EXPIRE_DATE"], format=date_format)
        else:
            df["QUOTE_DATE"]  = pd.to_datetime(df["QUOTE_DATE"])
            df["EXPIRE_DATE"] = pd.to_datetime(df["EXPIRE_DATE"])

        df["QUOTE_DATE"]  = df["QUOTE_DATE"].dt.strftime("%Y-%m-%d")
        df["EXPIRE_DATE"] = df["EXPIRE_DATE"].dt.strftime("%Y-%m-%d")

        df = df[KEEP_COLS]
        print(f"  Loaded: {os.path.basename(file_path)} | rows = {df.shape[0]:,}")
        return df

    except Exception as e:
        print(f"  Failed to process {os.path.basename(file_path)}: {e}")
        return None


# ============================================================
# 3. MAIN PIPELINE
# ============================================================

def main():
    frames = []

    print("--- Processing 2019-2023 data ---")
    frames.extend(load_txt_files(RAW_DIR))

    print("\n--- Processing 2024 data ---")
    df_24 = load_csv_mapped(FILE_2024, MAPPING_2024,
                            type_col="type", date_format="%m/%d/%Y")
    if df_24 is not None:
        frames.append(df_24)

    print("\n--- Processing 2025 data ---")
    df_25 = load_csv_mapped(FILE_2025, MAPPING_2025,
                            type_col="Type", date_format=None)
    if df_25 is not None:
        frames.append(df_25)

    # Validate the read results.
    if not frames:
        raise ValueError(
            "No files were read successfully. "
            "Check the folder path and the structure of the source files."
        )

    # Combine all DataFrames.
    print("\n--- Combining all data ---")
    combined = pd.concat(frames, ignore_index=True)
    print(f"Total rows before type validation : {combined.shape[0]:,}")

    # Coerce numeric columns to float; invalid values -> NaN.
    for col in NUMERIC_COLS:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    # Report the number of duplicates (not dropped automatically; the decision is
    # left to the cleaning stage for transparency in data auditing).
    n_dup = combined.duplicated(subset=["QUOTE_DATE", "EXPIRE_DATE", "STRIKE"]).sum()
    print(f"Indicated duplicate rows          : {n_dup:,}")

    # Save the final result.
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"\nFinal dataset saved to            : {OUTPUT_FILE}")
    print(f"Final dimensions                  : "
          f"{combined.shape[0]:,} rows x {combined.shape[1]} columns")


if __name__ == "__main__":
    main()