"""
Script: Merging the Option Dataset with the SOFR Interest Rate
--------------------------------------------------------------
Merges the consolidated option dataset (script 1) with Secured
Overnight Financing Rate (SOFR) data, which serves as the proxy for
the risk-free rate (r).

Merge strategy:
    Uses pd.merge_asof with direction='backward' so that each
    QUOTE_DATE is paired with the most recent SOFR rate in effect on
    or before that date. This approach automatically handles holidays
    and non-business days without manual forward-filling.

    If a QUOTE_DATE is earlier than the oldest SOFR observation, the
    row is filled with the first SOFR value as a fallback (with a
    report of the number of affected rows).
"""

import os
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Input: output of script 01_combine_options_data.py
INPUT_OPTIONS = os.path.join("data", "processed", "options_dataset_final.csv")

# Input: SOFR data from the official source (New York Fed)
INPUT_SOFR    = os.path.join("data", "raw", "SOFR.csv")

# Output
OUTPUT_FILE   = os.path.join("data", "processed", "options_with_sofr.csv")


# ============================================================
# 2. DATA-READING & CLEANING FUNCTIONS
# ============================================================

def load_sofr(path):
    """
    Load SOFR data and standardize it into two columns:
        - 'Effective Date' : datetime type
        - 'r'              : interest rate in decimal form
    """
    sofr = pd.read_csv(path, delimiter=";", decimal=",")
    sofr.columns = sofr.columns.str.strip()

    sofr["Effective Date"] = pd.to_datetime(sofr["Effective Date"])

    # Numeric coercion; invalid values are forced to NaN.
    sofr["r"] = pd.to_numeric(sofr["Rate (%)"], errors="coerce") / 100

    sofr = sofr[["Effective Date", "r"]].dropna()
    sofr = sofr.sort_values("Effective Date").reset_index(drop=True)
    return sofr


def load_options(path):
    """Load the option dataset and ensure QUOTE_DATE is of datetime type."""
    df = pd.read_csv(path)
    df["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"])
    return df


# ============================================================
# 3. MERGE FUNCTION
# ============================================================

def merge_with_sofr(options, sofr):
    """
    Merge the option table with the SOFR table using an as-of join
    (direction='backward'). For each QUOTE_DATE, the most recent SOFR
    rate in effect on or before that date is taken.

    If a QUOTE_DATE is earlier than the oldest SOFR observation, the
    first SOFR value is used as a fallback, with an explicit report.
    """
    # merge_asof requires both tables to be sorted on the merge key.
    options_sorted = options.sort_values("QUOTE_DATE").reset_index(drop=True)

    merged = pd.merge_asof(
        options_sorted,
        sofr.rename(columns={"Effective Date": "_rate_date"}),
        left_on="QUOTE_DATE",
        right_on="_rate_date",
        direction="backward",
    )
    merged = merged.drop(columns=["_rate_date"])

    # Fallback: rows with a QUOTE_DATE earlier than the oldest SOFR.
    n_before_sofr = int(merged["r"].isna().sum())
    if n_before_sofr > 0:
        fallback_rate = float(sofr["r"].iloc[0])
        merged["r"] = merged["r"].fillna(fallback_rate)
        print(
            f"Note: {n_before_sofr:,} rows have a QUOTE_DATE before the "
            f"oldest SOFR observation; filled with the first SOFR value "
            f"(r = {fallback_rate:.6f})."
        )

    return merged


# ============================================================
# 4. MAIN PIPELINE
# ============================================================

def main():
    print("--- Loading data ---")
    options = load_options(INPUT_OPTIONS)
    sofr    = load_sofr(INPUT_SOFR)

    print(f"Option rows       : {options.shape[0]:,}")
    print(f"SOFR observations : {sofr.shape[0]:,}")
    print(f"SOFR range        : "
          f"{sofr['Effective Date'].min().date()} to "
          f"{sofr['Effective Date'].max().date()}")

    print("\n--- Merging options + SOFR (as-of backward) ---")
    merged = merge_with_sofr(options, sofr)

    print("\n--- Saving result ---")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    merged.to_csv(OUTPUT_FILE, index=False)

    print(f"Output file       : {OUTPUT_FILE}")
    print(f"Total rows        : {merged.shape[0]:,}")
    print(f"NaNs in column r  : {merged['r'].isna().sum():,}")


if __name__ == "__main__":
    main()