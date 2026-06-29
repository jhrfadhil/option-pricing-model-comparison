"""
SPX Option Dataset Filtering Script
-----------------------------------
Applies a series of data-quality and economic filters to the option
dataset to produce a clean research sample ready for use in model
estimation.

Filtering stages:
    0. Data quality      : drop rows with missing values, non-positive
                           values, or invalid dates.
    1. Time to maturity  : compute TTM in days; keep 7 ≤ TTM ≤ 365.
    2. Minimum price      : C_LAST ≥ 0.375 (remove tick noise).
    3. No-arbitrage      : C > max(S − K·e^(−rT), 0).
    4. Moneyness         : 0.8 ≤ S/K ≤ 1.2.

After all filters are applied, the dataset is labeled with a moneyness
classification (ITM / ATM / OTM) based on the S/K ratio, and the
composition of each category is reported to the console. The S_over_K
and MONEYNESS columns are included in the output file so that downstream
analysis does not need to recompute them.

Each filtering stage is reported explicitly: the number of rows dropped,
the percentage, and the rows remaining after the filter.
"""

import os
import pandas as pd
import numpy as np


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FILE  = os.path.join("data", "processed", "options_with_sofr.csv")
OUTPUT_FILE = os.path.join("data", "processed", "options_dataset_filtered.csv")

# Set to False to inspect the filter log without saving the file.
SAVE_OUTPUT = True

# Numeric columns that must be populated (non-NaN).
REQUIRED_NUM_COLS = ["C_LAST", "UNDERLYING_LAST", "STRIKE", "r"]

# Filter parameters (easy to change for sensitivity analysis).
TTM_MIN        = 7       # days
TTM_MAX        = 365     # days
PRICE_MIN      = 0.375   # minimum option price threshold
MONEYNESS_LOW  = 0.8     # lower bound on S/K
MONEYNESS_HIGH = 1.2     # upper bound on S/K

# Moneyness classification parameters (S/K ratio).
ITM_THRESHOLD = 1.05   # S/K > 1.05           -> ITM
ATM_LOW       = 0.97   # 0.97 ≤ S/K ≤ 1.05    -> ATM
ATM_HIGH      = 1.05   # S/K < 0.97           -> OTM


# ============================================================
# 2. REPORTING UTILITY
# ============================================================

def report_filter(label, n_before, df_after):
    """Print a summary of the impact of one filtering stage."""
    n_after = len(df_after)
    dropped = n_before - n_after
    pct = (dropped / n_before * 100) if n_before > 0 else 0.0
    print(f"  [{label}] dropped: {dropped:,} ({pct:.2f}%) | remaining: {n_after:,}")
    return df_after


# ============================================================
# 3. FILTER FUNCTIONS
# ============================================================

def coerce_types(df):
    """Coerce the correct data types on the date and numeric columns."""
    df["QUOTE_DATE"]  = pd.to_datetime(df["QUOTE_DATE"],  errors="coerce")
    df["EXPIRE_DATE"] = pd.to_datetime(df["EXPIRE_DATE"], errors="coerce")
    for col in REQUIRED_NUM_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def apply_data_quality_filters(df):
    """
    Stage 0 — Data-quality filter:
        - Drop rows with invalid dates (NaT).
        - Drop rows with missing values in the essential numeric columns.
        - Drop rows with UNDERLYING_LAST ≤ 0 or STRIKE ≤ 0.
        - Drop rows with C_LAST < 0 (anomalous data).
    """
    print("\nStage 0: Data-quality filter")

    n = len(df)
    df = df.dropna(subset=["QUOTE_DATE", "EXPIRE_DATE"])
    df = report_filter("Invalid dates (NaT)", n, df)

    n = len(df)
    df = df.dropna(subset=REQUIRED_NUM_COLS)
    df = report_filter("NaN in numeric columns", n, df)

    n = len(df)
    df = df[df["UNDERLYING_LAST"] > 0]
    df = report_filter("UNDERLYING_LAST ≤ 0", n, df)

    n = len(df)
    df = df[df["STRIKE"] > 0]
    df = report_filter("STRIKE ≤ 0", n, df)

    n = len(df)
    df = df[df["C_LAST"] >= 0]
    df = report_filter("C_LAST < 0", n, df)

    return df


def apply_ttm_filter(df):
    """
    Stage 1 — Compute Time to Maturity (TTM) in days and apply the
    constraint TTM_MIN ≤ TTM ≤ TTM_MAX.
    """
    print(f"\nStage 1: Time-to-maturity filter ({TTM_MIN}–{TTM_MAX} days)")

    df["TIME_TO_MATURITY"] = (df["EXPIRE_DATE"] - df["QUOTE_DATE"]).dt.days

    # Drop contracts that have already expired or mature today.
    n = len(df)
    df = df[df["TIME_TO_MATURITY"] > 0]
    df = report_filter("TTM ≤ 0 (expired)", n, df)

    # Apply the TTM range.
    n = len(df)
    below = (df["TIME_TO_MATURITY"] < TTM_MIN).sum()
    above = (df["TIME_TO_MATURITY"] > TTM_MAX).sum()
    print(f"  TTM distribution before filter: "
          f"<{TTM_MIN}: {below:,} | >{TTM_MAX}: {above:,} | "
          f"within range: {n - below - above:,}")

    df = df[
        (df["TIME_TO_MATURITY"] >= TTM_MIN) &
        (df["TIME_TO_MATURITY"] <= TTM_MAX)
    ]
    df = report_filter(f"TTM outside {TTM_MIN}–{TTM_MAX}", n, df)

    return df


def apply_price_filter(df):
    """
    Stage 2 — Remove contracts whose price is too low (tick noise):
    C_LAST < PRICE_MIN.
    """
    print(f"\nStage 2: Minimum-price filter (C_LAST ≥ {PRICE_MIN})")

    n = len(df)
    df = df[df["C_LAST"] >= PRICE_MIN]
    df = report_filter(f"C_LAST < {PRICE_MIN}", n, df)

    return df


def apply_no_arbitrage_filter(df):
    """
    Stage 3 — No-arbitrage lower bound for European call options:
        C > max(S − K·e^(−rT), 0)
    Rows that violate this bound indicate anomalous data or recording
    errors.
    """
    print("\nStage 3: No-arbitrage lower-bound filter")

    n = len(df)
    T_years = df["TIME_TO_MATURITY"] / 365
    lower_bound = np.maximum(
        df["UNDERLYING_LAST"] - df["STRIKE"] * np.exp(-df["r"] * T_years),
        0,
    )
    df = df[df["C_LAST"] > lower_bound]
    df = report_filter("No-arbitrage lower-bound violation", n, df)

    return df


def apply_moneyness_filter(df):
    """
    Stage 4 — Restrict moneyness (S/K) to the range
    MONEYNESS_LOW ≤ S/K ≤ MONEYNESS_HIGH to remove illiquid
    deep-in-the-money and deep-out-of-the-money options.
    """
    print(f"\nStage 4: Moneyness filter "
          f"({MONEYNESS_LOW} ≤ S/K ≤ {MONEYNESS_HIGH})")

    n = len(df)
    moneyness = df["UNDERLYING_LAST"] / df["STRIKE"]
    df = df[(moneyness >= MONEYNESS_LOW) & (moneyness <= MONEYNESS_HIGH)]
    df = report_filter("Moneyness outside range", n, df)

    return df


# ============================================================
# 4. MONEYNESS CLASSIFICATION (POST-FILTER)
# ============================================================

def classify_moneyness(df):
    """
    Moneyness classification based on the S/K ratio:
        - ITM : S/K > ITM_THRESHOLD
        - ATM : ATM_LOW ≤ S/K ≤ ATM_HIGH
        - OTM : S/K < ATM_LOW

    Adds the 'S_over_K' and 'MONEYNESS' columns to the dataframe, then
    reports the composition of each category. This step is analytical —
    no rows are dropped.
    """
    print("\nMoneyness classification (ITM / ATM / OTM)")

    df = df.copy()
    df["S_over_K"] = df["UNDERLYING_LAST"] / df["STRIKE"]

    conditions = [
        df["S_over_K"] > ITM_THRESHOLD,
        (df["S_over_K"] >= ATM_LOW) & (df["S_over_K"] <= ATM_HIGH),
        df["S_over_K"] < ATM_LOW,
    ]
    labels = ["ITM", "ATM", "OTM"]
    df["MONEYNESS"] = np.select(conditions, labels, default="UNK")

    total = len(df)
    itm = int((df["MONEYNESS"] == "ITM").sum())
    atm = int((df["MONEYNESS"] == "ATM").sum())
    otm = int((df["MONEYNESS"] == "OTM").sum())
    pct = lambda n: (n / total * 100) if total > 0 else 0.0

    print(f"  Total options                            : {total:,}")
    print(f"  ITM (S/K > {ITM_THRESHOLD})            : "
          f"{itm:,} ({pct(itm):.2f}%)")
    print(f"  ATM ({ATM_LOW} ≤ S/K ≤ {ATM_HIGH})     : "
          f"{atm:,} ({pct(atm):.2f}%)")
    print(f"  OTM (S/K < {ATM_LOW})                  : "
          f"{otm:,} ({pct(otm):.2f}%)")
    print(f"  Check count (ITM + ATM + OTM)          : {itm + atm + otm:,}")

    return df


# ============================================================
# 5. MAIN PIPELINE
# ============================================================

def main():
    print("=== SPX Option Dataset Filtering ===")

    df = pd.read_csv(INPUT_FILE)
    n_awal = len(df)
    print(f"Initial rows: {n_awal:,}")

    # Coerce data types.
    df = coerce_types(df)

    # Apply filters sequentially.
    df = apply_data_quality_filters(df)
    df = apply_ttm_filter(df)
    df = apply_price_filter(df)
    df = apply_no_arbitrage_filter(df)
    df = apply_moneyness_filter(df)

    # Drop the EXPIRE_DATE column (its information is already represented in TTM).
    df = df.drop(columns=["EXPIRE_DATE"])

    # Classify moneyness on the post-filter sample.
    df = classify_moneyness(df)

    # Final summary.
    n_akhir = len(df)
    pct_retained = (n_akhir / n_awal * 100) if n_awal > 0 else 0.0
    print("\n=== Summary ===")
    print(f"Initial rows        : {n_awal:,}")
    print(f"Final rows          : {n_akhir:,} ({pct_retained:.2f}% retained)")
    print(f"S  (min / max)      : "
          f"{df['UNDERLYING_LAST'].min():.2f} / {df['UNDERLYING_LAST'].max():.2f}")
    print(f"K  (min / max)      : "
          f"{df['STRIKE'].min():.2f} / {df['STRIKE'].max():.2f}")
    print(f"TTM (min / max)     : "
          f"{df['TIME_TO_MATURITY'].min()} / {df['TIME_TO_MATURITY'].max()} days")
    print(f"r  (min / max)      : "
          f"{df['r'].min():.6f} / {df['r'].max():.6f}")
    print(f"S/K (min / max)     : "
          f"{df['S_over_K'].min():.4f} / {df['S_over_K'].max():.4f}")

    # Save / skip.
    if SAVE_OUTPUT:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nFile saved          : {OUTPUT_FILE}")
    else:
        print("\nInspection mode: file NOT saved.")

    print("Filtering complete.")


if __name__ == "__main__":
    main()
