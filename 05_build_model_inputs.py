"""
Volatility Merge & Model-Input Preparation Script
-------------------------------------------------
Merges the filtered option dataset (script 3) with the volatility
estimates (script 4) to produce three input files ready for use in the
modeling stage:

Output:
    1. input_ANN.csv
       All option data + relative volatility (HV5 … EGARCH).
       Used as the training input for the Artificial Neural Network.

    2. input_BS_MC.csv
       The last 20% of the data (by QUOTE_DATE) from input_ANN.csv.
       Used as the test set for the Black–Scholes & Monte Carlo models.

    3. input_Bachelier.csv
       The last 20% of the data (by QUOTE_DATE) from option data +
       absolute volatility (σ_abs = σ_rel × S).
       Used as the test set for the Bachelier model.

Merge strategy:
    Uses pd.merge_asof(direction='backward') so that each QUOTE_DATE is
    paired with the most recent volatility available on or before that
    date (avoiding look-ahead bias).
"""

import os
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Input
INPUT_OPTIONS = os.path.join("data", "processed", "options_dataset_filtered.csv")
INPUT_VOL_REL = os.path.join("data", "processed", "SPX_volatility_forecast.csv")
INPUT_VOL_ABS = os.path.join("data", "processed", "SPX_absolute_volatility_forecast.csv")

# Output
OUTPUT_ANN       = os.path.join("data", "processed", "input_ANN.csv")
OUTPUT_BS_MC     = os.path.join("data", "processed", "input_BS_MC.csv")
OUTPUT_BACHELIER = os.path.join("data", "processed", "input_Bachelier.csv")

# Proportion of the final data used as the test set.
TEST_FRACTION = 0.20

# Relative and absolute volatility columns.
VOL_REL_COLS = ["HV5", "HV20", "HV60", "HV100", "GARCH", "EGARCH"]
VOL_ABS_COLS = [f"{c}_abs" for c in VOL_REL_COLS]


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def load_and_parse(path, date_col):
    """Load CSV, clean column names, and parse the date column."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    return df


def asof_merge(options, vol_df, vol_date_col, vol_cols):
    """
    Merge the option table with the volatility table using an as-of
    backward join on the date column.

    Each QUOTE_DATE is paired with the most recent volatility observation
    in effect on or before that date, so that no look-ahead bias occurs.
    """
    merged = pd.merge_asof(
        options,
        vol_df[[vol_date_col] + vol_cols].rename(
            columns={vol_date_col: "_vol_date"}
        ),
        left_on="QUOTE_DATE",
        right_on="_vol_date",
        direction="backward",
    )
    merged = merged.drop(columns=["_vol_date"])
    return merged


def fill_and_validate(df, vol_cols, label):
    """
    Fill remaining NaNs in the volatility columns (rows at the start of
    the dataset not yet covered by the backward merge) using bfill, then
    report the number of NaNs that remain.
    """
    df[vol_cols] = df[vol_cols].bfill()

    n_missing = df[vol_cols].isna().sum().sum()
    if n_missing > 0:
        print(f"  WARNING [{label}]: {n_missing:,} NaNs still remain "
              f"in the volatility columns after bfill.")
    else:
        print(f"  [{label}] NaNs in volatility columns: 0 (clean)")

    return df


def extract_tail(df, fraction):
    """
    Take the last 'fraction' of the rows based on the already-sorted
    QUOTE_DATE order.
    """
    n = int(len(df) * fraction)
    return df.tail(n).reset_index(drop=True)


# ============================================================
# 3. MAIN PIPELINE
# ============================================================

def main():
    # --- Load data ---
    print("--- Loading data ---")
    opt     = load_and_parse(INPUT_OPTIONS, "QUOTE_DATE")
    vol_rel = load_and_parse(INPUT_VOL_REL, "Date")
    vol_abs = load_and_parse(INPUT_VOL_ABS, "Date")

    print(f"Option rows         : {len(opt):,}")
    print(f"Relative vol obs.   : {len(vol_rel):,}")
    print(f"Absolute vol obs.   : {len(vol_abs):,}")

    # ----------------------------------------------------------
    # OUTPUT 1 — input_ANN.csv (options + relative volatility)
    # ----------------------------------------------------------
    print(f"\n--- Preparing {OUTPUT_ANN} ---")
    df_ann = asof_merge(opt.copy(), vol_rel, "Date", VOL_REL_COLS)
    df_ann = fill_and_validate(df_ann, VOL_REL_COLS, OUTPUT_ANN)

    os.makedirs(os.path.dirname(OUTPUT_ANN), exist_ok=True)
    df_ann.to_csv(OUTPUT_ANN, index=False)
    print(f"  Saved: {OUTPUT_ANN} | rows: {len(df_ann):,}")

    # ----------------------------------------------------------
    # OUTPUT 2 — input_BS_MC.csv (last 20% of input_ANN)
    # ----------------------------------------------------------
    print(f"\n--- Preparing {OUTPUT_BS_MC} ---")
    df_bsmc = extract_tail(df_ann, TEST_FRACTION)

    os.makedirs(os.path.dirname(OUTPUT_BS_MC), exist_ok=True)
    df_bsmc.to_csv(OUTPUT_BS_MC, index=False)
    print(f"  Saved: {OUTPUT_BS_MC} | rows: {len(df_bsmc):,} "
          f"({TEST_FRACTION:.0%} of {OUTPUT_ANN})")
    print(f"  QUOTE_DATE range: "
          f"{df_bsmc['QUOTE_DATE'].min().date()} to "
          f"{df_bsmc['QUOTE_DATE'].max().date()}")

    # ----------------------------------------------------------
    # OUTPUT 3 — input_Bachelier.csv (last 20% of options + absolute vol)
    # ----------------------------------------------------------
    print(f"\n--- Preparing {OUTPUT_BACHELIER} ---")
    df_bach = asof_merge(opt.copy(), vol_abs, "Date", VOL_ABS_COLS)
    df_bach = fill_and_validate(df_bach, VOL_ABS_COLS, OUTPUT_BACHELIER)

    df_bach = extract_tail(df_bach, TEST_FRACTION)

    os.makedirs(os.path.dirname(OUTPUT_BACHELIER), exist_ok=True)
    df_bach.to_csv(OUTPUT_BACHELIER, index=False)
    print(f"  Saved: {OUTPUT_BACHELIER} | rows: {len(df_bach):,} "
          f"(last {TEST_FRACTION:.0%})")
    print(f"  QUOTE_DATE range: "
          f"{df_bach['QUOTE_DATE'].min().date()} to "
          f"{df_bach['QUOTE_DATE'].max().date()}")

    # --- Summary ---
    print("\n=== Summary ===")
    print(f"  {OUTPUT_ANN:<36s} : {len(df_ann):>10,} rows")
    print(f"  {OUTPUT_BS_MC:<36s} : {len(df_bsmc):>10,} rows")
    print(f"  {OUTPUT_BACHELIER:<36s} : {len(df_bach):>10,} rows")
    print("\nDone.")


if __name__ == "__main__":
    main()