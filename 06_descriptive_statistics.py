"""
Option Dataset Descriptive Statistics Script
--------------------------------------------
Computes descriptive statistics (N, Min, Max, Mean, Median, Std. Dev.,
Skewness, Kurtosis) for all the main variables and volatility estimates
in the option dataset that will be used as modeling input.

The output is formatted for direct presentation in the data-description
chapter of the thesis.

Output:
    descriptive_statistics_options.csv
"""

import os
import pandas as pd
import numpy as np
from scipy import stats as sp_stats


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FILE  = os.path.join("data", "processed", "input_ANN.csv")
OUTPUT_FILE = os.path.join("outputs", "descriptive_statistics_options.csv")

# Optional metric toggles.
INCLUDE_N        = True
INCLUDE_MEDIAN   = True
INCLUDE_SKEWNESS = True
INCLUDE_KURTOSIS = True

# Variable definitions: (label for the table, column name in the dataset)
CORE_VARS = [
    ("Asset Price (S)",            "UNDERLYING_LAST"),
    ("Strike Price (K)",          "STRIKE"),
    ("Time to Maturity (T, days)", "TIME_TO_MATURITY"),
    ("Risk-Free Rate (r)",        "r"),
    ("Option Price (C)",          "C_LAST"),
]

VOL_VARS = [
    ("HV 5-Day",      "HV5"),
    ("HV 20-Day",     "HV20"),
    ("HV 60-Day",     "HV60"),
    ("HV 100-Day",    "HV100"),
    ("GARCH (1,1)",   "GARCH"),
    ("EGARCH (1,1)",  "EGARCH"),
]


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def desc_stats(series):
    """
    Compute descriptive statistics for a single variable.
    Infinite values are converted to NaN before computation.
    """
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()

    stats = {}
    if INCLUDE_N:
        stats["N"] = int(len(s))

    stats["Minimum"]   = s.min()
    stats["Maximum"]   = s.max()
    stats["Mean"]      = s.mean()

    if INCLUDE_MEDIAN:
        stats["Median"] = s.median()

    stats["Std. Dev."] = s.std(ddof=1)

    if INCLUDE_SKEWNESS:
        stats["Skewness"] = sp_stats.skew(s, bias=False)

    if INCLUDE_KURTOSIS:
        stats["Kurtosis"] = sp_stats.kurtosis(s, bias=False)

    return stats


# ============================================================
# 3. MAIN PIPELINE
# ============================================================

def main():
    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.strip()
    print(f"Rows: {len(df):,}")

    # Validate core columns.
    core_cols = [col for _, col in CORE_VARS]
    missing = [c for c in core_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Core columns not found: {missing}")

    # Build the statistics table.
    rows = []

    for label, col in CORE_VARS:
        stats = desc_stats(df[col])
        stats["Variable"] = label
        rows.append(stats)

    for label, col in VOL_VARS:
        if col in df.columns:
            stats = desc_stats(df[col])
            stats["Variable"] = label
            rows.append(stats)
        else:
            print(f"  Info: column '{col}' not available, skipped.")

    # Arrange column order.
    ordered = ["Variable"]
    if INCLUDE_N:
        ordered.append("N")
    ordered += ["Minimum", "Maximum", "Mean"]
    if INCLUDE_MEDIAN:
        ordered.append("Median")
    ordered.append("Std. Dev.")
    if INCLUDE_SKEWNESS:
        ordered.append("Skewness")
    if INCLUDE_KURTOSIS:
        ordered.append("Kurtosis")

    result = pd.DataFrame(rows)[ordered]

    # Save.
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False, float_format="%.6f")

    print(f"\nFile saved: {OUTPUT_FILE}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()