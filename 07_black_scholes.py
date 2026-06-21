"""
Call Option Pricing Script — Black–Scholes Model
------------------------------------------------
Computes theoretical European call option prices using the Black–Scholes
formula for each volatility estimate (HV5 … EGARCH), then evaluates
accuracy against the market price (C_LAST).

Evaluation is performed at two levels:
    1. Overall   — the entire sample.
    2. Moneyness — ITM / ATM / OTM based on the Varch (2019) definition:
                   ITM  : S/K > 1.05
                   ATM  : 0.97 ≤ S/K ≤ 1.05
                   OTM  : S/K < 0.97

Accuracy metrics:
    MAE, RMSE, MAPE (%)

Output:
    1. BS_theoretical_prices.csv    — theoretical prices per volatility
    2. BS_accuracy_overall.csv      — overall metrics
    3. BS_accuracy_by_moneyness.csv — metrics per moneyness category
"""

import os
import time
import pandas as pd
import numpy as np
from scipy.stats import norm


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FILE  = os.path.join("data", "processed", "input_BS_MC.csv")
OUTPUT_THEO = os.path.join("outputs", "Black-Scholes", "BS_theoretical_prices.csv")
OUTPUT_OVR  = os.path.join("outputs", "Black-Scholes", "BS_accuracy_overall.csv")
OUTPUT_MON  = os.path.join("outputs", "Black-Scholes", "BS_accuracy_by_moneyness.csv")

# Volatility columns used as the σ input.
VOL_COLS = ["HV5", "HV20", "HV60", "HV100", "GARCH", "EGARCH"]

# Moneyness definition — Varch (2019).
ITM_LOWER = 1.05      # S/K > 1.05
OTM_UPPER = 0.97      # S/K < 0.97
                      # ATM : 0.97 ≤ S/K ≤ 1.05

# Original columns retained in the theoretical-price output.
KEEP_COLS = [
    "QUOTE_DATE", "UNDERLYING_LAST", "STRIKE",
    "r", "TIME_TO_MATURITY", "C_LAST",
]


# ============================================================
# 2. BLACK–SCHOLES FORMULA
# ============================================================

def bs_call(S, K, r, T, sigma):
    """
    European call option price using the Black–Scholes formula.

    Parameters
    ----------
    S     : array-like   Underlying (spot) price.
    K     : array-like   Strike price.
    r     : array-like   Risk-free rate (decimal, annualized).
    T     : array-like   Time to maturity (years).
    sigma : array-like   Volatility (decimal, annualized).

    Returns
    -------
    np.ndarray  Theoretical call option price.
    """
    sigma = np.maximum(sigma, 1e-8)
    T     = np.maximum(T, 1e-8)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# ============================================================
# 3. EVALUATION FUNCTIONS
# ============================================================

def compute_metrics(predicted, actual):
    """
    Compute MAE, RMSE, and MAPE between theoretical and market prices.
    MAPE is computed only on rows with a market price > 0.
    """
    errors = predicted - actual
    abs_err = np.abs(errors)

    mae  = np.mean(abs_err)
    rmse = np.sqrt(np.mean(errors**2))

    # Guard: avoid division by zero in MAPE.
    mask = actual > 0
    if mask.sum() > 0:
        mape = np.mean(np.abs(errors[mask] / actual[mask])) * 100
    else:
        mape = np.nan

    return mae, rmse, mape


def classify_moneyness(S, K):
    """
    Classify moneyness based on the S/K ratio.
        ITM : S/K > ITM_LOWER
        OTM : S/K < OTM_UPPER
        ATM : otherwise
    """
    m = S / K
    return np.where(m > ITM_LOWER, "ITM",
           np.where(m < OTM_UPPER, "OTM", "ATM"))


# ============================================================
# 4. MAIN PIPELINE
# ============================================================

def main():
    start = time.time()

    # --- Load data ---
    print("--- Loading data ---")
    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.strip()
    df["T"] = df["TIME_TO_MATURITY"] / 365.0
    print(f"Rows: {len(df):,}")

    # Input vectors (computed once, reused).
    S     = df["UNDERLYING_LAST"].values
    K     = df["STRIKE"].values
    r     = df["r"].values
    T     = df["T"].values
    C_mkt = df["C_LAST"].values

    # --- Moneyness classification ---
    df["Moneyness"] = classify_moneyness(S, K)

    for cat in ["ITM", "ATM", "OTM"]:
        n = (df["Moneyness"] == cat).sum()
        print(f"  {cat}: {n:,} observations")

    # --- Compute theoretical prices (once for all volatilities) ---
    print("\n--- Computing Black–Scholes prices ---")
    output_prices = df[KEEP_COLS + ["Moneyness"]].copy()

    for vol in VOL_COLS:
        sigma = df[vol].values
        output_prices[f"BS_{vol}"] = bs_call(S, K, r, T, sigma)

    os.makedirs(os.path.dirname(OUTPUT_THEO), exist_ok=True)
    output_prices.to_csv(OUTPUT_THEO, index=False)
    print(f"Theoretical prices saved: {OUTPUT_THEO}")

    # --- Overall metrics ---
    print("\n--- Overall evaluation ---")
    rows_overall = []

    for vol in VOL_COLS:
        theo = output_prices[f"BS_{vol}"].values
        mae, rmse, mape = compute_metrics(theo, C_mkt)
        rows_overall.append([vol, mae, rmse, mape])
        print(f"  {vol:<8s}  MAE={mae:>10.4f}  RMSE={rmse:>10.4f}  MAPE={mape:>8.2f}%")

    df_overall = pd.DataFrame(
        rows_overall, columns=["Volatility", "MAE", "RMSE", "MAPE"]
    )
    os.makedirs(os.path.dirname(OUTPUT_OVR), exist_ok=True)
    df_overall.to_csv(OUTPUT_OVR, index=False)

    # --- Metrics per moneyness ---
    print("\n--- Evaluation per moneyness ---")
    rows_mon = []

    for cat in ["ITM", "ATM", "OTM"]:
        mask = df["Moneyness"].values == cat
        n_cat = mask.sum()
        if n_cat == 0:
            continue

        C_sub = C_mkt[mask]
        print(f"\n  [{cat}] n = {n_cat:,}")

        for vol in VOL_COLS:
            theo_sub = output_prices[f"BS_{vol}"].values[mask]
            mae, rmse, mape = compute_metrics(theo_sub, C_sub)
            rows_mon.append([cat, vol, n_cat, mae, rmse, mape])
            print(f"    {vol:<8s}  MAE={mae:>10.4f}  "
                  f"RMSE={rmse:>10.4f}  MAPE={mape:>8.2f}%")

    df_mon = pd.DataFrame(
        rows_mon,
        columns=["Moneyness", "Volatility", "N", "MAE", "RMSE", "MAPE"],
    )
    os.makedirs(os.path.dirname(OUTPUT_MON), exist_ok=True)
    df_mon.to_csv(OUTPUT_MON, index=False)

    # --- Summary ---
    elapsed = time.time() - start
    print(f"\n=== Done ({elapsed:.2f} s) ===")
    print(f"  {OUTPUT_THEO}")
    print(f"  {OUTPUT_OVR}")
    print(f"  {OUTPUT_MON}")


if __name__ == "__main__":
    main()