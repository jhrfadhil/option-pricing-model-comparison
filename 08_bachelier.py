"""
Call Option Pricing Script — Bachelier Model
--------------------------------------------
Computes theoretical call option prices using the Bachelier model
(normal model) for each absolute volatility estimate
(HV5_abs … EGARCH_abs), then evaluates accuracy against the market
price (C_LAST).

The Bachelier model assumes the underlying price follows arithmetic
Brownian motion (rather than geometric, as in Black–Scholes), so
volatility is measured in absolute price units per √year.

Call formula:
    C = e^(−rT) [ (F − K) Φ(d) + σ_T φ(d) ]
    where F = S·e^(rT),  σ_T = σ_abs · √T,  d = (F − K) / σ_T

Evaluation is performed at two levels:
    1. Overall   — the entire sample.
    2. Moneyness — ITM / ATM / OTM (Varch, 2019 definition).

Output:
    1. Bachelier_theoretical_prices.csv    — theoretical prices per volatility
    2. Bachelier_accuracy_overall.csv      — overall metrics
    3. Bachelier_accuracy_by_moneyness.csv — metrics per moneyness category
"""

import os
import time
import pandas as pd
import numpy as np
from scipy.stats import norm


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FILE  = os.path.join("data", "processed", "input_Bachelier.csv")
OUTPUT_THEO = os.path.join("outputs", "Bachelier", "Bachelier_theoretical_prices.csv")
OUTPUT_OVR  = os.path.join("outputs", "Bachelier", "Bachelier_accuracy_overall.csv")
OUTPUT_MON  = os.path.join("outputs", "Bachelier", "Bachelier_accuracy_by_moneyness.csv")

# Mapping of absolute volatility columns -> theoretical-price columns.
VOL_MAP = {
    "HV5_abs":    "Bach_HV5",
    "HV20_abs":   "Bach_HV20",
    "HV60_abs":   "Bach_HV60",
    "HV100_abs":  "Bach_HV100",
    "GARCH_abs":  "Bach_GARCH",
    "EGARCH_abs": "Bach_EGARCH",
}

# Moneyness definition — Varch (2019).
ITM_LOWER = 1.05
OTM_UPPER = 0.97

# Original columns retained in the theoretical-price output.
KEEP_COLS = [
    "QUOTE_DATE", "UNDERLYING_LAST", "STRIKE",
    "r", "TIME_TO_MATURITY", "C_LAST",
]


# ============================================================
# 2. BACHELIER FORMULA
# ============================================================

def bachelier_call(S, K, r, T, sigma_abs):
    """
    Call option price using the Bachelier model (normal model).

    Parameters
    ----------
    S         : array-like  Underlying (spot) price.
    K         : array-like  Strike price.
    r         : array-like  Risk-free rate (decimal, annualized).
    T         : array-like  Time to maturity (years).
    sigma_abs : array-like  Absolute volatility (price units, annualized).

    Returns
    -------
    np.ndarray  Theoretical call option price.
    """
    T = np.maximum(T, 1e-8)

    # Forward price and discount factor.
    F    = S * np.exp(r * T)
    disc = np.exp(-r * T)

    # Total volatility to maturity.
    sigma_T = sigma_abs * np.sqrt(T)

    # d-score (zero-division guard).
    d = np.where(sigma_T > 0, (F - K) / sigma_T, 0.0)

    price = disc * ((F - K) * norm.cdf(d) + sigma_T * norm.pdf(d))

    # Fallback: if T ≈ 0, use the intrinsic value.
    intrinsic = np.maximum(S - K, 0.0)
    price = np.where(T < 1e-6, intrinsic, price)

    return price


# ============================================================
# 3. EVALUATION FUNCTIONS (identical to the Black–Scholes script)
# ============================================================

def compute_metrics(predicted, actual):
    """
    Compute MAE, RMSE, and MAPE between theoretical and market prices.
    MAPE is computed only on rows with a market price > 0.
    """
    errors  = predicted - actual
    abs_err = np.abs(errors)

    mae  = np.mean(abs_err)
    rmse = np.sqrt(np.mean(errors**2))

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

    # Validate columns.
    required = list(VOL_MAP.keys()) + [
        "UNDERLYING_LAST", "STRIKE", "C_LAST", "r", "TIME_TO_MATURITY",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {INPUT_FILE}: {missing}")

    df["T"] = (df["TIME_TO_MATURITY"] / 365.0).clip(lower=1e-8)
    print(f"Rows: {len(df):,}")

    # Input vectors.
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

    # --- Compute theoretical prices ---
    print("\n--- Computing Bachelier prices ---")
    output_prices = df[KEEP_COLS + ["Moneyness"]].copy()

    for vcol, outcol in VOL_MAP.items():
        sigma_abs = df[vcol].values
        output_prices[outcol] = bachelier_call(S, K, r, T, sigma_abs)

    os.makedirs(os.path.dirname(OUTPUT_THEO), exist_ok=True)
    output_prices.to_csv(OUTPUT_THEO, index=False)
    print(f"Theoretical prices saved: {OUTPUT_THEO}")

    # --- Overall metrics ---
    print("\n--- Overall evaluation ---")
    rows_overall = []

    for vcol, outcol in VOL_MAP.items():
        theo = output_prices[outcol].values
        mae, rmse, mape = compute_metrics(theo, C_mkt)
        rows_overall.append([vcol, mae, rmse, mape])
        print(f"  {vcol:<12s}  MAE={mae:>10.4f}  "
              f"RMSE={rmse:>10.4f}  MAPE={mape:>8.2f}%")

    df_overall = pd.DataFrame(
        rows_overall, columns=["Volatility", "MAE", "RMSE", "MAPE"]
    )
    os.makedirs(os.path.dirname(OUTPUT_OVR), exist_ok=True)
    df_overall.to_csv(OUTPUT_OVR, index=False)

    # --- Metrics per moneyness ---
    print("\n--- Evaluation per moneyness ---")
    rows_mon = []

    for cat in ["ITM", "ATM", "OTM"]:
        mask  = df["Moneyness"].values == cat
        n_cat = mask.sum()
        if n_cat == 0:
            continue

        C_sub = C_mkt[mask]
        print(f"\n  [{cat}] n = {n_cat:,}")

        for vcol, outcol in VOL_MAP.items():
            theo_sub = output_prices[outcol].values[mask]
            mae, rmse, mape = compute_metrics(theo_sub, C_sub)
            rows_mon.append([cat, vcol, n_cat, mae, rmse, mape])
            print(f"    {vcol:<12s}  MAE={mae:>10.4f}  "
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