"""
SPX Volatility Estimation Script (Relative & Absolute)
------------------------------------------------------
Computes various volatility measures from S&P 500 historical price data
for use as inputs to the option-pricing models.

Relative volatility (annualized):
    - Historical Volatility : HV5, HV20, HV60, HV100
      (standard deviation of log-returns over rolling windows of
       5/20/60/100 days)
    - GARCH(1,1)            : rolling 1-step-ahead forecast
    - EGARCH(1,1)           : rolling 1-step-ahead forecast

Absolute volatility (for the Bachelier model):
    σ_abs = σ_rel × S

Output:
    1. SPX_volatility_forecast.csv          — relative volatility + price
    2. SPX_absolute_volatility_forecast.csv — absolute volatility
"""

import os
import warnings
import pandas as pd
import numpy as np
from arch import arch_model
from tqdm import tqdm


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_PRICE  = os.path.join("data", "raw", "S&P 500 Historical Data.csv")
OUTPUT_REL   = os.path.join("data", "processed", "SPX_volatility_forecast.csv")
OUTPUT_ABS   = os.path.join("data", "processed", "SPX_absolute_volatility_forecast.csv")

# Historical volatility parameters (rolling window in trading days).
HV_WINDOWS = [5, 20, 60, 100]

# GARCH / EGARCH parameters.
GARCH_WINDOW = 1000          # length of the rolling estimation window
TRADING_DAYS = 252           # number of trading days per year

# Volatility columns to be computed.
VOL_COLS = [f"HV{w}" for w in HV_WINDOWS] + ["GARCH", "EGARCH"]


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def load_price_data(path):
    """
    Load S&P 500 historical price data. The 'Price' column is cleaned of
    thousands separators (commas) and converted to float.
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"])

    # Clean the price format: "4,769.83" -> 4769.83
    df["Price"] = (
        df["Price"]
          .astype(str)
          .str.strip()
          .str.replace(",", "", regex=False)
    )
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    df = df[["Date", "Price"]].dropna()
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def compute_log_returns(df):
    """Compute daily log-returns and drop the first row (NaN)."""
    df["log_ret"] = np.log(df["Price"] / df["Price"].shift(1))
    df = df.dropna(subset=["log_ret"]).reset_index(drop=True)
    return df


# ============================================================
# 3. HISTORICAL VOLATILITY
# ============================================================

def compute_historical_vol(df, windows):
    """
    Compute historical volatility (annualized) for each given
    rolling-window size.

    Formula: HV = std(log_ret, window) × √252
    """
    for w in windows:
        col_name = f"HV{w}"
        df[col_name] = df["log_ret"].rolling(w).std() * np.sqrt(TRADING_DAYS)
    return df


# ============================================================
# 4. ROLLING GARCH & EGARCH FORECAST
# ============================================================

def rolling_garch_egarch(returns, window_size):
    """
    Estimate 1-step-ahead volatility using GARCH(1,1) and EGARCH(1,1) on
    a rolling window.

    Each iteration fits the model on the last 'window_size' observations,
    then forecasts the variance one day ahead. The forecast is converted
    from daily variance (in %²) to annualized volatility (decimal).

    If estimation fails to converge on a given window, the value is
    replaced with NaN and reported at the end of the process.

    Parameters
    ----------
    returns     : pd.Series   Daily log-returns (decimal, not percent).
    window_size : int         Number of observations per estimation window.

    Returns
    -------
    garch_vals, egarch_vals : list[float]
    """
    n = len(returns)
    if n < window_size:
        raise ValueError(
            f"Data ({n:,} rows) is insufficient for window_size={window_size:,}."
        )

    returns_pct = returns * 100     # the arch library works in percent

    garch_vals  = [np.nan] * window_size
    egarch_vals = [np.nan] * window_size
    n_fail_g, n_fail_e = 0, 0

    print(f"Rolling forecast started "
          f"(data = {n:,}, window = {window_size:,}, "
          f"iterations = {n - window_size:,})")

    # Suppress convergence warnings to keep the log clean.
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings(
        "ignore",
        message=".*ConvergenceWarning.*|.*convergence.*",
        category=UserWarning,
    )

    for t in tqdm(range(window_size, n), desc="GARCH/EGARCH"):
        past = returns_pct.iloc[t - window_size : t]

        # --- GARCH(1,1) ---
        try:
            g_model = arch_model(past, vol="Garch", p=1, q=1, dist="normal")
            g_res   = g_model.fit(disp="off", show_warning=False)
            g_var   = g_res.forecast(horizon=1).variance.values[-1, 0]
            garch_vals.append(np.sqrt(g_var) / 100.0 * np.sqrt(TRADING_DAYS))
        except Exception:
            garch_vals.append(np.nan)
            n_fail_g += 1

        # --- EGARCH(1,1) ---
        try:
            e_model = arch_model(past, vol="EGarch", p=1, q=1, dist="normal")
            e_res   = e_model.fit(disp="off", show_warning=False)
            e_var   = e_res.forecast(horizon=1).variance.values[-1, 0]
            egarch_vals.append(np.sqrt(e_var) / 100.0 * np.sqrt(TRADING_DAYS))
        except Exception:
            egarch_vals.append(np.nan)
            n_fail_e += 1

    # Reset warning filters.
    warnings.resetwarnings()

    if n_fail_g > 0 or n_fail_e > 0:
        print(f"Convergence warning — "
              f"GARCH failures: {n_fail_g:,}, EGARCH failures: {n_fail_e:,} "
              f"(filled with NaN).")

    return garch_vals, egarch_vals


# ============================================================
# 5. ABSOLUTE VOLATILITY (BACHELIER)
# ============================================================

def compute_absolute_vol(df, vol_cols):
    """
    Convert relative volatility to absolute volatility:
        σ_abs = σ_rel × S
    Used as input to the Bachelier model.
    """
    for col in vol_cols:
        df[f"{col}_abs"] = df[col] * df["Price"]
    return df


# ============================================================
# 6. MAIN PIPELINE
# ============================================================

def main():
    # --- Load and clean price data ---
    print("--- Loading price data ---")
    df = load_price_data(INPUT_PRICE)
    print(f"Price observations  : {len(df):,}")

    # --- Log-returns ---
    df = compute_log_returns(df)
    print(f"Log-return obs.     : {len(df):,}")

    # --- Historical volatility ---
    print("\n--- Computing historical volatility ---")
    df = compute_historical_vol(df, HV_WINDOWS)

    # --- Rolling GARCH & EGARCH ---
    print("\n--- Computing GARCH(1,1) & EGARCH(1,1) ---")
    garch_vals, egarch_vals = rolling_garch_egarch(df["log_ret"], GARCH_WINDOW)
    df["GARCH"]  = garch_vals
    df["EGARCH"] = egarch_vals

    # --- Save relative volatility ---
    out_rel = df[["Date", "Price"] + VOL_COLS].copy()
    os.makedirs(os.path.dirname(OUTPUT_REL), exist_ok=True)
    out_rel.to_csv(OUTPUT_REL, index=False)
    print(f"\nRelative volatility saved : {OUTPUT_REL}")

    # --- Absolute volatility ---
    print("\n--- Computing absolute volatility (σ_abs = σ_rel × S) ---")
    df = compute_absolute_vol(df, VOL_COLS)

    abs_cols = [f"{c}_abs" for c in VOL_COLS]
    out_abs = df[["Date"] + abs_cols].copy()
    os.makedirs(os.path.dirname(OUTPUT_ABS), exist_ok=True)
    out_abs.to_csv(OUTPUT_ABS, index=False)
    print(f"Absolute volatility saved : {OUTPUT_ABS}")

    # --- Summary ---
    print("\n=== Summary ===")
    print(f"Date range          : {df['Date'].min().date()} to "
          f"{df['Date'].max().date()}")
    print(f"Total observations  : {len(df):,}")
    print(f"NaN GARCH           : {df['GARCH'].isna().sum():,}")
    print(f"NaN EGARCH          : {df['EGARCH'].isna().sum():,}")
    print(f"\nFinal snapshot (relative):")
    print(out_rel.tail(5).to_string(index=False))
    print(f"\nFinal snapshot (absolute):")
    print(out_abs.tail(5).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()