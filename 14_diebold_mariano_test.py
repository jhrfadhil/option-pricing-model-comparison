"""
Diebold-Mariano Test with Harvey-Leybourne-Newbold Correction
-------------------------------------------------------------
Combines the theoretical prices from four models (ANN, Bachelier,
Black–Scholes, Monte Carlo MJD) into a single dataset, then runs
the pairwise Diebold-Mariano (DM) test to assess the significance
of differences in predictive accuracy between models.

Test specification:
    - Loss function     : Squared Error (SE)
    - Aggregation       : mean SE per day (QUOTE_DATE)
    - Variance          : Newey-West HAC with Bartlett weights
    - Lag rule          : T^(1/3) (default)
    - Sample correction : Harvey-Leybourne-Newbold (HLN)
    - Distribution      : Student-t (df = T − 1)
    - Horizon           : h = 1

Output:
    DM_results_squared_HLN.csv — DM statistic, p-value, and the
    best model for each comparison pair.
"""

import os
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import t as student_t


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Monte Carlo simulation-path count whose output is compared here.
# Script 09 writes MCJD_theoretical_prices.csv under
# outputs/Monte Carlo/<N_PATHS>/ -> change this to compare a different run.
MCJD_N_PATHS = 500000

# Candidate price-column names for the ANN entry. Scripts 10-13 each write a
# single column named after the architecture (MLP_price / Residual_price /
# MoE_price / KAN_price); the loader auto-detects whichever one is present, so
# switching ANN architecture only requires changing the ANN path below.
ANN_PRICE_CANDIDATES = ["MLP_price", "Residual_price", "MoE_price", "KAN_price"]

# Input: theoretical-price file for each model. The classical-model columns
# (Bach/BS/MCJD) are fixed strings; the ANN column is resolved from
# ANN_PRICE_CANDIDATES at load time.
INPUT_FILES = {
    "ANN":  (os.path.join("outputs", "ANN", "MoE", "MoE_theoretical_prices.csv"),                      ANN_PRICE_CANDIDATES),
    "Bach": (os.path.join("outputs", "Bachelier", "Bachelier_theoretical_prices.csv"),                 "Bach_EGARCH"),
    "BS":   (os.path.join("outputs", "Black-Scholes", "BS_theoretical_prices.csv"),                    "BS_EGARCH"),
    "MCJD": (os.path.join("outputs", "Monte Carlo", str(MCJD_N_PATHS), "MCJD_theoretical_prices.csv"), "MCJD_EGARCH"),
}

# Output
OUTPUT_FILE = os.path.join("outputs", "DM_results_squared_HLN.csv")

# Key columns for merging across models
MERGE_KEYS = ["QUOTE_DATE", "UNDERLYING_LAST", "STRIKE", "TIME_TO_MATURITY"]

# Fraction of the most recent data used as the DM test set
# (applied only to the ANN file, which contains the full dataset;
#  the BS/Bachelier/MCJD files already contain the last 20%)
ANN_TAIL_FRAC = 0.20

# Diebold-Mariano test parameters
DM_HORIZON   = 1         # forecast horizon (1-step ahead)
NW_LAG_RULE  = "T13"     # "T13" = T^(1/3)


# ============================================================
# 2. DATA PREPARATION
# ============================================================

def load_model_prices(files_config):
    """
    Load each model's theoretical-price file, resolve and validate
    its price column, and return {model_name: (DataFrame, price_col)}.

    The price spec is either a fixed column name (Bach/BS/MCJD) or a
    list of candidate names (ANN); a candidate list is resolved to the
    single name present in the file.
    """
    frames = {}

    for model_name, (filepath, price_spec) in files_config.items():
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()

        # Resolve the price column. A plain string is used as-is; a list of
        # candidates (the ANN entry) is resolved to whichever one is present,
        # so the ANN architecture can be switched by changing the path only.
        if isinstance(price_spec, str):
            price_col = price_spec
        else:
            present = [c for c in price_spec if c in df.columns]
            if len(present) != 1:
                raise ValueError(
                    f"Expected exactly one of {list(price_spec)} in "
                    f"{filepath}, found {present}"
                )
            price_col = present[0]

        # Validate columns
        required = MERGE_KEYS + ["C_LAST", price_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing columns in {filepath}: {missing}"
            )

        df["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"])

        # ANN contains the full dataset -> take the last 20%
        if model_name == "ANN":
            tail_n = int(len(df) * ANN_TAIL_FRAC)
            df = df.tail(tail_n).copy()

        # Keep only the required columns
        keep = MERGE_KEYS + ["C_LAST", price_col]
        df = df[keep].sort_values("QUOTE_DATE").reset_index(drop=True)

        frames[model_name] = (df, price_col)
        print(f"  {model_name:<5s}: {len(df):>10,} rows  [{price_col}]  ({filepath})")

    return frames


def merge_all_models(frames):
    """
    Merge all models' theoretical prices into a single DataFrame
    using an inner join on MERGE_KEYS.
    """
    model_names = list(frames.keys())

    # Start from the first model (which carries C_LAST)
    first_name = model_names[0]
    first_df, first_col = frames[first_name]
    master = first_df.rename(columns={first_col: first_col})

    for name in model_names[1:]:
        df_m, col_m = frames[name]
        # Drop C_LAST from subsequent models to avoid duplication
        cols_to_merge = MERGE_KEYS + [col_m]
        master = master.merge(df_m[cols_to_merge], on=MERGE_KEYS, how="inner")

    # Collect the model price-column names
    price_cols = [frames[m][1] for m in model_names]

    # Report
    n_nan = master[price_cols].isna().sum().sum()
    print(f"\nMerge result: {len(master):,} rows")
    if n_nan > 0:
        print(f"  WARNING: {n_nan:,} NaN in model price columns.")
    else:
        print(f"  NaN in price columns: 0 (clean)")

    return master, price_cols


# ============================================================
# 3. DIEBOLD-MARIANO TEST
# ============================================================

def choose_nw_lag(T, rule="T13"):
    """
    Compute the maximum Newey-West lag from a rule-of-thumb.
        T13 : floor(T^(1/3))
    """
    if rule.upper() == "T13":
        return int(np.floor(T ** (1 / 3)))
    return int(np.floor(T ** 0.25))


def newey_west_var(d, max_lag):
    """
    Newey-West HAC variance estimator for the mean of series d.
    Uses Bartlett kernel weights: w_k = 1 − k/(L+1).

    Returns: γ̂₀ + 2 Σ w_k γ̂_k  (not yet divided by T).
    """
    d = np.asarray(d, dtype=np.float64)
    T = d.size
    u = d - d.mean()

    gamma0  = np.dot(u, u) / T
    var_hat = gamma0

    for k in range(1, max_lag + 1):
        gamma_k = np.dot(u[k:], u[:-k]) / T
        w_k = 1.0 - k / (max_lag + 1.0)
        var_hat += 2.0 * w_k * gamma_k

    return var_hat


def dm_test(loss1, loss2, h=1, lag_rule="T13"):
    """
    Two-sided Diebold-Mariano test with HLN correction.

    Parameters
    ----------
    loss1, loss2 : array (T,)  Mean daily loss per model.
    h            : int          Forecast horizon.
    lag_rule     : str          Rule-of-thumb for the Newey-West lag.

    Returns
    -------
    dm_stat, p_value, d_bar, max_lag
    """
    loss1 = np.asarray(loss1, dtype=np.float64)
    loss2 = np.asarray(loss2, dtype=np.float64)

    d     = loss1 - loss2
    T     = d.size
    d_bar = d.mean()

    max_lag = max(0, choose_nw_lag(T, rule=lag_rule))
    var_hat = newey_west_var(d, max_lag)

    # DM statistic
    dm = d_bar / np.sqrt(var_hat / T)

    # Harvey-Leybourne-Newbold correction (small-sample)
    h = int(h)
    hln_factor = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm *= hln_factor

    # Two-sided p-value (t-distribution, df = T − 1)
    p_value = 2.0 * (1.0 - student_t.cdf(np.abs(dm), df=T - 1))

    return dm, p_value, d_bar, max_lag


# ============================================================
# 4. MAIN PIPELINE
# ============================================================

def main():
    print("=== Diebold-Mariano test (SE + Newey-West + HLN) ===\n")

    # --- Load and merge theoretical prices ---
    print("--- Loading theoretical prices ---")
    frames = load_model_prices(INPUT_FILES)
    master, price_cols = merge_all_models(frames)

    y = master["C_LAST"].to_numpy(dtype=np.float64)

    # --- Compute Squared Error per model ---
    model_names = list(INPUT_FILES.keys())
    se_cols = {}

    for name in model_names:
        _, col = frames[name]          # resolved price column (ANN auto-detected)
        se_col = f"SE_{name}"
        master[se_col] = (y - master[col].to_numpy(dtype=np.float64)) ** 2
        se_cols[name] = se_col

    # --- Aggregate to daily loss (mean SE per QUOTE_DATE) ---
    agg_dict = {se: "mean" for se in se_cols.values()}
    daily_loss = master.groupby("QUOTE_DATE").agg(agg_dict).sort_index()

    T = len(daily_loss)
    print(f"\nUnique days (T): {T:,}")
    print(f"Newey-West lag rule : {NW_LAG_RULE}")
    print(f"DM horizon          : {DM_HORIZON}")

    # --- Run the DM test for all pairs ---
    print("\n--- DM test results ---")
    results = []

    for (m1, _), (m2, _) in combinations(INPUT_FILES.items(), 2):
        l1 = daily_loss[se_cols[m1]].to_numpy()
        l2 = daily_loss[se_cols[m2]].to_numpy()

        dm_stat, p_val, d_bar, used_lag = dm_test(
            l1, l2, h=DM_HORIZON, lag_rule=NW_LAG_RULE
        )

        mean_l1 = l1.mean()
        mean_l2 = l2.mean()

        if mean_l1 < mean_l2:
            better = m1
        elif mean_l2 < mean_l1:
            better = m2
        else:
            better = "Tie"

        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""

        results.append([
            m1, m2, T, mean_l1, mean_l2, d_bar,
            used_lag, dm_stat, p_val, better, sig,
        ])

        print(f"  {m1} vs {m2}:  DM = {dm_stat:>8.4f}  "
              f"p = {p_val:.4f} {sig:<3s}  best = {better}")

    # --- Save ---
    res_df = pd.DataFrame(results, columns=[
        "Model_1", "Model_2", "N_days",
        "Mean_SE_1", "Mean_SE_2", "Mean_Diff_d_bar",
        "NW_lag", "DM_HLN_stat", "p_value",
        "Lower_Loss", "Signif",
    ])
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    res_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nFile saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()