"""
Granular Analysis Script by Moneyness x Maturity & Volatility Regime
--------------------------------------------------------------------
Produces in-depth analysis tables to evaluate the performance of
the four option-pricing models at a granular level.

Moneyness classification follows Gross et al. (2025), per Chapter III:
    OTM  : S/K < 0.97
    ATM  : 0.97 <= S/K <= 1.05
    ITM  : S/K > 1.05

Days-to-expiration categories:
    Short   : < 60 days
    Medium  : 60-180 days
    Long    : >= 180 days

Volatility regime (based on annualized EGARCH):
    Low     : lower tertile  (<= P33)
    Medium  : middle tertile (P33-P67)
    High    : upper tertile  (> P67)

Output:
    1. table1_sample_properties.csv
       Mean option price and observation count per category.

    2. table2_vol_regime_performance.csv
       MAE, RMSE, MAPE per model per EGARCH volatility regime.

    3. table3_performance_by_category.csv
       MAE, RMSE, MAPE per model per moneyness x TTM category.

    4. table4_DM_by_category.csv
       Diebold-Mariano statistic per moneyness x TTM category
       for each model pair.
"""

import os
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import t as student_t


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Monte Carlo simulation-path count whose output is analysed here
# (script 09 writes under outputs/Monte Carlo/<N_PATHS>/).
MCJD_N_PATHS = 500000

# Candidate price-column names for the ANN entry. Scripts 10-13 each write a
# single column named after the architecture (MLP_price / Residual_price /
# MoE_price / KAN_price); the loader auto-detects whichever one is present,
# so switching ANN architecture only requires changing the ANN path below.
ANN_PRICE_CANDIDATES = ["MLP_price", "Residual_price", "MoE_price", "KAN_price"]

# Input: theoretical-price file for each model. Classical-model columns are
# fixed strings; the ANN column is resolved from ANN_PRICE_CANDIDATES.
INPUT_FILES = {
    "ANN":  (os.path.join("outputs", "ANN", "MoE", "MoE_theoretical_prices.csv"),                      ANN_PRICE_CANDIDATES),
    "BS":   (os.path.join("outputs", "Black-Scholes", "BS_theoretical_prices.csv"),                    "BS_EGARCH"),
    "Bach": (os.path.join("outputs", "Bachelier", "Bachelier_theoretical_prices.csv"),                 "Bach_EGARCH"),
    "MCJD": (os.path.join("outputs", "Monte Carlo", str(MCJD_N_PATHS), "MCJD_theoretical_prices.csv"), "MCJD_EGARCH"),
}

# Volatility file (for the EGARCH column as the regime basis)
INPUT_VOL = os.path.join("data", "processed", "SPX_volatility_forecast.csv")

# Filtered options file (for Table 1: the full sample)
INPUT_OPTIONS_FULL = os.path.join("data", "processed", "input_ANN.csv")

# Tail fraction for the test set (consistent with the previous scripts)
ANN_TAIL_FRAC = 0.20

# Output
OUT_TABLE1   = os.path.join("outputs", "table1_sample_properties.csv")
OUT_TABLE3   = os.path.join("outputs", "table3_performance_by_category.csv")
OUT_TABLE4   = os.path.join("outputs", "table4_DM_by_category.csv")
OUT_REGIME   = os.path.join("outputs", "table2_vol_regime_performance.csv")

# Merge keys
MERGE_KEYS = ["QUOTE_DATE", "UNDERLYING_LAST", "STRIKE", "TIME_TO_MATURITY"]

# --- Moneyness categories (S/K) following Gross et al. (2025) ---
# With right=False, the intervals become:
#   OTM = [0, 0.97)
#   ATM = [0.97, 1.05)
#   ITM = [1.05, inf)
# Total must be 3,417,592.
MONEYNESS_LABELS = ["OTM", "ATM", "ITM"]

# Target expectation for the Table 1 sanity check.
EXPECTED_MONEYNESS_COUNTS = {
    "OTM": 1_390_656,
    "ATM": 1_501_935,
    "ITM":   525_001,
}

# --- Days-to-expiration categories ---
TTM_BINS   = [-np.inf, 60, 180, np.inf]
TTM_LABELS = ["< 60", "60-180", ">= 180"]

# --- DM parameters ---
DM_HORIZON  = 1
NW_LAG_RULE = "T13"


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def classify(df):
    """Add moneyness and TTM category columns to the DataFrame."""
    m = df["UNDERLYING_LAST"] / df["STRIKE"]
    
    conditions = [
        m < 0.97,                       # OTM
        (m >= 0.97) & (m <= 1.05),      # ATM
        m > 1.05                        # ITM
    ]
    
    df["M_cat"] = np.select(conditions, MONEYNESS_LABELS, default="Unknown")
    df["M_cat"] = pd.Categorical(df["M_cat"], categories=MONEYNESS_LABELS, ordered=True)
    df["TTM_cat"] = pd.cut(
        df["TIME_TO_MATURITY"], bins=TTM_BINS, labels=TTM_LABELS, right=False
    )
    
    return df


def compute_metrics(pred, actual):
    """MAE, RMSE, MAPE. MAPE only on actual > 0."""
    err = pred - actual
    mae  = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    mask = actual > 0
    mape = (np.mean(np.abs(err[mask] / actual[mask])) * 100
            if mask.sum() > 0 else np.nan)
    return mae, rmse, mape


# ---------- Diebold-Mariano ----------

def nw_lag(T, rule="T13"):
    if rule.upper() == "T13":
        return int(np.floor(T ** (1 / 3)))
    return int(np.floor(T ** 0.25))


def newey_west_var(d, max_lag):
    T = d.size
    u = d - d.mean()
    gamma0 = np.dot(u, u) / T
    v = gamma0
    for k in range(1, max_lag + 1):
        gk = np.dot(u[k:], u[:-k]) / T
        v += 2.0 * (1.0 - k / (max_lag + 1.0)) * gk
    return v


def dm_test(l1, l2, h=1, rule="T13"):
    """DM + HLN, two-sided, t-distribution."""
    d     = l1 - l2
    T     = d.size
    d_bar = d.mean()
    lag   = max(0, nw_lag(T, rule))
    v     = newey_west_var(d, lag)
    dm    = d_bar / np.sqrt(v / T)
    hln   = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm   *= hln
    pval  = 2.0 * (1.0 - student_t.cdf(np.abs(dm), df=T - 1))
    return dm, pval


# ============================================================
# 3. TABLE 1 - SAMPLE PROPERTIES
# ============================================================

def build_table1():
    """
    Mean option price (C_LAST) and observation count per
    moneyness x days-to-expiration category, over the full sample.

    Sanity check at the end: the per-moneyness subtotals must match
    (OTM 1,390,656; ATM 1,501,935; ITM 525,001; Total 3,417,592).
    """
    print("\n=== Table 1: Sample Properties ===")
    df = pd.read_csv(INPUT_OPTIONS_FULL)
    df.columns = df.columns.str.strip()
    df["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"])
    df = classify(df)

    rows = []
    for m_cat in MONEYNESS_LABELS:
        for t_cat in TTM_LABELS:
            sub = df[(df["M_cat"] == m_cat) & (df["TTM_cat"] == t_cat)]
            n = len(sub)
            avg_price = sub["C_LAST"].mean() if n > 0 else np.nan
            rows.append([m_cat, t_cat, n, avg_price])

        # Subtotal per moneyness
        sub_all = df[df["M_cat"] == m_cat]
        rows.append([m_cat, "Subtotal", len(sub_all),
                     sub_all["C_LAST"].mean() if len(sub_all) > 0 else np.nan])

    # Grand total
    rows.append(["Total", "Total", len(df), df["C_LAST"].mean()])

    out = pd.DataFrame(rows, columns=[
        "Moneyness", "Days_to_Exp", "N", "Mean_Price"
    ])
    os.makedirs(os.path.dirname(OUT_TABLE1), exist_ok=True)
    out.to_csv(OUT_TABLE1, index=False, float_format="%.2f")
    print(f"  Saved: {OUT_TABLE1} ({len(df):,} total observations)")

    # ------- Sanity check -------
    print("\n  Verification against the Proposal:")
    print(f"  {'Category':<6s} {'Script':>12s} {'Expected':>12s} {'Match':>8s}")
    all_ok = True
    for m_cat, expected in EXPECTED_MONEYNESS_COUNTS.items():
        actual = int(out.loc[
            (out["Moneyness"] == m_cat) & (out["Days_to_Exp"] == "Subtotal"),
            "N"
        ].values[0])
        ok = (actual == expected)
        all_ok &= ok
        print(f"  {m_cat:<6s} {actual:>12,d} {expected:>12,d} "
              f"{'OK' if ok else 'DIFF':>8s}")
    if not all_ok:
        print("  WARNING: discrepancy. "
              "Check input_ANN.csv or the moneyness bins.")
    return out


# ============================================================
# 4. LOAD & MERGE THEORETICAL PRICES (TEST SET)
# ============================================================

def load_test_set():
    """
    Load and merge the four models' theoretical prices (test set).

    The ANN price column is resolved from ANN_PRICE_CANDIDATES, so the
    ANN architecture can be switched by changing its path only.

    Returns the merged DataFrame and {model_name: resolved_price_col}.
    """
    print("\n--- Loading theoretical prices (test set) ---")
    frames = {}
    price_map = {}

    for name, (fpath, pspec) in INPUT_FILES.items():
        d = pd.read_csv(fpath)
        d.columns = d.columns.str.strip()
        d["QUOTE_DATE"] = pd.to_datetime(d["QUOTE_DATE"])

        # Resolve the price column: a string is used as-is; a candidate
        # list (ANN) is resolved to whichever one is present in the file.
        if isinstance(pspec, str):
            pcol = pspec
        else:
            present = [c for c in pspec if c in d.columns]
            if len(present) != 1:
                raise ValueError(
                    f"Expected exactly one of {list(pspec)} in "
                    f"{fpath}, found {present}"
                )
            pcol = present[0]

        if name == "ANN":
            tail_n = int(len(d) * ANN_TAIL_FRAC)
            d = d.tail(tail_n).copy()

        d = d[MERGE_KEYS + ["C_LAST", pcol]].sort_values("QUOTE_DATE")
        frames[name] = (d, pcol)
        price_map[name] = pcol
        print(f"  {name:<5s}: {len(d):>10,} rows  [{pcol}]")

    # Merge
    names = list(frames.keys())
    master, _ = frames[names[0]]
    print(f"\n  [DIAG] Start with {names[0]}: {len(master):,} rows")

    for nm in names[1:]:
        df_m, col_m = frames[nm]
        before = len(master)

        dup_pair = df_m.duplicated(subset=MERGE_KEYS).sum()
        dup_self = master.duplicated(subset=MERGE_KEYS).sum()

        master = master.merge(df_m[MERGE_KEYS + [col_m]],
                            on=MERGE_KEYS, how="inner")
        after = len(master)
        print(f"  [DIAG] + {nm:<5s}: {after:>10,} rows "
            f"(Δ={after - before:+,d}, dup_self={dup_self}, dup_{nm}={dup_pair})")

    master = classify(master)
    print(f"  Merged: {len(master):,} rows")
    return master, price_map


# ============================================================
# 5. TABLE 2 - VOLATILITY REGIME (EGARCH)
# ============================================================

def build_table2(master, price_map):
    """
    Model performance per volatility regime based on EGARCH.

    The regime is set by the EGARCH tertile on the observation date:
        Low     : EGARCH <= P33
        Medium  : P33 < EGARCH <= P67
        High    : EGARCH > P67
    """
    print("\n=== Table 2: Volatility Regime (EGARCH) ===")

    # Load the volatility data
    vol = pd.read_csv(INPUT_VOL)
    vol.columns = vol.columns.str.strip()
    vol["Date"] = pd.to_datetime(vol["Date"])

    if "EGARCH" not in vol.columns:
        print("  WARNING: EGARCH column not found. Table skipped.")
        return None

    vol = vol[["Date", "EGARCH"]].dropna().sort_values("Date")

    # Merge EGARCH into master (as-of backward)
    master_sorted = master.sort_values("QUOTE_DATE").reset_index(drop=True)
    merged = pd.merge_asof(
        master_sorted,
        vol.rename(columns={"Date": "QUOTE_DATE", "EGARCH": "EGARCH_level"}),
        on="QUOTE_DATE",
        direction="backward",
    )

    # Determine the tertiles
    p33 = merged["EGARCH_level"].quantile(0.33)
    p67 = merged["EGARCH_level"].quantile(0.67)

    merged["Vol_Regime"] = pd.cut(
        merged["EGARCH_level"],
        bins=[-np.inf, p33, p67, np.inf],
        labels=["Low", "Medium", "High"],
    )

    print(f"  EGARCH tertiles: Low <= {p33:.4f} | "
          f"Medium <= {p67:.4f} | High > {p67:.4f}")

    y = merged["C_LAST"].values
    rows = []

    for regime in ["Low", "Medium", "High"]:
        mask = (merged["Vol_Regime"] == regime).values
        n = mask.sum()
        if n == 0:
            continue

        y_sub = y[mask]
        print(f"\n  [{regime}] n = {n:,}")

        for name, pcol in price_map.items():
            pred = merged[pcol].values[mask]
            mae, rmse, mape = compute_metrics(pred, y_sub)
            rows.append([regime, name, n, mae, rmse, mape])
            print(f"    {name:<5s}  MAE={mae:>10.4f}  "
                  f"RMSE={rmse:>10.4f}  MAPE={mape:>8.2f}%")

    out = pd.DataFrame(rows, columns=[
        "Vol_Regime", "Model", "N", "MAE", "RMSE", "MAPE"
    ])
    os.makedirs(os.path.dirname(OUT_REGIME), exist_ok=True)
    out.to_csv(OUT_REGIME, index=False, float_format="%.4f")
    print(f"\n  Saved: {OUT_REGIME}")
    return out


# ============================================================
# 6. TABLE 3 - PERFORMANCE BY MONEYNESS x TTM
# ============================================================

def build_table3(master, price_map):
    """MAE, RMSE, MAPE per model per category (3 moneyness x 3 TTM)."""
    print("\n=== Table 3: Performance by Moneyness x TTM ===")
    y = master["C_LAST"].values

    rows = []
    for m_cat in MONEYNESS_LABELS:
        for t_cat in TTM_LABELS:
            mask = ((master["M_cat"] == m_cat) &
                    (master["TTM_cat"] == t_cat)).values
            n = mask.sum()
            if n == 0:
                continue

            y_sub = y[mask]
            for name, pcol in price_map.items():
                pred = master[pcol].values[mask]
                mae, rmse, mape = compute_metrics(pred, y_sub)
                rows.append([m_cat, t_cat, name, n, mae, rmse, mape])

    out = pd.DataFrame(rows, columns=[
        "Moneyness", "Days_to_Exp", "Model", "N", "MAE", "RMSE", "MAPE"
    ])
    os.makedirs(os.path.dirname(OUT_TABLE3), exist_ok=True)
    out.to_csv(OUT_TABLE3, index=False, float_format="%.4f")
    print(f"  Saved: {OUT_TABLE3} ({len(out)} rows)")
    return out


# ============================================================
# 7. TABLE 4 - DM BY MONEYNESS x TTM
# ============================================================

def build_table4(master, price_map):
    """
    Pairwise DM test per moneyness x TTM category.
    Loss = squared error, aggregated to a daily mean per category.
    """
    print("\n=== Table 4: DM by Moneyness x TTM ===")
    y = master["C_LAST"].values
    model_names = list(INPUT_FILES.keys())

    # Compute SE per model
    for name, pcol in price_map.items():
        master[f"SE_{name}"] = (y - master[pcol].values) ** 2

    rows = []
    for m_cat in MONEYNESS_LABELS:
        for t_cat in TTM_LABELS:
            mask = ((master["M_cat"] == m_cat) &
                    (master["TTM_cat"] == t_cat)).values
            sub = master.loc[mask]
            if len(sub) < 10:
                continue

            # Daily aggregation
            se_cols = [f"SE_{nm}" for nm in model_names]
            daily = (sub.groupby("QUOTE_DATE")[se_cols]
                        .mean().sort_index())
            T = len(daily)
            if T < 5:
                continue

            for (m1, _), (m2, _) in combinations(INPUT_FILES.items(), 2):
                l1 = daily[f"SE_{m1}"].values
                l2 = daily[f"SE_{m2}"].values

                dm_stat, p_val = dm_test(l1, l2, h=DM_HORIZON,
                                         rule=NW_LAG_RULE)

                better = m1 if l1.mean() < l2.mean() else m2
                sig = ("***" if p_val < 0.01 else
                       "**"  if p_val < 0.05 else
                       "*"   if p_val < 0.10 else "")

                rows.append([
                    m_cat, t_cat, m1, m2, T,
                    dm_stat, p_val, better, sig
                ])

    out = pd.DataFrame(rows, columns=[
        "Moneyness", "Days_to_Exp",
        "Model_1", "Model_2", "N_days",
        "DM_HLN", "p_value", "Lower_Loss", "Signif",
    ])
    os.makedirs(os.path.dirname(OUT_TABLE4), exist_ok=True)
    out.to_csv(OUT_TABLE4, index=False, float_format="%.4f")
    print(f"  Saved: {OUT_TABLE4} ({len(out)} rows)")
    return out


# ============================================================
# 8. MAIN PIPELINE
# ============================================================

def main():
    print("=" * 60)
    print("GRANULAR ANALYSIS: Moneyness x Maturity & Volatility Regime")
    print("Moneyness classification: Gross et al. (2025) -- 3 categories")
    print("=" * 60)

    # Table 1: sample properties (full data)
    build_table1()

    # Load the test set (merge the four models)
    master, price_map = load_test_set()

    # Table 2: performance per volatility regime
    build_table2(master, price_map)

    # Table 3: performance by moneyness x TTM
    build_table3(master, price_map)

    # Table 4: DM by moneyness x TTM
    build_table4(master, price_map)

    print("\n" + "=" * 60)
    print("Done. All tables have been saved.")
    print("=" * 60)


if __name__ == "__main__":
    main()