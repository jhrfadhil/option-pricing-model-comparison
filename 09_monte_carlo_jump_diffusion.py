"""
Monte Carlo Merton Jump-Diffusion Script (Calibration + Pricing)
----------------------------------------------------------------
Estimates European call option prices using Monte Carlo simulation
based on the Merton Jump-Diffusion (MJD) model. This script runs two
stages in sequence:

Stage 1 — Jump-parameter calibration
    Calibrates the jump parameters (λ, μ_J, σ_J) from historical
    S&P 500 data using Maximum Likelihood Estimation (MLE) on a
    Poisson-Normal mixture (Merton, 1976).

Stage 2 — Monte Carlo simulation
    Uses the calibrated parameters to simulate N price paths under the
    risk-neutral measure, then computes the option price as the average
    discounted payoff.

    Price dynamics (risk-neutral):
        dS/S = (r − λk) dt + σ dW + J dN
    where:
        k = E[J − 1] = exp(μ_J + ½σ_J²) − 1
        N ~ Poisson(λ)
        ln(J) ~ N(μ_J, σ_J²)

Evaluation:
    1. Overall   — the entire sample.
    2. Moneyness — ITM / ATM / OTM (Varch, 2019 definition).

Output:
    1. MCJD_theoretical_prices.csv    — theoretical prices per volatility
    2. MCJD_accuracy_overall.csv      — overall metrics
    3. MCJD_accuracy_by_moneyness.csv — metrics per moneyness category
"""

import os
import math
import time
import warnings
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.stats import norm


# ============================================================
# 1. CONFIGURATION
# ============================================================

# --- Input ---
INPUT_PRICE   = os.path.join("data", "raw", "S&P 500 Weekly Historical Data.csv")
INPUT_OPTIONS = os.path.join("data", "processed", "input_BS_MC.csv")

# --- Simulation parameters ---
N_PATHS    = 500_000       # number of simulation paths per option
BATCH_SIZE = 512           # number of options per batch (GPU memory management)
SEED       = 42

# --- Output (written under outputs/Monte Carlo/<N_PATHS>/) ---
OUTPUT_DIR  = os.path.join("outputs", "Monte Carlo", str(N_PATHS))
OUTPUT_THEO = os.path.join(OUTPUT_DIR, "MCJD_theoretical_prices.csv")
OUTPUT_OVR  = os.path.join(OUTPUT_DIR, "MCJD_accuracy_overall.csv")
OUTPUT_MON  = os.path.join(OUTPUT_DIR, "MCJD_accuracy_by_moneyness.csv")

# --- Volatility columns ---
VOL_COLS = ["HV5", "HV20", "HV60", "HV100", "GARCH", "EGARCH"]

# --- Moneyness definition — Varch (2019) ---
ITM_LOWER = 1.05
OTM_UPPER = 0.97

# --- Columns retained in the theoretical-price output ---
KEEP_COLS = [
    "QUOTE_DATE", "UNDERLYING_LAST", "STRIKE",
    "r", "TIME_TO_MATURITY", "C_LAST",
]

# --- MLE calibration ---
MAX_POISSON_K   = 20      # number of terms in the Poisson expansion
OBS_PER_YEAR    = 52      # weekly data (52 weeks/year)


# ============================================================
# 2. STAGE 1 — JUMP-PARAMETER CALIBRATION
# ============================================================

def load_returns(path):
    """Load historical S&P 500 price data and compute log-returns."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["Date"]  = pd.to_datetime(df["Date"])
    df["Price"] = (
        df["Price"].astype(str)
          .str.strip()
          .str.replace(",", "", regex=False)
    )
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

    df = df.dropna(subset=["Price"]).sort_values("Date").reset_index(drop=True)
    df["log_ret"] = np.log(df["Price"] / df["Price"].shift(1))

    returns = df["log_ret"].dropna().values
    print(f"Log-return observations: {len(returns):,}")
    return returns


def merton_pdf(x, mu, sigma, lam, mu_j, sigma_j, dt=1/OBS_PER_YEAR):
    """
    Poisson-Normal mixture density for a single time step dt under the
    Merton Jump-Diffusion model.

    PDF = Σ_{k=0}^{K} P(N=k) · φ(x; μ_k, σ_k²)
    """
    pdf = np.zeros_like(x, dtype=np.float64)
    rate = lam * dt

    for k in range(MAX_POISSON_K):
        poisson_wt = np.exp(-rate) * (rate ** k) / math.factorial(k)
        mu_k    = mu * dt + k * mu_j
        sigma_k = np.sqrt(sigma**2 * dt + k * sigma_j**2)
        pdf += poisson_wt * norm.pdf(x, loc=mu_k, scale=sigma_k)

    return pdf


def neg_log_likelihood(params, returns):
    """Negative log-likelihood for MLE optimization."""
    mu, sigma, lam, mu_j, sigma_j = params

    if sigma <= 0 or lam < 0 or sigma_j <= 0:
        return 1e10

    pdf_vals = merton_pdf(returns, mu, sigma, lam, mu_j, sigma_j)
    pdf_vals = np.maximum(pdf_vals, 1e-300)

    return -np.sum(np.log(pdf_vals))


def calibrate_jump_params(returns):
    """
    Calibrate the Merton Jump-Diffusion parameters via MLE (Nelder-Mead).

    Returns
    -------
    dict  {'lambda_j', 'mu_j', 'sigma_j', 'mu', 'sigma'}
    """
    # Initial guess: [μ, σ, λ, μ_J, σ_J]
    x0 = [0.08, 0.15, 0.5, -0.05, 0.05]

    print("Calibrating jump parameters (Nelder-Mead) ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(neg_log_likelihood, x0, args=(returns,),
                       method="Nelder-Mead",
                       options={"maxiter": 50_000, "xatol": 1e-8})

    mu, sigma, lam, mu_j, sigma_j = res.x

    print(f"\n  Calibration results:")
    print(f"    μ  (annual drift)         = {mu:.4f}")
    print(f"    σ  (diffusion volatility) = {sigma:.4f}")
    print(f"    λ  (jump frequency/yr)    = {lam:.4f}")
    print(f"    μ_J (mean log-jump)       = {mu_j:.6f}")
    print(f"    σ_J (std log-jump)        = {sigma_j:.6f}")
    print(f"    Convergence: {res.success} | iterations: {res.nit}")

    return {
        "lambda_j": lam,
        "mu_j":     mu_j,
        "sigma_j":  sigma_j,
        "mu":       mu,
        "sigma":    sigma,
    }


# ============================================================
# 3. STAGE 2 — MONTE CARLO SIMULATION
# ============================================================

def mc_merton_jd(df, params, device):
    """
    Batched Monte Carlo simulation of the Merton Jump-Diffusion model
    with GPU support (PyTorch).

    Parameters
    ----------
    df     : pd.DataFrame  Option dataset with columns S, K, r, T, σ.
    params : dict           Calibrated jump parameters.
    device : torch.device   CPU or CUDA.

    Returns
    -------
    dict  {vol_name: np.ndarray of theoretical prices}
    """
    lam_j   = params["lambda_j"]
    mu_j    = params["mu_j"]
    sigma_j = params["sigma_j"]

    # k = E[J − 1]
    k_jump = math.exp(mu_j + 0.5 * sigma_j**2) - 1.0
    print(f"  k = E[J−1] = {k_jump:.6f}")

    n = len(df)
    T_all = (df["TIME_TO_MATURITY"].values / 365.0).clip(min=1e-8).astype(np.float32)
    S_all = df["UNDERLYING_LAST"].values.astype(np.float32)
    K_all = df["STRIKE"].values.astype(np.float32)
    r_all = df["r"].values.astype(np.float32)

    mcjd_prices = {vol: np.zeros(n, dtype=np.float32) for vol in VOL_COLS}
    num_batches = math.ceil(n / BATCH_SIZE)

    for b in range(num_batches):
        i0 = b * BATCH_SIZE
        i1 = min(i0 + BATCH_SIZE, n)
        B  = i1 - i0

        if (b + 1) % 50 == 0 or b == 0 or b == num_batches - 1:
            print(f"  Batch {b+1}/{num_batches} "
                  f"(rows {i0}–{i1-1})")

        # Input tensors (1, B) -> broadcast to (N_PATHS, B)
        S0    = torch.tensor(S_all[i0:i1], device=device).unsqueeze(0)
        K_b   = torch.tensor(K_all[i0:i1], device=device).unsqueeze(0)
        r_b   = torch.tensor(r_all[i0:i1], device=device).unsqueeze(0)
        T_b   = torch.tensor(T_all[i0:i1], device=device).unsqueeze(0)

        S0  = S0.expand(N_PATHS, B)
        K_b = K_b.expand(N_PATHS, B)
        r_b = r_b.expand(N_PATHS, B)
        T_b = T_b.expand(N_PATHS, B)

        # Random draws (reused for all volatilities).
        Z_diff = torch.randn(N_PATHS, B, device=device)
        Z_jump = torch.randn(N_PATHS, B, device=device)

        # Poisson jumps: N ~ Poisson(λ·T)
        lambda_T = (lam_j * T_b)
        N_jumps  = torch.poisson(lambda_T)

        # Cumulative log-jump: ~ N(N·μ_J, N·σ_J²)
        sqrt_N   = torch.sqrt(torch.clamp(N_jumps, min=0.0))
        jump_sum = mu_j * N_jumps + sigma_j * sqrt_N * Z_jump

        # Components independent of σ.
        base_logS = torch.log(S0) + jump_sum
        disc      = torch.exp(-r_b * T_b)

        for vol in VOL_COLS:
            sigma_np = df[vol].values[i0:i1].astype(np.float32)
            sigma    = torch.tensor(sigma_np, device=device).unsqueeze(0)
            sigma_b  = sigma.expand(N_PATHS, B)

            # Risk-neutral drift: (r − λk − ½σ²) T
            drift = (r_b - lam_j * k_jump - 0.5 * sigma_b**2) * T_b

            logS_T = base_logS + drift + sigma_b * torch.sqrt(T_b) * Z_diff
            S_T    = torch.exp(logS_T)

            payoff = torch.clamp(S_T - K_b, min=0.0)
            price  = (disc * payoff).mean(dim=0)

            mcjd_prices[vol][i0:i1] = price.detach().cpu().numpy()

    torch.cuda.empty_cache()
    return mcjd_prices


# ============================================================
# 4. EVALUATION FUNCTIONS (identical to the BS / Bachelier scripts)
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
    mape = (np.mean(np.abs(errors[mask] / actual[mask])) * 100
            if mask.sum() > 0 else np.nan)

    return mae, rmse, mape


def classify_moneyness(S, K):
    """ITM: S/K > 1.05 | OTM: S/K < 0.97 | ATM: otherwise."""
    m = S / K
    return np.where(m > ITM_LOWER, "ITM",
           np.where(m < OTM_UPPER, "OTM", "ATM"))


# ============================================================
# 5. MAIN PIPELINE
# ============================================================

def main():
    start = time.time()

    # --- Select device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # =============================================
    # STAGE 1: Jump-parameter calibration
    # =============================================
    print("\n=== Stage 1: Jump-parameter calibration ===")
    returns = load_returns(INPUT_PRICE)
    params  = calibrate_jump_params(returns)

    # =============================================
    # STAGE 2: Monte Carlo simulation
    # =============================================
    print("\n=== Stage 2: Monte Carlo MJD simulation ===")
    df = pd.read_csv(INPUT_OPTIONS)
    df.columns = df.columns.str.strip()

    # Validate columns.
    required = ["UNDERLYING_LAST", "STRIKE", "C_LAST",
                "r", "TIME_TO_MATURITY"] + VOL_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {INPUT_OPTIONS}: {missing}")

    print(f"Rows: {len(df):,}")

    mcjd_prices = mc_merton_jd(df, params, device)

    # --- Moneyness classification ---
    S     = df["UNDERLYING_LAST"].values
    K     = df["STRIKE"].values
    C_mkt = df["C_LAST"].values

    df["Moneyness"] = classify_moneyness(S, K)

    for cat in ["ITM", "ATM", "OTM"]:
        n = (df["Moneyness"] == cat).sum()
        print(f"  {cat}: {n:,} observations")

    # --- Save theoretical prices ---
    print(f"\n--- Saving {OUTPUT_THEO} ---")
    output_prices = df[KEEP_COLS + ["Moneyness"]].copy()
    for vol in VOL_COLS:
        output_prices[f"MCJD_{vol}"] = mcjd_prices[vol]

    os.makedirs(os.path.dirname(OUTPUT_THEO), exist_ok=True)
    output_prices.to_csv(OUTPUT_THEO, index=False)

    # --- Overall metrics ---
    print("\n--- Overall evaluation ---")
    rows_overall = []
    for vol in VOL_COLS:
        theo = mcjd_prices[vol]
        mae, rmse, mape = compute_metrics(theo, C_mkt)
        rows_overall.append([vol, mae, rmse, mape])
        print(f"  {vol:<8s}  MAE={mae:>10.4f}  "
              f"RMSE={rmse:>10.4f}  MAPE={mape:>8.2f}%")

    os.makedirs(os.path.dirname(OUTPUT_OVR), exist_ok=True)
    pd.DataFrame(
        rows_overall, columns=["Volatility", "MAE", "RMSE", "MAPE"]
    ).to_csv(OUTPUT_OVR, index=False)

    # --- Metrics per moneyness ---
    print("\n--- Evaluation per moneyness ---")
    rows_mon = []
    moneyness_arr = df["Moneyness"].values

    for cat in ["ITM", "ATM", "OTM"]:
        mask  = moneyness_arr == cat
        n_cat = mask.sum()
        if n_cat == 0:
            continue

        C_sub = C_mkt[mask]
        print(f"\n  [{cat}] n = {n_cat:,}")

        for vol in VOL_COLS:
            theo_sub = mcjd_prices[vol][mask]
            mae, rmse, mape = compute_metrics(theo_sub, C_sub)
            rows_mon.append([cat, vol, n_cat, mae, rmse, mape])
            print(f"    {vol:<8s}  MAE={mae:>10.4f}  "
                  f"RMSE={rmse:>10.4f}  MAPE={mape:>8.2f}%")

    os.makedirs(os.path.dirname(OUTPUT_MON), exist_ok=True)
    pd.DataFrame(
        rows_mon,
        columns=["Moneyness", "Volatility", "N", "MAE", "RMSE", "MAPE"],
    ).to_csv(OUTPUT_MON, index=False)

    # --- Summary ---
    elapsed = time.time() - start
    print(f"\n=== Done ({elapsed:.2f} s, device: {device}) ===")
    print(f"  {OUTPUT_THEO}")
    print(f"  {OUTPUT_OVR}")
    print(f"  {OUTPUT_MON}")


if __name__ == "__main__":
    main()