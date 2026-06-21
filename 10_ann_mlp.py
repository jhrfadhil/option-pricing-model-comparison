"""
Call Option Pricing with a Multi-Layer Perceptron (MLP)
------------------------------------------------------------
Predicts call option prices using an artificial neural network
with a standard Multi-Layer Perceptron architecture. This script 
serves as the neural-network baseline for assessing the contribution
of architecture to pricing accuracy.

Prediction target:
    y = log(1 + tv/K × 100),   tv = max(C − max(S − K, 0), 0)

    Using the time value (tv) removes the already-deterministic
    intrinsic component, so the network only learns the optionality
    value. The log1p scaling stabilizes the range of the target.

Input features (22 total):
    • Base (6)            : log(S/K), log(S/K)², T, √T, r, intrinsic_ratio
    • Volatility (6)      : HV5, HV20, HV60, HV100, GARCH, EGARCH
    • σ√T (6)             : natural scale of the stochastic process per volatility
    • Interaction (2)     : log(S/K)×√T, log(S/K)×HV20
    • Term structure (2)  : (HV5−HV100)/HV100, std(HV5,20,60,100)

    Note: HV20×√T is not included separately in the Interaction
    block because it is identical to vsqrtT_HV20 in the σ√T block.

Architecture per model:
    Input(22) -> Linear(128) -> GELU
              -> 8 × [Linear(128) -> LayerNorm -> GELU -> Dropout]
              -> Linear(1) -> Softplus

Data split (chronological, no shuffling):
    Train 60% | Validation 20% | Test 20%

Ensemble:
    5 models with different seeds; mean aggregation in price space
    to avoid Jensen bias from the log transformation.

Evaluation is performed at two levels (test set):
    1. Overall   — the entire test set.
    2. Moneyness — ITM / ATM / OTM (definition from Varch, 2019).

Output:
    1. MLP_theoretical_prices.csv    — theoretical prices for the full dataset
    2. MLP_accuracy_overall.csv      — overall metrics (test set)
    3. MLP_accuracy_by_moneyness.csv — metrics by moneyness (test set)
"""

import copy
import math
import os
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FILE  = os.path.join("data", "processed", "input_ANN.csv")

# All outputs for this architecture live under outputs/ANN/MLP/.
OUTPUT_DIR  = os.path.join("outputs", "ANN", "MLP")
OUTPUT_THEO = os.path.join(OUTPUT_DIR, "MLP_theoretical_prices.csv")
OUTPUT_OVR  = os.path.join(OUTPUT_DIR, "MLP_accuracy_overall.csv")
OUTPUT_MON  = os.path.join(OUTPUT_DIR, "MLP_accuracy_by_moneyness.csv")

# Training hyperparameters
EPOCHS       = 200
BATCH_SIZE   = 2048
LR_INIT      = 2e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 30

# Ensemble: number of models with different seeds
N_MODELS = 5
SEEDS    = [42, 123, 2024, 7, 999][:N_MODELS]

# MLP architecture
HIDDEN_DIM = 128
NUM_LAYERS = 8       # number of hidden layers (Linear -> LN -> GELU -> Dropout)
DROPOUT    = 0.10

# Volatility columns
VOL_COLS = ["HV5", "HV20", "HV60", "HV100", "GARCH", "EGARCH"]

# Feature list (built in build_features)
BASE_FEATURES = [
    "log_moneyness", "log_moneyness_sq", "T_years", "sqrt_T", "r",
    "intrinsic_ratio",
]
VSQRT_FEATURES = [f"vsqrtT_{v}" for v in VOL_COLS]
INTERACTION_FEATURES = [
    "log_m_x_sqrtT",
    "log_m_x_hv20",
]
TERM_STRUCTURE = [
    "hv_slope",
    "hv_dispersion",
]

FEATURE_COLS = (
    BASE_FEATURES
    + VOL_COLS
    + VSQRT_FEATURES
    + INTERACTION_FEATURES
    + TERM_STRUCTURE
)

# Chronological split proportions.
# Train = first int(TRAIN_FRAC * n) rows; Test = last int(TEST_FRAC * n) rows
# (this matches script 05's extract_tail rounding). Validation is the middle
# remainder, so VAL_FRAC is documented here but no longer used directly.
TRAIN_FRAC = 0.60
VAL_FRAC   = 0.20
TEST_FRAC  = 0.20

# Moneyness definition — Varch (2019)
ITM_LOWER = 1.05
OTM_UPPER = 0.97

# Original columns retained in the theoretical-prices output
KEEP_COLS = [
    "QUOTE_DATE", "UNDERLYING_LAST", "STRIKE",
    "r", "TIME_TO_MATURITY", "C_LAST",
]

MODEL_LABEL = "MLP"


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def build_features(df):
    """
    Build 22 model-agnostic features (no Black–Scholes formulas).
    All features are pure market information: price, strike, maturity,
    risk-free rate, volatility, and their direct derivatives.
    """
    S = df["UNDERLYING_LAST"].astype(float).values
    K = df["STRIKE"].astype(float).values
    T = (df["TIME_TO_MATURITY"].astype(float) / 365.0).values

    T_safe = np.maximum(T, 1e-6)
    sqrt_T = np.sqrt(T_safe)
    log_SK = np.log(S / K)

    # --- Base features ---
    df["log_moneyness"]    = log_SK
    df["log_moneyness_sq"] = log_SK ** 2
    df["T_years"]          = T
    df["sqrt_T"]           = sqrt_T
    df["intrinsic_ratio"]  = np.maximum(S / K - 1.0, 0.0)

    # --- σ√T per volatility (natural scale of the stochastic process) ---
    for vcol in VOL_COLS:
        sigma = np.maximum(df[vcol].astype(float).values, 1e-6)
        df[f"vsqrtT_{vcol}"] = sigma * sqrt_T

    # --- Non-linear interactions ---
    # Note: HV20×√T is not computed here because it already exists
    # as vsqrtT_HV20 in the σ√T block above (the values are identical).
    hv20 = np.maximum(df["HV20"].astype(float).values, 1e-6)
    df["log_m_x_sqrtT"] = log_SK * sqrt_T
    df["log_m_x_hv20"]  = log_SK * hv20

    # --- Volatility term structure ---
    hv5   = np.maximum(df["HV5"].astype(float).values,   1e-6)
    hv60  = np.maximum(df["HV60"].astype(float).values,  1e-6)
    hv100 = np.maximum(df["HV100"].astype(float).values, 1e-6)

    df["hv_slope"]      = (hv5 - hv100) / hv100
    df["hv_dispersion"] = np.std(
        np.stack([hv5, hv20, hv60, hv100], axis=1), axis=1
    )

    return df


# ============================================================
# 3. MLP ARCHITECTURE
# ============================================================

class MLPTimeValue(nn.Module):
    """
    Multi-Layer Perceptron for predicting the normalized target
    log(1 + tv/K × 100).

    Structure:
        Input -> Linear -> GELU
              -> N × [Linear -> LayerNorm -> GELU -> Dropout]
              -> Linear -> Softplus

    Softplus on the output guarantees predictions ≥ 0 (consistent
    with the non-negative log1p target).
    """

    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()

        layers = []

        # Input projection
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())

        # Hidden layers
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))

        self.network = nn.Sequential(*layers)

        # Output head
        self.head     = nn.Linear(hidden_dim, 1)
        self.softplus = nn.Softplus()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.network(x)
        return self.softplus(self.head(h))


# ============================================================
# 4. LOSS: LOG-COSH
# ============================================================

class LogCoshLoss(nn.Module):
    """
    Log-cosh loss: a smooth approximation of L1, robust to
    outliers, with a continuous gradient everywhere.

    Numerically stable form:
        log(cosh(x)) = |x| + softplus(-2|x|) − log(2)
    """

    def forward(self, pred, target):
        x = pred - target
        abs_x = torch.abs(x)
        return (abs_x + F.softplus(-2.0 * abs_x) - math.log(2.0)).mean()


# ============================================================
# 5. METRICS & MONEYNESS
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
    """ITM: S/K > 1.05 | OTM: S/K < 0.97 | ATM: the remainder."""
    m = S / K
    return np.where(m > ITM_LOWER, "ITM",
           np.where(m < OTM_UPPER, "OTM", "ATM"))


# ============================================================
# 6. DATA PREPARATION
# ============================================================

def prepare_data(df):
    """
    Build features, filter valid rows, and determine the chronological
    train/val/test split indices.
    """
    df = build_features(df)

    target_col = "C_LAST"
    hv_cols    = ["HV5", "HV20", "HV60", "HV100"]

    mask = (
        df[FEATURE_COLS + [target_col]].notna().all(axis=1) &
        np.isfinite(df[FEATURE_COLS].values).all(axis=1) &
        (df[hv_cols] > 0).all(axis=1) &
        (df[target_col] > 0) &
        (df["GARCH"] > 0) &
        (df["EGARCH"] > 0)
    )
    df_clean = df[mask].reset_index(drop=True)
    print(f"Samples after cleaning: {len(df_clean):,}")

    X = df_clean[FEATURE_COLS].values.astype(np.float32)

    S_clean = df_clean["UNDERLYING_LAST"].values.astype(np.float32)
    K_clean = df_clean["STRIKE"].values.astype(np.float32)
    C_clean = df_clean[target_col].values.astype(np.float32)
    intrinsic_val = np.maximum(S_clean - K_clean, 0.0)

    n = len(df_clean)
    train_end = int(TRAIN_FRAC * n)                 # unchanged
    val_end   = n - int(TEST_FRAC * n)              # test = last int(0.20*n) rows -> matches script 05

    return df_clean, X, S_clean, K_clean, C_clean, intrinsic_val, train_end, val_end


# ============================================================
# 7. TRAIN ONE MODEL
# ============================================================

def train_one_model(seed, X_train_s, y_train, X_val_s, y_val,
                    input_dim, device):
    """Train one MLP with a given seed. Returns the model with the best weights."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MLPTimeValue(input_dim=input_dim).to(device)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_s), torch.tensor(y_train)),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val_s), torch.tensor(y_val)),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    criterion = LogCoshLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # --- Validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(criterion(model(xb), yb).item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]["lr"]
            print(f"    Epoch {epoch:03d} | "
                  f"Train: {train_loss:.6f} | "
                  f"Val: {val_loss:.6f} | LR: {lr:.2e}")

        # --- Early stopping ---
        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            best_state    = copy.deepcopy(model.state_dict())
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"    Early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


# ============================================================
# 8. PREDICTION & PRICE RECONSTRUCTION
# ============================================================

def predict_target(model, X_scaled, device):
    """Forward pass over all data. Returns raw predictions in target space."""
    loader = DataLoader(
        TensorDataset(torch.tensor(X_scaled)),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    model.eval()
    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            out = model(xb.to(device)).cpu().numpy().reshape(-1)
            preds.append(out)
    return np.concatenate(preds)


def reconstruct_price(y_pred_log, K, intrinsic):
    """
    Inverse of the target transformation:
        y         = log(1 + tv/K × 100)
        tv/K×100  = exp(y) − 1     (use expm1 for numerical stability)
        tv        = ((exp(y) − 1) / 100) × K
        C_pred    = intrinsic + max(tv, 0)
    """
    ratio_x100 = np.expm1(y_pred_log)
    tv         = (ratio_x100 / 100.0) * K
    tv         = np.maximum(tv, 0.0)
    C_pred     = intrinsic + tv
    return np.maximum(C_pred, intrinsic)


# ============================================================
# 9. MAIN PIPELINE
# ============================================================

def main():
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load data ---
    print("\n--- Loading data ---")
    df = pd.read_csv(INPUT_FILE)
    df.columns = df.columns.str.strip()

    required = [
        "QUOTE_DATE", "UNDERLYING_LAST", "C_LAST", "STRIKE",
        "r", "TIME_TO_MATURITY",
        "HV5", "HV20", "HV60", "HV100", "GARCH", "EGARCH",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {INPUT_FILE}: {missing}")

    df["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"])
    df = df.sort_values("QUOTE_DATE").reset_index(drop=True)

    # --- Prepare data ---
    (df_clean, X, S_clean, K_clean, C_clean,
     intrinsic_val, train_end, val_end) = prepare_data(df)

    n = len(df_clean)
    print(f"  Train : {train_end:,}  "
          f"({df_clean['QUOTE_DATE'].iloc[0].date()} — "
          f"{df_clean['QUOTE_DATE'].iloc[train_end-1].date()})")
    print(f"  Val   : {val_end - train_end:,}  "
          f"({df_clean['QUOTE_DATE'].iloc[train_end].date()} — "
          f"{df_clean['QUOTE_DATE'].iloc[val_end-1].date()})")
    print(f"  Test  : {n - val_end:,}  "
          f"({df_clean['QUOTE_DATE'].iloc[val_end].date()} — "
          f"{df_clean['QUOTE_DATE'].iloc[-1].date()})")

    # --- Target: log(1 + tv/K × 100) ---
    time_value = np.maximum(C_clean - intrinsic_val, 0.0)
    ratio_x100 = (time_value / K_clean) * 100.0
    y_target   = np.log1p(ratio_x100).reshape(-1, 1).astype(np.float32)

    # --- Chronological split ---
    X_train = X[:train_end]
    X_val   = X[train_end:val_end]
    y_train = y_target[:train_end]
    y_val   = y_target[train_end:val_end]

    # --- Scaling (fit on train only) ---
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s   = scaler.transform(X_val).astype(np.float32)
    X_all_s   = scaler.transform(X).astype(np.float32)

    # --- Architecture info ---
    input_dim    = len(FEATURE_COLS)
    sample_model = MLPTimeValue(input_dim=input_dim)
    total_params = sum(p.numel() for p in sample_model.parameters() if p.requires_grad)
    print(f"\nMLP architecture:")
    print(f"  Input features : {input_dim}")
    print(f"  Hidden dim     : {HIDDEN_DIM}")
    print(f"  Num layers     : {NUM_LAYERS}")
    print(f"  Dropout        : {DROPOUT}")
    print(f"  Total params   : {total_params:,} per model")
    print(f"  Ensemble       : {N_MODELS} models × max {EPOCHS} epochs")
    del sample_model

    # ============================================================
    # TRAINING ENSEMBLE
    # ============================================================
    print(f"\n--- Ensemble training ({N_MODELS} models) ---")

    # Averaging in price space (not log space) avoids Jensen bias
    # and yields an estimator that is unbiased with respect to RMSE.
    all_preds_usd = []

    for i, seed in enumerate(SEEDS, 1):
        print(f"\n  [Model {i}/{N_MODELS}]  seed = {seed}")
        model = train_one_model(
            seed, X_train_s, y_train, X_val_s, y_val,
            input_dim, device,
        )

        y_pred_log = predict_target(model, X_all_s, device)
        pred_usd   = reconstruct_price(y_pred_log, K_clean, intrinsic_val)
        all_preds_usd.append(pred_usd)

        # Clean up GPU memory before the next model
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Ensemble average in price space
    all_pred_usd = np.mean(np.stack(all_preds_usd, axis=0), axis=0)

    # --- Moneyness classification ---
    df_clean["Moneyness"] = classify_moneyness(S_clean, K_clean)

    # --- Save theoretical prices ---
    output_prices = df_clean[KEEP_COLS + ["Moneyness"]].copy()
    output_prices["MLP_price"] = all_pred_usd
    os.makedirs(os.path.dirname(OUTPUT_THEO), exist_ok=True)
    output_prices.to_csv(OUTPUT_THEO, index=False)
    print(f"\nTheoretical prices saved: {OUTPUT_THEO}")

    # --- Overall evaluation (test set) ---
    C_test_actual = C_clean[val_end:]
    C_test_pred   = all_pred_usd[val_end:]

    print("\n--- Overall evaluation (test set) ---")
    mae, rmse, mape = compute_metrics(C_test_pred, C_test_actual)
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAPE = {mape:.2f}%")

    os.makedirs(os.path.dirname(OUTPUT_OVR), exist_ok=True)
    pd.DataFrame(
        [[MODEL_LABEL, mae, rmse, mape]],
        columns=["Model", "MAE", "RMSE", "MAPE"],
    ).to_csv(OUTPUT_OVR, index=False)

    # --- Evaluation by moneyness (test set) ---
    print("\n--- Evaluation by moneyness (test set) ---")
    moneyness_test = df_clean["Moneyness"].values[val_end:]
    rows_mon = []

    for cat in ["ITM", "ATM", "OTM"]:
        mask  = moneyness_test == cat
        n_cat = mask.sum()
        if n_cat == 0:
            continue

        mae_c, rmse_c, mape_c = compute_metrics(
            C_test_pred[mask], C_test_actual[mask]
        )
        rows_mon.append([cat, MODEL_LABEL, n_cat, mae_c, rmse_c, mape_c])
        print(f"  [{cat}] n = {n_cat:,}  "
              f"MAE={mae_c:.4f}  RMSE={rmse_c:.4f}  MAPE={mape_c:.2f}%")

    os.makedirs(os.path.dirname(OUTPUT_MON), exist_ok=True)
    pd.DataFrame(
        rows_mon,
        columns=["Moneyness", "Model", "N", "MAE", "RMSE", "MAPE"],
    ).to_csv(OUTPUT_MON, index=False)

    # --- Summary ---
    elapsed = time.time() - start
    print(f"\n=== Done ({elapsed:.2f} s, device: {device}) ===")
    print(f"  {OUTPUT_THEO}")
    print(f"  {OUTPUT_OVR}")
    print(f"  {OUTPUT_MON}")


if __name__ == "__main__":
    main()