"""
Call Option Pricing Script with a Mixture-of-Experts (MoE)
----------------------------------------------------------
Predicts call option prices using an artificial neural network
with a Mixture-of-Experts architecture. Three "experts" are
combined through a gating network that produces soft weights
(rather than a hard split) based on market context — so the
transition between moneyness regions is smooth and not broken
at sharp boundaries.

Prediction target:
    y = log(1 + tv/K × 100),   tv = max(C − max(S − K, 0), 0)

    Using the time value (tv) removes the already-deterministic
    intrinsic component, so the network only learns the
    optionality value. The log1p scale stabilizes the target range.

Input features (22 total):
    • Base (6)            : log(S/K), log(S/K)², T, √T, r, intrinsic_ratio
    • Volatility (6)      : HV5, HV20, HV60, HV100, GARCH, EGARCH
    • σ√T (6)             : natural scale of the stochastic process per volatility
    • Interaction (2)     : log(S/K)×√T, log(S/K)×HV20
    • Term structure (2)  : (HV5−HV100)/HV100, std(HV5,20,60,100)

    Note: HV20×√T is not included separately in the Interaction
    block because it is identical to vsqrtT_HV20 in the σ√T block.

Gating context (3 of the 22 features):
    log(S/K), T, √T — the main determinants for moneyness ×
    maturity classification. The gating network produces a softmax
    distribution over the 3 experts.

Architecture per model:

    ┌──────────────────────────────────────────────────────────────┐
    │                    Input (22 features)                       │
    │                                                              │
    │  ┌─────────────┐           ┌──────────────────────────────┐  │
    │  │ Gating Net  │           │  3 Experts (parallel)        │  │
    │  │ ctx (3)  →  │           │  ├─ Expert 1                 │  │
    │  │ softmax(3)  │           │  ├─ Expert 2                 │  │
    │  └──────┬──────┘           │  └─ Expert 3                 │  │
    │         │ gate_w           │  (each: proj → 2×ResBlock →  │  │
    │         │                  │   LayerNorm → scalar)        │  │
    │         │                  └──────────┬───────────────────┘  │
    │         │                             │ expert_outs          │
    │         ▼                             ▼                      │
    │         └────── Σ gate[i] × softplus(expert_out[i]) ───→     │
    └──────────────────────────────────────────────────────────────┘

    Every expert sees ALL 22 features — gating only adjusts each
    one's contribution weight. Per-expert softplus before mixing
    guarantees a non-negative output after the convex combination.

Data split (chronological, no shuffling):
    Train 60% | Validation 20% | Test 20%

Ensemble:
    5 models with different seeds; averaged aggregation in price
    space to avoid Jensen bias from the log transform.

Diagnostics:
    The script prints the average gate weights per moneyness
    category (a random sample from the test set) to verify the
    expert specialization that emerges implicitly from training.

Evaluation is performed at two levels (test set):
    1. Overall   — the entire test set.
    2. Moneyness — ITM / ATM / OTM (definition from Varch, 2019).

Computational profiling:
    The script records separately:
      - Total ensemble training time (5 models)
      - Total inference time over all data (for the CSV export)
      - Inference throughput in µs per observation per model
      - Equivalent inference time for the test set (comparator vs BS/Bach/MCJD)
    GPU measurement uses torch.cuda.synchronize() so the reported
    wall-clock time is accurate (CUDA is asynchronous).

Output:
    1. MoE_theoretical_prices.csv    — theoretical prices for all data
    2. MoE_accuracy_overall.csv      — overall metrics (test set)
    3. MoE_accuracy_by_moneyness.csv — per-moneyness metrics (test set)
"""

import os
import copy
import math
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

OUTPUT_DIR  = os.path.join("outputs", "ANN", "MoE")
OUTPUT_THEO = os.path.join(OUTPUT_DIR, "MoE_theoretical_prices.csv")
OUTPUT_OVR  = os.path.join(OUTPUT_DIR, "MoE_accuracy_overall.csv")
OUTPUT_MON  = os.path.join(OUTPUT_DIR, "MoE_accuracy_by_moneyness.csv")

# Training hyperparameters
EPOCHS       = 200
BATCH_SIZE   = 2048
LR_INIT      = 2e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 30

# Ensemble: number of models with different seeds
N_MODELS = 5
SEEDS    = [42, 123, 2024, 7, 999][:N_MODELS]

# MoE architecture
N_EXPERTS     = 3        # specialist per region (ITM/ATM/OTM, soft assignment)
EXPERT_HIDDEN = 96       # hidden dim per expert (smaller — compensated by the number of experts)
EXPERT_BLOCKS = 2        # 2 residual blocks per expert
GATE_HIDDEN   = 32       # hidden dim for the gating network
DROPOUT       = 0.10

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

# Context for the gating network — the 3 most determinant features for
# moneyness × maturity classification. Indices computed from FEATURE_COLS.
GATE_CTX_NAMES = ["log_moneyness", "T_years", "sqrt_T"]
GATE_CTX_IDX   = [FEATURE_COLS.index(name) for name in GATE_CTX_NAMES]

# Split proportions (chronological)
TRAIN_FRAC = 0.60
VAL_FRAC   = 0.20   # now implied: validation is the middle remainder between train and test
TEST_FRAC  = 0.20   # test = last int(TEST_FRAC * n) rows, matching script 05

# Moneyness definition — Varch (2019)
ITM_LOWER = 1.05
OTM_UPPER = 0.97

# Original columns retained in the theoretical-price output
KEEP_COLS = [
    "QUOTE_DATE", "UNDERLYING_LAST", "STRIKE",
    "r", "TIME_TO_MATURITY", "C_LAST",
]

MODEL_LABEL = "MoE"


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

def build_features(df):
    """
    Build 22 model-agnostic features (without the Black–Scholes
    formula). All features are purely market information: price,
    strike, maturity, risk-free rate, volatility, and their
    direct derivatives.
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
    # as vsqrtT_HV20 in the σ√T block above (its value is identical).
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
# 3. MIXTURE-OF-EXPERTS ARCHITECTURE
# ============================================================

class ResidualBlock(nn.Module):
    """Pre-activation residual block: LN -> Linear -> GELU -> Drop -> LN -> Linear -> Drop -> (+ skip)."""

    def __init__(self, dim, dropout=DROPOUT):
        super().__init__()
        self.ln1  = nn.LayerNorm(dim)
        self.fc1  = nn.Linear(dim, dim)
        self.ln2  = nn.LayerNorm(dim)
        self.fc2  = nn.Linear(dim, dim)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.fc1(self.ln1(x))
        h = self.drop(self.act(h))
        h = self.fc2(self.ln2(h))
        h = self.drop(h)
        return x + h


class Expert(nn.Module):
    """
    A single expert: proj(input) -> stack of ResidualBlocks -> scalar.
    Raw output (no softplus); softplus is applied at the MoE level
    after mixing.
    """

    def __init__(self, input_dim, hidden_dim=EXPERT_HIDDEN,
                 n_blocks=EXPERT_BLOCKS, dropout=DROPOUT):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.act  = nn.GELU()
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)
        ])
        self.ln_out = nn.LayerNorm(hidden_dim)
        self.head   = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.act(self.proj(x))
        for b in self.blocks:
            h = b(h)
        h = self.ln_out(h)
        return self.head(h).squeeze(-1)        # (B,)


class GatingNetwork(nn.Module):
    """
    Produces a softmax distribution over n_experts based on a small
    market context (3 features). A 2-layer MLP is enough to capture
    the smooth boundary between moneyness × maturity regions.
    """

    def __init__(self, ctx_dim, n_experts=N_EXPERTS,
                 hidden_dim=GATE_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_experts),
        )

    def forward(self, ctx):
        return F.softmax(self.net(ctx), dim=-1)


class MoENet(nn.Module):
    """
    Mixture-of-Experts for predicting the normalized target
    log(1 + tv/K × 100).

    Every expert receives all input features; the gating network
    receives only the 3 context features and sets the mixing weights.

    Output non-negativity is guaranteed by:
      1. Softplus on each expert before mixing.
      2. Convex combination with softmax weights (≥ 0, sum = 1).
    """

    def __init__(self, input_dim, ctx_idx, n_experts=N_EXPERTS):
        super().__init__()
        # Store as a buffer so it follows the device (.to(device))
        self.register_buffer(
            "ctx_idx", torch.tensor(ctx_idx, dtype=torch.long)
        )
        self.n_experts = n_experts

        self.gate = GatingNetwork(
            ctx_dim=len(ctx_idx), n_experts=n_experts
        )
        self.experts = nn.ModuleList([
            Expert(input_dim) for _ in range(n_experts)
        ])
        self.softplus = nn.Softplus()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Gating context (3 features from x)
        ctx    = x.index_select(1, self.ctx_idx)        # (B, ctx_dim)
        gate_w = self.gate(ctx)                         # (B, n_experts)

        # Each expert -> scalar, softplus-ed to be ≥ 0
        expert_outs = torch.stack(
            [self.softplus(e(x)) for e in self.experts], dim=1
        )                                               # (B, n_experts)

        # Convex combination -> non-negative scalar
        y = (gate_w * expert_outs).sum(dim=1, keepdim=True)
        return y                                        # (B, 1)


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
    Compute MAE, RMSE, and MAPE between theoretical prices and
    market prices. MAPE is computed only on rows with market
    price > 0.
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
    """ITM: S/K > 1.05 | OTM: S/K < 0.97 | ATM: the rest."""
    m = S / K
    return np.where(m > ITM_LOWER, "ITM",
           np.where(m < OTM_UPPER, "OTM", "ATM"))


# ============================================================
# 6. DATA PREPARATION
# ============================================================

def prepare_data(df):
    """
    Build features, filter valid rows, and determine the
    chronological train/val/test split indices.
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
    train_end = int(TRAIN_FRAC * n)            # unchanged
    val_end   = n - int(TEST_FRAC * n)         # test = last int(TEST_FRAC * n) rows -> matches script 05

    return df_clean, X, S_clean, K_clean, C_clean, intrinsic_val, train_end, val_end


# ============================================================
# 7. TRAINING A SINGLE MODEL
# ============================================================

def train_one_model(seed, X_train_s, y_train, X_val_s, y_val,
                    input_dim, device):
    """Train a single MoE with a given seed. Returns the model with the best weights."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MoENet(
        input_dim=input_dim,
        ctx_idx=GATE_CTX_IDX,
        n_experts=N_EXPERTS,
    ).to(device)

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
    Inverse of the target transform:
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


def analyze_gate_usage(model, X_scaled_sample, moneyness_sample, device):
    """
    Diagnostic: how the gating network allocates expert weights for
    each moneyness category on the sample.
    """
    loader = DataLoader(
        TensorDataset(torch.tensor(X_scaled_sample)),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    model.eval()
    gate_weights = []
    with torch.no_grad():
        for (xb,) in loader:
            xb  = xb.to(device)
            ctx = xb.index_select(1, model.ctx_idx)
            gw  = model.gate(ctx).cpu().numpy()
            gate_weights.append(gw)
    gate_weights = np.concatenate(gate_weights, axis=0)

    print(f"\n  Average gate weights per moneyness "
          f"(N_EXPERTS = {N_EXPERTS}):")
    for cat in ["ITM", "ATM", "OTM"]:
        mask = moneyness_sample == cat
        if mask.sum() == 0:
            continue
        mean_w = gate_weights[mask].mean(axis=0)
        w_str  = "  ".join([f"E{i}={w:.3f}" for i, w in enumerate(mean_w)])
        print(f"    [{cat}] n={mask.sum():>6,}  {w_str}")


# ============================================================
# 9. PROFILING HELPER
# ============================================================

def sync_time(device):
    """
    Accurate wall-clock time for GPU operations.
    CUDA is asynchronous — without synchronize, time.time() only
    records when the kernel is submitted to the queue, not when
    execution finishes.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.time()


# ============================================================
# 10. MAIN PIPELINE
# ============================================================

def main():
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Profiling accumulators ---
    total_train_time = 0.0
    total_inf_time   = 0.0

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

    # --- Data preparation ---
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
    input_dim     = len(FEATURE_COLS)
    sample_model  = MoENet(input_dim, GATE_CTX_IDX, N_EXPERTS)
    total_params  = sum(p.numel() for p in sample_model.parameters() if p.requires_grad)
    gate_params   = sum(p.numel() for p in sample_model.gate.parameters() if p.requires_grad)
    expert_params = sum(p.numel() for p in sample_model.experts[0].parameters() if p.requires_grad)
    print(f"\nMoE architecture:")
    print(f"  Input features    : {input_dim}")
    print(f"  Gating context    : {GATE_CTX_NAMES} (indices {GATE_CTX_IDX})")
    print(f"  Number of experts : {N_EXPERTS}")
    print(f"  Params per expert : {expert_params:,}")
    print(f"  Params gating     : {gate_params:,}")
    print(f"  Total per model   : {total_params:,}")
    print(f"  Ensemble          : {N_MODELS} models × max {EPOCHS} epochs")
    del sample_model

    # ============================================================
    # TRAINING ENSEMBLE — with separate profiling
    # ============================================================
    print(f"\n--- Ensemble training ({N_MODELS} models) ---")

    # Averaging in price space (not log space) avoids Jensen bias
    # and gives an unbiased estimator with respect to RMSE.
    all_preds_usd = []
    last_model    = None      # kept for the gate-usage diagnostic

    for i, seed in enumerate(SEEDS, 1):
        print(f"\n  [Model {i}/{N_MODELS}]  seed = {seed}")

        # --- Training (timed) ---
        t0 = sync_time(device)
        model = train_one_model(
            seed, X_train_s, y_train, X_val_s, y_val,
            input_dim, device,
        )
        t1 = sync_time(device)
        train_dur = t1 - t0
        total_train_time += train_dur
        print(f"    [profile] training finished: {train_dur:.2f} s")

        # --- Inference over all data (timed) ---
        t0 = sync_time(device)
        y_pred_log = predict_target(model, X_all_s, device)
        t1 = sync_time(device)
        inf_dur = t1 - t0
        total_inf_time += inf_dur
        print(f"    [profile] inference {n:,} obs: {inf_dur:.2f} s")

        pred_usd = reconstruct_price(y_pred_log, K_clean, intrinsic_val)
        all_preds_usd.append(pred_usd)

        if i == N_MODELS:
            last_model = model
        else:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Ensemble average in price space
    all_pred_usd = np.mean(np.stack(all_preds_usd, axis=0), axis=0)

    # --- Moneyness classification ---
    df_clean["Moneyness"] = classify_moneyness(S_clean, K_clean)

    # --- Save theoretical prices ---
    output_prices = df_clean[KEEP_COLS + ["Moneyness"]].copy()
    output_prices["MoE_price"] = all_pred_usd
    os.makedirs(os.path.dirname(OUTPUT_THEO), exist_ok=True)
    output_prices.to_csv(OUTPUT_THEO, index=False)
    print(f"\nTheoretical prices saved: {OUTPUT_THEO}")

    # --- Diagnostic: gate usage on a test sample (random 10K) ---
    print("\n--- Gating Network diagnostic (last model) ---")
    rng      = np.random.default_rng(0)
    test_idx = np.arange(val_end, n)
    sample   = rng.choice(test_idx, size=min(10_000, len(test_idx)), replace=False)
    analyze_gate_usage(
        last_model,
        X_all_s[sample],
        df_clean["Moneyness"].values[sample],
        device,
    )
    del last_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

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

    # --- Per-moneyness evaluation (test set) ---
    print("\n--- Per-moneyness evaluation (test set) ---")
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

    # ============================================================
    # COMPUTATIONAL PROFILING
    # ============================================================
    n_all  = n
    n_test = n - val_end

    # Inference throughput: total time / (N_MODELS × N_obs)
    # Divided by N_MODELS because total_inf_time covers 5x predict_target
    inf_us_per_obs_per_model = (total_inf_time / (N_MODELS * n_all)) * 1e6

    # Equivalent inference time for the test set only (5 full models)
    # Assumes linear throughput (valid for batched GPU inference)
    inf_test_equiv = total_inf_time * (n_test / n_all)

    print(f"\n--- Computational Profiling ---")
    print(f"  Hardware                          : {device}")
    print(f"  Training ensemble ({N_MODELS} models)       : "
          f"{total_train_time:>10.2f} s "
          f"({total_train_time/60:.2f} min)")
    print(f"  Inference total ({N_MODELS} × {n_all:,} obs) : "
          f"{total_inf_time:>10.2f} s")
    print(f"  Inference per obs per model       : "
          f"{inf_us_per_obs_per_model:>10.4f} µs")
    print(f"  Inference test-set equivalent     : "
          f"{inf_test_equiv:>10.2f} s "
          f"(for {N_MODELS} × {n_test:,} obs)")

    # --- Final summary ---
    elapsed = time.time() - start
    print(f"\n=== Done ({elapsed:.2f} s wall-clock, "
          f"device: {device}) ===")
    print(f"  {OUTPUT_THEO}")
    print(f"  {OUTPUT_OVR}")
    print(f"  {OUTPUT_MON}")


if __name__ == "__main__":
    main()