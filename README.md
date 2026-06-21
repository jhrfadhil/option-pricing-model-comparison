# SPX Option-Pricing Model Comparison

A reproducible pipeline that compares **classical** and **machine-learning** option-pricing
models on **S&P 500 (SPX) call options** (2019–2025), using **SOFR** as the risk-free rate.

The pipeline is a sequence of 15 numbered Python scripts that run in order; each script's
output feeds the next. Every script is path-portable and runs straight from a fresh clone
of this repository.

**Models compared**

- **Classical:** Black–Scholes, Bachelier (normal model), and a Monte Carlo
  Merton jump-diffusion simulation.
- **Machine learning:** four artificial-neural-network architectures — a baseline
  multilayer perceptron (MLP), a Residual network, a Mixture-of-Experts (MoE), and a
  Kolmogorov–Arnold Network (KAN).

**Inputs & evaluation.** Volatility is estimated from the S&P 500 index series using
historical estimators (HV5, HV20, HV60, HV100) plus GARCH and EGARCH forecasts. The test
set is the chronological **last 20%** of the sample. Models are compared on accuracy
(MAE / RMSE / MAPE), by pairwise **Diebold–Mariano** tests with the
Harvey–Leybourne–Newbold small-sample correction, and through granular breakdowns by
**moneyness × time-to-maturity** and **EGARCH volatility regime**.

> **Note.** No datasets are included in this repository. The scripts read external data
> from `data/raw/`, which you must download yourself (see [Data sources](#data-sources)).
> Full reproduction requires obtaining the licensed data described below.

---

## Repository structure

```
option-pricing-model-comparison/
├── README.md
├── requirements.txt
├── .gitignore
├── 01_combine_options_data.py
├── 02_merge_options_sofr.py
├── 03_filter_data.py
├── 04_volatility_forecast.py
├── 05_build_model_inputs.py
├── 06_descriptive_statistics.py
├── 07_black_scholes.py
├── 08_bachelier.py
├── 09_monte_carlo_jump_diffusion.py
├── 10_ann_mlp.py
├── 11_ann_residual.py
├── 12_ann_moe.py
├── 13_ann_kan.py
├── 14_diebold_mariano_test.py
├── 15_analysis.py
├── data/
│   ├── raw/         (.gitkeep; external downloads — NOT committed)
│   └── processed/   (.gitkeep; intermediate CSVs from scripts — NOT committed)
└── outputs/
    ├── descriptive_statistics_options.csv          (committed)
    ├── DM_results_squared_HLN.csv                  (committed)
    ├── table1_sample_properties.csv … table4_DM_by_category.csv   (committed)
    ├── Black-Scholes/   BS_accuracy_overall.csv, BS_accuracy_by_moneyness.csv (committed);
    │                    BS_theoretical_prices.csv (gitignored)
    ├── Bachelier/       (same pattern)
    ├── Monte Carlo/<10000|50000|100000|500000>/   (per simulation-path count; same pattern)
    └── ANN/<MLP|Residual|MoE|KAN>/                 (per architecture; same pattern)
```

All 15 scripts live at the repository root and **must be run from the root**, so that the
relative `data/` and `outputs/` paths resolve correctly.

---

## Requirements

- **Python 3.10+**
- The following third-party packages:

```
pandas
numpy
scipy
arch
tqdm
torch
scikit-learn
```

Install them with:

```bash
pip install -r requirements.txt
```

The ANN scripts (10–13) use PyTorch and will use a CUDA GPU if one is available;
otherwise they run on CPU.

---

## Data sources

**Nothing in `data/` is committed.** Download each dataset from its original source and
place the files in `data/raw/` using the exact filenames listed in
[Required raw files](#required-raw-files). The links below point to the original
providers — they are **not** mirrored or redistributed here.

| Dataset | Period | Source (download manually) | Terms |
|---|---|---|---|
| SPX option quotes | 2019–2023 | OptionsDX — <https://www.optionsdx.com/> | Free |
| SPX option quotes | 2024–2025 | Delta Neutral / Historical Option Data — <https://www.historicaloptiondata.com/> | Paid / licensed |
| SOFR (risk-free rate) | 2019–2025 | FRED, series `SOFR` — <https://fred.stlouisfed.org/series/SOFR> (original source: Federal Reserve Bank of New York — <https://www.newyorkfed.org/markets/reference-rates/sofr>) | Citation + disclaimer (see below) |
| S&P 500 index, **daily** | 2019–2025 | investing.com (Fusion Media) — <https://www.investing.com/indices/us-spx-500-historical-data> | Redistribution prohibited; link only |
| S&P 500 index, **weekly** | 2019–2025 | investing.com (Fusion Media) — same page, weekly interval | Redistribution prohibited; link only |

> The specific date ranges and export options vary by provider; download the equivalent
> coverage for the 2019–2025 study window. The provider links above are placeholders for
> manual download — confirm the exact product/series on each site before downloading.

### SOFR — citation and required disclaimer

SOFR is published by the **Federal Reserve Bank of New York** and is retrieved here via
**FRED** (Federal Reserve Bank of St. Louis), series `SOFR`. When using SOFR data, the New
York Fed requires that the following notice accompany it:

> The Secured Overnight Financing Rate (SOFR) data is sourced from newyorkfed.org and is
> subject to the Terms of Use posted at newyorkfed.org. The New York Fed is not responsible
> for publication of the SOFR data by this project, does not sanction or endorse any
> particular republication, and has no liability for use.
>
> The SOFR data is calculated using data provided under a license granted to the New York
> Fed by DTCC Solutions LLC, an affiliate of The Depository Trust & Clearing Corporation.
> Solutions, its affiliates, and third parties from which they obtained data have no
> liability for the content of this material.

*(See the New York Fed Terms of Use: <https://www.newyorkfed.org/privacy/termsofuse>.)*

### investing.com — link only

The S&P 500 index series (daily and weekly) are obtained from investing.com (Fusion Media).
Their terms prohibit redistribution, so these files are referenced by link only and are
**never** committed to this repository.

### Required raw files

Place these in `data/raw/` (filenames must match exactly):

| File | Used by | Description |
|---|---|---|
| `*.txt` (one per year, 2019–2023) | `01` | SPX call quotes, OptionsDX bracketed-column format |
| `SPX_2024.csv` | `01` | SPX call quotes, 2024 |
| `SPX_2025.csv` | `01` | SPX call quotes, 2025 |
| `SOFR.csv` | `02` | Daily SOFR series (from FRED) |
| `S&P 500 Historical Data.csv` | `04` | S&P 500 index, **daily** |
| `S&P 500 Weekly Historical Data.csv` | `09` | S&P 500 index, **weekly** |

---

## Run order

Run the scripts in numerical order from the repository root. Scripts `01`–`05` are strictly
sequential (each consumes the previous output). After `05`, the model scripts `06`–`13` read
from the prepared inputs; `14` and `15` must run last because they consume the model outputs.

```bash
python 01_combine_options_data.py
python 02_merge_options_sofr.py
python 03_filter_data.py
python 04_volatility_forecast.py
python 05_build_model_inputs.py
python 06_descriptive_statistics.py
python 07_black_scholes.py
python 08_bachelier.py
python 09_monte_carlo_jump_diffusion.py
python 10_ann_mlp.py
python 11_ann_residual.py
python 12_ann_moe.py
python 13_ann_kan.py
python 14_diebold_mariano_test.py
python 15_analysis.py
```

| # | Script | What it does |
|---|---|---|
| 01 | `01_combine_options_data.py` | Combines SPX call-option quotes from three source formats (2019–2023 `.txt`, 2024 `.csv`, 2025 `.csv`) into one unified dataset with a common schema. |
| 02 | `02_merge_options_sofr.py` | Attaches the SOFR risk-free rate to each option quote by date. |
| 03 | `03_filter_data.py` | Applies data-quality filters to produce the cleaned option dataset. |
| 04 | `04_volatility_forecast.py` | Estimates relative and absolute volatility from the **daily** S&P 500 series (historical HV5–HV100 plus GARCH/EGARCH forecasts). |
| 05 | `05_build_model_inputs.py` | Merges filtered options with volatility (as-of backward join, no look-ahead) and builds the three model-input files; the chronological last 20% becomes the test set. |
| 06 | `06_descriptive_statistics.py` | Computes descriptive statistics of the option sample → committed summary table. |
| 07 | `07_black_scholes.py` | Prices the test set with **Black–Scholes**; writes theoretical prices (gitignored) and accuracy tables (committed). |
| 08 | `08_bachelier.py` | Prices the test set with the **Bachelier** (normal) model; same output pattern. |
| 09 | `09_monte_carlo_jump_diffusion.py` | Prices the test set with a **Monte Carlo Merton jump-diffusion** simulation (calibrated on the **weekly** S&P 500 series), per simulation-path count. |
| 10 | `10_ann_mlp.py` | Trains and evaluates the **MLP** ANN; writes `MLP_price` theoretical prices and accuracy tables. |
| 11 | `11_ann_residual.py` | Trains and evaluates the **Residual** ANN; writes `Residual_price` outputs. |
| 12 | `12_ann_moe.py` | Trains and evaluates the **Mixture-of-Experts** ANN; writes `MoE_price` outputs. |
| 13 | `13_ann_kan.py` | Trains and evaluates the **Kolmogorov–Arnold Network**; writes `KAN_price` outputs. |
| 14 | `14_diebold_mariano_test.py` | Runs pairwise **Diebold–Mariano** tests (squared-error loss, Newey–West HAC variance, Harvey–Leybourne–Newbold small-sample correction) across the four models → `DM_results_squared_HLN.csv`. |
| 15 | `15_analysis.py` | Produces granular performance tables by **moneyness × maturity** and **EGARCH volatility regime**, plus DM tests by category → `table1`–`table4`. |

---

## Outputs: committed vs regenerated

Running the full pipeline regenerates **everything** under `data/` and `outputs/`. To keep
the repository free of licensed data, only the small aggregate result tables are committed;
the rest is produced locally when you run the pipeline.

**Committed to the repository** (small aggregate results, no row-level licensed data):

- `outputs/descriptive_statistics_options.csv`
- `outputs/DM_results_squared_HLN.csv`
- `outputs/table1_sample_properties.csv` … `outputs/table4_DM_by_category.csv`
- every `*_accuracy_overall.csv` and `*_accuracy_by_moneyness.csv`
  (under `Black-Scholes/`, `Bachelier/`, `Monte Carlo/<N>/`, and `ANN/<architecture>/`)

**Not committed — regenerated by running the pipeline:**

- every `*_theoretical_prices.csv` — these retain `C_LAST` (licensed market option price)
  and `UNDERLYING_LAST` (index level), so they are treated as licensed data and are
  gitignored.
- everything under `data/raw/` and `data/processed/`.

---

## Selecting the ANN architecture and Monte Carlo run for scripts 14–15

Scripts `14` and `15` compare one ANN architecture against the three classical models. By
default they point at the **Mixture-of-Experts (MoE)** output:

```
outputs/ANN/MoE/MoE_theoretical_prices.csv
```

**Switching architecture is path-only.** Change just the ANN entry in the `INPUT_FILES`
configuration at the top of `14_diebold_mariano_test.py` and `15_analysis.py` to point at a
different architecture's file (e.g. `outputs/ANN/KAN/KAN_theoretical_prices.csv`). The price
column (`MLP_price` / `Residual_price` / `MoE_price` / `KAN_price`) is **auto-detected**, so
no column edits are needed. The classical-model columns (`BS_EGARCH`, `Bach_EGARCH`,
`MCJD_EGARCH`) are fixed.

**Monte Carlo run.** Both scripts read the Monte Carlo output for `MCJD_N_PATHS = 500000`
by default. To compare a different simulation-path count, change that constant (it selects
the `outputs/Monte Carlo/<N_PATHS>/` subfolder).

---

## Reproducibility

Random seeds, model architectures, and hyperparameters are fixed in the scripts.
**However, full reproduction requires obtaining the licensed datasets** — in particular the
2024–2025 SPX option quotes (paid) and the investing.com S&P 500 index series (redistribution
prohibited). Those files are not, and cannot be, included here. Once the required raw files
are in `data/raw/`, running scripts `01`–`15` in order reproduces all processed data and
results.

---

## Citation, license, and contact

**Citation.** If you use this code, please cite the associated thesis:

> Muhammad Jauhar Fadhil (2026). *Option Pricing in the Post-LIBOR Era: A Comparative
> Analysis of Black–Scholes, Bachelier, Monte Carlo, and Neural Network Models on the
> S&P 500.* Undergraduate thesis (Skripsi), Universitas Syiah Kuala.

BibTeX:

```bibtex
@mastersthesis{fadhil2026optionpricing,
  author = {{Muhammad Jauhar Fadhil}},
  title  = {{Option Pricing in the Post-LIBOR Era: A Comparative Analysis of
            Black--Scholes, Bachelier, Monte Carlo, and Neural Network Models
            on the S\&P 500}},
  school = {Universitas Syiah Kuala},
  year   = {2026},
  type   = {Undergraduate thesis (Skripsi)},
  note   = {Code repository: \url{https://github.com/jhrfadhil/option-pricing-model-comparison}}
}
```

> If a DOI or published version becomes available later, add it as a `doi` or `url` field.

**License.** The code in this repository is released under the **MIT License** — see the
[`LICENSE`](LICENSE) file at the repository root. This license covers the **code only**. No
datasets are included; all external data remains subject to the terms of its original
provider (OptionsDX, Delta Neutral / historicaloptiondata.com, the Federal Reserve Bank of
New York / FRED for SOFR, and investing.com / Fusion Media). You must obtain and use those
datasets under their respective terms.

**Contact.** Muhammad Jauhar Fadhil — <jauharfadhil@protonmail.com>
(GitHub: [@jhrfadhil](https://github.com/jhrfadhil))
