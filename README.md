# SP500 Correlation Bot

Algorithmic trading bot for Interactive Brokers that predicts short-term 7-day price movements using Pearson correlations across the full S&P 500 universe. Both direct and inverse correlations are used as features for a Random Forest model, with the universe size configurable at runtime to always select the **N most valuable** companies by market cap. Price data is fetched free from Yahoo Finance; Interactive Brokers is only connected when live order placement is enabled. Every position is protected by a stop-loss / trailing-stop, and a portfolio-level max-drawdown circuit breaker halts new buys if losses accumulate — the strategy itself is validated by a walk-forward backtester that charges realistic commissions and slippage before any of this touches real money. A Streamlit dashboard visualises signals, fundamentals, correlation heatmaps, the backtest equity curve, and a weekly DCA simulator after each run.

**Main technologies:** Python · scikit-learn (Random Forest) · pandas · yfinance · ib-insync (Interactive Brokers API) · Streamlit · Plotly · pyarrow

**Monthly cost:** $0. Yahoo Finance data is free; IB connection is local. No cloud services, no paid APIs, no subscriptions required. Running on a local machine or a shared EC2 instance adds no incremental cost.

---

## Table of Contents

1. [Installation](#installation)
2. [Libraries](#libraries)
3. [Usage](#usage)
4. [Data Processing Pipeline](#data-processing-pipeline)
5. [Data Flow Diagram](#data-flow-diagram)
6. [Predictive Model](#predictive-model)
7. [File Structure](#file-structure)
8. [Key Configuration (`config.py`)](#key-configuration-configpy)
9. [Selecting Companies](#selecting-companies)
10. [Signal Selection](#signal-selection)
11. [Position Sizing](#position-sizing)
12. [Risk Management](#risk-management)
13. [Backtesting](#backtesting)
14. [Weekly DCA Simulator](#weekly-dca-simulator)
15. [Inverse Correlation Logic](#inverse-correlation-logic)
16. [Output Plots (`save_plots=True`)](#output-plots-saveplotstrue)
17. [Auditing](#auditing)
18. [CI/CD](#cicd)

---

## Installation

```bash
pip install ib_insync pandas numpy scikit-learn plotly kaleido requests yfinance streamlit nest_asyncio
```

- Interactive Brokers TWS or IB Gateway running locally (only needed to place orders)
- Port `7497` for paper trading (recommended for testing)
- Port `7496` for live trading (use with caution)

> **Price data** is fetched from Yahoo Finance via `yfinance` (free, no IB needed).
> IB is only connected when `execute_trades=True` to place the actual orders.

---

## Libraries

| Library | Version | Role in this project |
|---|---|---|
| **pandas** | 3.0.2 | Core data structure throughout the pipeline. All price matrices, returns, signals, and fundamentals are `DataFrame` objects. Used for `pct_change()`, `corr()`, CSV/Parquet I/O, and data alignment. |
| **numpy** | 2.4.4 | Numerical operations in the model and charting layers: cumulative-return arrays, `polyfit` trend lines, `corrcoef` for R² computation in charts, and array slicing for downsampling. |
| **scikit-learn** | 1.8.0 | Provides the `RandomForestRegressor` that predicts 7-day returns and `TimeSeriesSplit` for walk-forward cross-validation. The only ML framework used. |
| **yfinance** | 1.3.0 | Primary market data source (no API key needed). Used to download close prices, daily volume, market capitalisations, and per-ticker fundamental metadata (`Ticker.info`). |
| **plotly** | 6.7.0 | All charts — correlation heatmaps, price-series, market-cap bars, volume series, cumulative returns, and prediction scatter plots — are built with Plotly's `graph_objects` and `make_subplots`. |
| **kaleido** | 0.2.1 | Renders Plotly figures to static PNG files without a browser. Pinned to 0.2.1 because this version bundles its own renderer and does not require Chrome/Chromium. |
| **streamlit** | 1.57.0 | Powers the interactive web dashboard (`dashboard.py`). Renders signals tables, fundamentals, price charts, and correlation heatmaps from the output CSVs in a browser. |
| **ib-insync** | 0.9.86 | Python wrapper for the Interactive Brokers TWS/Gateway API. Used exclusively when `execute_trades=True` to qualify contracts, retrieve portfolio value, and place market or limit orders. |
| **requests** | 2.33.1 | HTTP client used to scrape the Wikipedia S&P 500 constituents page and fetch company metadata (name, sector, founding year). |
| **beautifulsoup4 / lxml** | 4.14.3 / 6.1.0 | HTML parsers invoked indirectly via `pandas.read_html()` when extracting the S&P 500 table from Wikipedia. `lxml` is the fast back-end parser. |
| **pyarrow** | 24.0.0 | Parquet serialisation for the incremental price and volume caches (`cache/prices_cache.parquet`, `cache/volume_cache.parquet`). Enables column-efficient storage and fast partial reads. |
| **joblib** | 1.5.3 | Used internally by scikit-learn to parallelise tree construction across all available CPU cores (`n_jobs=-1`). No direct calls in the project code. |
| **nest_asyncio** | — | Patches Python's event loop to allow `ib_insync`'s async calls inside Jupyter notebooks. Only needed when running `Main.ipynb`. |

> **Note on AI technologies:** This project uses only classical machine learning (Random Forest) from scikit-learn. No generative AI, large language models, retrieval-augmented generation, chatbots, speech-to-text, image diffusion, or agentic AI frameworks are used.

---

## Usage

```bash
# Demo mode — no IB required, synthetic data with inverse correlations
python main.py demo

# Top 50 companies by market cap, signals only
python main.py signals 50

# Top 100 companies, paper trading
python main.py paper 100

# Full S&P 500 (~503 tickers), paper trading
python main.py paper

# Live trading — real money, requires manual confirmation
python main.py live 50

# Walk-forward backtest — top 20 tickers, net of commissions/slippage
python main.py backtest 20

# MIN_R2 sensitivity sweep on the full universe — one expensive pass, cheap sweep
python main.py sweep full
```

Or use `Main.ipynb` in Jupyter — set `n_tickers` and `mode` in the run cell.

### Dashboard

After running the bot, launch the dashboard to explore results interactively:

```bash
streamlit run dashboard.py
```

Opens in your browser at `http://localhost:8501`. Nine tabs:
- **Signals** — colour-coded BUY/SELL/HOLD table
- **Fundamentals** — 10-metric scoring table with likelihood_pct highlighted
- **Mkt Cap / Prices / Volume / Returns / Correlation** — all charts from `General/` and `Correlation_method/`
- **Backtest** — walk-forward equity curve vs buy-and-hold, from the latest `python main.py backtest` run
- **Simulator** — weekly DCA calculator (fixed EUR amount into an S&P 500 ETF over N years)

---

## Data Processing Pipeline

The bot executes a fixed seven-stage pipeline on each run.

### Stage 1 — Universe Selection (`analysis/universe.py`)

The S&P 500 constituent list is scraped from Wikipedia's *List of S&P 500 companies* HTML table using `requests` + `pandas.read_html`. When a specific `n` is requested, `yfinance.fast_info['market_cap']` is called in parallel (10 threads) for all ~503 tickers to obtain real-time market capitalisations. Results are sorted descending and only the top-N are kept. Market caps are cached in `cache/market_caps_cache.json` with a 24-hour TTL to avoid redundant network calls on repeated runs.

### Stage 2 — Data Ingestion (`broker/data.py`)

Historical daily **close prices** and **trading volume** are downloaded in a single `yfinance.download()` batch call (all tickers at once, much faster than one-by-one). Data is persisted in Parquet format (`cache/prices_cache.parquet`, `cache/volume_cache.parquet`). On subsequent runs only the new days are fetched (incremental update), with a 15-day overlap window to capture dividend/split adjustments. `HISTORY_DAYS` controls how many trading days are retained for analysis.

When `execute_trades=True`, the pipeline additionally connects to Interactive Brokers via `ib_insync` to retrieve the portfolio's net liquidation value and to place orders.

### Stage 3 — Correlation Analysis (`analysis/correlations.py`)

Daily percentage returns are computed from the close-price DataFrame using `pct_change()`. A full **Pearson correlation matrix** is then calculated across all tickers with `DataFrame.corr(method='pearson')`. This produces an N×N symmetric matrix where each cell holds the linear correlation coefficient r ∈ [−1, 1] between the daily return streams of two stocks. Both positive (co-moving) and negative (counter-moving) relationships are preserved.

### Stage 4 — Prediction Model (`analysis/model.py`)

For each target ticker the model:

1. **Selects predictors** — any ticker whose absolute Pearson r with the target is ≥ `MIN_CORRELATION` (default 0.50).
2. **Constructs the feature matrix** — `X` = daily returns of all selected predictors; `y` = N-day cumulative return of the target starting from the same day (`PREDICTION_DAYS = 7`).
3. **Cross-validates** — `TimeSeriesSplit(n_splits=3)` creates three non-overlapping, chronologically ordered train/validation splits. The mean R² across splits is the model's confidence score.
4. **Trains the final model** — a `RandomForestRegressor` is re-fit on the full available history.
5. **Predicts** — the model predicts the 7-day forward return from the most recent feature vector.

See the [Predictive Model](#predictive-model) section for full details.

### Stage 5 — Fundamental Analysis (`analysis/fundamentals.py`)

Ten fundamental metrics are fetched per ticker from `yfinance.Ticker.info` using 10 parallel threads (with 2 retries each). Each metric is scored against calibrated thresholds and multiplied by its weight; scores are summed to a `likelihood_pct` (0–100) representing the probability that the stock price will be higher in 12 months. The 10 metrics and their weights are:

| Metric | Weight | Rationale |
|---|---|---|
| Revenue growth | 15 | Growing top line |
| Gross margin | 10 | Pricing power / moat |
| Operating margin | 10 | Operational efficiency |
| Free cash flow | 10 | Real cash generation |
| Current ratio | 10 | Short-term solvency |
| Debt-to-equity | 10 | Leverage risk |
| P/E ratio | 10 | Valuation vs earnings |
| PEG ratio | 10 | Valuation vs growth |
| Earnings growth | 10 | Profit momentum |
| P/B ratio | 5 | Valuation vs book value |

### Stage 6 — Signal Generation (`analysis/signals.py`)

Signals are generated by applying threshold rules to the model's predicted return and confidence score:

| Condition | Signal |
|---|---|
| R² < `MIN_R2` (0.01) | `LOW_CONFIDENCE` |
| predicted return > `BUY_THRESHOLD` (+1%) | `BUY` |
| predicted return < `SELL_THRESHOLD` (−10%) | `SELL` |
| otherwise | `HOLD` |
| insufficient predictors | `INSUF_DATA` |

Each signal row also lists the top-5 predictors split into `direct_top5_predictors` (r > 0) and `inverse_top5_predictors` (r < 0).

### Stage 7 — Reporting & Export (`reporting/`)

`report.py` prints the signal table to the console and saves `signals.csv`. `charts.py` renders nine PNG files (see [Output Plots](#output-plots-save_plotstrue)). Fundamental data is saved to `fundamentals.csv`. All CSV and PNG outputs are written into a timestamped `outputs/YYYY-MM-DD_HH-MM/` folder. The Streamlit dashboard reads these files to render the interactive UI.

---

## Data Flow Diagram

```mermaid
flowchart TD
    CLI([Start · main.py / Main.ipynb])

    CLI --> A

    subgraph A["1 · Universe Selection  (analysis/universe.py)"]
        A1[Wikipedia HTML scrape\n~503 S&P 500 tickers] --> A3
        A2["yfinance market caps\n(cache/market_caps_cache.json · 24 h TTL)"] --> A3
        A3[Sort by market cap → select top-N]
    end

    A --> B

    subgraph B["2 · Data Ingestion  (broker/data.py)"]
        B1["yfinance batch download → Close prices\n(cache/prices_cache.parquet · incremental)"]
        B2["yfinance batch download → Daily volume\n(cache/volume_cache.parquet · incremental)"]
    end

    B --> C

    subgraph C["3 · Correlation Analysis  (analysis/correlations.py)"]
        C1[daily_returns = prices.pct_change] --> C2
        C2["Pearson correlation matrix  (N × N)"]
    end

    C --> D

    subgraph D["4 · Prediction Model  (analysis/model.py)  · loop per ticker"]
        D1["Filter predictors  |r| ≥ 0.50"] --> D2
        D2["X = predictor daily returns\ny = 7-day cumulative return"] --> D3
        D3["TimeSeriesSplit  n = 3 folds\ncross-validate → mean R²"] --> D4
        D4["Final fit: RandomForestRegressor\n200 trees · max_depth = 4 · min_samples_leaf = 10"] --> D5
        D5["Predict next 7-day return  +  corr_signs"]
    end

    A --> E

    subgraph E["5 · Fundamental Analysis  (analysis/fundamentals.py)  · parallel"]
        E1["yfinance.Ticker.info · 10 threads · 2 retries"] --> E2
        E2["10 raw metrics: P/E · PEG · P/B\nmargins · FCF · current ratio · D/E · growth"] --> E3
        E3["Weighted score → likelihood_pct  (0–100)"]
    end

    D --> F

    subgraph F["6 · Signal Generation  (analysis/signals.py)"]
        F1{"R² ≥ MIN_R2 = 0.01?"}
        F1 -->|No| F2[LOW_CONFIDENCE]
        F1 -->|Yes| F3{"predicted return"}
        F3 -->|"> +1%"| F4[BUY]
        F3 -->|"< −10%"| F5[SELL]
        F3 -->|otherwise| F6[HOLD]
    end

    F --> G
    E --> G
    C --> G
    B --> G

    subgraph G["7 · Reporting  (reporting/charts.py · report.py)"]
        G1["signals.csv · fundamentals.csv\nprices.csv · volume.csv"]
        G2["PNG charts\nGeneral/  ·  Correlation_method/"]
    end

    G --> H[("Streamlit Dashboard\ndashboard.py · localhost:8501")]

    F -->|"execute_trades = True"| IB[("Interactive Brokers\nbroker/orders.py\nMarket orders")]
```

---

## Predictive Model

### Algorithm: Random Forest Regressor

`predict_price()` (`analysis/model.py`) uses scikit-learn's `RandomForestRegressor`. A Random Forest builds many independent decision trees on random sub-samples of the training data and averages their predictions. This ensemble approach provides several properties that make it well-suited to this task:

- **Non-linearity** — captures interactions between correlated stocks that a simple linear regression cannot (e.g. a pair with r = 0.70 may only be predictive when a third stock is also up).
- **Scale invariance** — decision trees split on rank order, not magnitude, so all predictor returns (which vary in scale) feed directly into the model without normalisation or `StandardScaler`.
- **Outlier robustness** — averaging over 200 trees smooths out the effect of extreme return days that would skew a linear model.
- **Implicit feature selection** — trees with low-information splits are averaged away; correlated predictors compete for splits rather than inflating coefficients as in OLS regression.

### Why not a linear model?

Stock-to-stock return relationships are conditionally nonlinear. Two stocks that are highly correlated on average may decorrelate during sector rotations, earnings seasons, or macro shocks. A forest of shallow trees (`max_depth=4`) captures these regime-dependent patterns while the depth limit prevents overfitting to noise.

### Configuration

| Parameter | Value | Effect |
|---|---|---|
| `n_estimators` (cross-val) | 100 | Faster CV pass; enough trees for a stable R² estimate |
| `n_estimators` (final fit) | 200 | Double the trees for the production prediction |
| `max_depth` | 4 | Limits each tree to 4 decision levels — shallow trees generalise better on financial time-series |
| `min_samples_leaf` | 10 | A leaf must represent at least 10 observations; prevents the model from memorising micro-patterns in small training sets |
| `random_state` | 42 | Reproducible results across runs |
| `n_jobs` | −1 | Uses all available CPU cores to build trees in parallel |

### Walk-Forward Cross-Validation

`TimeSeriesSplit(n_splits=3)` is used instead of the standard k-fold to respect temporal order:

```
Fold 1:  [train ───────]  [val ─]
Fold 2:  [train ──────────]  [val ─]
Fold 3:  [train ─────────────]  [val ─]
```

Each fold extends the training window forward in time; the validation set is always *after* the training set. This prevents look-ahead bias — the model is never evaluated on data that preceded its training period. The mean R² across the three folds is the confidence score reported in `model_r2`.

### Feature Engineering

The target label is the **cumulative return** of the target stock over the next `PREDICTION_DAYS` (default 7) trading days:

```python
y[i] = sum(daily_returns[i : i + 7])   # ≈ 7-day total return
```

The base feature set is the contemporaneous daily returns of all correlated predictor stocks (the cross-sectional structure). As of October 2026, four additional engineered features are appended when `prices`/`volume`/`vix` are passed to `predict_price()` (see `_engineered_features()`), added after research into the return-predictability literature identified momentum, liquidity, and volatility as signals the correlation-only design didn't capture:

| Feature | Construction | Source |
|---|---|---|
| `momentum` | Trailing `MOMENTUM_LOOKBACK_DAYS` (60) return, skipping the most recent `MOMENTUM_SKIP_DAYS` (5) days | Target's own price history (already cached) |
| `realized_vol` | Rolling `FEATURE_VOL_WINDOW` (20) day std of daily returns, annualized | Target's own price history (already cached) |
| `illiquidity` | Amihud ratio — rolling mean of `\|return\| / dollar_volume` | Target's own price + volume history (already cached) |
| `vix_level` | CBOE VIX close, forward-filled | `broker/data.py`'s `fetch_vix_cached()` — same yfinance provider, new `cache/vix_cache.parquet` |

All four are **optional and additive**: a source not passed (e.g. `vix=None`) simply omits that column rather than crashing or forcing the model to use a placeholder — see the no-lookahead and missing-source handling in `analysis/model.py`'s `_engineered_features()`. No explicit technical indicators (RSI, MACD, etc.) are computed beyond these four.

---

## File Structure

```
V3/
├── config.py               # All constants + OUTPUTS_DIR path
├── main.py                 # run_bot(n_tickers) + CLI entry point
├── demo.py                 # run_demo() — synthetic data, no IB connection needed
│
├── broker/
│   ├── __init__.py
│   ├── connection.py       # connect_ib(), get_contract(), nest_asyncio fix
│   ├── data.py             # fetch_prices() via IB; fetch_prices_free() via yfinance;
│   │                       # fetch_vix_cached() — market-wide volatility feature
│   ├── orders.py           # execute_order(), close_position(), calculate_position_size()
│   └── risk.py             # stop-loss / trailing-stop / max-drawdown guard,
│                           # persisted to cache/risk_state.json
│
├── analysis/
│   ├── __init__.py
│   ├── universe.py         # get_sp500_tickers(n) → (tickers, caps); fetch_market_caps();
│   │                       # fetch_company_metadata() — name, sector, founded, cap ($B)
│   ├── correlations.py     # compute_correlations(), get_top_correlated_pairs(),
│   │                       # get_top_inverse_pairs()
│   ├── fundamentals.py     # fetch_fundamentals(), score_fundamentals(),
│   │                       # save_fundamentals_csv() — 10-metric scoring → likelihood_pct
│   ├── model.py            # predict_price() — RandomForestRegressor + TimeSeriesSplit,
│   │                       # + optional momentum/illiquidity/realized_vol/vix_level
│   │                       # engineered features; returns corr_signs, y_actual, y_predicted
│   ├── signals.py          # generate_signals() — BUY/SELL/HOLD with
│   │                       # direct_top5_predictors / inverse_top5_predictors
│   ├── backtest.py         # run_backtest() — walk-forward, no-lookahead replay with
│   │                       # commissions/slippage + the same risk.py rules
│   └── simulator.py        # simulate_weekly_dca() — weekly EUR→ETF DCA calculator
│                           # for the dashboard's Simulator tab
│
├── reporting/
│   ├── __init__.py
│   ├── charts.py           # plot_correlation_matrix(); plot_price_series();
│   │                       # plot_market_cap_bars(); plot_prediction_analysis()
│   └── report.py           # print_report() — direct (↑↑) and inverse (↑↓) pair
│                           # sections; save_signals_csv()
│
├── cache/                  # Local data cache — git-ignored contents
│   ├── prices_cache.parquet
│   ├── volume_cache.parquet
│   ├── market_caps_cache.json
│   ├── vix_cache.parquet   # CBOE VIX history, written by broker/data.py's fetch_vix_cached()
│   ├── risk_state.json     # entry/peak prices + equity peak, written by broker/risk.py
│   └── backtest_latest/    # written by `python main.py backtest` — read by the dashboard
│       ├── equity_curve.csv
│       ├── trades.csv
│       └── metrics.json
│
├── outputs/                # Generated files — git-ignored contents
│   ├── 2026-05-07_14-30/   # timestamped folder per run (YYYY-MM-DD_HH-MM)
│   │   ├── signals.csv
│   │   ├── prices.csv          # raw close prices — used by dashboard for interactive charts
│   │   ├── fundamentals.csv
│   │   ├── General/
│   │   │   ├── price_series_market-cap.png
│   │   │   ├── price_series_stock-price-absolute.png
│   │   │   ├── price_series_normalized-return.png
│   │   │   ├── market_cap_bars.png
│   │   │   ├── market_cap_series_absolute.png
│   │   │   ├── market_cap_series_normalized.png
│   │   │   ├── volume_series_absolute.png
│   │   │   ├── volume_series_normalized.png
│   │   │   └── cumulative_returns.png
│   │   └── Correlation_method/
│   │       ├── correlation_matrix.png
│   │       └── analysis_{TICKER}.png
│   └── demo_signals.csv    # demo mode outputs (no timestamp)
│
├── Main.ipynb              # Jupyter entry point
└── dashboard.py            # Streamlit dashboard
```

---

## Key Configuration (`config.py`)

| Variable | Default | Description |
|---|---|---|
| `IB_HOST` | `127.0.0.1` | TWS / IB Gateway host |
| `IB_PORT` | `7497` | 7497 = paper, 7496 = live |
| `HISTORY_DAYS` | `99999` | Trading days of history (99999 = use full cache) |
| `PREDICTION_DAYS` | `7` | Forecast horizon (days) |
| `MIN_CORRELATION` | `0.50` | Minimum absolute Pearson r to use a predictor (direct or inverse) |
| `MIN_R2` | `0.01` | Informational only — drives the display-only `LOW_CONFIDENCE` label, not live selection (see [Signal Selection](#signal-selection)) |
| `BUY_THRESHOLD` | `0.01` | Informational only — drives the display-only `BUY` label, not live selection |
| `SELL_THRESHOLD` | `-0.10` | Informational only — drives the display-only `SELL` label, not live selection |
| `TOP_N_POSITIONS` | `15` | Live trading: top N tickers by predicted return (among positive predictions) held, equal-weighted |
| `MAX_POSITION_PCT` | `0.10` | Per-position safety ceiling — no single position exceeds 10% of portfolio even under equal-weight |
| `TOP_N_HIGHLIGHT` | `15` | Companies highlighted in price series and bar charts |
| `FALLBACK_PORTFOLIO` | `1000.0` | Portfolio value used when IB does not return NetLiquidation |
| `FALLBACK_TICKERS` | top 20 | Used when all online sources fail |
| `PRICE_CACHE_OVERLAP_DAYS` | `15` | Calendar days re-fetched on incremental update (for adjustments) |
| `MCAP_CACHE_MAX_AGE_HOURS` | `24` | Hours before market-cap snapshot is considered stale |
| `STOP_LOSS_PCT` | `0.08` | Close a position 8% below its entry price |
| `TRAILING_STOP_PCT` | `0.05` | Close a position 5% below its peak price since entry |
| `MAX_DRAWDOWN_PCT` | `0.15` | Halt new BUY orders once equity drawdown from peak exceeds 15% |
| `BACKTEST_START_CAPITAL` | `10000.0` | Simulated starting capital for `python main.py backtest` |
| `BACKTEST_TRANSACTION_COST_PCT` | `0.0010` | Commission per simulated fill (10 bps) |
| `BACKTEST_SLIPPAGE_PCT` | `0.0005` | Slippage per simulated fill (5 bps) |
| `BACKTEST_REBALANCE_DAYS` | `7` | Trading days between backtest re-scoring points |
| `SIM_WEEKLY_AMOUNT_EUR` | `100.0` | Default weekly contribution in the DCA Simulator tab |
| `SIM_YEARS` | `2` | Default lookback window in the DCA Simulator tab |
| `SIM_BENCHMARK_TICKER` | `SPY` | ETF used as the S&P 500 proxy in the DCA Simulator |
| `MOMENTUM_LOOKBACK_DAYS` | `60` | Trailing-return window for the `momentum` model feature |
| `MOMENTUM_SKIP_DAYS` | `5` | Most-recent days excluded from momentum (short-term reversal filter) |
| `FEATURE_VOL_WINDOW` | `20` | Rolling window for the `illiquidity` / `realized_vol` model features |

---

## Selecting Companies

`get_sp500_tickers(n)` fetches the **N most valuable** S&P 500 companies at runtime:

| Source | Order | Used when |
|---|---|---|
| Wikipedia + yfinance sort | By actual market cap ✓ | Always (when `n` is specified) |
| Wikipedia only | Alphabetical | `n=None` (full universe) |
| `FALLBACK_TICKERS` | Hardcoded top-20 | Wikipedia unavailable, or `n_tickers='FALLBACK_TICKERS'` |

When `n` is specified, all ~503 Wikipedia tickers are fetched then sorted by real yfinance market caps (~30s) before slicing to top N. Plots also re-sort by market cap so `price_series.png` and `market_cap_bars.png` always show the correct companies.

`n_tickers` accepts three types:
- **`int`** — top N companies by market cap (e.g. `50`)
- **`None`** — full S&P 500 (~503 tickers)
- **`'FALLBACK_TICKERS'`** — hardcoded top-20 list, no web request

---

## Signal Selection

**As of October 2026, live trading selection is rank-based, not threshold-based.** `analysis/signals.py`'s `generate_signals()` still computes and displays the legacy per-ticker `signal` label (`BUY`/`SELL`/`HOLD`/`LOW_CONFIDENCE`/`INSUF_DATA`, based on the absolute `BUY_THRESHOLD`/`MIN_R2`/`SELL_THRESHOLD` cutoffs) purely for informational display — it no longer drives trading. What actually gets traded is the separate `selected` column: **the top `TOP_N_POSITIONS` tickers by predicted return among those with a positive prediction, regardless of raw magnitude or R².**

This replaced the original design after a multi-day backtest investigation (October 2026) found, and then validated out-of-sample:

- **`BUY_THRESHOLD` (an absolute magnitude cutoff) was filtering for noise, not against it.** A sweep found predicted-return magnitude is essentially uncorrelated with R² or realized outcome quality (r ≈ −0.03 to −0.05). Replacing the magnitude cutoff with rank-based selection (always trade the top N candidates, whatever their raw values) nearly **3×'d the Sharpe ratio** (0.39 → 1.10) on an independent, non-overlapping out-of-sample window — a real, reproducible improvement.
- **R²-scaled position sizing (`strength = min(1.0, r2)`) was a structural bug, not a calibration nuance.** Real R² values from this model are almost always well below 1.0 (frequently negative), so at realistic share prices the old formula produced `qty = 0` — no trade at all — for nearly every signal that wasn't already near-maximum confidence, and went negative (silently skipped) whenever R² < 0.

See [Auditing](#auditing) for the full investigation, including an equally important negative result: a longer 21-day prediction horizon looked like a further improvement on one universe/window, then completely reversed (best → worst result in the whole study) on retest — it was not shipped.

---

## Position Sizing

The spend per order is calculated in `broker/orders.py`'s `calculate_position_size()`: **equal-weight across the tickers selected this run**, capped by `MAX_POSITION_PCT` as a per-position safety ceiling.

```
portfolio_value / n_selected,  capped at  portfolio_value × MAX_POSITION_PCT
```

- `TOP_N_POSITIONS = 15` → up to 15 equal-weight positions held at once
- `MAX_POSITION_PCT = 0.10` → no single position exceeds 10% of portfolio even if fewer than 15 names are selected
- Returns `0` (not a forced minimum of 1 share) when the allocated budget can't afford even one share at that price — the old formula's `max(1, ...)` could force-buy one share of an expensive stock on a tiny allocated budget, silently blowing through the intended position size.

Example — $100,000 portfolio, 10 tickers selected this run, buying a $300 stock:
```
equal_weight_value = 100,000 / 10          = $10,000
max_value           = min(10,000, 10,000)   = $10,000   (MAX_POSITION_PCT cap not binding here)
quantity             = int(10,000 / 300)     = 33 shares
```

To change position count or sizing, adjust `TOP_N_POSITIONS` or `MAX_POSITION_PCT` in `config.py`.

---

## Risk Management

Position sizing alone does not stop a loss from running — it only caps the size of the initial bet. `broker/risk.py` adds three independent guards, all evaluated at the start of every trading run (`execute_trades=True`) before any new order is placed:

| Guard | Config | Behaviour |
|---|---|---|
| **Stop-loss** | `STOP_LOSS_PCT = 0.08` | Closes a position if its price falls 8% below the recorded entry price. |
| **Trailing stop** | `TRAILING_STOP_PCT = 0.05` | Closes a position if its price falls 5% below the highest price observed since entry — locks in gains on winners instead of only protecting against losses. |
| **Max drawdown (circuit breaker)** | `MAX_DRAWDOWN_PCT = 0.15` | If portfolio equity has fallen 15% from its all-time run peak, **new BUY orders are halted for that run** (SELL orders and stop-loss closes still execute — de-risking is never blocked). |

Entry price, peak price, and the portfolio equity peak are persisted to `cache/risk_state.json` between runs (each cron invocation is a fresh process). The flow inside `main.py`'s `execute_trades` block is:

1. Load `risk_state.json`.
2. Check every open position's current price against its stop-loss / trailing-stop; close any breach immediately.
3. Recompute portfolio value, update the equity peak, and evaluate the drawdown circuit breaker.
4. Walk the BUY/SELL signal table — BUY orders are skipped entirely if the circuit breaker is active; every filled BUY registers its entry price for future stop-loss checks.
5. Save `risk_state.json`.

---

## Backtesting

`analysis/backtest.py` answers the question the live pipeline can't: *would this strategy actually have made money, net of costs?* It replays history day by day rather than fitting once on "all data up to today":

- At every rebalance point (`BACKTEST_REBALANCE_DAYS = PREDICTION_DAYS`, i.e. weekly) the model is fit and scored using **only price history available up to that day** — no lookahead into the future.
- Every simulated fill pays `BACKTEST_TRANSACTION_COST_PCT` (10 bps commission) and `BACKTEST_SLIPPAGE_PCT` (5 bps slippage), applied against the strategy, not in its favour.
- The same stop-loss, trailing-stop, and max-drawdown rules from [Risk Management](#risk-management) run inside the simulation, so the backtest tests the whole system, not just the raw model.
- Positions are fully liquidated and rebuilt at each rebalance from the fresh signal set; an equal-weight buy-and-hold of the same universe is tracked in parallel as the benchmark.

Run it with:
```bash
python main.py backtest 20     # top 20 tickers by market cap
python main.py backtest full   # full ~502-ticker universe — several hours; see below
```

Output (`cache/backtest_latest/` — deliberately not under `outputs/`, where the dashboard treats the newest folder as the latest run and the weekly cron deletes all but the newest):
- `equity_curve.csv` — daily strategy equity vs. the buy-and-hold benchmark
- `trades.csv` — every simulated fill with reason (`SIGNAL` / `STOP_LOSS` / `TRAILING_STOP` / `REBALANCE`) and P&L
- `metrics.json` — total return, CAGR, Sharpe ratio, max drawdown, win rate, trade count, benchmark return

The dashboard's **Backtest** tab reads these files directly. `fetch_prices_cached()` returns each ticker's full history back to its IPO, so the walked window is capped by `BACKTEST_LOOKBACK_DAYS` (default ~2 trading years, plus a `BACKTEST_MIN_HISTORY_DAYS` warm-up) to keep runtime bounded — without this cap a `python main.py backtest` run would try to replay 60+ years of history. Even bounded, this is compute-heavy: each rebalance fits one Random Forest (+ 3 walk-forward CV folds) *per ticker*, so a 2-year run takes roughly 1–2 seconds per ticker per rebalance — about 20–40 minutes for 20 tickers, and several hours for the full ~502-ticker universe (deliberately not the default; you have to type `full`). This is an offline analysis tool meant to be run occasionally (e.g. after a config change) and left to finish in the background on the server — not part of the live trading path, and not something the dashboard triggers on page load.

### MIN_R2 sweep

Walk-forward R² on a next-week-return regression is very often negative — the model explains less variance than a flat forecast would — so `MIN_R2 = 0.01` can end up blocking *every* signal on a given universe, which is exactly what a real ~502-ticker/2-year run produced: 0 trades in 72 rebalances while a naive buy-and-hold of the same universe returned +167%. Re-running the whole backtest once per candidate `MIN_R2` value would multiply an already multi-hour job, so `analysis/backtest.py` splits the work in two:

- `compute_signal_history()` — the expensive step, one Random Forest (+3 CV folds) per ticker per rebalance, run **once**.
- `simulate_from_signals()` — the cheap step, replaying rebalance/stop-loss/drawdown logic against those cached signals for a single `MIN_R2` value; runs in well under a second regardless of universe size.

`run_min_r2_sweep()` calls the expensive step once and the cheap step once per candidate threshold, so comparing a dozen `MIN_R2` values costs almost nothing extra over a single backtest. It also prints the R² distribution among tickers that already cleared `BUY_THRESHOLD` — the range `MIN_R2` actually has to work with — which is usually far below 0 (individual tickers scoring R² as low as −1.2 are common).

```bash
python main.py sweep 20     # top 20 tickers
python main.py sweep full   # full universe — pays the expensive step once, sweeps for free
```

Output (`cache/backtest_sweep/`):
- `summary.csv` — one row per `MIN_R2` candidate: trades, return, CAGR, Sharpe, max drawdown, win rate, benchmark return
- `signal_history.csv` — the raw `(date, ticker, pred_return, r2)` history, so *additional* thresholds can be tried later by calling `simulate_from_signals()` directly on this file, without recomputing anything

`run_sweep_cli()` also refreshes `cache/backtest_latest/` with the `config.MIN_R2` result from the same signal pass, so the dashboard's Backtest tab reflects the sweep run too.

---

## Weekly DCA Simulator

The dashboard's **Simulator** tab is a separate, much simpler tool: not a backtest of the bot's own signals, but a plain dollar-cost-averaging calculator — *"what if I had just invested a fixed amount every week?"* It simulates contributing `SIM_WEEKLY_AMOUNT_EUR` (default €100) every week into `SIM_BENCHMARK_TICKER` (default `SPY`, the tradable S&P 500 ETF proxy) over `SIM_YEARS` (default 2), converting each contribution at that week's historical EUR/USD rate (`analysis/simulator.py`, via `yfinance`). No fees, spread, or tax are modelled — it exists purely as an always-available, honest baseline to compare the bot's own performance against passive investing, and is adjustable live from the tab (weekly amount, years) without touching `config.py`.

---

## Inverse Correlation Logic

Stocks with absolute Pearson r ≥ 0.50 relative to the target are included as predictors regardless of sign. The Random Forest assigns the correct weight to each — inverse correlators (r < 0) contribute negatively to the prediction. The report labels them **↓ inverso** and lists the most negatively correlated pairs under **↑↓ TOP 5 PARES CORRELACIÓN INVERSA**.

---

## Output Plots (`save_plots=True`)

**`Correlation_method/correlation_matrix.png`** — Pearson correlation heatmap for all analyzed tickers.

**`Correlation_method/analysis_{TICKER}.png`** — generated for the top 5 signals by predicted return:
- **Left** — scatter of actual vs predicted cumulative returns (1:1 aspect ratio) with R² trend line.
- **Right** — normalized price time series of the target + its top 5 correlated tickers. Direct correlators as solid lines, inverse correlators as dashed lines.

**`General/price_series_market-cap.png`** — Two-subplot price time series. Top 15 by market cap in distinct colors; all others in light gray.
- **Top subplot** — normalized prices (base = 100).
- **Bottom subplot** — absolute close prices ($).

**`General/price_series_stock-price-absolute.png`** — Same layout but top 15 highlighted by highest last close stock price ($).

**`General/price_series_normalized-return.png`** — Same layout but top 15 highlighted by highest normalized return (best performers: largest % gain from the start of the history window).

**`General/market_cap_bars.png`** — Two-subplot bar chart for top 15 and bottom 15 companies by market cap.
- **Top subplot** — last closing stock price ($).
- **Bottom subplot** — market capitalization ($B / $T) from yfinance.

**`General/market_cap_series_absolute.png`** / **`market_cap_series_normalized.png`** — Market cap time series for top 15 companies (estimated as `close_price × shares_outstanding`), shown as absolute $B and normalised to base 100.

**`General/volume_series_absolute.png`** / **`volume_series_normalized.png`** — Daily trading volume time series, highlighted by average volume and normalized volume growth respectively.

**`General/cumulative_returns.png`** — Cumulative return (%) and dollar return ($ per share) since the start of the history window, with top 15 best performers highlighted.

---

## Auditing

This section provides a structured checklist for review by an IT expert and a quantitative-finance / algorithmic-trading subject-matter expert.

### Audit Items

- **Cost & resource minimization** — $0. Yahoo Finance data is free; IB connection is local. No cloud services or paid APIs are used. Parquet caching and market-cap TTL minimize redundant network calls.
- **IT architecture** — Seven-stage pipeline with clean module separation (analysis / broker / reporting). Parquet incremental caching avoids full re-downloads. Paper trading mode (`port 7497`) provides a safe testing environment before live deployment. The Streamlit dashboard reads output CSV files, decoupling analysis from visualization.
- **Code efficiency** — Parquet incremental update with a 15-day overlap window correctly handles dividend/split adjustments. Parallel fundamentals fetching (10 threads, 2 retries) minimizes I/O wait. Market-cap cache (24h TTL) avoids redundant yfinance calls. `n_jobs=-1` parallelizes Random Forest tree construction across all CPU cores.
- **Cybersecurity** — Interactive Brokers credentials are managed by TWS/Gateway locally; no API keys are stored in the project. Live trading requires an explicit manual confirmation step. All data sources (Yahoo Finance, Wikipedia, IB) are accessed over standard HTTPS/local socket connections.
- **Readability & maintainability** — All constants are centralized in `config.py`. The walk-forward cross-validation rationale and each model hyperparameter are documented. The signal generation logic is compact and auditable.
- **AI / ML model adequacy** — Random Forest with `TimeSeriesSplit` is sound as a modeling choice, but an October 2026 investigation (below) found walk-forward R² on the 7-day return regression is negative more often than not (median ≈ −0.07 even among signals the model is bullish on), and is essentially uncorrelated with predicted-return magnitude or realized outcome quality (r ≈ −0.03 to −0.05). Pearson correlation assumes linear relationships; non-linear cross-stock dependencies are not captured at the predictor-selection stage. No configuration tested — across MIN_R2, BUY_THRESHOLD, position count, three market-cap tiers, and two prediction horizons — beat simply holding the index. See [Strategy Validation Findings](#strategy-validation-findings-october-2026) below for the full investigation.
- **Financial risk** — Live trading operates with real money. [Stop-loss, trailing-stop, and a portfolio max-drawdown circuit breaker](#risk-management) are now enforced on every run (`broker/risk.py`), and the strategy is validated net of commissions/slippage by a [walk-forward backtester](#backtesting) before being trusted live. `MAX_POSITION_PCT=0.10` limits per-position concentration, but multiple correlated BUY signals can still create sector concentration — the drawdown circuit breaker is the backstop for that scenario, not a substitute for diversification. Correlation-based strategies historically break down during market dislocations (e.g., credit crises, flash crashes); the backtest windows used so far do not include a crisis period, so the max-drawdown guard remains the primary defense against a genuine regime break.
- **Other** — Wikipedia HTML scraping for S&P 500 constituents is fragile; a format change could break the entire universe-selection stage. yfinance data quality and availability are not guaranteed and should not be the sole data source for live trading decisions.

### Strategy Validation Findings (October 2026)

A multi-day investigation tested whether the correlation/Random-Forest strategy has any validated edge, and whether its confidence thresholds were well-calibrated. Full methodology and raw results are preserved in `analysis/backtest.py` (`run_min_r2_sweep`, `simulate_ranked_from_signals`, `compute_momentum_signal`) and this conversation's scripts; the findings below are the ones that changed the live code.

1. **`MIN_R2` sweep (full ~497-ticker universe, 2024–2026 window)** — at the original production default (`MIN_R2=0.01`), the strategy traded 349 times for essentially a breakeven return (−0.01%) while an equal-weight buy-and-hold of the same universe returned **+40.33%**. Every threshold from −2.0 to +0.01 produced *identical* results — later traced to a separate bug (next finding), not a genuine plateau. Only a much stricter `MIN_R2=0.10` showed a thin positive edge (8 trades, Sharpe +0.93) — too few trades to trust.
2. **Root cause found: `BUY_THRESHOLD` and R²-scaled sizing were both broken.** A grid search crossing `MIN_R2` with `BUY_THRESHOLD` showed *loosening* `BUY_THRESHOLD` to 0% (trade any positive prediction, not just ones exceeding +1%) consistently **beat** the original cutoff at every `MIN_R2` level — the model's predicted-return *magnitude* carries ~no quality signal, so requiring a bigger one filters for noise. Separately, `strength = min(1.0, r2)` was found to produce `qty = 0` (no trade at all) at realistic share prices for nearly any real R² value, and to go negative (silently skipped) whenever R² < 0 — explaining the earlier "identical results from −2.0 to +0.01": negative-R² candidates were being zeroed out by the sizing formula regardless of the nominal threshold.
3. **Out-of-sample validation #1: CONFIRMED.** Replaying the exact `BUY_THRESHOLD=0` fix on a non-overlapping earlier window (2021-10 → 2024-04, 100 tickers) reproduced the improvement independently: Sharpe 0.39 → 1.10 at `MIN_R2=0.01`. This is why the fix shipped — see [Signal Selection](#signal-selection).
4. **Universe-tier sensitivity (top/middle/bottom 20 by market cap, rank-based strategy).** None of nine strategy/universe combinations beat buy-and-hold. Top-20 (megacap) lost money on every strategy tested while the universe itself returned +122.73% — consistent with megacaps being the most efficiently-priced, hardest-to-beat segment. Bottom-20's buy-and-hold itself *lost* 32%; all three active strategies lost markedly less (−6.7% to −12.8%) than passive holding — a real, if modest, capital-preservation benefit from the risk management, not alpha generation.
5. **21-day horizon: tested, then disconfirmed.** Academic research on return predictability suggests short horizons are the noisiest (motivating a longer-horizon test). On the recent window, middle-20 + 21-day was the single best result in the whole investigation (+3.42% return, Sharpe +0.22). Out-of-sample validation #2, replaying the *exact same* 20 tickers on the 2021–2024 window, reversed it completely: **-17.79% return, Sharpe -1.18, 0% win rate** — the single worst result in the study. Not shipped; this closes the question rather than leaving it open for re-litigation without new evidence.
6. **Momentum/liquidity/volatility/VIX features: implemented and shipped (see [Feature Engineering](#feature-engineering)), tested, no improvement found.** Motivated by the Gu-Kelly-Xiu finding that price-correlation alone isn't among the dominant real return-predictability signals. Momentum and Amihud illiquidity compute from data already cached (no new source); VIX required one new fetch (`fetch_vix_cached()`, same yfinance provider). A direct with/without comparison on a 30-ticker out-of-sample window (2021-10 → 2024-04) found **no improvement**: total return was statistically identical (−15.68% vs −15.70%), while Sharpe was slightly worse (−0.83 → −1.04) and win rate dropped (31.4% → 24.0%, on fewer trades — 70 vs 50, since the engineered features' rolling/lookback windows drop additional rows to NaN). The features remain live (they're additive and don't hurt when unused elsewhere), but this is one test on one universe/window — not proof the features never help, just no evidence yet that they do. A natural next step, not yet done, would be testing whether they help specifically during high-VIX regimes rather than averaged across a whole window.

**Net result:** the rank-based selection + equal-weight sizing fix is real and shipped (validated out-of-sample, ~3× Sharpe improvement over the old defaults). It is a bug fix, not a discovered edge — no tested configuration, including the fixed one and the engineered-features one, outperforms simply holding the index. This matches the academic literature on cross-sectional return prediction (e.g., Gu, Kelly & Xiu 2020): individual-stock signal-to-noise is low, aggregation into ranked portfolios is what survives. The dominant real signals the literature identifies (momentum, liquidity, volatility) are now available as model inputs, but adding them hasn't yet moved the needle — the honest conclusion after this whole investigation is that the model's limiting factor may not be which features it has, but something more fundamental (universe/window choice, the RandomForest architecture itself, or the underlying premise that this signal set predicts short-horizon equity returns at all).

### Summary Table

| Audit Item | Claude's Assessment | Human Expert Assessment |
|---|---|---|
| Cost & resource minimization | $0. Parquet + TTL caching minimize redundant yfinance calls. No cloud dependency. | |
| IT architecture | Seven-stage pipeline with clean module separation. Paper trading mode is a correct safeguard. | |
| Code efficiency | Incremental Parquet update, parallel fundamentals fetch, and CPU-parallel Random Forest are all appropriate. | |
| Cybersecurity | IB credentials managed locally by TWS. No API keys in code. Live trading has manual confirmation gate. | |
| Readability & maintainability | Configuration centralized in config.py. Hyperparameter and model rationale well-documented. | |
| AI / ML model adequacy | Random Forest with TimeSeriesSplit is appropriate as a modeling choice. Validated (Oct 2026): R² is ~uncorrelated with predicted-return quality; no tested config beats buy-and-hold. Rank-based selection fix shipped (confirmed out-of-sample); 21-day horizon tested and disconfirmed out-of-sample. | |
| Financial risk | Stop-loss, trailing-stop, and max-drawdown circuit breaker enforced (broker/risk.py). Correlated BUY signals could still create sector concentration. Strategy vulnerable to market dislocation events not yet seen in backtest windows. | |
| Other | Wikipedia scraping for constituents is fragile. yfinance is not a guaranteed production data source. No backtesting framework for signal validation. | |


---

## CI/CD

### What is CI/CD?

**CI/CD** stands for **Continuous Integration / Continuous Deployment**. It is a software engineering practice that automates the steps of verifying, packaging, and releasing code whenever a change is pushed to the repository.

- **Continuous Integration (CI)** — each push triggers automated checks that confirm the change doesn't break the codebase.
- **Continuous Deployment (CD)** — once checks pass, the new version is automatically shipped to the live environment with no manual steps.

### Key Benefits

| Benefit | Description |
|---|---|
| **Speed** | Changes go live in seconds, not hours |
| **Consistency** | Every deploy follows the exact same steps — no human error |
| **Safety** | Broken code is caught before it reaches production |
| **Traceability** | Every deployment is linked to a specific commit and author |
| **Zero-downtime iterations** | Small frequent releases are safer than large rare ones |

### How it is applied here

The SP500 Bot runs as a **Docker Compose** service on a shared EC2 instance (`t3.small`, `eu-west-1`). The Streamlit dashboard is containerised — the Python code is copied into the image at build time (`COPY . .` in the Dockerfile). On every push to `main`, GitHub Actions SSH-es into the EC2, resets the code, and rebuilds the Docker image with `docker compose up -d --build`. The `cache/` and `outputs/` volumes are mounted and preserved across rebuilds so historical data and run artefacts are never lost. If the directory isn't yet a git repository (first deploy), the workflow initialises it and adds the remote automatically.

**Trigger:** push to `main`
**Runner:** `ubuntu-latest` (GitHub-hosted)
**Secrets required:** `EC2_HOST`, `EC2_USER`, `EC2_SSH_KEY`

### Implementation Diagram

```mermaid
flowchart TD
    DEV([👨‍💻 Developer\npushes to main])
    GH[GitHub repository\nfborbon/sp500-correlation-bot]
    GA[GitHub Actions\nubuntu-latest runner]
    SSH[appleboy/ssh-action\nSSH connection]
    EC2[EC2 t3.small\n54.78.82.101]
    INIT{.git present\nat /opt/forwardforecasting?}
    CLONE[git init + add remote]
    FETCH[git fetch origin main]
    RESET[git reset --hard origin/main]
    BUILD[docker compose up -d --build\nrebuilds image with new code]
    VOL[(cache/ + outputs/\nvolumes preserved)]
    DONE[✅ Dashboard live at\n/SP500bot]

    DEV --> GH
    GH --> GA
    GA --> SSH
    SSH -->|authenticated via\nEC2_SSH_KEY secret| EC2
    EC2 --> INIT
    INIT -->|No| CLONE --> FETCH
    INIT -->|Yes| FETCH
    FETCH --> RESET
    RESET --> BUILD
    BUILD --> VOL
    BUILD --> DONE

    style DEV fill:#4a90d9,color:#fff
    style DONE fill:#27ae60,color:#fff
    style EC2 fill:#e67e22,color:#fff
    style BUILD fill:#2c3e50,color:#fff
    style VOL fill:#7f8c8d,color:#fff
```

### Disk Space Maintenance

Two separate weekly cron jobs keep the shared EC2 instance from filling up — split because one is host-wide (not specific to this project) and one is specific to this repo's own log output:

- **`/home/ubuntu/weekly-disk-cleanup.sh`** (Sundays 04:00, not part of this repo — lives directly on the host since it also covers job-hunter-suite, windward, energy-trader, and anything else sharing the instance). Prunes Docker's build cache and dangling images/containers, and vacuums the systemd journal down to 200MB. Docker build cache in particular grows unbounded with every `docker compose up -d --build` across every project on the box — in October 2026 it reached 17GB before a cleanup.
- **`scripts/run_paper_daily.sh`** (Mondays 16:15 ET, this repo — see [CI/CD](#how-it-is-applied-here)) already kept only the newest `outputs/` folder; it now also deletes its own `logs/paper_*.log` files older than 30 days. These had grown unbounded since the first deploy (106 files, 375MB by October 2026) because nothing previously cleaned them — the host-level script above doesn't know this path exists, so retention belongs here, next to the code that creates them.

Neither job touches `cache/prices_cache.parquet`, `volume_cache.parquet`, `vix_cache.parquet`, `risk_state.json`, or the current `outputs/` run — only build artifacts, old logs, and already-superseded output folders.
