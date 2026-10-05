import datetime
from pathlib import Path

BASE_DIR    = Path(__file__).parent
OUTPUTS_DIR = BASE_DIR / 'outputs'
CACHE_DIR   = BASE_DIR / 'cache'
OUTPUTS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

def create_run_dirs() -> tuple:
    """Create a timestamped output folder tree for a single bot run.

    Returns (run_dir, general_dir, correlation_dir).
    Folder name format: 'YYYY-MM-DD_HH-MM'
    """
    ts      = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    run_dir = OUTPUTS_DIR / ts
    gen_dir  = run_dir / 'General'
    corr_dir = run_dir / 'Correlation_method'
    run_dir.mkdir(exist_ok=True)
    gen_dir.mkdir(exist_ok=True)
    corr_dir.mkdir(exist_ok=True)
    print(f"  Run outputs → {run_dir}")
    return run_dir, gen_dir, corr_dir

# Interactive Brokers connection
IB_HOST   = '127.0.0.1'
IB_PORT   = 7497           # 7497 = paper trading | 7496 = live
CLIENT_ID = 1
ACCOUNT   = ''             # Empty → use default account

# Number of tickers to use in paper/signals/live runs.
# int              → top N companies by market cap (e.g. 50)
# None             → full S&P 500 (~503 tickers)
# 'FALLBACK_TICKERS' → hardcoded top-20 list, no web request
N_TICKERS = None

# Used as fallback when Wikipedia is unreachable, and for demo mode
FALLBACK_TICKERS = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOG',
    'META', 'LLY',  'TSLA', 'JPM',  'V',
    'UNH',  'XOM',  'JNJ',  'WMT',  'MA',
    'PG',   'HD',   'AVGO', 'COST', 'NFLX'
]

# Duplicate share classes to exclude (keep only the preferred class listed above)
EXCLUDED_TICKERS = {'GOOGL'}

# Number of top companies highlighted in price series and bar charts
TOP_N_HIGHLIGHT  = 15

# Data cache
PRICE_CACHE_OVERLAP_DAYS  = 15   # calendar days to re-fetch for split/dividend adjustments
MCAP_CACHE_MAX_AGE_HOURS  = 24   # hours before market-cap snapshot is considered stale

# Model parameters
HISTORY_DAYS     = 99999   # Use all cached history; set lower to limit analysis window
PREDICTION_DAYS  = 7       # Prediction horizon (days)
MIN_CORRELATION  = 0.50    # Minimum correlation to use as predictor
MIN_R2           = 0.01    # Informational only below (see TOP_N_POSITIONS) — kept for the
                            # LOW_CONFIDENCE display label and backward compat with
                            # analysis/backtest.py's simulate_from_signals()/run_backtest().
BUY_THRESHOLD    = 0.01    # Informational only below (see TOP_N_POSITIONS) — kept for the
                            # 'signal' display label and the same backward compat.
SELL_THRESHOLD   = -0.10   # Informational only — drives the 'signal' SELL display label;
                            # actual live selling is driven by falling out of the top-N
                            # selection at rebalance, or stop-loss/trailing-stop.
ORDER_QUANTITY        = 10       # Shares per order (paper trading)
MAX_POSITION_PCT      = 0.10     # Safety ceiling on any single position, even under equal-weight
TOP_N_POSITIONS       = 15       # Live trading selection: rank all tickers with predicted_return
                                  # > 0 and hold the top N, equal-weighted. Replaces BUY_THRESHOLD
                                  # (an absolute magnitude cutoff) + R2-scaled sizing — an October
                                  # 2026 backtest investigation found predicted-return magnitude
                                  # and R2 are both ~uncorrelated with outcome quality, while
                                  # rank-based selection with flat sizing held up out-of-sample
                                  # (~3x the Sharpe ratio of the old approach on an independent
                                  # window). See analysis/backtest.py's simulate_ranked_from_signals
                                  # and the README's Signal Selection / Auditing sections.
FALLBACK_PORTFOLIO    = 1_000.0  # Used when IB does not return NetLiquidation

# Engineered model features (analysis/model.py) — added October 2026 after research into
# proven return-predictability signals (momentum, liquidity, volatility) the correlation-only
# feature set didn't capture. All optional: predict_price() falls back to pure correlation
# features when prices/volume/vix aren't passed in, so existing call sites are unaffected.
MOMENTUM_LOOKBACK_DAYS = 60   # Trailing-return window for the target ticker's own momentum.
                              # Shorter than the classic 252-day equity-factor convention,
                              # scaled to fit this bot's shorter history/rebalance cadence.
MOMENTUM_SKIP_DAYS     = 5    # Skip the most recent N days (short-term reversal filter)
FEATURE_VOL_WINDOW     = 20   # Rolling window for Amihud illiquidity / realized volatility

# Risk management
STOP_LOSS_PCT      = 0.08   # Close a position if it falls X% below its entry price
TRAILING_STOP_PCT  = 0.05   # Close a position if it falls X% below its peak price since entry
MAX_DRAWDOWN_PCT   = 0.15   # Halt new BUY orders once portfolio equity drawdown from peak exceeds X%
RISK_STATE_FILE    = CACHE_DIR / 'risk_state.json'  # Persists entry/peak prices and equity peak across runs

# Backtesting (analysis/backtest.py)
BACKTEST_START_CAPITAL      = 10_000.0  # Simulated starting capital
BACKTEST_TRANSACTION_COST_PCT = 0.0010  # Commission per trade (10 bps)
BACKTEST_SLIPPAGE_PCT          = 0.0005  # Market-impact/slippage per trade (5 bps)
BACKTEST_REBALANCE_DAYS        = PREDICTION_DAYS  # Re-score signals every N trading days
BACKTEST_MIN_HISTORY_DAYS      = 120    # Warm-up window before the first rebalance
BACKTEST_LOOKBACK_DAYS         = 500    # ~2 trading years actually walked (excl. warm-up) —
                                         # fetch_prices_cached() returns the FULL cache (back
                                         # to each ticker's IPO), so this bounds runtime; each
                                         # rebalance fits one RandomForest per ticker

# DCA simulator (dashboard "Simulator" tab)
SIM_WEEKLY_AMOUNT_EUR = 100.0   # Fictitious amount invested every week
SIM_YEARS             = 2       # Lookback window
SIM_BENCHMARK_TICKER  = 'SPY'   # S&P 500 ETF proxy (tradable, unlike the ^GSPC index)
