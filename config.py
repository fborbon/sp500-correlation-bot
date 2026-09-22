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
MIN_R2           = 0.01    # Minimum R² to trust the signal. =0.40
BUY_THRESHOLD    = 0.01    # Predicted return > X% → BUY
SELL_THRESHOLD   = -0.10   # Predicted return < -X% → SELL
ORDER_QUANTITY        = 10       # Shares per order (paper trading)
MAX_POSITION_PCT      = 0.10     # Maximum % of portfolio per position
FALLBACK_PORTFOLIO    = 1_000.0  # Used when IB does not return NetLiquidation

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
