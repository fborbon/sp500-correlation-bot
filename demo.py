import datetime

import numpy as np
import pandas as pd

from config import OUTPUTS_DIR
from analysis.correlations import compute_correlations, get_top_correlated_pairs, get_top_inverse_pairs
from analysis.signals import generate_signals
from reporting.charts import plot_correlation_matrix
from reporting.report import print_report, save_signals_csv


def run_demo():
    """Run the bot in demo mode using synthetic prices — no IB connection needed.

    Uses 20 synthetic tickers with realistic correlation structure:
    tech stocks are directly correlated with each other, while defensive
    sectors (utilities, healthcare) are inversely correlated with tech.
    """
    print("=" * 62)
    print("  SP500 CORRELATION BOT — DEMO MODE")
    print("  (Synthetic data · No IB connection needed)")
    print("=" * 62)

    np.random.seed(42)
    # NOT HISTORY_DAYS (99999 -- that's "don't limit the real cached history", not a literal
    # simulated-day count): compounding synthetic daily returns for ~100,000 steps decays
    # every price toward zero (geometric compounding of noise), breaking the demo entirely.
    n_days = 500
    base_prices = {
        'AAPL': 189.0,  'MSFT': 412.0,  'NVDA': 876.0,  'AMZN': 185.0,
        'GOOGL': 167.0, 'META': 502.0,  'LLY': 734.0,   'TSLA': 178.0,
        'JPM': 198.0,   'V': 274.0,     'UNH': 521.0,   'XOM': 114.0,
        'JNJ': 157.0,   'WMT': 64.0,    'MA': 461.0,    'PG': 162.0,
        'HD': 342.0,    'AVGO': 1312.0, 'COST': 728.0,  'NFLX': 615.0,
    }

    # Sector factor loadings:
    #   tech → positive market-factor weight  (direct correlation with market)
    #   defensive → negative weight           (inverse correlation with market)
    tech      = ['AAPL', 'MSFT', 'NVDA', 'META', 'GOOGL', 'AVGO', 'NFLX', 'AMZN', 'TSLA']
    defensive = ['JNJ', 'PG', 'WMT', 'COST', 'UNH']

    market_factor = np.random.normal(0, 0.008, n_days)

    prices = {}
    for ticker, base in base_prices.items():
        idio = np.random.normal(0, 0.012, n_days)
        if ticker in tech:
            weight = 0.7
        elif ticker in defensive:
            weight = -0.3    # moves against the market factor → inverse corr with tech
        else:
            weight = 0.3     # financials / energy: mild positive
        prices[ticker] = base * np.cumprod(1 + market_factor * weight + idio)

    prices_df = pd.DataFrame(prices)
    prices_df.index = pd.date_range(end=datetime.date.today(), periods=n_days, freq='B')

    corr_matrix, returns = compute_correlations(prices_df)
    top_pairs     = get_top_correlated_pairs(corr_matrix)
    inverse_pairs = get_top_inverse_pairs(corr_matrix)
    signals_df    = generate_signals(prices_df, returns, corr_matrix)

    print_report(signals_df, top_pairs, inverse_pairs)
    save_signals_csv(signals_df, OUTPUTS_DIR / 'demo_signals.csv')
    plot_correlation_matrix(corr_matrix, OUTPUTS_DIR / 'demo_correlation_matrix.png')
    print(f"\n✓ Demo complete. Check the outputs/ folder.")
