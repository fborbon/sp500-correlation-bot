"""Weekly DCA (dollar-cost averaging) simulator for the dashboard's Simulator tab.

Illustrative only: simulates investing a fixed EUR amount every week into an
S&P 500 ETF proxy, converting at the historical EUR/USD rate on each
contribution date. Unlike analysis/backtest.py this is a passive buy-and-hold
benchmark (no signals, no commissions/slippage) — it answers "what if I had
just kept investing a fixed amount every week?" rather than testing the bot's
own strategy.
"""
import pandas as pd
import yfinance as yf

from config import SIM_BENCHMARK_TICKER, SIM_WEEKLY_AMOUNT_EUR, SIM_YEARS


def simulate_weekly_dca(ticker: str = None, weekly_amount_eur: float = None,
                        years: int = None) -> dict:
    """Simulate a fixed weekly EUR contribution into `ticker` over `years` years.

    Returns {'curve': DataFrame[date -> invested_eur, value_eur], 'metrics': dict}.
    """
    ticker            = ticker or SIM_BENCHMARK_TICKER
    weekly_amount_eur = weekly_amount_eur or SIM_WEEKLY_AMOUNT_EUR
    years             = years or SIM_YEARS
    period = f'{years}y'

    price_raw = yf.download(ticker, period=period, interval='1d',
                            auto_adjust=True, progress=False)
    fx_raw = yf.download('EURUSD=X', period=period, interval='1d',
                         auto_adjust=True, progress=False)

    if price_raw.empty:
        raise ValueError(f"No price data returned for {ticker}.")

    price = price_raw['Close'][ticker].dropna()
    fx    = fx_raw['Close']['EURUSD=X'].reindex(price.index).ffill().bfill()

    df = pd.DataFrame({'price_usd': price, 'eurusd': fx}).dropna()

    # One contribution every 7 calendar days, snapped to the next trading day.
    contribution_targets = pd.date_range(df.index[0], df.index[-1], freq='7D')
    contribution_dates = sorted({
        df.index[df.index >= d][0]
        for d in contribution_targets
        if len(df.index[df.index >= d])
    })

    shares = 0.0
    invested_eur = 0.0
    contrib_set = set(contribution_dates)
    rows = []

    for date, row in df.iterrows():
        if date in contrib_set:
            usd_amount = weekly_amount_eur * row['eurusd']
            shares += usd_amount / row['price_usd']
            invested_eur += weekly_amount_eur

        value_eur = (shares * row['price_usd']) / row['eurusd']
        rows.append({'date': date, 'invested_eur': invested_eur, 'value_eur': value_eur})

    curve = pd.DataFrame(rows).set_index('date')

    final_value    = float(curve['value_eur'].iloc[-1]) if len(curve) else 0.0
    total_invested = float(curve['invested_eur'].iloc[-1]) if len(curve) else 0.0
    total_return_pct = ((final_value / total_invested) - 1) * 100 if total_invested else 0.0
    n_days = len(curve)
    cagr_pct = (((final_value / total_invested) ** (365 / max(n_days, 1)) - 1) * 100
                if total_invested and n_days else 0.0)

    metrics = {
        'ticker':             ticker,
        'weekly_amount_eur':  weekly_amount_eur,
        'years':              years,
        'num_contributions':  len(contribution_dates),
        'total_invested_eur': round(total_invested, 2),
        'final_value_eur':    round(final_value, 2),
        'profit_eur':         round(final_value - total_invested, 2),
        'total_return_pct':   round(total_return_pct, 2),
        'cagr_pct':           round(cagr_pct, 2),
    }

    return {'curve': curve, 'metrics': metrics}
