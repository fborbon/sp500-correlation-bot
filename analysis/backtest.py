"""Walk-forward backtest of the correlation / Random Forest strategy.

Unlike the live pipeline (which fits on all available history and predicts
"today"), this replays history day by day: at each rebalance point only price
data up to and including that day is used to fit the model and generate
signals, exactly mirroring what the bot would have known live. Commissions,
slippage, and the same stop-loss / trailing-stop / max-drawdown rules used in
production (broker/risk.py) are applied to every simulated fill, so the
resulting equity curve reflects net-of-cost, out-of-sample performance rather
than an in-sample fit that looks better than it would trade.
"""
import numpy as np
import pandas as pd

from config import (BACKTEST_MIN_HISTORY_DAYS, BACKTEST_REBALANCE_DAYS,
                    BACKTEST_SLIPPAGE_PCT, BACKTEST_START_CAPITAL,
                    BACKTEST_TRANSACTION_COST_PCT, BUY_THRESHOLD, MAX_DRAWDOWN_PCT,
                    MAX_POSITION_PCT, MIN_R2, STOP_LOSS_PCT, TRAILING_STOP_PCT)
from analysis.correlations import compute_correlations
from analysis.model import predict_price


def _fill_price(price: float, side: str) -> float:
    """Effective execution price after slippage (worse than mid, both directions)."""
    slip = price * BACKTEST_SLIPPAGE_PCT
    return price + slip if side == 'BUY' else price - slip


def run_backtest(prices_df: pd.DataFrame, n_tickers: int = 20,
                 start_capital: float = None, verbose: bool = True) -> dict:
    """Backtest the strategy over the history already present in `prices_df`.

    Args:
        prices_df:     Close-price DataFrame (date index, one column per ticker),
                        e.g. loaded from cache/prices_cache.parquet. Columns should
                        already be ordered by market-cap rank (as universe.py does).
        n_tickers:      Universe size. Kept modest by default — each rebalance fits
                        one Random Forest (+3 CV folds) per ticker, so cost scales
                        linearly with this.
        start_capital:  Overrides config.BACKTEST_START_CAPITAL.

    Returns:
        {'equity_curve': DataFrame[date -> equity, benchmark_equity],
         'trades': DataFrame, 'metrics': dict}
    """
    capital = start_capital if start_capital is not None else BACKTEST_START_CAPITAL

    universe = prices_df.columns[:n_tickers].tolist() if n_tickers else list(prices_df.columns)
    prices = prices_df[universe].dropna(axis=1, how='any')
    universe = prices.columns.tolist()
    dates = prices.index

    if len(dates) <= BACKTEST_MIN_HISTORY_DAYS + BACKTEST_REBALANCE_DAYS:
        raise ValueError(
            f"Not enough price history ({len(dates)} rows) for a backtest — need at least "
            f"{BACKTEST_MIN_HISTORY_DAYS + BACKTEST_REBALANCE_DAYS} trading days."
        )

    cash = capital
    positions = {}         # ticker -> {'qty', 'entry_price', 'peak_price'}
    equity_curve = []
    trades = []
    portfolio_peak = capital
    halted = False
    rebalance_idxs = set(range(BACKTEST_MIN_HISTORY_DAYS, len(dates), BACKTEST_REBALANCE_DAYS))

    for i in range(BACKTEST_MIN_HISTORY_DAYS, len(dates)):
        today = dates[i]
        today_prices = prices.iloc[i]

        # 1. Daily stop-loss / trailing-stop check on open positions
        for ticker in list(positions.keys()):
            price = today_prices.get(ticker)
            if price is None or np.isnan(price):
                continue
            pos = positions[ticker]
            pos['peak_price'] = max(pos['peak_price'], price)
            hard_stop = price <= pos['entry_price'] * (1 - STOP_LOSS_PCT)
            trailing  = price <= pos['peak_price']  * (1 - TRAILING_STOP_PCT)
            if hard_stop or trailing:
                fill = _fill_price(price, 'SELL')
                proceeds = fill * pos['qty'] * (1 - BACKTEST_TRANSACTION_COST_PCT)
                cash += proceeds
                pnl = proceeds - pos['qty'] * pos['entry_price']
                trades.append({'date': today, 'ticker': ticker, 'action': 'SELL',
                               'reason': 'STOP_LOSS' if hard_stop else 'TRAILING_STOP',
                               'qty': pos['qty'], 'price': fill, 'pnl': pnl})
                del positions[ticker]

        # 2. Rebalance — re-score every ticker using ONLY data up to `today`
        if i in rebalance_idxs:
            hist_prices = prices.iloc[:i + 1]
            corr_matrix, returns = compute_correlations(hist_prices)

            for ticker, pos in list(positions.items()):
                price = today_prices.get(ticker)
                if price is None or np.isnan(price):
                    continue
                fill = _fill_price(price, 'SELL')
                proceeds = fill * pos['qty'] * (1 - BACKTEST_TRANSACTION_COST_PCT)
                cash += proceeds
                pnl = proceeds - pos['qty'] * pos['entry_price']
                trades.append({'date': today, 'ticker': ticker, 'action': 'SELL',
                               'reason': 'REBALANCE', 'qty': pos['qty'], 'price': fill, 'pnl': pnl})
            positions = {}

            equity_now = cash
            portfolio_peak = max(portfolio_peak, equity_now)
            drawdown = (portfolio_peak - equity_now) / portfolio_peak if portfolio_peak > 0 else 0.0
            halted = drawdown >= MAX_DRAWDOWN_PCT

            if verbose:
                status = ' [HALTED — drawdown limit]' if halted else ''
                print(f"  {today.date()}  equity=${equity_now:,.0f}{status}")

            if not halted:
                for ticker in universe:
                    pred_ret, r2, *_ = predict_price(ticker, returns, corr_matrix)
                    if pred_ret is None or r2 < MIN_R2 or pred_ret <= BUY_THRESHOLD:
                        continue

                    price = today_prices.get(ticker)
                    if price is None or np.isnan(price):
                        continue

                    strength = min(1.0, r2)
                    max_value = cash * MAX_POSITION_PCT * strength
                    qty = int(max_value / price)
                    if qty < 1:
                        continue

                    fill = _fill_price(price, 'BUY')
                    cost = fill * qty * (1 + BACKTEST_TRANSACTION_COST_PCT)
                    if cost > cash:
                        continue
                    cash -= cost
                    positions[ticker] = {'qty': qty, 'entry_price': fill, 'peak_price': fill}
                    trades.append({'date': today, 'ticker': ticker, 'action': 'BUY',
                                   'reason': 'SIGNAL', 'qty': qty, 'price': fill, 'pnl': None})

        # 3. Mark to market
        holdings_value = sum(
            pos['qty'] * today_prices[t]
            for t, pos in positions.items()
            if not np.isnan(today_prices.get(t, np.nan))
        )
        equity = cash + holdings_value
        portfolio_peak = max(portfolio_peak, equity)
        equity_curve.append({'date': today, 'equity': equity})

    equity_df = pd.DataFrame(equity_curve).set_index('date')

    # Buy-and-hold benchmark: equal-weight the same universe, bought on day 1
    bh_start_prices = prices.iloc[BACKTEST_MIN_HISTORY_DAYS]
    bh_shares = (capital / len(universe)) / bh_start_prices
    equity_df['benchmark_equity'] = (
        prices.iloc[BACKTEST_MIN_HISTORY_DAYS:] * bh_shares
    ).sum(axis=1).values

    trades_df = pd.DataFrame(trades)
    metrics = _compute_metrics(equity_df, trades_df, capital)

    return {'equity_curve': equity_df, 'trades': trades_df, 'metrics': metrics}


def _compute_metrics(equity_df: pd.DataFrame, trades_df: pd.DataFrame, capital: float) -> dict:
    equity = equity_df['equity']
    daily_returns = equity.pct_change().dropna()

    total_return_pct = (equity.iloc[-1] / capital - 1) * 100
    n_days = len(equity)
    cagr_pct = ((equity.iloc[-1] / capital) ** (252 / max(n_days, 1)) - 1) * 100

    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)
              if daily_returns.std() > 0 else 0.0)

    running_peak = equity.cummax()
    drawdown = (equity - running_peak) / running_peak
    max_drawdown_pct = drawdown.min() * 100 if len(drawdown) else 0.0

    closed = trades_df[trades_df['pnl'].notna()] if not trades_df.empty else trades_df
    win_rate_pct = (closed['pnl'] > 0).mean() * 100 if not closed.empty else 0.0

    benchmark_return_pct = (equity_df['benchmark_equity'].iloc[-1] / capital - 1) * 100

    return {
        'total_return_pct':      round(float(total_return_pct), 2),
        'cagr_pct':               round(float(cagr_pct), 2),
        'sharpe_ratio':           round(float(sharpe), 2),
        'max_drawdown_pct':       round(float(max_drawdown_pct), 2),
        'win_rate_pct':           round(float(win_rate_pct), 1),
        'num_trades':             int(len(trades_df)),
        'benchmark_return_pct':   round(float(benchmark_return_pct), 2),
        'final_equity':           round(float(equity.iloc[-1]), 2),
    }
