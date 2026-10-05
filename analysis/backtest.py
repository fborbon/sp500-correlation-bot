"""Walk-forward backtest of the correlation / Random Forest strategy.

Unlike the live pipeline (which fits on all available history and predicts
"today"), this replays history day by day: at each rebalance point only price
data up to and including that day is used to fit the model and generate
signals, exactly mirroring what the bot would have known live. Commissions,
slippage, and the same stop-loss / trailing-stop / max-drawdown rules used in
production (broker/risk.py) are applied to every simulated fill, so the
resulting equity curve reflects net-of-cost, out-of-sample performance rather
than an in-sample fit that looks better than it would trade.

The expensive step — fitting one Random Forest (+3 walk-forward CV folds) per
ticker at every rebalance date — is separated from the cheap step of turning
those (predicted_return, r2) pairs into trades under a given MIN_R2/
BUY_THRESHOLD. This lets many threshold combinations be compared (see
run_min_r2_sweep) without re-fitting anything: compute_signal_history() runs
once, simulate_from_signals() replays the same signals in milliseconds per
threshold.
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


def _prep_universe(prices_df: pd.DataFrame, n_tickers: int):
    universe = prices_df.columns[:n_tickers].tolist() if n_tickers else list(prices_df.columns)
    prices = prices_df[universe].dropna(axis=1, how='any')
    universe = prices.columns.tolist()
    dates = prices.index
    if len(dates) <= BACKTEST_MIN_HISTORY_DAYS + BACKTEST_REBALANCE_DAYS:
        raise ValueError(
            f"Not enough price history ({len(dates)} rows) for a backtest — need at least "
            f"{BACKTEST_MIN_HISTORY_DAYS + BACKTEST_REBALANCE_DAYS} trading days."
        )
    return universe, prices, dates


def compute_signal_history(prices_df: pd.DataFrame, n_tickers: int = 20,
                           verbose: bool = True, lookahead: int = None,
                           rebalance_days: int = None, volume_df: pd.DataFrame = None,
                           vix: pd.Series = None) -> tuple:
    """Walk forward once, scoring every ticker at every rebalance date.

    This is the expensive step: one RandomForestRegressor fit (+3 CV-fold fits)
    per ticker per rebalance, using only price history up to (and including)
    that day — no lookahead. Runtime scales linearly with n_tickers and with
    the number of rebalance points (history length / rebalance_days).

    Args:
        lookahead:      Prediction horizon in trading days, passed straight to
                        predict_price(). Defaults to config.PREDICTION_DAYS (7).
                        Academic cross-sectional return prediction (e.g. Gu,
                        Kelly & Xiu) studies monthly horizons — short horizons
                        like 7 days sit at the noisiest end of the signal-to-
                        noise spectrum, so this is exposed to test longer ones.
        rebalance_days: How often to re-score and rebuild positions. Defaults
                        to `lookahead` if given (so positions are held exactly
                        as long as the prediction horizon they're based on —
                        rebalancing weekly against a 21-day forecast would
                        close positions before the predicted move has time to
                        happen), else config.BACKTEST_REBALANCE_DAYS.
        volume_df:      Optional — enables the Amihud illiquidity feature (see
                        analysis/model.py). Truncated to each rebalance date
                        before use, same no-lookahead discipline as prices.
        vix:            Optional — enables the market-wide VIX-level feature.
                        Also truncated per rebalance date.

    Returns (universe, prices, signal_df) where signal_df has one row per
    (date, ticker) with columns ['pred_return', 'r2'] — everything a strategy
    needs to decide BUY/SELL, without touching the model again.
    """
    universe, prices, dates = _prep_universe(prices_df, n_tickers)
    rebalance_days = rebalance_days or lookahead or BACKTEST_REBALANCE_DAYS
    rebalance_pts = list(range(BACKTEST_MIN_HISTORY_DAYS, len(dates), rebalance_days))

    rows = []
    for k, i in enumerate(rebalance_pts, 1):
        today = dates[i]
        hist_prices = prices.iloc[:i + 1]
        # Truncated to `today` (not the un-sliced full series) so an engineered feature can
        # never see a future value, regardless of how analysis/model.py's reindex logic
        # might change later — same no-lookahead discipline as hist_prices.
        hist_volume = volume_df.loc[:today] if volume_df is not None else None
        hist_vix    = vix.loc[:today] if vix is not None else None
        corr_matrix, returns = compute_correlations(hist_prices)

        for ticker in universe:
            pred_ret, r2, *_ = predict_price(ticker, returns, corr_matrix, lookahead=lookahead,
                                             prices=hist_prices, volume=hist_volume, vix=hist_vix)
            rows.append({'date': today, 'ticker': ticker, 'pred_return': pred_ret, 'r2': r2})

        if verbose:
            print(f"  [{k}/{len(rebalance_pts)}] {today.date()}  scored {len(universe)} tickers",
                  flush=True)

    signal_df = pd.DataFrame(rows)
    return universe, prices, signal_df


def compute_momentum_signal(prices_df: pd.DataFrame, n_tickers: int = 20,
                            lookback: int = 90, skip: int = 5,
                            rebalance_days: int = None, verbose: bool = True) -> tuple:
    """Pure price-based momentum benchmark — no model fitting at all.

    Standard factor-investing momentum: rank by trailing return over `lookback`
    days, skipping the most recent `skip` days (short-term reversal filter).
    Classic equity momentum uses a 252-day lookback / 21-day skip (~12mo minus
    the most recent month); scaled down here (90/5) to fit inside the same
    BACKTEST_MIN_HISTORY_DAYS warm-up window used for the RF strategy, so both
    can be compared on identical rebalance dates.

    Returns the same (universe, prices, signal_df) shape as
    compute_signal_history() — with 'r2' fixed at 1.0 (unused by
    simulate_ranked_from_signals, which never scales size by it) — so it can
    be run through the exact same simulator as the RF-based strategy.
    """
    universe, prices, dates = _prep_universe(prices_df, n_tickers)
    rebalance_days = rebalance_days or BACKTEST_REBALANCE_DAYS
    rebalance_pts = list(range(BACKTEST_MIN_HISTORY_DAYS, len(dates), rebalance_days))

    rows = []
    for k, i in enumerate(rebalance_pts, 1):
        today = dates[i]
        start_px = prices.iloc[i - lookback - skip]
        end_px   = prices.iloc[i - skip]
        momentum = (end_px / start_px) - 1
        for ticker in universe:
            rows.append({'date': today, 'ticker': ticker,
                        'pred_return': float(momentum[ticker]), 'r2': 1.0})
        if verbose:
            print(f"  [{k}/{len(rebalance_pts)}] {today.date()}  momentum scored "
                  f"{len(universe)} tickers", flush=True)

    signal_df = pd.DataFrame(rows)
    return universe, prices, signal_df


def simulate_from_signals(universe: list, prices: pd.DataFrame, signal_df: pd.DataFrame,
                          min_r2: float = None, buy_threshold: float = None,
                          start_capital: float = None) -> dict:
    """Cheap step: replay stop-loss/trailing-stop/rebalance/max-drawdown logic
    against precomputed signals for one MIN_R2 threshold. No model fitting —
    just filtering signal_df and bookkeeping, so this runs in well under a
    second even for hundreds of tickers, letting many thresholds be compared
    from a single compute_signal_history() pass.
    """
    min_r2 = MIN_R2 if min_r2 is None else min_r2
    buy_threshold = BUY_THRESHOLD if buy_threshold is None else buy_threshold
    capital = start_capital if start_capital is not None else BACKTEST_START_CAPITAL
    dates = prices.index

    sig = signal_df.set_index(['date', 'ticker'])[['pred_return', 'r2']]
    rebalance_dates = set(signal_df['date'].unique())

    cash = capital
    positions = {}
    equity_curve = []
    trades = []
    portfolio_peak = capital
    halted = False

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

        # 2. Rebalance — apply the threshold to the precomputed signals for `today`
        if today in rebalance_dates:
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

            if not halted:
                for ticker in universe:
                    key = (today, ticker)
                    if key not in sig.index:
                        continue
                    pred_ret, r2 = sig.loc[key]
                    if pred_ret is None or pd.isna(pred_ret) or r2 < min_r2 or pred_ret <= buy_threshold:
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
    metrics['min_r2'] = min_r2

    return {'equity_curve': equity_df, 'trades': trades_df, 'metrics': metrics}


def simulate_ranked_from_signals(universe: list, prices: pd.DataFrame, signal_df: pd.DataFrame,
                                 top_n: int = 10, min_r2: float = None,
                                 require_positive: bool = True,
                                 start_capital: float = None) -> dict:
    """Rank-based, equal-weight strategy — the methodology used in the academic
    literature (e.g. Gu, Kelly & Xiu decile-sort predicted returns and trade the
    top decile) rather than an absolute BUY_THRESHOLD cutoff.

    Two deliberate departures from simulate_from_signals(), both motivated by
    findings from the MIN_R2 sweep on this project's own data:
      - Selection is by RANK (top `top_n` predicted returns each rebalance),
        not by whether the raw predicted magnitude clears a fixed bar — the
        sweep showed predicted-return magnitude is ~uncorrelated with R² or
        outcome quality, so a magnitude cutoff filters for noise, not against it.
      - Sizing is FLAT equal-weight across the selected names, not scaled by
        `min(1.0, r2)` — with r2 typically in [-1, 0.1], that formula produces
        qty=0 (no trade at all) for nearly every real r2 value at realistic
        share prices, and goes negative (silently skipped, but via a position-
        sizing accident, not an intentional risk decision) whenever r2 < 0. R2
        isn't a calibrated edge estimate, so using it to size bets is sizing by
        noise dressed up as confidence.

    `min_r2` is still available as an optional floor (None = no R² filter at
    all, since R² has already been shown to barely track anything); it exists
    to test whether even a token quality floor changes results.
    """
    capital = start_capital if start_capital is not None else BACKTEST_START_CAPITAL
    dates = prices.index

    sig = signal_df.set_index(['date', 'ticker'])[['pred_return', 'r2']]
    rebalance_dates = set(signal_df['date'].unique())

    cash = capital
    positions = {}
    equity_curve = []
    trades = []
    portfolio_peak = capital
    halted = False

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

        # 2. Rebalance — rank candidates, equal-weight the top N
        if today in rebalance_dates:
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

            if not halted:
                today_sig = signal_df[signal_df['date'] == today].dropna(subset=['pred_return'])
                if min_r2 is not None:
                    today_sig = today_sig[today_sig['r2'] >= min_r2]
                if require_positive:
                    today_sig = today_sig[today_sig['pred_return'] > 0]
                picks = today_sig.sort_values('pred_return', ascending=False).head(top_n)

                n_picks = len(picks)
                if n_picks:
                    alloc_per_position = cash / n_picks
                    for _, row in picks.iterrows():
                        ticker = row['ticker']
                        price = today_prices.get(ticker)
                        if price is None or np.isnan(price):
                            continue
                        fill = _fill_price(price, 'BUY')
                        qty = int(alloc_per_position / (fill * (1 + BACKTEST_TRANSACTION_COST_PCT)))
                        if qty < 1:
                            continue
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

    bh_start_prices = prices.iloc[BACKTEST_MIN_HISTORY_DAYS]
    bh_shares = (capital / len(universe)) / bh_start_prices
    equity_df['benchmark_equity'] = (
        prices.iloc[BACKTEST_MIN_HISTORY_DAYS:] * bh_shares
    ).sum(axis=1).values

    trades_df = pd.DataFrame(trades)
    metrics = _compute_metrics(equity_df, trades_df, capital)
    metrics['top_n'] = top_n

    return {'equity_curve': equity_df, 'trades': trades_df, 'metrics': metrics}


def run_backtest(prices_df: pd.DataFrame, n_tickers: int = 20, start_capital: float = None,
                 verbose: bool = True, min_r2: float = None, volume_df: pd.DataFrame = None,
                 vix: pd.Series = None) -> dict:
    """Single-threshold backtest — thin wrapper kept for backward compatibility
    (dashboard's Backtest tab and `python main.py backtest` both use this).
    For comparing several MIN_R2 values without re-fitting, use
    run_min_r2_sweep() instead. `volume_df`/`vix` are optional and enable the
    engineered momentum/liquidity/volatility features — see analysis/model.py.
    """
    universe, prices, signal_df = compute_signal_history(
        prices_df, n_tickers, verbose, volume_df=volume_df, vix=vix)
    return simulate_from_signals(universe, prices, signal_df, min_r2=min_r2, start_capital=start_capital)


def run_min_r2_sweep(prices_df: pd.DataFrame, n_tickers: int = 20,
                     min_r2_values: list = None, start_capital: float = None,
                     verbose: bool = True, volume_df: pd.DataFrame = None,
                     vix: pd.Series = None) -> dict:
    """Score every ticker/rebalance ONCE, then cheaply evaluate several MIN_R2
    thresholds against those same signals — answers "where is the confidence
    gate actually selective?" without paying the fitting cost N times.
    `volume_df`/`vix` are optional and enable the engineered momentum/
    liquidity/volatility features — see analysis/model.py.

    Returns {'summary': DataFrame (one row per min_r2, all metrics),
             'results': {min_r2: run_backtest()-shaped dict},
             'signal_df': the raw (date, ticker, pred_return, r2) history —
                          save this to re-sweep additional thresholds later
                          without recomputing anything.}
    """
    if min_r2_values is None:
        # Walk-forward R² on daily-return regressions is very often negative (the
        # model explaining less variance than a flat mean forecast) — a realistic
        # sweep has to reach well below 0 to find where the gate actually opens up.
        min_r2_values = [-2.0, -1.0, -0.5, -0.2, -0.1, -0.05, -0.02, 0.0, 0.01, 0.05, 0.1, 0.2]

    universe, prices, signal_df = compute_signal_history(
        prices_df, n_tickers, verbose, volume_df=volume_df, vix=vix)

    valid = signal_df.dropna(subset=['pred_return'])
    buy_candidates = valid[valid['pred_return'] > BUY_THRESHOLD]
    if verbose:
        print(f"\n  {len(valid)} scored (ticker, rebalance) pairs; "
              f"{len(buy_candidates)} cleared BUY_THRESHOLD ({BUY_THRESHOLD:+.1%}) "
              f"before any R² filter.")
        if len(buy_candidates):
            q = buy_candidates['r2'].quantile([0, .1, .25, .5, .75, .9, 1.0])
            print("  R² distribution among those candidates (this is the range MIN_R2 "
                  "actually has to work with):")
            for p, v in q.items():
                print(f"    p{int(p*100):>3}: {v:+.3f}")

    rows = []
    results = {}
    for r2_val in min_r2_values:
        res = simulate_from_signals(universe, prices, signal_df, min_r2=r2_val,
                                    start_capital=start_capital)
        results[r2_val] = res
        rows.append({'min_r2': r2_val, **res['metrics']})
        if verbose:
            m = res['metrics']
            print(f"  MIN_R2={r2_val:+.2f}  trades={m['num_trades']:>4}  "
                  f"return={m['total_return_pct']:+8.2f}%  sharpe={m['sharpe_ratio']:+6.2f}  "
                  f"maxDD={m['max_drawdown_pct']:7.2f}%  win_rate={m['win_rate_pct']:5.1f}%  "
                  f"vs_bh={m['benchmark_return_pct']:+8.2f}%", flush=True)

    summary_df = pd.DataFrame(rows)
    return {'summary': summary_df, 'results': results, 'signal_df': signal_df}


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
