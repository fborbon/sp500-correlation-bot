import warnings
warnings.filterwarnings('ignore')

from config import MAX_DRAWDOWN_PCT, OUTPUTS_DIR, TOP_N_HIGHLIGHT, create_run_dirs
from broker.connection import connect_ib
from broker.data import (fetch_prices, fetch_prices_free, fetch_prices_cached,
                         fetch_vix_cached, fetch_volume_free, fetch_volume_cached)
from broker.orders import calculate_position_size, close_position, execute_order, get_portfolio_value
from broker.risk import (check_max_drawdown, check_stop_losses, load_risk_state,
                         register_entry, remove_position, save_risk_state)
from analysis.universe import fetch_company_metadata, fetch_market_caps_cached, get_sp500_tickers
from analysis.fundamentals import fetch_fundamentals, score_fundamentals, save_fundamentals_csv
from analysis.correlations import compute_correlations, get_top_correlated_pairs, get_top_inverse_pairs
from analysis.model import predict_price
from analysis.signals import generate_signals
from reporting.charts import (plot_correlation_matrix, plot_cumulative_returns,
                               plot_market_cap_bars, plot_market_cap_series,
                               plot_prediction_analysis, plot_price_series,
                               plot_volume_series)
from reporting.precompute import precompute_dashboard_cache
from reporting.report import print_report, save_signals_csv


def run_bot(execute_trades: bool = False, save_plots: bool = True,
            n_tickers: int = None):
    """Full pipeline: connect → download → correlate → predict → signal → (trade).

    Args:
        execute_trades: If True, places real orders in IB. Use with caution.
        save_plots:     If True, saves heatmap + price series + per-signal analysis PNGs.
        n_tickers:      Number of top S&P 500 companies by market cap to use.
                        None = full universe (~503 tickers).
    """
    print("\nCreate local folders")
    run_dir, gen_dir, corr_dir = create_run_dirs()

    print("\nFetching S&P 500 tickers and market_caps")
    tickers, market_caps = get_sp500_tickers(n=n_tickers)

    print("\nFetch stock prices")
    prices_df = fetch_prices_cached(tickers)
    if prices_df.empty or len(prices_df.columns) < 5:
        print("✗ Insufficient data. Aborting.")
        return

    print("\nCalculating correlations")
    corr_matrix, returns = compute_correlations(prices_df)
    top_pairs     = get_top_correlated_pairs(corr_matrix, top_n=10)
    inverse_pairs = get_top_inverse_pairs(corr_matrix, top_n=10)

    print("\nFetch volume and VIX for engineered features (momentum/liquidity/volatility)")
    volume_df = fetch_volume_cached(list(prices_df.columns))
    vix = fetch_vix_cached()

    print("\nGenerate the signals table")
    signals_df = generate_signals(prices_df, returns, corr_matrix, volume_df=volume_df, vix=vix)

    print("\nEnrich signals with company name, sector, founded year, market cap (B)")
    company_meta = fetch_company_metadata(list(prices_df.columns), market_caps)
    signals_df = signals_df.merge(
        company_meta.reset_index().rename(columns={'index': 'ticker'}),
        on='ticker', how='left'
    )
    
    print("\nReorder columns so metadata appears right after ticker")
    meta_cols = ['company_name', 'sector', 'founded', 'market_cap_B']
    other_cols = [c for c in signals_df.columns if c not in ['ticker'] + meta_cols]
    signals_df = signals_df[['ticker'] + meta_cols + other_cols]

    print("\nSave signals table to file")
    print_report(signals_df, top_pairs, inverse_pairs)
    save_signals_csv(signals_df, run_dir / 'signals.csv')

    print("\nSave prices for the dashboard interactive charts")
    prices_df.to_csv(run_dir / 'prices.csv')

    print("\nSave volume data (already fetched above for engineered features)")
    volume_df.to_csv(run_dir / 'volume.csv')

    print("\nFetch and save the Fundamental analysis table")
    # Fundamental analysis table
    fund_raw = fetch_fundamentals(list(prices_df.columns))
    fund_df  = score_fundamentals(fund_raw)
    save_fundamentals_csv(fund_df, run_dir / 'fundamentals.csv')

    print("\nSave plots")
    if save_plots:
        print("\nSlice to top N by market cap for a legible heatmap")
        top_t = [t for t in tickers if t in corr_matrix.columns][:TOP_N_HIGHLIGHT]
        plot_correlation_matrix(corr_matrix.loc[top_t, top_t],
                                save_path=corr_dir / 'correlation_matrix.png')

        print("\nGeneral/ — price series highlighted by market cap")
        plot_price_series(prices_df, tickers, top_n=TOP_N_HIGHLIGHT, label='market cap',
                          save_path=gen_dir / 'price_series_market-cap.png')

        print("\nGeneral/ — price series highlighted by highest absolute stock price")
        tickers_by_price = sorted(
            prices_df.columns.tolist(),
            key=lambda t: prices_df[t].iloc[-1],
            reverse=True
        )
        plot_price_series(prices_df, tickers_by_price, top_n=TOP_N_HIGHLIGHT, label='stock price',
                          save_path=gen_dir / 'price_series_stock-price-absolute.png')

        print("\nGeneral/ — price series highlighted by highest normalized return (best performers)")
        tickers_by_norm = sorted(
            prices_df.columns.tolist(),
            key=lambda t: prices_df[t].iloc[-1] / prices_df[t].iloc[0],
            reverse=True
        )
        plot_price_series(prices_df, tickers_by_norm, top_n=TOP_N_HIGHLIGHT, label='normalized return',
                          save_path=gen_dir / 'price_series_normalized-return.png')

        print("\nGeneral/ — bar chart: top 15 vs bottom 15 by market cap")
        plot_market_cap_bars(prices_df, tickers, market_caps=market_caps, top_n=TOP_N_HIGHLIGHT,
                             save_path=gen_dir / 'market_cap_bars.png')

        print("\nGeneral/ — market cap time series (absolute + normalized growth)")
        plot_market_cap_series(prices_df, market_caps, top_n=TOP_N_HIGHLIGHT,
                               save_path_abs=gen_dir  / 'market_cap_series_absolute.png',
                               save_path_norm=gen_dir / 'market_cap_series_normalized.png')

        print("\nGeneral/ — volume time series (absolute + normalized growth)")
        plot_volume_series(volume_df, top_n=TOP_N_HIGHLIGHT,
                           save_path_abs=gen_dir  / 'volume_series_absolute.png',
                           save_path_norm=gen_dir / 'volume_series_normalized.png')

        print("\nGeneral/ — cumulative returns (top N best performers highlighted)")
        plot_cumulative_returns(prices_df, top_n=TOP_N_HIGHLIGHT,
                                save_path=gen_dir / 'cumulative_returns.png')

        print("\nGenerate tables for: top N by predicted return  +  all BUY/SELL tickers")
        # Correlation_method/ — per-ticker prediction analysis
        # Generate tables for: top N by predicted return  +  all BUY/SELL tickers
        top_return_tickers = set(signals_df.head(TOP_N_HIGHLIGHT)['ticker'])
        buysell_tickers    = set(signals_df[signals_df['signal'].isin(['BUY', 'SELL'])]['ticker'])
        analysis_tickers   = top_return_tickers | buysell_tickers

        print("\nGenerating predition analysis plots")
        if analysis_tickers:
            print(f"\nGenerating analysis charts ({len(analysis_tickers)} tickers)...")
        for ticker in sorted(analysis_tickers):
            pred_ret, r2, top5, corr_signs, y_actual, y_pred = predict_price(
                ticker, returns, corr_matrix
            )
            if y_actual is not None:
                plot_prediction_analysis(
                    ticker, returns, prices_df, top5, corr_signs,
                    y_actual, y_pred,
                    save_path=corr_dir / f'analysis_{ticker}.png'
                )

    print("\nPre-computing dashboard cache")
    precompute_dashboard_cache(run_dir)

    print("\nExecute trades in IBKR")
    if execute_trades:
        ib = connect_ib()
        risk_state = load_risk_state()
        try:
            current_prices = prices_df.iloc[-1].to_dict()

            print("\nChecking stop-loss / trailing-stop on open positions...")
            for ticker, reason in check_stop_losses(risk_state, current_prices):
                print(f"  ✗ {reason} hit on {ticker} @ ${current_prices[ticker]:.2f} — closing position")
                close_position(ib, ticker)
                remove_position(risk_state, ticker)

            portfolio_value = get_portfolio_value(ib)
            halted = check_max_drawdown(risk_state, portfolio_value)
            if halted:
                print(f"\n  ⚠ Max drawdown ({MAX_DRAWDOWN_PCT:.0%}) breached — rebalance skipped "
                      f"this run (no new buys, no selection-driven sells). Stop-loss/"
                      f"trailing-stop closes above still applied.")
            else:
                # Rank-based rebalance: hold exactly this run's top TOP_N_POSITIONS tickers
                # by predicted return, equal-weighted. Validated (Oct 2026) to outperform the
                # old per-ticker BUY/SELL-signal + R2-scaled-sizing approach — see
                # analysis/backtest.py's simulate_ranked_from_signals and the README.
                selected = set(signals_df[signals_df['selected']]['ticker'])
                held_positions = {p.contract.symbol: p for p in ib.positions()}
                held_tickers = set(held_positions.keys())

                print(f"\nRebalancing to {len(selected)} ranked positions "
                      f"(portfolio: ${portfolio_value:,.0f})...")

                for ticker in held_tickers - selected:
                    pos = held_positions[ticker]
                    execute_order(ib, ticker, 'SELL', abs(int(pos.position)))
                    remove_position(risk_state, ticker)

                price_lookup = signals_df.set_index('ticker')['current_price']
                for ticker in selected - held_tickers:
                    price = price_lookup.get(ticker)
                    if price is None or price <= 0:
                        continue
                    qty = calculate_position_size(portfolio_value, price, len(selected))
                    if qty < 1:
                        continue
                    execute_order(ib, ticker, 'BUY', qty)
                    register_entry(risk_state, ticker, price, qty)
        finally:
            save_risk_state(risk_state)
            ib.disconnect()
            print("\n✓ Disconnected from Interactive Brokers.")
    else:
        print("\n  ℹ Simulation mode — no orders placed.")
        print("    To execute on paper trading: run_bot(execute_trades=True)")


def _load_backtest_prices(n_tickers) -> "pd.DataFrame":
    from config import BACKTEST_LOOKBACK_DAYS, BACKTEST_MIN_HISTORY_DAYS

    tickers, _ = get_sp500_tickers(n=n_tickers)
    prices_df = fetch_prices_cached(tickers)
    if prices_df.empty:
        return prices_df
    # fetch_prices_cached() returns the FULL cache (back to each ticker's IPO) — bound the
    # walk to BACKTEST_LOOKBACK_DAYS (+warm-up) so runtime stays practical.
    return prices_df.tail(BACKTEST_LOOKBACK_DAYS + BACKTEST_MIN_HISTORY_DAYS)


def _load_backtest_features(prices_df) -> tuple:
    """Volume + VIX, sliced to the same tickers/window as prices_df, for the
    engineered momentum/liquidity/volatility features (see analysis/model.py)."""
    volume_df = fetch_volume_cached(list(prices_df.columns))
    volume_df = volume_df.reindex(prices_df.index)
    vix = fetch_vix_cached().reindex(prices_df.index)
    return volume_df, vix


def run_backtest_cli(n_tickers: int = 20) -> None:
    """Walk-forward backtest of the LIVE strategy — rank-based top-TOP_N_POSITIONS
    selection, equal-weight sizing, momentum/illiquidity/realized_vol/vix_level
    features — net of commissions/slippage. Matches main.py's execute_trades
    exactly (analysis/backtest.py's simulate_ranked_from_signals()), so this is
    what the dashboard's Backtest tab should reflect. The older threshold-based
    simulate_from_signals()/run_backtest() still exist for comparison/research
    (see `python main.py sweep`) but no longer drive this CLI, since they test
    a strategy that isn't the one actually trading.

    Reuses whatever is already in cache/prices_cache.parquet (populated by any
    prior `signals`/`paper`/`live` run) — run `python main.py signals` first if
    the cache is empty.
    """
    import json
    from analysis.backtest import compute_signal_history, simulate_ranked_from_signals
    from config import CACHE_DIR, TOP_N_POSITIONS

    label = 'full S&P 500 universe (~502 tickers)' if n_tickers is None else f'top {n_tickers} tickers'
    print(f"\nBacktest (live strategy: rank-based top {TOP_N_POSITIONS}) — {label} by market cap")
    prices_df = _load_backtest_prices(n_tickers)
    if prices_df.empty:
        print("✗ No cached price data. Run `python main.py signals <n>` first to populate the cache.")
        return
    volume_df, vix = _load_backtest_features(prices_df)

    universe, prices, signal_df = compute_signal_history(
        prices_df, n_tickers=n_tickers, volume_df=volume_df, vix=vix)
    result = simulate_ranked_from_signals(universe, prices, signal_df, top_n=TOP_N_POSITIONS)
    metrics = result['metrics']

    print("\n=== Backtest results ===")
    for k, v in metrics.items():
        print(f"  {k:<24} {v}")

    out_dir = CACHE_DIR / 'backtest_latest'
    out_dir.mkdir(parents=True, exist_ok=True)
    result['equity_curve'].to_csv(out_dir / 'equity_curve.csv')
    result['trades'].to_csv(out_dir / 'trades.csv', index=False)
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(f"\n✓ Saved to {out_dir}")


def run_sweep_cli(n_tickers: int = 20, min_r2_values: list = None) -> None:
    """Score the universe ONCE, then compare several MIN_R2 thresholds against
    those same signals (cheap — no re-fitting). Writes cache/backtest_sweep/
    (summary.csv + the raw signal history, so more thresholds can be tried
    later without recomputing anything) and refreshes cache/backtest_latest/
    with the config.MIN_R2 result, so the dashboard's Backtest tab reflects
    this run too.
    """
    import json
    from analysis.backtest import run_min_r2_sweep
    from config import CACHE_DIR, MIN_R2

    if min_r2_values is not None and MIN_R2 not in min_r2_values:
        min_r2_values = sorted(set(min_r2_values) | {MIN_R2})

    label = 'full S&P 500 universe (~502 tickers)' if n_tickers is None else f'top {n_tickers} tickers'
    print(f"\nMIN_R2 sweep — {label} by market cap")
    prices_df = _load_backtest_prices(n_tickers)
    if prices_df.empty:
        print("✗ No cached price data. Run `python main.py signals <n>` first to populate the cache.")
        return
    volume_df, vix = _load_backtest_features(prices_df)

    sweep = run_min_r2_sweep(prices_df, n_tickers=n_tickers, min_r2_values=min_r2_values,
                             volume_df=volume_df, vix=vix)

    print("\n=== MIN_R2 sweep summary ===")
    print(sweep['summary'].to_string(index=False))

    out_dir = CACHE_DIR / 'backtest_sweep'
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep['summary'].to_csv(out_dir / 'summary.csv', index=False)
    sweep['signal_df'].to_csv(out_dir / 'signal_history.csv', index=False)
    print(f"\n✓ Sweep saved to {out_dir}")

    standard = sweep['results'][MIN_R2]
    latest_dir = CACHE_DIR / 'backtest_latest'
    latest_dir.mkdir(parents=True, exist_ok=True)
    standard['equity_curve'].to_csv(latest_dir / 'equity_curve.csv')
    standard['trades'].to_csv(latest_dir / 'trades.csv', index=False)
    (latest_dir / 'metrics.json').write_text(json.dumps(standard['metrics'], indent=2))
    print(f"✓ Dashboard Backtest tab refreshed with the MIN_R2={MIN_R2} result")


if __name__ == '__main__':
    import sys
    from demo import run_demo
    import config

    mode   = sys.argv[1] if len(sys.argv) > 1 else 'demo'
    _n_arg = sys.argv[2] if len(sys.argv) > 2 else None
    n      = (_n_arg if _n_arg == 'FALLBACK_TICKERS'
               else None if _n_arg == 'full'
               else int(_n_arg) if _n_arg is not None
               else config.N_TICKERS)

    if mode == 'demo':
        run_demo()

    elif mode == 'paper':
        run_bot(execute_trades=False, save_plots=True, n_tickers=n)

    elif mode == 'live':
        confirm = input("Confirm execution on LIVE account? (type YES): ")
        if confirm.strip() == 'YES':
            config.IB_PORT = 7496   # override before connect_ib() reads it
            run_bot(execute_trades=True, save_plots=True, n_tickers=n)
        else:
            print("Cancelled.")

    elif mode == 'signals':
        run_bot(execute_trades=False, n_tickers=n)

    elif mode == 'backtest':
        bt_n = None if _n_arg == 'full' else (n if isinstance(n, int) else 20)
        run_backtest_cli(n_tickers=bt_n)

    elif mode == 'sweep':
        sw_n = None if _n_arg == 'full' else (n if isinstance(n, int) else 20)
        run_sweep_cli(n_tickers=sw_n)

    else:
        print("Usage: python main.py [demo|paper|live|signals|backtest|sweep] [n_tickers|full]")
