import math
import time

import pandas as pd
import yfinance as yf
from ib_insync import IB, util

from config import CACHE_DIR, HISTORY_DAYS, PRICE_CACHE_OVERLAP_DAYS
from broker.connection import get_contract


def _build_duration(trading_days: int) -> str:
    """Convert a number of trading days to an IB durationStr.

    IB requires years (Y) for requests longer than 365 days.
    1 year ≈ 252 trading days.
    """
    total = trading_days + 10   # small buffer for holidays
    if total > 365:
        return f'{math.ceil(total / 252)} Y'
    return f'{total} D'


def fetch_prices(ib: IB, tickers: list, duration: str = None,
                 bar_size: str = '1 day') -> pd.DataFrame:
    """Download historical close prices via Interactive Brokers.

    Requires an active IB TWS / Gateway connection.
    Returns a DataFrame with dates as index and tickers as columns.
    """
    if duration is None:
        duration = _build_duration(HISTORY_DAYS)

    prices = {}
    print(f"\nDownloading historical data ({duration}, bar size {bar_size})...")

    for ticker in tickers:
        try:
            contract = get_contract(ticker)
            bars = ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow='ADJUSTED_LAST',
                useRTH=True,
                keepUpToDate=False,
            )
            if not bars:
                print(f"  ✗ {ticker}: no data")
                continue
            df = util.df(bars)[['date', 'close']].set_index('date')
            prices[ticker] = df['close']
            print(f"  ✓ {ticker}: {len(df)} bars — last close ${df['close'].iloc[-1]:.2f}")
            time.sleep(0.2)   # avoid API throttling
        except Exception as e:
            print(f"  ✗ {ticker}: {e}")

    prices_df = pd.DataFrame(prices).tail(HISTORY_DAYS)
    prices_df.dropna(axis=1, how='all', inplace=True)
    print(f"\n  Tickers with data: {len(prices_df.columns)}/{len(tickers)}")
    return prices_df


def fetch_prices_free(tickers: list) -> pd.DataFrame:
    """Download historical close prices via Yahoo Finance (yfinance) — no IB needed.

    All tickers are fetched in a single batch request. Dot notation is converted
    to Yahoo Finance format (e.g. BRK.B → BRK-B) automatically.
    Data is 15-min delayed for the latest bar; all historical bars are accurate.

    Returns a DataFrame with dates as index and original ticker names as columns.
    """
    yf_tickers  = [t.replace('.', '-') for t in tickers]
    ticker_map  = dict(zip(yf_tickers, tickers))   # yf name → original name

    # HISTORY_DAYS are trading days; yfinance period uses calendar days
    calendar_days = math.ceil(HISTORY_DAYS * 365 / 252) + 10
    period = f'{calendar_days}d'

    print(f"\nDownloading historical data via yfinance ({HISTORY_DAYS} trading days)...")
    raw = yf.download(
        yf_tickers,
        period=period,
        interval='1d',
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # yf.download returns MultiIndex columns when >1 ticker; single ticker is flat
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw['Close'].rename(columns=ticker_map)
    else:
        prices = raw[['Close']].rename(columns={'Close': tickers[0]})

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.tail(HISTORY_DAYS)
    prices.dropna(axis=1, how='all', inplace=True)

    fetched = len(prices.columns)
    print(f"  Tickers with data: {fetched}/{len(tickers)}")
    for ticker in prices.columns:
        print(f"  ✓ {ticker}: {prices[ticker].count()} bars"
              f" — last close ${prices[ticker].dropna().iloc[-1]:.2f}")
    return prices


def fetch_volume_free(tickers: list) -> pd.DataFrame:
    """Download historical daily volume via Yahoo Finance — no IB needed.

    Same batch request as fetch_prices_free; extracts the Volume column.
    Returns a DataFrame with dates as index and original ticker names as columns.
    """
    yf_tickers = [t.replace('.', '-') for t in tickers]
    ticker_map = dict(zip(yf_tickers, tickers))

    calendar_days = math.ceil(HISTORY_DAYS * 365 / 252) + 10

    print(f"\nDownloading historical volume via yfinance ({HISTORY_DAYS} trading days)...")
    raw = yf.download(
        yf_tickers,
        period=f'{calendar_days}d',
        interval='1d',
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        volume = raw['Volume'].rename(columns=ticker_map)
    else:
        volume = raw[['Volume']].rename(columns={'Volume': tickers[0]})

    volume.index = pd.to_datetime(volume.index).tz_localize(None)
    volume = volume.tail(HISTORY_DAYS)
    volume.dropna(axis=1, how='all', inplace=True)

    print(f"  Volume data: {len(volume.columns)}/{len(tickers)} tickers")
    return volume


# ── Cached fetch helpers ──────────────────────────────────────────────────────

def _fetch_col_from_date(tickers: list, start: str, col: str) -> pd.DataFrame:
    """Download a single OHLCV column for tickers starting from a given date."""
    yf_tickers = [t.replace('.', '-') for t in tickers]
    ticker_map  = dict(zip(yf_tickers, tickers))
    raw = yf.download(yf_tickers, start=start, interval='1d',
                      auto_adjust=True, progress=False, threads=True)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw[col].rename(columns=ticker_map)
    else:
        df = raw[[col]].rename(columns={col: tickers[0]})
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def _update_ohlcv_cache(cache_path, tickers: list, col: str,
                        full_fetch_fn) -> pd.DataFrame:
    """Load parquet cache, fetch only new days, append, save, return tail.

    - Re-fetches a PRICE_CACHE_OVERLAP_DAYS window to capture price adjustments.
    - Fetches full history for any tickers not yet in the cache.
    - Cache grows over time; all tickers ever fetched are retained.
    """
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        cached.index = pd.to_datetime(cached.index).tz_localize(None)
        cached = cached.sort_index()

        last_date   = cached.index[-1]
        days_behind = (pd.Timestamp.today() - last_date).days

        if days_behind < 1:
            print(f"  Cache up to date (last: {last_date.date()})")
        else:
            existing = [t for t in tickers if t in cached.columns]
            new_t    = [t for t in tickers if t not in cached.columns]

            # Overlap start: go back enough calendar days to cover PRICE_CACHE_OVERLAP_DAYS trading days
            overlap_start = (last_date - pd.Timedelta(days=PRICE_CACHE_OVERLAP_DAYS * 2)).strftime('%Y-%m-%d')

            if existing:
                print(f"  Fetching incremental {col} for {len(existing)} tickers (from {overlap_start})...")
                fresh = _fetch_col_from_date(existing, overlap_start, col)
                if not fresh.empty:
                    cut = fresh.index[0]
                    cached = pd.concat([cached[cached.index < cut], fresh])
                    cached = cached.sort_index()
                    cached = cached[~cached.index.duplicated(keep='last')]

            if new_t:
                print(f"  Fetching full history for {len(new_t)} new tickers...")
                full_new = full_fetch_fn(new_t)
                for t in full_new.columns:
                    cached[t] = full_new[t].reindex(cached.index)
                cached = cached.sort_index()
    else:
        print(f"  No cache found — fetching full history for {len(tickers)} tickers...")
        CACHE_DIR.mkdir(exist_ok=True)
        cached = full_fetch_fn(tickers)

    cached.to_parquet(cache_path)
    print(f"  Cache saved: {cache_path.name}  "
          f"({len(cached.columns)} tickers, {len(cached)} days, "
          f"last: {cached.index[-1].date()})")

    result = cached[[t for t in tickers if t in cached.columns]].tail(HISTORY_DAYS)
    result.dropna(axis=1, how='all', inplace=True)
    return result


def _fetch_ohlcv_max(tickers: list, col: str) -> pd.DataFrame:
    """Download full available history (period='max') for the given OHLCV column."""
    yf_tickers = [t.replace('.', '-') for t in tickers]
    ticker_map  = dict(zip(yf_tickers, tickers))
    print(f"\nDownloading full history (period=max) via yfinance — {len(tickers)} tickers...")
    raw = yf.download(yf_tickers, period='max', interval='1d',
                      auto_adjust=True, progress=False, threads=True)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw[col].rename(columns=ticker_map)
    else:
        df = raw[[col]].rename(columns={col: tickers[0]})
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.dropna(axis=1, how='all', inplace=True)
    print(f"  ✓ {len(df.columns)}/{len(tickers)} tickers — {len(df)} trading days "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


def fetch_prices_max(tickers: list) -> pd.DataFrame:
    """Download full price history since IPO for all tickers."""
    return _fetch_ohlcv_max(tickers, 'Close')


def fetch_volume_max(tickers: list) -> pd.DataFrame:
    """Download full volume history since IPO for all tickers."""
    return _fetch_ohlcv_max(tickers, 'Volume')


def fetch_prices_cached(tickers: list) -> pd.DataFrame:
    """Cached version — initial download uses period='max'; subsequent runs are incremental."""
    print("\nLoading prices from cache + incremental update...")
    return _update_ohlcv_cache(
        CACHE_DIR / 'prices_cache.parquet', tickers, 'Close', fetch_prices_max
    )


def fetch_volume_cached(tickers: list) -> pd.DataFrame:
    """Cached version — initial download uses period='max'; subsequent runs are incremental."""
    print("\nLoading volume from cache + incremental update...")
    return _update_ohlcv_cache(
        CACHE_DIR / 'volume_cache.parquet', tickers, 'Volume', fetch_volume_max
    )


def fetch_vix_cached() -> pd.Series:
    """Cached CBOE VIX close price — reuses the same incremental-cache machinery as
    fetch_prices_cached(), kept in its own file (not mixed into prices_cache.parquet)
    since VIX is a volatility index, not a tradable S&P 500 constituent, and
    shouldn't appear as a candidate in the universe's correlation matrix.

    Feeds analysis/model.py's engineered features as a market-wide volatility-
    regime signal.
    """
    print("\nLoading VIX from cache + incremental update...")
    df = _update_ohlcv_cache(CACHE_DIR / 'vix_cache.parquet', ['^VIX'], 'Close', fetch_prices_max)
    return df['^VIX'] if '^VIX' in df.columns else pd.Series(dtype=float)
