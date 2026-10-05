import datetime

import pandas as pd

from config import OUTPUTS_DIR


def _fmt_cap(val) -> str:
    if pd.isna(val) or val is None:
        return 'N/A'
    return f'${val:,.0f}B'


def _fmt_meta(r) -> str:
    sector  = r.get('sector',  None)
    founded = r.get('founded', None)
    cap     = r.get('market_cap_B', None)
    parts   = []
    if pd.notna(sector)  and sector:  parts.append(str(sector))
    if pd.notna(founded) and founded: parts.append(f'est. {int(founded)}')
    if pd.notna(cap)     and cap:     parts.append(f'Cap: {_fmt_cap(cap)}')
    return '  ·  '.join(parts) if parts else ''


def _print_signal_row(prefix, r):
    name = r.get('company_name', '')
    name_str = f'  {name}' if pd.notna(name) and name else ''
    print(f"  {prefix} {r['ticker']:<6} ${r['current_price']:.2f} → ${r['target_price_7d']:.2f}  "
          f"({r['predicted_return']:+.1f}%)  R²={r['model_r2']:.2f}{name_str}")
    meta = _fmt_meta(r)
    if meta:
        print(f"           {meta}")
    if r['direct_top5_predictors']:
        print(f"      ↑ direct:   {r['direct_top5_predictors']}")
    if r['inverse_top5_predictors']:
        print(f"      ↓ inverse:  {r['inverse_top5_predictors']}")


def print_report(signals_df: pd.DataFrame, top_pairs: list,
                 inverse_pairs: list = None):
    """Print a human-readable summary of signals, direct pairs, and inverse pairs."""
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    print("\n" + "═" * 68)
    print(f"  SP500 CORRELATION BOT — Report {now}")
    print("═" * 68)

    buys  = signals_df[signals_df['signal'] == 'BUY']
    sells = signals_df[signals_df['signal'] == 'SELL']
    holds = signals_df[signals_df['signal'] == 'HOLD']

    print(f"\n  SIGNALS: {len(buys)} BUY · {len(holds)} HOLD · {len(sells)} SELL\n")
    if 'selected' in signals_df.columns:
        n_selected = int(signals_df['selected'].sum())
        print(f"  → {n_selected} selected for live rebalancing (rank-based, independent of the "
              f"BUY/SELL labels above — see README § Signal Selection)\n")

    print("  ── BUY ──────────────────────────────────────────────────────")
    for _, r in buys.iterrows():
        _print_signal_row('▲', r)

    if not sells.empty:
        print("\n  ── SELL ─────────────────────────────────────────────────────")
        for _, r in sells.iterrows():
            _print_signal_row('▼', r)

    print("\n  ── TOP 5 DIRECT CORRELATION PAIRS ───────────────────────────")
    for a, b, r in top_pairs[:5]:
        if r > 0:
            bar = '█' * int(abs(r) * 20)
            print(f"  {a:<6} ↑↑ {b:<6}  r={r:+.3f}  {bar}")

    if inverse_pairs:
        print("\n  ── TOP 5 INVERSE CORRELATION PAIRS ──────────────────────────")
        for a, b, r in inverse_pairs[:5]:
            bar = '█' * int(abs(r) * 20)
            print(f"  {a:<6} ↑↓ {b:<6}  r={r:+.3f}  {bar}")

    print("\n" + "═" * 68)


def save_signals_csv(signals_df: pd.DataFrame, path=None):
    """Save signals to CSV in the outputs directory."""
    if path is None:
        path = OUTPUTS_DIR / 'signals.csv'
    signals_df.to_csv(path, index=False)
    print(f"  Signals saved to: {path}")
