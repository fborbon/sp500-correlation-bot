import numpy as np
import pandas as pd

from config import BUY_THRESHOLD, MIN_R2, SELL_THRESHOLD, TOP_N_POSITIONS
from analysis.model import predict_price


def generate_signals(prices_df: pd.DataFrame, returns: pd.DataFrame,
                     corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """Generate a BUY/SELL/HOLD signal for each ticker based on predicted return.

    Predictors are split into direct (positive r) and inverse (negative r) groups.
    Returns a DataFrame sorted by predicted_return descending, with:
      - 'signal': the per-ticker BUY/SELL/HOLD/LOW_CONFIDENCE/INSUF_DATA label —
        informational, based on the legacy absolute BUY_THRESHOLD/MIN_R2 cutoffs.
      - 'selected': the column that actually drives live trading (see main.py's
        execute_trades) — True for the top TOP_N_POSITIONS tickers by predicted
        return among those with predicted_return > 0, regardless of 'signal' or
        R². This rank-based selection was validated out-of-sample to outperform
        the old absolute-threshold + R2-scaled-sizing approach; see
        analysis/backtest.py's simulate_ranked_from_signals and the README.
    """
    current_prices = prices_df.iloc[-1]
    signals = []
    total = len(returns.columns)

    print(f"\nGenerating prediction signals ({total} tickers)...")
    for i, ticker in enumerate(returns.columns, 1):
        pred_ret, r2, top_preds, corr_signs, _, _ = predict_price(ticker, returns, corr_matrix)

        if pred_ret is None:
            signal = 'INSUF_DATA'
        elif r2 < MIN_R2:
            signal = 'LOW_CONFIDENCE'
        elif pred_ret > BUY_THRESHOLD:
            signal = 'BUY'
        elif pred_ret < SELL_THRESHOLD:
            signal = 'SELL'
        else:
            signal = 'HOLD'

        current = current_prices.get(ticker, np.nan)
        target  = (current * (1 + pred_ret)
                   if pred_ret is not None and not np.isnan(current) else np.nan)

        direct  = [t for t in top_preds if corr_signs.get(t, 1) > 0]
        inverse = [t for t in top_preds if corr_signs.get(t, 1) < 0]

        signals.append({
            'ticker':                    ticker,
            'current_price':             round(current, 2),
            'target_price_7d':           round(target, 2) if not np.isnan(target) else None,
            'predicted_return':          round(pred_ret * 100, 2) if pred_ret is not None else None,
            'model_r2':                  round(r2, 3),
            'direct_top5_predictors':    ', '.join(direct),
            'inverse_top5_predictors':   ', '.join(inverse),
            'signal':                    signal,
        })

        icon    = {'BUY': '▲', 'SELL': '▼', 'HOLD': '─'}.get(signal, '?')
        ret_str = f"{pred_ret * 100:+.2f}%" if pred_ret is not None else 'N/A'
        inv_str = f"  ↕ inv:{','.join(inverse)}" if inverse else ''
        print(f"  [{i:>3}/{total}] {icon} {ticker:<6} ${current:.2f} → ${target:.2f}  "
              f"ret={ret_str}  R²={r2:.2f}  [{signal}]{inv_str}")

    df = pd.DataFrame(signals)
    # Explicit numeric coercion: if every ticker hit INSUF_DATA, predicted_return is all None
    # and pandas infers dtype=object, which nlargest() below can't handle.
    df['predicted_return'] = pd.to_numeric(df['predicted_return'], errors='coerce')
    df.sort_values('predicted_return', ascending=False, inplace=True, na_position='last')

    candidates = df[df['predicted_return'] > 0]
    top_tickers = set(candidates.nlargest(TOP_N_POSITIONS, 'predicted_return')['ticker'])
    df['selected'] = df['ticker'].isin(top_tickers)

    return df
