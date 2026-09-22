"""Stop-loss, trailing-stop, and portfolio max-drawdown guard.

State (entry price, peak price per open position, and portfolio equity peak)
is persisted to RISK_STATE_FILE so it survives across daily cron runs — each
run is a fresh process with no in-memory history of prior fills.
"""
import json

from config import MAX_DRAWDOWN_PCT, RISK_STATE_FILE, STOP_LOSS_PCT, TRAILING_STOP_PCT


def load_risk_state() -> dict:
    if RISK_STATE_FILE.exists():
        try:
            return json.loads(RISK_STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {'positions': {}, 'portfolio_peak': 0.0}


def save_risk_state(state: dict) -> None:
    RISK_STATE_FILE.write_text(json.dumps(state, indent=2))


def register_entry(state: dict, ticker: str, price: float, qty: int) -> None:
    """Record a new BUY fill so future runs can evaluate stop-loss/trailing-stop."""
    state['positions'][ticker] = {'entry_price': price, 'peak_price': price, 'qty': qty}


def remove_position(state: dict, ticker: str) -> None:
    state['positions'].pop(ticker, None)


def check_stop_losses(state: dict, current_prices: dict) -> list:
    """Update peak prices and return tickers whose stop-loss or trailing-stop triggered.

    current_prices: {ticker: last_close_price} for every ticker with a tracked position.
    Does not mutate positions itself — the caller closes the position on IB and then
    calls remove_position() once the order is confirmed.
    """
    triggered = []
    for ticker, pos in state['positions'].items():
        price = current_prices.get(ticker)
        if price is None:
            continue

        pos['peak_price'] = max(pos['peak_price'], price)

        hard_stop_hit = price <= pos['entry_price'] * (1 - STOP_LOSS_PCT)
        trailing_hit  = price <= pos['peak_price']  * (1 - TRAILING_STOP_PCT)

        if hard_stop_hit or trailing_hit:
            reason = 'STOP_LOSS' if hard_stop_hit else 'TRAILING_STOP'
            triggered.append((ticker, reason))

    return triggered


def check_max_drawdown(state: dict, portfolio_value: float) -> bool:
    """Update the equity peak and return True if new BUY orders should be halted.

    SELL orders and stop-loss closes are never blocked — only new risk-taking is.
    """
    state['portfolio_peak'] = max(state.get('portfolio_peak', 0.0), portfolio_value)
    peak = state['portfolio_peak']
    if peak <= 0:
        return False
    drawdown = (peak - portfolio_value) / peak
    return drawdown >= MAX_DRAWDOWN_PCT
