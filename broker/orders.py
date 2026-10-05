from ib_insync import IB, LimitOrder, MarketOrder, Trade

from config import FALLBACK_PORTFOLIO, MAX_POSITION_PCT
from broker.connection import get_contract


def get_portfolio_value(ib: IB) -> float:
    """Return net liquidation value from IB account summary."""
    for av in ib.accountSummary():
        if av.tag == 'NetLiquidation' and av.currency == 'USD':
            return float(av.value)
    return FALLBACK_PORTFOLIO


def calculate_position_size(portfolio_value: float, price: float, n_selected: int) -> int:
    """Equal-weight share count: portfolio_value split evenly across the
    n_selected ranked positions this run, capped by MAX_POSITION_PCT as a
    per-position safety ceiling.

    Replaces the old R2-scaled sizing (`strength = min(1.0, r2)`), which was
    found to produce qty=0 at realistic share prices for nearly any real R2
    value (R2 is almost never near 1.0) and to go negative whenever R2 < 0 —
    not an intentional risk decision, a sizing-formula accident. See
    analysis/backtest.py's simulate_ranked_from_signals and the README.

    Returns 0 (not a forced minimum of 1) when the allocated value can't
    afford even one share — buying 1 share of an expensive stock on a tiny
    allocated budget would silently blow through the intended position size.
    """
    if n_selected < 1:
        return 0
    equal_weight_value = portfolio_value / n_selected
    max_value = min(equal_weight_value, portfolio_value * MAX_POSITION_PCT)
    return int(max_value / price)


def execute_order(ib: IB, ticker: str, action: str, quantity: int,
                  order_type: str = 'MKT') -> Trade:
    """Place a market or limit order on IB."""
    contract = get_contract(ticker)
    ib.qualifyContracts(contract)

    if order_type == 'MKT':
        order = MarketOrder(action, quantity)
    else:
        ticker_data = ib.reqMktData(contract, '', False, False)
        ib.sleep(1)
        limit_price = ticker_data.ask if action == 'BUY' else ticker_data.bid
        order = LimitOrder(action, quantity, round(limit_price, 2))

    trade = ib.placeOrder(contract, order)
    ib.sleep(0.5)
    print(f"  → Order {action} {quantity}x {ticker}: {trade.orderStatus.status}")
    return trade


def close_position(ib: IB, ticker: str):
    """Close an open position for the given ticker if one exists."""
    positions = {p.contract.symbol: p for p in ib.positions()}
    if ticker in positions:
        pos = positions[ticker]
        action = 'SELL' if pos.position > 0 else 'BUY'
        execute_order(ib, ticker, action, abs(int(pos.position)))
