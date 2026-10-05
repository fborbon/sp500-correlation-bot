import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

from config import (FEATURE_VOL_WINDOW, MIN_CORRELATION, MOMENTUM_LOOKBACK_DAYS,
                    MOMENTUM_SKIP_DAYS, PREDICTION_DAYS)


def _engineered_features(target: str, index: pd.DatetimeIndex, prices: pd.DataFrame,
                         volume: pd.DataFrame = None, vix: pd.Series = None) -> pd.DataFrame:
    """Momentum, Amihud illiquidity, and realized volatility for `target` itself,
    plus the market-wide VIX level — all reindexed to `index` (the returns
    DataFrame's date index) so they line up as extra feature columns.

    A column is only ADDED when its underlying source was passed at all
    (volume/vix not None) — a source that's simply unavailable this call is
    omitted entirely rather than filled with NaN, because predict_price() drops
    any training row with a NaN in it: an all-NaN column (every row) would
    silently drop every row and return "insufficient data" even though the
    correlation-based features alone were perfectly usable. A column that IS
    included can still legitimately be NaN per-row (target missing from that
    source, or before its rolling/lookback window has enough history) — those
    rows get dropped individually, which is the intended behavior.
    """
    feat = pd.DataFrame(index=index)
    px = prices[target] if target in prices.columns else None

    if px is not None:
        ret = px.pct_change()
        # Trailing return, skipping the most recent few days (short-term reversal filter) —
        # standard momentum-factor construction, scaled down to this bot's shorter windows.
        momentum = px.shift(MOMENTUM_SKIP_DAYS) / px.shift(MOMENTUM_SKIP_DAYS + MOMENTUM_LOOKBACK_DAYS) - 1
        feat['momentum'] = momentum.reindex(index)
        feat['realized_vol'] = (ret.rolling(FEATURE_VOL_WINDOW).std() * np.sqrt(252)).reindex(index)
    else:
        feat['momentum'] = np.nan
        feat['realized_vol'] = np.nan

    if volume is not None:
        if px is not None and target in volume.columns:
            dollar_vol = px * volume[target]
            illiquidity = (ret.abs() / dollar_vol).replace([np.inf, -np.inf], np.nan)
            feat['illiquidity'] = illiquidity.rolling(FEATURE_VOL_WINDOW).mean().reindex(index)
        else:
            feat['illiquidity'] = np.nan

    if vix is not None and len(vix):
        feat['vix_level'] = vix.reindex(index).ffill()

    return feat


def predict_price(target: str, returns: pd.DataFrame, corr_matrix: pd.DataFrame,
                  lookahead: int = None, prices: pd.DataFrame = None,
                  volume: pd.DataFrame = None, vix: pd.Series = None) -> tuple:
    """Predict future return of `target` using correlated tickers as features,
    plus (when `prices` is passed) the target's own momentum, liquidity, and
    realized volatility, and (when `vix` is passed) the market-wide VIX level.

    Both direct (positive r) and inverse (negative r) correlators are used as
    predictors. RandomForestRegressor captures non-linear relationships and is
    scale-invariant so no StandardScaler is needed.

    `prices`/`volume`/`vix` are optional and default to None, which reproduces
    the original correlation-only feature set exactly — existing callers that
    don't pass them are unaffected.

    Returns:
        (predicted_return, r2_score, top_predictors, corr_signs, y_actual, y_predicted)
        corr_signs    : dict mapping predictor ticker → raw Pearson r (float).
                        Only covers the correlation-based predictors, not the
                        engineered features.
        y_actual      : np.ndarray of actual cumulative returns used for training.
        y_predicted   : np.ndarray of in-sample model predictions (same length).
        All arrays are None on early-exit paths.
    """
    if lookahead is None:
        lookahead = PREDICTION_DAYS

    predictors = [t for t in returns.columns if t != target]
    if target not in corr_matrix.columns:
        return None, 0.0, [], {}, None, None

    corrs = corr_matrix[target][predictors].abs()  # Looks for positive and inverse correlations
    top_pred = corrs[corrs >= MIN_CORRELATION].sort_values(ascending=False)

    if len(top_pred) < 2:
        return None, 0.0, [], {}, None, None

    pred_cols = top_pred.index.tolist()
    X = returns[pred_cols].values

    if prices is not None:
        extra = _engineered_features(target, returns.index, prices, volume, vix)
        X = np.hstack([X, extra.values])

    y_raw = returns[target].values

    n_samples = len(X) - lookahead
    if n_samples < 20:
        return None, 0.0, [], {}, None, None

    X_train_full = X[:n_samples]
    y_train_full = np.array([
        y_raw[i:i + lookahead].sum()   # cumulative return over N days
        for i in range(n_samples)
    ])

    # Engineered features are NaN until their rolling/lookback windows fill in (and whenever
    # a source wasn't available for this ticker) — drop those rows rather than feed NaN to
    # the model. No-op when prices wasn't passed, since there are no extra columns to be NaN.
    valid = ~np.isnan(X_train_full).any(axis=1)
    X_train = X_train_full[valid]
    y_train = y_train_full[valid]

    if len(X_train) < 20:
        return None, 0.0, [], {}, None, None

    last_row = X[-1:]
    if np.isnan(last_row).any():
        return None, 0.0, [], {}, None, None   # can't predict today without valid features

    tscv = TimeSeriesSplit(n_splits=3)
    r2_scores = []
    for train_idx, val_idx in tscv.split(X_train):
        if len(train_idx) < 10:
            continue
        m = RandomForestRegressor(
            n_estimators=100, max_depth=4, min_samples_leaf=10,
            random_state=42, n_jobs=-1
        )
        m.fit(X_train[train_idx], y_train[train_idx])
        r2_scores.append(m.score(X_train[val_idx], y_train[val_idx]))

    r2 = float(np.mean(r2_scores)) if r2_scores else 0.0

    model = RandomForestRegressor(
        n_estimators=200, max_depth=4, min_samples_leaf=10,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_predicted  = model.predict(X_train)
    pred_return  = float(model.predict(last_row.reshape(1, -1))[0])

    top5       = pred_cols[:5]  # Hardcoded. Top 5 most correlated tickers for target
    corr_signs = {col: float(corr_matrix[target][col]) for col in top5}

    return pred_return, r2, top5, corr_signs, y_train, y_predicted
