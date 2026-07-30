"""Vectorised technical indicators (pure pandas/numpy, no TA-Lib dependency).

All functions are causal: the value at bar *t* uses only information from bars
``<= t``. This is essential to avoid look-ahead bias when the outputs feed a
predictive model. ``ewm(alpha=1/period, adjust=False)`` implements Wilder's
smoothing (a.k.a. RMA), the convention used by ATR/ADX/RSI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (running moving average) with the given period."""
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def log_returns(close: pd.Series) -> pd.Series:
    return np.log(close / close.shift(1))


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def realized_volatility(close: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation of log returns over ``window`` bars."""
    return log_returns(close).rolling(window, min_periods=window).std(ddof=0)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder)."""
    return _wilder(true_range(high, low, close), period)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder smoothing)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = _wilder(gain, period)
    avg_loss = _wilder(loss, period)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    result = 100.0 - 100.0 / (1.0 + rs)
    # When average loss is zero and there were gains, RSI is 100.
    result = result.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    return result


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """Average Directional Index with the directional indicators.

    Returns a frame with columns ``adx``, ``plus_di`` and ``minus_di``.
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    atr_ = _wilder(true_range(high, low, close), period)
    plus_di = 100.0 * _wilder(plus_dm, period) / atr_.replace(0.0, np.nan)
    minus_di = 100.0 * _wilder(minus_dm, period) / atr_.replace(0.0, np.nan)

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx_ = _wilder(dx, period)

    return pd.DataFrame(
        {"adx": adx_, "plus_di": plus_di, "minus_di": minus_di},
        index=high.index,
    )


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line and histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig}, index=close.index)


def stoch_rsi(
    close: pd.Series, rsi_period: int = 14, stoch_period: int = 14, k: int = 3, d: int = 3
) -> pd.DataFrame:
    """Stochastic RSI (%K, %D) in 0-100 — a faster, leading momentum oscillator."""
    r = rsi(close, rsi_period)
    lo = r.rolling(stoch_period, min_periods=stoch_period).min()
    hi = r.rolling(stoch_period, min_periods=stoch_period).max()
    stoch = (r - lo) / (hi - lo).replace(0.0, np.nan)
    k_line = (stoch * 100.0).rolling(k, min_periods=k).mean()
    d_line = k_line.rolling(d, min_periods=d).mean()
    return pd.DataFrame({"k": k_line, "d": d_line}, index=close.index)


def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger bands and the %B / bandwidth derived features."""
    mid = sma(close, period)
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / mid.replace(0.0, np.nan)
    pct_b = (close - lower) / (upper - lower).replace(0.0, np.nan)
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_width": width, "bb_pct_b": pct_b},
        index=close.index,
    )
