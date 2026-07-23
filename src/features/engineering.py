"""Multi-timeframe feature engineering with strict causal alignment.

Two conventions keep the feature matrix free of look-ahead bias:

1. Per-timeframe indicators are causal (see :mod:`.indicators`), so the feature
   row for a bar uses only that bar and earlier bars.
2. Higher-timeframe (HTF) features are attached to a base bar only if the HTF
   bar has already **closed** by the time the base bar closes. This is done with
   a backward ``merge_asof`` keyed on close times, so at base bar ``b`` the model
   never sees an HTF candle that is still forming.

Labels (phase 3) are defined strictly forward from a base bar's close, so a
feature row at bar ``b`` (using data through the close of ``b``) paired with a
forward label is causally consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..logging_utils import get_logger
from .indicators import (
    adx,
    atr,
    bollinger,
    ema,
    log_returns,
    realized_volatility,
    rolling_zscore,
    rsi,
)
from ..data.resample import TIMEFRAME_TO_PANDAS

logger = get_logger(__name__)


@dataclass(frozen=True)
class FeatureParams:
    """Indicator periods used by :func:`timeframe_features`."""

    rsi_period: int = 14
    atr_period: int = 14
    adx_period: int = 14
    ema_fast: int = 12
    ema_slow: int = 26
    vol_window: int = 20
    bb_period: int = 20
    zscore_window: int = 50
    momentum_lags: tuple[int, ...] = (1, 3, 6, 12)


def timeframe_features(
    df: pd.DataFrame,
    params: FeatureParams | None = None,
) -> pd.DataFrame:
    """Compute the indicator feature set for a single timeframe.

    Parameters
    ----------
    df:
        OHLCV frame indexed by open time. Requires ``open/high/low/close`` and,
        for volume features, ``volume``.
    """
    params = params or FeatureParams()
    high, low, close = df["high"], df["low"], df["close"]

    feats: dict[str, pd.Series] = {}
    feats["ret"] = log_returns(close)

    feats["rsi"] = rsi(close, params.rsi_period)

    atr_ = atr(high, low, close, params.atr_period)
    feats["atr"] = atr_
    feats["atr_pct"] = atr_ / close

    adx_df = adx(high, low, close, params.adx_period)
    feats["adx"] = adx_df["adx"]
    feats["plus_di"] = adx_df["plus_di"]
    feats["minus_di"] = adx_df["minus_di"]
    feats["di_spread"] = adx_df["plus_di"] - adx_df["minus_di"]

    feats["rvol"] = realized_volatility(close, params.vol_window)

    ema_fast = ema(close, params.ema_fast)
    ema_slow = ema(close, params.ema_slow)
    feats["ema_ratio"] = ema_fast / ema_slow - 1.0
    feats["price_vs_ema_slow"] = close / ema_slow - 1.0
    # EMA crossover events (fast crossing slow) — a classic trend-shift trigger.
    ema_gap = ema_fast - ema_slow
    feats["ema_cross_up"] = _cross_up(ema_gap, 0.0).astype("float64")
    feats["ema_cross_dn"] = _cross_dn(ema_gap, 0.0).astype("float64")

    bb = bollinger(close, params.bb_period)
    feats["bb_width"] = bb["bb_width"]
    feats["bb_pct_b"] = bb["bb_pct_b"]
    # Bollinger mid-band distance and crossings, plus a wide-band regime flag —
    # signals are more reliable when price crosses the mid-band with wide bands.
    feats["bb_mid_dist"] = close / bb["bb_mid"] - 1.0
    feats["bb_mid_cross_up"] = _cross_up(close, bb["bb_mid"]).astype("float64")
    feats["bb_mid_cross_dn"] = _cross_dn(close, bb["bb_mid"]).astype("float64")
    bb_width_med = bb["bb_width"].rolling(params.zscore_window, min_periods=params.bb_period).median()
    feats["bb_wide"] = (bb["bb_width"] > bb_width_med).astype("float64")

    for lag in params.momentum_lags:
        feats[f"mom_{lag}"] = close / close.shift(lag) - 1.0

    if "volume" in df.columns:
        feats["vol_z"] = rolling_zscore(df["volume"], params.zscore_window)

    return pd.DataFrame(feats, index=df.index)


def _cross_up(a: pd.Series, b) -> pd.Series:
    """True on the bar where ``a`` crosses from <= ``b`` to > ``b``."""
    b_series = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    prev = a.shift(1) <= b_series.shift(1)
    now = a > b_series
    return (prev & now).fillna(False)


def _cross_dn(a: pd.Series, b) -> pd.Series:
    """True on the bar where ``a`` crosses from >= ``b`` to < ``b``."""
    b_series = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    prev = a.shift(1) >= b_series.shift(1)
    now = a < b_series
    return (prev & now).fillna(False)


def _timeframe_delta(timeframe: str) -> pd.Timedelta:
    return pd.Timedelta(TIMEFRAME_TO_PANDAS.get(timeframe, timeframe))


def build_feature_matrix(
    frames: dict[str, pd.DataFrame],
    *,
    base_timeframe: str,
    htf_timeframes: list[str] | None = None,
    params: FeatureParams | None = None,
    dropna: bool = True,
) -> pd.DataFrame:
    """Assemble a causal multi-timeframe feature matrix on the base timeframe.

    Parameters
    ----------
    frames:
        Mapping ``{timeframe: ohlcv_frame}`` (e.g. the output of
        :func:`src.data.resample.resample_to_timeframes`). Must contain
        ``base_timeframe``.
    htf_timeframes:
        Higher timeframes to merge onto the base. Defaults to every key in
        ``frames`` other than the base timeframe.
    dropna:
        Drop leading rows that contain NaNs from indicator warm-up.
    """
    if base_timeframe not in frames:
        raise KeyError(f"frames is missing the base timeframe {base_timeframe!r}")

    params = params or FeatureParams()
    if htf_timeframes is None:
        htf_timeframes = [tf for tf in frames if tf != base_timeframe]

    base_df = frames[base_timeframe]
    base_feats = timeframe_features(base_df, params)
    base_feats = base_feats.add_prefix(f"{base_timeframe}_")

    matrix = base_feats.copy()
    base_close = base_df.index + _timeframe_delta(base_timeframe)

    for tf in htf_timeframes:
        if tf not in frames:
            raise KeyError(f"frames is missing higher timeframe {tf!r}")
        htf_feats = timeframe_features(frames[tf], params).add_prefix(f"{tf}_")
        htf_close = frames[tf].index + _timeframe_delta(tf)

        left = pd.DataFrame({"_base_close": base_close}, index=base_df.index).reset_index(names="_base_open")
        right = htf_feats.copy()
        right["_htf_close"] = htf_close
        right = right.sort_values("_htf_close")

        merged = pd.merge_asof(
            left.sort_values("_base_close"),
            right,
            left_on="_base_close",
            right_on="_htf_close",
            direction="backward",
            # Strict: only HTF bars that closed *before* this base bar's close,
            # so HTF features stay constant across a forming HTF window and can
            # never incorporate the simultaneously-closing HTF candle.
            allow_exact_matches=False,
        )
        merged = merged.set_index("_base_open").sort_index()
        merged = merged.drop(columns=["_base_close", "_htf_close"])
        matrix = matrix.join(merged)

    matrix.index.name = base_df.index.name or "open_time"
    if dropna:
        matrix = matrix.dropna()
    return matrix
