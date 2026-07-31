"""Resample a base-timeframe OHLCV series to higher timeframes (HTF).

Binance klines are labelled by their *open* time and cover the half-open
interval ``[open_time, open_time + interval)``. Resampling therefore uses
``closed="left", label="left"`` so an aggregated bar keeps the open time of its
first constituent bar — matching how the exchange itself labels candles.

The aggregation rule is the standard OHLCV rollup: ``open`` from the first bar,
``high``/``low`` as the extrema, ``close`` from the last bar and additive
columns (volume, quote volume, trade count, taker-buy volumes) summed.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import pandas as pd

from ..logging_utils import get_logger

logger = get_logger(__name__)

# Canonical timeframe strings -> pandas offset aliases (pandas >= 2.2 spelling).
TIMEFRAME_TO_PANDAS: dict[str, str] = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "12h": "12h",
    "1d": "1D",
}

# How each known column is aggregated. Columns not listed here are dropped.
_AGGREGATIONS: Mapping[str, str] = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "quote_volume": "sum",
    "trades": "sum",
    "taker_buy_base": "sum",
    "taker_buy_quote": "sum",
}


def _to_pandas_rule(timeframe: str) -> str:
    """Translate a canonical timeframe (``"1h"``) to a pandas offset alias."""
    if timeframe in TIMEFRAME_TO_PANDAS:
        return TIMEFRAME_TO_PANDAS[timeframe]
    # Allow passing a pandas alias through unchanged.
    return timeframe


def _normalise_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("resample requires a DatetimeIndex (open_time)")
    if df.index.tz is None:
        df = df.tz_localize("UTC")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    return df


def resample_ohlcv(
    df: pd.DataFrame,
    timeframe: str,
    *,
    base_timeframe: str | None = None,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """Resample an OHLCV frame to ``timeframe``.

    Parameters
    ----------
    df:
        OHLCV frame indexed by a (UTC) ``DatetimeIndex`` of open times. Must at
        least contain ``open``, ``high``, ``low``, ``close``; additive columns
        are aggregated when present.
    timeframe:
        Target timeframe, canonical (``"1h"``) or a pandas alias.
    base_timeframe:
        Timeframe of the input bars. Required when ``drop_incomplete`` is set so
        the completeness of the final aggregated bar can be determined.
    drop_incomplete:
        Drop the last aggregated bar if it is not fully covered by base bars
        (i.e. the currently forming candle). Defaults to ``True`` to avoid
        leaking look-ahead from a partially formed HTF candle.
    """
    if df.empty:
        return df.copy()

    df = _normalise_index(df)

    agg = {col: how for col, how in _AGGREGATIONS.items() if col in df.columns}
    missing = {"open", "high", "low", "close"} - set(agg)
    if missing:
        raise ValueError(f"missing required OHLC column(s): {sorted(missing)}")

    rule = _to_pandas_rule(timeframe)
    resampler = df.resample(rule, closed="left", label="left")
    resampled = resampler.agg(agg)
    counts = resampler.size().reindex(resampled.index, fill_value=0)

    # Drop empty periods created by gaps in the base series.
    non_empty = counts > 0
    resampled = resampled[non_empty]
    counts = counts[non_empty]

    if drop_incomplete and len(resampled):
        if base_timeframe is None:
            raise ValueError("base_timeframe is required when drop_incomplete=True")
        expected = pd.Timedelta(rule) / pd.Timedelta(_to_pandas_rule(base_timeframe))
        if counts.iloc[-1] < expected:
            logger.debug(
                "Dropping incomplete %s bar at %s (%d/%g base bars)",
                timeframe,
                resampled.index[-1],
                counts.iloc[-1],
                expected,
            )
            resampled = resampled.iloc[:-1]

    return resampled


def resample_to_timeframes(
    df: pd.DataFrame,
    timeframes: Iterable[str],
    *,
    base_timeframe: str,
    drop_incomplete: bool = True,
) -> dict[str, pd.DataFrame]:
    """Resample ``df`` to each timeframe, returning ``{timeframe: frame}``.

    The base timeframe is included unchanged (only index-normalised) when it
    appears in ``timeframes``.
    """
    out: dict[str, pd.DataFrame] = {}
    for tf in timeframes:
        if tf == base_timeframe:
            out[tf] = _normalise_index(df.copy())
        else:
            out[tf] = resample_ohlcv(
                df,
                tf,
                base_timeframe=base_timeframe,
                drop_incomplete=drop_incomplete,
            )
    return out
