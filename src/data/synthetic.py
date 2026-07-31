"""Offline synthetic market-data source.

A drop-in replacement for :class:`~src.data.binance_client.BinanceClient` that
generates deterministic, regime-switching OHLCV data. It exists for two reasons:

1. the full pipeline can be trained, scanned and demonstrated **without network
   access** (useful in sandboxes where the exchange is unreachable);
2. tests and examples get reproducible data with mild, learnable structure
   (regime-dependent drift/volatility plus momentum autocorrelation), so the
   modelling stack has something real to find.

It implements the same read methods the rest of the code calls on the Binance
client (``get_klines``, ``get_klines_range``) and the context-manager protocol,
so it is interchangeable in ``main.py``.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from ..utils.timing import INTERVAL_MINUTES, floor_to_interval

logger = get_logger(__name__)

# Hidden regimes: (drift per bar, volatility per bar). Loosely trend / range /
# high-vol, mirroring the classifier's vocabulary.
_REGIMES = {
    "trend": (0.0006, 0.004),
    "range": (0.0000, 0.003),
    "high_vol": (0.0000, 0.011),
}
_REGIME_NAMES = list(_REGIMES)

# Markov transition matrix (rows sum to 1); regimes are persistent.
_TRANSITION = np.array(
    [
        [0.985, 0.010, 0.005],  # trend ->
        [0.010, 0.985, 0.005],  # range ->
        [0.020, 0.020, 0.960],  # high_vol ->
    ]
)

_MOMENTUM_RHO = 0.08  # AR(1) coefficient giving returns mild persistence


def _seed_for(symbol: str, base_seed: int) -> int:
    digest = hashlib.md5(f"{symbol}:{base_seed}".encode()).hexdigest()
    return int(digest[:8], 16)


def generate_ohlcv(
    n: int,
    *,
    freq_minutes: int,
    seed: int,
    end_time: datetime,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Generate ``n`` OHLCV bars ending at ``end_time`` (exclusive of the future)."""
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)

    # Regime path via the Markov chain.
    states = np.empty(n, dtype=int)
    states[0] = rng.integers(0, len(_REGIME_NAMES))
    for i in range(1, n):
        states[i] = rng.choice(len(_REGIME_NAMES), p=_TRANSITION[states[i - 1]])

    drifts = np.array([_REGIMES[_REGIME_NAMES[s]][0] for s in states])
    vols = np.array([_REGIMES[_REGIME_NAMES[s]][1] for s in states])

    # Returns with mild AR(1) momentum on top of regime drift/vol.
    eps = rng.normal(0.0, 1.0, n)
    rets = np.empty(n)
    rets[0] = drifts[0] + vols[0] * eps[0]
    for i in range(1, n):
        rets[i] = drifts[i] + _MOMENTUM_RHO * rets[i - 1] + vols[i] * eps[i]

    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]

    # Intrabar wiggle scaled by the bar's volatility.
    wig = np.abs(rng.normal(0.0, 1.0, n)) * vols * close
    high = np.maximum(open_, close) + wig
    low = np.minimum(open_, close) - wig
    volume = np.abs(rng.normal(1.0, 0.3, n)) * (1.0 + 40.0 * vols) * 1000.0

    end_aligned = floor_to_interval(end_time, _minutes_to_interval(freq_minutes))
    idx = pd.date_range(end=end_aligned, periods=n, freq=f"{freq_minutes}min", tz="UTC", name="open_time")

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume,
         "trades": (volume / 10).astype("int64")},
        index=idx,
    )


def _minutes_to_interval(minutes: int) -> str:
    for name, m in INTERVAL_MINUTES.items():
        if m == minutes:
            return name
    return f"{minutes}m"


class SyntheticClient:
    """Offline stand-in for BinanceClient exposing the read API surface."""

    def __init__(self, *, base_seed: int = 12345, now: datetime | None = None) -> None:
        self.base_seed = base_seed
        self._now = now or datetime.now(timezone.utc)

    def __enter__(self) -> "SyntheticClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:  # symmetry with BinanceClient
        pass

    def _freq_minutes(self, interval: str) -> int:
        if interval not in INTERVAL_MINUTES:
            raise ValueError(f"unsupported interval {interval!r}")
        return INTERVAL_MINUTES[interval]

    def get_klines(self, symbol: str, interval: str, *, limit: int = 1000, **_: object) -> pd.DataFrame:
        minutes = self._freq_minutes(interval)
        return generate_ohlcv(
            limit, freq_minutes=minutes, seed=_seed_for(symbol, self.base_seed), end_time=self._now
        )

    def get_klines_range(
        self, symbol: str, interval: str, *, start_time: object, end_time: object = None, **_: object
    ) -> pd.DataFrame:
        minutes = self._freq_minutes(interval)
        end = pd.Timestamp(end_time) if end_time is not None else pd.Timestamp(self._now)
        if end.tzinfo is None:
            end = end.tz_localize("UTC")
        start = pd.Timestamp(start_time)
        if start.tzinfo is None:
            start = start.tz_localize("UTC")
        span_minutes = max((end - start).total_seconds() / 60.0, minutes)
        n = min(int(span_minutes // minutes) + 1, 200_000)
        return generate_ohlcv(
            n, freq_minutes=minutes, seed=_seed_for(symbol, self.base_seed), end_time=end.to_pydatetime()
        )
