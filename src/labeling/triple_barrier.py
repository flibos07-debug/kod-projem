"""Triple-barrier labelling (Lopez de Prado).

Each event bar is assigned a profit-take (upper) and stop-loss (lower) barrier,
plus a vertical (time) barrier. Looking forward from the event, the label is
determined by whichever barrier is touched first:

- ``+1`` if the profit-take barrier is touched first;
- ``-1`` if the stop-loss barrier is touched first;
- ``0`` if the vertical barrier (max holding) is reached first (a timeout).

Barrier widths are set from a per-bar volatility estimate (typically ATR), so
they adapt to changing market conditions. Intrabar ``high``/``low`` are used to
detect touches, which is more realistic than close-to-close.

The returned ``t1`` (time of first touch / label resolution) is what the
walk-forward splitter uses to *purge* training samples whose label windows
overlap the validation/test periods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class BarrierParams:
    """Configuration for :func:`triple_barrier_labels`.

    Attributes
    ----------
    tp_mult, sl_mult:
        Profit-take / stop-loss widths as multiples of the volatility unit.
    max_holding:
        Vertical barrier, in bars.
    side:
        ``+1`` for long events, ``-1`` for short. The label sign is expressed
        relative to the trade direction (``+1`` always means "worked out").
    """

    tp_mult: float = 2.0
    sl_mult: float = 1.0
    max_holding: int = 12
    side: int = 1

    def __post_init__(self) -> None:
        if self.side not in (1, -1):
            raise ValueError("side must be +1 (long) or -1 (short)")
        if self.tp_mult <= 0 or self.sl_mult <= 0:
            raise ValueError("tp_mult and sl_mult must be positive")
        if self.max_holding < 1:
            raise ValueError("max_holding must be >= 1")


def triple_barrier_labels(
    df: pd.DataFrame,
    volatility: pd.Series,
    params: BarrierParams | None = None,
    *,
    event_index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Apply the triple-barrier method to ``df``.

    Parameters
    ----------
    df:
        OHLCV frame indexed by open time; needs ``high``, ``low``, ``close``.
    volatility:
        Per-bar volatility unit (e.g. ATR) aligned to ``df.index``. Barrier
        distances are ``mult * volatility`` at the event bar.
    event_index:
        Subset of ``df.index`` at which to create events. Defaults to every bar
        that has a finite volatility value.

    Returns
    -------
    DataFrame indexed by event time with columns: ``entry_price``, ``tp``,
    ``sl``, ``t1`` (first-touch time), ``exit_price``, ``ret`` (signed return in
    the trade direction) and ``label`` (+1/-1/0).
    """
    params = params or BarrierParams()
    required = {"high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing required column(s): {sorted(missing)}")

    df = df.sort_index()
    volatility = volatility.reindex(df.index)

    if event_index is None:
        event_index = df.index[volatility.notna() & (volatility > 0)]
    event_index = pd.DatetimeIndex(event_index)

    highs = df["high"].to_numpy(dtype="float64")
    lows = df["low"].to_numpy(dtype="float64")
    closes = df["close"].to_numpy(dtype="float64")
    times = df.index
    pos_of = {ts: i for i, ts in enumerate(times)}

    n = len(times)
    side = params.side

    records = []
    for ts in event_index:
        i = pos_of.get(ts)
        if i is None:
            continue
        vol = volatility.iat[i]
        if not np.isfinite(vol) or vol <= 0:
            continue

        entry = closes[i]
        tp = entry + side * params.tp_mult * vol
        sl = entry - side * params.sl_mult * vol

        end = min(i + params.max_holding, n - 1)
        label = 0
        t1_pos = end
        exit_price = closes[end]

        # Scan forward bar-by-bar; the entry bar itself (i) is excluded so the
        # label uses only strictly future price action.
        for j in range(i + 1, end + 1):
            hi, lo = highs[j], lows[j]
            hit_tp = hi >= tp if side == 1 else lo <= tp
            hit_sl = lo <= sl if side == 1 else hi >= sl
            if hit_tp and hit_sl:
                # Both barriers within the same bar: assume the stop is hit
                # first (conservative).
                label, t1_pos, exit_price = -1, j, sl
                break
            if hit_tp:
                label, t1_pos, exit_price = 1, j, tp
                break
            if hit_sl:
                label, t1_pos, exit_price = -1, j, sl
                break

        ret = side * (exit_price / entry - 1.0)
        records.append(
            {
                "entry_time": ts,
                "entry_price": entry,
                "tp": tp,
                "sl": sl,
                "t1": times[t1_pos],
                "exit_price": exit_price,
                "ret": ret,
                "label": label,
            }
        )

    if not records:
        return pd.DataFrame(
            columns=["entry_price", "tp", "sl", "t1", "exit_price", "ret", "label"]
        )

    out = pd.DataFrame.from_records(records).set_index("entry_time")
    out.index.name = df.index.name or "open_time"
    return out


def binary_target(labels: pd.DataFrame, *, timeout_as_loss: bool = True) -> pd.Series:
    """Map triple-barrier labels to a binary classification target.

    ``+1`` (profit-take) -> 1. ``-1`` (stop) -> 0. ``0`` (timeout) -> 0 when
    ``timeout_as_loss`` else it is dropped (NaN).
    """
    mapping = labels["label"].map({1: 1, -1: 0, 0: 0 if timeout_as_loss else np.nan})
    return mapping.rename("target")
