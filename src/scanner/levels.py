"""Derive actionable trade levels from a signal and volatility.

Given the current price and ATR, this computes a professional trade plan for a
side:

- an **entry zone** (a band, not a single price): for a long you want to buy a
  small dip, for a short to sell a small bounce;
- a **stop loss** at ``sl_mult`` ATR against the position;
- **TP1** at 1 ATR (the ~1:1 partial) and **TP2** at ``tp_mult`` ATR (the
  triple-barrier target the model was trained on).

Multipliers default to the training barrier (``tp_mult=2``, ``sl_mult=1``) but
are read from the artifact when present, so levels always match the labels the
model actually learned.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TradeLevels:
    side: str          # "long" or "short"
    entry_low: float
    entry_high: float
    stop_loss: float
    tp1: float
    tp2: float
    sl_pct: float      # signed % from reference price
    tp1_pct: float
    tp2_pct: float

    def as_dict(self) -> dict:
        return {
            "entry_low": self.entry_low,
            "entry_high": self.entry_high,
            "stop_loss": self.stop_loss,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "sl_pct": self.sl_pct,
            "tp1_pct": self.tp1_pct,
            "tp2_pct": self.tp2_pct,
        }


def compute_levels(
    close: float,
    atr: float,
    side: str,
    *,
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    tp1_mult: float = 1.0,
    entry_band_mult: float = 0.5,
) -> TradeLevels:
    """Compute entry zone, stop and take-profits for ``side`` from ATR.

    ``close`` is the reference (last closed price); ``atr`` is the current ATR
    in price units. Percentages are relative to ``close``.
    """
    if side not in ("long", "short"):
        raise ValueError("side must be 'long' or 'short'")
    if not (atr > 0):
        # Degenerate volatility: collapse the zone to the price.
        atr = 0.0

    band = entry_band_mult * atr
    if side == "long":
        entry_low, entry_high = close - band, close
        stop_loss = close - sl_mult * atr
        tp1 = close + tp1_mult * atr
        tp2 = close + tp_mult * atr
    else:  # short
        entry_low, entry_high = close, close + band
        stop_loss = close + sl_mult * atr
        tp1 = close - tp1_mult * atr
        tp2 = close - tp_mult * atr

    def pct(level: float) -> float:
        return (level - close) / close * 100.0 if close else 0.0

    return TradeLevels(
        side=side,
        entry_low=entry_low,
        entry_high=entry_high,
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        sl_pct=pct(stop_loss),
        tp1_pct=pct(tp1),
        tp2_pct=pct(tp2),
    )
