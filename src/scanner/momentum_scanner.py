"""Training-free technical breakout / momentum scanner.

Complements the ML scanner: no models, no training — it scores every symbol
from pure price/volume structure on the last closed bar, so it can sweep the
whole futures universe instantly and surface the *movers* the ML model (tuned
for modest, liquid moves) tends to filter out.

Signals scored per side (the user's own playbook):
- Bollinger band breakout (close beyond the outer band);
- Bollinger **mid-band** crossover;
- RSI reversal out of oversold/overbought;
- reaction off a recent low/high (dip bounce / top rejection);
- EMA fast/slow crossover;
- volume spike and band-width expansion as confirmation.

Output uses the same schema as the ML signals report, so the existing terminal
and HTML renderers apply unchanged (``prob`` carries the technical score).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features.indicators import atr, bollinger, ema, rsi
from ..logging_utils import get_logger
from .futures_scanner import _SIGNAL_COLUMNS, _empty_signal_frame
from .levels import compute_levels

logger = get_logger(__name__)


@dataclass(frozen=True)
class MomentumParams:
    bb_period: int = 20
    atr_period: int = 14
    atr_ma: int = 50
    rsi_period: int = 14
    ema_fast: int = 12
    ema_slow: int = 26
    vol_ma: int = 20
    lookback: int = 48          # bars for the recent high/low (dip/top proximity)
    vol_mult: float = 1.5       # volume-spike threshold vs its moving average
    atr_mult: float = 1.2       # band-expansion threshold vs its moving average
    near_pct: float = 0.02      # "within X% of the recent low/high"
    rsi_low: float = 35.0
    rsi_high: float = 65.0
    min_atr_pct: float = 0.0015  # skip untradeably quiet coins (ATR < 0.15% of price)
    tp_mult: float = 3.0
    sl_mult: float = 1.5


def _verdict(score: float) -> str:
    if score >= 0.6:
        return "UYGUN"
    if score >= 0.35:
        return "DİKKATLİ"
    return "ZAYIF"


def momentum_signal(df: pd.DataFrame, params: MomentumParams | None = None) -> dict | None:
    """Score the last closed bar for bullish (long) and bearish (short) setups."""
    p = params or MomentumParams()
    if df is None or len(df) < max(p.lookback, p.atr_ma) + 5:
        return None

    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    bb = bollinger(close, p.bb_period)
    atr_ = atr(high, low, close, p.atr_period)
    atr_ma = atr_.rolling(p.atr_ma, min_periods=p.atr_period).mean()
    rsi_ = rsi(close, p.rsi_period)
    ema_f, ema_s = ema(close, p.ema_fast), ema(close, p.ema_slow)
    vol_ma = vol.rolling(p.vol_ma, min_periods=p.vol_ma).mean()
    rlow = low.rolling(p.lookback, min_periods=p.lookback).min()
    rhigh = high.rolling(p.lookback, min_periods=p.lookback).max()

    c, cp = close.iloc[-1], close.iloc[-2]
    a = float(atr_.iloc[-1])
    if not np.isfinite(a) or a <= 0:
        return None
    # Volatility floor: on a near-flat series the Bollinger bands collapse and
    # noise would fake breakouts. Skip coins that aren't actually moving.
    if c <= 0 or a / c < p.min_atr_pct:
        return {"close": float(c), "atr": a, "long_score": 0.0, "short_score": 0.0,
                "long_reasons": [], "short_reasons": []}

    long_s, short_s = 0.0, 0.0
    lr: list[str] = []
    sr: list[str] = []

    # Band breakout.
    if c > bb["bb_upper"].iloc[-1]:
        long_s += 0.30; lr.append("üst bant kırılımı")
    if c < bb["bb_lower"].iloc[-1]:
        short_s += 0.30; sr.append("alt bant kırılımı")

    # Mid-band crossover (this bar).
    mid, midp = bb["bb_mid"].iloc[-1], bb["bb_mid"].iloc[-2]
    if cp <= midp and c > mid:
        long_s += 0.20; lr.append("orta bant yukarı kesişim")
    if cp >= midp and c < mid:
        short_s += 0.20; sr.append("orta bant aşağı kesişim")

    # RSI reversal.
    if rsi_.iloc[-2] < p.rsi_low <= rsi_.iloc[-1]:
        long_s += 0.20; lr.append("RSI dipten dönüş")
    if rsi_.iloc[-2] > p.rsi_high >= rsi_.iloc[-1]:
        short_s += 0.20; sr.append("RSI tepeden dönüş")

    # Reaction off a recent extreme (dip bounce / top rejection).
    if np.isfinite(rlow.iloc[-1]) and (cp - rlow.iloc[-1]) / rlow.iloc[-1] < p.near_pct and c > cp:
        long_s += 0.20; lr.append("dipten dönüş")
    if np.isfinite(rhigh.iloc[-1]) and (rhigh.iloc[-1] - cp) / rhigh.iloc[-1] < p.near_pct and c < cp:
        short_s += 0.20; sr.append("tepeden dönüş")

    # EMA crossover.
    if ema_f.iloc[-2] <= ema_s.iloc[-2] and ema_f.iloc[-1] > ema_s.iloc[-1]:
        long_s += 0.15; lr.append("EMA yukarı kesişim")
    if ema_f.iloc[-2] >= ema_s.iloc[-2] and ema_f.iloc[-1] < ema_s.iloc[-1]:
        short_s += 0.15; sr.append("EMA aşağı kesişim")

    # Confirmations (only reinforce an existing directional signal).
    vspike = np.isfinite(vol_ma.iloc[-1]) and vol.iloc[-1] > p.vol_mult * vol_ma.iloc[-1]
    expanding = np.isfinite(atr_ma.iloc[-1]) and a > p.atr_mult * atr_ma.iloc[-1]
    for flag, txt, pts in ((vspike, "hacim spike", 0.15), (expanding, "bant genişliyor", 0.10)):
        if not flag:
            continue
        if long_s > 0:
            long_s += pts; lr.append(txt)
        if short_s > 0:
            short_s += pts; sr.append(txt)

    return {
        "close": float(c),
        "atr": a,
        "long_score": min(long_s, 1.0),
        "short_score": min(short_s, 1.0),
        "long_reasons": lr,
        "short_reasons": sr,
    }


class MomentumScanner:
    def __init__(self, client, *, base_timeframe: str = "1h", history_bars: int = 300,
                 params: MomentumParams | None = None) -> None:
        self.client = client
        self.base_timeframe = base_timeframe
        self.history_bars = history_bars
        self.params = params or MomentumParams()

    def scan_signals(
        self,
        symbols: list[str],
        *,
        top_n: int = 10,
        max_move_pct: float | None = None,
        fundamentals: dict | None = None,
        min_quote_volume: float = 0.0,
        min_score: float = 0.35,
    ) -> dict[str, pd.DataFrame]:
        fundamentals = fundamentals or {}
        rows = []
        for sym in symbols:
            try:
                df = self.client.get_klines(sym, self.base_timeframe, limit=self.history_bars)
                sig = momentum_signal(df, self.params)
            except Exception as exc:
                logger.warning("Momentum scan failed for %s: %s", sym, exc)
                continue
            if sig is None:
                continue
            fund = fundamentals.get(sym, {})
            vol = fund.get("quote_volume", float("nan"))
            if min_quote_volume > 0 and (vol != vol or vol < min_quote_volume):
                continue
            side = "long" if sig["long_score"] >= sig["short_score"] else "short"
            score = sig[f"{side}_score"]
            if score < min_score:
                continue
            rows.append({**sig, "symbol": sym, "side": side, "score": score,
                         "reasons": sig[f"{side}_reasons"], "fund": fund})

        longs = [r for r in rows if r["side"] == "long"]
        shorts = [r for r in rows if r["side"] == "short"]
        return {
            "long": self._build(longs, "long", top_n, max_move_pct),
            "short": self._build(shorts, "short", top_n, max_move_pct),
        }

    def _build(self, items: list[dict], side: str, top_n: int, max_move_pct: float | None) -> pd.DataFrame:
        p = self.params
        out = []
        for r in sorted(items, key=lambda x: x["score"], reverse=True):
            lv = compute_levels(r["close"], r["atr"], side, tp_mult=p.tp_mult, sl_mult=p.sl_mult)
            if max_move_pct is not None and max_move_pct > 0 and abs(lv.tp2_pct) > max_move_pct:
                continue
            fund = r["fund"]
            out.append({
                "symbol": r["symbol"], "side": side.upper(), "prob": r["score"],
                "confident": r["score"] >= 0.6, "verdict": _verdict(r["score"]),
                "note": ", ".join(r["reasons"]) or "-", "regime": "-", "price": r["close"],
                "quote_volume": fund.get("quote_volume", float("nan")),
                "funding": fund.get("funding", float("nan")),
                "ls_ratio": fund.get("ls_ratio", float("nan")),
                "open_interest": fund.get("open_interest", float("nan")),
                "entry_low": lv.entry_low, "entry_high": lv.entry_high,
                "stop_loss": lv.stop_loss, "tp1": lv.tp1, "tp2": lv.tp2,
                "sl_pct": lv.sl_pct, "tp1_pct": lv.tp1_pct, "tp2_pct": lv.tp2_pct,
            })
            if len(out) >= top_n:
                break
        return pd.DataFrame(out, columns=_SIGNAL_COLUMNS) if out else _empty_signal_frame()
