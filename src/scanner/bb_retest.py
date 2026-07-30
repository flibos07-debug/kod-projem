"""Bollinger squeeze -> mid-band break -> retest strategy scanner.

A precise, mechanical setup (no ML):

Higher timeframe (15m) sets the direction and requires low volatility first:
- Bollinger bands must be **squeezed and flat** (volatility contracted), then
- price breaks the mid-band (20-SMA) and **closes** beyond it:
  up = long bias, down = short bias.

Entry timeframe (5m) times the entry via a retest:
- LONG: after the 15m up-break, price pulls back to the 5m **lower** band.
- SHORT: after the 15m down-break, price pulls back to the 5m **upper** band.

Stops and targets:
- SL just beyond the entry (last 5m) candle's low/high.
- TP at the opposite outer band on 15m (long -> upper band, short -> lower band).

Each coin is classified into a stage: GİRİŞ (retest hit — enter now),
BEKLE (broke out, waiting for the retest) or İZLE (squeezing, watch).
"""

from __future__ import annotations

import concurrent.futures as cf
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.resample import resample_to_timeframes
from ..features.indicators import atr, bollinger
from ..logging_utils import get_logger

logger = get_logger(__name__)

_STAGE_SCORE = {"GİRİŞ": 3, "BEKLE": 2, "İZLE": 1}


@dataclass(frozen=True)
class BBStratParams:
    bb_period: int = 20
    squeeze_lookback: int = 30
    squeeze_q: float = 0.30       # bandwidth in the lowest 30% = squeezed
    flat_bars: int = 6            # mid-band slope measured over this many 15m bars
    flat_max_slope: float = 0.006  # |slope| < 0.6% over flat_bars = "flat"
    break_window: int = 6         # mid-band break must be within the last N 15m bars
    retest_low: float = 0.15      # 5m %B <= this = at the lower band (long retest)
    retest_high: float = 0.85     # 5m %B >= this = at the upper band (short retest)
    sl_buffer_atr: float = 0.3    # stop = candle extreme -/+ this * 5m ATR


def _mid_break_bars_ago(close: pd.Series, mid: pd.Series, window: int, direction: str) -> int | None:
    """Bars since price crossed and closed beyond the mid-band (None if not in window)."""
    n = len(close)
    for j in range(n - 1, max(n - 1 - window, 0), -1):
        if j - 1 < 0:
            break
        cp, c = close.iloc[j - 1], close.iloc[j]
        mp, m = mid.iloc[j - 1], mid.iloc[j]
        if direction == "up" and cp <= mp and c > m:
            return n - 1 - j
        if direction == "down" and cp >= mp and c < m:
            return n - 1 - j
    return None


def bb_retest_signal(f5: pd.DataFrame, f15: pd.DataFrame, p: BBStratParams | None = None) -> dict | None:
    p = p or BBStratParams()
    if f5 is None or f15 is None or len(f15) < p.squeeze_lookback + 5 or len(f5) < 30:
        return None

    c15 = f15["close"]
    bb15 = bollinger(c15, p.bb_period)
    mid15, up15, lo15, w15 = bb15["bb_mid"], bb15["bb_upper"], bb15["bb_lower"], bb15["bb_width"]

    bb5 = bollinger(f5["close"], p.bb_period)
    pctb5 = float(bb5["bb_pct_b"].iloc[-1])
    price = float(f5["close"].iloc[-1])
    atr5 = float(atr(f5["high"], f5["low"], f5["close"], 14).iloc[-1])
    if not np.isfinite(atr5) or atr5 <= 0:
        return None

    # 15m squeeze + flat mid-band.
    w_thresh = w15.rolling(p.squeeze_lookback, min_periods=p.bb_period).quantile(p.squeeze_q).iloc[-1]
    squeezed = bool(np.isfinite(w_thresh) and w15.iloc[-1] <= w_thresh)
    m_now, m_prev = float(mid15.iloc[-1]), float(mid15.iloc[-1 - p.flat_bars])
    flat = bool(m_prev and abs(m_now / m_prev - 1.0) < p.flat_max_slope)

    long_ago = _mid_break_bars_ago(c15, mid15, p.break_window, "up")
    short_ago = _mid_break_bars_ago(c15, mid15, p.break_window, "down")

    side, stage, notes, break_ago = None, None, [], None

    if long_ago is not None and price > m_now:  # up-break holding above mid
        break_ago = long_ago
        notes.append(f"15m orta bant YUKARI kırıldı ({long_ago} bar önce)")
        if pctb5 <= p.retest_low:
            side, stage = "long", "GİRİŞ"
            notes.append("5m alt bant retest ✓ (giriş)")
        else:
            side, stage = "long", "BEKLE"
            notes.append(f"5m retest bekleniyor (%B {pctb5:.2f})")
    elif short_ago is not None and price < m_now:  # down-break holding below mid
        break_ago = short_ago
        notes.append(f"15m orta bant AŞAĞI kırıldı ({short_ago} bar önce)")
        if pctb5 >= p.retest_high:
            side, stage = "short", "GİRİŞ"
            notes.append("5m üst bant retest ✓ (giriş)")
        else:
            side, stage = "short", "BEKLE"
            notes.append(f"5m retest bekleniyor (%B {pctb5:.2f})")
    elif squeezed and flat:
        side, stage = "watch", "İZLE"
        notes.append("15m sıkışma + yatay (kırılım bekleniyor)")

    if stage is None:
        return None

    # Entry / SL / TP.
    entry = price
    if side == "long":
        stop = float(f5["low"].iloc[-1]) - p.sl_buffer_atr * atr5
        tp = float(up15.iloc[-1])
    elif side == "short":
        stop = float(f5["high"].iloc[-1]) + p.sl_buffer_atr * atr5
        tp = float(lo15.iloc[-1])
    else:
        stop = tp = float("nan")

    sl_pct = (stop / entry - 1.0) * 100.0 if entry else float("nan")
    tp_pct = (tp / entry - 1.0) * 100.0 if entry and np.isfinite(tp) else float("nan")

    return {
        "side": side, "stage": stage, "stage_score": _STAGE_SCORE[stage],
        "price": price, "squeeze15": squeezed, "flat15": flat, "break_ago": break_ago,
        "pctb5": pctb5, "entry": entry, "stop_loss": stop, "tp": tp,
        "sl_pct": sl_pct, "tp_pct": tp_pct, "atr5": atr5,
        "note": ", ".join(notes),
    }


class BBRetestScanner:
    def __init__(self, client, *, history_bars: int = 500, params: BBStratParams | None = None,
                 max_workers: int = 16) -> None:
        self.client = client
        self.history_bars = history_bars
        self.params = params or BBStratParams()
        self.max_workers = max_workers

    def _fetch(self, sym: str):
        try:
            b5 = self.client.get_klines(sym, "5m", limit=self.history_bars)
            frames = resample_to_timeframes(b5, ["5m", "15m"], base_timeframe="5m", drop_incomplete=True)
            return sym, frames
        except Exception as exc:
            logger.warning("Fetch failed for %s: %s", sym, exc)
            return sym, None

    def scan(
        self,
        symbols: list[str],
        *,
        top_n: int = 20,
        fundamentals: dict | None = None,
        min_quote_volume: float = 0.0,
        include_watch: bool = False,
    ) -> dict[str, pd.DataFrame]:
        fundamentals = fundamentals or {}
        if min_quote_volume > 0:
            symbols = [s for s in symbols
                       if (fundamentals.get(s, {}).get("quote_volume", float("nan")) != fundamentals.get(s, {}).get("quote_volume", float("nan")))
                       or fundamentals.get(s, {}).get("quote_volume", 0) >= min_quote_volume]

        total = len(symbols)
        frames_by: dict[str, dict] = {}
        with cf.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            for i, (sym, frames) in enumerate(ex.map(self._fetch, symbols), 1):
                if frames is not None:
                    frames_by[sym] = frames
                if i % 100 == 0:
                    logger.info("Fetched %d/%d…", i, total)

        rows = []
        for sym, frames in frames_by.items():
            sig = bb_retest_signal(frames.get("5m"), frames.get("15m"), self.params)
            if sig is None:
                continue
            if sig["stage"] == "İZLE" and not include_watch:
                continue
            fund = fundamentals.get(sym, {})
            vr = frames["5m"]["volume"].iloc[-1] / frames["5m"]["volume"].rolling(20).mean().iloc[-1] \
                if len(frames["5m"]) >= 20 else float("nan")
            rows.append({**sig, "symbol": sym, "vol_ratio": float(vr), "fund": fund})

        return {
            "long": self._build([r for r in rows if r["side"] == "long"], top_n),
            "short": self._build([r for r in rows if r["side"] == "short"], top_n),
            "watch": self._build([r for r in rows if r["side"] == "watch"], top_n),
        }

    def _build(self, items: list[dict], top_n: int) -> pd.DataFrame:
        out = []
        for r in sorted(items, key=lambda x: (x["stage_score"], -abs(x.get("tp_pct") or 0)), reverse=True)[:top_n]:
            fund = r["fund"]
            out.append({
                "symbol": r["symbol"], "stage": r["stage"], "side": r["side"].upper(),
                "price": r["price"], "break_ago": r["break_ago"], "pctb5": r["pctb5"],
                "squeeze15": r["squeeze15"], "vol_ratio": r["vol_ratio"],
                "entry": r["entry"], "stop_loss": r["stop_loss"], "tp": r["tp"],
                "sl_pct": r["sl_pct"], "tp_pct": r["tp_pct"],
                "chg24h": fund.get("chg24h", float("nan")), "funding": fund.get("funding", float("nan")),
                "ls_ratio": fund.get("ls_ratio", float("nan")), "open_interest": fund.get("open_interest", float("nan")),
                "oi_change": fund.get("oi_change", float("nan")), "note": r["note"],
            })
        return pd.DataFrame(out)
