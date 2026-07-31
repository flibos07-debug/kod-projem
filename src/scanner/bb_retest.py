"""Bollinger mid-band cross entry strategy (every-bar-close, no repaint).

Rules (as specified by the user):

- On the **15m** chart, find a fresh cross of the mid-band (20-SMA) that
  happened within the last 0-2 completed candles. If none on 15m, look on the
  **1h** chart.
- Confirm on the **5m** chart: for an up-cross, price must be **above** its 5m
  mid-band -> LONG; for a down-cross, price **below** its 5m mid-band -> SHORT.
- Stop: just beyond the last completed 5m candle's low/high.
- Target: the opposite outer band on the trigger timeframe (long -> upper,
  short -> lower).

Anti-repaint: the currently-forming candle is dropped, so every value comes from
the last **completed** bar. The scanner is meant to run once per closed 5m bar.

Stages: GİRİŞ (cross + 5m confirms -> enter now) and BEKLE (cross found, 5m not
yet confirming).
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

_STAGE_SCORE = {"GİRİŞ": 3, "BEKLE": 2}


@dataclass(frozen=True)
class BBStratParams:
    bb_period: int = 20
    fresh_bars: int = 2         # cross must be within the last 0..fresh_bars bars
    sl_buffer_atr: float = 0.3  # stop = 5m candle extreme -/+ this * 5m ATR


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


def bb_retest_signal(
    f5: pd.DataFrame, f15: pd.DataFrame, f1h: pd.DataFrame, p: BBStratParams | None = None
) -> dict | None:
    p = p or BBStratParams()
    if f5 is None or len(f5) < p.bb_period + 3:
        return None

    bb5 = bollinger(f5["close"], p.bb_period)
    mid5 = float(bb5["bb_mid"].iloc[-1])
    price = float(f5["close"].iloc[-1])
    if not np.isfinite(mid5) or not np.isfinite(price):
        return None
    atr5 = float(atr(f5["high"], f5["low"], f5["close"], 14).iloc[-1])
    if not np.isfinite(atr5) or atr5 <= 0:
        return None
    above5 = price > mid5

    window = p.fresh_bars + 1  # 0..fresh_bars bars ago
    need = p.bb_period + window + 2

    for tf_name, f in (("15m", f15), ("1h", f1h)):
        if f is None or len(f) < need:
            continue
        bb = bollinger(f["close"], p.bb_period)
        mid, up, lo = bb["bb_mid"], bb["bb_upper"], bb["bb_lower"]
        up_ago = _mid_break_bars_ago(f["close"], mid, window, "up")
        dn_ago = _mid_break_bars_ago(f["close"], mid, window, "down")

        if up_ago is not None:
            side, break_ago, tf = "long", up_ago, tf_name
            confirmed = above5
            stage = "GİRİŞ" if confirmed else "BEKLE"
            notes = [f"{tf} orta bant YUKARI kesişim ({break_ago} bar önce)",
                     "5m orta bant ÜSTÜNDE ✓ (giriş)" if confirmed else "5m onay bekleniyor (fiyat 5m orta bandın altında)"]
            stop = float(f5["low"].iloc[-1]) - p.sl_buffer_atr * atr5
            tp = float(up.iloc[-1])
            return _row(side, stage, tf, break_ago, price, mid5, stop, tp, above5, notes, p)

        if dn_ago is not None:
            side, break_ago, tf = "short", dn_ago, tf_name
            confirmed = not above5
            stage = "GİRİŞ" if confirmed else "BEKLE"
            notes = [f"{tf} orta bant AŞAĞI kesişim ({break_ago} bar önce)",
                     "5m orta bant ALTINDA ✓ (giriş)" if confirmed else "5m onay bekleniyor (fiyat 5m orta bandın üstünde)"]
            stop = float(f5["high"].iloc[-1]) + p.sl_buffer_atr * atr5
            tp = float(lo.iloc[-1])
            return _row(side, stage, tf, break_ago, price, mid5, stop, tp, above5, notes, p)

    return None


def _row(side, stage, tf, break_ago, price, mid5, stop, tp, above5, notes, p) -> dict:
    entry = price
    sl_pct = (stop / entry - 1.0) * 100.0 if entry else float("nan")
    tp_pct = (tp / entry - 1.0) * 100.0 if entry and np.isfinite(tp) else float("nan")
    # Distance to a GİRİŞ: 0 if 5m already confirms, else how far 5m price is
    # from its mid-band (about to cross = smaller).
    entry_dist = 0.0 if stage == "GİRİŞ" else (abs(price - mid5) / mid5 if mid5 else 1.0)
    return {
        "side": side, "stage": stage, "stage_score": _STAGE_SCORE[stage], "tf": tf,
        "break_ago": break_ago, "price": price, "entry": entry, "stop_loss": stop, "tp": tp,
        "sl_pct": sl_pct, "tp_pct": tp_pct, "entry_dist": entry_dist, "note": ", ".join(notes),
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
            if len(b5) > 1:
                b5 = b5.iloc[:-1]  # drop the currently-forming candle (anti-repaint)
            frames = resample_to_timeframes(b5, ["5m", "15m", "1h"], base_timeframe="5m", drop_incomplete=True)
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
            sig = bb_retest_signal(frames.get("5m"), frames.get("15m"), frames.get("1h"), self.params)
            if sig is None:
                continue
            if sig["stage"] == "BEKLE" and not include_watch:
                continue  # default: only actionable GİRİŞ
            f5 = frames["5m"]
            vr = f5["volume"].iloc[-1] / f5["volume"].rolling(20).mean().iloc[-1] if len(f5) >= 20 else float("nan")
            rows.append({**sig, "symbol": sym, "vol_ratio": float(vr), "fund": fundamentals.get(sym, {})})

        return {
            "long": self._build([r for r in rows if r["side"] == "long"], top_n),
            "short": self._build([r for r in rows if r["side"] == "short"], top_n),
        }

    def _build(self, items: list[dict], top_n: int) -> pd.DataFrame:
        def _vol(x):
            v = x.get("vol_ratio", 0.0)
            return v if v == v else 0.0

        ordered = sorted(items, key=lambda x: (-x["stage_score"], x.get("entry_dist", 1.0), -_vol(x)))
        out = []
        for r in ordered[:top_n]:
            fund = r["fund"]
            out.append({
                "symbol": r["symbol"], "stage": r["stage"], "side": r["side"].upper(), "tf": r["tf"],
                "price": r["price"], "break_ago": r["break_ago"], "vol_ratio": r["vol_ratio"],
                "entry": r["entry"], "stop_loss": r["stop_loss"], "tp": r["tp"],
                "sl_pct": r["sl_pct"], "tp_pct": r["tp_pct"],
                "chg24h": fund.get("chg24h", float("nan")), "funding": fund.get("funding", float("nan")),
                "ls_ratio": fund.get("ls_ratio", float("nan")), "open_interest": fund.get("open_interest", float("nan")),
                "oi_change": fund.get("oi_change", float("nan")), "note": r["note"],
            })
        return pd.DataFrame(out)
