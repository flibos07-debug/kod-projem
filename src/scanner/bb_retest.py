"""Bollinger mid-band cross strategy with a leading-indicator confidence score.

The pattern we want (as the user described it): price turns off a **top or
bottom**, goes **sideways (a Bollinger squeeze)**, and then breaks up or down —
confirmed on both the **15m** and the **1h** timeframe.

Signal construction
-------------------
1. On the **15m** chart (fallback **1h**), find a *fresh* cross of the mid-band
   (20-SMA) within the last 0-2 completed candles.
2. Require the break to come out of a **squeeze** (bandwidth in the lower part
   of its recent range) — i.e. the "yatay konuma geçti" phase.
3. Require the **1h** timeframe to agree with the direction (higher-timeframe
   bias). When the trigger itself is 1h, this is inherent.
4. Confirm on the **5m** chart: for an up-cross price must be **above** its 5m
   mid-band -> LONG; for a down-cross **below** -> SHORT.
5. Score the setup 0-100 from leading confirmations (squeeze depth, 1h bias,
   reversal-from-extreme via RSI, volume expansion, ADX/DI trend, MACD).

Anti-repaint: the currently-forming candle is dropped, so every value comes from
the last **completed** bar. The scanner runs once per closed 5m bar.

Stages: GİRİŞ (cross + 5m confirms + score >= min_confidence -> enter now),
BEKLE (cross found, still building confirmation).
"""

from __future__ import annotations

import concurrent.futures as cf
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.resample import resample_to_timeframes
from ..features.indicators import adx, atr, bollinger, ema, macd, rsi
from ..logging_utils import get_logger

logger = get_logger(__name__)

_STAGE_SCORE = {"GİRİŞ": 3, "BEKLE": 2}


@dataclass(frozen=True)
class BBStratParams:
    bb_period: int = 20
    fresh_bars: int = 2          # cross must be within the last 0..fresh_bars bars
    sl_buffer_atr: float = 0.3   # stop = 5m candle extreme -/+ this * 5m ATR
    # --- leading-indicator / precision knobs ---
    require_squeeze: bool = True     # only accept breaks out of a sideways squeeze
    require_htf: bool = True         # 1h must agree with the 15m direction
    squeeze_lookback: int = 40       # bars to rank current bandwidth against
    squeeze_pct: float = 0.5         # bandwidth must be in the lower this fraction
    vol_expansion: float = 1.2       # breakout-bar volume vs 20-bar average
    rsi_period: int = 14
    adx_period: int = 14
    min_confidence: float = 60.0     # GİRİŞ needs at least this score
    watch_confidence: float = 45.0   # below this we don't even list it as BEKLE


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


def _squeeze_rank(width: pd.Series, lookback: int, ago: int) -> float:
    """Rank the *pre-break* bandwidth within its recent range (0=tightest).

    Bandwidth expands at the moment of the break, so the "squeeze" is measured
    just before the cross (``ago`` bars back from the last bar), which is the
    "yatay konuma geçti" phase we want to require.
    """
    w = width.dropna()
    if len(w) < 6:
        return 1.0
    pre = len(w) - 2 - ago          # bar just before the break
    if pre < 1:
        pre = len(w) - 2
    lb = min(lookback, pre + 1)
    ref = w.iloc[pre - lb + 1: pre + 1]
    pre_w = float(w.iloc[pre])
    return float((ref < pre_w).mean())  # fraction of bars tighter than the pre-break bar


def _htf_bias(f1h: pd.DataFrame, p: BBStratParams) -> tuple[int, str]:
    """Higher-timeframe (1h) directional bias: +1 up, -1 down, 0 neutral."""
    if f1h is None or len(f1h) < 30:
        return 0, "1h verisi yetersiz"
    close = f1h["close"]
    e = ema(close, 50)
    price = float(close.iloc[-1])
    ref = float(e.iloc[-1])
    if not np.isfinite(ref):
        return 0, "1h EMA hazır değil"
    slope_up = e.iloc[-1] >= e.iloc[-3]
    if price > ref and slope_up:
        return 1, "1h EMA50 üstünde ve yukarı eğimli"
    if price < ref and not slope_up:
        return -1, "1h EMA50 altında ve aşağı eğimli"
    return 0, "1h yönsüz (EMA50 civarı)"


def _confidence(
    f: pd.DataFrame, side: str, p: BBStratParams, *,
    squeezed: bool, sq_rank: float, htf_dir: int, tf_name: str,
) -> tuple[float, list[str]]:
    """0-100 confidence from leading confirmations, plus human-readable notes."""
    score = 0.0
    notes: list[str] = []
    up = side == "long"

    # 1) Squeeze / sideways phase (the "yatay konuma geçti" requirement).
    if squeezed:
        score += 20
        notes.append(f"sıkışmadan çıkış (bant genişliği alt %{int(sq_rank * 100)})")

    # 2) 1h higher-timeframe agreement.
    want = 1 if up else -1
    if tf_name == "1h":
        score += 20  # trigger is already the 1h itself
    elif htf_dir == want:
        score += 20
        notes.append("1h aynı yönde ✓")
    elif htf_dir == 0:
        score += 8
        notes.append("1h yönsüz")

    close = f["close"]
    # 3) Reversal from an extreme (dip/tepe dönüşü) via RSI.
    r = rsi(close, p.rsi_period)
    if len(r.dropna()) >= 7:
        recent = r.iloc[-6:]
        cur = float(r.iloc[-1])
        if up and float(recent.min()) < 40 and cur > float(recent.min()):
            score += 15
            notes.append("RSI dipten dönüyor")
        elif (not up) and float(recent.max()) > 60 and cur < float(recent.max()):
            score += 15
            notes.append("RSI tepeden dönüyor")

    # 4) Volume expansion on the breakout bar.
    vol = f["volume"]
    if len(vol) >= 20:
        avg = float(vol.rolling(20).mean().iloc[-1])
        if avg > 0 and float(vol.iloc[-1]) >= p.vol_expansion * avg:
            score += 15
            notes.append(f"hacim patlaması ({vol.iloc[-1] / avg:.1f}x)")

    # 5) ADX / DI — a trend is building in the right direction.
    a = adx(f["high"], f["low"], close, p.adx_period)
    if len(a.dropna()) >= 3:
        adx_v = float(a["adx"].iloc[-1])
        di_ok = (a["plus_di"].iloc[-1] > a["minus_di"].iloc[-1]) if up else (a["minus_di"].iloc[-1] > a["plus_di"].iloc[-1])
        rising = a["adx"].iloc[-1] >= a["adx"].iloc[-3]
        if di_ok and rising and adx_v >= 15:
            score += 15
            notes.append(f"ADX güçleniyor ({adx_v:.0f})")

    # 6) MACD histogram agrees and is expanding.
    m = macd(close)
    if len(m.dropna()) >= 2:
        hist = m["hist"]
        if up and hist.iloc[-1] > 0 and hist.iloc[-1] >= hist.iloc[-2]:
            score += 15
            notes.append("MACD pozitif/güçleniyor")
        elif (not up) and hist.iloc[-1] < 0 and hist.iloc[-1] <= hist.iloc[-2]:
            score += 15
            notes.append("MACD negatif/güçleniyor")

    return min(score, 100.0), notes


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

    htf_dir, htf_note = _htf_bias(f1h, p)

    window = p.fresh_bars + 1  # 0..fresh_bars bars ago
    need = p.bb_period + p.adx_period + window + 6

    for tf_name, f in (("15m", f15), ("1h", f1h)):
        if f is None or len(f) < need:
            continue
        bb = bollinger(f["close"], p.bb_period)
        mid, up, lo, width = bb["bb_mid"], bb["bb_upper"], bb["bb_lower"], bb["bb_width"]

        for side, ago, dirn in (
            ("long", _mid_break_bars_ago(f["close"], mid, window, "up"), 1),
            ("short", _mid_break_bars_ago(f["close"], mid, window, "down"), -1),
        ):
            if ago is None:
                continue
            sq_rank = _squeeze_rank(width, p.squeeze_lookback, ago)
            squeezed = sq_rank <= p.squeeze_pct
            # Hard filter 1: must break out of a sideways squeeze.
            if p.require_squeeze and not squeezed:
                continue
            # Hard filter 2: 1h must not oppose (only when trigger is the 15m).
            if p.require_htf and tf_name == "15m" and htf_dir != 0 and htf_dir != dirn:
                continue

            conf, notes = _confidence(
                f, side, p, squeezed=squeezed, sq_rank=sq_rank, htf_dir=htf_dir, tf_name=tf_name,
            )
            confirmed5 = above5 if side == "long" else (not above5)
            stage = "GİRİŞ" if (confirmed5 and conf >= p.min_confidence) else "BEKLE"
            if stage == "BEKLE" and conf < p.watch_confidence:
                continue  # too weak even to watch

            cross_txt = "YUKARI" if side == "long" else "AŞAĞI"
            head = [f"{tf_name} orta bant {cross_txt} kesişim ({ago} bar önce)"]
            head.append("5m onay ✓" if confirmed5 else "5m onay bekleniyor")
            if tf_name == "15m":
                head.append(htf_note)
            all_notes = head + notes

            if side == "long":
                stop = float(f5["low"].iloc[-1]) - p.sl_buffer_atr * atr5
                tp = float(up.iloc[-1])
            else:
                stop = float(f5["high"].iloc[-1]) + p.sl_buffer_atr * atr5
                tp = float(lo.iloc[-1])
            return _row(side, stage, tf_name, ago, price, mid5, stop, tp, conf, all_notes)

    return None


def _row(side, stage, tf, break_ago, price, mid5, stop, tp, conf, notes) -> dict:
    entry = price
    sl_pct = (stop / entry - 1.0) * 100.0 if entry else float("nan")
    tp_pct = (tp / entry - 1.0) * 100.0 if entry and np.isfinite(tp) else float("nan")
    entry_dist = 0.0 if stage == "GİRİŞ" else (abs(price - mid5) / mid5 if mid5 else 1.0)
    return {
        "side": side, "stage": stage, "stage_score": _STAGE_SCORE[stage], "tf": tf,
        "break_ago": break_ago, "price": price, "entry": entry, "stop_loss": stop, "tp": tp,
        "sl_pct": sl_pct, "tp_pct": tp_pct, "entry_dist": entry_dist, "confidence": conf,
        "note": ", ".join(notes),
    }


class BBRetestScanner:
    def __init__(self, client, *, history_bars: int = 800, params: BBStratParams | None = None,
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

        ordered = sorted(
            items,
            key=lambda x: (-x["stage_score"], -x.get("confidence", 0.0), x.get("entry_dist", 1.0), -_vol(x)),
        )
        out = []
        for r in ordered[:top_n]:
            fund = r["fund"]
            out.append({
                "symbol": r["symbol"], "stage": r["stage"], "side": r["side"].upper(), "tf": r["tf"],
                "price": r["price"], "break_ago": r["break_ago"], "vol_ratio": r["vol_ratio"],
                "confidence": r["confidence"],
                "entry": r["entry"], "stop_loss": r["stop_loss"], "tp": r["tp"],
                "sl_pct": r["sl_pct"], "tp_pct": r["tp_pct"],
                "chg24h": fund.get("chg24h", float("nan")), "funding": fund.get("funding", float("nan")),
                "ls_ratio": fund.get("ls_ratio", float("nan")), "open_interest": fund.get("open_interest", float("nan")),
                "oi_change": fund.get("oi_change", float("nan")), "note": r["note"],
            })
        return pd.DataFrame(out)
