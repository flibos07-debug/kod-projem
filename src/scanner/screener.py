"""Multi-timeframe day-trader screener (no ML, no training).

For every symbol it fetches 5m klines once, resamples to 15m and 1h, and reads
the indicators a scalper/day-trader actually watches on each timeframe: EMA
trend, RSI, MACD, Stochastic RSI, Bollinger position/squeeze, volume and ATR.
It combines them (weighting the higher timeframe more) into a LONG or SHORT
lean and surfaces **leading** ("öncü") triggers — Bollinger squeeze, volume
spike, StochRSI turn, MACD flip — that tend to precede a move.

It offers no opinion beyond a compact tilt score and the raw readout: the trader
makes the entry/exit call. Funding / open interest / long-short ratio are added
as context.
"""

from __future__ import annotations

import concurrent.futures as cf
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.resample import resample_to_timeframes
from ..features.indicators import atr, bollinger, ema, macd, rsi, stoch_rsi
from ..logging_utils import get_logger
from .levels import compute_levels

logger = get_logger(__name__)

TIMEFRAMES = ("5m", "15m", "1h")
_TF_WEIGHT = {"5m": 1.0, "15m": 1.5, "1h": 2.0}


@dataclass(frozen=True)
class ScreenerParams:
    rsi_period: int = 14
    ema_fast: int = 9
    ema_slow: int = 21
    bb_period: int = 20
    vol_ma: int = 20
    squeeze_lookback: int = 50
    squeeze_q: float = 0.15     # bandwidth in the lowest 15% -> squeeze
    vol_spike: float = 2.0
    tp_mult: float = 2.0
    sl_mult: float = 1.0
    timeframes: tuple[str, ...] = TIMEFRAMES


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _tf_snapshot(df: pd.DataFrame, p: ScreenerParams) -> dict | None:
    if df is None or len(df) < max(p.squeeze_lookback, p.bb_period) + 5:
        return None
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]
    c = float(close.iloc[-1])

    rsi_v = float(rsi(close, p.rsi_period).iloc[-1])
    ema_f = float(ema(close, p.ema_fast).iloc[-1])
    ema_s = float(ema(close, p.ema_slow).iloc[-1])
    if c > ema_s and ema_f >= ema_s:
        trend = "up"
    elif c < ema_s and ema_f <= ema_s:
        trend = "down"
    else:
        trend = "flat"

    m = macd(close)
    hist = m["hist"]
    macd_dir = "up" if hist.iloc[-1] > 0 else "down"
    macd_flip = _sign(hist.iloc[-1]) != _sign(hist.iloc[-2])

    sr = stoch_rsi(close)
    k, d = float(sr["k"].iloc[-1]), float(sr["d"].iloc[-1])
    kp = float(sr["k"].iloc[-2]) if len(sr) > 1 else k
    stoch_turn_up = kp < 20 and k >= kp and k > d
    stoch_turn_dn = kp > 80 and k <= kp and k < d

    bb = bollinger(close, p.bb_period)
    pctb = float(bb["bb_pct_b"].iloc[-1])
    width = bb["bb_width"]
    thresh = width.rolling(p.squeeze_lookback, min_periods=p.bb_period).quantile(p.squeeze_q).iloc[-1]
    squeeze = bool(np.isfinite(thresh) and width.iloc[-1] <= thresh)

    vol_ma = vol.rolling(p.vol_ma, min_periods=p.vol_ma).mean().iloc[-1]
    vol_ratio = float(vol.iloc[-1] / vol_ma) if np.isfinite(vol_ma) and vol_ma > 0 else float("nan")
    atr_pct = float(atr(high, low, close, 14).iloc[-1] / c) if c else float("nan")

    # Trend is the primary directional vote; momentum/mean-reversion refine it.
    # RSI is reported for the trader but not scored directly, so an uptrend coin
    # is not pushed short merely for being overbought.
    bull = bear = 0.0
    if trend == "up":
        bull += 1.5
    elif trend == "down":
        bear += 1.5
    bull += 0.5 if macd_dir == "up" else 0.0
    bear += 0.5 if macd_dir == "down" else 0.0
    if k > d and k < 80:
        bull += 0.5        # stoch turning up (leading)
    if k < d and k > 20:
        bear += 0.5
    if pctb < 0.15 or stoch_turn_up:
        bull += 0.5        # oversold / turning up at band (reversal long)
    elif pctb > 0.85 or stoch_turn_dn:
        bear += 0.5

    return {
        "price": c, "rsi": rsi_v, "trend": trend, "macd_dir": macd_dir, "macd_flip": macd_flip,
        "stoch_k": k, "stoch_d": d, "stoch_turn_up": stoch_turn_up, "stoch_turn_dn": stoch_turn_dn,
        "pctb": pctb, "squeeze": squeeze, "vol_ratio": vol_ratio, "atr_pct": atr_pct,
        "bull": bull, "bear": bear,
    }


def screen_symbol(frames: dict[str, pd.DataFrame], p: ScreenerParams | None = None) -> dict | None:
    p = p or ScreenerParams()
    snaps = {}
    for tf in p.timeframes:
        s = _tf_snapshot(frames.get(tf), p)
        if s is None:
            return None
        snaps[tf] = s

    wsum = sum(_TF_WEIGHT.get(tf, 1.0) for tf in p.timeframes)
    max_per_tf = 3.0
    long_score = sum(_TF_WEIGHT.get(tf, 1.0) * snaps[tf]["bull"] for tf in p.timeframes) / (wsum * max_per_tf)
    short_score = sum(_TF_WEIGHT.get(tf, 1.0) * snaps[tf]["bear"] for tf in p.timeframes) / (wsum * max_per_tf)

    side = "long" if long_score >= short_score else "short"

    # Leading ("öncü") triggers, scanned across timeframes.
    leading: list[str] = []
    if any(snaps[tf]["squeeze"] for tf in p.timeframes):
        leading.append("SIKIŞMA (kırılım yakın)")
    vmax = max((snaps[tf]["vol_ratio"] for tf in p.timeframes if snaps[tf]["vol_ratio"] == snaps[tf]["vol_ratio"]), default=float("nan"))
    if vmax == vmax and vmax >= p.vol_spike:
        leading.append(f"HACİM x{vmax:.1f}")
    if side == "long" and any(snaps[tf]["stoch_turn_up"] for tf in p.timeframes):
        leading.append("StochRSI dipten dönüş")
    if side == "short" and any(snaps[tf]["stoch_turn_dn"] for tf in p.timeframes):
        leading.append("StochRSI tepeden dönüş")
    if any(snaps[tf]["macd_flip"] for tf in ("15m", "1h") if tf in snaps):
        leading.append("MACD dönüş")

    ref = snaps.get("1h", snaps[p.timeframes[-1]])
    return {
        "side": side,
        "long_score": long_score,
        "short_score": short_score,
        "score": max(long_score, short_score),
        "price": ref["price"],
        "atr_pct_1h": ref["atr_pct"],
        "rsi_5m": snaps["5m"]["rsi"] if "5m" in snaps else float("nan"),
        "rsi_15m": snaps["15m"]["rsi"] if "15m" in snaps else float("nan"),
        "rsi_1h": snaps["1h"]["rsi"] if "1h" in snaps else float("nan"),
        "trend_5m": snaps["5m"]["trend"] if "5m" in snaps else "-",
        "trend_15m": snaps["15m"]["trend"] if "15m" in snaps else "-",
        "trend_1h": snaps["1h"]["trend"] if "1h" in snaps else "-",
        "macd_1h": ref["macd_dir"],
        "stoch_1h": ref["stoch_k"],
        "bb_1h": ref["pctb"],
        "squeeze": any(snaps[tf]["squeeze"] for tf in p.timeframes),
        "vol_ratio": vmax,
        "leading": leading,
        "atr_abs": ref["atr_pct"] * ref["price"] if ref["atr_pct"] == ref["atr_pct"] else float("nan"),
    }


class Screener:
    def __init__(self, client, *, base_timeframe: str = "5m", history_bars: int = 1000,
                 params: ScreenerParams | None = None, max_workers: int = 16) -> None:
        self.client = client
        self.base_timeframe = base_timeframe
        self.history_bars = history_bars
        self.params = params or ScreenerParams()
        self.max_workers = max_workers

    def _fetch(self, sym: str):
        try:
            base = self.client.get_klines(sym, self.base_timeframe, limit=self.history_bars)
            frames = resample_to_timeframes(base, list(self.params.timeframes),
                                            base_timeframe=self.base_timeframe, drop_incomplete=True)
            return sym, frames
        except Exception as exc:
            logger.warning("Fetch failed for %s: %s", sym, exc)
            return sym, None

    def scan(
        self,
        symbols: list[str],
        *,
        top_n: int = 15,
        fundamentals: dict | None = None,
        min_quote_volume: float = 0.0,
        min_score: float = 0.0,
        rsi_below: float | None = None,
        rsi_above: float | None = None,
        require_squeeze: bool = False,
        require_vol_spike: bool = False,
    ) -> dict[str, pd.DataFrame]:
        fundamentals = fundamentals or {}
        if min_quote_volume > 0:
            symbols = [s for s in symbols
                       if (fundamentals.get(s, {}).get("quote_volume", float("nan")) != fundamentals.get(s, {}).get("quote_volume", float("nan")))
                       or fundamentals.get(s, {}).get("quote_volume", 0) >= min_quote_volume]

        total = len(symbols)
        results: dict[str, dict] = {}
        with cf.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            for i, (sym, frames) in enumerate(ex.map(self._fetch, symbols), 1):
                if frames is not None:
                    results[sym] = frames
                if i % 100 == 0:
                    logger.info("Fetched %d/%d…", i, total)

        rows = []
        for sym, frames in results.items():
            row = screen_symbol(frames, self.params)
            if row is None or row["score"] < min_score:
                continue
            # Trader filters.
            focus_rsi = row["rsi_1h"]
            if rsi_below is not None and not (focus_rsi < rsi_below):
                continue
            if rsi_above is not None and not (focus_rsi > rsi_above):
                continue
            if require_squeeze and not row["squeeze"]:
                continue
            if require_vol_spike and not (row["vol_ratio"] == row["vol_ratio"] and row["vol_ratio"] >= self.params.vol_spike):
                continue
            row["symbol"] = sym
            row["fund"] = fundamentals.get(sym, {})
            rows.append(row)

        longs = [r for r in rows if r["side"] == "long"]
        shorts = [r for r in rows if r["side"] == "short"]
        return {
            "long": self._build(longs, "long", top_n),
            "short": self._build(shorts, "short", top_n),
        }

    def _build(self, items: list[dict], side: str, top_n: int) -> pd.DataFrame:
        p = self.params
        out = []
        for r in sorted(items, key=lambda x: x["score"], reverse=True)[:top_n]:
            atr_abs = r["atr_abs"]
            lv = compute_levels(r["price"], atr_abs, side, tp_mult=p.tp_mult, sl_mult=p.sl_mult) if atr_abs == atr_abs else None
            fund = r["fund"]
            out.append({
                "symbol": r["symbol"], "side": side.upper(), "score": round(r["score"], 3),
                "price": r["price"], "chg24h": fund.get("chg24h", float("nan")),
                "rsi_5m": r["rsi_5m"], "rsi_15m": r["rsi_15m"], "rsi_1h": r["rsi_1h"],
                "trend_5m": r["trend_5m"], "trend_15m": r["trend_15m"], "trend_1h": r["trend_1h"],
                "macd_1h": r["macd_1h"], "stoch_1h": r["stoch_1h"], "bb_1h": r["bb_1h"],
                "squeeze": r["squeeze"], "vol_ratio": r["vol_ratio"], "atr_pct": r["atr_pct_1h"],
                "funding": fund.get("funding", float("nan")), "ls_ratio": fund.get("ls_ratio", float("nan")),
                "open_interest": fund.get("open_interest", float("nan")),
                "leading": ", ".join(r["leading"]) or "-",
                "entry_low": lv.entry_low if lv else float("nan"),
                "entry_high": lv.entry_high if lv else float("nan"),
                "stop_loss": lv.stop_loss if lv else float("nan"),
                "tp1": lv.tp1 if lv else float("nan"), "tp2": lv.tp2 if lv else float("nan"),
                "sl_pct": lv.sl_pct if lv else float("nan"),
                "tp1_pct": lv.tp1_pct if lv else float("nan"), "tp2_pct": lv.tp2_pct if lv else float("nan"),
            })
        return pd.DataFrame(out)
