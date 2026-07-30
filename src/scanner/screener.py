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
from ..features.indicators import (
    adx,
    atr,
    bollinger,
    ema,
    macd,
    rolling_vwap,
    rsi,
    stoch_rsi,
    supertrend,
)
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
    squeeze_lookback: int = 30  # keep < the ~41 1h bars a 500-bar 5m fetch yields
    squeeze_q: float = 0.15     # bandwidth in the lowest 15% -> squeeze
    vol_spike: float = 2.0
    fresh_bars: int = 3          # Supertrend flip within this many bars = "early"
    max_ext_pct: float = 5.0     # price >5% from 21-EMA = already extended (chaser)
    tp_mult: float = 2.0
    sl_mult: float = 1.0
    timeframes: tuple[str, ...] = TIMEFRAMES


def _sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _tf_snapshot(df: pd.DataFrame, p: ScreenerParams, *, want_ema200: bool = False, want_bias: bool = False) -> dict | None:
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

    # Heavy "big-picture" indicators are computed only where they are used:
    # EMA200 on the 5m frame (enough bars), Supertrend/VWAP/ADX on the 1h frame.
    ema200_pos = "na"
    if want_ema200:
        e200 = float(ema(close, 200).iloc[-1])
        ema200_pos = "na" if e200 != e200 else ("up" if c > e200 else "down")

    st_dir, st_flip, vwap_pos, adx_v = 0, False, "na", float("nan")
    bars_since_flip = 999
    if want_bias:
        st = supertrend(high, low, close)
        dir_arr = st["direction"].to_numpy()
        st_dir = int(dir_arr[-1])
        st_flip = len(dir_arr) > 1 and st_dir != int(dir_arr[-2])
        # How many bars ago did the trend flip? (0 = flipped this bar = earliest.)
        bars_since_flip = 0
        for j in range(len(dir_arr) - 2, -1, -1):
            if dir_arr[j] == st_dir:
                bars_since_flip += 1
            else:
                break
        vw = float(rolling_vwap(df, 24).iloc[-1])
        vwap_pos = "na" if vw != vw else ("up" if c > vw else "down")
        adx_v = float(adx(high, low, close, 14)["adx"].iloc[-1])

    # Extension: how far price has run from its 21-EMA (in %). Small = early.
    ext_pct = (c / ema_s - 1.0) * 100.0 if ema_s and np.isfinite(ema_s) else float("nan")

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
        "ema200_pos": ema200_pos, "st_dir": st_dir, "st_flip": st_flip,
        "vwap_pos": vwap_pos, "adx": adx_v, "ext_pct": ext_pct,
        "bars_since_flip": bars_since_flip, "bull": bull, "bear": bear,
    }


def screen_symbol(frames: dict[str, pd.DataFrame], p: ScreenerParams | None = None) -> dict | None:
    p = p or ScreenerParams()
    snaps = {}
    for tf in p.timeframes:
        s = _tf_snapshot(frames.get(tf), p, want_ema200=(tf == "5m"), want_bias=(tf == "1h"))
        if s is None:
            return None
        snaps[tf] = s

    ref = snaps.get("1h", snaps[p.timeframes[-1]])  # higher-timeframe = the bias
    # EMA200 comes from the 5m frame, which has enough bars for it (the 1h frame
    # resampled from 500 5m bars is too short for a 200-EMA).
    ema200_pos = snaps.get("5m", ref)["ema200_pos"]

    # Directional score: higher-TF bias (EMA200, Supertrend, VWAP, MACD) is the
    # backbone; lower-timeframe EMA trend refines it; reversal signals catch turns.
    bull = bear = 0.0
    if ema200_pos == "up":
        bull += 2.0
    elif ema200_pos == "down":
        bear += 2.0
    if ref["st_dir"] == 1:
        bull += 2.0
    elif ref["st_dir"] == -1:
        bear += 2.0
    if ref["vwap_pos"] == "up":
        bull += 1.0
    elif ref["vwap_pos"] == "down":
        bear += 1.0
    if ref["macd_dir"] == "up":
        bull += 1.0
    else:
        bear += 1.0
    for tf in ("5m", "15m"):
        if tf in snaps:
            if snaps[tf]["trend"] == "up":
                bull += 0.5
            elif snaps[tf]["trend"] == "down":
                bear += 0.5

    # Reversal ("dönüş") signals catch turns against the higher-TF bias.
    rev_up = any(snaps[tf]["stoch_turn_up"] for tf in p.timeframes) or ref["pctb"] < 0.1
    rev_dn = any(snaps[tf]["stoch_turn_dn"] for tf in p.timeframes) or ref["pctb"] > 0.9
    if rev_up:
        bull += 1.0
    if rev_dn:
        bear += 1.0

    squeeze_any = any(snaps[tf]["squeeze"] for tf in p.timeframes)

    # Early vs extended: a fresh Supertrend flip means the move just started
    # (what we want); a large distance from the 21-EMA means it already ran.
    ext = ref.get("ext_pct", float("nan"))
    bsf = ref.get("bars_since_flip", 999)
    fresh_flip = bsf <= p.fresh_bars
    if fresh_flip:
        if ref["st_dir"] == 1:
            bull += 1.0
        elif ref["st_dir"] == -1:
            bear += 1.0

    extended = bool(np.isfinite(ext) and abs(ext) >= p.max_ext_pct)
    if extended:  # penalise chasers so early setups rank higher
        bull *= 0.55
        bear *= 0.55

    net = bull - bear
    side = "long" if net >= 0 else "short"
    long_score = max(bull - bear, 0.0) / 9.0
    short_score = max(bear - bull, 0.0) / 9.0
    long_score, short_score = min(long_score, 1.0), min(short_score, 1.0)

    if fresh_flip or squeeze_any:
        state = "BAŞLANGIÇ"
    elif extended:
        state = "UZAMIŞ"
    else:
        state = "ORTA"

    # Setup type: squeeze -> breakout; strong ADX -> trend; else a reversal.
    if squeeze_any:
        setup = "KIRILIM"
    elif ref["adx"] >= 25:
        setup = "TREND"
    elif (side == "long" and rev_up) or (side == "short" and rev_dn):
        setup = "DÖNÜŞ"
    else:
        setup = "TREND" if ref["adx"] >= 20 else "NÖTR"

    # Leading ("öncü") triggers.
    leading: list[str] = []
    if squeeze_any:
        leading.append("SIKIŞMA (kırılım yakın)")
    vmax = max((snaps[tf]["vol_ratio"] for tf in p.timeframes if snaps[tf]["vol_ratio"] == snaps[tf]["vol_ratio"]), default=float("nan"))
    if vmax == vmax and vmax >= p.vol_spike:
        leading.append(f"HACİM x{vmax:.1f}")
    if fresh_flip:
        leading.append(f"YENİ dönüş ({bsf} bar önce)")
    if side == "long" and rev_up:
        leading.append("dipten dönüş")
    if side == "short" and rev_dn:
        leading.append("tepeden dönüş")
    if any(snaps[tf]["macd_flip"] for tf in ("15m", "1h") if tf in snaps):
        leading.append("MACD dönüş")
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
        "squeeze": squeeze_any,
        "vol_ratio": vmax,
        "setup": setup,
        "state": state,
        "ext_pct": ext,
        "bars_since_flip": bsf,
        "ema200_1h": ema200_pos,
        "supertrend_1h": "up" if ref["st_dir"] == 1 else "down",
        "vwap_1h": ref["vwap_pos"],
        "adx_1h": ref["adx"],
        "leading": leading,
        "atr_abs": ref["atr_pct"] * ref["price"] if ref["atr_pct"] == ref["atr_pct"] else float("nan"),
    }


class Screener:
    def __init__(self, client, *, base_timeframe: str = "5m", history_bars: int = 500,
                 params: ScreenerParams | None = None, max_workers: int = 16) -> None:
        self.client = client
        self.base_timeframe = base_timeframe
        self.history_bars = history_bars
        self.params = params or ScreenerParams()
        self.max_workers = max_workers

    def _fetch(self, sym: str):
        # ONE request per coin (low weight -> fast, no rate-limit stalls): pull
        # 5m and resample to 15m/1h. EMA200 is taken from the 5m frame (500 bars
        # ~ 17h), which is a solid intraday major-trend filter.
        try:
            b5 = self.client.get_klines(sym, "5m", limit=self.history_bars)
            frames = resample_to_timeframes(b5, ["5m", "15m", "1h"], base_timeframe="5m", drop_incomplete=True)
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
        only_early: bool = False,
        max_ext_pct: float | None = None,
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
            if only_early and row["state"] == "UZAMIŞ":
                continue          # hide coins that already ran
            if max_ext_pct is not None and row["ext_pct"] == row["ext_pct"] and abs(row["ext_pct"]) > max_ext_pct:
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
                "setup": r["setup"], "state": r["state"], "ext_pct": r["ext_pct"],
                "ema200": r["ema200_1h"], "supertrend": r["supertrend_1h"],
                "vwap": r["vwap_1h"], "adx": r["adx_1h"],
                "funding": fund.get("funding", float("nan")), "ls_ratio": fund.get("ls_ratio", float("nan")),
                "open_interest": fund.get("open_interest", float("nan")),
                "oi_change": fund.get("oi_change", float("nan")),
                "leading": ", ".join(r["leading"]) or "-",
                "entry_low": lv.entry_low if lv else float("nan"),
                "entry_high": lv.entry_high if lv else float("nan"),
                "stop_loss": lv.stop_loss if lv else float("nan"),
                "tp1": lv.tp1 if lv else float("nan"), "tp2": lv.tp2 if lv else float("nan"),
                "sl_pct": lv.sl_pct if lv else float("nan"),
                "tp1_pct": lv.tp1_pct if lv else float("nan"), "tp2_pct": lv.tp2_pct if lv else float("nan"),
            })
        return pd.DataFrame(out)
