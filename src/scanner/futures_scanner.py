"""Two-sided (long/short) scanner for Binance Futures.

For each symbol it scores both the long and short models, wraps each calibrated
probability in a conformal set, and turns the pair into a discrete signal:

- **LONG** when the long set is a confident ``{1}`` (and beats the short side);
- **SHORT** when the short set is a confident ``{1}`` (and beats the long side);
- **FLAT** otherwise (conformal abstains — the deliberately conservative state).

Confident signals are ranked cross-sectionally within each direction and the top
``max_positions_per_direction`` are marked as actionable, so at most N longs and
N shorts are selected per scan.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Callable

import numpy as np
import pandas as pd

from ..config_schema import AppConfig
from ..data.resample import resample_to_timeframes
from ..features.engineering import build_feature_matrix
from ..logging_utils import get_logger
from ..pipeline.directional import DirectionalArtifact, SideModel
from ..regime.classifier import RegimeClassifier
from ..reporting.report import cleanup_old_reports, render_table, save_scan
from ..utils.timing import INTERVAL_MINUTES, next_boundary, seconds_until
from .levels import compute_levels
from .live_scanner import _set_repr

# Feature warm-up budget, expressed in bars of *any* timeframe. The scanner must
# fetch enough base bars that even the highest timeframe has this many complete
# bars for its indicators to warm up.
_WARMUP_BARS = 120

_SIGNAL_COLUMNS = [
    "symbol", "side", "prob", "confident", "regime", "price",
    "quote_volume", "funding",
    "entry_low", "entry_high", "stop_loss", "tp1", "tp2",
    "sl_pct", "tp1_pct", "tp2_pct",
]


def _empty_signal_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_SIGNAL_COLUMNS)

logger = get_logger(__name__)


class FuturesScanner:
    def __init__(
        self,
        config: AppConfig,
        artifacts: dict[str, DirectionalArtifact],
        client,
        *,
        history_bars: int = 1500,
        vol_window: int = 500,
    ) -> None:
        if not artifacts:
            raise ValueError("at least one directional artifact is required")
        self.config = config
        self.artifacts = artifacts
        self.client = client
        self.history_bars = history_bars
        self.vol_window = vol_window

        meta = next(iter(artifacts.values())).metadata
        self.base_timeframe = meta.get("base_timeframe", "5m")
        self.htf_timeframes = list(meta.get("htf_timeframes", ["15m", "1h", "4h"]))
        self._regime_clf = RegimeClassifier(config.regime, vol_window=vol_window)
        # Enough base bars so the highest timeframe also gets _WARMUP_BARS
        # complete bars (single get_klines caps at 1000, which is too few for a
        # 5m base with a 4h HTF — hence range fetching below).
        self.required_bars = self._required_base_bars()

    def _required_base_bars(self) -> int:
        base_m = INTERVAL_MINUTES.get(self.base_timeframe, 5)
        need = _WARMUP_BARS
        for tf in self.htf_timeframes:
            htf_m = INTERVAL_MINUTES.get(tf, base_m)
            need = max(need, int(_WARMUP_BARS * htf_m / base_m))
        return int(min(need + 250, 30_000))

    def _fetch_base(self, symbol: str) -> pd.DataFrame:
        """Fetch enough base history for multi-timeframe warm-up (paginated)."""
        base_m = INTERVAL_MINUTES.get(self.base_timeframe, 5)
        if hasattr(self.client, "get_klines_range"):
            # Anchor to the client's clock when it exposes one (e.g. the offline
            # SyntheticClient), else the real UTC clock.
            end = getattr(self.client, "_now", None) or datetime.now(timezone.utc)
            start = end - timedelta(minutes=self.required_bars * base_m)
            return self.client.get_klines_range(
                symbol, self.base_timeframe, start_time=start, end_time=end
            )
        return self.client.get_klines(symbol, self.base_timeframe, limit=self.required_bars)

    def _side_score(self, model: SideModel | None, x_last: pd.DataFrame, regime_last, alpha_90) -> tuple[float, str]:
        if model is None:
            return float("nan"), "{}"
        raw = float(model.ensemble.predict_proba_positive(x_last)[0])
        cal = float(model.calibrator.transform(np.array([raw]), np.array([regime_last]))[0])
        r90 = model.conformal.predict(np.array([cal]), alpha_90)
        return cal, _set_repr(bool(r90.include_0[0]), bool(r90.include_1[0]))

    def score_symbol(self, symbol: str, base_df: pd.DataFrame) -> dict | None:
        artifact = self.artifacts.get(symbol)
        if artifact is None or base_df is None or len(base_df) < 100:
            logger.warning("%s: too little base history (%s bars)", symbol, 0 if base_df is None else len(base_df))
            return None

        frames = resample_to_timeframes(
            base_df, [self.base_timeframe, *self.htf_timeframes],
            base_timeframe=self.base_timeframe, drop_incomplete=True,
        )
        matrix = build_feature_matrix(
            frames, base_timeframe=self.base_timeframe, htf_timeframes=self.htf_timeframes,
            params=artifact.feature_params, dropna=True,
        )
        if matrix.empty:
            logger.warning(
                "%s: empty feature matrix (%d base bars insufficient for %s warm-up); "
                "need ~%d bars", symbol, len(base_df), self.htf_timeframes, self.required_bars,
            )
            return None
        missing = [f for f in artifact.feature_names if f not in matrix.columns]
        if missing:
            logger.warning("%s: features missing from matrix: %s", symbol, missing[:5])
            return None

        regime = self._regime_clf.classify(
            matrix, adx_col=f"{self.base_timeframe}_adx", volatility_col=f"{self.base_timeframe}_rvol"
        ).labels
        regime_last = regime.iloc[-1]
        x_last = matrix[artifact.feature_names].iloc[[-1]]
        alpha_90 = artifact.metadata.get("alpha_90", self.config.conformal.alpha_90)

        long_prob, long_set = self._side_score(artifact.long, x_last, regime_last, alpha_90)
        short_prob, short_set = self._side_score(artifact.short, x_last, regime_last, alpha_90)

        long_conf = long_set == "{1}"
        short_conf = short_set == "{1}"
        if long_conf and (not short_conf or long_prob >= short_prob):
            signal, direction, score = "LONG", "long", long_prob
        elif short_conf and (not long_conf or short_prob > long_prob):
            signal, direction, score = "SHORT", "short", short_prob
        else:
            signal, direction, score = "FLAT", "flat", float(np.nanmax([long_prob, short_prob]))

        atr_col = f"{self.base_timeframe}_atr"
        atr_val = float(matrix[atr_col].iloc[-1]) if atr_col in matrix.columns else float("nan")

        return {
            "symbol": symbol,
            "time": matrix.index[-1],
            "close": float(base_df["close"].iloc[-1]),
            "atr": atr_val,
            "long_prob": long_prob,
            "short_prob": short_prob,
            "long_conf": long_set == "{1}",
            "short_conf": short_set == "{1}",
            "regime": regime_last,
            "long_set90": long_set,
            "short_set90": short_set,
            "signal": signal,
            "direction": direction,
            "score": score,
        }

    def scan_once(self) -> pd.DataFrame:
        rows = []
        for symbol in self.artifacts:
            try:
                base_df = self._fetch_base(symbol)
                scored = self.score_symbol(symbol, base_df)
                if scored is not None:
                    rows.append(scored)
            except Exception as exc:
                logger.warning("Scoring failed for %s: %s", symbol, exc)

        cols = ["symbol", "time", "close", "long_prob", "short_prob", "regime",
                "long_set90", "short_set90", "signal", "direction", "score", "rank", "selected"]
        if not rows:
            return pd.DataFrame(columns=cols)

        df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        df["selected"] = False

        cap = self.config.backtest.max_positions_per_direction
        actionable = df[df["signal"].isin(["LONG", "SHORT"])]
        selected_idx = actionable.groupby("direction").head(cap).index
        df.loc[selected_idx, "selected"] = True
        return df

    def scan_signals(
        self,
        *,
        top_n: int = 5,
        max_move_pct: float | None = 5.0,
        fundamentals: dict | None = None,
        min_quote_volume: float = 0.0,
    ) -> dict[str, pd.DataFrame]:
        """Return the best ``top_n`` LONG and ``top_n`` SHORT trade plans.

        Each coin is assigned to a **single** side — whichever direction its
        model leans toward — so the LONG and SHORT lists never contain the same
        symbol. Each row carries an entry zone, stop and TP1/TP2 with
        percentages from the current ATR and the model's training barrier.

        ``fundamentals`` (``{symbol: {"quote_volume": .., "funding": ..}}``) adds
        liquidity/funding context; ``min_quote_volume`` drops illiquid coins
        (a basic tradeability gate). ``max_move_pct`` keeps only setups whose
        TP2 move is within that percentage (e.g. the 0-5% swing requested).
        """
        raw = []
        for symbol in self.artifacts:
            try:
                base_df = self._fetch_base(symbol)
                scored = self.score_symbol(symbol, base_df)
                if scored is not None:
                    raw.append(scored)
            except Exception as exc:
                logger.warning("Scoring failed for %s: %s", symbol, exc)

        if not raw:
            empty = _empty_signal_frame()
            return {"long": empty, "short": empty}

        df = pd.DataFrame(raw)
        fundamentals = fundamentals or {}
        df["quote_volume"] = df["symbol"].map(lambda s: fundamentals.get(s, {}).get("quote_volume", float("nan")))
        df["funding"] = df["symbol"].map(lambda s: fundamentals.get(s, {}).get("funding", float("nan")))

        # Fundamental (tradeability) gate: drop illiquid coins.
        if min_quote_volume > 0:
            liquid = df["quote_volume"].fillna(0.0) >= min_quote_volume
            dropped = (~liquid).sum()
            if dropped:
                logger.info("Liquidity filter dropped %d coin(s) below %.0f volume", dropped, min_quote_volume)
            df = df[liquid]
        if df.empty:
            empty = _empty_signal_frame()
            return {"long": empty, "short": empty}

        # Assign each coin to its dominant side -> disjoint LONG/SHORT lists.
        dominant = np.where(df["long_prob"] >= df["short_prob"], "long", "short")
        longs = df[dominant == "long"]
        shorts = df[dominant == "short"]
        return {
            "long": self._side_signals(longs, "long", top_n, max_move_pct),
            "short": self._side_signals(shorts, "short", top_n, max_move_pct),
        }

    def _side_signals(self, df: pd.DataFrame, side: str, top_n: int, max_move_pct: float | None) -> pd.DataFrame:
        prob_col = f"{side}_prob"
        conf_col = f"{side}_conf"
        meta = next(iter(self.artifacts.values())).metadata
        tp_mult = meta.get("tp_mult", 2.0)
        sl_mult = meta.get("sl_mult", 1.0)

        rows = []
        for _, r in df.sort_values(prob_col, ascending=False).iterrows():
            lv = compute_levels(r["close"], r["atr"], side, tp_mult=tp_mult, sl_mult=sl_mult)
            if max_move_pct is not None and abs(lv.tp2_pct) > max_move_pct:
                continue
            rows.append({
                "symbol": r["symbol"], "side": side.upper(), "prob": r[prob_col],
                "confident": bool(r[conf_col]), "regime": r["regime"], "price": r["close"],
                "quote_volume": r.get("quote_volume", float("nan")),
                "funding": r.get("funding", float("nan")),
                "entry_low": lv.entry_low, "entry_high": lv.entry_high,
                "stop_loss": lv.stop_loss, "tp1": lv.tp1, "tp2": lv.tp2,
                "sl_pct": lv.sl_pct, "tp1_pct": lv.tp1_pct, "tp2_pct": lv.tp2_pct,
            })
            if len(rows) >= top_n:
                break
        return pd.DataFrame(rows) if rows else _empty_signal_frame()

    def run(
        self,
        *,
        max_iterations: int | None = None,
        buffer_seconds: float = 5.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        persist: bool = True,
    ) -> None:
        report_dir = self.config.storage.report_directory
        retention = self.config.storage.retention_days_scans
        iteration = 0
        while max_iterations is None or iteration < max_iterations:
            now = now_fn()
            target = next_boundary(now, self.base_timeframe)
            sleep_fn(seconds_until(target, now=now) + buffer_seconds)
            results = self.scan_once()
            self._emit(results)
            if persist and not results.empty:
                save_scan(results, report_dir)
                cleanup_old_reports(report_dir, retention)
            iteration += 1

    def _emit(self, results: pd.DataFrame) -> None:  # pragma: no cover - I/O
        cols = ["rank", "symbol", "signal", "long_prob", "short_prob", "regime", "selected"]
        available = [c for c in cols if c in results.columns]
        print(f"\n=== Futures scan @ {datetime.now(timezone.utc).isoformat()} ===")
        print(render_table(results, columns=available))
