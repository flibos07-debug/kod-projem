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
from datetime import datetime, timezone
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
from ..utils.timing import next_boundary, seconds_until
from .live_scanner import _set_repr

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

        return {
            "symbol": symbol,
            "time": matrix.index[-1],
            "close": float(base_df["close"].iloc[-1]),
            "long_prob": long_prob,
            "short_prob": short_prob,
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
                base_df = self.client.get_klines(symbol, self.base_timeframe, limit=self.history_bars)
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
