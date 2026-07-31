"""Live scanner: score a universe on each closed base-timeframe candle.

On every base-timeframe boundary (5m by default) the scanner pulls recent
klines per symbol, rebuilds the exact feature pipeline stored in each model
artifact, scores the **last closed** bar, calibrates and wraps the score in a
conformal prediction set, then ranks symbols cross-sectionally and marks the
top-N per direction as actionable signals.

Timing and I/O are injectable (``client``, ``sleep_fn``, ``now_fn``) so a single
scan and the loop are both testable without the network or real waiting.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd

from ..config_schema import AppConfig
from ..data.binance_client import BinanceClient
from ..data.resample import resample_to_timeframes
from ..features.engineering import build_feature_matrix
from ..logging_utils import get_logger
from ..pipeline.artifacts import ModelArtifact
from ..regime.classifier import RegimeClassifier
from ..reporting.report import cleanup_old_reports, render_table, save_scan
from ..utils.timing import next_boundary, seconds_until

logger = get_logger(__name__)


def _set_repr(include_0: bool, include_1: bool) -> str:
    if include_0 and include_1:
        return "{0,1}"
    if include_1:
        return "{1}"
    if include_0:
        return "{0}"
    return "{}"


class LiveScanner:
    def __init__(
        self,
        config: AppConfig,
        artifacts: dict[str, ModelArtifact],
        client: BinanceClient,
        *,
        history_bars: int = 1500,
        vol_window: int = 500,
    ) -> None:
        if not artifacts:
            raise ValueError("at least one model artifact is required")
        self.config = config
        self.artifacts = artifacts
        self.client = client
        self.history_bars = history_bars
        self.vol_window = vol_window

        meta = next(iter(artifacts.values())).metadata
        self.base_timeframe = meta.get("base_timeframe", "5m")
        self.htf_timeframes = list(meta.get("htf_timeframes", ["15m", "1h", "4h"]))
        self._regime_clf = RegimeClassifier(config.regime, vol_window=vol_window)

    # -- scoring ------------------------------------------------------------
    def score_symbol(self, symbol: str, base_df: pd.DataFrame) -> dict | None:
        artifact = self.artifacts.get(symbol)
        if artifact is None:
            return None
        if base_df is None or len(base_df) < 100:
            logger.debug("Not enough history for %s", symbol)
            return None

        frames = resample_to_timeframes(
            base_df, [self.base_timeframe, *self.htf_timeframes],
            base_timeframe=self.base_timeframe, drop_incomplete=True,
        )
        matrix = build_feature_matrix(
            frames,
            base_timeframe=self.base_timeframe,
            htf_timeframes=self.htf_timeframes,
            params=artifact.feature_params,
            dropna=True,
        )
        if matrix.empty:
            return None

        # Regime from the full matrix (needs adx/rvol columns).
        adx_col = f"{self.base_timeframe}_adx"
        vol_col = f"{self.base_timeframe}_rvol"
        regime = self._regime_clf.classify(matrix, adx_col=adx_col, volatility_col=vol_col).labels
        last_time = matrix.index[-1]
        regime_last = regime.iloc[-1]

        missing = [f for f in artifact.feature_names if f not in matrix.columns]
        if missing:
            logger.warning("%s: artifact features missing from matrix: %s", symbol, missing[:5])
            return None
        x_last = matrix[artifact.feature_names].iloc[[-1]]

        raw = float(artifact.ensemble.predict_proba_positive(x_last)[0])
        cal = float(artifact.calibrator.transform(np.array([raw]), np.array([regime_last]))[0])

        alpha_90 = artifact.metadata.get("alpha_90", self.config.conformal.alpha_90)
        alpha_80 = artifact.metadata.get("alpha_80", self.config.conformal.alpha_80)
        r90 = artifact.conformal.predict(np.array([cal]), alpha_90)
        r80 = artifact.conformal.predict(np.array([cal]), alpha_80)
        set90 = _set_repr(bool(r90.include_0[0]), bool(r90.include_1[0]))
        set80 = _set_repr(bool(r80.include_0[0]), bool(r80.include_1[0]))

        return {
            "symbol": symbol,
            "time": last_time,
            "close": float(base_df["close"].iloc[-1]),
            "prob_raw": raw,
            "prob": cal,
            "regime": regime_last,
            "set90": set90,
            "set80": set80,
            "direction": "long",
            "score": cal,
            "confident": set90 == "{1}",
        }

    def scan_once(self, *, now: datetime | None = None) -> pd.DataFrame:
        rows = []
        for symbol in self.artifacts:
            try:
                base_df = self.client.get_klines(symbol, self.base_timeframe, limit=self.history_bars)
                scored = self.score_symbol(symbol, base_df)
                if scored is not None:
                    rows.append(scored)
            except Exception as exc:  # keep scanning the rest of the universe
                logger.warning("Scoring failed for %s: %s", symbol, exc)

        if not rows:
            return pd.DataFrame(
                columns=["symbol", "time", "close", "prob_raw", "prob", "regime",
                         "set90", "set80", "direction", "score", "confident", "rank", "selected"]
            )

        df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)

        # Select top-N confident longs per direction (position cap).
        cap = self.config.backtest.max_positions_per_direction
        df["selected"] = False
        confident = df[df["confident"]]
        selected_idx = confident.groupby("direction").head(cap).index
        df.loc[selected_idx, "selected"] = True
        return df

    # -- loop ---------------------------------------------------------------
    def run(
        self,
        *,
        max_iterations: int | None = None,
        buffer_seconds: float = 5.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        now_fn: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        persist: bool = True,
    ) -> None:
        """Run the aligned scan loop (blocking). ``max_iterations`` bounds it."""
        report_dir = self.config.storage.report_directory
        retention = self.config.storage.retention_days_scans

        iteration = 0
        while max_iterations is None or iteration < max_iterations:
            now = now_fn()
            target = next_boundary(now, self.base_timeframe)
            wait = seconds_until(target, now=now) + buffer_seconds
            logger.info("Sleeping %.1fs until %s close", wait, self.base_timeframe)
            sleep_fn(wait)

            results = self.scan_once(now=now_fn())
            self._emit(results)
            if persist and not results.empty:
                save_scan(results, report_dir)
                cleanup_old_reports(report_dir, retention)
            iteration += 1

    def _emit(self, results: pd.DataFrame) -> None:  # pragma: no cover - I/O
        cols = ["rank", "symbol", "prob", "regime", "set90", "set80", "selected"]
        available = [c for c in cols if c in results.columns]
        print(f"\n=== Scan @ {datetime.now(timezone.utc).isoformat()} ===")
        print(render_table(results, columns=available))
