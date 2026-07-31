"""Tests for the multi-timeframe technical screener."""

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.synthetic import SyntheticClient  # noqa: E402
from src.features.indicators import macd, stoch_rsi  # noqa: E402
from src.reporting.screener_report import render_screener_html  # noqa: E402
from src.scanner.screener import Screener, screen_symbol  # noqa: E402


def _series(n, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.Series(100 + np.cumsum(rng.normal(0, 0.3, n)), index=idx)


class IndicatorTests(unittest.TestCase):
    def test_macd_columns(self):
        m = macd(_series(200))
        self.assertEqual(list(m.columns), ["macd", "signal", "hist"])
        # histogram = macd - signal
        np.testing.assert_allclose((m["macd"] - m["signal"]).dropna(), m["hist"].dropna(), atol=1e-9)

    def test_stochrsi_bounded(self):
        sr = stoch_rsi(_series(300)).dropna()
        self.assertTrue((sr["k"] >= -1e-6).all() and (sr["k"] <= 100 + 1e-6).all())


class ScreenSymbolTests(unittest.TestCase):
    def _frames(self, seed):
        client = SyntheticClient(now=datetime(2024, 6, 1, tzinfo=timezone.utc))
        from src.data.resample import resample_to_timeframes
        base = client.get_klines("XUSDT", "5m", limit=1000)  # deterministic per 'XUSDT'
        return resample_to_timeframes(base, ["5m", "15m", "1h"], base_timeframe="5m", drop_incomplete=True)

    def test_screen_returns_multitf_readout(self):
        row = screen_symbol(self._frames(0))
        self.assertIsNotNone(row)
        for key in ["rsi_5m", "rsi_15m", "rsi_1h", "trend_1h", "macd_1h", "stoch_1h", "side", "score", "leading"]:
            self.assertIn(key, row)
        self.assertIn(row["side"], {"long", "short"})
        self.assertTrue(0.0 <= row["score"] <= 1.0)

    def test_uptrend_leans_long(self):
        # A clean rising base across all timeframes should read long.
        idx = pd.date_range("2024-01-01", periods=1000, freq="5min", tz="UTC")
        close = pd.Series(np.linspace(100, 140, 1000), index=idx)
        o = close.shift(1).fillna(close)
        df = pd.DataFrame({"open": o, "high": close + 0.1, "low": close - 0.1,
                           "close": close, "volume": pd.Series(1000.0, index=idx)}, index=idx)
        from src.data.resample import resample_to_timeframes
        frames = resample_to_timeframes(df, ["5m", "15m", "1h"], base_timeframe="5m", drop_incomplete=True)
        row = screen_symbol(frames)
        self.assertEqual(row["side"], "long")
        self.assertEqual(row["trend_1h"], "up")


class ScreenerScanTests(unittest.TestCase):
    def test_scan_and_disjoint(self):
        client = SyntheticClient(now=datetime(2024, 6, 1, tzinfo=timezone.utc))
        scanner = Screener(client)
        sig = scanner.scan(["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"], top_n=10, min_score=0.0)
        longs = set(sig["long"]["symbol"]) if not sig["long"].empty else set()
        shorts = set(sig["short"]["symbol"]) if not sig["short"].empty else set()
        self.assertEqual(longs & shorts, set())
        self.assertGreater(len(longs) + len(shorts), 0)

    def test_rsi_filter(self):
        client = SyntheticClient(now=datetime(2024, 6, 1, tzinfo=timezone.utc))
        scanner = Screener(client)
        sig = scanner.scan(["BTCUSDT", "ETHUSDT", "SOLUSDT"], top_n=10, min_score=0.0, rsi_below=40)
        for side in ("long", "short"):
            if not sig[side].empty:
                self.assertTrue((sig[side]["rsi_1h"] < 40).all())

    def test_html_renders(self):
        client = SyntheticClient(now=datetime(2024, 6, 1, tzinfo=timezone.utc))
        sig = Screener(client).scan(["BTCUSDT", "ETHUSDT"], top_n=5, min_score=0.0)
        out = render_screener_html(sig, meta={"scanned": 2})
        self.assertIn("<!doctype html>", out)
        self.assertIn("Screener", out)
        self.assertIn("RSI", out)


if __name__ == "__main__":
    unittest.main()
