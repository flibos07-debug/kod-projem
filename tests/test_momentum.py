"""Tests for the training-free breakout/momentum scanner."""

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.synthetic import SyntheticClient  # noqa: E402
from src.scanner.momentum_scanner import MomentumScanner, momentum_signal  # noqa: E402


def _frame(close_arr, vol_arr=None, buf=0.2):
    idx = pd.date_range("2024-01-01", periods=len(close_arr), freq="1h", tz="UTC")
    close = pd.Series(close_arr, index=idx, dtype="float64")
    o = close.shift(1).fillna(close)
    high = np.maximum(o, close) + buf
    low = np.minimum(o, close) - buf
    vol = pd.Series(vol_arr if vol_arr is not None else np.full(len(close_arr), 1000.0), index=idx)
    return pd.DataFrame({"open": o, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


class MomentumSignalTests(unittest.TestCase):
    def test_v_bottom_reversal_scores_long(self):
        down = np.linspace(100, 80, 210)
        tail = np.array([80, 80.1, 80, 79.8, 80, 80.2, 80.1, 80, 80.5, 84.0])
        vol = np.r_[np.full(219, 1000.0), [4000.0]]
        sig = momentum_signal(_frame(np.concatenate([down, tail]), vol))
        self.assertGreater(sig["long_score"], sig["short_score"])
        self.assertGreater(sig["long_score"], 0.4)
        self.assertTrue(any("dip" in r for r in sig["long_reasons"]))

    def test_top_rejection_scores_short(self):
        up = np.linspace(80, 100, 210)
        tail = np.array([100, 99.9, 100, 100.2, 100, 99.8, 99.9, 100, 99.5, 96.0])
        vol = np.r_[np.full(219, 1000.0), [4000.0]]
        sig = momentum_signal(_frame(np.concatenate([up, tail]), vol))
        self.assertGreater(sig["short_score"], sig["long_score"])
        self.assertTrue(any("tepe" in r for r in sig["short_reasons"]))

    def test_quiet_series_scores_low(self):
        # A near-flat coin (ATR well under the 0.15% floor) must not fire.
        flat = np.full(220, 100.0) + np.linspace(0, 0.02, 220)  # ~0.02% drift, no range
        sig = momentum_signal(_frame(flat, buf=0.0))
        self.assertLess(max(sig["long_score"], sig["short_score"]), 0.35)

    def test_too_short_returns_none(self):
        self.assertIsNone(momentum_signal(_frame(np.linspace(100, 101, 10))))


class MomentumScannerTests(unittest.TestCase):
    def test_scan_returns_signal_schema(self):
        client = SyntheticClient(now=datetime(2024, 6, 1, tzinfo=timezone.utc))
        scanner = MomentumScanner(client, base_timeframe="1h", history_bars=300)
        sig = scanner.scan_signals(["BTCUSDT", "ETHUSDT", "SOLUSDT"], top_n=5, min_score=0.0)
        self.assertIn("long", sig)
        self.assertIn("short", sig)
        total = len(sig["long"]) + len(sig["short"])
        self.assertGreater(total, 0)  # something scored
        for side in ("long", "short"):
            if not sig[side].empty:
                for col in ["symbol", "prob", "verdict", "note", "entry_low", "tp2"]:
                    self.assertIn(col, sig[side].columns)

    def test_disjoint_sides(self):
        client = SyntheticClient(now=datetime(2024, 6, 1, tzinfo=timezone.utc))
        scanner = MomentumScanner(client)
        sig = scanner.scan_signals(["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"], top_n=10, min_score=0.0)
        longs = set(sig["long"]["symbol"]) if not sig["long"].empty else set()
        shorts = set(sig["short"]["symbol"]) if not sig["short"].empty else set()
        self.assertEqual(longs & shorts, set())


if __name__ == "__main__":
    unittest.main()
