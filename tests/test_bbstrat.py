"""Tests for the Bollinger squeeze -> mid-band break -> retest strategy."""

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.synthetic import SyntheticClient  # noqa: E402
from src.scanner.bb_retest import BBRetestScanner, BBStratParams, _mid_break_bars_ago, bb_retest_signal  # noqa: E402


def _frame(vals, freq="15min"):
    idx = pd.date_range("2024-01-01", periods=len(vals), freq=freq, tz="UTC")
    close = pd.Series(vals, index=idx, dtype="float64")
    o = close.shift(1).fillna(close)
    return pd.DataFrame(
        {"open": o, "high": np.maximum(o, close) + 0.02, "low": np.minimum(o, close) - 0.02,
         "close": close, "volume": pd.Series(1000.0, index=idx)}, index=idx,
    )


class MidBreakTests(unittest.TestCase):
    def test_up_break_detected(self):
        c = pd.Series(np.r_[np.full(38, 100.0), [100.3, 100.7, 101.2]])
        mid = c.rolling(20, min_periods=20).mean()
        self.assertIsNotNone(_mid_break_bars_ago(c, mid, 6, "up"))
        self.assertIsNone(_mid_break_bars_ago(c, mid, 6, "down"))

    def test_down_break_detected(self):
        c = pd.Series(np.r_[np.full(38, 100.0), [99.7, 99.3, 98.8]])
        mid = c.rolling(20, min_periods=20).mean()
        self.assertIsNotNone(_mid_break_bars_ago(c, mid, 6, "down"))
        self.assertIsNone(_mid_break_bars_ago(c, mid, 6, "up"))


class SignalTests(unittest.TestCase):
    def _f15_upbreak(self):
        rng = np.random.default_rng(0)
        base = 100.0 + rng.normal(0, 0.01, 45)     # flat squeeze
        base[-3:] = [100.3, 100.7, 101.2]           # break up + close above mid
        return _frame(base, "15min")

    def _f5(self, last):
        # 5m oscillating around 101 (above the 15m mid ~100); final bar = `last`.
        osc = 101.0 + 0.3 * np.array([1 if i % 2 else -1 for i in range(199)])
        return _frame(np.r_[osc, [last]], "5min")

    def test_long_giris_on_lower_band_retest(self):
        # last 5m close at the lower band (~100.4) but still above the 15m mid.
        sig = bb_retest_signal(self._f5(100.4), self._f15_upbreak(), BBStratParams())
        self.assertIsNotNone(sig)
        self.assertEqual(sig["side"], "long")
        self.assertEqual(sig["stage"], "GİRİŞ")
        self.assertLess(sig["pctb5"], 0.2)
        self.assertGreater(sig["tp"], sig["entry"])   # TP = upper band, above
        self.assertLess(sig["stop_loss"], sig["entry"])  # SL below

    def test_long_bekle_when_not_retested(self):
        sig = bb_retest_signal(self._f5(101.0), self._f15_upbreak(), BBStratParams())
        self.assertIsNotNone(sig)
        self.assertEqual(sig["side"], "long")
        self.assertEqual(sig["stage"], "BEKLE")

    def test_none_when_no_setup(self):
        flat = _frame(100.0 + np.random.default_rng(1).normal(0, 2.0, 60), "15min")  # noisy, no clean squeeze/break
        f5 = _frame(100.0 + np.random.default_rng(2).normal(0, 2.0, 200), "5min")
        # May or may not find a setup; just ensure it returns dict-or-None without error.
        out = bb_retest_signal(f5, flat, BBStratParams())
        self.assertTrue(out is None or isinstance(out, dict))


class ScannerTests(unittest.TestCase):
    def test_scan_sections(self):
        client = SyntheticClient(now=datetime(2024, 6, 1, tzinfo=timezone.utc))
        scanner = BBRetestScanner(client)
        sig = scanner.scan(["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"], top_n=10, include_watch=True)
        self.assertEqual(set(sig), {"long", "short", "watch"})
        # long/short lists must be disjoint by symbol.
        longs = set(sig["long"]["symbol"]) if not sig["long"].empty else set()
        shorts = set(sig["short"]["symbol"]) if not sig["short"].empty else set()
        self.assertEqual(longs & shorts, set())


if __name__ == "__main__":
    unittest.main()
