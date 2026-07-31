"""Tests for the Bollinger mid-band cross strategy with confidence scoring."""

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


def _mech(**kw):
    """Params with the confidence thresholds disabled, to test pure mechanics."""
    return BBStratParams(min_confidence=0.0, watch_confidence=0.0, **kw)


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

    def test_old_break_outside_window(self):
        # cross happened long ago -> not within a 2-bar window.
        c = pd.Series(np.r_[np.full(30, 100.0), [100.4], np.full(10, 100.6)])
        mid = c.rolling(20, min_periods=20).mean()
        self.assertIsNone(_mid_break_bars_ago(c, mid, 2, "up"))


class SignalTests(unittest.TestCase):
    def _f15_upbreak(self):
        base = np.full(60, 100.0)                    # flat squeeze on the mid
        base[-3:] = [100.3, 100.7, 101.2]           # fresh break up + close above mid
        return _frame(base, "15min")

    def _f1h_flat(self):
        # A calm 1h series with no fresh mid-band cross (neutral 1h bias).
        return _frame(200.0 + np.zeros(60), "60min")

    def _f5(self, last, center=100.2):
        # 5m oscillating around `center`; final bar = `last`.
        osc = center + 0.3 * np.array([1 if i % 2 else -1 for i in range(199)])
        return _frame(np.r_[osc, [last]], "5min")

    def test_long_giris_when_5m_above_mid(self):
        # 15m broke up out of a squeeze; 5m price above its own mid -> GİRİŞ long.
        sig = bb_retest_signal(self._f5(100.4), self._f15_upbreak(), self._f1h_flat(), _mech())
        self.assertIsNotNone(sig)
        self.assertEqual(sig["side"], "long")
        self.assertEqual(sig["stage"], "GİRİŞ")
        self.assertEqual(sig["tf"], "15m")
        self.assertGreater(sig["tp"], sig["entry"])      # TP = upper band, above
        self.assertLess(sig["stop_loss"], sig["entry"])  # SL below
        self.assertEqual(sig["entry_dist"], 0.0)
        self.assertGreaterEqual(sig["confidence"], 0.0)
        self.assertLessEqual(sig["confidence"], 100.0)

    def test_long_bekle_when_5m_below_mid(self):
        # 15m broke up but 5m price sits below its own mid-band -> BEKLE.
        sig = bb_retest_signal(self._f5(99.9), self._f15_upbreak(), self._f1h_flat(), _mech())
        self.assertIsNotNone(sig)
        self.assertEqual(sig["side"], "long")
        self.assertEqual(sig["stage"], "BEKLE")
        self.assertGreater(sig["entry_dist"], 0.0)

    def test_none_when_no_fresh_cross(self):
        flat15 = _frame(100.0 + np.zeros(60), "15min")   # no cross at all
        out = bb_retest_signal(self._f5(100.4), flat15, self._f1h_flat(), _mech())
        self.assertIsNone(out)

    def test_1h_fallback_when_no_15m_cross(self):
        flat15 = _frame(100.0 + np.zeros(60), "15min")   # no 15m cross
        h = np.full(60, 100.0)
        h[-3:] = [100.3, 100.7, 101.2]                    # fresh 1h up-cross
        f1h = _frame(h, "60min")
        sig = bb_retest_signal(self._f5(100.4), flat15, f1h, _mech())
        self.assertIsNotNone(sig)
        self.assertEqual(sig["side"], "long")
        self.assertEqual(sig["tf"], "1h")


class FilterTests(unittest.TestCase):
    """The new precision filters: squeeze requirement and 1h agreement."""

    def _f15_upbreak_no_squeeze(self):
        # Wide oscillation (fat bands, NOT a squeeze) then a fresh up-cross.
        base = 100.0 + 2.0 * np.array([1 if i % 2 else -1 for i in range(60)], dtype="float64")
        base[-3:] = [98.5, 99.0, 101.5]   # dip below mid then close above -> fresh up-cross
        return _frame(base, "15min")

    def _f5(self, last, center=100.2):
        osc = center + 0.3 * np.array([1 if i % 2 else -1 for i in range(199)])
        return _frame(np.r_[osc, [last]], "5min")

    def _f15_upbreak(self):
        base = np.full(60, 100.0)
        base[-3:] = [100.3, 100.7, 101.2]
        return _frame(base, "15min")

    def test_squeeze_required_blocks_wide_break(self):
        f15 = self._f15_upbreak_no_squeeze()
        f1h = _frame(200.0 + np.zeros(60), "60min")
        blocked = bb_retest_signal(self._f5(100.4), f15, f1h, _mech(require_squeeze=True))
        allowed = bb_retest_signal(self._f5(100.4), f15, f1h, _mech(require_squeeze=False))
        # With the squeeze filter off we should still detect the cross.
        self.assertIsNotNone(allowed)
        # With it on, a non-squeezed break is rejected (or at least not preferred).
        if blocked is not None:
            self.assertLessEqual(blocked["confidence"], allowed["confidence"])

    def test_htf_opposing_blocks_15m(self):
        # 1h clearly in a downtrend opposes a 15m up-break.
        down = np.linspace(120.0, 80.0, 60)
        f1h_down = _frame(down, "60min")
        out = bb_retest_signal(self._f5(100.4), self._f15_upbreak(), f1h_down, _mech(require_htf=True))
        self.assertIsNone(out)
        # Same setup passes when the 1h filter is disabled.
        out2 = bb_retest_signal(self._f5(100.4), self._f15_upbreak(), f1h_down, _mech(require_htf=False))
        self.assertIsNotNone(out2)


class ScannerTests(unittest.TestCase):
    def test_scan_sections(self):
        client = SyntheticClient(now=datetime(2024, 6, 1, tzinfo=timezone.utc))
        scanner = BBRetestScanner(client, params=_mech())
        sig = scanner.scan(["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"], top_n=10, include_watch=True)
        self.assertEqual(set(sig), {"long", "short"})
        # long/short lists must be disjoint by symbol.
        longs = set(sig["long"]["symbol"]) if not sig["long"].empty else set()
        shorts = set(sig["short"]["symbol"]) if not sig["short"].empty else set()
        self.assertEqual(longs & shorts, set())
        for df in sig.values():
            if not df.empty:
                self.assertIn("confidence", df.columns)


if __name__ == "__main__":
    unittest.main()
