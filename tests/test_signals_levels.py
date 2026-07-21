"""Tests for trade-level computation and the pro signals report."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reporting.report import render_signals  # noqa: E402
from src.scanner.levels import compute_levels  # noqa: E402


class LevelsTests(unittest.TestCase):
    def test_long_levels_ordering(self):
        lv = compute_levels(100.0, 2.0, "long", tp_mult=2.0, sl_mult=1.0)
        # entry zone is a dip band below/at price; stop below; TPs above.
        self.assertLessEqual(lv.entry_low, lv.entry_high)
        self.assertEqual(lv.entry_high, 100.0)
        self.assertEqual(lv.entry_low, 99.0)         # 100 - 0.5*2
        self.assertEqual(lv.stop_loss, 98.0)         # 100 - 1*2
        self.assertEqual(lv.tp1, 102.0)              # 100 + 1*2
        self.assertEqual(lv.tp2, 104.0)              # 100 + 2*2
        self.assertAlmostEqual(lv.tp2_pct, 4.0)
        self.assertAlmostEqual(lv.sl_pct, -2.0)

    def test_short_levels_mirror(self):
        lv = compute_levels(100.0, 2.0, "short", tp_mult=2.0, sl_mult=1.0)
        self.assertEqual(lv.entry_low, 100.0)
        self.assertEqual(lv.entry_high, 101.0)       # 100 + 0.5*2
        self.assertEqual(lv.stop_loss, 102.0)        # 100 + 1*2
        self.assertEqual(lv.tp1, 98.0)
        self.assertEqual(lv.tp2, 96.0)
        self.assertAlmostEqual(lv.tp2_pct, -4.0)     # price falls 4%
        self.assertAlmostEqual(lv.sl_pct, 2.0)

    def test_zero_atr_collapses_zone(self):
        lv = compute_levels(50.0, 0.0, "long")
        self.assertEqual(lv.entry_low, lv.entry_high)
        self.assertEqual(lv.stop_loss, 50.0)
        self.assertEqual(lv.tp2, 50.0)

    def test_invalid_side(self):
        with self.assertRaises(ValueError):
            compute_levels(100.0, 1.0, "sideways")


class RenderSignalsTests(unittest.TestCase):
    def _frame(self, side):
        return pd.DataFrame([{
            "symbol": "SOLUSDT", "side": side.upper(), "prob": 0.47, "confident": True,
            "regime": "trend", "price": 100.0, "entry_low": 99.0, "entry_high": 100.0,
            "stop_loss": 98.0, "tp1": 102.0, "tp2": 104.0,
            "sl_pct": -2.0, "tp1_pct": 2.0, "tp2_pct": 4.0,
        }])

    def test_render_contains_levels(self):
        out = render_signals({"long": self._frame("long"), "short": pd.DataFrame()})
        self.assertIn("LONG", out)
        self.assertIn("SOLUSDT", out)
        self.assertIn("TP1", out)
        self.assertIn("TP2", out)
        self.assertIn("+4.00%", out)  # tp2 pct

    def test_render_empty_side(self):
        out = render_signals({"long": pd.DataFrame(), "short": pd.DataFrame()})
        self.assertIn("uygun sinyal yok", out)


if __name__ == "__main__":
    unittest.main()
