"""Tests for the suitability verdict, crossover features and HTML report."""

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.synthetic import generate_ohlcv  # noqa: E402
from src.features.engineering import timeframe_features  # noqa: E402
from src.reporting.html_report import render_html  # noqa: E402
from src.scanner.suitability import compute_suitability  # noqa: E402


class CrossoverFeatureTests(unittest.TestCase):
    def test_new_features_present_and_binary(self):
        df = generate_ohlcv(400, freq_minutes=60, seed=2, end_time=datetime(2024, 6, 1, tzinfo=timezone.utc))
        f = timeframe_features(df)
        for col in ["ema_cross_up", "ema_cross_dn", "bb_mid_cross_up", "bb_mid_cross_dn", "bb_wide", "bb_mid_dist"]:
            self.assertIn(col, f.columns)
        for col in ["ema_cross_up", "ema_cross_dn", "bb_wide"]:
            vals = set(f[col].dropna().unique())
            self.assertTrue(vals <= {0.0, 1.0}, f"{col} not binary: {vals}")

    def test_crossovers_are_events(self):
        df = generate_ohlcv(600, freq_minutes=60, seed=3, end_time=datetime(2024, 6, 1, tzinfo=timezone.utc))
        f = timeframe_features(df)
        # Crossovers happen occasionally, not on every bar.
        rate = f["bb_mid_cross_up"].mean()
        self.assertTrue(0 < rate < 0.5)


class SuitabilityTests(unittest.TestCase):
    def test_confident_favorable_long_is_uygun(self):
        s = compute_suitability("long", prob=0.55, confident=True, funding_pct=-0.01, ls_ratio=0.9)
        self.assertEqual(s.verdict, "UYGUN")
        self.assertGreaterEqual(s.score, 3)

    def test_weak_low_prob_is_zayif(self):
        s = compute_suitability("long", prob=0.12, confident=False, funding_pct=0.10, ls_ratio=2.5)
        self.assertEqual(s.verdict, "ZAYIF")

    def test_crowded_long_penalized(self):
        crowded = compute_suitability("long", prob=0.45, confident=False, funding_pct=0.08, ls_ratio=2.5)
        clean = compute_suitability("long", prob=0.45, confident=False, funding_pct=-0.02, ls_ratio=0.8)
        self.assertLess(crowded.score, clean.score)

    def test_short_funding_alignment(self):
        s = compute_suitability("short", prob=0.5, confident=True, funding_pct=0.05, ls_ratio=1.6)
        self.assertIn(s.verdict, {"UYGUN", "DİKKATLİ"})

    def test_nan_context_is_safe(self):
        s = compute_suitability("long", prob=0.5, confident=True, funding_pct=float("nan"), ls_ratio=float("nan"))
        self.assertIn(s.verdict, {"UYGUN", "DİKKATLİ", "ZAYIF"})


class HtmlReportTests(unittest.TestCase):
    def _frame(self, side):
        return pd.DataFrame([{
            "symbol": "SOLUSDT", "side": side.upper(), "prob": 0.47, "confident": True,
            "verdict": "UYGUN", "note": "model emin, funding uygun", "regime": "trend",
            "price": 100.0, "quote_volume": 1.2e9, "funding": 0.01, "ls_ratio": 1.3,
            "open_interest": 500000.0, "entry_low": 99.0, "entry_high": 100.0,
            "stop_loss": 98.0, "tp1": 102.0, "tp2": 104.0,
            "sl_pct": -2.0, "tp1_pct": 2.0, "tp2_pct": 4.0,
        }])

    def test_render_html_contains_key_parts(self):
        out = render_html({"long": self._frame("long"), "short": pd.DataFrame()},
                          meta={"scanned": 20, "max_move": 5, "min_volume": 50})
        self.assertIn("<!doctype html>", out)
        self.assertIn("SOLUSDT", out)
        self.assertIn("UYGUN", out)
        self.assertIn("TP2", out)
        self.assertIn("badge", out)  # verdict badge styling
        self.assertIn("Uygun sinyal yok", out)  # empty short side

    def test_html_escapes_symbol(self):
        df = self._frame("long")
        df.loc[0, "symbol"] = "<script>"
        out = render_html({"long": df, "short": pd.DataFrame()})
        self.assertNotIn("<script>", out)
        self.assertIn("&lt;script&gt;", out)


if __name__ == "__main__":
    unittest.main()
