"""Tests for the offline synthetic data source and the CLI wiring."""

import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.synthetic import SyntheticClient, generate_ohlcv  # noqa: E402
from src.main import build_parser  # noqa: E402


class SyntheticDataTests(unittest.TestCase):
    def test_generate_shape_and_columns(self):
        end = datetime(2024, 6, 1, tzinfo=timezone.utc)
        df = generate_ohlcv(500, freq_minutes=60, seed=1, end_time=end)
        self.assertEqual(len(df), 500)
        for col in ["open", "high", "low", "close", "volume"]:
            self.assertIn(col, df.columns)
        # OHLC invariants hold on every bar.
        self.assertTrue((df["high"] >= df["low"]).all())
        self.assertTrue((df["high"] >= df["close"]).all())
        self.assertTrue((df["low"] <= df["close"]).all())
        self.assertTrue((df["close"] > 0).all())

    def test_deterministic_per_symbol(self):
        client = SyntheticClient(now=datetime(2024, 6, 1, tzinfo=timezone.utc))
        a1 = client.get_klines("BTCUSDT", "1h", limit=100)
        a2 = client.get_klines("BTCUSDT", "1h", limit=100)
        b = client.get_klines("ETHUSDT", "1h", limit=100)
        pd.testing.assert_frame_equal(a1, a2)  # reproducible
        self.assertFalse(a1["close"].equals(b["close"]))  # symbol-specific

    def test_klines_range_spans_requested_window(self):
        client = SyntheticClient(now=datetime(2024, 6, 1, tzinfo=timezone.utc))
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 2, 1, tzinfo=timezone.utc)
        df = client.get_klines_range("BTCUSDT", "1h", start_time=start, end_time=end)
        # ~31 days of hourly bars.
        self.assertGreater(len(df), 700)
        self.assertLess(len(df), 800)

    def test_index_is_utc_and_sorted(self):
        df = generate_ohlcv(50, freq_minutes=5, seed=0, end_time=datetime(2024, 1, 1, tzinfo=timezone.utc))
        self.assertEqual(str(df.index.tz), "UTC")
        self.assertTrue(df.index.is_monotonic_increasing)


class CLIParserTests(unittest.TestCase):
    def test_demo_defaults(self):
        args = build_parser().parse_args(["demo"])
        self.assertEqual(args.command, "demo")
        self.assertIn("BTCUSDT", args.symbols)

    def test_source_flag(self):
        args = build_parser().parse_args(["--source", "synthetic", "scan", "--symbols", "BTCUSDT", "--once"])
        self.assertEqual(args.source, "synthetic")
        self.assertTrue(args.once)

    def test_train_requires_symbols(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["train"])


if __name__ == "__main__":
    unittest.main()
