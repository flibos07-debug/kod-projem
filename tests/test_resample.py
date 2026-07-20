"""Tests for HTF resampling."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.resample import resample_ohlcv, resample_to_timeframes  # noqa: E402


def _make_5m(n: int, start: str = "2024-01-01 00:00:00") -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=n, freq="5min", tz="UTC")
    # Deterministic, monotone-ish OHLCV.
    close = pd.Series(range(1, n + 1), dtype="float64", index=idx)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": pd.Series([10.0] * n, index=idx),
            "trades": pd.Series([3] * n, index=idx),
        },
        index=idx,
    )


class ResampleOHLCVTests(unittest.TestCase):
    def test_5m_to_1h_aggregation(self):
        df = _make_5m(24)  # exactly two full hours
        out = resample_ohlcv(df, "1h", base_timeframe="5m", drop_incomplete=True)
        self.assertEqual(len(out), 2)
        first = out.iloc[0]
        # Hour 1 = bars 1..12
        self.assertEqual(first["open"], df.iloc[0]["open"])
        self.assertEqual(first["high"], df.iloc[:12]["high"].max())
        self.assertEqual(first["low"], df.iloc[:12]["low"].min())
        self.assertEqual(first["close"], df.iloc[11]["close"])
        self.assertEqual(first["volume"], 120.0)
        self.assertEqual(first["trades"], 36)

    def test_label_is_open_time(self):
        df = _make_5m(12)
        out = resample_ohlcv(df, "1h", base_timeframe="5m", drop_incomplete=True)
        self.assertEqual(out.index[0], pd.Timestamp("2024-01-01 00:00:00", tz="UTC"))

    def test_drop_incomplete_removes_partial_last_bar(self):
        df = _make_5m(18)  # one full hour + 6 bars (half of the next hour)
        out = resample_ohlcv(df, "1h", base_timeframe="5m", drop_incomplete=True)
        self.assertEqual(len(out), 1)  # partial second hour dropped

    def test_keep_incomplete_bar(self):
        df = _make_5m(18)
        out = resample_ohlcv(df, "1h", base_timeframe="5m", drop_incomplete=False)
        self.assertEqual(len(out), 2)

    def test_drop_incomplete_requires_base_timeframe(self):
        df = _make_5m(12)
        with self.assertRaises(ValueError):
            resample_ohlcv(df, "1h", drop_incomplete=True)

    def test_missing_ohlc_column_raises(self):
        df = _make_5m(12).drop(columns=["high"])
        with self.assertRaises(ValueError):
            resample_ohlcv(df, "1h", base_timeframe="5m", drop_incomplete=False)

    def test_gaps_produce_no_empty_bars(self):
        df = _make_5m(24)
        # Remove the whole second hour -> a gap.
        df = df.drop(df.index[12:24])
        out = resample_ohlcv(df, "1h", base_timeframe="5m", drop_incomplete=False)
        self.assertEqual(len(out), 1)

    def test_naive_index_is_localised(self):
        df = _make_5m(12)
        df.index = df.index.tz_localize(None)
        out = resample_ohlcv(df, "1h", base_timeframe="5m", drop_incomplete=False)
        self.assertEqual(str(out.index.tz), "UTC")

    def test_unsorted_and_duplicated_index_handled(self):
        df = _make_5m(12)
        shuffled = pd.concat([df.iloc[6:], df.iloc[:6], df.iloc[:1]])  # unsorted + dup
        out = resample_ohlcv(shuffled, "1h", base_timeframe="5m", drop_incomplete=False)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["close"], df.iloc[11]["close"])


class ResampleMultiTests(unittest.TestCase):
    def test_resample_to_timeframes_includes_base(self):
        df = _make_5m(48)  # 4 hours
        out = resample_to_timeframes(
            df, ["5m", "15m", "1h"], base_timeframe="5m", drop_incomplete=True
        )
        self.assertEqual(set(out), {"5m", "15m", "1h"})
        self.assertEqual(len(out["5m"]), 48)
        self.assertEqual(len(out["1h"]), 4)
        self.assertEqual(len(out["15m"]), 16)


if __name__ == "__main__":
    unittest.main()
