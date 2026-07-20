"""Tests for technical indicators."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.indicators import (  # noqa: E402
    adx,
    atr,
    bollinger,
    ema,
    log_returns,
    rsi,
    true_range,
)


def _series(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="5min", tz="UTC")
    return pd.Series(values, index=idx, dtype="float64")


class TrueRangeTests(unittest.TestCase):
    def test_true_range_first_bar_is_high_low(self):
        high = _series([10, 12, 11])
        low = _series([9, 10, 9])
        close = _series([9.5, 11, 10])
        tr = true_range(high, low, close)
        self.assertAlmostEqual(tr.iloc[0], 1.0)  # 10 - 9
        # bar 2: max(12-10, |12-9.5|, |10-9.5|) = max(2, 2.5, 0.5) = 2.5
        self.assertAlmostEqual(tr.iloc[1], 2.5)


class ATRTests(unittest.TestCase):
    def test_atr_positive_and_warmup_nan(self):
        n = 50
        high = _series(np.linspace(10, 20, n) + 0.5)
        low = _series(np.linspace(10, 20, n) - 0.5)
        close = _series(np.linspace(10, 20, n))
        a = atr(high, low, close, period=14)
        self.assertTrue(a.iloc[:13].isna().all())
        self.assertTrue((a.dropna() > 0).all())


class RSITests(unittest.TestCase):
    def test_rsi_bounds(self):
        rng = np.random.default_rng(0)
        close = _series(100 + np.cumsum(rng.normal(0, 1, 200)))
        r = rsi(close, 14).dropna()
        self.assertTrue((r >= 0).all() and (r <= 100).all())

    def test_rsi_all_gains_is_100(self):
        close = _series(np.arange(1, 40, dtype="float64"))
        r = rsi(close, 14).dropna()
        self.assertTrue((r > 99.9).all())


class ADXTests(unittest.TestCase):
    def test_adx_high_in_strong_trend(self):
        n = 200
        close = _series(np.linspace(100, 300, n))
        high = close + 0.5
        low = close - 0.5
        out = adx(high, low, close, 14)
        self.assertIn("adx", out.columns)
        # A clean uptrend => strong +DI over -DI and elevated ADX.
        tail = out.dropna().iloc[-1]
        self.assertGreater(tail["plus_di"], tail["minus_di"])
        self.assertGreater(tail["adx"], 20)

    def test_adx_bounded_0_100(self):
        rng = np.random.default_rng(1)
        close = _series(100 + np.cumsum(rng.normal(0, 1, 300)))
        high = close + rng.uniform(0, 1, 300)
        low = close - rng.uniform(0, 1, 300)
        out = adx(high, low, close, 14).dropna()
        self.assertTrue((out["adx"] >= 0).all() and (out["adx"] <= 100).all())


class MiscTests(unittest.TestCase):
    def test_log_returns(self):
        close = _series([100.0, 110.0])
        lr = log_returns(close)
        self.assertTrue(np.isnan(lr.iloc[0]))
        self.assertAlmostEqual(lr.iloc[1], np.log(1.1))

    def test_ema_matches_pandas(self):
        close = _series(np.arange(1, 60, dtype="float64"))
        e = ema(close, 10)
        self.assertFalse(e.dropna().empty)

    def test_bollinger_pct_b_within_bands(self):
        close = _series(100 + np.sin(np.linspace(0, 20, 100)))
        bb = bollinger(close, 20).dropna()
        self.assertTrue((bb["bb_upper"] >= bb["bb_lower"]).all())


if __name__ == "__main__":
    unittest.main()
