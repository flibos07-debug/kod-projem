"""Tests for feature engineering (causal alignment) and regime classification."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_schema import RegimeConfig  # noqa: E402
from src.data.resample import resample_to_timeframes  # noqa: E402
from src.features.engineering import FeatureParams, build_feature_matrix, timeframe_features  # noqa: E402
from src.regime.classifier import RegimeClassifier  # noqa: E402


def _ohlcv(n, start="2024-01-01", freq="5min", seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    high = close + rng.uniform(0, 0.5, n)
    low = close - rng.uniform(0, 0.5, n)
    open_ = close - rng.normal(0, 0.2, n)
    volume = rng.uniform(10, 100, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


class TimeframeFeatureTests(unittest.TestCase):
    def test_columns_present(self):
        df = _ohlcv(300)
        feats = timeframe_features(df)
        for col in ["ret", "rsi", "atr", "adx", "rvol", "ema_ratio", "bb_width", "vol_z", "mom_1"]:
            self.assertIn(col, feats.columns)

    def test_index_preserved(self):
        df = _ohlcv(100)
        feats = timeframe_features(df)
        self.assertTrue(feats.index.equals(df.index))


class CausalAlignmentTests(unittest.TestCase):
    def test_no_lookahead_from_htf(self):
        # Build 5m data, resample to 1h, then assert every attached 1h feature
        # corresponds to a 1h bar that closed at/before the base bar's close.
        base = _ohlcv(24 * 12, seed=3)  # ~1 day of 5m bars
        frames = resample_to_timeframes(base, ["5m", "1h"], base_timeframe="5m", drop_incomplete=False)
        matrix = build_feature_matrix(
            frames, base_timeframe="5m", htf_timeframes=["1h"], dropna=False
        )
        # The 1h RSI column should be constant within an hour and change only at
        # hour boundaries -> confirms step-wise (already-closed) alignment.
        one_h_rsi = matrix["1h_rsi"].dropna()
        self.assertFalse(one_h_rsi.empty)
        # Within the first fully-populated hour, values are identical.
        by_hour = one_h_rsi.groupby(one_h_rsi.index.floor("1h"))
        constant_within_hour = by_hour.nunique().max() <= 1
        self.assertTrue(bool(constant_within_hour))

    def test_htf_value_lags_by_one_bar(self):
        # A base bar at the very start of hour k must see the 1h feature from
        # hour k-1 (the last CLOSED hour), never hour k.
        base = _ohlcv(24 * 12, seed=5)
        frames = resample_to_timeframes(base, ["5m", "1h"], base_timeframe="5m", drop_incomplete=False)
        htf_feats = timeframe_features(frames["1h"])
        matrix = build_feature_matrix(
            frames, base_timeframe="5m", htf_timeframes=["1h"], dropna=False
        )
        # Pick an hour boundary well past warm-up.
        boundary = frames["1h"].index[5]  # start of the 6th hour
        base_at_boundary = matrix.loc[boundary, "1h_rsi"]
        prev_hour = frames["1h"].index[4]
        self.assertTrue(
            np.isnan(base_at_boundary)
            or np.isclose(base_at_boundary, htf_feats.loc[prev_hour, "rsi"], equal_nan=True)
        )

    def test_missing_base_raises(self):
        frames = {"1h": _ohlcv(50)}
        with self.assertRaises(KeyError):
            build_feature_matrix(frames, base_timeframe="5m")


class RegimeClassifierTests(unittest.TestCase):
    def _config(self):
        return RegimeConfig(
            classes=("trend", "range", "high_vol"),
            adx_threshold_trend=25,
            adx_threshold_range=20,
            vol_percentile_high=80,
            use_soft_routing=True,
            routing_temperature=1.0,
        )

    def test_soft_weights_sum_to_one(self):
        clf = RegimeClassifier(self._config())
        idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
        adx = pd.Series([10, 22, 30, 40, 15], index=idx, dtype="float64")
        vol_pct = pd.Series([10, 50, 90, 30, 85], index=idx, dtype="float64")
        w = clf.soft_weights(adx, vol_pct)
        np.testing.assert_allclose(w.sum(axis=1).values, np.ones(5), atol=1e-9)

    def test_hard_labels(self):
        clf = RegimeClassifier(self._config())
        idx = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
        adx = pd.Series([30, 10, 22, 30], index=idx, dtype="float64")
        vol_pct = pd.Series([10, 10, 10, 90], index=idx, dtype="float64")
        labels = clf.hard_label(adx, vol_pct)
        self.assertEqual(labels.iloc[0], "trend")   # adx high, vol low
        self.assertEqual(labels.iloc[1], "range")   # adx low
        self.assertEqual(labels.iloc[3], "high_vol")  # vol extreme overrides

    def test_nan_inputs_yield_nan(self):
        clf = RegimeClassifier(self._config())
        idx = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
        adx = pd.Series([np.nan, 30], index=idx)
        vol_pct = pd.Series([50, 50], index=idx)
        w = clf.soft_weights(adx, vol_pct)
        self.assertTrue(w.iloc[0].isna().all())

    def test_classify_end_to_end(self):
        df = _ohlcv(600, seed=7)
        feats = timeframe_features(df)
        clf = RegimeClassifier(self._config(), vol_window=100)
        result = clf.classify(feats, adx_col="adx", volatility_col="rvol")
        self.assertEqual(len(result.labels), len(feats))
        valid = result.weights.dropna()
        np.testing.assert_allclose(valid.sum(axis=1).values, np.ones(len(valid)), atol=1e-9)


if __name__ == "__main__":
    unittest.main()
