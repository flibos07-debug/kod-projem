"""Tests for triple-barrier labelling and walk-forward validation."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_schema import ValidationConfig  # noqa: E402
from src.labeling.triple_barrier import BarrierParams, binary_target, triple_barrier_labels  # noqa: E402
from src.validation.walk_forward import WalkForwardSplitter  # noqa: E402


def _frame(highs, lows, closes, start="2024-01-01", freq="5min"):
    idx = pd.date_range(start=start, periods=len(closes), freq=freq, tz="UTC")
    return pd.DataFrame(
        {"high": highs, "low": lows, "close": closes}, index=idx, dtype="float64"
    )


class TripleBarrierTests(unittest.TestCase):
    def test_take_profit_hit_first(self):
        # Entry at bar 0 (close=100), vol=1, tp_mult=2 -> tp=102, sl=99.
        closes = [100, 101, 103, 100]
        highs = [100, 101.5, 103.5, 100]  # bar 2 reaches 103.5 >= 102
        lows = [100, 100.5, 102, 99.5]
        df = _frame(highs, lows, closes)
        vol = pd.Series(1.0, index=df.index)
        out = triple_barrier_labels(
            df, vol, BarrierParams(tp_mult=2, sl_mult=1, max_holding=3),
            event_index=df.index[[0]],
        )
        self.assertEqual(out.iloc[0]["label"], 1)
        self.assertEqual(out.iloc[0]["t1"], df.index[2])
        self.assertGreater(out.iloc[0]["ret"], 0)

    def test_stop_loss_hit_first(self):
        closes = [100, 98, 100, 100]
        highs = [100, 99, 100, 100]
        lows = [100, 98, 100, 100]  # bar 1 low 98 <= sl=99
        df = _frame(highs, lows, closes)
        vol = pd.Series(1.0, index=df.index)
        out = triple_barrier_labels(
            df, vol, BarrierParams(tp_mult=2, sl_mult=1, max_holding=3),
            event_index=df.index[[0]],
        )
        self.assertEqual(out.iloc[0]["label"], -1)
        self.assertLess(out.iloc[0]["ret"], 0)

    def test_timeout_label_zero(self):
        closes = [100, 100.5, 100.2, 100.3, 100.1]
        highs = [100, 100.6, 100.4, 100.5, 100.2]  # never reaches 102
        lows = [100, 100.1, 100.0, 100.1, 100.0]   # never reaches 99
        df = _frame(highs, lows, closes)
        vol = pd.Series(1.0, index=df.index)
        out = triple_barrier_labels(
            df, vol, BarrierParams(tp_mult=2, sl_mult=1, max_holding=3),
            event_index=df.index[[0]],
        )
        self.assertEqual(out.iloc[0]["label"], 0)
        self.assertEqual(out.iloc[0]["t1"], df.index[3])  # vertical barrier

    def test_short_side_inverts(self):
        # Short: tp is below entry. entry=100, tp=98, sl=101.
        closes = [100, 97, 100]
        highs = [100, 97.5, 100]
        lows = [100, 97, 100]  # reaches 97 <= tp(98) for short
        df = _frame(highs, lows, closes)
        vol = pd.Series(1.0, index=df.index)
        out = triple_barrier_labels(
            df, vol, BarrierParams(tp_mult=2, sl_mult=1, max_holding=2, side=-1),
            event_index=df.index[[0]],
        )
        self.assertEqual(out.iloc[0]["label"], 1)  # profit for the short
        self.assertGreater(out.iloc[0]["ret"], 0)

    def test_label_uses_only_future_bars(self):
        # A spike on the entry bar itself must not trigger a barrier.
        closes = [100, 100, 100]
        highs = [500, 100, 100]  # huge high on entry bar
        lows = [100, 100, 100]
        df = _frame(highs, lows, closes)
        vol = pd.Series(1.0, index=df.index)
        out = triple_barrier_labels(
            df, vol, BarrierParams(tp_mult=2, sl_mult=1, max_holding=2),
            event_index=df.index[[0]],
        )
        self.assertEqual(out.iloc[0]["label"], 0)  # not +1 from its own bar

    def test_binary_target_mapping(self):
        labels = pd.DataFrame({"label": [1, -1, 0]})
        tgt = binary_target(labels, timeout_as_loss=True)
        self.assertEqual(list(tgt), [1, 0, 0])
        tgt2 = binary_target(labels, timeout_as_loss=False)
        self.assertTrue(np.isnan(tgt2.iloc[2]))


class WalkForwardTests(unittest.TestCase):
    def _config(self, **kw):
        base = dict(
            train_months=6, validation_months=1, test_months=1, step_months=1,
            purge_bars=12, embargo_bars=12, n_folds_nested=3,
        )
        base.update(kw)
        return ValidationConfig(**base)

    def _daily_index(self, months=12):
        return pd.date_range("2022-01-01", periods=months * 30, freq="D", tz="UTC")

    def test_chronological_order(self):
        idx = self._daily_index(14)
        splitter = WalkForwardSplitter(self._config(purge_bars=0, embargo_bars=0))
        folds = splitter.split(idx)
        self.assertTrue(len(folds) >= 1)
        for f in folds:
            train_times = idx[f.train]
            test_times = idx[f.test]
            # Every train sample precedes every test sample.
            self.assertLess(train_times.max(), test_times.min())

    def test_purge_creates_train_val_gap(self):
        idx = self._daily_index(14)
        purge = 5
        splitter = WalkForwardSplitter(self._config(purge_bars=purge, embargo_bars=0))
        folds = splitter.split(idx)
        self.assertTrue(folds)
        f = folds[0]
        # Gap between last train sample and validation start.
        val_start = f.val_span[0]
        last_train = idx[f.train].max()
        self.assertLess(last_train, val_start)

    def test_purge_by_label_overlap(self):
        idx = self._daily_index(14)
        # Give every sample a 40-day label horizon so many train labels spill
        # into the validation window and must be purged.
        t1 = pd.Series(idx + pd.Timedelta(days=40), index=idx)
        splitter = WalkForwardSplitter(self._config(purge_bars=0, embargo_bars=0))
        folds = splitter.split(idx, label_end_times=t1)
        self.assertTrue(folds)
        f = folds[0]
        val_start = f.val_span[0]
        # No training sample's label window reaches into validation.
        self.assertTrue((t1.iloc[f.train] < val_start).all())

    def test_inner_folds_present_and_ordered(self):
        idx = self._daily_index(18)
        splitter = WalkForwardSplitter(self._config(purge_bars=2, embargo_bars=2))
        folds = splitter.split(idx)
        self.assertTrue(folds)
        f = folds[0]
        self.assertEqual(len(f.inner), 3)
        for inner_train, inner_val in f.inner:
            self.assertLess(idx[inner_train].max(), idx[inner_val].min())

    def test_no_folds_when_span_too_short(self):
        idx = pd.date_range("2022-01-01", periods=20, freq="D", tz="UTC")
        splitter = WalkForwardSplitter(self._config())
        self.assertEqual(splitter.split(idx), [])

    def test_positions_index_original_array(self):
        idx = self._daily_index(14)
        splitter = WalkForwardSplitter(self._config(purge_bars=0, embargo_bars=0))
        folds = splitter.split(idx)
        f = folds[0]
        # Positions are valid indices into the original index.
        self.assertTrue(f.train.max() < len(idx))
        self.assertEqual(idx[f.test][0], f.test_span[0])


if __name__ == "__main__":
    unittest.main()
