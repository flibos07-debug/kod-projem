"""Tests for the quality gate, cross-sectional ranking and backtest engine."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_schema import BacktestConfig, QualityGateConfig  # noqa: E402
from src.backtest.engine import BacktestEngine, max_drawdown  # noqa: E402
from src.quality.gate import (  # noqa: E402
    QualityGate,
    brier_score,
    expected_calibration_error,
    top_k_precision,
)
from src.ranking.cross_sectional import cross_sectional_rank, select_top_n  # noqa: E402


class QualityMetricTests(unittest.TestCase):
    def test_brier_perfect_is_zero(self):
        y = np.array([0, 1, 1, 0])
        self.assertAlmostEqual(brier_score(y.astype(float), y), 0.0)

    def test_ece_of_calibrated_is_low(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, 20000)
        y = (rng.uniform(0, 1, 20000) < p).astype(int)  # perfectly calibrated
        self.assertLess(expected_calibration_error(p, y), 0.03)

    def test_top_k_precision(self):
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        y = np.array([1, 1, 0, 0])
        self.assertEqual(top_k_precision(scores, y, 2), 1.0)


class QualityGateTests(unittest.TestCase):
    def _cfg(self, **kw):
        base = dict(
            min_brier_improvement_vs_baseline=0.005, max_ece=0.05,
            min_top5_precision_vs_event=1.30, min_feature_stability=0.70,
            min_oos_folds=3, require_all_pass=True,
        )
        base.update(kw)
        return QualityGateConfig(**base)

    def test_all_pass(self):
        gate = QualityGate(self._cfg())
        report = gate.evaluate(
            brier=0.18, brier_baseline=0.20, ece=0.03,
            top5_precision=0.65, event_base_rate=0.45,
            feature_stability=0.80, n_oos_folds=4,
        )
        self.assertTrue(report.passed)
        self.assertTrue(all(c.passed for c in report.criteria))

    def test_require_all_pass_fails_on_one(self):
        gate = QualityGate(self._cfg(require_all_pass=True))
        report = gate.evaluate(
            brier=0.199, brier_baseline=0.20, ece=0.03,  # tiny improvement -> fail
            top5_precision=0.65, event_base_rate=0.45,
            feature_stability=0.80, n_oos_folds=4,
        )
        self.assertFalse(report.passed)

    def test_tolerant_mode_allows_single_miss(self):
        gate = QualityGate(self._cfg(require_all_pass=False))
        report = gate.evaluate(
            brier=0.199, brier_baseline=0.20, ece=0.03,  # one miss
            top5_precision=0.65, event_base_rate=0.45,
            feature_stability=0.80, n_oos_folds=4,
        )
        self.assertTrue(report.passed)


class RankingTests(unittest.TestCase):
    def _long_df(self):
        rows = []
        for t in pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC"):
            for sym, score in [("A", 0.9), ("B", 0.5), ("C", 0.1), ("D", 0.7)]:
                rows.append({"time": t, "symbol": sym, "score": score})
        return pd.DataFrame(rows)

    def test_rank_and_percentile(self):
        ranked = cross_sectional_rank(self._long_df())
        first_ts = ranked["time"].iloc[0]
        block = ranked[ranked["time"] == first_ts].set_index("symbol")
        self.assertEqual(block.loc["A", "rank"], 1)  # highest score
        self.assertEqual(block.loc["C", "rank"], 4)  # lowest
        self.assertAlmostEqual(block.loc["A", "percentile"], 1.0)

    def test_select_top_n_per_timestamp(self):
        top2 = select_top_n(self._long_df(), 2)
        counts = top2.groupby("time").size()
        self.assertTrue((counts == 2).all())
        # Top-2 per timestamp are A(0.9) and D(0.7).
        first = top2[top2["time"] == top2["time"].iloc[0]]
        self.assertEqual(set(first["symbol"]), {"A", "D"})

    def test_select_top_n_per_direction(self):
        df = self._long_df()
        df["direction"] = ["long", "long", "short", "short"] * 3
        top1 = select_top_n(df, 1, direction_col="direction")
        counts = top1.groupby(["time", "direction"]).size()
        self.assertTrue((counts == 1).all())


class BacktestTests(unittest.TestCase):
    def _cfg(self, **kw):
        base = dict(
            assumed_taker_fee_percent_per_side=0.04,
            assumed_slippage_percent_per_side=0.02,
            max_positions_per_direction=5, prevent_overlapping=True,
            monte_carlo_iterations=500,
        )
        base.update(kw)
        return BacktestConfig(**base)

    def _trades(self, rets, freq_h=2):
        times = pd.date_range("2024-01-01", periods=len(rets), freq=f"{freq_h}h", tz="UTC")
        return pd.DataFrame(
            {
                "entry_time": times,
                "exit_time": times + pd.Timedelta(hours=1),
                "ret": rets,
                "direction": ["long"] * len(rets),
            }
        )

    def test_max_drawdown(self):
        equity = np.array([1.0, 1.2, 0.9, 1.1])
        # Peak 1.2 -> trough 0.9 => 0.25 drawdown.
        self.assertAlmostEqual(max_drawdown(equity), 0.25)

    def test_costs_reduce_returns(self):
        eng = BacktestEngine(self._cfg())
        trades = self._trades([0.0, 0.0, 0.0])  # zero gross
        result = eng.run(trades)
        # Every trade loses exactly the round-trip cost.
        self.assertTrue((result.net_returns < 0).all())
        self.assertAlmostEqual(result.net_returns[0], -eng.round_trip_cost)

    def test_concurrency_cap_limits_trades(self):
        # 10 overlapping trades, cap of 2 -> at most 2 accepted concurrently.
        times = pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC")
        trades = pd.DataFrame(
            {
                "entry_time": times,
                "exit_time": times + pd.Timedelta(hours=1),  # all overlap
                "ret": [0.05] * 10,
                "direction": ["long"] * 10,
            }
        )
        eng = BacktestEngine(self._cfg(max_positions_per_direction=2))
        result = eng.run(trades)
        self.assertEqual(result.stats["n_trades"], 2.0)

    def test_no_overlap_prevention_accepts_all(self):
        times = pd.date_range("2024-01-01", periods=10, freq="1min", tz="UTC")
        trades = pd.DataFrame(
            {
                "entry_time": times,
                "exit_time": times + pd.Timedelta(hours=1),
                "ret": [0.05] * 10,
                "direction": ["long"] * 10,
            }
        )
        eng = BacktestEngine(self._cfg(prevent_overlapping=False, max_positions_per_direction=2))
        result = eng.run(trades)
        self.assertEqual(result.stats["n_trades"], 10.0)

    def test_monte_carlo_distribution(self):
        eng = BacktestEngine(self._cfg(monte_carlo_iterations=1000))
        rng = np.random.default_rng(0)
        rets = rng.normal(0.01, 0.05, 200)  # positive-edge trades
        result = eng.run(self._trades(rets, freq_h=3))
        mc = eng.monte_carlo(result.net_returns, position_fraction=result.stats["position_fraction"])
        self.assertEqual(mc.iterations, 1000)
        self.assertLess(mc.final_return["p5"], mc.final_return["p95"])
        self.assertTrue(0.0 <= mc.prob_negative <= 1.0)

    def test_empty_trades(self):
        eng = BacktestEngine(self._cfg())
        empty = pd.DataFrame(columns=["entry_time", "exit_time", "ret"])
        result = eng.run(empty)
        self.assertEqual(result.stats["n_trades"], 0.0)


if __name__ == "__main__":
    unittest.main()
