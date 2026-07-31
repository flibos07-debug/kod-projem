"""Tests for conformal prediction, drift detection and stability selection."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_schema import DriftConfig  # noqa: E402
from src.conformal.predictor import BinaryConformalPredictor  # noqa: E402
from src.drift.detector import (  # noqa: E402
    DriftDetector,
    jensen_shannon_divergence,
    population_stability_index,
)
from src.selection.stability import StabilitySelector  # noqa: E402


class ConformalTests(unittest.TestCase):
    def _make(self, n=4000, seed=0):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, n)
        # Informative but noisy probabilities.
        p1 = np.clip(0.5 + 0.25 * (y - 0.5) + rng.normal(0, 0.2, n), 1e-3, 1 - 1e-3)
        return p1, y

    def test_coverage_meets_guarantee(self):
        p1, y = self._make()
        cal_p, cal_y = p1[:2000], y[:2000]
        test_p, test_y = p1[2000:], y[2000:]
        cp = BinaryConformalPredictor().fit(cal_p, cal_y)
        for alpha in (0.10, 0.20):
            cov = cp.coverage(test_p, test_y, alpha)
            # Allow a small finite-sample slack below the 1-alpha target.
            self.assertGreaterEqual(cov, (1 - alpha) - 0.05)

    def test_tighter_alpha_gives_larger_sets(self):
        p1, y = self._make(seed=3)
        cp = BinaryConformalPredictor().fit(p1[:2000], y[:2000])
        r90 = cp.predict(p1[2000:], alpha=0.10)
        r80 = cp.predict(p1[2000:], alpha=0.20)
        size90 = r90.include_0.astype(int) + r90.include_1.astype(int)
        size80 = r80.include_0.astype(int) + r80.include_1.astype(int)
        self.assertGreaterEqual(size90.mean(), size80.mean())

    def test_singletons_and_abstain_partition(self):
        p1, y = self._make(seed=5)
        cp = BinaryConformalPredictor().fit(p1[:2000], y[:2000])
        r = cp.predict(p1[2000:], alpha=0.10)
        # Every instance is exactly one of: singleton+, singleton-, abstain, empty.
        total = (
            r.singleton_positive().astype(int)
            + r.singleton_negative().astype(int)
            + r.abstain().astype(int)
            + (~r.include_0 & ~r.include_1).astype(int)
        )
        np.testing.assert_array_equal(total, np.ones_like(total))

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            BinaryConformalPredictor().predict(np.array([0.5]), 0.1)


class DriftTests(unittest.TestCase):
    def test_psi_zero_for_identical(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 5000)
        self.assertAlmostEqual(population_stability_index(x, x), 0.0, places=3)

    def test_psi_grows_with_shift(self):
        rng = np.random.default_rng(1)
        ref = rng.normal(0, 1, 5000)
        small = rng.normal(0.2, 1, 5000)
        large = rng.normal(2.0, 1, 5000)
        self.assertLess(
            population_stability_index(ref, small),
            population_stability_index(ref, large),
        )

    def test_js_bounded(self):
        rng = np.random.default_rng(2)
        ref = rng.normal(0, 1, 3000)
        cur = rng.normal(3, 1, 3000)
        js = jensen_shannon_divergence(ref, cur)
        self.assertTrue(0.0 <= js <= 1.0)

    def test_detector_report_and_severity(self):
        rng = np.random.default_rng(3)
        n = 4000
        ref = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
        cur = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(3, 1, n)})
        cfg = DriftConfig(
            psi_threshold_high=0.25, psi_threshold_medium=0.10,
            js_threshold=0.05, check_interval_hours=24, enable_drift_detection=True,
        )
        report = DriftDetector(cfg).compare(ref, cur)
        self.assertEqual(report.index[0], "b")  # most drifted first
        self.assertEqual(report.loc["b", "severity"], "high")
        self.assertTrue(DriftDetector(cfg).any_high_drift(report))


class StabilitySelectionTests(unittest.TestCase):
    def test_selects_informative_features(self):
        rng = np.random.default_rng(0)
        n = 800
        informative = rng.normal(0, 1, (n, 3))
        noise = rng.normal(0, 1, (n, 7))
        logits = informative @ np.array([2.5, -2.0, 1.8])
        y = (logits + rng.normal(0, 0.5, n) > 0).astype(int)
        X = pd.DataFrame(
            np.hstack([informative, noise]),
            columns=[f"info_{i}" for i in range(3)] + [f"noise_{i}" for i in range(7)],
        )
        sel = StabilitySelector(n_bootstraps=40, threshold=0.6, sample_fraction=0.6).fit(X, y)
        selected = set(sel.selected_features())
        # All informative features should be selected...
        self.assertTrue({"info_0", "info_1", "info_2"} <= selected)
        # ...and most noise features dropped.
        noise_kept = [f for f in selected if f.startswith("noise")]
        self.assertLessEqual(len(noise_kept), 2)

    def test_transform_subsets_columns(self):
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.normal(0, 1, (200, 5)), columns=list("abcde"))
        y = (X["a"] + X["b"] > 0).astype(int)
        sel = StabilitySelector(n_bootstraps=20, threshold=0.5).fit(X, y)
        Xt = sel.transform(X)
        self.assertEqual(list(Xt.columns), sel.selected_features())

    def test_fit_before_support_raises(self):
        with self.assertRaises(RuntimeError):
            StabilitySelector().get_support()


if __name__ == "__main__":
    unittest.main()
