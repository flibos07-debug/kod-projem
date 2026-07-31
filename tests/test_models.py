"""Tests for base models, stacking ensemble and per-regime calibration."""

import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.base_models import make_base_models, make_meta_model  # noqa: E402
from src.models.calibration import PerRegimeCalibrator  # noqa: E402
from src.models.stacking import StackingEnsemble, blend_probability_ranking  # noqa: E402


def _dataset(n=600, seed=0):
    X, y = make_classification(
        n_samples=n,
        n_features=12,
        n_informative=6,
        n_redundant=2,
        random_state=seed,
        class_sep=1.0,
    )
    return X.astype("float64"), y.astype(int)


class BaseModelTests(unittest.TestCase):
    def test_factory_returns_diverse_models(self):
        models = make_base_models(fast=True)
        self.assertIn("logreg", models)
        self.assertIn("random_forest", models)
        self.assertGreaterEqual(len(models), 3)

    def test_meta_model_unknown_raises(self):
        with self.assertRaises(ValueError):
            make_meta_model("does_not_exist")


class StackingTests(unittest.TestCase):
    def test_fit_predict_shapes_and_range(self):
        X, y = _dataset()
        ens = StackingEnsemble(fast=True).fit(X, y, cv=4)
        proba = ens.predict_proba_positive(X)
        self.assertEqual(proba.shape, (len(X),))
        self.assertTrue(((proba >= 0) & (proba <= 1)).all())

    def test_stacking_beats_random(self):
        X, y = _dataset(seed=1)
        n_train = 450
        ens = StackingEnsemble(fast=True).fit(X[:n_train], y[:n_train], cv=4)
        proba = ens.predict_proba_positive(X[n_train:])
        acc = ((proba >= 0.5).astype(int) == y[n_train:]).mean()
        self.assertGreater(acc, 0.6)

    def test_averaging_mode_when_stacking_off(self):
        X, y = _dataset()
        ens = StackingEnsemble(fast=True, use_stacking=False).fit(X, y, cv=3)
        self.assertIsNone(ens.fitted_meta_)
        proba = ens.predict_proba_positive(X)
        self.assertTrue(((proba >= 0) & (proba <= 1)).all())

    def test_predict_before_fit_raises(self):
        with self.assertRaises(RuntimeError):
            StackingEnsemble(fast=True).predict_proba_positive(np.zeros((3, 12)))

    def test_custom_cv_splits(self):
        X, y = _dataset()
        splits = [(np.arange(0, 300), np.arange(300, 400)), (np.arange(0, 400), np.arange(400, 500))]
        ens = StackingEnsemble(fast=True).fit(X, y, cv=splits)
        self.assertIsNotNone(ens.fitted_base_)


class BlendTests(unittest.TestCase):
    def test_blend_renormalises_weights(self):
        prob = np.array([0.2, 0.8])
        rank = np.array([1.0, 0.0])
        out = blend_probability_ranking(prob, rank, probability_weight=0.75, ranking_weight=0.25)
        np.testing.assert_allclose(out, [0.75 * 0.2 + 0.25 * 1.0, 0.75 * 0.8 + 0.25 * 0.0])

    def test_blend_zero_weights_raise(self):
        with self.assertRaises(ValueError):
            blend_probability_ranking(np.array([0.5]), np.array([0.5]), probability_weight=0, ranking_weight=0)


class CalibrationTests(unittest.TestCase):
    def test_isotonic_improves_or_preserves_range(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 1000)
        # Distorted probabilities (squashed).
        raw = np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.2, 1000), 0, 1)
        cal = PerRegimeCalibrator(method="isotonic", per_regime=False)
        out = cal.fit_transform(raw, y)
        self.assertTrue(((out >= 0) & (out <= 1)).all())

    def test_per_regime_falls_back_when_too_few_samples(self):
        rng = np.random.default_rng(2)
        n = 800
        y = rng.integers(0, 2, n)
        raw = np.clip(0.5 + 0.2 * (y - 0.5) + rng.normal(0, 0.2, n), 0, 1)
        regimes = np.array(["trend"] * 700 + ["range"] * 100)  # 'range' < min_samples
        cal = PerRegimeCalibrator(method="sigmoid", per_regime=True, min_samples=200)
        cal.fit(raw, y, regimes)
        self.assertIn("trend", cal.by_regime_)
        self.assertNotIn("range", cal.by_regime_)  # too few -> global fallback
        out = cal.transform(raw, regimes)
        self.assertEqual(out.shape, raw.shape)

    def test_single_class_identity_fallback(self):
        raw = np.array([0.2, 0.4, 0.6])
        y = np.array([1, 1, 1])
        cal = PerRegimeCalibrator(method="isotonic")
        out = cal.fit_transform(raw, y)
        np.testing.assert_allclose(out, raw)


if __name__ == "__main__":
    unittest.main()
