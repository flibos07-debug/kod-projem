"""Tests for timing, reporting, the live scanner and the training pipeline."""

import dataclasses
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_schema import AppConfig, ValidationConfig  # noqa: E402
from src.conformal.predictor import BinaryConformalPredictor  # noqa: E402
from src.data.resample import resample_to_timeframes  # noqa: E402
from src.features.engineering import FeatureParams, build_feature_matrix  # noqa: E402
from src.models.calibration import PerRegimeCalibrator  # noqa: E402
from src.models.stacking import StackingEnsemble  # noqa: E402
from src.pipeline.artifacts import ModelArtifact, load_artifact, save_artifact  # noqa: E402
from src.reporting.report import cleanup_old_reports, render_table, save_scan  # noqa: E402
from src.scanner.live_scanner import LiveScanner  # noqa: E402
from src.utils.timing import floor_to_interval, next_boundary, seconds_until  # noqa: E402

DEFAULT_CONFIG = AppConfig.load(Path(__file__).resolve().parent.parent / "config" / "default.yaml")


def _ohlcv(n, start="2024-01-01", freq="5min", seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    high = close + rng.uniform(0, 0.5, n)
    low = close - rng.uniform(0, 0.5, n)
    open_ = close - rng.normal(0, 0.2, n)
    volume = rng.uniform(10, 100, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


class TimingTests(unittest.TestCase):
    def test_floor_to_5m(self):
        t = datetime(2024, 1, 1, 12, 37, 45, tzinfo=timezone.utc)
        self.assertEqual(floor_to_interval(t, "5m"), datetime(2024, 1, 1, 12, 35, tzinfo=timezone.utc))

    def test_next_boundary_strictly_after(self):
        t = datetime(2024, 1, 1, 12, 35, 0, tzinfo=timezone.utc)
        self.assertEqual(next_boundary(t, "5m"), datetime(2024, 1, 1, 12, 40, tzinfo=timezone.utc))

    def test_seconds_until_non_negative(self):
        now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        past = datetime(2024, 1, 1, 11, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(seconds_until(past, now=now), 0.0)
        future = datetime(2024, 1, 1, 12, 0, 30, tzinfo=timezone.utc)
        self.assertEqual(seconds_until(future, now=now), 30.0)

    def test_naive_treated_as_utc(self):
        t = datetime(2024, 1, 1, 12, 37)
        self.assertEqual(floor_to_interval(t, "15m"), datetime(2024, 1, 1, 12, 30, tzinfo=timezone.utc))


class ReportingTests(unittest.TestCase):
    def test_render_table(self):
        df = pd.DataFrame({"symbol": ["BTC", "ETH"], "score": [0.9123, 0.5]})
        table = render_table(df)
        self.assertIn("symbol", table)
        self.assertIn("0.9123", table)

    def test_render_empty(self):
        self.assertEqual(render_table(pd.DataFrame()), "(no rows)")

    def test_save_and_cleanup(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            df = pd.DataFrame({"a": [1, 2]})
            old = datetime(2020, 1, 1, tzinfo=timezone.utc)
            save_scan(df, d, timestamp=old)
            recent = datetime.now(timezone.utc)
            save_scan(df, d, timestamp=recent)
            # Prune anything older than 90 days -> the 2020 file goes.
            removed = cleanup_old_reports(d, 90, now=recent)
            self.assertEqual(removed, 1)
            self.assertEqual(len(list(Path(d).glob("scan_*.csv"))), 1)


def _make_small_artifact(seed=0):
    base = _ohlcv(2500, seed=seed)
    frames = resample_to_timeframes(base, ["5m", "15m"], base_timeframe="5m", drop_incomplete=True)
    matrix = build_feature_matrix(frames, base_timeframe="5m", htf_timeframes=["15m"], dropna=True)
    feats = ["5m_rsi", "5m_adx", "5m_mom_1", "15m_rsi"]
    X = matrix[feats]
    # Label: next-bar up move (toy target just to fit the estimators).
    fwd = matrix["5m_ret"].shift(-1)
    y = (fwd > 0).astype(int).loc[X.index]
    valid = y.notna()
    X, y = X[valid], y[valid]

    ens = StackingEnsemble(fast=True).fit(X, y, cv=3)
    raw = ens.predict_proba_positive(X)
    regimes = np.array(["trend"] * len(X))
    cal = PerRegimeCalibrator(method="isotonic", per_regime=False).fit(raw, y.to_numpy(), regimes)
    conf = BinaryConformalPredictor().fit(cal.transform(raw), y.to_numpy())

    return ModelArtifact(
        ensemble=ens,
        calibrator=cal,
        conformal=conf,
        feature_names=feats,
        feature_params=FeatureParams(),
        metadata={"base_timeframe": "5m", "htf_timeframes": ["15m"], "alpha_90": 0.10, "alpha_80": 0.20},
    )


class FakeClient:
    def __init__(self, base_df):
        self._df = base_df

    def get_klines(self, symbol, interval, *, limit=1500, **kw):
        return self._df.tail(limit)


class ScannerTests(unittest.TestCase):
    def test_score_symbol_and_scan_once(self):
        artifact = _make_small_artifact()
        base = _ohlcv(2500, seed=1)
        scanner = LiveScanner(DEFAULT_CONFIG, {"BTCUSDT": artifact}, FakeClient(base))
        results = scanner.scan_once()
        self.assertEqual(len(results), 1)
        row = results.iloc[0]
        self.assertEqual(row["symbol"], "BTCUSDT")
        self.assertTrue(0.0 <= row["prob"] <= 1.0)
        self.assertIn(row["set90"], {"{0}", "{1}", "{0,1}", "{}"})
        self.assertIn("selected", results.columns)

    def test_run_loop_bounded(self):
        artifact = _make_small_artifact()
        base = _ohlcv(2500, seed=2)
        scanner = LiveScanner(DEFAULT_CONFIG, {"BTCUSDT": artifact}, FakeClient(base))
        slept = []
        scanner.run(
            max_iterations=2,
            sleep_fn=lambda s: slept.append(s),
            now_fn=lambda: datetime(2024, 1, 1, 12, 3, 0, tzinfo=timezone.utc),
            persist=False,
        )
        self.assertEqual(len(slept), 2)
        # 12:03 -> next 5m boundary 12:05 => ~120s + buffer.
        self.assertGreater(slept[0], 100)

    def test_artifact_roundtrip(self):
        import tempfile

        artifact = _make_small_artifact()
        with tempfile.TemporaryDirectory() as d:
            path = save_artifact(artifact, Path(d) / "BTCUSDT.joblib")
            loaded = load_artifact(path)
            self.assertEqual(loaded.feature_names, artifact.feature_names)


class PipelineSmokeTest(unittest.TestCase):
    def test_pipeline_end_to_end_fast(self):
        from src.pipeline.training import PipelineParams, TrainingPipeline

        # ~5 months of 1h bars keeps the walk-forward windows satisfiable while
        # staying fast; shrink the validation windows to match.
        base = _ohlcv(24 * 30 * 5, freq="1h", seed=7)
        frames = resample_to_timeframes(base, ["1h", "4h"], base_timeframe="1h", drop_incomplete=True)

        small_validation = ValidationConfig(
            train_months=2, validation_months=1, test_months=1, step_months=1,
            purge_bars=6, embargo_bars=6, n_folds_nested=2,
        )
        config = dataclasses.replace(DEFAULT_CONFIG, validation=small_validation)
        params = PipelineParams(base_timeframe="1h", htf_timeframes=("4h",))
        pipeline = TrainingPipeline(config, params, fast=True)

        result = pipeline.run(frames, do_selection=False, symbol="TEST")
        self.assertGreaterEqual(result.n_folds, 1)
        self.assertIn("brier", result.metrics)
        # Calibrated probabilities are usable.
        art = result.artifact
        self.assertTrue(len(art.feature_names) > 0)
        self.assertIsNotNone(art.conformal)


if __name__ == "__main__":
    unittest.main()
