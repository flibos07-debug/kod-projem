"""Tests for the configuration schema and its validation rules."""

import copy
import sys
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_schema import AppConfig, ConfigError, load_default  # noqa: E402

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def _base_dict() -> dict:
    """A fresh, valid config dict loaded from the default YAML."""
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))


class LoadDefaultTests(unittest.TestCase):
    def test_default_config_loads_and_validates(self):
        config = load_default()
        self.assertIsInstance(config, AppConfig)

    def test_default_values_are_parsed(self):
        config = load_default()
        self.assertEqual(config.regime.classes, ("trend", "range", "high_vol"))
        self.assertEqual(config.ensemble.meta_model, "logistic_regression")
        self.assertAlmostEqual(
            config.ensemble.ranking_weight + config.ensemble.probability_weight, 1.0
        )
        self.assertEqual(config.calibration.method, "isotonic")
        self.assertTrue(config.conformal.use_conformal)
        self.assertEqual(config.storage.retention_days_scans, 90)

    def test_config_is_immutable(self):
        config = load_default()
        with self.assertRaises(Exception):
            config.regime.routing_temperature = 2.0  # type: ignore[misc]


class ValidationRuleTests(unittest.TestCase):
    def _assert_invalid(self, data: dict):
        with self.assertRaises(ConfigError):
            AppConfig.from_dict(data)

    def test_base_dict_is_valid(self):
        # Sanity check that the helper produces a config the parser accepts.
        AppConfig.from_dict(_base_dict())

    def test_missing_section_rejected(self):
        data = _base_dict()
        del data["ensemble"]
        self._assert_invalid(data)

    def test_missing_key_rejected(self):
        data = _base_dict()
        del data["regime"]["adx_threshold_trend"]
        self._assert_invalid(data)

    def test_unknown_key_rejected(self):
        data = _base_dict()
        data["regime"]["surprise"] = 1
        self._assert_invalid(data)

    def test_unknown_section_rejected(self):
        data = _base_dict()
        data["extra_section"] = {}
        self._assert_invalid(data)

    def test_ensemble_weights_must_sum_to_one(self):
        data = _base_dict()
        data["ensemble"]["ranking_weight"] = 0.5  # probability_weight still 0.75
        self._assert_invalid(data)

    def test_conformal_alpha_ordering_enforced(self):
        data = _base_dict()
        data["conformal"]["alpha_90"] = 0.30  # must be < alpha_80 (0.20)
        self._assert_invalid(data)

    def test_drift_psi_ordering_enforced(self):
        data = _base_dict()
        data["drift"]["psi_threshold_medium"] = 0.30  # must be < high (0.25)
        self._assert_invalid(data)

    def test_adx_threshold_ordering_enforced(self):
        data = _base_dict()
        data["regime"]["adx_threshold_range"] = 30  # must be <= trend (25)
        self._assert_invalid(data)

    def test_calibration_method_whitelist(self):
        data = _base_dict()
        data["calibration"]["method"] = "linear"
        self._assert_invalid(data)

    def test_bool_not_accepted_for_numeric(self):
        data = _base_dict()
        data["validation"]["train_months"] = True
        self._assert_invalid(data)

    def test_negative_purge_bars_rejected(self):
        data = _base_dict()
        data["validation"]["purge_bars"] = -1
        self._assert_invalid(data)

    def test_routing_temperature_must_be_positive(self):
        data = _base_dict()
        data["regime"]["routing_temperature"] = 0
        self._assert_invalid(data)

    def test_max_ece_out_of_range_rejected(self):
        data = _base_dict()
        data["quality_gate"]["max_ece"] = 1.5
        self._assert_invalid(data)

    def test_duplicate_regime_classes_rejected(self):
        data = _base_dict()
        data["regime"]["classes"] = ["trend", "trend"]
        self._assert_invalid(data)

    def test_deep_copy_does_not_leak_between_cases(self):
        data = _base_dict()
        mutated = copy.deepcopy(data)
        mutated["ensemble"]["ranking_weight"] = 0.9
        # Original still valid; mutated invalid.
        AppConfig.from_dict(data)
        self._assert_invalid(mutated)


if __name__ == "__main__":
    unittest.main()
