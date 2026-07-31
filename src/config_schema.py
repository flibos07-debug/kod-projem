"""Typed, validated schema for the ML model configuration.

The configuration is grouped into logical sections (regime routing, ensemble,
walk-forward validation, calibration, conformal prediction, drift detection,
the quality gate, backtesting assumptions and storage paths). Each section is a
frozen dataclass with a ``from_dict`` factory that coerces/validates its inputs.

The module depends only on the standard library plus PyYAML for loading, so it
can be imported and tested without installing heavier ML/validation frameworks.

Example
-------
>>> from src.config_schema import AppConfig
>>> config = AppConfig.load("config/default.yaml")
>>> config.ensemble.meta_model
'logistic_regression'
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


class ConfigError(ValueError):
    """Raised when a configuration value is missing or invalid.

    The message is prefixed with the dotted path to the offending value
    (e.g. ``ensemble.ranking_weight``) to make problems easy to locate.
    """


def _require(mapping: Mapping[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"{section}.{key}: missing required key")
    return mapping[key]


def _as_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path}: expected a number, got {type(value).__name__}")
    return float(value)


def _as_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path}: expected an integer, got {type(value).__name__}")
    return value


def _as_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path}: expected a boolean, got {type(value).__name__}")
    return value


def _as_str(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{path}: expected a string, got {type(value).__name__}")
    return value


def _check_range(
    value: float,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    inclusive: bool = True,
) -> float:
    if minimum is not None and (value < minimum if inclusive else value <= minimum):
        bound = ">=" if inclusive else ">"
        raise ConfigError(f"{path}: {value} must be {bound} {minimum}")
    if maximum is not None and (value > maximum if inclusive else value >= maximum):
        bound = "<=" if inclusive else "<"
        raise ConfigError(f"{path}: {value} must be {bound} {maximum}")
    return value


def _reject_unknown_keys(mapping: Mapping[str, Any], known: Sequence[str], section: str) -> None:
    unknown = set(mapping) - set(known)
    if unknown:
        joined = ", ".join(sorted(unknown))
        raise ConfigError(f"{section}: unknown key(s): {joined}")


@dataclass(frozen=True)
class RegimeConfig:
    """Market-regime classification and routing settings."""

    classes: tuple[str, ...]
    adx_threshold_trend: float
    adx_threshold_range: float
    vol_percentile_high: float
    use_soft_routing: bool
    routing_temperature: float

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RegimeConfig":
        section = "regime"
        _reject_unknown_keys(data, [f.name for f in fields(cls)], section)

        raw_classes = _require(data, "classes", section)
        if not isinstance(raw_classes, list) or not raw_classes:
            raise ConfigError(f"{section}.classes: expected a non-empty list")
        classes = tuple(_as_str(c, f"{section}.classes[{i}]") for i, c in enumerate(raw_classes))
        if len(set(classes)) != len(classes):
            raise ConfigError(f"{section}.classes: duplicate regime names are not allowed")

        adx_trend = _check_range(
            _as_float(_require(data, "adx_threshold_trend", section), f"{section}.adx_threshold_trend"),
            f"{section}.adx_threshold_trend",
            minimum=0,
            maximum=100,
        )
        adx_range = _check_range(
            _as_float(_require(data, "adx_threshold_range", section), f"{section}.adx_threshold_range"),
            f"{section}.adx_threshold_range",
            minimum=0,
            maximum=100,
        )
        if adx_range > adx_trend:
            raise ConfigError(
                f"{section}: adx_threshold_range ({adx_range}) must be <= "
                f"adx_threshold_trend ({adx_trend})"
            )
        vol_pct = _check_range(
            _as_float(_require(data, "vol_percentile_high", section), f"{section}.vol_percentile_high"),
            f"{section}.vol_percentile_high",
            minimum=0,
            maximum=100,
        )
        temperature = _check_range(
            _as_float(_require(data, "routing_temperature", section), f"{section}.routing_temperature"),
            f"{section}.routing_temperature",
            minimum=0,
            inclusive=False,
        )

        return cls(
            classes=classes,
            adx_threshold_trend=adx_trend,
            adx_threshold_range=adx_range,
            vol_percentile_high=vol_pct,
            use_soft_routing=_as_bool(_require(data, "use_soft_routing", section), f"{section}.use_soft_routing"),
            routing_temperature=temperature,
        )


@dataclass(frozen=True)
class EnsembleConfig:
    """Stacking / ranking ensemble settings."""

    use_stacking: bool
    use_ranking: bool
    ranking_weight: float
    probability_weight: float
    meta_model: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EnsembleConfig":
        section = "ensemble"
        _reject_unknown_keys(data, [f.name for f in fields(cls)], section)

        ranking_weight = _check_range(
            _as_float(_require(data, "ranking_weight", section), f"{section}.ranking_weight"),
            f"{section}.ranking_weight",
            minimum=0,
            maximum=1,
        )
        probability_weight = _check_range(
            _as_float(_require(data, "probability_weight", section), f"{section}.probability_weight"),
            f"{section}.probability_weight",
            minimum=0,
            maximum=1,
        )
        if abs((ranking_weight + probability_weight) - 1.0) > 1e-6:
            raise ConfigError(
                f"{section}: ranking_weight ({ranking_weight}) + probability_weight "
                f"({probability_weight}) must sum to 1.0"
            )

        return cls(
            use_stacking=_as_bool(_require(data, "use_stacking", section), f"{section}.use_stacking"),
            use_ranking=_as_bool(_require(data, "use_ranking", section), f"{section}.use_ranking"),
            ranking_weight=ranking_weight,
            probability_weight=probability_weight,
            meta_model=_as_str(_require(data, "meta_model", section), f"{section}.meta_model"),
        )


@dataclass(frozen=True)
class ValidationConfig:
    """Walk-forward / nested cross-validation windowing."""

    train_months: int
    validation_months: int
    test_months: int
    step_months: int
    purge_bars: int
    embargo_bars: int
    n_folds_nested: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ValidationConfig":
        section = "validation"
        _reject_unknown_keys(data, [f.name for f in fields(cls)], section)

        values = {}
        positive = {"train_months", "validation_months", "test_months", "step_months", "n_folds_nested"}
        for name in (f.name for f in fields(cls)):
            path = f"{section}.{name}"
            val = _as_int(_require(data, name, section), path)
            minimum = 1 if name in positive else 0
            values[name] = int(_check_range(val, path, minimum=minimum))

        return cls(**values)


@dataclass(frozen=True)
class CalibrationConfig:
    """Probability calibration settings."""

    method: str
    per_regime: bool
    min_samples_calibration: int

    _ALLOWED_METHODS = ("isotonic", "sigmoid", "beta")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CalibrationConfig":
        section = "calibration"
        _reject_unknown_keys(data, [f.name for f in fields(cls)], section)

        method = _as_str(_require(data, "method", section), f"{section}.method")
        if method not in cls._ALLOWED_METHODS:
            allowed = ", ".join(cls._ALLOWED_METHODS)
            raise ConfigError(f"{section}.method: {method!r} is not one of ({allowed})")

        return cls(
            method=method,
            per_regime=_as_bool(_require(data, "per_regime", section), f"{section}.per_regime"),
            min_samples_calibration=int(
                _check_range(
                    _as_int(_require(data, "min_samples_calibration", section), f"{section}.min_samples_calibration"),
                    f"{section}.min_samples_calibration",
                    minimum=1,
                )
            ),
        )


@dataclass(frozen=True)
class ConformalConfig:
    """Conformal prediction miscoverage levels."""

    alpha_90: float
    alpha_80: float
    use_conformal: bool

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConformalConfig":
        section = "conformal"
        _reject_unknown_keys(data, [f.name for f in fields(cls)], section)

        alpha_90 = _check_range(
            _as_float(_require(data, "alpha_90", section), f"{section}.alpha_90"),
            f"{section}.alpha_90",
            minimum=0,
            maximum=1,
            inclusive=False,
        )
        alpha_80 = _check_range(
            _as_float(_require(data, "alpha_80", section), f"{section}.alpha_80"),
            f"{section}.alpha_80",
            minimum=0,
            maximum=1,
            inclusive=False,
        )
        if alpha_90 >= alpha_80:
            raise ConfigError(
                f"{section}: alpha_90 ({alpha_90}) must be < alpha_80 ({alpha_80}); "
                "a 90% interval requires smaller miscoverage than an 80% interval"
            )

        return cls(
            alpha_90=alpha_90,
            alpha_80=alpha_80,
            use_conformal=_as_bool(_require(data, "use_conformal", section), f"{section}.use_conformal"),
        )


@dataclass(frozen=True)
class DriftConfig:
    """Feature/prediction drift detection thresholds."""

    psi_threshold_high: float
    psi_threshold_medium: float
    js_threshold: float
    check_interval_hours: int
    enable_drift_detection: bool

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DriftConfig":
        section = "drift"
        _reject_unknown_keys(data, [f.name for f in fields(cls)], section)

        psi_high = _check_range(
            _as_float(_require(data, "psi_threshold_high", section), f"{section}.psi_threshold_high"),
            f"{section}.psi_threshold_high",
            minimum=0,
        )
        psi_medium = _check_range(
            _as_float(_require(data, "psi_threshold_medium", section), f"{section}.psi_threshold_medium"),
            f"{section}.psi_threshold_medium",
            minimum=0,
        )
        if psi_medium >= psi_high:
            raise ConfigError(
                f"{section}: psi_threshold_medium ({psi_medium}) must be < "
                f"psi_threshold_high ({psi_high})"
            )

        return cls(
            psi_threshold_high=psi_high,
            psi_threshold_medium=psi_medium,
            js_threshold=_check_range(
                _as_float(_require(data, "js_threshold", section), f"{section}.js_threshold"),
                f"{section}.js_threshold",
                minimum=0,
            ),
            check_interval_hours=int(
                _check_range(
                    _as_int(_require(data, "check_interval_hours", section), f"{section}.check_interval_hours"),
                    f"{section}.check_interval_hours",
                    minimum=1,
                )
            ),
            enable_drift_detection=_as_bool(
                _require(data, "enable_drift_detection", section), f"{section}.enable_drift_detection"
            ),
        )


@dataclass(frozen=True)
class QualityGateConfig:
    """Promotion gate the model must clear before deployment."""

    min_brier_improvement_vs_baseline: float
    max_ece: float
    min_top5_precision_vs_event: float
    min_feature_stability: float
    min_oos_folds: int
    require_all_pass: bool

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QualityGateConfig":
        section = "quality_gate"
        _reject_unknown_keys(data, [f.name for f in fields(cls)], section)

        return cls(
            min_brier_improvement_vs_baseline=_check_range(
                _as_float(
                    _require(data, "min_brier_improvement_vs_baseline", section),
                    f"{section}.min_brier_improvement_vs_baseline",
                ),
                f"{section}.min_brier_improvement_vs_baseline",
                minimum=0,
            ),
            max_ece=_check_range(
                _as_float(_require(data, "max_ece", section), f"{section}.max_ece"),
                f"{section}.max_ece",
                minimum=0,
                maximum=1,
            ),
            min_top5_precision_vs_event=_check_range(
                _as_float(
                    _require(data, "min_top5_precision_vs_event", section),
                    f"{section}.min_top5_precision_vs_event",
                ),
                f"{section}.min_top5_precision_vs_event",
                minimum=0,
            ),
            min_feature_stability=_check_range(
                _as_float(_require(data, "min_feature_stability", section), f"{section}.min_feature_stability"),
                f"{section}.min_feature_stability",
                minimum=0,
                maximum=1,
            ),
            min_oos_folds=int(
                _check_range(
                    _as_int(_require(data, "min_oos_folds", section), f"{section}.min_oos_folds"),
                    f"{section}.min_oos_folds",
                    minimum=1,
                )
            ),
            require_all_pass=_as_bool(_require(data, "require_all_pass", section), f"{section}.require_all_pass"),
        )


@dataclass(frozen=True)
class BacktestConfig:
    """Backtesting cost assumptions and position constraints."""

    assumed_taker_fee_percent_per_side: float
    assumed_slippage_percent_per_side: float
    max_positions_per_direction: int
    prevent_overlapping: bool
    monte_carlo_iterations: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BacktestConfig":
        section = "backtest"
        _reject_unknown_keys(data, [f.name for f in fields(cls)], section)

        return cls(
            assumed_taker_fee_percent_per_side=_check_range(
                _as_float(
                    _require(data, "assumed_taker_fee_percent_per_side", section),
                    f"{section}.assumed_taker_fee_percent_per_side",
                ),
                f"{section}.assumed_taker_fee_percent_per_side",
                minimum=0,
            ),
            assumed_slippage_percent_per_side=_check_range(
                _as_float(
                    _require(data, "assumed_slippage_percent_per_side", section),
                    f"{section}.assumed_slippage_percent_per_side",
                ),
                f"{section}.assumed_slippage_percent_per_side",
                minimum=0,
            ),
            max_positions_per_direction=int(
                _check_range(
                    _as_int(
                        _require(data, "max_positions_per_direction", section),
                        f"{section}.max_positions_per_direction",
                    ),
                    f"{section}.max_positions_per_direction",
                    minimum=1,
                )
            ),
            prevent_overlapping=_as_bool(
                _require(data, "prevent_overlapping", section), f"{section}.prevent_overlapping"
            ),
            monte_carlo_iterations=int(
                _check_range(
                    _as_int(_require(data, "monte_carlo_iterations", section), f"{section}.monte_carlo_iterations"),
                    f"{section}.monte_carlo_iterations",
                    minimum=1,
                )
            ),
        )


@dataclass(frozen=True)
class StorageConfig:
    """Filesystem layout for data, models, reports and logs."""

    data_directory: str
    model_directory: str
    report_directory: str
    logs_directory: str
    retention_days_scans: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StorageConfig":
        section = "storage"
        _reject_unknown_keys(data, [f.name for f in fields(cls)], section)

        return cls(
            data_directory=_as_str(_require(data, "data_directory", section), f"{section}.data_directory"),
            model_directory=_as_str(_require(data, "model_directory", section), f"{section}.model_directory"),
            report_directory=_as_str(_require(data, "report_directory", section), f"{section}.report_directory"),
            logs_directory=_as_str(_require(data, "logs_directory", section), f"{section}.logs_directory"),
            retention_days_scans=int(
                _check_range(
                    _as_int(_require(data, "retention_days_scans", section), f"{section}.retention_days_scans"),
                    f"{section}.retention_days_scans",
                    minimum=1,
                )
            ),
        )


@dataclass(frozen=True)
class AppConfig:
    """Top-level configuration composed of all sections."""

    regime: RegimeConfig
    ensemble: EnsembleConfig
    validation: ValidationConfig
    calibration: CalibrationConfig
    conformal: ConformalConfig
    drift: DriftConfig
    quality_gate: QualityGateConfig
    backtest: BacktestConfig
    storage: StorageConfig

    _SECTION_TYPES = {
        "regime": RegimeConfig,
        "ensemble": EnsembleConfig,
        "validation": ValidationConfig,
        "calibration": CalibrationConfig,
        "conformal": ConformalConfig,
        "drift": DriftConfig,
        "quality_gate": QualityGateConfig,
        "backtest": BacktestConfig,
        "storage": StorageConfig,
    }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AppConfig":
        if not isinstance(data, Mapping):
            raise ConfigError("top-level configuration must be a mapping")
        _reject_unknown_keys(data, list(cls._SECTION_TYPES), "<root>")

        parsed: dict[str, Any] = {}
        for name, section_cls in cls._SECTION_TYPES.items():
            if name not in data:
                raise ConfigError(f"<root>.{name}: missing required section")
            section_data = data[name]
            if not isinstance(section_data, Mapping):
                raise ConfigError(f"{name}: expected a mapping, got {type(section_data).__name__}")
            parsed[name] = section_cls.from_dict(section_data)
        return cls(**parsed)

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        """Load and validate a YAML configuration file."""
        text = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(text)
        if data is None:
            raise ConfigError(f"{path}: configuration file is empty")
        return cls.from_dict(data)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def load_default() -> AppConfig:
    """Load and validate the bundled default configuration."""
    return AppConfig.load(DEFAULT_CONFIG_PATH)


if __name__ == "__main__":
    config = load_default()
    print(f"Loaded and validated config from {DEFAULT_CONFIG_PATH}")
    for section in fields(config):
        print(f"  - {section.name}: OK")
