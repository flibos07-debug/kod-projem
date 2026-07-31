"""Per-regime probability calibration.

A model's raw scores are rarely well-calibrated, and mis-calibration often
differs by market regime. This calibrator fits a separate isotonic or sigmoid
(Platt) mapping per regime, falling back to a global mapping for regimes with
fewer than ``min_samples`` observations (or unseen regimes at inference).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ..logging_utils import get_logger

logger = get_logger(__name__)


def _fit_one(method: str, prob: np.ndarray, y: np.ndarray) -> Any:
    prob = np.asarray(prob, dtype="float64")
    y = np.asarray(y)
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(prob, y)
        return ("isotonic", model)
    if method == "sigmoid":
        # Platt scaling: logistic regression on the raw probability.
        model = LogisticRegression(max_iter=1000)
        model.fit(prob.reshape(-1, 1), y)
        return ("sigmoid", model)
    raise ValueError(f"unknown calibration method {method!r}")


def _predict_one(fitted: Any, prob: np.ndarray) -> np.ndarray:
    kind, model = fitted
    prob = np.asarray(prob, dtype="float64")
    if kind == "isotonic":
        return np.clip(model.predict(prob), 0.0, 1.0)
    return model.predict_proba(prob.reshape(-1, 1))[:, 1]


class PerRegimeCalibrator:
    def __init__(self, method: str = "isotonic", *, per_regime: bool = True, min_samples: int = 500) -> None:
        if method not in ("isotonic", "sigmoid"):
            raise ValueError(f"unknown calibration method {method!r}")
        self.method = method
        self.per_regime = per_regime
        self.min_samples = min_samples
        self.global_: Any | None = None
        self.by_regime_: dict[Any, Any] = {}

    def fit(self, probability: np.ndarray, y: np.ndarray, regimes: Any | None = None) -> "PerRegimeCalibrator":
        prob = np.asarray(probability, dtype="float64")
        y = np.asarray(y)
        if len(np.unique(y)) < 2:
            logger.warning("Calibration target has a single class; using identity fallback")
            self.global_ = None
            return self

        self.global_ = _fit_one(self.method, prob, y)
        self.by_regime_ = {}

        if self.per_regime and regimes is not None:
            regimes = np.asarray(regimes)
            for reg in np.unique(regimes[regimes != None]):  # noqa: E711 - keep object None out
                mask = regimes == reg
                if mask.sum() >= self.min_samples and len(np.unique(y[mask])) >= 2:
                    self.by_regime_[reg] = _fit_one(self.method, prob[mask], y[mask])
                else:
                    logger.debug("Regime %r has too few samples (%d) for its own calibrator", reg, int(mask.sum()))
        return self

    def transform(self, probability: np.ndarray, regimes: Any | None = None) -> np.ndarray:
        prob = np.asarray(probability, dtype="float64")
        if self.global_ is None:
            return prob  # identity fallback

        out = _predict_one(self.global_, prob)
        if self.per_regime and regimes is not None and self.by_regime_:
            regimes = np.asarray(regimes)
            for reg, fitted in self.by_regime_.items():
                mask = regimes == reg
                if mask.any():
                    out[mask] = _predict_one(fitted, prob[mask])
        return out

    def fit_transform(self, probability: np.ndarray, y: np.ndarray, regimes: Any | None = None) -> np.ndarray:
        return self.fit(probability, y, regimes).transform(probability, regimes)
