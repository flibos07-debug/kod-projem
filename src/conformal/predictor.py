"""Label-conditional (Mondrian) split-conformal prediction.

Given probabilities from a fitted model and a held-out **calibration** set, this
produces prediction *sets* with a finite-sample coverage guarantee: at level
``alpha`` the true label is contained in the set with probability ``>= 1 - alpha``.
Coverage is conditioned on the true class (Mondrian), which keeps guarantees
meaningful under class imbalance.

Nonconformity for the positive class is ``1 - p1`` and for the negative class
``p1``. A label is included in the prediction set when its nonconformity does
not exceed the calibrated ``(1 - alpha)`` quantile for that class.

For signal generation the useful states are the **singletons**: a set equal to
``{1}`` is a confident positive, ``{0}`` a confident negative; ``{0, 1}`` is
abstention and ``{}`` (empty) flags an out-of-distribution score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ConformalResult:
    """Per-instance conformal output at a given alpha.

    ``include_0`` / ``include_1`` are boolean arrays; ``p_value_0`` /
    ``p_value_1`` are the conformal p-values for each label.
    """

    alpha: float
    include_0: np.ndarray
    include_1: np.ndarray
    p_value_0: np.ndarray
    p_value_1: np.ndarray

    def singleton_positive(self) -> np.ndarray:
        """Mask of confident positive predictions (set == {1})."""
        return self.include_1 & ~self.include_0

    def singleton_negative(self) -> np.ndarray:
        return self.include_0 & ~self.include_1

    def abstain(self) -> np.ndarray:
        return self.include_0 & self.include_1


class BinaryConformalPredictor:
    def __init__(self) -> None:
        self.cal_scores_0_: np.ndarray | None = None
        self.cal_scores_1_: np.ndarray | None = None

    def fit(self, cal_probability: np.ndarray, cal_y: np.ndarray) -> "BinaryConformalPredictor":
        """Calibrate on held-out positive-class probabilities and labels."""
        p1 = np.asarray(cal_probability, dtype="float64")
        y = np.asarray(cal_y).astype(int)
        if p1.shape != y.shape:
            raise ValueError("probability and label arrays must have the same shape")

        # Nonconformity of the *true* label for each calibration point.
        self.cal_scores_1_ = (1.0 - p1)[y == 1]
        self.cal_scores_0_ = p1[y == 0]
        if len(self.cal_scores_1_) == 0 or len(self.cal_scores_0_) == 0:
            logger.warning("Calibration set is missing one of the classes; coverage may be void")
        return self

    @staticmethod
    def _p_value(cal_scores: np.ndarray, test_score: np.ndarray) -> np.ndarray:
        """Conformal p-value: fraction of calibration scores >= test score."""
        n = len(cal_scores)
        if n == 0:
            return np.ones_like(test_score)
        # (#{cal >= test} + 1) / (n + 1)
        ge = (cal_scores[None, :] >= test_score[:, None]).sum(axis=1)
        return (ge + 1.0) / (n + 1.0)

    def predict(self, probability: np.ndarray, alpha: float) -> ConformalResult:
        if self.cal_scores_0_ is None or self.cal_scores_1_ is None:
            raise RuntimeError("BinaryConformalPredictor is not fitted")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")

        p1 = np.asarray(probability, dtype="float64")
        score_1 = 1.0 - p1
        score_0 = p1

        pval_1 = self._p_value(self.cal_scores_1_, score_1)
        pval_0 = self._p_value(self.cal_scores_0_, score_0)

        # Include a label when its conformal p-value exceeds alpha.
        include_1 = pval_1 > alpha
        include_0 = pval_0 > alpha
        return ConformalResult(
            alpha=alpha,
            include_0=include_0,
            include_1=include_1,
            p_value_0=pval_0,
            p_value_1=pval_1,
        )

    def coverage(self, probability: np.ndarray, y: np.ndarray, alpha: float) -> float:
        """Empirical coverage: fraction of instances whose true label is in the set."""
        result = self.predict(probability, alpha)
        y = np.asarray(y).astype(int)
        covered = np.where(y == 1, result.include_1, result.include_0)
        return float(covered.mean())
