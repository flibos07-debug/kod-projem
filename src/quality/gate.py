"""Promotion quality gate.

A candidate model is only allowed into production if it clears every configured
criterion (when ``require_all_pass`` is set). The criteria come straight from
``QualityGateConfig``:

- **Brier improvement** over a baseline (lower Brier is better);
- **calibration** — expected calibration error below a ceiling;
- **top-5 precision lift** over the event base rate (does ranking add value?);
- **feature stability** — mean stability-selection frequency of kept features;
- **out-of-sample folds** — enough independent OOS evaluations to trust the rest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config_schema import QualityGateConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


def brier_score(probability: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error between predicted probability and outcome."""
    p = np.asarray(probability, dtype="float64")
    y = np.asarray(y, dtype="float64")
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    probability: np.ndarray, y: np.ndarray, *, n_bins: int = 10
) -> float:
    """Expected Calibration Error using equal-width probability bins."""
    p = np.asarray(probability, dtype="float64")
    y = np.asarray(y, dtype="float64")
    if len(p) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    n = len(p)
    for b in range(n_bins):
        mask = bins == b
        count = mask.sum()
        if count == 0:
            continue
        confidence = p[mask].mean()
        accuracy = y[mask].mean()
        ece += (count / n) * abs(accuracy - confidence)
    return float(ece)


def top_k_precision(scores: np.ndarray, y: np.ndarray, k: int) -> float:
    """Precision among the ``k`` highest-scoring instances."""
    scores = np.asarray(scores, dtype="float64")
    y = np.asarray(y, dtype="float64")
    if k <= 0 or len(scores) == 0:
        return float("nan")
    k = min(k, len(scores))
    top_idx = np.argsort(scores)[::-1][:k]
    return float(y[top_idx].mean())


@dataclass(frozen=True)
class GateCriterion:
    name: str
    value: float
    threshold: float
    passed: bool
    comparison: str  # ">=" or "<="


@dataclass(frozen=True)
class GateReport:
    criteria: list[GateCriterion]
    passed: bool

    def to_frame(self):  # pragma: no cover - convenience
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "criterion": c.name,
                    "value": c.value,
                    "threshold": c.threshold,
                    "comparison": c.comparison,
                    "passed": c.passed,
                }
                for c in self.criteria
            ]
        ).set_index("criterion")

    def summary(self) -> str:
        lines = [f"Quality gate: {'PASS' if self.passed else 'FAIL'}"]
        for c in self.criteria:
            mark = "ok" if c.passed else "XX"
            lines.append(f"  [{mark}] {c.name}: {c.value:.4f} {c.comparison} {c.threshold:.4f}")
        return "\n".join(lines)


class QualityGate:
    def __init__(self, config: QualityGateConfig) -> None:
        self.config = config

    def evaluate(
        self,
        *,
        brier: float,
        brier_baseline: float,
        ece: float,
        top5_precision: float,
        event_base_rate: float,
        feature_stability: float,
        n_oos_folds: int,
    ) -> GateReport:
        cfg = self.config
        brier_improvement = brier_baseline - brier
        top5_lift = top5_precision / event_base_rate if event_base_rate > 0 else float("nan")

        criteria = [
            GateCriterion(
                "brier_improvement_vs_baseline", brier_improvement,
                cfg.min_brier_improvement_vs_baseline,
                brier_improvement >= cfg.min_brier_improvement_vs_baseline, ">=",
            ),
            GateCriterion(
                "ece", ece, cfg.max_ece, ece <= cfg.max_ece, "<=",
            ),
            GateCriterion(
                "top5_precision_vs_event", top5_lift, cfg.min_top5_precision_vs_event,
                top5_lift >= cfg.min_top5_precision_vs_event, ">=",
            ),
            GateCriterion(
                "feature_stability", feature_stability, cfg.min_feature_stability,
                feature_stability >= cfg.min_feature_stability, ">=",
            ),
            GateCriterion(
                "oos_folds", float(n_oos_folds), float(cfg.min_oos_folds),
                n_oos_folds >= cfg.min_oos_folds, ">=",
            ),
        ]

        n_pass = sum(c.passed for c in criteria)
        if cfg.require_all_pass:
            overall = n_pass == len(criteria)
        else:
            overall = n_pass >= len(criteria) - 1  # tolerate a single miss

        report = GateReport(criteria=criteria, passed=overall)
        logger.info("\n%s", report.summary())
        return report
