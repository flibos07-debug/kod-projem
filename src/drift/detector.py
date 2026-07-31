"""Feature-distribution drift detection.

Two complementary statistics compare a *reference* distribution (e.g. the
training window) against a *current* one:

- **PSI** (Population Stability Index) — the classic banking metric. Common
  rules of thumb: ``< 0.10`` stable, ``0.10-0.25`` moderate shift, ``> 0.25``
  significant shift.
- **Jensen-Shannon divergence** — a bounded, symmetric divergence in ``[0, 1]``
  (log base 2), robust for comparing histograms.

Both bin continuous features using quantile edges derived from the reference so
that empty/steep regions don't dominate. Severity levels come from the
``DriftConfig`` thresholds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config_schema import DriftConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)

_EPS = 1e-6


def _quantile_edges(reference: np.ndarray, bins: int) -> np.ndarray:
    ref = reference[np.isfinite(reference)]
    if ref.size == 0:
        return np.array([-np.inf, np.inf])
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, qs))
    if edges.size < 2:  # constant reference
        edges = np.array([edges[0] - 1e-9, edges[0] + 1e-9])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _binned_fractions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    values = values[np.isfinite(values)]
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    if total == 0:
        return np.full(len(counts), 1.0 / len(counts))
    return counts / total


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, *, bins: int = 10
) -> float:
    """Population Stability Index between two samples of one feature."""
    edges = _quantile_edges(np.asarray(reference, dtype="float64"), bins)
    ref_frac = _binned_fractions(np.asarray(reference, dtype="float64"), edges) + _EPS
    cur_frac = _binned_fractions(np.asarray(current, dtype="float64"), edges) + _EPS
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def jensen_shannon_divergence(
    reference: np.ndarray, current: np.ndarray, *, bins: int = 10
) -> float:
    """Jensen-Shannon divergence (base 2, in [0, 1]) between two samples."""
    edges = _quantile_edges(np.asarray(reference, dtype="float64"), bins)
    p = _binned_fractions(np.asarray(reference, dtype="float64"), edges) + _EPS
    q = _binned_fractions(np.asarray(current, dtype="float64"), edges) + _EPS
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * np.log2(a / b)))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


class DriftDetector:
    def __init__(self, config: DriftConfig, *, bins: int = 10) -> None:
        self.config = config
        self.bins = bins

    def _severity(self, psi: float) -> str:
        if psi >= self.config.psi_threshold_high:
            return "high"
        if psi >= self.config.psi_threshold_medium:
            return "medium"
        return "none"

    def compare(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        *,
        features: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return a per-feature drift report sorted by PSI (descending)."""
        features = features or [c for c in reference.columns if c in current.columns]
        rows = []
        for col in features:
            ref = reference[col].to_numpy(dtype="float64")
            cur = current[col].to_numpy(dtype="float64")
            psi = population_stability_index(ref, cur, bins=self.bins)
            js = jensen_shannon_divergence(ref, cur, bins=self.bins)
            rows.append(
                {
                    "feature": col,
                    "psi": psi,
                    "js": js,
                    "severity": self._severity(psi),
                    "js_flag": js >= self.config.js_threshold,
                }
            )
        report = pd.DataFrame(rows).set_index("feature")
        return report.sort_values("psi", ascending=False)

    def any_high_drift(self, report: pd.DataFrame) -> bool:
        return bool((report["severity"] == "high").any())
