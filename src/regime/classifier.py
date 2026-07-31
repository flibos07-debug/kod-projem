"""Classify each bar into a market regime and produce routing weights.

Three canonical regimes are supported: ``trend``, ``range`` and ``high_vol``.
Classification combines a trend-strength signal (ADX) with a volatility signal
(the rolling percentile of realized volatility):

- **high_vol** when volatility sits at/above ``vol_percentile_high``;
- **trend** when ADX is at/above ``adx_threshold_trend``;
- **range** when ADX is at/below ``adx_threshold_range``;
- otherwise the ambiguous middle is assigned to whichever ADX threshold is
  nearer.

For soft routing, per-regime scores are turned into weights with a
temperature-scaled softmax so a bar can be handled by a blend of regime experts
rather than a single hard bucket.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config_schema import RegimeConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)

_EPS = 1e-9


@dataclass(frozen=True)
class RegimeResult:
    """Output of :meth:`RegimeClassifier.classify`.

    Attributes
    ----------
    labels:
        Hard regime label per bar (``Series`` of str).
    weights:
        Routing weights per regime (``DataFrame``; columns are regime names,
        rows sum to 1). Equals the soft-routing softmax when enabled, otherwise
        a one-hot of ``labels``.
    vol_percentile:
        The rolling volatility percentile used (0-100).
    """

    labels: pd.Series
    weights: pd.DataFrame
    vol_percentile: pd.Series


class RegimeClassifier:
    def __init__(self, config: RegimeConfig, *, vol_window: int = 500) -> None:
        self.config = config
        self.vol_window = vol_window
        self.classes = list(config.classes)
        self._known = {"trend", "range", "high_vol"}
        unknown = set(self.classes) - self._known
        if unknown:
            logger.warning("Regime classes not scored by rules: %s", sorted(unknown))

    # -- volatility percentile ---------------------------------------------
    def vol_percentile(self, volatility: pd.Series) -> pd.Series:
        """Rolling percentile rank (0-100) of the volatility series."""
        min_periods = max(20, self.vol_window // 5)
        ranked = volatility.rolling(self.vol_window, min_periods=min_periods).rank(pct=True)
        return ranked * 100.0

    # -- scoring ------------------------------------------------------------
    def _scores(self, adx: pd.Series, vol_pct: pd.Series) -> pd.DataFrame:
        cfg = self.config
        adx_scale = max(cfg.adx_threshold_trend - cfg.adx_threshold_range, _EPS)
        vol_scale = max(100.0 - cfg.vol_percentile_high, _EPS)

        raw = {
            "trend": (adx - cfg.adx_threshold_trend) / adx_scale,
            "range": (cfg.adx_threshold_range - adx) / adx_scale,
            "high_vol": (vol_pct - cfg.vol_percentile_high) / vol_scale,
        }
        # Restrict/order to configured classes; unknown classes score 0.
        data = {c: raw.get(c, pd.Series(0.0, index=adx.index)) for c in self.classes}
        return pd.DataFrame(data, index=adx.index)

    def soft_weights(self, adx: pd.Series, vol_pct: pd.Series) -> pd.DataFrame:
        scores = self._scores(adx, vol_pct)
        temp = max(self.config.routing_temperature, _EPS)
        scaled = scores / temp
        # Numerically stable softmax across regime columns.
        shifted = scaled.sub(scaled.max(axis=1), axis=0)
        exp = np.exp(shifted)
        weights = exp.div(exp.sum(axis=1), axis=0)
        # Rows with any NaN input yield NaN weights (undefined regime).
        invalid = scores.isna().any(axis=1)
        weights.loc[invalid] = np.nan
        return weights

    def hard_label(self, adx: pd.Series, vol_pct: pd.Series) -> pd.Series:
        cfg = self.config
        midpoint = 0.5 * (cfg.adx_threshold_trend + cfg.adx_threshold_range)

        labels = pd.Series(index=adx.index, dtype="object")
        is_high_vol = vol_pct >= cfg.vol_percentile_high
        is_trend = adx >= cfg.adx_threshold_trend
        is_range = adx <= cfg.adx_threshold_range
        # Ambiguous middle -> nearest ADX threshold.
        middle_trend = (~is_trend) & (~is_range) & (adx >= midpoint)

        labels[is_range] = "range"
        labels[middle_trend | is_trend] = "trend"
        # high_vol takes precedence when volatility is extreme.
        labels[is_high_vol] = "high_vol"
        labels[adx.isna() | vol_pct.isna()] = np.nan

        # Fall back to the first configured class for anything still unset.
        labels = labels.where(labels.notna() | (adx.isna() | vol_pct.isna()), self.classes[0])
        return labels

    # -- top-level ----------------------------------------------------------
    def classify(
        self,
        features: pd.DataFrame,
        *,
        adx_col: str,
        volatility_col: str,
    ) -> RegimeResult:
        """Classify each row of ``features``.

        Parameters
        ----------
        adx_col, volatility_col:
            Column names in ``features`` holding the ADX and the (raw) realized
            volatility used for the percentile.
        """
        for col in (adx_col, volatility_col):
            if col not in features.columns:
                raise KeyError(f"features is missing required column {col!r}")

        adx = features[adx_col]
        vol_pct = self.vol_percentile(features[volatility_col])
        labels = self.hard_label(adx, vol_pct)

        if self.config.use_soft_routing:
            weights = self.soft_weights(adx, vol_pct)
        else:
            weights = pd.get_dummies(labels).reindex(columns=self.classes, fill_value=0).astype(float)
            weights[labels.isna()] = np.nan

        return RegimeResult(labels=labels, weights=weights, vol_percentile=vol_pct)
