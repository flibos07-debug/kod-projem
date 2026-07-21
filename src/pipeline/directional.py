"""Directional (long **and** short) training pipeline.

Produces two independent binary models sharing one feature set:

- a **long** model — probability that a long trade hits its take-profit before
  its stop (triple-barrier ``side=+1``);
- a **short** model — the same for a short trade (``side=-1``).

Features are side-independent, so they are built once and stability-selection is
run once (on the long labels) to fix a single feature set used by both sides.
Each side is then trained, calibrated, conformalised and gated independently via
the shared :func:`~src.pipeline.training.train_side_model`.

At scan time the two calibrated probabilities and their conformal sets decide
LONG / SHORT / FLAT, so the system emits genuine two-sided signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from ..config_schema import AppConfig
from ..features.engineering import FeatureParams, build_feature_matrix
from ..features.indicators import atr
from ..labeling.triple_barrier import BarrierParams, binary_target, triple_barrier_labels
from ..logging_utils import get_logger
from ..regime.classifier import RegimeClassifier
from ..selection.stability import StabilitySelector
from .artifacts import ARTIFACT_VERSION
from .training import PipelineParams, SideTrainResult, train_side_model

logger = get_logger(__name__)


@dataclass
class SideModel:
    """One direction's fitted model bundle."""

    ensemble: object
    calibrator: object
    conformal: object
    metrics: dict[str, float]
    gate_passed: bool


@dataclass
class DirectionalArtifact:
    """Long+short model bundle sharing one feature set."""

    long: SideModel
    short: SideModel | None
    feature_names: list[str]
    feature_params: FeatureParams
    metadata: dict = field(default_factory=dict)
    version: int = ARTIFACT_VERSION
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class DirectionalResult:
    artifact: DirectionalArtifact
    long: SideTrainResult
    short: SideTrainResult | None


class DirectionalPipeline:
    def __init__(self, config: AppConfig, params: PipelineParams | None = None, *, fast: bool = False) -> None:
        self.config = config
        self.params = params or PipelineParams()
        self.fast = fast

    def _build_features(self, frames: dict[str, pd.DataFrame]):
        p = self.params
        base = frames[p.base_timeframe]
        X_full = build_feature_matrix(
            frames, base_timeframe=p.base_timeframe, htf_timeframes=list(p.htf_timeframes),
            params=p.feature_params, dropna=True,
        )
        vol = atr(base["high"], base["low"], base["close"], p.feature_params.atr_period)
        clf = RegimeClassifier(self.config.regime, vol_window=p.vol_window)
        regime = clf.classify(
            X_full, adx_col=f"{p.base_timeframe}_adx", volatility_col=f"{p.base_timeframe}_rvol"
        ).labels
        return X_full, base, vol, regime

    def _side_dataset(self, X_full, base, vol, regime, side):
        labels = triple_barrier_labels(
            base, vol,
            BarrierParams(tp_mult=self.params.barrier.tp_mult, sl_mult=self.params.barrier.sl_mult,
                          max_holding=self.params.barrier.max_holding, side=side),
            event_index=X_full.index,
        )
        y = binary_target(labels, timeout_as_loss=True)
        common = X_full.index.intersection(labels.index)
        common = common[y.reindex(common).notna()]
        return (
            X_full.loc[common],
            y.loc[common].astype(int),
            labels.loc[common, "t1"],
            regime.reindex(common),
        )

    def run(self, frames: dict[str, pd.DataFrame], *, do_selection: bool = True, symbol: str = "", with_short: bool = True) -> DirectionalResult:
        cfg = self.config
        X_full, base, vol, regime = self._build_features(frames)

        # -- select features once, on the long side ------------------------
        Xl, yl, t1l, regl = self._side_dataset(X_full, base, vol, regime, side=1)
        if len(Xl) < 50 or yl.nunique() < 2:
            raise ValueError("insufficient labelled data to train (long side)")

        if do_selection and len(Xl) >= 150:
            sel = StabilitySelector(
                n_bootstraps=30 if self.fast else 100,
                threshold=cfg.quality_gate.min_feature_stability,
            ).fit(Xl, yl)
            selected = sel.selected_features() or list(Xl.columns)
            feature_stability = float(sel.selection_frequencies_[selected].mean())
        else:
            selected = list(Xl.columns)
            feature_stability = 1.0

        long_res = train_side_model(
            cfg, Xl[selected], yl, t1l, regl,
            fast=self.fast, feature_stability=feature_stability, top_k=self.params.top_k,
        )
        long_model = SideModel(long_res.ensemble, long_res.calibrator, long_res.conformal,
                               long_res.metrics, long_res.gate.passed)

        short_res = None
        short_model = None
        if with_short:
            Xs, ys, t1s, regs = self._side_dataset(X_full, base, vol, regime, side=-1)
            if ys.nunique() >= 2 and len(Xs) >= 50:
                short_res = train_side_model(
                    cfg, Xs[selected], ys, t1s, regs,
                    fast=self.fast, feature_stability=feature_stability, top_k=self.params.top_k,
                )
                short_model = SideModel(short_res.ensemble, short_res.calibrator, short_res.conformal,
                                        short_res.metrics, short_res.gate.passed)
            else:
                logger.warning("Short side has insufficient/one-class data; skipping short model")

        artifact = DirectionalArtifact(
            long=long_model,
            short=short_model,
            feature_names=selected,
            feature_params=self.params.feature_params,
            metadata={
                "symbol": symbol,
                "market": "futures",
                "train_span": [str(Xl.index[0]), str(Xl.index[-1])],
                "n_samples": int(len(Xl)),
                "base_timeframe": self.params.base_timeframe,
                "htf_timeframes": list(self.params.htf_timeframes),
                "alpha_90": cfg.conformal.alpha_90,
                "alpha_80": cfg.conformal.alpha_80,
                "long_gate_passed": long_model.gate_passed,
                "short_gate_passed": bool(short_model.gate_passed) if short_model else None,
                "long_metrics": long_model.metrics,
                "short_metrics": short_model.metrics if short_model else None,
            },
        )
        logger.info(
            "Directional training for %s: long gate=%s, short gate=%s",
            symbol or "?", long_model.gate_passed,
            short_model.gate_passed if short_model else "n/a",
        )
        return DirectionalResult(artifact=artifact, long=long_res, short=short_res)
