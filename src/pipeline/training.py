"""End-to-end training pipeline.

Wires the phases into one honest, leakage-controlled flow:

data frames -> multi-timeframe features -> triple-barrier labels -> regime
labels -> stability selection -> purged/embargoed walk-forward -> per-fold
stacking ensemble -> **out-of-sample** raw predictions -> per-regime calibration
and conformal calibration (fit on the OOS predictions) -> quality-gate metrics.

The calibrator and conformal predictor are fit on genuinely out-of-sample raw
predictions gathered across folds, so the reported calibration/coverage are not
in-sample. The final ensemble is refit on all data for live inference and
bundled with the calibrator/conformal into a :class:`ModelArtifact`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config_schema import AppConfig
from ..conformal.predictor import BinaryConformalPredictor
from ..features.engineering import FeatureParams, build_feature_matrix
from ..features.indicators import atr
from ..labeling.triple_barrier import BarrierParams, binary_target, triple_barrier_labels
from ..logging_utils import get_logger
from ..models.calibration import PerRegimeCalibrator
from ..models.stacking import StackingEnsemble
from ..quality.gate import (
    GateReport,
    QualityGate,
    brier_score,
    expected_calibration_error,
    top_k_precision,
)
from ..regime.classifier import RegimeClassifier
from ..selection.stability import StabilitySelector
from ..validation.walk_forward import WalkForwardSplitter
from .artifacts import ModelArtifact

logger = get_logger(__name__)


@dataclass(frozen=True)
class PipelineParams:
    base_timeframe: str = "5m"
    htf_timeframes: tuple[str, ...] = ("15m", "1h", "4h")
    barrier: BarrierParams = field(default_factory=lambda: BarrierParams(tp_mult=2.0, sl_mult=1.0, max_holding=12))
    feature_params: FeatureParams = field(default_factory=FeatureParams)
    vol_window: int = 500
    top_k: int = 5


@dataclass
class TrainingResult:
    artifact: ModelArtifact
    gate: GateReport
    metrics: dict[str, float]
    n_folds: int


class TrainingPipeline:
    def __init__(self, config: AppConfig, params: PipelineParams | None = None, *, fast: bool = False) -> None:
        self.config = config
        self.params = params or PipelineParams()
        self.fast = fast

    # -- dataset ------------------------------------------------------------
    def prepare_dataset(self, frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        p = self.params
        base = frames[p.base_timeframe]

        X_full = build_feature_matrix(
            frames,
            base_timeframe=p.base_timeframe,
            htf_timeframes=list(p.htf_timeframes),
            params=p.feature_params,
            dropna=True,
        )

        vol = atr(base["high"], base["low"], base["close"], p.feature_params.atr_period)
        labels = triple_barrier_labels(base, vol, p.barrier, event_index=X_full.index)
        y = binary_target(labels, timeout_as_loss=True)

        clf = RegimeClassifier(self.config.regime, vol_window=p.vol_window)
        adx_col = f"{p.base_timeframe}_adx"
        vol_col = f"{p.base_timeframe}_rvol"
        regime = clf.classify(X_full, adx_col=adx_col, volatility_col=vol_col).labels

        common = X_full.index.intersection(labels.index)
        common = common[y.reindex(common).notna()]
        X = X_full.loc[common]
        y = y.loc[common].astype(int)
        meta = pd.DataFrame(
            {
                "t1": labels.loc[common, "t1"],
                "ret": labels.loc[common, "ret"],
                "regime": regime.reindex(common),
            }
        )
        return X, y, meta

    # -- training -----------------------------------------------------------
    def run(self, frames: dict[str, pd.DataFrame], *, do_selection: bool = True, symbol: str = "") -> TrainingResult:
        cfg = self.config
        X, y, meta = self.prepare_dataset(frames)
        if len(X) < 50 or y.nunique() < 2:
            raise ValueError("insufficient labelled data to train")

        # -- stability selection -------------------------------------------
        if do_selection and len(X) >= 150:
            sel = StabilitySelector(
                n_bootstraps=30 if self.fast else 100,
                threshold=cfg.quality_gate.min_feature_stability,
            ).fit(X, y)
            selected = sel.selected_features() or list(X.columns)
            feature_stability = float(sel.selection_frequencies_[selected].mean())
        else:
            selected = list(X.columns)
            feature_stability = 1.0
        X = X[selected]

        # -- walk-forward OOS collection -----------------------------------
        splitter = WalkForwardSplitter(cfg.validation)
        folds = splitter.split(X.index, meta["t1"])

        oos_raw: list[np.ndarray] = []
        oos_y: list[np.ndarray] = []
        oos_reg: list[np.ndarray] = []
        used_folds = 0
        for fold in folds:
            ytr = y.iloc[fold.train]
            yte = y.iloc[fold.test]
            if ytr.nunique() < 2 or len(yte) == 0:
                continue
            ens = StackingEnsemble(
                use_stacking=cfg.ensemble.use_stacking, fast=self.fast,
            ).fit(X.iloc[fold.train], ytr, cv=cfg.validation.n_folds_nested)
            raw = ens.predict_proba_positive(X.iloc[fold.test])
            oos_raw.append(raw)
            oos_y.append(yte.to_numpy())
            oos_reg.append(meta["regime"].iloc[fold.test].to_numpy())
            used_folds += 1

        if used_folds == 0:
            raise ValueError("walk-forward produced no usable folds; extend the data span")

        raw_all = np.concatenate(oos_raw)
        y_all = np.concatenate(oos_y)
        reg_all = np.concatenate(oos_reg)

        # -- calibration + conformal on OOS predictions --------------------
        calibrator = PerRegimeCalibrator(
            method=cfg.calibration.method,
            per_regime=cfg.calibration.per_regime,
            min_samples=cfg.calibration.min_samples_calibration,
        ).fit(raw_all, y_all, reg_all)
        cal_all = calibrator.transform(raw_all, reg_all)

        conformal = BinaryConformalPredictor().fit(cal_all, y_all)

        # -- metrics + quality gate ----------------------------------------
        base_rate = float(y_all.mean())
        metrics = {
            "brier": brier_score(cal_all, y_all),
            "brier_baseline": brier_score(np.full_like(cal_all, base_rate), y_all),
            "ece": expected_calibration_error(cal_all, y_all),
            "top5_precision": top_k_precision(cal_all, y_all, self.params.top_k),
            "event_base_rate": base_rate,
            "feature_stability": feature_stability,
            "n_oos_folds": float(used_folds),
            "n_oos_samples": float(len(y_all)),
        }
        gate = QualityGate(cfg.quality_gate).evaluate(
            brier=metrics["brier"],
            brier_baseline=metrics["brier_baseline"],
            ece=metrics["ece"],
            top5_precision=metrics["top5_precision"],
            event_base_rate=base_rate,
            feature_stability=feature_stability,
            n_oos_folds=used_folds,
        )

        # -- final refit on all data ---------------------------------------
        final_ens = StackingEnsemble(
            use_stacking=cfg.ensemble.use_stacking, fast=self.fast,
        ).fit(X, y, cv=cfg.validation.n_folds_nested)

        artifact = ModelArtifact(
            ensemble=final_ens,
            calibrator=calibrator,
            conformal=conformal,
            feature_names=selected,
            feature_params=self.params.feature_params,
            metadata={
                "symbol": symbol,
                "train_span": [str(X.index[0]), str(X.index[-1])],
                "n_samples": int(len(X)),
                "metrics": metrics,
                "gate_passed": gate.passed,
                "regime_classes": list(cfg.regime.classes),
                "base_timeframe": self.params.base_timeframe,
                "htf_timeframes": list(self.params.htf_timeframes),
                "alpha_90": cfg.conformal.alpha_90,
                "alpha_80": cfg.conformal.alpha_80,
            },
        )
        logger.info("Training complete for %s: gate=%s, folds=%d", symbol or "?", gate.passed, used_folds)
        return TrainingResult(artifact=artifact, gate=gate, metrics=metrics, n_folds=used_folds)
