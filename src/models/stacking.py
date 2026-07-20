"""Out-of-fold stacking ensemble for binary classification.

Base learners are combined by training a meta-model on their **out-of-fold**
(OOF) predictions, never on in-sample fits — the standard guard against the
meta-model simply learning the base models' overfit. For inference the base
learners are refit on the full training set.

The class is written for a binary target with labels ``{0, 1}`` (the positive
class is ``1``), matching the triple-barrier ``binary_target`` output.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from ..logging_utils import get_logger
from .base_models import make_base_models, make_meta_model

logger = get_logger(__name__)


def _as_array(X: Any) -> np.ndarray:
    if hasattr(X, "to_numpy"):
        return X.to_numpy(dtype="float64")
    return np.asarray(X, dtype="float64")


def _positive_proba(model: Any, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 in classes:
        return proba[:, classes.index(1)]
    # Degenerate: model saw a single class in training.
    return np.full(len(X), float(classes[0] == 1))


class StackingEnsemble:
    def __init__(
        self,
        base_models: dict[str, Any] | None = None,
        meta_model: Any | None = None,
        *,
        use_stacking: bool = True,
        use_ranking: bool = False,
        ranking_weight: float = 0.25,
        probability_weight: float = 0.75,
        random_state: int = 42,
        fast: bool = False,
    ) -> None:
        self.base_models = base_models or make_base_models(random_state, fast=fast)
        self.meta_model = meta_model if meta_model is not None else make_meta_model()
        self.use_stacking = use_stacking
        self.use_ranking = use_ranking
        self.ranking_weight = ranking_weight
        self.probability_weight = probability_weight
        self.random_state = random_state
        self.fitted_base_: dict[str, Any] = {}
        self.fitted_meta_: Any | None = None
        self.base_names_: list[str] = list(self.base_models)

    def _make_cv(self, n_samples: int, cv: Any) -> list[tuple[np.ndarray, np.ndarray]]:
        if isinstance(cv, int):
            splitter = TimeSeriesSplit(n_splits=cv)
            return list(splitter.split(np.arange(n_samples)))
        return [(np.asarray(tr), np.asarray(va)) for tr, va in cv]

    def _oof_matrix(
        self, X: np.ndarray, y: np.ndarray, splits: Sequence[tuple[np.ndarray, np.ndarray]]
    ) -> np.ndarray:
        n, m = len(X), len(self.base_names_)
        oof = np.full((n, m), np.nan)
        for j, name in enumerate(self.base_names_):
            for train_idx, val_idx in splits:
                if len(np.unique(y[train_idx])) < 2:
                    continue
                model = clone(self.base_models[name])
                model.fit(X[train_idx], y[train_idx])
                oof[val_idx, j] = _positive_proba(model, X[val_idx])
        return oof

    def fit(self, X: Any, y: Any, cv: Any = 5) -> "StackingEnsemble":
        Xa, ya = _as_array(X), np.asarray(y)
        splits = self._make_cv(len(Xa), cv)

        if self.use_stacking:
            oof = self._oof_matrix(Xa, ya, splits)
            covered = ~np.isnan(oof).any(axis=1)
            if covered.sum() < 10 or len(np.unique(ya[covered])) < 2:
                logger.warning("Insufficient OOF coverage; falling back to averaging")
                self.fitted_meta_ = None
            else:
                self.fitted_meta_ = clone(self.meta_model)
                self.fitted_meta_.fit(oof[covered], ya[covered])
        else:
            self.fitted_meta_ = None

        # Refit base models on the full training set for inference.
        self.fitted_base_ = {}
        for name in self.base_names_:
            model = clone(self.base_models[name])
            model.fit(Xa, ya)
            self.fitted_base_[name] = model
        return self

    def _base_matrix(self, X: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [_positive_proba(self.fitted_base_[name], X) for name in self.base_names_]
        )

    def predict_proba_positive(self, X: Any) -> np.ndarray:
        """Return the probability of the positive class (1D array)."""
        if not self.fitted_base_:
            raise RuntimeError("StackingEnsemble is not fitted")
        Xa = _as_array(X)
        base = self._base_matrix(Xa)
        if self.use_stacking and self.fitted_meta_ is not None:
            return _positive_proba(self.fitted_meta_, base)
        return base.mean(axis=1)

    def predict(self, X: Any, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba_positive(X) >= threshold).astype(int)


def blend_probability_ranking(
    probability: np.ndarray,
    rank_score: np.ndarray,
    *,
    probability_weight: float = 0.75,
    ranking_weight: float = 0.25,
) -> np.ndarray:
    """Convex blend of a probability and a (0-1) ranking score.

    Weights are renormalised so they sum to 1, so callers can pass the raw
    config weights directly.
    """
    total = probability_weight + ranking_weight
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    pw, rw = probability_weight / total, ranking_weight / total
    return pw * np.asarray(probability) + rw * np.asarray(rank_score)
