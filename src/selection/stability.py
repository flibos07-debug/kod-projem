"""Stability selection (Meinshausen & Buhlmann).

An L1-penalised logistic model is refit on many random subsamples of the data;
a feature's **selection frequency** is the fraction of subsamples in which it
receives a non-zero coefficient. Features whose frequency clears a threshold are
kept. This is far more robust than a single L1 fit, whose selected set is
sensitive to the particular sample — exactly the instability that matters when a
model must survive out-of-sample.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..logging_utils import get_logger

logger = get_logger(__name__)

_SKLEARN_VER = tuple(int(p) for p in sklearn.__version__.split(".")[:2])


def _l1_logistic(c: float, random_state: int) -> LogisticRegression:
    """L1-penalised logistic regression, using the API of the installed sklearn.

    sklearn 1.8 deprecated ``penalty='l1'`` in favour of ``l1_ratio``; older
    versions require the ``penalty``/``liblinear`` spelling.
    """
    if _SKLEARN_VER >= (1, 8):
        return LogisticRegression(
            solver="saga", l1_ratio=1.0, C=c, max_iter=2000, random_state=random_state
        )
    return LogisticRegression(
        penalty="l1", solver="liblinear", C=c, max_iter=500, random_state=random_state
    )


class StabilitySelector:
    def __init__(
        self,
        *,
        n_bootstraps: int = 100,
        sample_fraction: float = 0.5,
        threshold: float = 0.7,
        c_grid: tuple[float, ...] = (0.05, 0.1, 0.25),
        random_state: int = 42,
    ) -> None:
        if not 0 < sample_fraction <= 1:
            raise ValueError("sample_fraction must be in (0, 1]")
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        self.n_bootstraps = n_bootstraps
        self.sample_fraction = sample_fraction
        self.threshold = threshold
        self.c_grid = c_grid
        self.random_state = random_state
        self.feature_names_: list[str] = []
        self.selection_frequencies_: pd.Series | None = None
        self.support_: np.ndarray | None = None

    def fit(self, X: Any, y: Any) -> "StabilitySelector":
        if hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)
            Xa = X.to_numpy(dtype="float64")
        else:
            Xa = np.asarray(X, dtype="float64")
            self.feature_names_ = [f"f{i}" for i in range(Xa.shape[1])]
        ya = np.asarray(y).astype(int)

        n, p = Xa.shape
        rng = np.random.default_rng(self.random_state)
        counts = np.zeros(p, dtype="float64")
        effective = 0
        sub_n = max(2, int(self.sample_fraction * n))

        for b in range(self.n_bootstraps):
            idx = rng.choice(n, size=sub_n, replace=False)
            y_sub = ya[idx]
            if len(np.unique(y_sub)) < 2:
                continue
            X_sub = StandardScaler().fit_transform(Xa[idx])
            c = self.c_grid[b % len(self.c_grid)]
            model = _l1_logistic(c, self.random_state + b)
            model.fit(X_sub, y_sub)
            counts += (np.abs(model.coef_.ravel()) > 1e-8).astype("float64")
            effective += 1

        if effective == 0:
            logger.warning("No valid bootstrap had both classes; nothing selected")
            freqs = np.zeros(p)
        else:
            freqs = counts / effective

        self.selection_frequencies_ = pd.Series(freqs, index=self.feature_names_).sort_values(ascending=False)
        self.support_ = freqs >= self.threshold
        return self

    def get_support(self, indices: bool = False) -> np.ndarray:
        if self.support_ is None:
            raise RuntimeError("StabilitySelector is not fitted")
        return np.where(self.support_)[0] if indices else self.support_

    def selected_features(self) -> list[str]:
        if self.support_ is None:
            raise RuntimeError("StabilitySelector is not fitted")
        return [name for name, keep in zip(self.feature_names_, self.support_) if keep]

    def transform(self, X: Any) -> Any:
        if self.support_ is None:
            raise RuntimeError("StabilitySelector is not fitted")
        if hasattr(X, "loc"):
            return X.loc[:, self.selected_features()]
        return np.asarray(X)[:, self.support_]
