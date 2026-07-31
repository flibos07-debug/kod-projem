"""Factories for the base learners and the stacking meta-model.

Base learners are deliberately diverse (a linear model, bagged trees and
gradient-boosted trees) so their errors decorrelate — the property that makes
stacking worthwhile. Each is returned fresh (unfitted) so callers can clone and
fit them independently per fold.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_base_models(random_state: int = 42, *, fast: bool = False) -> dict[str, Any]:
    """Return the default set of base learners keyed by name.

    Parameters
    ----------
    fast:
        Use small, quick configurations (handy for tests / smoke runs).
    """
    n_trees = 50 if fast else 300
    return {
        "logreg": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=1000, C=1.0, random_state=random_state),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=n_trees,
            max_depth=None if not fast else 6,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=random_state,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=n_trees,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=random_state,
        ),
        "hist_gb": HistGradientBoostingClassifier(
            max_iter=100 if fast else 400,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=random_state,
        ),
    }


def make_meta_model(name: str = "logistic_regression", random_state: int = 42) -> Any:
    """Return the meta-model estimator selected by ``name``."""
    key = name.lower()
    if key in ("logistic_regression", "logreg", "logistic"):
        return LogisticRegression(max_iter=1000, C=1.0, random_state=random_state)
    if key in ("random_forest", "rf"):
        return RandomForestClassifier(
            n_estimators=200, min_samples_leaf=20, n_jobs=-1, random_state=random_state
        )
    raise ValueError(f"unknown meta_model {name!r}")
