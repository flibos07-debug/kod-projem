"""Nested, purged, embargoed walk-forward cross-validation.

Outer folds move forward in calendar time using the month windows from
``ValidationConfig`` (train / validation / test / step). Within each fold:

- **Purge** (``purge_bars``): a gap between train and validation, and — when
  label end-times (``t1``) are supplied — training samples whose label windows
  extend into the validation period are removed. This prevents a sample's
  forward-looking label from leaking information about the evaluation window.
- **Embargo** (``embargo_bars``): a gap between validation and test, guarding
  against serial-correlation leakage across the boundary.

Each outer fold also carries an inner nested split of its (purged) training set
for hyper-parameter selection, built with the same purge gap.

Positions returned in :class:`Fold` are integer indices into the *original*
``sample_times`` array, so ``X.iloc[fold.train]`` is valid when ``X`` is aligned
row-for-row with ``sample_times``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config_schema import ValidationConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    train_span: tuple[pd.Timestamp, pd.Timestamp]
    val_span: tuple[pd.Timestamp, pd.Timestamp]
    test_span: tuple[pd.Timestamp, pd.Timestamp]
    inner: list[tuple[np.ndarray, np.ndarray]]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Fold(id={self.fold_id}, train={len(self.train)}, "
            f"val={len(self.val)}, test={len(self.test)}, inner={len(self.inner)})"
        )


class WalkForwardSplitter:
    def __init__(self, config: ValidationConfig, *, expanding: bool = False) -> None:
        self.config = config
        self.expanding = expanding

    def _months(self, origin: pd.Timestamp, m: int) -> pd.Timestamp:
        return origin + pd.DateOffset(months=int(m))

    def split(
        self,
        sample_times: pd.DatetimeIndex,
        label_end_times: pd.Series | None = None,
    ) -> list[Fold]:
        times = pd.DatetimeIndex(sample_times)
        if len(times) == 0:
            return []

        order = np.argsort(times.values, kind="mergesort")
        sorted_times = times[order]

        if label_end_times is not None:
            t1 = pd.Series(label_end_times).reindex(times)
            sorted_t1 = pd.DatetimeIndex(t1.iloc[order])  # preserves tz
        else:
            sorted_t1 = sorted_times  # no purge-by-overlap; only positional gaps

        cfg = self.config
        origin = sorted_times[0]
        last_time = sorted_times[-1]

        folds: list[Fold] = []
        fold_id = 0
        i = 0
        while True:
            base = i * cfg.step_months
            train_start = origin if self.expanding else self._months(origin, base)
            train_end = self._months(origin, base + cfg.train_months)
            val_start = train_end
            val_end = self._months(origin, base + cfg.train_months + cfg.validation_months)
            test_start = val_end
            test_end = self._months(
                origin, base + cfg.train_months + cfg.validation_months + cfg.test_months
            )
            if test_start > last_time:
                break
            i += 1

            train_mask = (sorted_times >= train_start) & (sorted_times < train_end)
            val_mask = (sorted_times >= val_start) & (sorted_times < val_end)
            test_mask = (sorted_times >= test_start) & (sorted_times < test_end)

            # Purge by label overlap.
            if label_end_times is not None:
                train_mask &= ~(sorted_t1 >= val_start)
                val_mask &= ~(sorted_t1 >= test_start)

            train_pos = np.where(train_mask)[0]
            val_pos = np.where(val_mask)[0]
            test_pos = np.where(test_mask)[0]

            # Positional gaps: purge (train->val) and embargo (val->test).
            if cfg.purge_bars > 0 and len(train_pos) > cfg.purge_bars:
                train_pos = train_pos[: -cfg.purge_bars]
            elif cfg.purge_bars > 0:
                train_pos = train_pos[:0]
            if cfg.embargo_bars > 0 and len(val_pos) > cfg.embargo_bars:
                val_pos = val_pos[: -cfg.embargo_bars]
            elif cfg.embargo_bars > 0:
                val_pos = val_pos[:0]

            if len(train_pos) == 0 or len(test_pos) == 0:
                logger.debug("Skipping empty fold at test_start=%s", test_start)
                continue

            inner = self._inner_splits(train_pos)

            folds.append(
                Fold(
                    fold_id=fold_id,
                    train=order[train_pos],
                    val=order[val_pos],
                    test=order[test_pos],
                    train_span=(train_start, train_end),
                    val_span=(val_start, val_end),
                    test_span=(test_start, test_end),
                    inner=[(order[a], order[b]) for a, b in inner],
                )
            )
            fold_id += 1

        if not folds:
            logger.warning("Walk-forward produced no folds; check window sizes vs data span")
        return folds

    def _inner_splits(self, train_pos: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """Expanding purged time-series splits within the training positions."""
        n_folds = self.config.n_folds_nested
        gap = self.config.purge_bars
        if n_folds < 1 or len(train_pos) < (n_folds + 1) * (gap + 1):
            logger.debug("Training set too small for %d inner folds", n_folds)
            return []

        blocks = np.array_split(train_pos, n_folds + 1)
        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for k in range(n_folds):
            inner_train = np.concatenate(blocks[: k + 1])
            inner_val = blocks[k + 1]
            if gap > 0 and len(inner_train) > gap:
                inner_train = inner_train[:-gap]
            if len(inner_train) and len(inner_val):
                splits.append((inner_train, inner_val))
        return splits
