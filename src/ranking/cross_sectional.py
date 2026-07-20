"""Cross-sectional ranking of signals across the traded universe.

At each timestamp the model emits a score per symbol. Trading the *relative*
ranking (top-N per timestamp) rather than an absolute threshold makes the
strategy robust to overall score drift and naturally caps exposure. Scores are
converted to a within-timestamp percentile (0-1) and the top-N per direction are
selected.

Input is a tidy/long frame with a timestamp column, a symbol column and a score
column, so it composes directly with the live scanner output.
"""

from __future__ import annotations

import pandas as pd

from ..logging_utils import get_logger

logger = get_logger(__name__)


def cross_sectional_rank(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    symbol_col: str = "symbol",
    score_col: str = "score",
) -> pd.DataFrame:
    """Add ``rank`` (1 = best) and ``percentile`` columns per timestamp."""
    for col in (time_col, symbol_col, score_col):
        if col not in df.columns:
            raise KeyError(f"missing required column {col!r}")

    out = df.copy()
    grp = out.groupby(time_col)[score_col]
    # rank(method="first") to break ties deterministically; 1 = highest score.
    out["rank"] = grp.rank(ascending=False, method="first").astype(int)
    # percentile in [0, 1]; higher score -> higher percentile.
    out["percentile"] = grp.rank(ascending=True, pct=True)
    return out


def select_top_n(
    df: pd.DataFrame,
    n: int,
    *,
    time_col: str = "time",
    score_col: str = "score",
    direction_col: str | None = None,
    ascending: bool = False,
) -> pd.DataFrame:
    """Select the top ``n`` rows per timestamp (optionally per direction).

    With ``direction_col`` given, the top ``n`` are chosen within each
    (timestamp, direction) group, implementing a per-direction position cap.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    keys = [time_col] if direction_col is None else [time_col, direction_col]
    ranked = df.sort_values(score_col, ascending=ascending)
    return (
        ranked.groupby(keys, group_keys=False, sort=False)
        .head(n)
        .sort_values([time_col, score_col], ascending=[True, ascending])
        .reset_index(drop=True)
    )
