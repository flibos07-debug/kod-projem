"""Event-driven backtest with realistic costs and Monte Carlo robustness.

The engine consumes triple-barrier trades (each already resolved to an entry,
an exit and a signed gross return) and applies:

- **round-trip costs** — taker fee and slippage on *both* sides;
- **a concurrency cap** — at most ``max_positions_per_direction`` positions open
  per direction at once (when ``prevent_overlapping`` is set), so the equity
  curve reflects a capacity-constrained portfolio rather than an impossible
  infinitely-levered one;
- **fixed-fractional sizing** — capital split evenly across the available slots.

Point estimates from a single historical path are fragile, so
:meth:`monte_carlo` bootstraps the trade sequence to produce distributions of
final return and max drawdown, and the probability of an unprofitable run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config_schema import BacktestConfig
from ..logging_utils import get_logger

logger = get_logger(__name__)


def max_drawdown(equity: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of an equity curve (as a fraction)."""
    equity = np.asarray(equity, dtype="float64")
    if equity.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    return float(-drawdowns.min())


@dataclass(frozen=True)
class BacktestResult:
    stats: dict[str, float]
    equity_curve: pd.Series
    trades: pd.DataFrame
    net_returns: np.ndarray = field(repr=False)

    def summary(self) -> str:  # pragma: no cover - cosmetic
        return "\n".join(f"{k}: {v:.4f}" for k, v in self.stats.items())


@dataclass(frozen=True)
class MonteCarloResult:
    final_return: dict[str, float]
    max_drawdown: dict[str, float]
    prob_negative: float
    iterations: int


class BacktestEngine:
    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self.round_trip_cost = 2.0 * (
            config.assumed_taker_fee_percent_per_side
            + config.assumed_slippage_percent_per_side
        ) / 100.0

    # -- position filtering -------------------------------------------------
    def _accept_mask(self, trades: pd.DataFrame) -> np.ndarray:
        """Which trades survive the per-direction concurrency cap."""
        if not self.config.prevent_overlapping:
            return np.ones(len(trades), dtype=bool)

        cap = self.config.max_positions_per_direction
        open_by_dir: dict[object, list[pd.Timestamp]] = {}
        accepted = np.zeros(len(trades), dtype=bool)

        # trades has a reset RangeIndex, so each row label equals its position.
        for idx, row in trades.sort_values("entry_time").iterrows():
            direction = row.get("direction", "long")
            entry, exit_ = row["entry_time"], row["exit_time"]
            open_list = open_by_dir.setdefault(direction, [])
            # Drop positions that have already closed by this entry.
            open_list[:] = [t for t in open_list if t > entry]
            if len(open_list) < cap:
                accepted[idx] = True
                open_list.append(exit_)
        return accepted

    # -- run ----------------------------------------------------------------
    def run(self, trades: pd.DataFrame) -> BacktestResult:
        required = {"entry_time", "exit_time", "ret"}
        missing = required - set(trades.columns)
        if missing:
            raise ValueError(f"trades missing column(s): {sorted(missing)}")

        if trades.empty:
            empty = pd.Series(dtype="float64")
            return BacktestResult(_empty_stats(), empty, trades.copy(), np.array([]))

        trades = trades.reset_index(drop=True)
        accepted = trades[self._accept_mask(trades)].copy()
        gross = accepted["ret"].to_numpy(dtype="float64")
        net = gross - self.round_trip_cost
        accepted["net_ret"] = net

        directions = accepted.get("direction")
        n_dir = int(directions.nunique()) if directions is not None else 1
        n_dir = max(n_dir, 1)
        total_slots = max(self.config.max_positions_per_direction * n_dir, 1)
        f = 1.0 / total_slots

        # Realise PnL at exit time to build the equity curve.
        realised = accepted.sort_values("exit_time")
        pnl = f * realised["net_ret"].to_numpy(dtype="float64")
        equity = 1.0 + np.cumsum(pnl)
        equity_curve = pd.Series(equity, index=realised["exit_time"].to_numpy(), name="equity")

        stats = self._stats(net, equity, f)
        return BacktestResult(stats, equity_curve, accepted, net)

    def _stats(self, net: np.ndarray, equity: np.ndarray, f: float) -> dict[str, float]:
        wins = net[net > 0]
        losses = net[net < 0]
        gross_profit = wins.sum()
        gross_loss = -losses.sum()
        std = net.std(ddof=1) if len(net) > 1 else 0.0
        return {
            "n_trades": float(len(net)),
            "total_return": float(equity[-1] - 1.0) if len(equity) else 0.0,
            "avg_net_return": float(net.mean()) if len(net) else 0.0,
            "win_rate": float((net > 0).mean()) if len(net) else 0.0,
            "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
            "sharpe_per_trade": float(net.mean() / std) if std > 0 else 0.0,
            "max_drawdown": max_drawdown(equity),
            "position_fraction": f,
        }

    # -- Monte Carlo --------------------------------------------------------
    def monte_carlo(
        self,
        net_returns: np.ndarray,
        *,
        position_fraction: float | None = None,
        iterations: int | None = None,
        random_state: int = 42,
    ) -> MonteCarloResult:
        """Bootstrap the trade sequence to estimate outcome distributions."""
        net = np.asarray(net_returns, dtype="float64")
        iterations = iterations or self.config.monte_carlo_iterations
        if net.size == 0:
            return MonteCarloResult({}, {}, float("nan"), iterations)

        f = position_fraction if position_fraction is not None else 1.0 / max(
            self.config.max_positions_per_direction, 1
        )
        rng = np.random.default_rng(random_state)
        n = len(net)
        finals = np.empty(iterations)
        mdds = np.empty(iterations)
        for i in range(iterations):
            sample = net[rng.integers(0, n, size=n)]
            equity = 1.0 + np.cumsum(f * sample)
            finals[i] = equity[-1] - 1.0
            mdds[i] = max_drawdown(equity)

        def _pct(a: np.ndarray) -> dict[str, float]:
            return {
                "p5": float(np.percentile(a, 5)),
                "p50": float(np.percentile(a, 50)),
                "p95": float(np.percentile(a, 95)),
                "mean": float(a.mean()),
            }

        return MonteCarloResult(
            final_return=_pct(finals),
            max_drawdown=_pct(mdds),
            prob_negative=float((finals < 0).mean()),
            iterations=iterations,
        )


def _empty_stats() -> dict[str, float]:
    return {
        "n_trades": 0.0,
        "total_return": 0.0,
        "avg_net_return": 0.0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "sharpe_per_trade": 0.0,
        "max_drawdown": 0.0,
        "position_fraction": 0.0,
    }
