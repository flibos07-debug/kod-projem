# kod-projem

Production-grade ML pipeline for crypto trading signals, built in phases. Read
only from Binance **public** market-data endpoints — no keys, no order/trade
endpoints, no ability to place trades.

## Build phases

| Phase | Scope | Status |
| ----- | ----- | ------ |
| 1 | Skeleton, config schema/loader, Binance public REST client, HTF resample, logging | ✅ done |
| 2 | Feature engineering (5m/15m/1h/4h), ATR/ADX, regime classification | ✅ done |
| 3 | Triple-barrier labelling, nested walk-forward split (purge + embargo) | ✅ done |
| 4 | Base models + stacking meta-model, per-regime calibration | ✅ done |
| 5 | Conformal prediction, drift detection (PSI/JS), stability selection | ✅ done |
| 6 | Quality gate, cross-sectional ranking, backtest engine (Monte Carlo) | ✅ done |
| 7 | Live scanner loop (5m alignment), terminal output, reporting | ✅ done |

## Layout

```
config/default.yaml         # the default configuration
src/
  config_schema.py          # dataclass schema + validation + YAML loader
  logging_utils.py          # centralised logging setup
  data/
    binance_client.py       # read-only Binance public REST client
    resample.py             # base-timeframe -> HTF OHLCV resampling
  features/
    indicators.py           # ATR/ADX/RSI/EMA/Bollinger/vol (causal)
    engineering.py          # multi-timeframe feature matrix (no look-ahead)
  regime/
    classifier.py           # trend/range/high_vol + soft routing weights
  labeling/
    triple_barrier.py       # triple-barrier labels (+ t1 for purging)
  validation/
    walk_forward.py         # nested, purged, embargoed walk-forward folds
  models/
    base_models.py          # diverse base learners + meta-model factory
    stacking.py             # out-of-fold stacking ensemble + rank blend
    calibration.py          # per-regime isotonic/sigmoid calibration
  conformal/
    predictor.py            # Mondrian split-conformal prediction sets
  drift/
    detector.py             # PSI / Jensen-Shannon drift report
  selection/
    stability.py            # L1 stability selection
  quality/
    gate.py                 # promotion gate: Brier/ECE/top-k/stability/OOS
  ranking/
    cross_sectional.py      # per-timestamp ranking + top-N selection
  backtest/
    engine.py               # costed, capacity-capped backtest + Monte Carlo
  pipeline/
    training.py             # end-to-end training orchestrator
    artifacts.py            # model artifact save/load (joblib)
  scanner/
    live_scanner.py         # 5m-aligned scan loop + cross-sectional ranking
  reporting/
    report.py               # terminal table, CSV persistence, retention
  utils/
    timing.py               # timeframe-boundary scheduling helpers
  main.py                   # CLI: `train` and `scan`
tests/
```

## Command line

```bash
# Train per-symbol models from Binance public data (last 180 days)
python -m src.main train --symbols BTCUSDT ETHUSDT --days 180

# Run a single aligned scan using the trained artifacts
python -m src.main scan --symbols BTCUSDT ETHUSDT --once

# Or run the continuous 5m-aligned scanner loop
python -m src.main scan --symbols BTCUSDT ETHUSDT
```

`train` runs the full pipeline (features → labels → regime → purged
walk-forward → stacking → per-regime calibration → conformal → quality gate)
and saves a `ModelArtifact` per symbol under `models/`. Only the market
data-fetching commands touch the network, and only public read-only endpoints.

## Library usage

```python
from src.config_schema import load_default, AppConfig

# Load and validate the bundled default config
config = load_default()
print(config.ensemble.meta_model)  # -> "logistic_regression"

# Or load any YAML file
config = AppConfig.load("config/default.yaml")
```

Loading raises `ConfigError` with a dotted path (e.g. `ensemble.ranking_weight`)
if any value is missing, mistyped, out of range, or violates a cross-field rule.

## Config sections

| Section        | Purpose                                                        |
| -------------- | -------------------------------------------------------------- |
| `regime`       | Market-regime classification and (soft) routing thresholds     |
| `ensemble`     | Stacking / ranking ensemble; weights must sum to 1.0           |
| `validation`   | Walk-forward + nested CV windowing (purge/embargo bars)        |
| `calibration`  | Probability calibration method, per-regime toggle              |
| `conformal`    | Conformal prediction miscoverage levels (`alpha_90 < alpha_80`)|
| `drift`        | PSI / JS drift thresholds and check interval                   |
| `quality_gate` | Promotion gate the model must clear before deployment          |
| `backtest`     | Fee/slippage assumptions, position limits, Monte Carlo iters   |
| `storage`      | Filesystem layout and scan retention                           |

## Validation

The schema enforces types, ranges, allowed-value whitelists, unknown-key
rejection, and cross-field invariants, including:

- `ensemble.ranking_weight + ensemble.probability_weight == 1.0`
- `conformal.alpha_90 < conformal.alpha_80`
- `drift.psi_threshold_medium < drift.psi_threshold_high`
- `regime.adx_threshold_range <= regime.adx_threshold_trend`
- `calibration.method` in `{isotonic, sigmoid, beta}`

## Requirements

```
pip install -r requirements.txt   # PyYAML
```

## Tests

```
python -m unittest discover -s tests -v
```
