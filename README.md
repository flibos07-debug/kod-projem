# kod-projem

Typed, validated configuration schema for an ML trading-signal model.

## Layout

```
config/default.yaml     # the default configuration
src/config_schema.py    # dataclass schema + validation + YAML loader
tests/test_config_schema.py
```

## Usage

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
