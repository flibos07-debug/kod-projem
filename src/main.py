"""Command-line entrypoint: train models and run the live scanner.

Examples
--------
Train per-symbol models from Binance public data::

    python -m src.main train --symbols BTCUSDT ETHUSDT --days 180

Run one aligned scan (or loop) using the trained artifacts::

    python -m src.main scan --symbols BTCUSDT ETHUSDT --once
"""

from __future__ import annotations

import argparse
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Keep the CLI output readable: solver non-convergence during selection is
# expected and harmless, so don't flood the terminal with it.
try:  # pragma: no cover - defensive import
    from sklearn.exceptions import ConvergenceWarning

    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except Exception:  # pragma: no cover
    pass
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from .config_schema import AppConfig
from .data.binance_client import BinanceClient
from .data.binance_futures_client import BinanceFuturesClient
from .data.resample import resample_to_timeframes
from .data.synthetic import SyntheticClient
from .logging_utils import setup_logging, get_logger
from .pipeline.artifacts import load_artifact, save_artifact
from .pipeline.directional import DirectionalPipeline
from .pipeline.training import PipelineParams, TrainingPipeline
from .scanner.futures_scanner import FuturesScanner
from .scanner.live_scanner import LiveScanner

logger = get_logger(__name__)


def _load_config(path: str) -> AppConfig:
    return AppConfig.load(path)


def _artifact_path(config: AppConfig, symbol: str) -> Path:
    return Path(config.storage.model_directory) / f"{symbol.upper()}.joblib"


def _make_client(args: argparse.Namespace):
    """Return a data client for the selected source/market."""
    source = getattr(args, "source", "binance")
    market = getattr(args, "market", "spot")
    if source == "synthetic":
        logger.info("Using OFFLINE synthetic data source")
        return SyntheticClient()
    if market == "futures":
        return BinanceFuturesClient()
    return BinanceClient()


def cmd_train(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    params = PipelineParams()
    pipeline = TrainingPipeline(config, params, fast=args.fast)

    start = datetime.now(timezone.utc) - timedelta(days=args.days)
    passed = 0
    with _make_client(args) as client:
        for symbol in args.symbols:
            logger.info("Fetching %s %s klines since %s", symbol, params.base_timeframe, start.date())
            base = client.get_klines_range(symbol, params.base_timeframe, start_time=start)
            frames = resample_to_timeframes(
                base, [params.base_timeframe, *params.htf_timeframes],
                base_timeframe=params.base_timeframe, drop_incomplete=True,
            )
            try:
                result = pipeline.run(frames, symbol=symbol)
            except ValueError as exc:
                logger.error("Training skipped for %s: %s", symbol, exc)
                continue
            print(f"\n[{symbol}] {result.gate.summary()}")
            save_artifact(result.artifact, _artifact_path(config, symbol))
            passed += int(result.gate.passed)
    logger.info("Trained %d symbol(s); %d passed the quality gate", len(args.symbols), passed)
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    artifacts = {}
    for symbol in args.symbols:
        path = _artifact_path(config, symbol)
        if path.exists():
            artifacts[symbol.upper()] = load_artifact(path)
        else:
            logger.warning("No artifact for %s at %s; run `train` first", symbol, path)
    if not artifacts:
        logger.error("No artifacts available to scan")
        return 1

    with _make_client(args) as client:
        scanner = LiveScanner(config, artifacts, client)
        if args.once:
            results = scanner.scan_once()
            scanner._emit(results)
        else:
            scanner.run(max_iterations=args.iterations)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Fully offline end-to-end run: train on synthetic data, then scan once.

    Uses a 1h base timeframe so the multi-month walk-forward stays fast while
    still exercising the entire pipeline and scanner.
    """
    config = _load_config(args.config)
    params = PipelineParams(base_timeframe="1h", htf_timeframes=("4h",))
    pipeline = TrainingPipeline(config, params, fast=True)

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=args.days)
    client = SyntheticClient(now=now)

    artifacts = {}
    logger.info("=== DEMO: training on synthetic data (%d-day span, 1h base) ===", args.days)
    for symbol in args.symbols:
        base = client.get_klines_range(symbol, params.base_timeframe, start_time=start, end_time=now)
        frames = resample_to_timeframes(
            base, [params.base_timeframe, *params.htf_timeframes],
            base_timeframe=params.base_timeframe, drop_incomplete=True,
        )
        try:
            result = pipeline.run(frames, symbol=symbol, do_selection=True)
        except ValueError as exc:
            logger.error("Demo training skipped for %s: %s", symbol, exc)
            continue
        print(f"\n[{symbol}] {result.gate.summary()}")
        save_artifact(result.artifact, _artifact_path(config, symbol))
        artifacts[symbol.upper()] = result.artifact

    if not artifacts:
        logger.error("Demo produced no artifacts")
        return 1

    logger.info("=== DEMO: running a single offline scan ===")
    scanner = LiveScanner(config, artifacts, client)
    results = scanner.scan_once()
    scanner._emit(results)
    return 0


def cmd_futures_demo(args: argparse.Namespace) -> int:
    """Offline two-sided (long/short) futures scan on synthetic data.

    Trains a long AND a short model per symbol, then runs one futures scan
    emitting LONG / SHORT / FLAT signals — entirely offline.
    """
    config = _load_config(args.config)
    params = PipelineParams(base_timeframe="1h", htf_timeframes=("4h",))
    pipeline = DirectionalPipeline(config, params, fast=True)

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=args.days)
    client = SyntheticClient(now=now)

    artifacts = {}
    logger.info("=== FUTURES DEMO: training long+short on synthetic data (%d-day span) ===", args.days)
    for symbol in args.symbols:
        base = client.get_klines_range(symbol, params.base_timeframe, start_time=start, end_time=now)
        frames = resample_to_timeframes(
            base, [params.base_timeframe, *params.htf_timeframes],
            base_timeframe=params.base_timeframe, drop_incomplete=True,
        )
        try:
            result = pipeline.run(frames, symbol=symbol, with_short=True)
        except ValueError as exc:
            logger.error("Futures demo training skipped for %s: %s", symbol, exc)
            continue
        lg = result.long.gate.passed
        sg = result.short.gate.passed if result.short else None
        print(f"[{symbol}] long gate={'PASS' if lg else 'FAIL'}, "
              f"short gate={'PASS' if sg else ('FAIL' if sg is False else 'n/a')}")
        save_artifact(result.artifact, Path(config.storage.model_directory) / f"{symbol.upper()}_dir.joblib")
        artifacts[symbol.upper()] = result.artifact

    if not artifacts:
        logger.error("Futures demo produced no artifacts")
        return 1

    logger.info("=== FUTURES DEMO: running a single offline long/short scan ===")
    scanner = FuturesScanner(config, artifacts, client)
    results = scanner.scan_once()
    scanner._emit(results)
    return 0


def _directional_artifact_path(config: AppConfig, symbol: str) -> Path:
    return Path(config.storage.model_directory) / f"{symbol.upper()}_dir.joblib"


def _discover_top_futures(client, n: int, *, quote_asset: str = "USDT") -> list[str]:
    """Top-N USDT perpetuals by 24h quote volume (universe discovery + ranking)."""
    symbols = set(client.get_perpetual_symbols(quote_asset=quote_asset))
    tickers = client.get_ticker_24h()  # all symbols
    tickers = tickers[tickers["symbol"].isin(symbols)]
    if "quoteVolume" in tickers.columns:
        tickers = tickers.sort_values("quoteVolume", ascending=False)
    top = tickers["symbol"].head(n).tolist()
    logger.info("Universe: %d perpetuals, scanning top %d by volume", len(symbols), len(top))
    return top


def _htf_for_base(base_timeframe: str) -> tuple[str, ...]:
    """A sensible higher-timeframe set for a given base timeframe."""
    return {
        "5m": ("15m", "1h", "4h"),
        "15m": ("1h", "4h"),
        "1h": ("4h",),
    }.get(base_timeframe, ("4h",))


def cmd_futures_train(args: argparse.Namespace) -> int:
    """Train long+short models for a Futures symbol set (or top-N by volume)."""
    args.market = "futures"
    config = _load_config(args.config)
    tf = getattr(args, "timeframe", "5m")
    params = PipelineParams(base_timeframe=tf, htf_timeframes=_htf_for_base(tf))
    pipeline = DirectionalPipeline(config, params, fast=args.fast)
    do_selection = not getattr(args, "no_select", False)
    start = datetime.now(timezone.utc) - timedelta(days=args.days)

    with _make_client(args) as client:
        symbols = args.symbols
        if args.top and hasattr(client, "get_perpetual_symbols"):
            symbols = _discover_top_futures(client, args.top)
        if not symbols:
            logger.error("No symbols to train (pass --symbols or --top N)")
            return 1
        for symbol in symbols:
            base = client.get_klines_range(symbol, params.base_timeframe, start_time=start)
            frames = resample_to_timeframes(
                base, [params.base_timeframe, *params.htf_timeframes],
                base_timeframe=params.base_timeframe, drop_incomplete=True,
            )
            try:
                result = pipeline.run(frames, symbol=symbol, do_selection=do_selection)
            except ValueError as exc:
                logger.error("Training skipped for %s: %s", symbol, exc)
                continue
            lg = result.long.gate.passed
            sg = result.short.gate.passed if result.short else None
            print(f"[{symbol}] long gate={'PASS' if lg else 'FAIL'}, "
                  f"short gate={'PASS' if sg else ('FAIL' if sg is False else 'n/a')}")
            save_artifact(result.artifact, _directional_artifact_path(config, symbol))
    return 0


def cmd_futures_scan(args: argparse.Namespace) -> int:
    """Live two-sided scan: load directional artifacts and emit LONG/SHORT/FLAT."""
    args.market = "futures"
    config = _load_config(args.config)

    model_dir = Path(config.storage.model_directory)
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        symbols = [p.name[: -len("_dir.joblib")] for p in model_dir.glob("*_dir.joblib")]
    if not symbols:
        logger.error("No directional artifacts found; run `futures-train` first")
        return 1

    artifacts = {}
    for symbol in symbols:
        path = _directional_artifact_path(config, symbol)
        if path.exists():
            artifacts[symbol] = load_artifact(path)
        else:
            logger.warning("No artifact for %s at %s", symbol, path)
    if not artifacts:
        logger.error("No usable artifacts to scan")
        return 1

    with _make_client(args) as client:
        scanner = FuturesScanner(config, artifacts, client)
        if args.once:
            results = scanner.scan_once()
            scanner._emit(results)
            if not results.empty:
                from .reporting.report import save_scan
                save_scan(results, config.storage.report_directory)
        else:
            scanner.run(max_iterations=args.iterations)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ml-signals", description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="path to the YAML config")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--source", choices=["binance", "synthetic"], default="binance",
        help="data source: live Binance public data or offline synthetic",
    )
    parser.add_argument(
        "--market", choices=["spot", "futures"], default="spot",
        help="live market to read from (spot or USD‑M futures)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train per-symbol models")
    p_train.add_argument("--symbols", nargs="+", required=True)
    p_train.add_argument("--days", type=int, default=180, help="history window in days")
    p_train.add_argument("--fast", action="store_true", help="use fast/small model settings")
    p_train.set_defaults(func=cmd_train)

    p_scan = sub.add_parser("scan", help="run the live scanner")
    p_scan.add_argument("--symbols", nargs="+", required=True)
    p_scan.add_argument("--once", action="store_true", help="run a single scan and exit")
    p_scan.add_argument("--iterations", type=int, default=None, help="bound the loop")
    p_scan.set_defaults(func=cmd_scan)

    p_demo = sub.add_parser("demo", help="offline end-to-end run on synthetic data")
    p_demo.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    p_demo.add_argument("--days", type=int, default=300, help="synthetic history span in days")
    p_demo.set_defaults(func=cmd_demo)

    p_fdemo = sub.add_parser("futures-demo", help="offline long/short futures scan on synthetic data")
    p_fdemo.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    p_fdemo.add_argument("--days", type=int, default=300, help="synthetic history span in days")
    p_fdemo.set_defaults(func=cmd_futures_demo)

    p_ftrain = sub.add_parser("futures-train", help="train long+short models for futures symbols")
    p_ftrain.add_argument("--symbols", nargs="*", default=[], help="explicit symbols (or use --top)")
    p_ftrain.add_argument("--top", type=int, default=None, help="auto-pick top-N USDT perpetuals by 24h volume")
    p_ftrain.add_argument("--days", type=int, default=365, help="history window in days")
    p_ftrain.add_argument(
        "--timeframe", choices=["5m", "15m", "1h"], default="1h",
        help="base timeframe (1h is fast; 5m is heaviest)",
    )
    p_ftrain.add_argument("--no-select", action="store_true", help="skip stability selection (faster)")
    p_ftrain.add_argument("--fast", action="store_true")
    p_ftrain.set_defaults(func=cmd_futures_train)

    p_fscan = sub.add_parser("futures-scan", help="live long/short scan of trained futures symbols")
    p_fscan.add_argument("--symbols", nargs="*", default=[], help="symbols to scan (default: all trained)")
    p_fscan.add_argument("--once", action="store_true", help="run a single scan and exit")
    p_fscan.add_argument("--iterations", type=int, default=None, help="bound the loop")
    p_fscan.set_defaults(func=cmd_futures_scan)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level, log_dir=None)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
