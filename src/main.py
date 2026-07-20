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
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config_schema import AppConfig
from .data.binance_client import BinanceClient
from .data.resample import resample_to_timeframes
from .data.synthetic import SyntheticClient
from .logging_utils import setup_logging, get_logger
from .pipeline.artifacts import load_artifact, save_artifact
from .pipeline.training import PipelineParams, TrainingPipeline
from .scanner.live_scanner import LiveScanner

logger = get_logger(__name__)


def _load_config(path: str) -> AppConfig:
    return AppConfig.load(path)


def _artifact_path(config: AppConfig, symbol: str) -> Path:
    return Path(config.storage.model_directory) / f"{symbol.upper()}.joblib"


def _make_client(args: argparse.Namespace):
    """Return a data client for the selected source (live Binance or offline)."""
    source = getattr(args, "source", "binance")
    if source == "synthetic":
        logger.info("Using OFFLINE synthetic data source")
        return SyntheticClient()
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ml-signals", description=__doc__)
    parser.add_argument("--config", default="config/default.yaml", help="path to the YAML config")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--source", choices=["binance", "synthetic"], default="binance",
        help="data source: live Binance public data or offline synthetic",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level, log_dir=None)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
