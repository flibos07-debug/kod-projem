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
from .labeling.triple_barrier import BarrierParams
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


def _discover_top_futures(client, n: int, *, offset: int = 0, quote_asset: str = "USDT") -> list[str]:
    """USDT perpetuals ranked by 24h quote volume, sliced ``[offset:offset+n]``."""
    symbols = set(client.get_perpetual_symbols(quote_asset=quote_asset))
    tickers = client.get_ticker_24h()  # all symbols
    tickers = tickers[tickers["symbol"].isin(symbols)]
    if "quoteVolume" in tickers.columns:
        tickers = tickers.sort_values("quoteVolume", ascending=False)
    ranked = tickers["symbol"].tolist()
    chunk = ranked[offset: offset + n]
    logger.info(
        "Universe: %d perpetuals; training rank %d-%d (%d coins)",
        len(symbols), offset + 1, offset + len(chunk), len(chunk),
    )
    return chunk


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
    barrier = BarrierParams(
        tp_mult=args.tp_mult, sl_mult=args.sl_mult, max_holding=args.max_holding,
    )
    params = PipelineParams(base_timeframe=tf, htf_timeframes=_htf_for_base(tf), barrier=barrier)
    logger.info(
        "Barrier: TP=%.1f*ATR, SL=%.1f*ATR, max_holding=%d %s bars",
        args.tp_mult, args.sl_mult, args.max_holding, tf,
    )
    pipeline = DirectionalPipeline(config, params, fast=args.fast)
    do_selection = not getattr(args, "no_select", False)
    start = datetime.now(timezone.utc) - timedelta(days=args.days)

    with _make_client(args) as client:
        symbols = args.symbols
        if args.top and hasattr(client, "get_perpetual_symbols"):
            symbols = _discover_top_futures(client, args.top, offset=getattr(args, "offset", 0))
        if not symbols:
            logger.error("No symbols to train (pass --symbols or --top N)")
            return 1
        skip_existing = getattr(args, "skip_existing", False)
        total = len(symbols)
        trained = 0
        for i, symbol in enumerate(symbols, 1):
            path = _directional_artifact_path(config, symbol)
            if skip_existing and path.exists():
                logger.info("[%d/%d] %s already trained, skipping", i, total, symbol)
                continue
            try:
                base = client.get_klines_range(symbol, params.base_timeframe, start_time=start)
                frames = resample_to_timeframes(
                    base, [params.base_timeframe, *params.htf_timeframes],
                    base_timeframe=params.base_timeframe, drop_incomplete=True,
                )
                result = pipeline.run(frames, symbol=symbol, do_selection=do_selection)
            except (ValueError, KeyError) as exc:
                logger.error("[%d/%d] Training skipped for %s: %s", i, total, symbol, exc)
                continue
            except Exception as exc:  # keep going through a big universe
                logger.error("[%d/%d] Unexpected error for %s: %s", i, total, symbol, exc)
                continue
            lg = result.long.gate.passed
            sg = result.short.gate.passed if result.short else None
            print(f"[{i}/{total}] {symbol}: long={'PASS' if lg else 'FAIL'}, "
                  f"short={'PASS' if sg else ('FAIL' if sg is False else 'n/a')}")
            save_artifact(result.artifact, path)
            trained += 1
    logger.info("Trained %d/%d symbols", trained, total)
    return 0


def _load_directional_artifacts(config: AppConfig, symbols: list[str]) -> dict:
    model_dir = Path(config.storage.model_directory)
    if not symbols:
        symbols = [p.name[: -len("_dir.joblib")] for p in model_dir.glob("*_dir.joblib")]
    artifacts = {}
    for symbol in symbols:
        path = _directional_artifact_path(config, symbol.upper())
        if path.exists():
            artifacts[symbol.upper()] = load_artifact(path)
        else:
            logger.warning("No artifact for %s at %s", symbol, path)
    return artifacts


def cmd_futures_signals(args: argparse.Namespace) -> int:
    """Pro report: top-N LONG and SHORT setups with entry zone, SL, TP1, TP2."""
    args.market = "futures"
    config = _load_config(args.config)
    from .reporting.report import render_signals, save_signals

    artifacts = _load_directional_artifacts(config, [s.upper() for s in args.symbols])
    if not artifacts:
        logger.error("No directional artifacts found; run `futures-train` first")
        return 1

    with _make_client(args) as client:
        fundamentals = _fetch_fundamentals(client, list(artifacts))
        scanner = FuturesScanner(config, artifacts, client)
        max_move = None if args.max_move <= 0 else args.max_move
        verdict_rank = {"all": 0, "dikkatli": 1, "uygun": 2}[args.min_verdict]
        signals = scanner.scan_signals(
            top_n=args.top_n, max_move_pct=max_move,
            fundamentals=fundamentals, min_quote_volume=args.min_volume * 1e6,
            min_verdict=verdict_rank, confirm_trend=not args.no_confirm_trend,
        )
        _enrich_fundamentals(client, signals)

    print(f"\n=== Futures sinyalleri @ {datetime.now(timezone.utc).isoformat()} ===")
    print(f"(taranan: {len(artifacts)} coin | TP2 ≤ %{args.max_move} | min hacim: {args.min_volume}M$)\n")
    print(render_signals(signals))
    save_signals(signals, config.storage.report_directory)

    if args.html:
        from .reporting.html_report import save_html

        meta = {"scanned": len(artifacts), "max_move": args.max_move, "min_volume": args.min_volume}
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        html_path = Path(config.storage.report_directory) / f"signals_{ts}.html"
        save_html(signals, html_path, meta=meta)
        print(f"\n🌐 HTML rapor: {html_path.resolve()}")
        if args.open_html:
            import webbrowser
            webbrowser.open(html_path.resolve().as_uri())
    return 0


def _fetch_fundamentals(client, symbols: list[str]) -> dict:
    """Bulk liquidity + funding for the whole set in just two API calls.

    L/S ratio and open interest are per-symbol endpoints (no bulk form), so they
    are fetched later — only for the handful of coins that make the final list
    (see ``_enrich_fundamentals``).
    """
    fundamentals: dict[str, dict] = {s: {} for s in symbols}
    try:
        tickers = client.get_ticker_24h()
        vol = dict(zip(tickers["symbol"], tickers.get("quoteVolume", [])))
        chg = dict(zip(tickers["symbol"], tickers.get("priceChangePercent", [])))
        for s in symbols:
            fundamentals[s]["quote_volume"] = float(vol.get(s, float("nan")))
            try:
                fundamentals[s]["chg24h"] = float(chg.get(s, float("nan")))
            except (TypeError, ValueError):
                pass
    except Exception as exc:
        logger.debug("24h ticker unavailable: %s", exc)
    try:
        if hasattr(client, "get_all_funding"):
            fmap = client.get_all_funding()  # one request, all symbols
            for s in symbols:
                if s in fmap:
                    fundamentals[s]["funding"] = fmap[s]
    except Exception as exc:
        logger.debug("bulk funding unavailable: %s", exc)
    return fundamentals


def _enrich_fundamentals(client, signals: dict) -> None:
    """Fill L/S ratio and open interest for just the surfaced coins (in place)."""
    symbols = []
    for side in ("long", "short"):
        df = signals.get(side)
        if df is not None and not df.empty:
            symbols += list(df["symbol"])
    symbols = list(dict.fromkeys(symbols))
    if not symbols:
        return

    import concurrent.futures as cf

    ls: dict[str, float] = {}
    oi: dict[str, float] = {}
    oic: dict[str, float] = {}
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        ls_f = {ex.submit(client.get_long_short_ratio, s): s for s in symbols} if hasattr(client, "get_long_short_ratio") else {}
        oi_f = {ex.submit(client.get_open_interest, s): s for s in symbols} if hasattr(client, "get_open_interest") else {}
        oic_f = {ex.submit(client.get_open_interest_change, s): s for s in symbols} if hasattr(client, "get_open_interest_change") else {}
        for fut, s in ls_f.items():
            try:
                ls[s] = fut.result()
            except Exception:
                pass
        for fut, s in oi_f.items():
            try:
                oi[s] = fut.result()
            except Exception:
                pass
        for fut, s in oic_f.items():
            try:
                oic[s] = fut.result()
            except Exception:
                pass

    for side in ("long", "short"):
        df = signals.get(side)
        if df is not None and not df.empty:
            df["ls_ratio"] = df["symbol"].map(ls).fillna(df["ls_ratio"])
            df["open_interest"] = df["symbol"].map(oi).fillna(df["open_interest"])
            if "oi_change" in df.columns:
                df["oi_change"] = df["symbol"].map(oic).fillna(df["oi_change"])


def cmd_futures_screener(args: argparse.Namespace) -> int:
    """Multi-timeframe (5m/15m/1h) technical screener — LONG/SHORT candidates."""
    args.market = "futures"
    config = _load_config(args.config)
    from .reporting.screener_report import render_screener, save_screener_html
    from .scanner.screener import Screener, ScreenerParams

    with _make_client(args) as client:
        if args.symbols:
            symbols = [s.upper() for s in args.symbols]
        elif args.top and hasattr(client, "get_perpetual_symbols"):
            symbols = _discover_top_futures(client, args.top)
        elif hasattr(client, "get_perpetual_symbols"):
            symbols = client.get_perpetual_symbols()
        else:
            logger.error("No symbols and no universe discovery on this client")
            return 1

        logger.info("Screening %d symbols on 5m/15m/1h…", len(symbols))
        fundamentals = _fetch_fundamentals(client, symbols)
        params = ScreenerParams(tp_mult=args.tp_mult, sl_mult=args.sl_mult)
        screener = Screener(client, params=params)
        signals = screener.scan(
            symbols, top_n=args.top_n, fundamentals=fundamentals,
            min_quote_volume=args.min_volume * 1e6, min_score=args.min_score,
            rsi_below=args.rsi_below, rsi_above=args.rsi_above,
            require_squeeze=args.squeeze, require_vol_spike=args.vol_spike,
            only_early=args.early, max_ext_pct=args.max_ext,
        )
        _enrich_fundamentals(client, signals)

    print(f"\n=== Futures Screener (5m/15m/1h) @ {datetime.now(timezone.utc).isoformat()} ===")
    print(f"(taranan: {len(symbols)} coin | eğitim yok)\n")
    print(render_screener(signals))
    if args.html:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        html_path = Path(config.storage.report_directory) / f"screener_{ts}.html"
        save_screener_html(signals, html_path, meta={"scanned": len(symbols)})
        print(f"\n🌐 HTML: {html_path.resolve()}")
        if args.open_html:
            import webbrowser
            webbrowser.open(html_path.resolve().as_uri())
    return 0


def _giris_count(signals: dict) -> int:
    n = 0
    for side in ("long", "short"):
        df = signals.get(side)
        if df is not None and not df.empty and "stage" in df.columns:
            n += int((df["stage"] == "GİRİŞ").sum())
    return n


def cmd_futures_bbstrat(args: argparse.Namespace) -> int:
    """Bollinger mid-band cross strategy scan (+ live loop synced to 5m closes)."""
    args.market = "futures"
    config = _load_config(args.config)
    import os
    import time as _time

    from .reporting.strategy_report import render_strategy, save_strategy_html
    from .scanner.bb_retest import BBRetestScanner, BBStratParams
    from .utils.timing import next_boundary, seconds_until

    with _make_client(args) as client:
        if args.symbols:
            symbols = [s.upper() for s in args.symbols]
        elif args.top and hasattr(client, "get_perpetual_symbols"):
            symbols = _discover_top_futures(client, args.top)
        elif hasattr(client, "get_perpetual_symbols"):
            symbols = client.get_perpetual_symbols()
        else:
            logger.error("No symbols and no universe discovery")
            return 1
        params = BBStratParams(
            require_squeeze=not args.no_squeeze,
            require_htf=not args.no_htf,
            min_confidence=args.min_confidence,
        )
        scanner = BBRetestScanner(client, params=params)

        def one_pass(iteration: int) -> None:
            fundamentals = _fetch_fundamentals(client, symbols)
            signals = scanner.scan(
                symbols, top_n=args.top_n, fundamentals=fundamentals,
                min_quote_volume=args.min_volume * 1e6, include_watch=args.watch,
            )
            _enrich_fundamentals(client, {"long": signals["long"], "short": signals["short"]})

            if args.loop:
                os.system("cls" if os.name == "nt" else "clear")
            now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")
            print(f"=== BB Orta Bant Kırılımı @ {now} "
                  f"{'(canlı #%d)' % iteration if args.loop else ''} ===")
            print(f"(taranan: {len(symbols)} coin | min hacim: {args.min_volume}M$)\n")
            n_giris = _giris_count(signals)
            if n_giris:
                print("\a")  # terminal bell
                print(f"🔔🔔  {n_giris} COİN GİRİŞ AŞAMASINDA — 5m onayladı, gir!  🔔🔔\n")
            print(render_strategy(signals))
            if args.html:
                html_path = Path(config.storage.report_directory) / "bbstrat_live.html"
                save_strategy_html(signals, html_path, meta={"scanned": len(symbols)})
                if iteration == 1 and args.open_html:
                    import webbrowser
                    webbrowser.open(html_path.resolve().as_uri())

        iteration = 1
        try:
            while True:
                one_pass(iteration)
                if not args.loop:
                    break
                # Sync to 5m candle closes: wake just after the next boundary so
                # every scan reads freshly-closed candles ("every bar close").
                target = next_boundary(datetime.now(timezone.utc), "5m")
                wait = seconds_until(target) + 2.0  # small buffer for exchange lag
                mm = target.astimezone().strftime("%H:%M")
                print(f"\n⏳ Sıradaki 5m kapanışına senkron: {mm} "
                      f"(~{int(wait)}s) — durdurmak için Ctrl+C")
                _time.sleep(wait)
                iteration += 1
        except KeyboardInterrupt:
            print("\nDurduruldu.")
    return 0


def cmd_futures_breakout(args: argparse.Namespace) -> int:
    """Training-free technical scan of the whole universe (breakout/momentum)."""
    args.market = "futures"
    config = _load_config(args.config)
    from .reporting.report import render_signals, save_signals
    from .scanner.momentum_scanner import MomentumParams, MomentumScanner

    with _make_client(args) as client:
        if args.symbols:
            symbols = [s.upper() for s in args.symbols]
        elif args.top and hasattr(client, "get_perpetual_symbols"):
            symbols = _discover_top_futures(client, args.top)
        elif hasattr(client, "get_perpetual_symbols"):
            symbols = client.get_perpetual_symbols()  # the whole universe
        else:
            logger.error("No symbols and this client has no universe discovery")
            return 1

        logger.info("Breakout-scanning %d symbols (no training needed)…", len(symbols))
        fundamentals = _fetch_fundamentals(client, symbols)
        params = MomentumParams(tp_mult=args.tp_mult, sl_mult=args.sl_mult)
        scanner = MomentumScanner(client, base_timeframe=args.timeframe, params=params)
        max_move = None if args.max_move <= 0 else args.max_move
        signals = scanner.scan_signals(
            symbols, top_n=args.top_n, max_move_pct=max_move,
            fundamentals=fundamentals, min_quote_volume=args.min_volume * 1e6,
            min_score=args.min_score,
        )
        _enrich_fundamentals(client, signals)

    print(f"\n=== Breakout/Momentum taraması @ {datetime.now(timezone.utc).isoformat()} ===")
    print(f"(taranan: {len(symbols)} coin | {args.timeframe} | eğitim yok)\n")
    print(render_signals(signals))
    save_signals(signals, config.storage.report_directory)
    if args.html:
        from .reporting.html_report import save_html

        meta = {"scanned": len(symbols), "max_move": args.max_move, "min_volume": args.min_volume}
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        html_path = Path(config.storage.report_directory) / f"breakout_{ts}.html"
        save_html(signals, html_path, meta=meta)
        print(f"\n🌐 HTML rapor: {html_path.resolve()}")
        if args.open_html:
            import webbrowser
            webbrowser.open(html_path.resolve().as_uri())
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
    p_ftrain.add_argument("--offset", type=int, default=0, help="skip the first N ranked coins (for training in chunks)")
    p_ftrain.add_argument("--skip-existing", action="store_true", help="skip coins already trained (resume a big run)")
    p_ftrain.add_argument("--days", type=int, default=365, help="history window in days")
    p_ftrain.add_argument(
        "--timeframe", choices=["5m", "15m", "1h"], default="1h",
        help="base timeframe (1h is fast; 5m is heaviest)",
    )
    p_ftrain.add_argument("--no-select", action="store_true", help="skip stability selection (faster)")
    p_ftrain.add_argument("--tp-mult", type=float, default=2.0, help="take-profit distance in ATRs (bigger = larger TP)")
    p_ftrain.add_argument("--sl-mult", type=float, default=1.0, help="stop-loss distance in ATRs")
    p_ftrain.add_argument("--max-holding", type=int, default=12, help="max bars to reach the target (vertical barrier)")
    p_ftrain.add_argument("--fast", action="store_true")
    p_ftrain.set_defaults(func=cmd_futures_train)

    p_fscan = sub.add_parser("futures-scan", help="live long/short scan of trained futures symbols")
    p_fscan.add_argument("--symbols", nargs="*", default=[], help="symbols to scan (default: all trained)")
    p_fscan.add_argument("--once", action="store_true", help="run a single scan and exit")
    p_fscan.add_argument("--iterations", type=int, default=None, help="bound the loop")
    p_fscan.set_defaults(func=cmd_futures_scan)

    p_fsig = sub.add_parser("futures-signals", help="pro report: top LONG/SHORT setups with entry, SL, TP1, TP2")
    p_fsig.add_argument("--symbols", nargs="*", default=[], help="symbols (default: all trained)")
    p_fsig.add_argument("--top-n", type=int, default=10, help="how many per direction")
    p_fsig.add_argument("--max-move", type=float, default=5.0, help="max TP2 move %% (0 = no cap)")
    p_fsig.add_argument("--min-volume", type=float, default=0.0, help="min 24h quote volume in millions (liquidity gate)")
    p_fsig.add_argument("--min-verdict", choices=["all", "dikkatli", "uygun"], default="all",
                        help="only show setups at/above this verdict (hide ZAYIF etc.)")
    p_fsig.add_argument("--no-confirm-trend", action="store_true",
                        help="disable the trend filter (allow LONGs on falling coins)")
    p_fsig.add_argument("--html", action="store_true", help="also write a colored HTML dashboard")
    p_fsig.add_argument("--open-html", action="store_true", help="open the HTML report in the browser")
    p_fsig.set_defaults(func=cmd_futures_signals)

    p_fbrk = sub.add_parser("futures-breakout", help="training-free technical scan (breakout/momentum) of all coins")
    p_fbrk.add_argument("--symbols", nargs="*", default=[], help="symbols (default: whole universe)")
    p_fbrk.add_argument("--top", type=int, default=None, help="scan only the top-N by volume (default: all)")
    p_fbrk.add_argument("--top-n", type=int, default=10, help="how many per direction to show")
    p_fbrk.add_argument("--timeframe", choices=["15m", "1h", "4h"], default="1h", help="chart timeframe")
    p_fbrk.add_argument("--max-move", type=float, default=0.0, help="max TP2 move %% (0 = no cap)")
    p_fbrk.add_argument("--min-volume", type=float, default=0.0, help="min 24h quote volume in millions")
    p_fbrk.add_argument("--min-score", type=float, default=0.35, help="min technical score (0-1)")
    p_fbrk.add_argument("--tp-mult", type=float, default=3.0, help="take-profit distance in ATRs")
    p_fbrk.add_argument("--sl-mult", type=float, default=1.5, help="stop-loss distance in ATRs")
    p_fbrk.add_argument("--html", action="store_true", help="also write a colored HTML dashboard")
    p_fbrk.add_argument("--open-html", action="store_true", help="open the HTML report in the browser")
    p_fbrk.set_defaults(func=cmd_futures_breakout)

    p_scr = sub.add_parser("futures-screener", help="multi-timeframe (5m/15m/1h) technical screener")
    p_scr.add_argument("--symbols", nargs="*", default=[], help="symbols (default: whole universe)")
    p_scr.add_argument("--top", type=int, default=None, help="scan only top-N by volume (default: all)")
    p_scr.add_argument("--top-n", type=int, default=15, help="how many candidates per side to show")
    p_scr.add_argument("--min-score", type=float, default=0.35, help="min alignment score (0-1)")
    p_scr.add_argument("--min-volume", type=float, default=0.0, help="min 24h quote volume in millions")
    p_scr.add_argument("--rsi-below", type=float, default=None, help="only coins with 1h RSI below this")
    p_scr.add_argument("--rsi-above", type=float, default=None, help="only coins with 1h RSI above this")
    p_scr.add_argument("--squeeze", action="store_true", help="only Bollinger-squeeze coins (breakout imminent)")
    p_scr.add_argument("--vol-spike", action="store_true", help="only coins with a volume spike")
    p_scr.add_argument("--early", action="store_true", help="hide already-extended coins (only early/fresh moves)")
    p_scr.add_argument("--max-ext", type=float, default=None, help="max %% distance from 21-EMA (drop chasers)")
    p_scr.add_argument("--tp-mult", type=float, default=2.0, help="reference TP distance in ATRs")
    p_scr.add_argument("--sl-mult", type=float, default=1.0, help="reference SL distance in ATRs")
    p_scr.add_argument("--html", action="store_true", help="write a colored HTML screener")
    p_scr.add_argument("--open-html", action="store_true", help="open the HTML in the browser")
    p_scr.set_defaults(func=cmd_futures_screener)

    p_bb = sub.add_parser("futures-bbstrat", help="Bollinger orta bant kırılımı (15m/1h yön + 5m onay, retest yok)")
    p_bb.add_argument("--symbols", nargs="*", default=[], help="symbols (default: whole universe)")
    p_bb.add_argument("--top", type=int, default=None, help="scan only top-N by volume")
    p_bb.add_argument("--top-n", type=int, default=20, help="how many per section to show")
    p_bb.add_argument("--min-volume", type=float, default=0.0, help="min 24h quote volume in millions")
    p_bb.add_argument("--watch", action="store_true", help="ayrıca BEKLE (kesişti, onay bekliyor) coinleri göster")
    p_bb.add_argument("--min-confidence", type=float, default=60.0, help="GİRİŞ için minimum güven skoru (0-100); yükselt=daha az ama daha güçlü sinyal")
    p_bb.add_argument("--no-squeeze", action="store_true", help="sıkışma (yatay) şartını kaldır")
    p_bb.add_argument("--no-htf", action="store_true", help="1h teyidi şartını kaldır")
    p_bb.add_argument("--loop", action="store_true", help="canlı mod: her 5m mum kapanışında otomatik yenilenir")
    p_bb.add_argument("--html", action="store_true", help="write a colored HTML report")
    p_bb.add_argument("--open-html", action="store_true", help="open the HTML in the browser")
    p_bb.set_defaults(func=cmd_futures_bbstrat)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level, log_dir=None)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
