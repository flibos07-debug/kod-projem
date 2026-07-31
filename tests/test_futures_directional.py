"""Tests for the Futures client, directional pipeline and futures scanner."""

import dataclasses
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config_schema import AppConfig, ValidationConfig  # noqa: E402
from src.data.binance_futures_client import BinanceFuturesClient  # noqa: E402
from src.data.synthetic import SyntheticClient  # noqa: E402
from src.pipeline.directional import DirectionalArtifact, DirectionalPipeline  # noqa: E402
from src.pipeline.training import PipelineParams  # noqa: E402
from src.scanner.futures_scanner import FuturesScanner  # noqa: E402

DEFAULT_CONFIG = AppConfig.load(Path(__file__).resolve().parent.parent / "config" / "default.yaml")


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, headers=None, text=""):
        self.status_code = status_code
        self._json = json_data
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._json


class FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params or {}})
        return self._responses.pop(0)

    def close(self):
        pass


class FuturesClientTests(unittest.TestCase):
    def test_uses_fapi_paths(self):
        session = FakeSession([FakeResponse(200, [])])
        client = BinanceFuturesClient(session=session)
        client.ping_ok = client._request(client.PATH_KLINES, {"symbol": "BTCUSDT"})
        self.assertTrue(session.calls[0]["url"].endswith("/fapi/v1/klines"))

    def test_perpetual_universe_discovery(self):
        info = {
            "symbols": [
                {"symbol": "BTCUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL"},
                {"symbol": "ETHUSDT", "status": "TRADING", "quoteAsset": "USDT", "contractType": "PERPETUAL"},
                {"symbol": "BTCUSDT_240628", "status": "TRADING", "quoteAsset": "USDT", "contractType": "CURRENT_QUARTER"},
                {"symbol": "ADABUSD", "status": "TRADING", "quoteAsset": "BUSD", "contractType": "PERPETUAL"},
                {"symbol": "XRPUSDT", "status": "BREAK", "quoteAsset": "USDT", "contractType": "PERPETUAL"},
            ]
        }
        client = BinanceFuturesClient(session=FakeSession([FakeResponse(200, info)]))
        syms = client.get_perpetual_symbols(quote_asset="USDT")
        self.assertEqual(set(syms), {"BTCUSDT", "ETHUSDT"})  # quarterly/BUSD/BREAK excluded

    def test_base_url_is_futures(self):
        self.assertIn("fapi", BinanceFuturesClient().base_url)


def _small_config():
    small_validation = ValidationConfig(
        train_months=2, validation_months=1, test_months=1, step_months=1,
        purge_bars=6, embargo_bars=6, n_folds_nested=2,
    )
    return dataclasses.replace(DEFAULT_CONFIG, validation=small_validation)


class DirectionalPipelineTests(unittest.TestCase):
    def test_trains_long_and_short(self):
        client = SyntheticClient(now=datetime(2024, 6, 1, tzinfo=timezone.utc))
        start = datetime(2023, 8, 1, tzinfo=timezone.utc)
        base = client.get_klines_range("BTCUSDT", "1h", start_time=start, end_time=datetime(2024, 6, 1, tzinfo=timezone.utc))
        from src.data.resample import resample_to_timeframes

        frames = resample_to_timeframes(base, ["1h", "4h"], base_timeframe="1h", drop_incomplete=True)
        params = PipelineParams(base_timeframe="1h", htf_timeframes=("4h",))
        pipeline = DirectionalPipeline(_small_config(), params, fast=True)
        result = pipeline.run(frames, symbol="BTCUSDT", do_selection=False, with_short=True)

        self.assertIsInstance(result.artifact, DirectionalArtifact)
        self.assertIsNotNone(result.artifact.long)
        self.assertIsNotNone(result.artifact.short)
        self.assertGreaterEqual(result.long.n_folds, 1)
        self.assertGreaterEqual(result.short.n_folds, 1)
        # Both sides share one feature set.
        self.assertTrue(len(result.artifact.feature_names) > 0)


class FuturesScannerTests(unittest.TestCase):
    def _artifact(self):
        client = SyntheticClient(now=datetime(2024, 6, 1, tzinfo=timezone.utc))
        start = datetime(2023, 8, 1, tzinfo=timezone.utc)
        base = client.get_klines_range("BTCUSDT", "1h", start_time=start, end_time=datetime(2024, 6, 1, tzinfo=timezone.utc))
        from src.data.resample import resample_to_timeframes

        frames = resample_to_timeframes(base, ["1h", "4h"], base_timeframe="1h", drop_incomplete=True)
        params = PipelineParams(base_timeframe="1h", htf_timeframes=("4h",))
        pipeline = DirectionalPipeline(_small_config(), params, fast=True)
        return pipeline.run(frames, symbol="BTCUSDT", do_selection=False).artifact

    def test_scan_once_emits_signal(self):
        artifact = self._artifact()
        client = SyntheticClient(now=datetime(2024, 6, 2, tzinfo=timezone.utc))
        scanner = FuturesScanner(_small_config(), {"BTCUSDT": artifact}, client)
        results = scanner.scan_once()
        self.assertEqual(len(results), 1)
        row = results.iloc[0]
        self.assertIn(row["signal"], {"LONG", "SHORT", "FLAT"})
        self.assertIn("long_prob", results.columns)
        self.assertIn("short_prob", results.columns)
        self.assertIn(row["direction"], {"long", "short", "flat"})
        self.assertIn("selected", results.columns)


class CLITests(unittest.TestCase):
    def test_futures_train_parses_top(self):
        from src.main import build_parser

        args = build_parser().parse_args(["futures-train", "--top", "25", "--fast"])
        self.assertEqual(args.command, "futures-train")
        self.assertEqual(args.top, 25)
        self.assertTrue(args.fast)

    def test_futures_scan_defaults_to_all_trained(self):
        from src.main import build_parser

        args = build_parser().parse_args(["futures-scan", "--once"])
        self.assertEqual(args.symbols, [])  # empty -> discover from models dir
        self.assertTrue(args.once)

    def test_market_flag(self):
        from src.main import build_parser

        args = build_parser().parse_args(["--market", "futures", "futures-scan"])
        self.assertEqual(args.market, "futures")

    def test_discover_top_futures_ranks_by_volume(self):
        from src.main import _discover_top_futures

        class FakeFuturesClient:
            def get_perpetual_symbols(self, quote_asset="USDT"):
                return ["BTCUSDT", "ETHUSDT", "DOGEUSDT"]

            def get_ticker_24h(self):
                return pd.DataFrame(
                    {
                        "symbol": ["ETHUSDT", "BTCUSDT", "DOGEUSDT", "XRPUSDT"],
                        "quoteVolume": [5e9, 9e9, 1e9, 8e9],
                    }
                )

        top = _discover_top_futures(FakeFuturesClient(), 2)
        self.assertEqual(top, ["BTCUSDT", "ETHUSDT"])  # by volume, XRP not perpetual


if __name__ == "__main__":
    unittest.main()
