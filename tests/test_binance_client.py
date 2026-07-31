"""Tests for the Binance public REST client using a fake HTTP session.

No network access is required: a ``FakeSession`` returns canned responses and
records the requests it received.
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.binance_client import BinanceAPIError, BinanceClient, to_millis  # noqa: E402


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, headers=None, text=""):
        self.status_code = status_code
        self._json = json_data
        self.headers = headers or {}
        self.text = text

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


class FakeSession:
    """Returns queued responses; falls back to a default per-path handler."""

    def __init__(self, responses=None, handler=None):
        self._responses = list(responses or [])
        self._handler = handler
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params or {}})
        if self._responses:
            return self._responses.pop(0)
        if self._handler:
            return self._handler(url, params or {})
        return FakeResponse(200, [])

    def close(self):
        pass


def _kline_row(open_ms, close):
    return [
        open_ms,
        f"{close - 0.5}",
        f"{close + 1}",
        f"{close - 1}",
        f"{close}",
        "10.0",
        open_ms + 299_999,
        "1000.0",
        5,
        "6.0",
        "600.0",
        "0",
    ]


class ToMillisTests(unittest.TestCase):
    def test_none(self):
        self.assertIsNone(to_millis(None))

    def test_int_passthrough(self):
        self.assertEqual(to_millis(1_700_000_000_000), 1_700_000_000_000)

    def test_string_utc(self):
        self.assertEqual(to_millis("1970-01-01T00:00:01Z"), 1000)

    def test_naive_timestamp_treated_as_utc(self):
        self.assertEqual(to_millis(pd.Timestamp("1970-01-01 00:00:01")), 1000)

    def test_bool_rejected(self):
        with self.assertRaises(TypeError):
            to_millis(True)


class KlinesTests(unittest.TestCase):
    def test_klines_parsed_to_frame(self):
        rows = [_kline_row(0, 1), _kline_row(300_000, 2)]
        session = FakeSession(responses=[FakeResponse(200, rows)])
        client = BinanceClient(session=session)
        df = client.get_klines("btcusdt", "5m")
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns[:5]), ["open", "high", "low", "close", "volume"])
        self.assertEqual(str(df.index.tz), "UTC")
        self.assertEqual(df["trades"].dtype.kind, "i")
        self.assertAlmostEqual(df.iloc[1]["close"], 2.0)
        # Symbol upper-cased in the request.
        self.assertEqual(session.calls[0]["params"]["symbol"], "BTCUSDT")

    def test_invalid_interval_rejected(self):
        client = BinanceClient(session=FakeSession())
        with self.assertRaises(ValueError):
            client.get_klines("BTCUSDT", "7m")

    def test_empty_payload_returns_typed_empty_frame(self):
        session = FakeSession(responses=[FakeResponse(200, [])])
        client = BinanceClient(session=session)
        df = client.get_klines("BTCUSDT", "5m")
        self.assertTrue(df.empty)
        self.assertIn("close", df.columns)
        self.assertEqual(str(df.index.tz), "UTC")

    def test_pagination_walks_forward_and_stops(self):
        # Page 1: 1000 rows (full) -> triggers a second fetch. Page 2: 1 row.
        page1 = [_kline_row(i * 300_000, i + 1) for i in range(1000)]
        page2 = [_kline_row(1000 * 300_000, 1001)]
        session = FakeSession(
            responses=[FakeResponse(200, page1), FakeResponse(200, page2)]
        )
        client = BinanceClient(session=session)
        end = 1001 * 300_000 + 1
        df = client.get_klines_range("BTCUSDT", "5m", start_time=0, end_time=end)
        self.assertEqual(len(df), 1001)
        self.assertTrue(df.index.is_monotonic_increasing)
        self.assertFalse(df.index.has_duplicates)
        self.assertEqual(len(session.calls), 2)


class ErrorHandlingTests(unittest.TestCase):
    def test_4xx_raises_immediately(self):
        session = FakeSession(
            responses=[FakeResponse(400, {"code": -1121, "msg": "Invalid symbol."})]
        )
        client = BinanceClient(session=session, max_retries=3)
        with self.assertRaises(BinanceAPIError) as ctx:
            client.get_klines("NOPE", "5m")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.code, -1121)
        # Fail-fast: exactly one call, no retries.
        self.assertEqual(len(session.calls), 1)

    def test_5xx_retried_then_succeeds(self):
        rows = [_kline_row(0, 1)]
        session = FakeSession(
            responses=[FakeResponse(503, text="down"), FakeResponse(200, rows)]
        )
        client = BinanceClient(session=session, max_retries=3, backoff_base=0)
        df = client.get_klines("BTCUSDT", "5m")
        self.assertEqual(len(df), 1)
        self.assertEqual(len(session.calls), 2)

    def test_rate_limit_honoured_then_succeeds(self):
        rows = [_kline_row(0, 1)]
        session = FakeSession(
            responses=[
                FakeResponse(429, headers={"Retry-After": "0"}),
                FakeResponse(200, rows),
            ]
        )
        client = BinanceClient(session=session, max_retries=3, backoff_base=0)
        df = client.get_klines("BTCUSDT", "5m")
        self.assertEqual(len(df), 1)
        self.assertEqual(len(session.calls), 2)

    def test_exhausted_retries_raise(self):
        session = FakeSession(responses=[FakeResponse(500) for _ in range(5)])
        client = BinanceClient(session=session, max_retries=2, backoff_base=0)
        with self.assertRaises(BinanceAPIError):
            client.ping()


class ExchangeInfoTests(unittest.TestCase):
    def test_get_symbols_filters(self):
        info = {
            "symbols": [
                {"symbol": "BTCUSDT", "status": "TRADING", "quoteAsset": "USDT", "isSpotTradingAllowed": True},
                {"symbol": "ETHBTC", "status": "TRADING", "quoteAsset": "BTC", "isSpotTradingAllowed": True},
                {"symbol": "OLDUSDT", "status": "BREAK", "quoteAsset": "USDT", "isSpotTradingAllowed": True},
            ]
        }
        session = FakeSession(responses=[FakeResponse(200, info)])
        client = BinanceClient(session=session)
        symbols = client.get_symbols(quote_asset="USDT", status="TRADING")
        self.assertEqual(symbols, ["BTCUSDT"])

    def test_ticker_24h_numeric_coercion(self):
        payload = [
            {"symbol": "BTCUSDT", "lastPrice": "42000.5", "quoteVolume": "1000000.25", "priceChangePercent": "1.5"},
        ]
        session = FakeSession(responses=[FakeResponse(200, payload)])
        client = BinanceClient(session=session)
        df = client.get_ticker_24h()
        self.assertEqual(df.iloc[0]["lastPrice"], 42000.5)
        self.assertEqual(df["quoteVolume"].dtype.kind, "f")


if __name__ == "__main__":
    unittest.main()
