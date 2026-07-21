"""Read-only client for Binance public market-data endpoints.

This client deliberately implements **only** public market-data endpoints —
klines, exchange info and the 24h ticker. It never signs requests, holds API
keys or touches any account/order/trade endpoint, so it cannot place or cancel
orders. It is a data source, nothing more.

Design notes
------------
- Transient failures (HTTP 5xx, connection errors) are retried with exponential
  backoff. Rate-limit responses (429/418) honour the ``Retry-After`` header.
- Other 4xx responses raise :class:`BinanceAPIError` immediately with the
  exchange's error ``code``/``msg``.
- ``get_klines_range`` paginates the 1000-row klines limit automatically.
- Timestamps are returned as timezone-aware (UTC) ``DatetimeIndex``.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import pandas as pd
import requests

from ..logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_BASE_URL = "https://api.binance.com"

# Column names for the raw klines payload (12 fields per row).
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]

_FLOAT_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "taker_buy_base",
    "taker_buy_quote",
]

# Interval string -> milliseconds, used for kline pagination.
_INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}

_MAX_KLINES_LIMIT = 1000


class BinanceAPIError(RuntimeError):
    """Raised when Binance returns an error response."""

    def __init__(self, message: str, *, status_code: int | None = None, code: int | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


def to_millis(value: int | float | str | datetime | pd.Timestamp | None) -> int | None:
    """Normalise a variety of time representations to epoch milliseconds (UTC)."""
    if value is None:
        return None
    if isinstance(value, bool):  # guard: bool is an int subclass
        raise TypeError("boolean is not a valid timestamp")
    if isinstance(value, (int, float)):
        # Already epoch milliseconds.
        return int(value)
    if isinstance(value, str):
        value = pd.Timestamp(value)
    if isinstance(value, datetime) and not isinstance(value, pd.Timestamp):
        value = pd.Timestamp(value)
    if isinstance(value, pd.Timestamp):
        if value.tzinfo is None:
            value = value.tz_localize("UTC")
        return int(value.tz_convert("UTC").timestamp() * 1000)
    raise TypeError(f"cannot convert {type(value).__name__} to milliseconds")


class BinanceClient:
    """Minimal, resilient client for Binance public market data.

    Endpoint paths are class attributes so a Futures subclass can point the same
    request/retry machinery at the ``/fapi`` endpoints.
    """

    PATH_PING = "/api/v3/ping"
    PATH_TIME = "/api/v3/time"
    PATH_EXCHANGE_INFO = "/api/v3/exchangeInfo"
    PATH_TICKER_24H = "/api/v3/ticker/24hr"
    PATH_KLINES = "/api/v3/klines"

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        session: requests.Session | None = None,
        timeout: float = 10.0,
        max_retries: int = 4,
        backoff_base: float = 1.0,
        max_backoff: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._session = session or requests.Session()
        self._owns_session = session is None
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.max_backoff = max_backoff

    # -- context manager ----------------------------------------------------
    def __enter__(self) -> "BinanceClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_session:
            self._session.close()

    # -- low-level request --------------------------------------------------
    def _request(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        params = {k: v for k, v in (params or {}).items() if v is not None}

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:  # network / connection errors
                last_exc = exc
                self._sleep_backoff(attempt, reason=f"network error: {exc}")
                continue

            status = response.status_code

            if status == 200:
                return response.json()

            if status in (429, 418):  # rate limited / IP banned
                retry_after = self._retry_after_seconds(response, attempt)
                logger.warning("Rate limited (HTTP %s); sleeping %.1fs", status, retry_after)
                time.sleep(retry_after)
                last_exc = BinanceAPIError(f"rate limited (HTTP {status})", status_code=status)
                continue

            if 500 <= status < 600:  # transient server error
                last_exc = BinanceAPIError(f"server error (HTTP {status})", status_code=status)
                self._sleep_backoff(attempt, reason=f"HTTP {status}")
                continue

            # Other 4xx: parse the exchange error and fail fast.
            code, msg = self._parse_error(response)
            raise BinanceAPIError(
                f"Binance API error (HTTP {status}, code {code}): {msg}",
                status_code=status,
                code=code,
            )

        raise BinanceAPIError(f"request to {path} failed after {self.max_retries} retries") from last_exc

    def _sleep_backoff(self, attempt: int, *, reason: str) -> None:
        if attempt >= self.max_retries:
            return
        delay = min(self.backoff_base * (2**attempt), self.max_backoff)
        logger.warning("Retrying after %s (attempt %d/%d, sleeping %.1fs)", reason, attempt + 1, self.max_retries, delay)
        time.sleep(delay)

    def _retry_after_seconds(self, response: "requests.Response", attempt: int) -> float:
        header = response.headers.get("Retry-After")
        if header is not None:
            try:
                return float(header)
            except ValueError:
                pass
        return min(self.backoff_base * (2**attempt), self.max_backoff)

    @staticmethod
    def _parse_error(response: "requests.Response") -> tuple[int | None, str]:
        try:
            payload = response.json()
            return payload.get("code"), payload.get("msg", response.text)
        except (ValueError, AttributeError):
            return None, response.text

    # -- connectivity -------------------------------------------------------
    def ping(self) -> bool:
        self._request(self.PATH_PING)
        return True

    def get_server_time(self) -> pd.Timestamp:
        payload = self._request(self.PATH_TIME)
        return pd.Timestamp(payload["serverTime"], unit="ms", tz="UTC")

    # -- exchange info ------------------------------------------------------
    def get_exchange_info(self, symbols: Sequence[str] | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if symbols:
            # Binance expects a JSON-array string, e.g. ["BTCUSDT","ETHUSDT"].
            joined = ",".join(f'"{s.upper()}"' for s in symbols)
            params["symbols"] = f"[{joined}]"
        return self._request(self.PATH_EXCHANGE_INFO, params)

    def get_symbols(
        self,
        *,
        quote_asset: str | None = None,
        status: str | None = "TRADING",
        spot_only: bool = True,
    ) -> list[str]:
        """Return symbol names, optionally filtered by quote asset / status."""
        info = self.get_exchange_info()
        out: list[str] = []
        for sym in info.get("symbols", []):
            if status is not None and sym.get("status") != status:
                continue
            if quote_asset is not None and sym.get("quoteAsset") != quote_asset.upper():
                continue
            if spot_only and not sym.get("isSpotTradingAllowed", True):
                continue
            out.append(sym["symbol"])
        return out

    # -- 24h ticker ---------------------------------------------------------
    def get_ticker_24h(self, symbol: str | None = None) -> pd.DataFrame:
        params = {"symbol": symbol.upper()} if symbol else None
        payload = self._request(self.PATH_TICKER_24H, params)
        rows = [payload] if isinstance(payload, dict) else payload
        df = pd.DataFrame(rows)
        numeric = [
            "priceChange",
            "priceChangePercent",
            "weightedAvgPrice",
            "lastPrice",
            "volume",
            "quoteVolume",
            "highPrice",
            "lowPrice",
        ]
        for col in numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    # -- klines -------------------------------------------------------------
    def get_klines(
        self,
        symbol: str,
        interval: str,
        *,
        start_time: Any = None,
        end_time: Any = None,
        limit: int = _MAX_KLINES_LIMIT,
    ) -> pd.DataFrame:
        """Fetch a single page (<= 1000 rows) of klines as a DataFrame."""
        if interval not in _INTERVAL_MS:
            raise ValueError(f"unsupported interval {interval!r}; known: {sorted(_INTERVAL_MS)}")
        limit = max(1, min(limit, _MAX_KLINES_LIMIT))
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": to_millis(start_time),
            "endTime": to_millis(end_time),
            "limit": limit,
        }
        payload = self._request(self.PATH_KLINES, params)
        return self._klines_to_frame(payload)

    def get_klines_range(
        self,
        symbol: str,
        interval: str,
        *,
        start_time: Any,
        end_time: Any = None,
        page_limit: int = _MAX_KLINES_LIMIT,
        max_pages: int = 10_000,
    ) -> pd.DataFrame:
        """Fetch all klines in ``[start_time, end_time)`` with pagination."""
        if interval not in _INTERVAL_MS:
            raise ValueError(f"unsupported interval {interval!r}; known: {sorted(_INTERVAL_MS)}")

        step_ms = _INTERVAL_MS[interval]
        cursor = to_millis(start_time)
        end_ms = to_millis(end_time) or int(datetime.now(timezone.utc).timestamp() * 1000)

        frames: list[pd.DataFrame] = []
        for _ in range(max_pages):
            if cursor >= end_ms:
                break
            page = self.get_klines(
                symbol,
                interval,
                start_time=cursor,
                end_time=end_ms,
                limit=page_limit,
            )
            if page.empty:
                break
            frames.append(page)
            last_open_ms = int(page.index[-1].value // 1_000_000)
            next_cursor = last_open_ms + step_ms
            if next_cursor <= cursor:  # safety: no forward progress
                break
            cursor = next_cursor
            if len(page) < page_limit:
                break

        if not frames:
            return self._klines_to_frame([])

        combined = pd.concat(frames)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        return combined[combined.index < pd.Timestamp(end_ms, unit="ms", tz="UTC")]

    @staticmethod
    def _klines_to_frame(payload: Sequence[Sequence[Any]]) -> pd.DataFrame:
        df = pd.DataFrame(list(payload), columns=KLINE_COLUMNS)
        if df.empty:
            df = df.astype({c: "float64" for c in _FLOAT_COLUMNS})
            df["trades"] = pd.Series(dtype="int64")
            df.index = pd.DatetimeIndex([], tz="UTC", name="open_time")
            return df.drop(columns=["open_time", "ignore"])

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        for col in _FLOAT_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("int64")
        df = df.drop(columns=["ignore"]).set_index("open_time")
        return df
