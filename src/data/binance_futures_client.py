"""Read-only client for Binance USDⓈ-M **Futures** public market data.

Reuses the spot client's request/retry/pagination machinery but points at the
``/fapi`` endpoints. Like the spot client it is strictly read-only: klines,
exchange info, 24h ticker and (futures-only) funding rate. It never signs
requests and cannot open, modify or close a position.

The key futures-specific capability is **universe discovery**: listing the
tradable USDT-margined perpetual symbols so the scanner can sweep the market
instead of a hand-typed symbol list.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..logging_utils import get_logger
from .binance_client import BinanceClient

logger = get_logger(__name__)

FUTURES_BASE_URL = "https://fapi.binance.com"


class BinanceFuturesClient(BinanceClient):
    """Binance USDⓈ-M Futures market-data client (read-only)."""

    PATH_PING = "/fapi/v1/ping"
    PATH_TIME = "/fapi/v1/time"
    PATH_EXCHANGE_INFO = "/fapi/v1/exchangeInfo"
    PATH_TICKER_24H = "/fapi/v1/ticker/24hr"
    PATH_KLINES = "/fapi/v1/klines"
    PATH_FUNDING_RATE = "/fapi/v1/fundingRate"

    def __init__(self, base_url: str = FUTURES_BASE_URL, **kwargs: Any) -> None:
        super().__init__(base_url, **kwargs)

    def get_perpetual_symbols(
        self,
        *,
        quote_asset: str = "USDT",
        status: str = "TRADING",
        exclude_quarterly: bool = True,
    ) -> list[str]:
        """List tradable perpetual symbols (universe discovery).

        Parameters
        ----------
        quote_asset:
            Margin/quote asset, typically ``USDT``.
        exclude_quarterly:
            Keep only ``contractType == "PERPETUAL"`` (drop dated futures).
        """
        info = self.get_exchange_info()
        out: list[str] = []
        for sym in info.get("symbols", []):
            if sym.get("status") != status:
                continue
            if sym.get("quoteAsset") != quote_asset.upper():
                continue
            if exclude_quarterly and sym.get("contractType") != "PERPETUAL":
                continue
            out.append(sym["symbol"])
        logger.info("Discovered %d %s perpetual symbols", len(out), quote_asset)
        return out

    def get_funding_rate(self, symbol: str, *, limit: int = 100) -> pd.DataFrame:
        """Recent funding-rate history for a symbol (context only)."""
        payload = self._request(
            self.PATH_FUNDING_RATE, {"symbol": symbol.upper(), "limit": limit}
        )
        df = pd.DataFrame(payload)
        if not df.empty:
            df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
            df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        return df
