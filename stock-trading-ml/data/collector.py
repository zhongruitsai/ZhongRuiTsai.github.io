"""
Data collection layer.

Supports two sources:
  1. yfinance  – US stocks and Taiwan stocks with the ".TW" suffix
  2. TWSE OpenAPI – Taiwan Stock Exchange official data feed
     (real-time / daily OHLCV + institutional holdings)
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)


class DataCollector:
    """Fetch and cache OHLCV data from yfinance and the TWSE OpenAPI."""

    TWSE_BASE = "https://openapi.twse.com.tw/v1"

    def __init__(self, data_dir: str = "cached_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        symbols: List[str],
        start: str,
        end: str,
        use_cache: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Return {symbol: OHLCV DataFrame} for each symbol."""
        result: Dict[str, pd.DataFrame] = {}
        for sym in symbols:
            df = self._load_symbol(sym, start, end, use_cache)
            if df is not None and not df.empty:
                result[sym] = df
            else:
                logger.warning("No data returned for %s", sym)
        return result

    def fetch_twse_daily(self, date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch the full market daily summary from the TWSE OpenAPI.
        ``date`` format: 'YYYYMMDD'. Defaults to today.
        """
        if date is None:
            date = datetime.today().strftime("%Y%m%d")
        url = f"{self.TWSE_BASE}/exchangeReport/STOCK_DAY_ALL"
        params = {"date": date}
        return self._twse_get(url, params)

    def fetch_twse_institutional(self, stock_no: str) -> pd.DataFrame:
        """Fetch three-party institutional buy/sell for a TW stock."""
        url = f"{self.TWSE_BASE}/fund/TWT38U"
        params = {"stockNo": stock_no}
        return self._twse_get(url, params)

    def fetch_twse_margin(self, stock_no: str) -> pd.DataFrame:
        """Fetch margin trading data for a TW stock."""
        url = f"{self.TWSE_BASE}/exchangeReport/TWT93U"
        params = {"stockNo": stock_no}
        return self._twse_get(url, params)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_symbol(
        self, symbol: str, start: str, end: str, use_cache: bool
    ) -> Optional[pd.DataFrame]:
        cache_path = os.path.join(
            self.data_dir, f"{symbol.replace('.', '_')}_{start}_{end}.parquet"
        )
        if use_cache and os.path.exists(cache_path):
            logger.info("Loading %s from cache", symbol)
            return pd.read_parquet(cache_path)

        df = self._fetch_yfinance(symbol, start, end)
        if df is not None and not df.empty:
            df.to_parquet(cache_path)
        return df

    def _fetch_yfinance(
        self, symbol: str, start: str, end: str, retries: int = 3
    ) -> Optional[pd.DataFrame]:
        for attempt in range(retries):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start, end=end, auto_adjust=True)
                if df.empty:
                    return None
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                df.columns = ["open", "high", "low", "close", "volume"]
                df.dropna(inplace=True)
                logger.info("Fetched %d rows for %s", len(df), symbol)
                return df
            except Exception as exc:
                logger.warning("yfinance attempt %d failed for %s: %s", attempt + 1, symbol, exc)
                time.sleep(2 ** attempt)
        return None

    def _twse_get(self, url: str, params: dict) -> pd.DataFrame:
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return pd.DataFrame(data)
            return pd.DataFrame([data])
        except Exception as exc:
            logger.error("TWSE API error: %s", exc)
            return pd.DataFrame()
