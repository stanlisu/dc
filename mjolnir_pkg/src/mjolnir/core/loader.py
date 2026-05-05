"""MjolnirLoader: reads daily gzip shards from the Tardis dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .utils import normalize_symbol, tardis_path

logger = logging.getLogger(__name__)

# Known stream schemas: required columns after timestamp normalization
_STREAM_SCHEMAS: Dict[str, List[str]] = {
    "trades": ["side", "price", "amount"],
    "book_ticker": ["ask_price", "ask_amount", "bid_price", "bid_amount"],
    "book_snapshot_25": [],  # dynamic column names (asks[0].price, etc.)
    "derivative_ticker": ["mark_price", "index_price", "open_interest", "funding_rate"],
    "liquidations": ["side", "price", "amount"],
}


class MjolnirLoader:
    """Read daily gzip CSV shards from the Tardis Binance dataset.

    Args:
        tardis_root: Root directory of the Tardis data mount
                     (e.g. "/mnt/tardis/binance").
        symbol:      Agamotto-style or native symbol
                     (e.g. "BINANCE_PERP_BTC_USDT" or "BTCUSDT").
    """

    def __init__(self, tardis_root: str, symbol: str) -> None:
        self.tardis_root = Path(tardis_root)
        self.symbol_native = normalize_symbol(symbol)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_stream(self, date_str: str, stream: str) -> pd.DataFrame:
        """Load one stream for one day.

        Args:
            date_str: Date in YYYYMMDD format.
            stream:   One of trades, book_ticker, book_snapshot_25,
                      derivative_ticker, liquidations.

        Returns:
            DataFrame with microsecond-resolution UTC DatetimeIndex.
            Returns an empty DataFrame if the file is missing.
        """
        path = tardis_path(self.tardis_root, date_str, stream, self.symbol_native)
        if not path.exists():
            logger.debug("Missing shard: %s", path)
            return pd.DataFrame()

        try:
            df = self._read_gz(path)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            return pd.DataFrame()

        df = self._parse_timestamp(df)
        return df

    def load_day(self, date_str: str, streams: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Load all (or selected) streams for a single day.

        Args:
            date_str: Date in YYYYMMDD format.
            streams:  List of stream names. Defaults to all five streams.

        Returns:
            Dict mapping stream name -> DataFrame.
        """
        if streams is None:
            streams = list(_STREAM_SCHEMAS.keys())
        return {s: self.load_stream(date_str, s) for s in streams}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_gz(path: Path) -> pd.DataFrame:
        """Read a gzip-compressed Tardis CSV file using pandas.

        We use polars-style lazy evaluation when possible but fall back to
        pandas for full compatibility with the existing pipeline.
        """
        try:
            import polars as pl
            lf = pl.scan_csv(str(path), infer_schema_length=1000)
            df = lf.collect().to_pandas()
        except Exception:
            # Polars unavailable or failed — use pandas
            df = pd.read_csv(path, compression="gzip")
        return df

    @staticmethod
    def _parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
        """Convert Tardis microsecond epoch to UTC DatetimeIndex."""
        if df.empty:
            return df
        if "timestamp" not in df.columns:
            logger.warning("No 'timestamp' column found; returning raw DataFrame.")
            return df
        # Tardis timestamps are microseconds since Unix epoch
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="us", utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
