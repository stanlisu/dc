"""ValkyrieLoader: reads Deribit options_chain daily gzip CSV files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .utils import deribit_options_chain_path, get_dates_in_range, parse_asset

logger = logging.getLogger(__name__)

# Columns required by DailyFeatureBuilder — skip unused fields to reduce IO.
# Skipped: exchange, local_timestamp, last_price, bid_price, bid_amount,
#          ask_price, ask_amount, underlying_index, rho
_REQUIRED_COLUMNS = [
    "symbol", "timestamp", "type", "strike_price", "expiration",
    "open_interest", "bid_iv", "ask_iv", "mark_price", "mark_iv",
    "underlying_price", "delta", "gamma", "vega", "theta",
]


class ValkyrieLoader:
    """Read and parse Deribit options_chain daily CSV.gz files.

    Args:
        tardis_root: Root of the Tardis Deribit data mount (e.g. "/mnt/tardis/deribit").
        assets: Asset names to include (e.g. ["BTC", "ETH", "SOL_USDC"]).
        usecols: Optional list of column names to read. Defaults to _REQUIRED_COLUMNS
            (the minimal set needed by DailyFeatureBuilder). Pass None to read all columns.
    """

    def __init__(
        self,
        tardis_root: str,
        assets: List[str],
        usecols: Optional[List[str]] = _REQUIRED_COLUMNS,
    ) -> None:
        self.tardis_root = tardis_root
        self.assets = set(assets)
        self.usecols = usecols

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_day(self, date_str: str) -> pd.DataFrame:
        """Load one day's options_chain file, filtered to requested assets.

        Args:
            date_str: Date in YYYYMMDD format.

        Returns:
            Parsed DataFrame with columns: timestamp (tz-aware), asset,
            expiration (microseconds int), expiration_dt (tz-aware),
            strike_price, type (call/put), mark_iv, underlying_price,
            delta, gamma, vega, theta, open_interest, plus raw columns.
            Returns empty DataFrame if file missing or no matching assets.
        """
        path = deribit_options_chain_path(self.tardis_root, date_str)
        if not path.exists():
            logger.debug("Missing file: %s", path)
            return pd.DataFrame()

        try:
            read_kwargs = {"compression": "gzip"}
            if self.usecols is not None:
                read_kwargs["usecols"] = self.usecols
            df = pd.read_csv(path, **read_kwargs)
        except Exception as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            return pd.DataFrame()

        if df.empty:
            return df

        # Extract asset from symbol (e.g. "BTC-7MAR25-95000-C" -> "BTC")
        df["asset"] = df["symbol"].apply(parse_asset)

        # Filter to requested assets
        df = df[df["asset"].isin(self.assets)].copy()
        if df.empty:
            return df

        # Parse timestamps: microseconds UTC
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="us", utc=True)
        df["expiration_dt"] = pd.to_datetime(df["expiration"], unit="us", utc=True)

        # Normalize type to lowercase
        if "type" in df.columns:
            df["type"] = df["type"].str.lower().str.strip()

        # Fill NaN mark_iv from bid/ask midpoint where available
        if "bid_iv" in df.columns and "ask_iv" in df.columns:
            bid = pd.to_numeric(df["bid_iv"], errors="coerce")
            ask = pd.to_numeric(df["ask_iv"], errors="coerce")
            mid_iv = (bid.fillna(ask) + ask.fillna(bid)) / 2
            df["mark_iv"] = pd.to_numeric(df.get("mark_iv", pd.Series()), errors="coerce")
            df["mark_iv"] = df["mark_iv"].fillna(mid_iv)

        # Ensure numeric columns
        for col in ["strike_price", "underlying_price", "delta", "gamma", "vega", "theta",
                    "open_interest", "mark_iv"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df.sort_values("timestamp").reset_index(drop=True)

    def available_dates(self, start_date: str, end_date: str) -> List[str]:
        """Return YYYYMMDD dates with available options_chain files."""
        dates = get_dates_in_range(start_date, end_date)
        return [d for d in dates if deribit_options_chain_path(self.tardis_root, d).exists()]
