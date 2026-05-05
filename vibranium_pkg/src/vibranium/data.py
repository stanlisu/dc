"""
vibranium/data.py — Data loader for 15m liquid crypto perpetual OHLCV.

Loads close prices from data/BINANCEFUTURES/15m/liquid/, aligns on a common
DatetimeIndex (15m bars), and drops symbols with excessive missing data.
"""
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

_TIMESTAMP_COL = "open_time_ms"
_CLOSE_COL = "close"
# Drop symbols missing more than this fraction of bars
_MAX_MISSING_FRAC = 0.05
# Forward-fill at most this many consecutive NaN bars
_MAX_FFILL = 4


class VibraniumData:
    """Load and align 15m close prices for liquid crypto perpetuals."""

    def __init__(self, data_dir: str = "data/BINANCEFUTURES/15m/liquid/"):
        self.data_dir = data_dir

    def list_symbols(self, tf: str = "15m") -> list:
        """Return all available symbols (subdirectory names), sorted."""
        target_dir = self._resolve_dir(tf)
        if not os.path.isdir(target_dir):
            raise FileNotFoundError(f"Data directory not found: {target_dir}")
        return sorted(
            d for d in os.listdir(target_dir)
            if os.path.isdir(os.path.join(target_dir, d))
        )

    def _resolve_dir(self, tf: str) -> str:
        """Return the data directory, substituting tf if data_dir contains a placeholder."""
        # If data_dir already contains the tf, use as-is; otherwise keep self.data_dir
        return self.data_dir

    def _load_symbol(self, symbol: str, tf: str = "15m") -> pd.Series:
        """Load all monthly CSVs for a symbol; return close-price Series on DatetimeIndex."""
        sym_dir = os.path.join(self.data_dir, symbol)
        if not os.path.isdir(sym_dir):
            raise FileNotFoundError(f"Symbol directory not found: {sym_dir}")

        csv_files = sorted(f for f in os.listdir(sym_dir) if f.endswith(".csv"))
        if not csv_files:
            logger.warning("No CSV files found for %s", symbol)
            return pd.Series(dtype=float, name=symbol)

        frames = []
        for fname in csv_files:
            path = os.path.join(sym_dir, fname)
            try:
                df = pd.read_csv(path, usecols=[_TIMESTAMP_COL, _CLOSE_COL])
                frames.append(df)
            except Exception as exc:
                logger.warning("Failed to read %s: %s", path, exc)

        if not frames:
            return pd.Series(dtype=float, name=symbol)

        combined = pd.concat(frames, ignore_index=True)
        combined["ts"] = pd.to_datetime(combined[_TIMESTAMP_COL], unit="ms", utc=True)
        combined = combined.drop_duplicates(subset="ts").sort_values("ts")
        series = combined.set_index("ts")[_CLOSE_COL].rename(symbol)
        series.index = series.index.tz_localize(None)
        return series

    def _drop_sparse_symbols(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that exceed the maximum missing-data fraction."""
        n_rows = len(prices)
        if n_rows == 0:
            return prices
        missing_frac = prices.isna().sum() / n_rows
        keep = missing_frac[missing_frac <= _MAX_MISSING_FRAC].index.tolist()
        dropped = [c for c in prices.columns if c not in keep]
        if dropped:
            logger.info(
                "Dropping symbols with >%.0f%% missing: %s",
                _MAX_MISSING_FRAC * 100,
                dropped,
            )
        return prices[keep]

    def load_prices(
        self,
        symbols: list = None,
        start_date: str = None,
        end_date: str = None,
        tf: str = "15m",
    ) -> pd.DataFrame:
        """
        Load close prices for the requested symbols, aligned on a common DatetimeIndex.

        Parameters
        ----------
        symbols : list[str] or None
            Symbols to load.  If None, loads all available symbols.
        start_date : str or None
            ISO date string, e.g. "2023-01-01".
        end_date : str or None
            ISO date string, e.g. "2025-12-31".
        tf : str
            Timeframe tag (informational; data_dir should already point to correct tf).

        Returns
        -------
        pd.DataFrame
            DatetimeIndex (15m bars); columns are symbol names.
        """
        if symbols is None:
            symbols = self.list_symbols(tf=tf)

        series_list = []
        for sym in symbols:
            try:
                s = self._load_symbol(sym, tf=tf)
                if not s.empty:
                    series_list.append(s)
            except Exception as exc:
                logger.warning("Skipping %s: %s", sym, exc)

        if not series_list:
            return pd.DataFrame()

        prices = pd.concat(series_list, axis=1)
        prices.index = pd.to_datetime(prices.index)
        prices.sort_index(inplace=True)

        if start_date is not None:
            prices = prices[prices.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            prices = prices[prices.index <= pd.Timestamp(end_date)]

        prices = prices.ffill(limit=_MAX_FFILL)
        prices = self._drop_sparse_symbols(prices)
        prices = prices.dropna(how="all")

        logger.info(
            "Loaded %d symbols x %d bars  [%s -> %s]",
            len(prices.columns),
            len(prices),
            prices.index.min() if not prices.empty else "n/a",
            prices.index.max() if not prices.empty else "n/a",
        )
        return prices
