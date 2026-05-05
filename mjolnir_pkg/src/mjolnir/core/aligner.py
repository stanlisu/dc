"""StreamAligner: merge heterogeneous Tardis streams into uniform time bars."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .utils import bar_freq_to_seconds, bars_per_day

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column mappings for each stream
# ---------------------------------------------------------------------------

_BOOK_SNAP_LEVELS = 25  # number of levels in book_snapshot_25


class StreamAligner:
    """Merge five Tardis streams into a single time-bar DataFrame.

    Each day produces ``bars_per_day(bar_freq)`` rows (17 280 at 5s).

    Alignment strategy per stream:
    - ``trades``           → OHLCV aggregate + buy_vol, sell_vol, n_trades,
                             vwap, trade_imbalance
    - ``book_ticker``      → last-value in bar (spread, microprice, bid/ask)
    - ``book_snapshot_25`` → last-value in bar (depth by level)
    - ``derivative_ticker``→ forward-fill to bar grid
    - ``liquidations``     → sum within bar (count + notional per side)

    Args:
        bar_freq: Bar width as a string, e.g. "5s", "1m", "1h".
    """

    def __init__(self, bar_freq: str = "5s") -> None:
        self.bar_freq = bar_freq
        self._bar_seconds = bar_freq_to_seconds(bar_freq)
        # Pandas deprecated 'm' for minutes (now means month-end); use 'min' instead
        self._pd_freq = (
            bar_freq[:-1] + "min"
            if bar_freq.endswith("m") and not bar_freq.endswith("ms")
            else bar_freq
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def align(
        self,
        streams: Dict[str, pd.DataFrame],
        date_str: str,
    ) -> pd.DataFrame:
        """Align all streams to a uniform time-bar grid for one UTC day.

        Args:
            streams:  Dict of stream_name -> raw DataFrame from MjolnirLoader.
            date_str: "YYYYMMDD" — used to build the full-day bar index.

        Returns:
            DataFrame with a UTC DatetimeIndex (one row per bar, 86400/bar_seconds
            rows), indexed by bar *open* timestamp.  Missing bars are left as NaN.
        """
        bar_index = self._make_bar_index(date_str)
        parts: Dict[str, pd.DataFrame] = {}

        if "trades" in streams and not streams["trades"].empty:
            parts["trades"] = self._agg_trades(streams["trades"], bar_index)

        if "book_ticker" in streams and not streams["book_ticker"].empty:
            parts["book_ticker"] = self._last_value(
                streams["book_ticker"], bar_index, self._book_ticker_cols
            )

        if "book_snapshot_25" in streams and not streams["book_snapshot_25"].empty:
            parts["book_snapshot_25"] = self._last_value_snapshot(
                streams["book_snapshot_25"], bar_index
            )

        if "derivative_ticker" in streams and not streams["derivative_ticker"].empty:
            parts["derivative_ticker"] = self._ffill_derivative(
                streams["derivative_ticker"], bar_index
            )

        if "liquidations" in streams and not streams["liquidations"].empty:
            parts["liquidations"] = self._agg_liquidations(
                streams["liquidations"], bar_index
            )

        if not parts:
            return pd.DataFrame(index=bar_index)

        combined = pd.concat(list(parts.values()), axis=1)
        combined = combined.reindex(bar_index)
        return combined

    # ------------------------------------------------------------------
    # Trades aggregation
    # ------------------------------------------------------------------

    def _agg_trades(self, df: pd.DataFrame, bar_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Aggregate tick trades into OHLCV bars."""
        if "timestamp" not in df.columns or "price" not in df.columns:
            return pd.DataFrame(index=bar_index)

        df = df.copy()
        df["bar"] = df["timestamp"].dt.floor(self._pd_freq)
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["amount"] = pd.to_numeric(df.get("amount", pd.Series(np.nan)), errors="coerce")

        is_buy = df["side"].str.lower().isin(["buy", "b"]) if "side" in df.columns else pd.Series(False, index=df.index)

        agg = df.groupby("bar").agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("amount", "sum"),
            n_trades=("price", "count"),
        )
        agg["buy_vol"] = df[is_buy].groupby("bar")["amount"].sum()
        agg["sell_vol"] = df[~is_buy].groupby("bar")["amount"].sum()

        # VWAP
        df["pv"] = df["price"] * df["amount"]
        agg["vwap"] = df.groupby("bar")["pv"].sum() / (agg["volume"] + 1e-10)

        # Trade imbalance ∈ [-1, 1]
        agg["trade_imbalance"] = (
            (agg["buy_vol"].fillna(0) - agg["sell_vol"].fillna(0))
            / (agg["volume"].fillna(0) + 1e-10)
        )

        agg = agg.reindex(bar_index)

        # Forward-fill close for zero-trade bars, then derive OHLC and zero volumes.
        # Bars with no trades have NaN OHLCV, which propagates into all TA features
        # and causes Ridge/ElasticNet failures. Standard microstructure convention:
        # a zero-trade bar carries the last traded price with zero volume.
        agg["close"] = agg["close"].ffill()
        no_trade = agg["n_trades"].isna()
        agg.loc[no_trade, "open"] = agg.loc[no_trade, "close"]
        agg.loc[no_trade, "high"] = agg.loc[no_trade, "close"]
        agg.loc[no_trade, "low"] = agg.loc[no_trade, "close"]
        agg.loc[no_trade, "vwap"] = agg.loc[no_trade, "close"]
        for col in ["volume", "n_trades", "buy_vol", "sell_vol"]:
            agg[col] = agg[col].fillna(0.0)
        agg["trade_imbalance"] = agg["trade_imbalance"].fillna(0.0)

        return agg

    # ------------------------------------------------------------------
    # Book ticker (last value in bar)
    # ------------------------------------------------------------------

    @staticmethod
    def _book_ticker_cols(df: pd.DataFrame) -> pd.DataFrame:
        """Select and rename book-ticker columns."""
        keep = [c for c in ["bid_price", "bid_amount", "ask_price", "ask_amount"] if c in df.columns]
        return df[keep] if keep else df

    def _last_value(
        self,
        df: pd.DataFrame,
        bar_index: pd.DatetimeIndex,
        col_selector=None,
    ) -> pd.DataFrame:
        """Take the last tick value within each bar."""
        if "timestamp" not in df.columns:
            return pd.DataFrame(index=bar_index)

        df = df.copy()
        df["bar"] = df["timestamp"].dt.floor(self._pd_freq)

        if col_selector is not None:
            # Select value columns but keep "bar"
            selected = col_selector(df)
            selected["bar"] = df["bar"]
            df = selected

        last = df.groupby("bar").last()
        # Numeric coercion
        for col in last.columns:
            if col != "bar":
                last[col] = pd.to_numeric(last[col], errors="coerce")

        last = last.reindex(bar_index).ffill()
        return last

    # ------------------------------------------------------------------
    # Book snapshot (depth levels, last value in bar)
    # ------------------------------------------------------------------

    def _last_value_snapshot(
        self, df: pd.DataFrame, bar_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Extract depth columns from book_snapshot_25 and take last-in-bar."""
        if "timestamp" not in df.columns:
            return pd.DataFrame(index=bar_index)

        df = df.copy()
        df["bar"] = df["timestamp"].dt.floor(self._pd_freq)

        # Collect depth columns for top 5 levels
        depth_cols: Dict[str, pd.Series] = {}
        for side in ("bids", "asks"):
            for lvl in range(5):
                p_col = f"{side}[{lvl}].price"
                q_col = f"{side}[{lvl}].amount"
                if p_col in df.columns:
                    depth_cols[f"{side}_{lvl}_price"] = pd.to_numeric(df[p_col], errors="coerce")
                if q_col in df.columns:
                    depth_cols[f"{side}_{lvl}_qty"] = pd.to_numeric(df[q_col], errors="coerce")

        if not depth_cols:
            return pd.DataFrame(index=bar_index)

        snap = pd.DataFrame(depth_cols, index=df.index)
        snap["bar"] = df["bar"]
        last = snap.groupby("bar").last().reindex(bar_index).ffill()

        # Aggregate convenience depth columns
        for n in (1, 3, 5):
            bid_q = [f"bids_{i}_qty" for i in range(n) if f"bids_{i}_qty" in last.columns]
            ask_q = [f"asks_{i}_qty" for i in range(n) if f"asks_{i}_qty" in last.columns]
            if bid_q:
                last[f"depth_bid_L{n}"] = last[bid_q].sum(axis=1)
            if ask_q:
                last[f"depth_ask_L{n}"] = last[ask_q].sum(axis=1)

        return last

    # ------------------------------------------------------------------
    # Derivative ticker (forward-fill)
    # ------------------------------------------------------------------

    def _ffill_derivative(
        self, df: pd.DataFrame, bar_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Forward-fill derivative ticker to bar frequency."""
        if "timestamp" not in df.columns:
            return pd.DataFrame(index=bar_index)

        want = [
            "mark_price", "index_price", "open_interest",
            "funding_rate", "next_funding_time", "predicted_funding_rate",
        ]
        df = df.copy()
        df["bar"] = df["timestamp"].dt.floor(self._pd_freq)
        keep = [c for c in want if c in df.columns]
        if not keep:
            return pd.DataFrame(index=bar_index)

        df_sub = df[keep + ["bar"]].copy()
        for c in keep:
            df_sub[c] = pd.to_numeric(df_sub[c], errors="coerce")

        last = df_sub.groupby("bar").last()
        last = last.reindex(bar_index).ffill()
        return last

    # ------------------------------------------------------------------
    # Liquidations (sum within bar)
    # ------------------------------------------------------------------

    def _agg_liquidations(
        self, df: pd.DataFrame, bar_index: pd.DatetimeIndex
    ) -> pd.DataFrame:
        """Sum liquidation events within each bar."""
        if "timestamp" not in df.columns:
            return pd.DataFrame(index=bar_index)

        df = df.copy()
        df["bar"] = df["timestamp"].dt.floor(self._pd_freq)
        df["price"] = pd.to_numeric(df.get("price", pd.Series(np.nan)), errors="coerce")
        df["amount"] = pd.to_numeric(df.get("amount", pd.Series(np.nan)), errors="coerce")
        df["notional"] = df["price"] * df["amount"]

        is_long_liq = df["side"].str.lower().isin(["buy", "b"]) if "side" in df.columns else pd.Series(False, index=df.index)

        agg = df.groupby("bar").agg(liq_total_count=("amount", "count"))
        agg["liq_long_count"] = df[is_long_liq].groupby("bar")["amount"].count()
        agg["liq_short_count"] = df[~is_long_liq].groupby("bar")["amount"].count()
        agg["liq_long_notional"] = df[is_long_liq].groupby("bar")["notional"].sum()
        agg["liq_short_notional"] = df[~is_long_liq].groupby("bar")["notional"].sum()

        agg = agg.reindex(bar_index).fillna(0.0)
        return agg

    # ------------------------------------------------------------------
    # Bar index construction
    # ------------------------------------------------------------------

    def _make_bar_index(self, date_str: str) -> pd.DatetimeIndex:
        """Build a complete UTC bar index for one calendar day."""
        start = pd.Timestamp(
            f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}",
            tz="UTC",
        )
        return pd.date_range(start=start, periods=bars_per_day(self.bar_freq), freq=self._pd_freq)
