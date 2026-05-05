"""Utility functions for Agamotto package."""

from __future__ import annotations

import re
import pandas as pd


def _symbol_to_native(symbol: str) -> str | None:
    symbol = symbol.strip()
    if not symbol:
        return None
    parts = [seg for seg in symbol.split("_") if seg]
    upper = [seg.upper() for seg in parts]
    EXCHANGE_PREFIXES = {"BINANCE", "BINANCEFUTURES", "BYBIT",
                         "BYBITUNIFIED", "STOCKS"}
    INSTRUMENT_PREFIXES = {"PERP", "LINEAR", "UNIFIED", "SPOT", "INVERSE"}
    while upper and (
        upper[0] in EXCHANGE_PREFIXES
        or upper[0].startswith("BYBIT")
    ):
        upper.pop(0)
    while upper and upper[0] in INSTRUMENT_PREFIXES:
        upper.pop(0)
    if not upper:
        return None
    return "".join(upper)


def _timeframe_to_seconds(timeframe: str) -> int:
    match = re.fullmatch(r"(\d+)([mhdw])", timeframe)
    if not match:
        return 24 * 60 * 60
    value, unit = match.groups()
    value = int(value)
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 60 * 60
    if unit == "d":
        return value * 24 * 60 * 60
    if unit == "w":
        return value * 7 * 24 * 60 * 60
    return 24 * 60 * 60


def _round_down_to_timeframe_boundary(ts: pd.Timestamp, timeframe: str) -> pd.Timestamp:
    """
    Round down timestamp to the latest completed candle boundary based on TIME_UNIT.
    
    Examples:
        - 1h: 07:15:30 -> 07:00:00
        - 4h: 07:15:30 -> 04:00:00 (rounds to 0, 4, 8, 12, 16, 20)
        - 15m: 07:17:30 -> 07:15:00
        - 1d: 07:15:30 -> 00:00:00 (midnight)
    """
    match = re.fullmatch(r"(\d+)([mhdw])", timeframe)
    if not match:
        # Default: round to hour
        return ts.replace(minute=0, second=0, microsecond=0)
    
    value, unit = match.groups()
    value = int(value)
    
    if unit == "m":
        # Round down to nearest minute boundary
        minutes = (ts.minute // value) * value
        return ts.replace(minute=minutes, second=0, microsecond=0)
    elif unit == "h":
        if value == 1:
            # Round to hour
            return ts.replace(minute=0, second=0, microsecond=0)
        else:
            # Round to N-hour boundaries (e.g., 4h -> 0, 4, 8, 12, 16, 20)
            hours = (ts.hour // value) * value
            return ts.replace(hour=hours, minute=0, second=0, microsecond=0)
    elif unit == "d":
        if value == 1:
            # Round to midnight
            return ts.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            # For multi-day, round to midnight (could be enhanced for N-day boundaries)
            return ts.replace(hour=0, minute=0, second=0, microsecond=0)
    elif unit == "w":
        # Round to start of week (Monday 00:00:00)
        days_since_monday = ts.weekday()
        return (ts - pd.Timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Default: round to hour
    return ts.replace(minute=0, second=0, microsecond=0)
