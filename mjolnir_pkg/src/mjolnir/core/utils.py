"""Path helpers, frequency conversion, and symbol normalization for Mjolnir."""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path
from typing import List


# ---------------------------------------------------------------------------
# Frequency helpers
# ---------------------------------------------------------------------------

_FREQ_RE = re.compile(r"^(\d+)(s|m|h|d)$", re.IGNORECASE)
_MULTIPLIERS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def bar_freq_to_seconds(freq: str) -> int:
    """Convert bar frequency string to seconds.

    Examples:
        "5s" -> 5
        "1m" -> 60
        "5m" -> 300
        "1h" -> 3600
    """
    m = _FREQ_RE.match(freq.strip())
    if not m:
        raise ValueError(f"Unrecognised bar frequency: {freq!r}. Expected format: '5s', '1m', '1h'.")
    n, unit = int(m.group(1)), m.group(2).lower()
    return n * _MULTIPLIERS[unit]


def bars_per_day(freq: str) -> int:
    """Number of bars in a UTC calendar day for the given frequency."""
    return 86400 // bar_freq_to_seconds(freq)


# ---------------------------------------------------------------------------
# Symbol normalization
# ---------------------------------------------------------------------------

def normalize_symbol(sym: str) -> str:
    """Convert Agamotto-style symbol to native Binance ticker.

    Examples:
        "BINANCE_PERP_BTC_USDT" -> "BTCUSDT"
        "BTCUSDT"               -> "BTCUSDT"
    """
    _EXCHANGE = {"BINANCE", "BINANCEFUTURES", "BYBIT", "BYBITUNIFIED", "STOCKS"}
    _INSTRUMENT = {"PERP", "LINEAR", "UNIFIED", "SPOT", "INVERSE"}
    parts = [p.upper() for p in sym.strip().split("_") if p]
    while parts and parts[0] in _EXCHANGE:
        parts.pop(0)
    while parts and parts[0] in _INSTRUMENT:
        parts.pop(0)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Tardis path construction
# ---------------------------------------------------------------------------

def tardis_path(root: str | Path, date_str: str, stream: str, symbol: str) -> Path:
    """Construct the path to a Tardis gzip CSV shard.

    Args:
        root:      Tardis root directory (e.g. "/mnt/tardis/binance").
        date_str:  Date in YYYYMMDD format.
        stream:    Stream name (e.g. "trades", "book_ticker").
        symbol:    Native Binance ticker (e.g. "BTCUSDT").

    Returns:
        Path: {root}/{YYYYMMDD}/binance-futures_{stream}_{YYYY-MM-DD}_{symbol}.csv.gz
    """
    d = date_str  # e.g. "20250101"
    formatted = f"{d[:4]}-{d[4:6]}-{d[6:]}"  # "2025-01-01"
    filename = f"binance-futures_{stream}_{formatted}_{symbol}.csv.gz"
    return Path(root) / d / filename


# ---------------------------------------------------------------------------
# Date range utilities
# ---------------------------------------------------------------------------

def get_dates_in_range(start_date: str, end_date: str) -> List[str]:
    """Return list of YYYYMMDD strings from start_date to end_date (inclusive).

    Args:
        start_date: "YYYYMMDD" string.
        end_date:   "YYYYMMDD" string (inclusive).
    """
    def _parse(s: str) -> date:
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))

    current = _parse(start_date)
    end = _parse(end_date)
    result: List[str] = []
    while current <= end:
        result.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return result
