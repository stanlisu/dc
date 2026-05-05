"""Utility functions for Valkyrie: date helpers, expiry parsing, IV rank."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def get_dates_in_range(start_date: str, end_date: str) -> List[str]:
    """Return YYYYMMDD strings from start_date to end_date inclusive."""
    def _parse(s: str) -> date:
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))

    current = _parse(start_date)
    end = _parse(end_date)
    result: List[str] = []
    while current <= end:
        result.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    return result


def deribit_options_chain_path(tardis_root: str, date_str: str) -> Path:
    """Path to Deribit options_chain gzip file for a given date.

    Pattern: {root}/{YYYYMMDD}/deribit_options_chain_{YYYY-MM-DD}_OPTIONS.csv.gz
    """
    d = date_str
    formatted = f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return Path(tardis_root) / d / f"deribit_options_chain_{formatted}_OPTIONS.csv.gz"


def parse_asset(symbol: str) -> str:
    """Extract asset base name from a Deribit option symbol.

    Examples:
        "BTC-7MAR25-95000-C"    -> "BTC"
        "SOL_USDC-7MAR25-230-C" -> "SOL_USDC"
        "ETH-5APR19-170-P"      -> "ETH"
    """
    return symbol.split("-")[0]


def get_atm_strike(strikes: np.ndarray, spot_price: float) -> Optional[float]:
    """Return the strike price closest to spot_price."""
    if len(strikes) == 0 or spot_price <= 0:
        return None
    return float(min(strikes, key=lambda s: abs(float(s) - spot_price)))


def get_delta_strike(
    df: pd.DataFrame,
    target_delta: float,
    tolerance: float = 0.12,
) -> Optional[float]:
    """Find the strike whose mean delta is closest to target_delta.

    Args:
        df: Options DataFrame filtered to one expiry and option_type.
        target_delta: Target delta (e.g. 0.25 for 25Δ call, -0.25 for 25Δ put).
        tolerance: Maximum allowed deviation from target_delta.

    Returns:
        Strike price, or None if not found within tolerance.
    """
    if df.empty or "delta" not in df.columns:
        return None

    per_strike = df.groupby("strike_price")["delta"].mean()
    if per_strike.empty:
        return None

    diffs = (per_strike - target_delta).abs()
    best_strike = diffs.idxmin()
    if diffs[best_strike] > tolerance:
        return None
    return float(best_strike)


def iv_rank(iv_series: pd.Series, window: int = 252) -> pd.Series:
    """Rolling IV rank (0–100) over a trailing window of trading days.

    IV rank = (current_iv - rolling_min) / (rolling_max - rolling_min) * 100
    Returns 50.0 where insufficient history or zero range.
    """
    roll_min = iv_series.rolling(window, min_periods=10).min()
    roll_max = iv_series.rolling(window, min_periods=10).max()
    denom = (roll_max - roll_min).replace(0, np.nan)
    rank = (iv_series - roll_min) / denom * 100
    return rank.fillna(50.0)


def iv_percentile(iv_series: pd.Series, window: int = 252) -> pd.Series:
    """Rolling IV percentile — fraction of past days below current IV (×100)."""
    def _pctile(x: np.ndarray) -> float:
        if len(x) < 2:
            return 50.0
        return float(np.mean(x[:-1] < x[-1]) * 100)

    return iv_series.rolling(window, min_periods=10).apply(_pctile, raw=True)


def rsi_manual(prices: pd.Series, period: int = 7) -> pd.Series:
    """Simple RSI (Wilder smoothing via rolling mean for simplicity)."""
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50.0)


def get_expiry_buckets(
    expiration_us: pd.Series,
    midday_us: int,
    max_weekly_days: int = 7,
) -> Tuple[Optional[int], Optional[int]]:
    """Find the nearest weekly and ~30-day monthly expiry timestamps.

    Args:
        expiration_us: Series of expiry timestamps in microseconds UTC.
        midday_us: Reference timestamp (microseconds UTC) for the current day.
        max_weekly_days: Max days-to-expiry for the "weekly" bucket.

    Returns:
        (weekly_expiry_us, monthly_expiry_us) — either may be None.
    """
    ONE_DAY_US = 86_400 * 1_000_000
    max_weekly_us = max_weekly_days * ONE_DAY_US
    thirty_days_us = 30 * ONE_DAY_US

    unique_expiries = sorted(expiration_us.dropna().unique())
    future_expiries = [e for e in unique_expiries if e > midday_us]

    if not future_expiries:
        return None, None

    # Weekly: shortest future expiry within max_weekly_days
    weekly_candidates = [e for e in future_expiries if (e - midday_us) <= max_weekly_us]
    weekly = weekly_candidates[0] if weekly_candidates else future_expiries[0]

    # Monthly: closest to 30 days out (must be different from weekly)
    monthly_target = midday_us + thirty_days_us
    if len(future_expiries) > 1:
        monthly = min(future_expiries, key=lambda e: abs(e - monthly_target))
    else:
        monthly = future_expiries[0]

    return int(weekly), int(monthly)
