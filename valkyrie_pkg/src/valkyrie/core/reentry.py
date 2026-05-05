"""Daily re-entry simulation for Valkyrie weekly options.

When the predicted direction changes on a new day (Mon–Thu), close the
current option position at that day's open price and re-enter with the
new signal.  The final open leg uses the pre-computed ret_short_* from the
entry row (already handles hold-to-Friday + stop-loss on all hold days).
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

# y value → option position
_Y_TO_POSITION: Dict[int, str] = {
    0: "short_call",
    1: "short_straddle",
    2: "short_put",
}

# position → (open_price_col, high_price_col, hold_to_fri_ret_col)
_POS_COLS: Dict[str, tuple] = {
    "short_call":     ("atm_call_price_open",     "atm_call_price_high",     "ret_short_call"),
    "short_put":      ("atm_put_price_open",       "atm_put_price_high",      "ret_short_put"),
    "short_straddle": ("atm_straddle_price_open",  "atm_straddle_price_high", "ret_short_straddle"),
}


def _leg_pnl_early_close(
    entry_price: float,
    intermediate_rows: list,
    high_col: str,
    stoploss_pct: float,
    close_price: float,
) -> float:
    """P&L for a leg closed early (signal flip), checking stop-loss on intermediate days.

    Args:
        entry_price:        Option premium collected at leg entry.
        intermediate_rows:  Row dicts for days strictly between entry and flip day.
        high_col:           Column name for the daily high option price.
        stoploss_pct:       Stop-loss fraction (e.g. 0.10).
        close_price:        Option price at the open of the flip day.

    Returns:
        Float P&L: capped at -stoploss_pct if stopped, else
        (entry_price - close_price) / entry_price.
    """
    if not (np.isfinite(entry_price) and entry_price > 0):
        return 0.0
    stop_threshold = entry_price * (1.0 + stoploss_pct)
    for row in intermediate_rows:
        high = row.get(high_col)
        if high is not None and np.isfinite(high) and high >= stop_threshold:
            return -stoploss_pct  # stopped out before the flip
    # Not stopped — close at flip-day open
    if not (np.isfinite(close_price) and close_price > 0):
        return 0.0
    return float((entry_price - close_price) / entry_price)


def simulate_weekly_reentry(week_df: pd.DataFrame, stoploss_pct: float = 0.10) -> float:
    """Simulate daily re-entry for one (iso_week, asset) group.

    Args:
        week_df:     DataFrame with Mon–Thu rows (day_of_week 0–3) for one week+asset.
                     Required columns: day_of_week, y_pred, atm_*_price_open/high,
                     ret_short_call/put/straddle.
        stoploss_pct: Early-close stop threshold.

    Returns:
        Total weekly P&L (sum of all legs).
    """
    rows = week_df.sort_values("day_of_week").to_dict("records")
    if not rows:
        return np.nan

    current_pos: str | None = None
    entry_price: float = np.nan
    entry_dow: int = -1
    total_ret: float = 0.0

    for row in rows:
        dow = int(row["day_of_week"])
        y_pred = int(row.get("y_pred", 1))
        new_pos = _Y_TO_POSITION.get(y_pred, "short_straddle")

        if current_pos is None:
            # First entry
            current_pos = new_pos
            entry_price = float(row.get(_POS_COLS[new_pos][0]) or np.nan)
            entry_dow = dow
            continue

        if new_pos != current_pos:
            # Signal flip: close current leg at today's open
            cur_open_col, cur_high_col, _ = _POS_COLS[current_pos]
            intermediates = [r for r in rows if entry_dow < r["day_of_week"] < dow]
            close_price = float(row.get(cur_open_col) or np.nan)
            leg_ret = _leg_pnl_early_close(
                entry_price=entry_price,
                intermediate_rows=intermediates,
                high_col=cur_high_col,
                stoploss_pct=stoploss_pct,
                close_price=close_price,
            )
            total_ret += leg_ret
            # Re-enter with new position
            current_pos = new_pos
            entry_price = float(row.get(_POS_COLS[new_pos][0]) or np.nan)
            entry_dow = dow

    # Final leg: use pre-computed hold-to-Friday P&L from the entry row
    if current_pos is not None:
        entry_row = next((r for r in rows if r["day_of_week"] == entry_dow), None)
        if entry_row is not None:
            _, _, final_ret_col = _POS_COLS[current_pos]
            total_ret += float(entry_row.get(final_ret_col) or 0.0)

    return total_ret


def compute_reentry_pnl(df: pd.DataFrame, stoploss_pct: float = 0.10) -> pd.DataFrame:
    """Apply simulate_weekly_reentry across all (iso_year, iso_week, asset) groups.

    Args:
        df:  DataFrame with columns: iso_year, iso_week, asset, day_of_week,
             y_pred, atm_*_price_open/high, ret_short_*.
        stoploss_pct: Stop-loss fraction.

    Returns:
        DataFrame with columns: iso_year, iso_week, asset, weekly_ret.
    """
    records = []
    for (iso_year, iso_week, asset), grp in df.groupby(["iso_year", "iso_week", "asset"]):
        ret = simulate_weekly_reentry(grp, stoploss_pct=stoploss_pct)
        records.append({"iso_year": iso_year, "iso_week": iso_week,
                         "asset": asset, "weekly_ret": ret})
    return pd.DataFrame(records)
