"""Ladder-adjusted return computation for MjolnirResearch.

Standalone function — takes a config dict as first arg instead of reading
self.config.

Extracted from research.py to keep each module under ~700 lines for
PyArmor trial compatibility.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_ladder_returns(
    config: Dict,
    df: pd.DataFrame,
    close_col: str,
    low_col: str,
    high_col: str,
    horizon_bars: int = 1,
) -> pd.DataFrame:
    """Compute ladder-adjusted return columns for a single symbol.

    Mirrors the agamotto.research.AgamottoResearch.engineer_features ladder
    logic (same step_size = 1 bps, same clipping, same column names),
    generalized to a variable forward horizon so the fill window matches
    the prediction window.

    At horizon_bars == 1 the behaviour is identical to the original
    1-bar shift, preserving native-mode parity (TIME_UNIT == bar
    resolution, e.g. mjolnir.base.5s_1). For boundary-aligned
    experiments (e.g. mjolnir.base.30s_1: 5s bars predicting the next
    30s boundary close) callers should pass
    horizon_bars = TIME_UNIT_seconds / bar_tf_seconds so the low/high
    lookahead spans the full prediction horizon, instead of only the
    next single bar (which under-counted ladder fills and produced
    a frictionless close-to-close target).

    Args:
        config:       Dictionary loaded from setting.json.
        df:           DataFrame with at least close_col, low_col, high_col.
        close_col:    Name of the close price column.
        low_col:      Name of the low price column.
        high_col:     Name of the high price column.
        horizon_bars: Forward horizon used both for the price return
                      (close[t+h] / close[t] - 1) and for the
                      low/high min/max lookahead window. Must be >= 1.

    Returns:
        DataFrame with columns: return_long, return_short,
        return_long_raw, return_short_raw.
    """
    if horizon_bars < 1:
        raise ValueError(
            f"horizon_bars must be >= 1, got {horizon_bars}")

    step_size = 0.0001
    ladder = int(config.get("LADDER", 1) or 0)
    # FEE is required (no fallback): see commit 0500d8fa for rationale.
    # The historical `or 0.0` collapsed any falsy FEE to the default.
    fee_rate = float(config["FEE"]) / 10000.0

    close = df[close_col]
    low_series = df[low_col]
    high_series = df[high_col]

    # Forward h-bar price return: close[t+h] / close[t] - 1.
    price_return = close.pct_change(
        horizon_bars, fill_method=None).shift(-horizon_bars)
    close_safe = close.replace(0, np.nan)

    # Forward-rolling min low / max high over (t, t+horizon_bars]:
    # shift(-1) so position t holds low[t+1], then reverse-rolling so
    # the window covers the next horizon_bars bars (exclusive of t).
    # At horizon_bars=1 this reduces to low.shift(-1) / high.shift(-1).
    low_shifted = low_series.shift(-1)
    high_shifted = high_series.shift(-1)
    low_window = (
        low_shifted.iloc[::-1]
        .rolling(horizon_bars, min_periods=1)
        .min()
        .iloc[::-1]
    )
    high_window = (
        high_shifted.iloc[::-1]
        .rolling(horizon_bars, min_periods=1)
        .max()
        .iloc[::-1]
    )

    # Per-1bps fill layers from each side's forward excursion, capped at
    # LADDER. low_layers = how far price DUG DOWN below cost (long entries /
    # short exits); high_layers = how far it RALLIED UP above cost (short
    # entries / long exits).
    # 2026-06-20 refinement #1: use n layers, NOT 1 + n — no free base rung,
    # so an excursion of < 1bps fills nothing.
    distance_long = ((close_safe - low_window) / close_safe).replace(
        [np.inf, -np.inf], np.nan)
    low_layers = np.floor(distance_long / step_size).clip(
        lower=0, upper=ladder).fillna(0).astype(int)

    distance_short = ((high_window - close_safe) / close_safe).replace(
        [np.inf, -np.inf], np.nan)
    high_layers = np.floor(distance_short / step_size).clip(
        lower=0, upper=ladder).fillna(0).astype(int)

    # 2026-06-20 refinement #2 (Stan — mark-to-market exit): size each position
    # by its ENTRY-side penetration and mark the WHOLE opened stack at the
    # horizon close, i.e. force-liquidate at close[t+h] whatever did not take
    # profit. Unclosed inventory therefore BOOKS the real directional move at
    # the close (underwater longs in a downtrend / underwater shorts in a bull
    # realize their loss) instead of being silently dropped — the earlier
    # size=min(low,high) gate discarded the carried, mostly-losing, inventory
    # and was optimistically biased. LONG accumulates on the DIP -> size_long =
    # low_layers; SHORT accumulates on the RALLY -> size_short = high_layers.
    # No look-ahead favorable exit is credited: selecting round-trips by
    # min(open,close) and paying them a take-profit would re-introduce a smaller
    # excursion-conditioned artifact, so the exit is purely liquidate-at-close.
    size_long = low_layers
    size_short = high_layers
    # Gross per-unit signed return BEFORE fee and size. Default ("ladder")
    # marks the whole opened stack at the horizon close (close-to-close):
    # long earns +price_return, short earns -price_return. fill_mode may
    # override either the size (flat) or the exit price (limit_then_taker).
    ret_long_gross = price_return
    ret_short_gross = -price_return

    fill_mode = str(config.get("LADDER_FILL_MODE", "ladder")).lower()
    if fill_mode == "flat":
        # Two-way TAKER model — fixed size 1 per bar, filled at the decision
        # (horizon-close) price, no laddered size and no maker rungs. This is
        # what aggressive taker execution actually realizes (precise fill
        # PRICE, fixed SIZE), as opposed to the laddered maker accumulation.
        size_long = 1
        size_short = 1
    elif fill_mode == "limit_then_taker":
        # Two-stage maker-close-then-taker-fallback exit (Stan 2026-06-22).
        # Open n on the entry-side penetration (size_long/size_short as
        # above), then try to close the WHOLE position with a single limit at
        # the next-boundary close close[t+h]. Decide the fill from the
        # FOLLOWING horizon window (t+h, t+2h]:
        #   LONG  (sell limit): fills if that window rallies back up to
        #          close_h (max high >= close_h) -> exit at close_h; else
        #          taker-close the leftover at close[t+2h].
        #   SHORT (buy limit):  fills if that window dips to close_h
        #          (min low <= close_h) -> exit at close_h; else taker-close
        #          the leftover at close[t+2h].
        # Unfilled ("leftover") inventory therefore books the real later move
        # at close[t+2h], not the optimistic single horizon close. Both
        # branches are charged the taker FEE (a deliberately strict
        # assumption). Requires the full t+2 window: the final 2h bars per
        # symbol have no close[t+2h] and are masked to NaN (dropped in
        # verticalize, which under this mode also gates on return_long/short).
        close_h = close.shift(-horizon_bars)
        close_2h = close.shift(-2 * horizon_bars)
        high_w2 = high_window.shift(-horizon_bars)   # max high over (t+h, t+2h]
        low_w2 = low_window.shift(-horizon_bars)      # min low over (t+h, t+2h]
        full = close_2h.notna()                       # full t+2 window observed
        exit_long = close_h.where(high_w2 >= close_h, close_2h)
        exit_short = close_h.where(low_w2 <= close_h, close_2h)
        ret_long_gross = (exit_long / close_safe - 1.0).where(full)
        ret_short_gross = (-(exit_short / close_safe - 1.0)).where(full)
    elif fill_mode != "ladder":
        raise ValueError(
            "LADDER_FILL_MODE must be 'ladder', 'flat' or 'limit_then_taker', "
            f"got {fill_mode!r}")

    fee_cost = (fee_rate * 2.0) if fee_rate else 0.0
    # return_X = (signed gross return - round-trip fee) x size. Short's gross
    # is already negated above; fee is paid either side. Matches features.py
    # `"return_short": -(forward_return + fee)` (i.e. -price_return - fee).
    return_long = ((ret_long_gross - fee_cost) * size_long).rename("return_long")
    return_short = ((ret_short_gross - fee_cost) * size_short).rename("return_short")
    return_long_raw = (ret_long_gross * size_long).rename("return_long_raw")
    return_short_raw = (ret_short_gross * size_short).rename("return_short_raw")

    return pd.concat(
        [return_long, return_short, return_long_raw, return_short_raw], axis=1)
