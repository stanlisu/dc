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

VALID_FILL_MODES = ("ladder", "flat", "limit_then_taker")


def resolve_fill_mode(config: Dict) -> str:
    """Read the REQUIRED ``LADDER_FILL_MODE`` key. No default.

    Single source of truth for the key. It used to be read as
    ``config.get("LADDER_FILL_MODE", "ladder")`` at THREE independent
    sites (``ladder.py``, ``research.py::verticalize``,
    ``streaming.py::stream_research``); an arm whose ``setting.json``
    omitted the key therefore got a silently DIFFERENT TARGET from one
    that set it — ``mjolnir.base.5s_1`` (omitted -> "ladder") vs
    ``mjolnir.base.30s_1`` ("limit_then_taker") — with nothing in the
    logs to say so. Magic-value defaults on required numeric/string
    config are banned by CLAUDE.md's no-silent-fallback rule, so a
    missing key now raises.

    Args:
        config: Dictionary loaded from setting.json.

    Returns:
        The lower-cased fill mode, guaranteed to be in VALID_FILL_MODES.

    Raises:
        KeyError:   LADDER_FILL_MODE absent from config.
        ValueError: LADDER_FILL_MODE present but not a known mode.
    """
    if "LADDER_FILL_MODE" not in config:
        raise KeyError(
            "LADDER_FILL_MODE is required in setting.json and has NO default "
            f"(VERSION={config.get('VERSION', '<unset>')!r}). It selects the "
            "target construction, so defaulting it silently changes what the "
            f"model is trained on. Set it explicitly to one of "
            f"{VALID_FILL_MODES}.")
    mode = str(config["LADDER_FILL_MODE"]).lower()
    if mode not in VALID_FILL_MODES:
        raise ValueError(
            "LADDER_FILL_MODE must be 'ladder', 'flat' or 'limit_then_taker', "
            f"got {mode!r}")
    return mode


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
    # Per-unit UNSIGNED price return on each side's own exit path, BEFORE fee and
    # size. Default ("ladder") marks the whole opened stack at the horizon close
    # (close-to-close), so both sides see the same close-to-close move; fill_mode
    # may override either the size (flat) or the exit price (limit_then_taker),
    # which is why the two are tracked separately.
    #
    # UNSIGNED, i.e. the MARKET move, NOT the trade's P&L: a short is profitable
    # when this is NEGATIVE. This matches agamotto (agamotto/research.py:379-380,
    # `return_{long,short}_raw = price_return * size`) so the shared marvel PnL
    # engine — which books `signal * y_true_raw`, with signal = -1 for a short —
    # is correct for BOTH algos. Until 2026-07-29 mjolnir negated the short here,
    # making its target position-SIGNED while agamotto's stayed unsigned under the
    # SAME column names; the engine then signed mjolnir's a second time and every
    # short leg booked +price_return. See tasks/lessons.md 2026-07-29.
    ret_long_px = price_return
    ret_short_px = price_return

    fill_mode = resolve_fill_mode(config)
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
        # Both UNSIGNED (see ret_{long,short}_px above): each side's own realized
        # price move on its own exit path. The short is NOT negated here — the
        # consumer applies the direction.
        ret_long_px = (exit_long / close_safe - 1.0).where(full)
        ret_short_px = (exit_short / close_safe - 1.0).where(full)
    # No trailing `elif fill_mode != "ladder": raise` — resolve_fill_mode()
    # already rejected anything outside VALID_FILL_MODES, so reaching here
    # means fill_mode == "ladder" (the close-to-close mark, set above).

    fee_cost = (fee_rate * 2.0) if fee_rate else 0.0
    # Targets are UNSIGNED market returns (the trade direction is applied by the
    # consumer). The round-trip fee is therefore SUBTRACTED for the long and ADDED
    # for the short: a long needs the move above +fee to profit, a short needs it
    # below -fee. Identical to agamotto (agamotto/research.py:370-380
    # `long: price_return - fee_cost`, `short: price_return + fee_cost`) and to
    # features.py's non-ladder path.
    return_long = ((ret_long_px - fee_cost) * size_long).rename("return_long")
    return_short = ((ret_short_px + fee_cost) * size_short).rename("return_short")
    return_long_raw = (ret_long_px * size_long).rename("return_long_raw")
    return_short_raw = (ret_short_px * size_short).rename("return_short_raw")

    return pd.concat(
        [return_long, return_short, return_long_raw, return_short_raw], axis=1)
