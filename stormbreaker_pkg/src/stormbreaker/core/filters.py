"""Tick-native regime filters for Stormbreaker.

Each filter operates on a bar DataFrame that contains MjolnirFeatures columns.
Returns a boolean Series (True = condition active on that bar).

Directional filters (unidirectional):
  LONG-only:  buy_flow, bid_heavy, ofi_positive, short_liq_spike
  SHORT-only: sell_flow, ask_heavy, ofi_negative, long_liq_spike

Bidirectional filters (context only — pair with a directional):
  BOTH: liq_spike, high_spread, low_spread
"""

from __future__ import annotations

import pandas as pd

# ── Thresholds ────────────────────────────────────────────────────────────────
# trade_imbalance, depth_imbalance_L5 are normalised to [-1, 1].
_FLOW_THRESH = 0.2           # 20% net buy/sell imbalance
_BOOK_THRESH = 0.2           # 20% net bid/ask book imbalance
_LIQ_BURST_THRESH = 2.0      # liquidation 2× rolling-60-bar average
_SPREAD_WINDOW = 300         # bars for rolling spread median

# Book-depth preference chain for bid_heavy/ask_heavy: any one carries the
# signal; none of them means the frame cannot express the regime (raises).
_DEPTH_COLS = ("depth_imbalance_L5", "depth_imbalance_L3", "depth_imbalance_L1")

# ── Position taxonomy ─────────────────────────────────────────────────────────
_LONG_ONLY = frozenset({"buy_flow", "bid_heavy", "ofi_positive", "short_liq_spike"})
_SHORT_ONLY = frozenset({"sell_flow", "ask_heavy", "ofi_negative", "long_liq_spike"})
_BOTH = frozenset({"liq_spike", "high_spread", "low_spread"})
_ALL_TICK_FILTERS = _LONG_ONLY | _SHORT_ONLY | _BOTH


def allowed_positions(regime_name: str) -> list[str]:
    """Return allowed positions for a (possibly compound) Stormbreaker regime.

    Scans every component of the regime name for known tick-filter suffixes.
    Rules:
      - Any LONG-only filter present → long side allowed
      - Any SHORT-only filter present → short side allowed
      - Both LONG-only and SHORT-only present → contradictory, return []
      - Neither present → BOTH (all bidirectional components)
    """
    has_long = any(f"_{f}" in regime_name or regime_name.endswith(f) for f in _LONG_ONLY)
    has_short = any(f"_{f}" in regime_name or regime_name.endswith(f) for f in _SHORT_ONLY)

    if has_long and has_short:
        return []            # contradictory combo — caller should skip
    if has_long:
        return ["long"]
    if has_short:
        return ["short"]
    return ["long", "short"]


def is_tick_filter(name: str) -> bool:
    """Return True if *name* (without TF prefix) is a known tick-native filter."""
    return name in _ALL_TICK_FILTERS


def _missing_column_error(
    df: pd.DataFrame, filter_name: str, detail: str
) -> ValueError:
    """Build the shared fail-loud error for an absent source column.

    2026-08-04: these guards used to return an all-True mask, i.e. the
    regime fires on EVERY bar. That is a silent `baseline` regime, which
    CLAUDE.md removed forever (2026-06-18) and enforces in code. It bites
    hardest here because `StormBreakerResearch._get_tf_view` returns a frame
    with NO columns but the SAME index when a cross-TF atom names a TF that
    has no `{tf}_` columns — so a mistyped or unbuilt context TF turned the
    whole regime unconditional across every row (marvel
    `stormbreaker/gauntlet/README.md`, "Any TF an atom can name must have
    real columns"). Mirrors the mjolnir fix in
    `mjolnir/core/regime_filters.py::_require_col`.
    """
    return ValueError(
        f"Stormbreaker filter {filter_name!r} requires {detail}; frame has "
        f"{len(df.columns)} columns. An all-True fallback here fires on "
        "every bar — a `baseline` regime under another name, banned by "
        "CLAUDE.md. If this is a cross-TF atom, the named TF has no "
        "columns: add it to CONTEXT_BAR_FREQS and build its bars, or drop "
        "the regime.")


def _require(df: pd.DataFrame, filter_name: str, *cols: str) -> None:
    """Raise unless EVERY column in *cols* is present. See _missing_column_error."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise _missing_column_error(
            df, filter_name, f"column(s) {missing}")


def _require_any(df: pd.DataFrame, filter_name: str, cols: tuple[str, ...]) -> str:
    """Return the first present column of *cols*; raise if none is.

    For atoms that accept a preference chain (e.g. depth_imbalance L5→L3→L1):
    any one of them carries the signal, but none of them means the frame
    cannot express the regime at all.
    """
    for col in cols:
        if col in df.columns:
            return col
    raise _missing_column_error(
        df, filter_name, f"one of {list(cols)}, and has none of them")


def apply_filter(df: pd.DataFrame, filter_name: str) -> pd.Series:
    """Apply a single tick-native filter to a bar DataFrame.

    Args:
        df:          Bar DataFrame with MjolnirFeatures columns.
        filter_name: Filter name WITHOUT timeframe prefix (e.g. 'buy_flow').

    Returns:
        Boolean Series aligned to df.index.

    Raises:
        ValueError: if *filter_name* is unknown, or if a required column is
            absent (see _require — never an all-True fallback).
    """

    # ── Trade-flow filters ────────────────────────────────────────────────────
    if filter_name == "buy_flow":
        _require(df, filter_name, "trade_imbalance")
        return df["trade_imbalance"] > _FLOW_THRESH

    if filter_name == "sell_flow":
        _require(df, filter_name, "trade_imbalance")
        return df["trade_imbalance"] < -_FLOW_THRESH

    # ── Order-book filters ────────────────────────────────────────────────────
    if filter_name == "bid_heavy":
        col = _require_any(df, filter_name, _DEPTH_COLS)
        return df[col] > _BOOK_THRESH

    if filter_name == "ask_heavy":
        col = _require_any(df, filter_name, _DEPTH_COLS)
        return df[col] < -_BOOK_THRESH

    # ── OFI filters ───────────────────────────────────────────────────────────
    if filter_name == "ofi_positive":
        _require(df, filter_name, "ofi_agg")
        return df["ofi_agg"] > 0

    if filter_name == "ofi_negative":
        _require(df, filter_name, "ofi_agg")
        return df["ofi_agg"] < 0

    # ── Liquidation filters ───────────────────────────────────────────────────
    if filter_name == "short_liq_spike":
        _require(df, filter_name, "liq_burst_ratio", "liq_directional_imbalance")
        return (df["liq_burst_ratio"] > _LIQ_BURST_THRESH) & (df["liq_directional_imbalance"] > 0)

    if filter_name == "long_liq_spike":
        _require(df, filter_name, "liq_burst_ratio", "liq_directional_imbalance")
        return (df["liq_burst_ratio"] > _LIQ_BURST_THRESH) & (df["liq_directional_imbalance"] < 0)

    if filter_name == "liq_spike":
        _require(df, filter_name, "liq_burst_ratio")
        return df["liq_burst_ratio"] > _LIQ_BURST_THRESH

    # oi_expanding REMOVED 2026-07-24 (with mjolnir oi_velocity): the gate
    # read oi_velocity, whose live-vs-offline step timing is unreplicable.

    # ── Spread filters ────────────────────────────────────────────────────────
    if filter_name == "high_spread":
        _require(df, filter_name, "relative_spread")
        median = df["relative_spread"].rolling(_SPREAD_WINDOW, min_periods=10).median()
        return df["relative_spread"] > median * 1.5

    if filter_name == "low_spread":
        _require(df, filter_name, "relative_spread")
        median = df["relative_spread"].rolling(_SPREAD_WINDOW, min_periods=10).median()
        return df["relative_spread"] < median * 0.7

    raise ValueError(f"Unknown Stormbreaker filter: {filter_name!r}")
