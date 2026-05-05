"""Tick-native regime filters for Stormbreaker.

Each filter operates on a bar DataFrame that contains MjolnirFeatures columns.
Returns a boolean Series (True = condition active on that bar).

Directional filters (unidirectional):
  LONG-only:  buy_flow, bid_heavy, ofi_positive, short_liq_spike
  SHORT-only: sell_flow, ask_heavy, ofi_negative, long_liq_spike

Bidirectional filters (context only — pair with a directional):
  BOTH: liq_spike, oi_expanding, high_spread, low_spread
"""

from __future__ import annotations

import pandas as pd

# ── Thresholds ────────────────────────────────────────────────────────────────
# trade_imbalance, depth_imbalance_L5 are normalised to [-1, 1].
_FLOW_THRESH = 0.2           # 20% net buy/sell imbalance
_BOOK_THRESH = 0.2           # 20% net bid/ask book imbalance
_LIQ_BURST_THRESH = 2.0      # liquidation 2× rolling-60-bar average
_OI_VEL_THRESH = 0.005       # 0.5% OI change per bar
_SPREAD_WINDOW = 300         # bars for rolling spread median

# ── Position taxonomy ─────────────────────────────────────────────────────────
_LONG_ONLY = frozenset({"buy_flow", "bid_heavy", "ofi_positive", "short_liq_spike"})
_SHORT_ONLY = frozenset({"sell_flow", "ask_heavy", "ofi_negative", "long_liq_spike"})
_BOTH = frozenset({"liq_spike", "oi_expanding", "high_spread", "low_spread"})
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


def apply_filter(df: pd.DataFrame, filter_name: str) -> pd.Series:
    """Apply a single tick-native filter to a bar DataFrame.

    Args:
        df:          Bar DataFrame with MjolnirFeatures columns.
        filter_name: Filter name WITHOUT timeframe prefix (e.g. 'buy_flow').

    Returns:
        Boolean Series aligned to df.index.  Falls back to all-True if a
        required column is missing so the research step can still run.
    """
    true_mask = pd.Series(True, index=df.index)
    false_mask = pd.Series(False, index=df.index)

    # ── Trade-flow filters ────────────────────────────────────────────────────
    if filter_name == "buy_flow":
        if "trade_imbalance" not in df.columns:
            return true_mask
        return df["trade_imbalance"] > _FLOW_THRESH

    if filter_name == "sell_flow":
        if "trade_imbalance" not in df.columns:
            return true_mask
        return df["trade_imbalance"] < -_FLOW_THRESH

    # ── Order-book filters ────────────────────────────────────────────────────
    if filter_name == "bid_heavy":
        col = next((c for c in ("depth_imbalance_L5", "depth_imbalance_L3",
                                "depth_imbalance_L1") if c in df.columns), None)
        if col is None:
            return true_mask
        return df[col] > _BOOK_THRESH

    if filter_name == "ask_heavy":
        col = next((c for c in ("depth_imbalance_L5", "depth_imbalance_L3",
                                "depth_imbalance_L1") if c in df.columns), None)
        if col is None:
            return true_mask
        return df[col] < -_BOOK_THRESH

    # ── OFI filters ───────────────────────────────────────────────────────────
    if filter_name == "ofi_positive":
        if "ofi_agg" not in df.columns:
            return true_mask
        return df["ofi_agg"] > 0

    if filter_name == "ofi_negative":
        if "ofi_agg" not in df.columns:
            return true_mask
        return df["ofi_agg"] < 0

    # ── Liquidation filters ───────────────────────────────────────────────────
    if filter_name == "short_liq_spike":
        if "liq_burst_ratio" not in df.columns or "liq_directional_imbalance" not in df.columns:
            return true_mask
        return (df["liq_burst_ratio"] > _LIQ_BURST_THRESH) & (df["liq_directional_imbalance"] > 0)

    if filter_name == "long_liq_spike":
        if "liq_burst_ratio" not in df.columns or "liq_directional_imbalance" not in df.columns:
            return true_mask
        return (df["liq_burst_ratio"] > _LIQ_BURST_THRESH) & (df["liq_directional_imbalance"] < 0)

    if filter_name == "liq_spike":
        if "liq_burst_ratio" not in df.columns:
            return true_mask
        return df["liq_burst_ratio"] > _LIQ_BURST_THRESH

    # ── Open-interest filters ─────────────────────────────────────────────────
    if filter_name == "oi_expanding":
        if "oi_velocity" not in df.columns:
            return true_mask
        return df["oi_velocity"] > _OI_VEL_THRESH

    # ── Spread filters ────────────────────────────────────────────────────────
    if filter_name == "high_spread":
        if "relative_spread" not in df.columns:
            return true_mask
        median = df["relative_spread"].rolling(_SPREAD_WINDOW, min_periods=10).median()
        return df["relative_spread"] > median * 1.5

    if filter_name == "low_spread":
        if "relative_spread" not in df.columns:
            return true_mask
        median = df["relative_spread"].rolling(_SPREAD_WINDOW, min_periods=10).median()
        return df["relative_spread"] < median * 0.7

    raise ValueError(f"Unknown Stormbreaker filter: {filter_name!r}")
