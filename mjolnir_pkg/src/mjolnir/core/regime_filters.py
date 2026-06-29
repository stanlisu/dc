"""Regime filter masks for MjolnirResearch.

Standalone functions — no class state needed. These only use the DataFrame,
filter name, and position to produce boolean Series masks.

Extracted from research.py to keep each module under ~700 lines for
PyArmor trial compatibility.
"""

from __future__ import annotations

import pandas as pd


def apply_filter_mask(
    df: pd.DataFrame,
    filter_name,
    position: str,
) -> pd.Series:
    """Return a boolean Series: True = include in filter.

    Supports:
    - String filter names with _and_ / _or_ composition
    - List filters: ["filter_a", "&", "filter_b"]
    - Mjolnir-specific microstructure filters
    - Standard Agamotto price filters (baseline, rsi, macd, etc.)
    """
    # --- List support (recursive) ---
    if isinstance(filter_name, list):
        mask = None
        op = None
        for item in filter_name:
            item = str(item).strip()
            if item in ("|", "&"):
                op = item
            else:
                sub = apply_filter_mask(df, item, position)
                if mask is None:
                    mask = sub
                elif op == "|":
                    mask = mask | sub
                else:
                    mask = mask & sub
        return mask if mask is not None else pd.Series(True, index=df.index)

    # --- String composition ---
    if isinstance(filter_name, str):
        name = filter_name.lower().strip()
        if "_and_" in name:
            parts = name.split("_and_")
            mask = None
            for p in parts:
                sub = apply_filter_mask(df, p.strip(), position)
                mask = sub if mask is None else (mask & sub)
            return mask if mask is not None else pd.Series(True, index=df.index)
        if "_or_" in name:
            parts = name.split("_or_")
            mask = None
            for p in parts:
                sub = apply_filter_mask(df, p.strip(), position)
                mask = sub if mask is None else (mask | sub)
            return mask if mask is not None else pd.Series(True, index=df.index)
    else:
        name = str(filter_name).lower().strip()

    if df.empty:
        return pd.Series(dtype=bool)

    # Strip trailing _long / _short from the filter name
    name = name.replace("_long", "").replace("_short", "")

    return named_filter(df, name, position)


def named_filter(df: pd.DataFrame, name: str, position: str) -> pd.Series:
    """Dispatch to a named filter implementation."""
    true = pd.Series(True, index=df.index)

    # ---- Microstructure filters (Mjolnir-specific) ----

    if name == "baseline":
        raise ValueError(
            "baseline regime removed 2026-06-18 — it is an unconditional "
            "fires-every-bar no-brainer; drop it from the regime stack. See CLAUDE.md.")

    if name == "high_liquidation_pressure":
        col = "liq_burst_ratio"
        if col not in df.columns:
            return true
        return df[col] > df[col].quantile(0.75)

    if name == "low_liquidation_pressure":
        col = "liq_burst_ratio"
        if col not in df.columns:
            return true
        return df[col] < df[col].quantile(0.25)

    if name == "funding_positive":
        col = "funding_rate"
        if col not in df.columns:
            return true
        return df[col] > 0

    if name == "funding_negative":
        col = "funding_rate"
        if col not in df.columns:
            return true
        return df[col] < 0

    if name == "deep_book":
        col = "depth_imbalance_L5"
        if col not in df.columns:
            return true
        if position == "long":
            return df[col] > df[col].quantile(0.6)
        return df[col] < df[col].quantile(0.4)

    if name == "trade_imbalance":
        col = "trade_imbalance"
        if col not in df.columns:
            return true
        if position == "long":
            return df[col] > 0
        return df[col] < 0

    if name == "basis_premium":
        col = "basis_pct"
        if col not in df.columns:
            return true
        return df[col] > 0

    if name == "basis_discount":
        col = "basis_pct"
        if col not in df.columns:
            return true
        return df[col] < 0

    if name == "pre_funding_settlement":
        col = "pre_funding"
        if col not in df.columns:
            return true
        return df[col] > 0

    if name == "oi_expansion":
        col = "oi_velocity"
        if col not in df.columns:
            return true
        return df[col] > 0

    if name == "oi_contraction":
        col = "oi_velocity"
        if col not in df.columns:
            return true
        return df[col] < 0

    if name == "ofi_positive":
        col = "ofi_agg"
        if col not in df.columns:
            return true
        if position == "long":
            return df[col] > 0
        return df[col] < 0

    if name == "tight_spread":
        col = "relative_spread"
        if col not in df.columns:
            return true
        return df[col] < df[col].quantile(0.5)

    if name == "wide_spread":
        col = "relative_spread"
        if col not in df.columns:
            return true
        return df[col] > df[col].quantile(0.5)

    # ---- Standard price filters (shared with Agamotto) ----

    mvg1 = df.get("mvg1")
    mvg2 = df.get("mvg2")
    close = df.get("close", df.get("mid_price"))

    if mvg1 is None or mvg2 is None or close is None:
        # Return true if base price columns missing
        return true

    up_trend = (close > mvg1) & (mvg1 > mvg2)
    down_trend = (close < mvg1) & (mvg1 < mvg2)

    if name in ("trend_aligned",):
        return up_trend if position == "long" else down_trend

    if name == "strong_trend":
        mvg3 = df.get("mvg3")
        if position == "long":
            return up_trend & (mvg2 > mvg3) if mvg3 is not None else up_trend
        return down_trend & (mvg2 < mvg3) if mvg3 is not None else down_trend

    if name == "ma_momentum":
        mvg3 = df.get("mvg3")
        if position == "long":
            return (mvg1 > mvg2) & (mvg1 > mvg3) if mvg3 is not None else (mvg1 > mvg2)
        return (mvg1 < mvg2) & (mvg1 < mvg3) if mvg3 is not None else (mvg1 < mvg2)

    if name == "rsi_oversold":
        col = "rsi"
        if col not in df.columns:
            return true
        return df[col] < 30

    if name == "rsi_overbought":
        col = "rsi"
        if col not in df.columns:
            return true
        return df[col] > 70

    if name == "macd_bullish":
        col = "macdhist"
        if col not in df.columns:
            return true
        return df[col] > 0 if position == "long" else df[col] < 0

    if name == "macd_bearish":
        col = "macdhist"
        if col not in df.columns:
            return true
        return df[col] < 0 if position == "long" else df[col] > 0

    if name == "adx_trend":
        col = "adx"
        if col not in df.columns:
            return true
        return df[col] > 25

    if name == "bb_rebound":
        if position == "long":
            col = "bb_lower"
            if col not in df.columns:
                return true
            return close < df[col]
        else:
            col = "bb_upper"
            if col not in df.columns:
                return true
            return close > df[col]

    if name in ("high_volume", "vol_breakout"):
        col = "vol_ratio"
        if col not in df.columns:
            return true
        threshold = 2.0 if name == "vol_breakout" else 1.0
        return df[col] > threshold

    if name == "low_volume":
        col = "vol_ratio"
        if col not in df.columns:
            return true
        return df[col] < 1.0

    if name == "high_vol":
        col = "price_range_pct"
        if col not in df.columns:
            return true
        return df[col] > df[col].quantile(0.5)

    if name == "low_vol":
        col = "price_range_pct"
        if col not in df.columns:
            return true
        return df[col] < df[col].quantile(0.5)

    if name == "mom_positive":
        col = "mom"
        if col not in df.columns:
            return true
        return df[col] > 0 if position == "long" else df[col] < 0

    raise ValueError(f"Unknown filter: {name!r}")
