"""agamotto research: regime filter-mask evaluation.

Split out of research.py to keep every script under the PyArmor trial
per-script ceiling (see research_features.py). The method body below is
VERBATIM from research.py.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from .research_common import _obf

logger = logging.getLogger(__name__)


class FilterMaskMixin:
    """Provides :meth:`_apply_filter_mask` to AgamottoResearch."""

    def _apply_filter_mask(self, df: pd.DataFrame, filter_name: str | list, position: str) -> pd.Series:
        """
        Applies a named filter or list of filters to the DataFrame.
        Returns a boolean Series (mask) where True = Keep signal.

        Dispatch order matters (2026-08-02): own-column atoms (high_vol,
        low_vol, rsi_*, macd_*, stoch_bullish, cci_reversal, adx_trend,
        strong_candle, *_volume, vol_breakout, mom_positive, buy_pressure,
        mfi_*, bop_*, roc_*) resolve BEFORE any close/mvg1/mvg2 check, because
        none of them read those columns. The MVG_DEPENDENT_FILTERS (plus the
        `combined_*` composites) RAISE when close/mvg1/mvg2 are missing instead
        of returning an all-True mask — an all-True mask there fires on every
        bar, the banned baseline-shaped always-on pattern. Mirrors the mjolnir
        fix (b5ea04a, mjolnir/core/regime_filters.py::named_filter).

        Unknown names are unaffected by the price-column check and still hit
        the strict-filters raise at the bottom.
        """
        # --- List Support (Recursive) ---
        if isinstance(filter_name, list):
            mask = None
            current_op = None
            for item in filter_name:
                item = str(item).strip()
                if item in ["|", "&"]:
                    current_op = item
                else:
                    sub_mask = self._apply_filter_mask(df, item, position)
                    if mask is None:
                        mask = sub_mask
                    else:
                        if current_op == "|":
                            mask = mask | sub_mask
                        elif current_op == "&":
                            mask = mask & sub_mask
                        else:
                            # Default to AND
                            mask = mask & sub_mask
            return mask if mask is not None else pd.Series(True, index=df.index)

        # --- Base Filters ---
        if isinstance(filter_name, str):
            filter_name = filter_name.lower().strip()

            # Skip metadata rows (e.g. __summary__ from filter_regime_stacks.py)
            if filter_name.startswith("__"):
                return pd.Series(True, index=df.index)

            # Accept coded regimes (rename rollout): decode code→real before the
            # real-name `filter_name == "..."` chain. Real names pass through;
            # genuinely-unknown tokens still hit the strict raise below.
            filter_name = _obf().decode_regime_tolerant(filter_name)

            # Support complex strings like "filterA_and_filterB" or "filterA_or_filterB"
            if "_and_" in filter_name:
                parts = filter_name.split("_and_")
                mask = None
                for part in parts:
                    sub_mask = self._apply_filter_mask(df, part.strip(), position)
                    mask = sub_mask if mask is None else (mask & sub_mask)
                return mask if mask is not None else pd.Series(True, index=df.index)
            
            if "_or_" in filter_name:
                parts = filter_name.split("_or_")
                mask = None
                for part in parts:
                    sub_mask = self._apply_filter_mask(df, part.strip(), position)
                    mask = sub_mask if mask is None else (mask | sub_mask)
                return mask if mask is not None else pd.Series(True, index=df.index)

        if df.empty:
            return pd.Series(dtype=bool)

        # Strip suffixes from filter name
        if isinstance(filter_name, str):
            filter_name = filter_name.replace("_long", "").replace("_short", "")
            # Strip TF prefix from ORB regime names (e.g. "15m_stoch_bullish" → "stoch_bullish")
            filter_name = re.sub(r'^(?:15m|1h|4h|1d)_', '', filter_name)

        # Enforce LONG_ONLY / SHORT_ONLY constraints via allowed_positions()
        _allowed = type(self).allowed_positions(filter_name) if isinstance(filter_name, str) else None
        if _allowed and position not in _allowed:
            return pd.Series(False, index=df.index)

        # ---- Own-column filters FIRST ----
        # None of the atoms below read close/mvg1/mvg2, so they must never be
        # gated on the price columns. Pre-2026-08-02 they sat BELOW the shared
        # required_base all-True guard, so on a frame lacking those columns they
        # silently fired on EVERY bar even when their own column was present
        # (high_vol AND low_vol simultaneously all-True) — the banned
        # baseline-shaped always-on failure. Mirrors the mjolnir fix (b5ea04a,
        # mjolnir/core/regime_filters.py). Their own missing-column all-True
        # guards are unchanged (longstanding, pinned by tests) — except the
        # three volume atoms, which RAISE via _volume_ratio() rather than
        # collapsing to a scalar-1.0 comparison (2026-08-02, see that helper).
        if position == "long":
            if filter_name == "low_vol":
                q50 = df["price_range_pct_q50"] if "price_range_pct_q50" in df.columns else df["price_range_pct"].rolling(700, min_periods=1).quantile(0.5)
                return df["price_range_pct"] < q50
            if filter_name == "high_vol":
                q50 = df["price_range_pct_q50"] if "price_range_pct_q50" in df.columns else df["price_range_pct"].rolling(700, min_periods=1).quantile(0.5)
                return df["price_range_pct"] > q50
            if filter_name == "strong_candle":
                return (df["open_close_pct"] > 0.005)

            # TA-Lib based
            if filter_name == "rsi_oversold":
                return df["rsi"] < 30 if "rsi" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "macd_bullish":
                return df["macdhist"] > 0 if "macdhist" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "stoch_bullish":
                return (df["stoch_k"] > df["stoch_d"]) if "stoch_k" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "cci_reversal":
                return df["cci"] > 100 if "cci" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "adx_trend":
                return (df["adx"] > 25) if "adx" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "mom_positive":
                return df["mom"] > 0 if "mom" in df.columns else pd.Series(True, index=df.index)

            # Volume-based (priority to quote_vol_ratio if available, else vol_ratio)
            if filter_name in ("low_volume", "high_volume", "vol_breakout"):
                v_ratio = type(self)._volume_ratio(df, filter_name)
                if filter_name == "low_volume": return (v_ratio < 1.0)
                if filter_name == "high_volume": return (v_ratio > 1.0)
                return (v_ratio > 2.0)  # vol_breakout

            # New TA-lab filters
            if filter_name == "buy_pressure":
                return df["buy_pressure"] > 0.55 if "buy_pressure" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "mfi_oversold":
                return df["mfi"] < 30 if "mfi" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "bop_bullish":
                return df["bop"] > 0.1 if "bop" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "roc_positive":
                return df["roc"] > 0 if "roc" in df.columns else pd.Series(True, index=df.index)
        else:  # short
            if filter_name == "low_vol":
                q50 = df["price_range_pct_q50"] if "price_range_pct_q50" in df.columns else df["price_range_pct"].rolling(700, min_periods=1).quantile(0.5)
                return df["price_range_pct"] < q50
            if filter_name == "high_vol":
                q50 = df["price_range_pct_q50"] if "price_range_pct_q50" in df.columns else df["price_range_pct"].rolling(700, min_periods=1).quantile(0.5)
                return df["price_range_pct"] > q50
            if filter_name == "strong_candle":
                return (df["open_close_pct"] < -0.005)

            # TA-Lib based
            if filter_name == "rsi_overbought":
                return df["rsi"] > 70 if "rsi" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "macd_bearish":
                return df["macdhist"] < 0 if "macdhist" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "stoch_bullish":
                return (df["stoch_k"] > df["stoch_d"]) if "stoch_k" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "cci_reversal":
                return df["cci"] < -100 if "cci" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "adx_trend":
                return (df["adx"] > 25) if "adx" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "mom_positive":
                return df["mom"] < 0 if "mom" in df.columns else pd.Series(True, index=df.index)

            # Volume-based (priority to quote_vol_ratio if available, else vol_ratio)
            # quote_vol_ratio = quote_vol / 7d_ma
            # vol_ratio = vol / 7d_ma
            if filter_name in ("low_volume", "high_volume", "vol_breakout"):
                v_ratio = type(self)._volume_ratio(df, filter_name)
                if filter_name == "low_volume": return (v_ratio < 1.0)
                if filter_name == "high_volume": return (v_ratio > 1.0)
                return (v_ratio > 2.0)  # vol_breakout

            # New TA-lab filters
            if filter_name == "buy_pressure":
                return df["buy_pressure"] < 0.45 if "buy_pressure" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "mfi_overbought":
                return df["mfi"] > 70 if "mfi" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "bop_bearish":
                return df["bop"] < -0.1 if "bop" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "roc_negative":
                return df["roc"] < 0 if "roc" in df.columns else pd.Series(True, index=df.index)

        # ---- Price-column dependent filters ----
        # These genuinely read close/mvg1/mvg2. Missing columns RAISE
        # (no-silent-fallback, CLAUDE.md): the old shared guard logged a warning
        # and returned all-True, so on a degenerate/warmup frame every one of
        # them fired on every bar. Unknown names are NOT gated here — they still
        # fall through to the strict-filters raise below, unchanged.
        if isinstance(filter_name, str) and (
                filter_name in type(self).MVG_DEPENDENT_FILTERS
                or filter_name.startswith("combined_")):
            required_base = ["close", "mvg1", "mvg2"]
            missing = [col for col in required_base if col not in df.columns]
            if missing:
                raise ValueError(
                    f"Filter {filter_name!r} requires price columns; frame is "
                    f"missing {missing}. An all-True fallback here fires on "
                    "every bar (baseline-shaped) — failing loud instead. "
                    "See CLAUDE.md no-silent-fallback rules.")

            # Common trend conditions
            up_trend = (df["close"] > df["mvg1"]) & (df["mvg1"] > df["mvg2"])
            down_trend = (df["close"] < df["mvg1"]) & (df["mvg1"] < df["mvg2"])

            if position == "long":
                if filter_name == "trend_aligned":  # baseline removed 2026-06-18 (no-brainer; falls through to strict raise)
                    return up_trend
                if filter_name == "strong_trend":
                    return up_trend & (df["mvg2"] > df["mvg3"]) if "mvg3" in df.columns else up_trend
                if filter_name == "ma_momentum":
                    return (df["mvg1"] > df["mvg2"]) & (df["mvg1"] > df["mvg3"]) if "mvg3" in df.columns else (df["mvg1"] > df["mvg2"])
                if filter_name == "above_all_mas":
                    mask = (df["close"] > df["mvg1"]) & (df["close"] > df["mvg2"])
                    if "mvg3" in df.columns:
                        mask &= (df["close"] > df["mvg3"])
                    return mask
                if filter_name == "near_ma":
                    return ((df["close"] - df["mvg1"]) / df["mvg1"] < 0.02)
                if filter_name == "bb_rebound":
                    return df["close"] < df["bb_lower"] if "bb_lower" in df.columns else pd.Series(True, index=df.index)
                if filter_name == "sar_aligned":
                    return df["close"] > df["sar"] if "sar" in df.columns else pd.Series(True, index=df.index)
            else:  # short
                if filter_name == "trend_aligned":  # baseline removed 2026-06-18 (no-brainer; falls through to strict raise)
                    return down_trend
                if filter_name == "strong_trend":
                    return down_trend & (df["mvg2"] < df["mvg3"]) if "mvg3" in df.columns else down_trend
                if filter_name == "ma_momentum":
                    return (df["mvg1"] < df["mvg2"]) & (df["mvg1"] < df["mvg3"]) if "mvg3" in df.columns else (df["mvg1"] < df["mvg2"])
                if filter_name == "above_all_mas":
                    mask = (df["close"] < df["mvg1"]) & (df["close"] < df["mvg2"])
                    if "mvg3" in df.columns:
                        mask &= (df["close"] < df["mvg3"])
                    return mask
                if filter_name == "near_ma":
                    return ((df["mvg1"] - df["close"]) / df["mvg1"] < 0.02)
                if filter_name == "bb_rebound":
                    return df["close"] > df["bb_upper"] if "bb_upper" in df.columns else pd.Series(True, index=df.index)
                if filter_name == "sar_aligned":
                    return df["close"] < df["sar"] if "sar" in df.columns else pd.Series(True, index=df.index)

                # Combined Strategies
                if filter_name.startswith("combined_"):
                    # Component masks
                    _q50 = df["price_range_pct_q50"] if "price_range_pct_q50" in df.columns else df["price_range_pct"].rolling(700, min_periods=1).quantile(0.5)
                    low_vol_mask = down_trend & (df["price_range_pct"] < _q50)
                    vol_breakout_mask = down_trend & (df.get("quote_vol_ratio", 0.0) > 2.0)

                    if filter_name == "combined_union":
                        # Trade if EITHER Low Vol OR Volume Breakout
                        return low_vol_mask | vol_breakout_mask

                    if filter_name == "combined_intersection":
                        # Trade only if BOTH Low Vol AND Volume Breakout
                        return low_vol_mask & vol_breakout_mask

                    # New Combinations
                    above_all_mask = (df["close"] < df["mvg1"]) & (df["close"] < df["mvg2"])
                    if "mvg3" in df.columns: above_all_mask &= (df["close"] < df["mvg3"])

                    if filter_name == "combined_union_plus":
                        # Low Vol OR Vol Breakout OR Above All MAs
                        return low_vol_mask | vol_breakout_mask | above_all_mask

                    if filter_name == "combined_union_alt":
                        # Low Vol OR Above All MAs
                        return low_vol_mask | above_all_mask

        # No fallback: raise or return False mask depending on context
        if getattr(self, '_strict_filters', True):
            raise ValueError(f"Unknown filter name: {filter_name}")
        logger.debug("Unknown filter '%s' for position '%s' — treating as no-match",
                      filter_name, position)
        return pd.Series(False, index=df.index)

