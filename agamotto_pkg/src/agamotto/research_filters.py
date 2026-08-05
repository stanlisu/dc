"""Filter evaluation functions and constants for Agamotto research."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional
import re
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _obf():
    """Lazy accessor for the vendored obfuscation codec (see _obf/codec.py)."""
    from ._obf.codec import default
    return default()


def _require_col(df: pd.DataFrame, filter_name: str, col: str) -> None:
    """Fail loud when a known filter's source column is absent."""
    if col not in df.columns:
        raise ValueError(
            f"Filter {filter_name!r} requires column {col!r}; frame is missing "
            f"it ({len(df.columns)} columns present). An all-True fallback "
            "here fires on every bar — a `baseline` regime under another "
            "name, which CLAUDE.md removed forever (2026-06-18). Failing loud "
            "instead; drop the regime from the stack or build the column.")


def _volume_ratio(df: pd.DataFrame, filter_name: str) -> pd.Series:
    """Volume-ratio series backing low_volume/high_volume/vol_breakout."""
    for col in ("quote_vol_ratio", "vol_ratio"):
        if col in df.columns:
            return df[col]
    raise ValueError(
        f"Filter {filter_name!r} requires a volume-ratio column; frame is "
        "missing both 'quote_vol_ratio' and 'vol_ratio'. A scalar 1.0 "
        "fallback here returns a bool instead of a per-row mask — "
        "failing loud instead. See CLAUDE.md no-silent-fallback rules.")


# Directionally-biased filters: these only make sense for one side.
LONG_ONLY_FILTERS = frozenset({
    "macd_bullish", "stoch_bullish", "rsi_oversold",
    "mfi_oversold", "bop_bullish", "roc_positive",
})
SHORT_ONLY_FILTERS = frozenset({
    "macd_bearish", "rsi_overbought",
    "mfi_overbought", "bop_bearish", "roc_negative",
})

MVG_DEPENDENT_FILTERS = frozenset({
    "trend_aligned", "strong_trend", "ma_momentum", "above_all_mas",
    "near_ma", "bb_rebound", "sar_aligned",
})

BASE_REGIMES = [
    "vol_breakout",
    "vol_breakout_and_strong_trend", "vol_breakout_and_ma_momentum",
    "vol_breakout_and_above_all_mas", "vol_breakout_and_rsi_oversold",
    "vol_breakout_and_rsi_overbought", "vol_breakout_and_macd_bullish",
    "vol_breakout_and_macd_bearish", "vol_breakout_and_cci_reversal",
    "vol_breakout_and_adx_trend", "vol_breakout_and_bb_rebound",
    "vol_breakout_and_mom_positive", "vol_breakout_and_strong_candle",
    "vol_breakout_and_near_ma", "vol_breakout_and_stoch_bullish",
    "high_volume_and_strong_trend", "high_volume_and_ma_momentum",
    "high_volume_and_above_all_mas", "high_volume_and_rsi_oversold",
    "high_volume_and_rsi_overbought", "high_volume_and_macd_bullish",
    "high_volume_and_macd_bearish", "high_volume_and_cci_reversal",
    "high_volume_and_adx_trend", "high_volume_and_bb_rebound",
    "high_volume_and_mom_positive", "high_volume_and_strong_candle",
    "low_volume_and_strong_trend", "low_volume_and_ma_momentum",
    "low_volume_and_above_all_mas", "low_volume_and_rsi_oversold",
    "low_volume_and_bb_rebound", "low_volume_and_cci_reversal",
]

_SWEEP_VOL_FILTERS = ["low_volume", "high_volume", "vol_breakout"]
_SWEEP_TECH_FILTERS = [
    "above_all_mas", "rsi_oversold", "rsi_overbought",
    "cci_reversal", "bb_rebound", "macd_bullish", "macd_bearish",
    "stoch_bullish", "adx_trend", "mom_positive", "strong_trend",
    "ma_momentum", "near_ma", "strong_candle",
]


def allowed_positions(filter_name: str) -> list:
    """Return ['long'], ['short'], or ['long','short'] for a regime name."""
    if isinstance(filter_name, str):
        filter_name = _obf().decode_regime_tolerant(filter_name)
    parts = filter_name.split("_and_") if "_and_" in filter_name else [filter_name]
    needs_long = False
    needs_short = False
    for part in parts:
        base = part.strip()
        tokens = base.split("_")
        if len(tokens) > 1 and tokens[0] in ("15m", "1h", "4h", "1d"):
            base = "_".join(tokens[1:])
        if base in LONG_ONLY_FILTERS:
            needs_long = True
        if base in SHORT_ONLY_FILTERS:
            needs_short = True
    if needs_long and needs_short:
        return []
    if needs_long:
        return ["long"]
    if needs_short:
        return ["short"]
    return ["long", "short"]


def comprehensive_sweep_regimes() -> list[str]:
    c = _obf()
    names = (
        _SWEEP_VOL_FILTERS
        + _SWEEP_TECH_FILTERS
        + [f"{v}_and_{t}" for v in _SWEEP_VOL_FILTERS
           for t in _SWEEP_TECH_FILTERS]
    )
    return [c.encode_regime(n) for n in names]


def generate_regime_stack() -> list[dict]:
    c = _obf()
    rows = []
    for regime in BASE_REGIMES:
        for pos in allowed_positions(regime):
            rows.append({"regime": c.encode_regime(regime), "position": pos})
    return rows


def apply_filter_mask(
    df: pd.DataFrame,
    filter_name: str | list,
    position: str,
    strict_filters: bool = True,
    allowed_positions_fn = allowed_positions,
) -> pd.Series:
    """Evaluates a named filter or list of filters against DataFrame df."""
    if isinstance(filter_name, list):
        mask = None
        current_op = None
        for item in filter_name:
            item = str(item).strip()
            if item in ["|", "&"]:
                current_op = item
            else:
                sub_mask = apply_filter_mask(df, item, position, strict_filters, allowed_positions_fn)
                if mask is None:
                    mask = sub_mask
                else:
                    if current_op == "|":
                        mask = mask | sub_mask
                    elif current_op == "&":
                        mask = mask & sub_mask
                    else:
                        mask = mask & sub_mask
        return mask if mask is not None else pd.Series(True, index=df.index)

    if isinstance(filter_name, str):
        filter_name = filter_name.lower().strip()

        if filter_name.startswith("__"):
            return pd.Series(True, index=df.index)

        filter_name = _obf().decode_regime_tolerant(filter_name)

        if "_and_" in filter_name:
            parts = filter_name.split("_and_")
            mask = None
            for part in parts:
                sub_mask = apply_filter_mask(df, part.strip(), position, strict_filters, allowed_positions_fn)
                mask = sub_mask if mask is None else (mask & sub_mask)
            return mask if mask is not None else pd.Series(True, index=df.index)

        if "_or_" in filter_name:
            parts = filter_name.split("_or_")
            mask = None
            for part in parts:
                sub_mask = apply_filter_mask(df, part.strip(), position, strict_filters, allowed_positions_fn)
                mask = sub_mask if mask is None else (mask | sub_mask)
            return mask if mask is not None else pd.Series(True, index=df.index)

    if df.empty:
        return pd.Series(dtype=bool)

    if isinstance(filter_name, str):
        filter_name = filter_name.replace("_long", "").replace("_short", "")
        filter_name = re.sub(r'^(?:15m|1h|4h|1d)_', '', filter_name)

    _allowed = allowed_positions_fn(filter_name) if isinstance(filter_name, str) else None
    if _allowed and position not in _allowed:
        return pd.Series(False, index=df.index)

    if position == "long":
        if filter_name == "low_vol":
            _require_col(df, filter_name, "price_range_pct")
            q50 = df["price_range_pct_q50"] if "price_range_pct_q50" in df.columns else df["price_range_pct"].rolling(700, min_periods=1).quantile(0.5)
            return df["price_range_pct"] < q50
        if filter_name == "high_vol":
            _require_col(df, filter_name, "price_range_pct")
            q50 = df["price_range_pct_q50"] if "price_range_pct_q50" in df.columns else df["price_range_pct"].rolling(700, min_periods=1).quantile(0.5)
            return df["price_range_pct"] > q50
        if filter_name == "strong_candle":
            _require_col(df, filter_name, "open_close_pct")
            return (df["open_close_pct"] > 0.005)

        if filter_name == "rsi_oversold":
            _require_col(df, filter_name, "rsi")
            return df["rsi"] < 30
        if filter_name == "macd_bullish":
            _require_col(df, filter_name, "macdhist")
            return df["macdhist"] > 0
        if filter_name == "stoch_bullish":
            _require_col(df, filter_name, "stoch_k")
            return (df["stoch_k"] > df["stoch_d"])
        if filter_name == "cci_reversal":
            _require_col(df, filter_name, "cci")
            return df["cci"] > 100
        if filter_name == "adx_trend":
            _require_col(df, filter_name, "adx")
            return (df["adx"] > 25)
        if filter_name == "mom_positive":
            _require_col(df, filter_name, "mom")
            return df["mom"] > 0

        if filter_name in ("low_volume", "high_volume", "vol_breakout"):
            v_ratio = _volume_ratio(df, filter_name)
            if filter_name == "low_volume": return (v_ratio < 1.0)
            if filter_name == "high_volume": return (v_ratio > 1.0)
            return (v_ratio > 2.0)

        if filter_name == "buy_pressure":
            _require_col(df, filter_name, "buy_pressure")
            return df["buy_pressure"] > 0.55
        if filter_name == "mfi_oversold":
            _require_col(df, filter_name, "mfi")
            return df["mfi"] < 30
        if filter_name == "bop_bullish":
            _require_col(df, filter_name, "bop")
            return df["bop"] > 0.1
        if filter_name == "roc_positive":
            _require_col(df, filter_name, "roc")
            return df["roc"] > 0
    else:  # short
        if filter_name == "low_vol":
            _require_col(df, filter_name, "price_range_pct")
            q50 = df["price_range_pct_q50"] if "price_range_pct_q50" in df.columns else df["price_range_pct"].rolling(700, min_periods=1).quantile(0.5)
            return df["price_range_pct"] < q50
        if filter_name == "high_vol":
            _require_col(df, filter_name, "price_range_pct")
            q50 = df["price_range_pct_q50"] if "price_range_pct_q50" in df.columns else df["price_range_pct"].rolling(700, min_periods=1).quantile(0.5)
            return df["price_range_pct"] > q50
        if filter_name == "strong_candle":
            _require_col(df, filter_name, "open_close_pct")
            return (df["open_close_pct"] < -0.005)

        if filter_name == "rsi_overbought":
            _require_col(df, filter_name, "rsi")
            return df["rsi"] > 70
        if filter_name == "macd_bearish":
            _require_col(df, filter_name, "macdhist")
            return df["macdhist"] < 0
        if filter_name == "stoch_bullish":
            _require_col(df, filter_name, "stoch_k")
            return (df["stoch_k"] > df["stoch_d"])
        if filter_name == "cci_reversal":
            _require_col(df, filter_name, "cci")
            return df["cci"] < -100
        if filter_name == "adx_trend":
            _require_col(df, filter_name, "adx")
            return (df["adx"] > 25)
        if filter_name == "mom_positive":
            _require_col(df, filter_name, "mom")
            return df["mom"] < 0

        if filter_name in ("low_volume", "high_volume", "vol_breakout"):
            v_ratio = _volume_ratio(df, filter_name)
            if filter_name == "low_volume": return (v_ratio < 1.0)
            if filter_name == "high_volume": return (v_ratio > 1.0)
            return (v_ratio > 2.0)

        if filter_name == "buy_pressure":
            _require_col(df, filter_name, "buy_pressure")
            return df["buy_pressure"] < 0.45
        if filter_name == "mfi_overbought":
            _require_col(df, filter_name, "mfi")
            return df["mfi"] > 70
        if filter_name == "bop_bearish":
            _require_col(df, filter_name, "bop")
            return df["bop"] < -0.1
        if filter_name == "roc_negative":
            _require_col(df, filter_name, "roc")
            return df["roc"] < 0

    if isinstance(filter_name, str) and (
            filter_name in MVG_DEPENDENT_FILTERS
            or filter_name.startswith("combined_")):
        required_base = ["close", "mvg1", "mvg2"]
        missing = [col for col in required_base if col not in df.columns]
        if missing:
            raise ValueError(
                f"Filter {filter_name!r} requires price columns; frame is "
                f"missing {missing}. An all-True fallback here fires on "
                "every bar (baseline-shaped) — failing loud instead. "
                "See CLAUDE.md no-silent-fallback rules.")

        up_trend = (df["close"] > df["mvg1"]) & (df["mvg1"] > df["mvg2"])
        down_trend = (df["close"] < df["mvg1"]) & (df["mvg1"] < df["mvg2"])

        if position == "long":
            if filter_name == "trend_aligned":
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
                _require_col(df, filter_name, "bb_lower")
                return df["close"] < df["bb_lower"]
            if filter_name == "sar_aligned":
                _require_col(df, filter_name, "sar")
                return df["close"] > df["sar"]
        else:  # short
            if filter_name == "trend_aligned":
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
                _require_col(df, filter_name, "bb_upper")
                return df["close"] > df["bb_upper"]
            if filter_name == "sar_aligned":
                _require_col(df, filter_name, "sar")
                return df["close"] < df["sar"]

            if filter_name.startswith("combined_"):
                _q50 = df["price_range_pct_q50"] if "price_range_pct_q50" in df.columns else df["price_range_pct"].rolling(700, min_periods=1).quantile(0.5)
                low_vol_mask = down_trend & (df["price_range_pct"] < _q50)
                vol_breakout_mask = down_trend & (df.get("quote_vol_ratio", 0.0) > 2.0)

                if filter_name == "combined_union":
                    return low_vol_mask | vol_breakout_mask
                if filter_name == "combined_intersection":
                    return low_vol_mask & vol_breakout_mask

                above_all_mask = (df["close"] < df["mvg1"]) & (df["close"] < df["mvg2"])
                if "mvg3" in df.columns: above_all_mask &= (df["close"] < df["mvg3"])

                if filter_name == "combined_union_plus":
                    return low_vol_mask | vol_breakout_mask | above_all_mask
                if filter_name == "combined_union_alt":
                    return low_vol_mask | above_all_mask

    if strict_filters:
        raise ValueError(f"Unknown filter name: {filter_name}")
    logger.debug("Unknown filter '%s' for position '%s' — treating as no-match",
                  filter_name, position)
    return pd.Series(False, index=df.index)
