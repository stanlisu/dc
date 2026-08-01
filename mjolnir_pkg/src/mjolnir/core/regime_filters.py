"""Regime filter masks for MjolnirResearch.

Standalone functions — no class state needed. These only use the DataFrame,
filter name, and position to produce boolean Series masks.

Extracted from research.py to keep each module under ~700 lines for
PyArmor trial compatibility.

2026-08-01 — opt-in rolling-quantile atom thresholds: distribution-relative
atoms (deep_book, spread, vol, liq-pressure) can compute their cutoffs as a
CAUSAL trailing per-day quantile instead of the legacy full-frame quantile,
so cutoffs track market level shifts across eras (deep_book fired 0 times in
2026Q2 under stale cuts). Opt in via REGIME_ATOM_MODE="rolling_quantile" in
setting.json (see rolling_quantile_atoms_from_config); default is the legacy
fixed behavior, byte-identical.
"""

from __future__ import annotations

from typing import Mapping, NamedTuple, Optional

import numpy as np
import pandas as pd


def _obf():
    """Lazy accessor for the vendored obfuscation codec (see _obf/codec.py)."""
    from .._obf.codec import default
    return default()


# Every name named_filter() implements. Membership is checked BEFORE any
# missing-column guard so an unknown/retired name raises on ANY frame —
# including degenerate/warmup frames lacking mvg1/mvg2/close, which previously
# fell into the all-True price-column guard and silently fired on every bar.
# "baseline" is listed so it reaches its dedicated removal raise below.
KNOWN_FILTERS = frozenset({
    "baseline",
    # microstructure (Mjolnir-specific)
    "high_liquidation_pressure", "low_liquidation_pressure",
    "funding_positive", "funding_negative",
    "deep_book", "trade_imbalance",
    "basis_premium", "basis_discount",
    "pre_funding_settlement",
    "ofi_positive",
    "tight_spread", "wide_spread",
    # standard price filters (shared with Agamotto)
    "trend_aligned", "strong_trend", "ma_momentum",
    "rsi_oversold", "rsi_overbought",
    "macd_bullish", "macd_bearish",
    "adx_trend", "bb_rebound",
    "high_volume", "vol_breakout", "low_volume",
    "high_vol", "low_vol",
    "mom_positive",
})


# ── Atom threshold modes (2026-08-01, opt-in rolling-quantile cutoffs) ────────

ATOM_MODE_FIXED = "fixed"
ATOM_MODE_ROLLING_QUANTILE = "rolling_quantile"

# Default trailing window for rolling-quantile cutoffs: 30 days of 5s bars
# (30 * 17280 = 518,400). Configurable via REGIME_ATOM_QUANTILE_WINDOW_BARS.
DEFAULT_QUANTILE_WINDOW_BARS = 30 * 17_280

# Quantile level per distribution-relative atom. These are the EXACT levels
# the legacy fixed rules use (`df[col].quantile(level)` over the whole frame):
# fixed mode reads its literals from here (masks stay byte-identical), and
# rolling mode reproduces each atom's historical firing rate by construction —
# e.g. deep_book short fires on the bottom 40% in both modes; rolling just
# measures that 40% on a causal trailing window instead of the full frame.
# (The 2026-08-01 design note floated a 0.8 default for deep_book because its
# fixed-rule percentile was assumed unknowable; the rule IS a frame quantile
# at 0.6/0.4, so those levels reproduce the 2025 firing rate exactly and are
# the defaults. Calibration overrides go in REGIME_ATOM_QUANTILE_LEVELS.)
ATOM_QUANTILE_LEVELS = {
    "high_liquidation_pressure": 0.75,  # fires when feature > cutoff
    "low_liquidation_pressure": 0.25,   # fires when feature < cutoff
    "deep_book_long": 0.6,              # fires when feature > cutoff
    "deep_book_short": 0.4,             # fires when feature < cutoff
    "tight_spread": 0.5,                # fires when feature < cutoff
    "wide_spread": 0.5,                 # fires when feature > cutoff
    "high_vol": 0.5,                    # fires when feature > cutoff
    "low_vol": 0.5,                     # fires when feature < cutoff
}


class RollingQuantileAtoms(NamedTuple):
    """Resolved rolling-quantile atom config: trailing window + levels."""

    window_bars: int
    levels: Mapping[str, float]


# Keys that only have meaning in rolling-quantile mode. Their presence with
# mode absent/"fixed" raises — a config carrying them while running fixed
# thresholds is a misconfiguration waiting to be misread as active.
_ROLLING_ONLY_KEYS = (
    "REGIME_ATOM_QUANTILE_WINDOW_BARS",
    "REGIME_ATOM_QUANTILE_LEVELS",
)


def rolling_quantile_atoms_from_config(
    cfg: Mapping,
) -> Optional[RollingQuantileAtoms]:
    """Parse REGIME_ATOM_* keys from a setting.json dict.

    Returns None for fixed (legacy) mode, or a RollingQuantileAtoms for
    rolling-quantile mode.

    REGIME_ATOM_MODE absent -> fixed. This absent-key default is the
    explicitly sanctioned back-compat default (agreed opt-in design,
    2026-08-01): every setting.json written before then predates the key and
    must produce byte-identical masks. A PRESENT but invalid value always
    raises — no silent fallback. Rolling-only keys present while the mode is
    absent/"fixed" also raise (they would be silently inert).
    """
    mode = cfg["REGIME_ATOM_MODE"] if "REGIME_ATOM_MODE" in cfg \
        else ATOM_MODE_FIXED
    if mode == ATOM_MODE_FIXED:
        stray = [k for k in _ROLLING_ONLY_KEYS if k in cfg]
        if stray:
            raise ValueError(
                f"{', '.join(stray)} present but inert: these keys require "
                f"REGIME_ATOM_MODE={ATOM_MODE_ROLLING_QUANTILE!r} "
                f"(mode is {cfg.get('REGIME_ATOM_MODE', '<absent>')!r})")
        return None
    if mode != ATOM_MODE_ROLLING_QUANTILE:
        raise ValueError(
            f"REGIME_ATOM_MODE={mode!r} invalid; expected "
            f"{ATOM_MODE_FIXED!r} or {ATOM_MODE_ROLLING_QUANTILE!r}")

    if "REGIME_ATOM_QUANTILE_WINDOW_BARS" in cfg:
        window_bars = cfg["REGIME_ATOM_QUANTILE_WINDOW_BARS"]
        if isinstance(window_bars, bool) or not isinstance(window_bars, int) \
                or window_bars <= 0:
            raise ValueError(
                "REGIME_ATOM_QUANTILE_WINDOW_BARS must be a positive int, "
                f"got {window_bars!r}")
    else:
        # Documented default for the NEW opt-in mode (agreed design:
        # 30 days of 5s bars); a named constant, not a magic number.
        window_bars = DEFAULT_QUANTILE_WINDOW_BARS

    levels = dict(ATOM_QUANTILE_LEVELS)
    if "REGIME_ATOM_QUANTILE_LEVELS" in cfg:
        overrides = cfg["REGIME_ATOM_QUANTILE_LEVELS"]
        if not isinstance(overrides, Mapping):
            raise ValueError(
                "REGIME_ATOM_QUANTILE_LEVELS must be a mapping "
                f"{{atom: level}}, got {type(overrides).__name__}")
        for key, level in overrides.items():
            if key not in levels:
                raise ValueError(
                    f"REGIME_ATOM_QUANTILE_LEVELS: unknown atom {key!r}; "
                    f"valid: {sorted(levels)}")
            if isinstance(level, bool) \
                    or not isinstance(level, (int, float)) \
                    or not 0.0 < float(level) < 1.0:
                raise ValueError(
                    f"REGIME_ATOM_QUANTILE_LEVELS[{key!r}] must be a number "
                    f"in (0, 1), got {level!r}")
            levels[key] = float(level)
    return RollingQuantileAtoms(window_bars=window_bars, levels=levels)


def _bar_timestamps(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Per-row timestamps for rolling-quantile cutoffs.

    Prefers a DatetimeIndex; otherwise uses the 'timestamp' column that the
    streaming/verticalize paths stamp before masks are applied (those frames
    carry an integer index). Anything else raises — rolling mode cannot be
    causal without real timestamps.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        ts = df.index
    elif "timestamp" in df.columns:
        ts = pd.DatetimeIndex(df["timestamp"])
    else:
        raise ValueError(
            "rolling_quantile atom mode requires a DatetimeIndex or a "
            "'timestamp' column to build causal trailing cutoffs")
    if ts.hasnans:
        raise ValueError("rolling_quantile atom mode: NaT in bar timestamps")
    if ts.tz is not None:
        # Normalize to UTC so day-bucket keys are true UTC days regardless
        # of the frame's timezone.
        ts = ts.tz_convert("UTC")
    return ts


def _window_days(ts_sorted: pd.DatetimeIndex, window_bars: int) -> int:
    """Convert a window in BARS to whole trailing DAYS via bar spacing.

    E.g. 518,400 bars at 5s spacing -> exactly 30 days. Uses the median
    positive spacing so occasional gaps don't skew the conversion.
    """
    if len(ts_sorted) < 2:
        raise ValueError(
            "rolling_quantile atom mode needs >= 2 bars to infer bar "
            f"spacing; got {len(ts_sorted)} row(s)")
    diffs = np.diff(ts_sorted.asi8)  # nanoseconds
    pos = diffs[diffs > 0]
    if len(pos) == 0:
        raise ValueError(
            "rolling_quantile atom mode: all bar timestamps identical — "
            "cannot infer bar spacing")
    bar_sec = float(np.median(pos)) / 1e9
    return max(1, int(round(window_bars * bar_sec / 86_400.0)))


def rolling_quantile_cutoff(
    df: pd.DataFrame,
    col: str,
    level: float,
    window_bars: int,
) -> pd.Series:
    """Causal trailing quantile cutoff for df[col], aligned to df.index.

    Exact per-bar trailing quantiles over the default window (518,400 bars
    per symbol) are O(n*window) — infeasible at 5s scale. Documented
    approximation (coarser, but strictly causal):

      1. per-UTC-day quantile of that day's values: q_d
      2. trailing mean over the last `window_days` daily quantiles: m_d
         (min_periods = window_days: a partial window yields NaN, so warmup
         behavior is deterministic across walk-forward windows)
      3. shift one full day: cutoff applied to day d = m_{d-1}
      4. broadcast each day's cutoff onto that day's bars

    Causality: step 3 shifts by one FULL day, so the cutoff applied to any
    bar of day d is built exclusively from days <= d-1 — no same-day leakage,
    and in particular bar t's own value never enters its cutoff.

    Warmup: the FIRST window_days per symbol (and any day whose trailing
    window has a data-gap day) get NaN cutoffs -> comparisons are False ->
    the regime fails CLOSED. Research runs must therefore extend LOAD_START
    by one full window (default 30 days) before the intended study start.

    Per-symbol: when a 'symbol' column is present, each symbol gets its own
    independent quantile stream (constraint: no cross-symbol pooling).

    Live deployment path (follow-up, NOT implemented here): live bots cannot
    span this window from their bar buffer — research is to dump the frozen
    per-day cutoffs alongside the weights and the live bot loads them like
    model artifacts. Until then MjolnirTrading rejects rolling mode at boot.
    """
    ts = _bar_timestamps(df)
    values = df[col].to_numpy(dtype=float)
    out = np.full(len(df), np.nan)

    if "symbol" in df.columns:
        group_ids = df["symbol"].to_numpy()
        groups = [
            np.flatnonzero(group_ids == g) for g in pd.unique(group_ids)
        ]
    else:
        groups = [np.arange(len(df))]

    for idx in groups:
        g_ts = ts[idx]
        order = np.argsort(g_ts.asi8, kind="stable")
        ts_sorted = g_ts[order]
        s = pd.Series(values[idx][order], index=ts_sorted)
        daily_q = s.resample("1D").quantile(level)
        w_days = _window_days(ts_sorted, window_bars)
        cut_daily = daily_q.rolling(w_days, min_periods=w_days).mean().shift(1)
        # Map each bar to ITS day's cutoff by exact day key (no ffill
        # ambiguity); restore original row order.
        out[idx[order]] = cut_daily.reindex(ts_sorted.normalize()).to_numpy()
    return pd.Series(out, index=df.index)


def _atom_cutoff(
    df: pd.DataFrame,
    col: str,
    atom_key: str,
    atom_cfg: Optional[RollingQuantileAtoms],
):
    """Cutoff for one distribution-relative atom.

    fixed (atom_cfg None): the legacy full-frame quantile at the atom's
    canonical level — the exact expression the pre-2026-08-01 code used
    (ATOM_QUANTILE_LEVELS holds those literals; masks byte-identical).
    rolling_quantile: causal trailing per-day quantile, see
    rolling_quantile_cutoff().
    """
    if atom_cfg is None:
        return df[col].quantile(ATOM_QUANTILE_LEVELS[atom_key])
    return rolling_quantile_cutoff(
        df, col, atom_cfg.levels[atom_key], atom_cfg.window_bars)


def apply_filter_mask(
    df: pd.DataFrame,
    filter_name,
    position: str,
    atom_cfg: Optional[RollingQuantileAtoms] = None,
) -> pd.Series:
    """Return a boolean Series: True = include in filter.

    Supports:
    - String filter names with _and_ / _or_ composition
    - List filters: ["filter_a", "&", "filter_b"]
    - Mjolnir-specific microstructure filters
    - Standard Agamotto price filters (baseline, rsi, macd, etc.)

    atom_cfg=None is the documented back-compat default = legacy fixed
    thresholds (byte-identical masks). Pass a RollingQuantileAtoms (built by
    rolling_quantile_atoms_from_config) to opt in to causal trailing
    rolling-quantile cutoffs for the distribution-relative atoms.
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
                sub = apply_filter_mask(df, item, position, atom_cfg)
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
        # Accept coded regimes (rename rollout): decode code->real before the
        # real-name named_filter() chain; real names pass through unchanged.
        name = _obf().decode_regime_tolerant(name)
        if "_and_" in name:
            parts = name.split("_and_")
            mask = None
            for p in parts:
                sub = apply_filter_mask(df, p.strip(), position, atom_cfg)
                mask = sub if mask is None else (mask & sub)
            return mask if mask is not None else pd.Series(True, index=df.index)
        if "_or_" in name:
            parts = name.split("_or_")
            mask = None
            for p in parts:
                sub = apply_filter_mask(df, p.strip(), position, atom_cfg)
                mask = sub if mask is None else (mask | sub)
            return mask if mask is not None else pd.Series(True, index=df.index)
    else:
        name = str(filter_name).lower().strip()

    if df.empty:
        return pd.Series(dtype=bool)

    # Strip trailing _long / _short from the filter name
    name = name.replace("_long", "").replace("_short", "")

    return named_filter(df, name, position, atom_cfg)


def named_filter(
    df: pd.DataFrame,
    name: str,
    position: str,
    atom_cfg: Optional[RollingQuantileAtoms] = None,
) -> pd.Series:
    """Dispatch to a named filter implementation.

    Raises ValueError for unknown names regardless of df contents; the
    per-branch missing-column all-True guards apply to KNOWN names only
    (and in BOTH atom modes — the guard precedes any cutoff computation).

    atom_cfg=None = legacy fixed thresholds (documented back-compat default;
    see apply_filter_mask / rolling_quantile_atoms_from_config).
    """
    if name not in KNOWN_FILTERS:
        raise ValueError(f"Unknown filter: {name!r}")

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
        return df[col] > _atom_cutoff(df, col, name, atom_cfg)

    if name == "low_liquidation_pressure":
        col = "liq_burst_ratio"
        if col not in df.columns:
            return true
        return df[col] < _atom_cutoff(df, col, name, atom_cfg)

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
            return df[col] > _atom_cutoff(df, col, "deep_book_long", atom_cfg)
        return df[col] < _atom_cutoff(df, col, "deep_book_short", atom_cfg)

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

    # oi_expansion/oi_contraction REMOVED 2026-07-24: they gated on the SIGN
    # of oi_velocity, which live (5s REST poll) and offline (Tardis poller)
    # disagree on for ~10% of 30s bars — an unreplicable regime gate. A stale
    # stack referencing them now hits the Unknown-filter raise below (never a
    # silent all-True mask).

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
        return df[col] < _atom_cutoff(df, col, name, atom_cfg)

    if name == "wide_spread":
        col = "relative_spread"
        if col not in df.columns:
            return true
        return df[col] > _atom_cutoff(df, col, name, atom_cfg)

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
        return df[col] > _atom_cutoff(df, col, name, atom_cfg)

    if name == "low_vol":
        col = "price_range_pct"
        if col not in df.columns:
            return true
        return df[col] < _atom_cutoff(df, col, name, atom_cfg)

    if name == "mom_positive":
        col = "mom"
        if col not in df.columns:
            return true
        return df[col] > 0 if position == "long" else df[col] < 0

    # Unreachable for unknown names (rejected at the top); firing here means
    # KNOWN_FILTERS drifted out of sync with the branches above.
    raise ValueError(
        f"Filter {name!r} is in KNOWN_FILTERS but has no implementation")
