"""Scale-free replacements for the raw price/volume level features.

`verticalize()` stacks every symbol into one frame, so a feature carried in price
or volume UNITS is partly a symbol ID. Seven features survive
`select_feature_columns` because they have no percentage twin, and all seven are
raw levels: sar, bb_upper, bb_lower, macd, macdhist, obv, ad.

Measured on 10 symbols x 360,010 rows of 15m klines (price scale $0.0045 to
$88,468), the raw levels are not merely diluted but SIGN-INVERTED by pooling:
`sar` reads +0.0042 pooled against a true -0.0388, and 3 of 10 symbols disagree
with the pooled sign. Every transform below scores 0/10 sign flips and collapses
the pooled-vs-within gap to <=0.0014. Full table in
`agamotto_pkg/tests/test_scale_free_levels.py`.

DERIVED, NOT RECOMPUTED. Each transform is built from the already-computed raw
column so it inherits that engine's indicator parameters — agamotto uses
`BBANDS(timeperiod=20, nbdev 2/2)` (`research.py:422`) while mjolnir uses the
talib default of 5 (`mjolnir/core/features.py:396`). Recomputing here would
silently impose one engine's choice on the other.

THE RAW COLUMNS MUST STAY IN THE FRAME. `research_filters.py` gates regimes on
them (`close > sar` :330, `close < bb_lower` :327, `close > bb_upper` :347,
`macdhist > 0` :213). They are kept out of the MODEL by
`gauntlet/rolling_predict_returns.select_feature_columns`, which governs training
input only — regimes are evaluated against the full frame (`research.py:628`).

MIRRORED IN MJOLNIR. `mjolnir/core/features_scalefree.py` carries an identical
copy because mjolnir_pkg does not depend on agamotto_pkg (no package declares
that dependency; orb only gets it via co-installation). A parity test asserts the
two produce identical output.
"""
from typing import Dict, List

import numpy as np
import pandas as pd

# The raw level -> its scale-free replacement(s).
REQUIRED_SOURCE_FIELDS = [
    "close", "sar", "bb_upper", "bb_lower", "macd", "macdhist", "obv", "ad",
    "volume",
]

# NOTE: the `_FEATURES` suffix is load-bearing — extract_inventory.py:33
# collects members of a `*_FEATURES` list as obfuscatable feature names.
# A `*_COLUMNS` name would instead mark them PASSTHROUGH (:36) and ship
# the real names into the filter parquet.
SCALE_FREE_FEATURES = [
    "sar_dist",        # (close - sar) / close        signed distance to the flip
    "bb_pctb",         # (close - bbl) / (bbu - bbl)  classic %B, band-relative
    "bb_width",        # (bbu - bbl) / close          volatility, scale-free
    "macd_norm",       # macd / close                 MACD is an EMA difference
    "macdhist_norm",   # macdhist / close
    "obv_slope",       # obv.diff(w) / volume.sum(w)  bounded flow, not a total
    "ad_slope",        # ad.diff(w)  / volume.sum(w)
]

DEFAULT_WINDOW = 20


def _safe(num: pd.Series, den: pd.Series) -> pd.Series:
    """Divide, mapping a zero/degenerate denominator to NaN rather than inf.

    An inf propagates through the |IC| ranker in
    `rolling_predict_returns._compute_train_window_top_n_ic` and silently
    poisons feature selection, so it must never leave this module.
    """
    out = num / den.replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _flow(series: pd.Series, window: int, is_cumulative: bool) -> pd.Series:
    """Net flow over `window`, from either a running total or a ready-made diff."""
    s = series.astype(float)
    return s.diff(window) if is_cumulative else s


def scale_free_levels(df: pd.DataFrame, window: int = DEFAULT_WINDOW,
                      prefix: str = "",
                      obv_is_cumulative: bool = True) -> pd.DataFrame:
    """Build the seven scale-free columns from an OHLCV+indicator frame.

    Args:
        df: frame carrying `{prefix}close`, `{prefix}sar`, `{prefix}bb_upper`,
            `{prefix}bb_lower`, `{prefix}macd`, `{prefix}macdhist`,
            `{prefix}obv`, `{prefix}ad`, `{prefix}volume`.
        window: lookback for the obv/ad flow measures.
        prefix: timeframe prefix for multi-TF panels (orb/scepter carry
            `15m_sar`, `1h_sar`, ...); empty for agamotto's single-TF panel.
        obv_is_cumulative: whether `obv`/`ad` arrive as running totals.
            True (default) matches `talib.OBV(c, v)` as mjolnir builds it
            (`core/features.py:420`) AND the experiment that justified this
            change, so the default reproduces the measured numbers.
            **agamotto must pass False** — `research.py:411-412` already stores
            `obv_raw.diff(14)`, so differencing again would take a SECOND
            difference. That failure is silent: on a steady flow it returns 0.0
            for every row while still looking like a valid feature.

    Returns:
        DataFrame of the seven columns, prefixed to match, indexed like `df`.

    Raises:
        KeyError: if any source column is missing. No silent fallback — a renamed
            upstream column would otherwise drop a top-3 feature from the panel.
    """
    missing = [c for c in REQUIRED_SOURCE_FIELDS if f"{prefix}{c}" not in df.columns]
    if missing:
        raise KeyError(
            f"scale_free_levels: missing source column(s) "
            f"{[f'{prefix}{c}' for c in missing]} — cannot build "
            f"{SCALE_FREE_FEATURES}. See CLAUDE.md 'no silent fallbacks'.")

    g = lambda name: df[f"{prefix}{name}"]                       # noqa: E731
    close = g("close").astype(float)
    bb_upper, bb_lower = g("bb_upper").astype(float), g("bb_lower").astype(float)
    vol_sum = g("volume").astype(float).rolling(window, min_periods=window).sum()

    cols: Dict[str, pd.Series] = {
        "sar_dist":      _safe(close - g("sar").astype(float), close),
        # %B subsumes both band distances (near-collinear with it) and measured
        # -0.0500, the strongest single feature in the panel.
        "bb_pctb":       _safe(close - bb_lower, bb_upper - bb_lower),
        "bb_width":      _safe(bb_upper - bb_lower, close),
        "macd_norm":     _safe(g("macd").astype(float), close),
        "macdhist_norm": _safe(g("macdhist").astype(float), close),
        # A running-total obv/ad encodes listing age as much as price action, so
        # difference it first. An ALREADY-differenced one (agamotto) is only in
        # the wrong UNITS — normalise, never difference twice.
        "obv_slope":     _safe(_flow(g("obv"), window, obv_is_cumulative), vol_sum),
        "ad_slope":      _safe(_flow(g("ad"), window, obv_is_cumulative), vol_sum),
    }
    return pd.DataFrame({f"{prefix}{k}": v for k, v in cols.items()},
                        index=df.index)
