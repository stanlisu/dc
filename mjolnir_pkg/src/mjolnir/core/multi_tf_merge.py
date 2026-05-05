"""Cross-TF feature merge — shared helper used by both research and live trading.

Bug 2 (audit A1): research and live were doing different things. Research
maintained its own MULTI_TF_BARS engines and merged with a +tf_seconds
shift; live only ran the base-TF feature engine, then silently dropped
unknown columns at scaler.transform. Models trained with cross-TF features
were getting only base-TF features at inference.

This helper is the single source of truth for the cross-TF align-and-shift
operation. Both ``mjolnir/core/research.py:engineer_features`` and
``mjolnir/trading.py:predict`` call it.

Contract
--------
``merge_cross_tf_features(base_feats, tf_feats)``:

- ``base_feats``: feature DataFrame on the base-TF index (one row per base bar,
  prefixed by nothing — the base-TF feature engine produced raw column names).
- ``tf_feats``: ``dict[str, DataFrame]`` mapping each higher-TF code (e.g.
  ``"15s"``, ``"30s"``) to that TF's feature DataFrame. Caller MUST construct
  these DataFrames with ``MjolnirFeatures(prefix=tf, ...)`` so columns are
  already named ``f"{tf}_{col}"``.
- Returns a single DataFrame on ``base_feats.index`` whose columns are the
  union of base columns and every higher-TF prefixed column. Higher-TF
  features are shifted forward by exactly one higher-TF bar duration to
  avoid look-ahead leakage (see research.py:333-365 for full rationale),
  then ffill-aligned to the base index.

Failure modes
-------------
- Unknown TF code (not in ``_TF_SECONDS``): raises ``RuntimeError`` so the
  operator can extend the map before re-running. No silent skip — that is
  precisely the failure pattern this helper is meant to prevent.
"""
from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd

from .features import _TF_SECONDS

# Columns that the per-TF feature engines may carry but which represent
# meta/target data — these must NOT be merged because:
#  (a) they would duplicate the base-TF target columns and shadow them, and
#  (b) MjolnirFeatures intentionally does NOT prefix meta columns even when
#      a prefix is set, so they collide name-for-name across TFs.
_META_COLS = frozenset({
    "ret", "ret_raw", "return", "return_long", "return_short",
    "return_long_raw", "return_short_raw",
    "position", "regime", "timestamp", "symbol", "year", "month",
})


def merge_cross_tf_features(
    base_feats: pd.DataFrame,
    tf_feats: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Merge per-TF prefixed feature frames into the base-TF index.

    See module docstring for full contract. Mirrors research.py:333-365
    inline logic exactly so live and research produce identical features.
    """
    if not tf_feats:
        return base_feats

    out = base_feats
    for tf, tf_df in tf_feats.items():
        if tf not in _TF_SECONDS:
            raise RuntimeError(
                f"Cross-TF merge: TF {tf!r} not in "
                f"_TF_SECONDS map {sorted(_TF_SECONDS)!r}. "
                "Add an entry to mjolnir/core/features.py "
                "before using this TF in MULTI_TF_BARS."
            )
        if tf_df is None or tf_df.empty:
            # WHY: caller is responsible for not passing empty frames; if it
            # happens here it is almost always because of an upstream load
            # failure — emit nothing for this TF and let the caller's
            # column-presence guard surface the issue.
            continue

        # Drop meta/target cols from higher-TF frame (would duplicate base targets).
        drop = [c for c in tf_df.columns if c in _META_COLS]
        tf_df = tf_df.drop(columns=drop, errors="ignore")

        # LEAKAGE FIX (research.py:333-365): higher-TF bars are stamped at
        # bar-OPEN but carry bar-CLOSE microstructure. Shift forward by one
        # full higher-TF bucket so a base row at T receives features from
        # the higher-TF bar that ENDED strictly before T.
        tf_seconds = _TF_SECONDS[tf]
        shifted = tf_df.copy()
        shifted.index = shifted.index + pd.Timedelta(seconds=tf_seconds)
        aligned = shifted.reindex(out.index, method="ffill")
        out = pd.concat([out, aligned], axis=1)
    return out


def expected_cross_tf_prefixes(multi_tfs: Iterable[str]) -> Dict[str, str]:
    """Return ``{tf: prefix_string}`` for caller-side guards (e.g. asserting
    every multi-TF carries at least one column into the merged frame)."""
    return {tf: f"{tf}_" for tf in multi_tfs}
