"""Reversible, structure-preserving codec for marvel regime & feature names.

PRIVATE (lives only in dc). Maps proprietary atom/feature names <-> opaque codes
while preserving the structural tokens every parser in marvel depends on:

  * conjunction     "_and_"            (cross-TF / multi-filter regime composition)
  * position suffix "_long" / "_short"
  * TF prefix       "<n><unit>_"       e.g. 15m_, 1h_, 4h_, 1d_, 5s_, 15s_, 30s_, 1m_, 5m_

Examples
  encode_regime("1d_strong_trend_and_1h_macd_bullish_long")
      -> "1d_r066_and_1h_r042_long"
  encode_feature("15m_price_range")     -> "15m_f050"
  encode_feature("kyle_lambda")         -> "f025"

Fail-fast: encoding/decoding an UNKNOWN base raises KeyError — never passes the
raw string through (per marvel CLAUDE.md: no silent fallbacks). Callers that
legitimately handle passthrough columns must filter them out before calling.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_MAP_PATH = Path(__file__).resolve().parent / "map.json"

_TF_PREFIX = re.compile(r"^(\d+[smhd])_(.+)$")
_POSITION = re.compile(r"_(long|short)$")
_AND = "_and_"
# Generic transform suffix on tick feature columns: a rolling-window stat applied
# to a base signal, e.g. relative_spread_roll30_mean. The window/stat is a generic
# transform (not IP), so it is PRESERVED like a TF prefix; only the base signal is
# coded. <tf>_<base>_roll<w>_<mean|std>  ->  <tf>_<code>_roll<w>_<mean|std>.
_ROLL_SUFFIX = re.compile(r"(_roll\d+_(?:mean|std))$")
# Conjunction/disjunction separators between regime atoms. Capturing group so
# re.split keeps them, letting _map_regime map atoms and pass separators through.
# No regime atom contains the substring `_and_`/`_or_`, so this never false-splits.
_SEP_RE = re.compile(r"(_and_|_or_)")


class Codec:
    def __init__(self, map_path: Path | str = _MAP_PATH):
        data = json.loads(Path(map_path).read_text())
        self.regimes: dict[str, str] = data["regimes"]
        self.features: dict[str, str] = data["features"]
        self._regimes_rev = {v: k for k, v in self.regimes.items()}
        self._features_rev = {v: k for k, v in self.features.items()}
        # Bijection guard — duplicate codes would make decode ambiguous.
        if len(self._regimes_rev) != len(self.regimes):
            raise ValueError("regime map is not bijective (duplicate codes)")
        if len(self._features_rev) != len(self.features):
            raise ValueError("feature map is not bijective (duplicate codes)")

    # ---- atom-level ---------------------------------------------------------
    def encode_atom(self, name: str) -> str:
        try:
            return self.regimes[name]
        except KeyError:
            raise KeyError(f"unknown regime atom: {name!r}") from None

    def decode_atom(self, code: str) -> str:
        try:
            return self._regimes_rev[code]
        except KeyError:
            raise KeyError(f"unknown regime code: {code!r}") from None

    def encode_feature_base(self, name: str) -> str:
        try:
            return self.features[name]
        except KeyError:
            raise KeyError(f"unknown feature: {name!r}") from None

    def decode_feature_base(self, code: str) -> str:
        try:
            return self._features_rev[code]
        except KeyError:
            raise KeyError(f"unknown feature code: {code!r}") from None

    # ---- structure-preserving regime names ----------------------------------
    def _split_tf(self, token: str) -> tuple[str, str]:
        m = _TF_PREFIX.match(token)
        return (m.group(1) + "_", m.group(2)) if m else ("", token)

    def encode_regime(self, regime: str) -> str:
        return self._map_regime(regime, self.encode_atom)

    def decode_regime(self, regime: str) -> str:
        return self._map_regime(regime, self.decode_atom)

    def is_regime_code(self, token: str) -> bool:
        return token in self._regimes_rev

    def decode_regime_tolerant(self, regime: str) -> str:
        """Decode a regime that may already be in real-name form.

        Per atom token: decode if it is a known code, else leave as-is. Used at
        dc filter-mask entry during the rename rollout so the same method accepts
        both coded (new) and real-name (existing tests / in-flight callers) input.

        A token that is neither a valid code nor a valid real filter is passed
        through unchanged here and rejected downstream by _apply_filter_mask —
        which raises ``ValueError("Unknown filter name")`` ONLY when
        ``self._strict_filters`` is True (the default). With strict filters off,
        that path instead logs and returns an all-False mask, so an unknown token
        yields zero signal rather than an error. Keep strict filters on for the
        obfuscation entry points.
        """
        return self._map_regime(
            regime, lambda b: self.decode_atom(b) if self.is_regime_code(b) else b
        )

    def _map_regime(self, regime: str, fn) -> str:
        pos = ""
        m = _POSITION.search(regime)
        if m:
            pos = regime[m.start():]
            regime = regime[: m.start()]
        out = []
        # Split on both _and_ and _or_, preserving the separators (mixed grammars
        # like a_and_b_or_c reassemble correctly).
        for tok in _SEP_RE.split(regime):
            if tok in ("_and_", "_or_") or tok == "":
                out.append(tok)
            else:
                tf, base = self._split_tf(tok)
                out.append(tf + fn(base))
        return "".join(out) + pos

    def has_baseline(self, regime: str) -> bool:
        """True if any conjunct of a (positioned, TF-prefixed, _and_/_or_-joined)
        regime is the banned ``baseline``. Operates on REAL names; coded regimes
        always return False because `baseline` is never in the map. Single shared
        home for the baseline-ban predicate (see CLAUDE.md, 2026-06-18)."""
        body = _POSITION.sub("", regime)
        for tok in _SEP_RE.split(body):
            if tok in ("_and_", "_or_") or tok == "":
                continue
            _, base = self._split_tf(tok)
            if base == "baseline":
                return True
        return False

    # ---- structure-preserving feature names ---------------------------------
    def _split_feature(self, col: str) -> tuple[str, str, str]:
        """Split a feature column into (tf_prefix, base, roll_suffix), preserving
        the generic TF prefix and rolling-stat transform suffix; only `base` maps."""
        tf, rest = self._split_tf(col)
        m = _ROLL_SUFFIX.search(rest)
        if m:
            return tf, rest[: m.start()], m.group(1)
        return tf, rest, ""

    def encode_feature(self, col: str) -> str:
        tf, base, suf = self._split_feature(col)
        return tf + self.encode_feature_base(base) + suf

    def decode_feature(self, col: str) -> str:
        tf, base, suf = self._split_feature(col)
        return tf + self.decode_feature_base(base) + suf

    def encode_columns(self, columns) -> dict[str, str]:
        """Rename-map for a DataFrame: {real_col: coded_col} for KNOWN feature
        columns only (TF prefix preserved). Passthrough / target / metadata
        columns (not in the feature map) are omitted, so callers can
        ``df.rename(columns=codec.encode_columns(df.columns))`` safely.
        """
        out = {}
        for col in columns:
            tf, base, suf = self._split_feature(col)
            if base in self.features:
                out[col] = tf + self.features[base] + suf
        return out

    def decode_columns(self, columns) -> dict[str, str]:
        """Inverse of encode_columns — {coded_col: real_col} for known codes."""
        out = {}
        for col in columns:
            tf, base, suf = self._split_feature(col)
            if base in self._features_rev:
                out[col] = tf + self._features_rev[base] + suf
        return out

    def add_feature_aliases(self, obj):
        """Add coded-name ALIAS columns alongside the real feature columns so a
        predict can select EITHER coded (new weights) or real (old weights)
        feature_columns during the rename rollout — and the scaler's name-check
        (feature_names_in_) matches whichever namespace meta uses. Accepts a
        DataFrame (columns) or a Series (index, e.g. mjolnir's per-bar row).
        Returns a copy; real columns + non-feature columns are left untouched.
        """
        import pandas as pd
        is_series = isinstance(obj, pd.Series)
        names = obj.index if is_series else obj.columns
        mapping = self.encode_columns(names)
        if not mapping:
            return obj
        obj = obj.copy()
        have = set(obj.index if is_series else obj.columns)
        for real, coded in mapping.items():
            if coded not in have:
                obj[coded] = obj[real]
        return obj

    def decode_feature_columns(self, columns) -> list[str]:
        """Real-name list for a coded feature_columns list, ORDER PRESERVED.

        Known codes -> real; unknown / passthrough names pass through unchanged.
        Use at the LIVE / reproduction predict boundary to reconcile a coded
        meta.feature_columns against a freshly-computed REAL-named feature frame:
        select these columns from the real frame, then scale/predict (the scaler
        and model use values in this order, so names don't matter past selection).
        Do NOT use on the research path — there the frame is already coded.
        """
        m = self.decode_columns(columns)
        return [m.get(c, c) for c in columns]


_DEFAULT: Codec | None = None


def default() -> Codec:
    """Process-wide singleton bound to the bundled map.json."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Codec()
    return _DEFAULT
