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
        NOT a silent fallback: a token that is neither a valid code nor a valid
        real filter still falls through to the existing strict raise in
        _apply_filter_mask (`unknown filter_name`). See marvel CLAUDE.md.
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
        parts = regime.split(_AND)
        out = []
        for part in parts:
            tf, base = self._split_tf(part)
            out.append(tf + fn(base))
        return _AND.join(out) + pos

    # ---- structure-preserving feature names ---------------------------------
    def encode_feature(self, col: str) -> str:
        tf, base = self._split_tf(col)
        return tf + self.encode_feature_base(base)

    def decode_feature(self, col: str) -> str:
        tf, base = self._split_tf(col)
        return tf + self.decode_feature_base(base)

    def encode_columns(self, columns) -> dict[str, str]:
        """Rename-map for a DataFrame: {real_col: coded_col} for KNOWN feature
        columns only (TF prefix preserved). Passthrough / target / metadata
        columns (not in the feature map) are omitted, so callers can
        ``df.rename(columns=codec.encode_columns(df.columns))`` safely.
        """
        out = {}
        for col in columns:
            tf, base = self._split_tf(col)
            if base in self.features:
                out[col] = tf + self.features[base]
        return out

    def decode_columns(self, columns) -> dict[str, str]:
        """Inverse of encode_columns — {coded_col: real_col} for known codes."""
        out = {}
        for col in columns:
            tf, base = self._split_tf(col)
            if base in self._features_rev:
                out[col] = tf + self._features_rev[base]
        return out


_DEFAULT: Codec | None = None


def default() -> Codec:
    """Process-wide singleton bound to the bundled map.json."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Codec()
    return _DEFAULT
