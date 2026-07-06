"""StormBreakerResearch: MjolnirResearch extended with CONTEXT_BAR_FREQS.

Differences from MjolnirResearch:
- CONTEXT_BAR_FREQS replaces MULTI_TF_BARS.  Context bar paths are
  auto-derived from the signal BARS_DIR as
    {archive_base}/pred_stormbreaker.base.{tf}_1/bars/
- _apply_filter_mask() intercepts '{tf}_{filter}' atoms BEFORE the parent
  strips '_long'/'_short', preventing '1m_long_liq_spike' → '1m_liq_spike'.
  Splits compound names (_and_/_or_) and recurses through self.
- _named_filter() is a secondary fallback for direct tick-native calls.
- All other pipeline steps (engineer_features, verticalize, filter_signals,
  create) are unchanged and inherited from MjolnirResearch.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import pandas as pd

from mjolnir.core.research import MjolnirResearch
from stormbreaker.core.filters import apply_filter, is_tick_filter


def _obf():
    """Lazy accessor for the vendored obfuscation codec (see _obf/codec.py)."""
    from .._obf.codec import default
    return default()

logger = logging.getLogger(__name__)

# TF strings in longest-first order so "15s" is tried before "5s"
_KNOWN_TFS = sorted(["5s", "15s", "30s", "1m", "5m", "15m"], key=len, reverse=True)


class StormBreakerResearch(MjolnirResearch):
    """Research pipeline for Stormbreaker tick-native cross-TF ML.

    Subclasses MjolnirResearch.  The only constructor change is translating
    CONTEXT_BAR_FREQS → MULTI_TF_BARS + MULTI_TF_BARS_DIRS so that the
    parent's load() and engineer_features() can operate unchanged.

    Args:
        config:    Dictionary loaded from setting.json.
        home_root: Repository root directory.
    """

    # Bar frequencies + cross-TF context/signal pairs (moved out of the public
    # marvel generator so real regime names live only here).
    _ALL_TFS = ["5s", "15s", "30s", "1m", "5m", "15m"]
    _TF_PAIRS = [
        ("15s", "5s"), ("1m", "5s"), ("1m", "15s"), ("5m", "15s"),
        ("5m", "30s"), ("15m", "30s"), ("5m", "1m"), ("15m", "1m"),
    ]

    @classmethod
    def generate_regime_stack(cls, config: Optional[Dict] = None) -> list[dict]:
        """Coded [{regime, position}] for the Stormbreaker tick-native cross-TF
        stack (single-TF + the 3 cross-TF section families). Regime names are
        returned OBFUSCATED so the public generator never holds real names."""
        from stormbreaker.core.filters import (
            allowed_positions, _LONG_ONLY, _SHORT_ONLY, _BOTH,
        )
        longs, shorts, both = sorted(_LONG_ONLY), sorted(_SHORT_ONLY), sorted(_BOTH)
        all_f = longs + shorts + both
        combos: list[tuple] = []
        for tf in cls._ALL_TFS:                                  # 1. single-TF
            combos += [(f"{tf}_{f}",) for f in all_f]
        for ctx_tf, sig_tf in cls._TF_PAIRS:                     # 2. BOTH ctx + dir sig
            for cf in both:
                for sf in longs + shorts:
                    combos.append((f"{ctx_tf}_{cf}", f"{sig_tf}_{sf}"))
        for ctx_tf, sig_tf in cls._TF_PAIRS:                     # 3. same-side dir x dir
            for side in (longs, shorts):
                for cf in side:
                    for sf in side:
                        if cf != sf:
                            combos.append((f"{ctx_tf}_{cf}", f"{sig_tf}_{sf}"))
        for ctx_tf, sig_tf in cls._TF_PAIRS:                     # 4. dir ctx + BOTH sig
            for cf in longs + shorts:
                for sf in both:
                    combos.append((f"{ctx_tf}_{cf}", f"{sig_tf}_{sf}"))

        c = _obf()
        seen, out = set(), []
        for combo in combos:
            name = "_and_".join(combo)
            for pos in allowed_positions(name):
                key = (name, pos)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"regime": c.encode_regime(name), "position": pos})
        return out

    def __init__(self, config: Dict, home_root: str) -> None:
        config = self._resolve_context_bar_dirs(config)
        super().__init__(config, home_root)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_context_bar_dirs(config: Dict) -> Dict:
        """Translate CONTEXT_BAR_FREQS into MULTI_TF_BARS + MULTI_TF_BARS_DIRS.

        Derives context bar paths as:
            {archive_base}/pred_stormbreaker.base.{tf}_1/bars/

        where archive_base = dirname(dirname(BARS_DIR)).

        Raises:
            KeyError: if CONTEXT_BAR_FREQS is set but BARS_DIR is missing.
        """
        context_tfs = config.get("CONTEXT_BAR_FREQS", [])
        if not context_tfs:
            return config

        config = dict(config)  # don't mutate caller's dict
        config["MULTI_TF_BARS"] = context_tfs

        bars_dir = config.get("BARS_DIR")
        if not bars_dir:
            raise KeyError(
                "BARS_DIR is required in config when CONTEXT_BAR_FREQS is set"
            )

        # /mnt/.../stormbreaker/pred_stormbreaker.base.5s_1/bars
        # → /mnt/.../stormbreaker/
        archive_base = os.path.dirname(os.path.dirname(bars_dir.rstrip("/")))

        tf_dirs: Dict[str, str] = {}
        for tf in context_tfs:
            tf_dirs[tf] = os.path.join(
                archive_base, f"pred_stormbreaker.base.{tf}_1", "bars"
            )
        config["MULTI_TF_BARS_DIRS"] = tf_dirs
        logger.info(
            "Resolved CONTEXT_BAR_FREQS %s → archive_base=%s",
            context_tfs, archive_base,
        )
        return config

    @staticmethod
    def _parse_tf_filter(name: str) -> tuple[str, str]:
        """Split '{tf}_{filter}' → (tf, base_filter_name).

        Returns ('', name) if no known TF prefix is found.
        _KNOWN_TFS is sorted longest-first to avoid '5s' matching '15s'.
        """
        for tf in _KNOWN_TFS:
            prefix = tf + "_"
            if name.startswith(prefix):
                return tf, name[len(prefix):]
        return "", name

    def _get_tf_view(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """Return a sub-DataFrame with columns for *tf* renamed to bare names.

        E.g. if tf='15s', columns '15s_trade_imbalance' → 'trade_imbalance'.
        Returns df unchanged when tf matches BAR_FREQ (base TF, no prefix).
        Returns an empty DataFrame (same index) if the TF has no columns yet
        — apply_filter() will then fall back to all-True.
        """
        bar_freq = self.config.get("BAR_FREQ", "5s")
        if not tf or tf == bar_freq:
            return df

        prefix = tf + "_"
        mapping = {c: c[len(prefix):] for c in df.columns if c.startswith(prefix)}
        if not mapping:
            return pd.DataFrame(index=df.index)
        return df[list(mapping.keys())].rename(columns=mapping)

    # ------------------------------------------------------------------
    # Filter dispatch
    # ------------------------------------------------------------------

    def _apply_filter_mask(
        self,
        df: pd.DataFrame,
        filter_name,
        position: str,
    ) -> pd.Series:
        """Override to route tick-native filters BEFORE the parent strips _long/_short.

        The parent's _apply_filter_mask calls
            name = name.replace("_long", "").replace("_short", "")
        which would corrupt tick-native names like '1m_long_liq_spike' →
        '1m_liq_spike'.  We intercept tick-native atomic filters here before
        any stripping happens.

        Compound names (_and_ / _or_) are split here and each atom is routed
        through self._apply_filter_mask so the interception applies recursively.
        """
        # List filters — parent handles correctly (recurses via self)
        if isinstance(filter_name, list):
            return super()._apply_filter_mask(df, filter_name, position)

        name = str(filter_name).lower().strip()
        # Accept coded regimes: decode code->real before tick-filter routing
        # (TF prefixes preserved; real names pass through; recursion re-decodes).
        name = _obf().decode_regime_tolerant(name)

        # Compound: must split here so recursion passes through our override
        if "_and_" in name:
            parts = name.split("_and_")
            mask = None
            for p in parts:
                sub = self._apply_filter_mask(df, p.strip(), position)
                mask = sub if mask is None else (mask & sub)
            return mask if mask is not None else pd.Series(True, index=df.index)

        if "_or_" in name:
            parts = name.split("_or_")
            mask = None
            for p in parts:
                sub = self._apply_filter_mask(df, p.strip(), position)
                mask = sub if mask is None else (mask | sub)
            return mask if mask is not None else pd.Series(True, index=df.index)

        if df.empty:
            return pd.Series(dtype=bool)

        # Atomic filter — intercept tick-native BEFORE parent strips _long/_short
        tf, base_filter = self._parse_tf_filter(name)
        if is_tick_filter(base_filter):
            sub_df = self._get_tf_view(df, tf)
            return apply_filter(sub_df, base_filter)

        # Non-tick filter — delegate to parent (safe to strip _long/_short there)
        return super()._apply_filter_mask(df, filter_name, position)

    def _named_filter(
        self, df: pd.DataFrame, name: str, position: str
    ) -> pd.Series:
        """Secondary dispatch for tick-native filters that reach _named_filter.

        Handles the case where tick-native filters are passed without going
        through the _apply_filter_mask path (e.g. direct calls or tests).
        Unknown names fall through to MjolnirResearch._named_filter().
        """
        tf, base_filter = self._parse_tf_filter(name)
        if is_tick_filter(base_filter):
            sub_df = self._get_tf_view(df, tf)
            return apply_filter(sub_df, base_filter)
        return super()._named_filter(df, name, position)
