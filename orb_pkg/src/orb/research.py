"""Orb research: cross-timeframe feature alignment on top of Agamotto."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence
import gc
import logging
import re

import numpy as np
import pandas as pd

from agamotto import AgamottoResearch
from agamotto.features_scalefree import SCALE_FREE_FEATURES
from agamotto.ladder import compute_ladder_multiplier, compute_ladder_return, ladder_params
from agamotto.research import VOL_QUANTILE_FEATURES
from agamotto.utils import _symbol_to_native

logger = logging.getLogger(__name__)

# Features that are DERIVED (used as ML features after TF-prefixing).
# Raw OHLCV / MAs / returns are excluded from TF-prefixed ML features
# but included unprefixed from TARGET_TF for filter logic.
_DERIVED_FEATURES = [
    "price_range", "price_range_pct", "price_range_pct_q50",
    "open_close_diff", "open_close_pct",
    "high_open_pct", "low_open_pct",
    "ret_lag1", "ret_lag2", "ret_lag3",
    # TA-Lib momentum
    "rsi", "rsi_7", "rsi_28",
    "macd", "macdhist",
    "stoch_k", "stoch_d",
    "cci", "adx", "dx", "plus_di", "minus_di",
    "mom", "roc", "willr", "cmo", "trix", "ultosc",
    "stochrsi_k", "stochrsi_d",
    # TA-Lib volume
    "obv", "ad", "mfi", "bop",
    # TA-Lib volatility
    "atr", "natr", "parkinson_vol", "bb_upper", "bb_lower",
    # TA-Lib trend
    "sar",
    # Rolling stats
    "std", "skew", "kurt", "acf_lag1",
    # Volume features
    "vol_ratio", "quote_vol_ratio", "buy_pressure", "trade_intensity",
    "vol_ret_lag1", "vol_ret_lag2", "vol_ret_lag3",
]

# The seven scale-free level replacements (agamotto/features_scalefree.py,
# added 2026-08-06). orb ALREADY COMPUTES THEM: engineer_features() delegates
# per TF to AgamottoResearch (:260), whose engineer_features writes
# `{native}_sar_dist` … into that TF's wide frame, and _align_timeframes carries
# them through as `{tf}_{native}_sar_dist`. They were simply absent from
# _DERIVED_FEATURES, so verticalize()'s step-1 loop — which iterates the LIST,
# not the frame — dropped all seven from the vertical panel in silence. That is
# the same failure SHAPE as agamotto's pre-104d740 rename-map drop, and it is
# what left orb's 15m block at 40 features, a strict subset of agamotto's 47
# (gauntlet/orb_vs_agamotto_features_20260822.md §1a); those twins accounted for
# 1,794 of agamotto's exclusive top-5 picks.
#
# APPENDED FROM THE CANONICAL LIST, never re-typed here: a second hardcoded copy
# drifting from features_scalefree.SCALE_FREE_FEATURES is exactly the defect
# this replaces. The obfuscation map already carries codes for all seven
# (f101–f107) — the map is global and append-only, so nothing renumbers.
_DERIVED_FEATURES += list(SCALE_FREE_FEATURES)

# Raw / MA / return columns that filters need but ML should NOT use.
_RAW_COLUMNS = [
    "close", "open", "high", "low", "volume",
    "quote_volume", "number_of_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume",
    "mvg1", "mvg2", "mvg3",
    # Precomputed own-state filter atom (agamotto/research.py::engineer_features),
    # not a plain raw column, but same bucket: filter-only, excluded from ML,
    # same failure shape as the two incidents documented above this list if left
    # out — verticalize() iterates this list, not the engineered frame.
    "convergence_tight",
]

# FILTER-ONLY trailing vol-quantile CUTOFFS. agamotto's `high_vol_q80/q90/q95`
# atoms compare `price_range_pct` against these per-symbol cutoffs
# (agamotto/research_filters.py:102-104), so a regime naming one is unevaluable
# without them. orb ALREADY COMPUTES THEM: engineer_features() delegates per TF
# to AgamottoResearch, whose engineer_features writes `{native}_price_range_pct_q80`
# ... into that TF's wide frame. They were simply absent from BOTH carry lists, so
# _remap_for_tf's `filter_cols` (= _RAW_COLUMNS | _DERIVED_FEATURES) never exposed
# the bare alias and _apply_filter_mask raised "requires column
# 'price_range_pct_q80'" on 144 of agamotto's 177 regimes (243 of 299 stack rows).
# Same failure SHAPE as the seven scale-free twins above: computed, then dropped
# by a list that did not name them.
#
# THEY BELONG HERE, NOT IN _DERIVED_FEATURES. They GATE ENTRY and are not model
# inputs -- gauntlet/rolling_predict_returns.py:1358 excludes all three by name
# AND by TF-stripped/obfuscated alias, and its comment says so outright ("they
# GATE ENTRY, they are not model inputs ... Excluded HERE, not deleted in dc").
# _RAW_COLUMNS carries them unprefixed from TARGET_TF, which is the only
# timeframe agamotto's single-TF regimes read; _DERIVED_FEATURES would instead
# mint 12 TF-prefixed copies that every consumer then has to exclude again.
#
# APPENDED FROM THE CANONICAL LIST, never re-typed here: a second hardcoded copy
# drifting from agamotto.research.VOL_QUANTILE_FEATURES is exactly the defect
# this repairs. Order follows VOL_Q_LEVELS (asserted in agamotto's tests).
_RAW_COLUMNS += list(VOL_QUANTILE_FEATURES)

_RETURN_COLUMNS = [
    "return", "return_long", "return_short",
    "return_long_raw", "return_short_raw",
]

# ── The timeframe ladder ─────────────────────────────────────────────────────
#
# ORDERING CONTRACT: `TIMEFRAMES` is the ladder ordered FINEST FIRST, and RANK
# is counted from the COARSE end — rank 0 = coarsest = the CONTEXT leg, the
# last rank = finest = the DECISION leg. Every cross-TF regime template below
# names ranks, never literal timeframes, so the same table generates today's
# 15m/1h/4h/1d stack and a 1m/5m/15m/1h one.
#
# LEGACY LADDER. Every orb arm on disk states TIMEFRAMES explicitly and states
# exactly this list, but `generate_regime_stack()` is also called with NO config
# at all (marvel gauntlet/generate_orb_regimes.py:33), so the zero-arg call has
# to keep reproducing the shipped 332-row stack byte for byte.
# DEPRECATED: drop after 2026-12-01 — by then the marvel generator passes the
# arm's setting.json and this constant becomes a test fixture only. The chain
# ends in a raise inside `_resolve_ladder`, never in a second silent default.
_DEFAULT_TIMEFRAMES = ["15m", "1h", "4h", "1d"]

_TF_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_TF_RE = re.compile(r"^(\d+)([smhd])$")
# A leading TF prefix on a regime atom, e.g. "4h_" in "4h_rsi_oversold".
_TF_PREFIX_ON_ATOM = re.compile(r"^(\d+[smhd])_")


def _tf_seconds(tf: str) -> int:
    """Duration of a timeframe token in seconds. Raises on anything unparseable.

    The obfuscation codec's TF-prefix regex is `^(\\d+[smhd])_` (obfuscation/
    codec.py:28), so a TF token that does not match this shape would silently
    stop being recognised as a TF prefix at write time. Refuse it here instead.
    """
    m = _TF_RE.match(tf)
    if m is None:
        raise ValueError(
            f"unparseable timeframe {tf!r}: expected <int><s|m|h|d>, e.g. '15m'"
        )
    return int(m.group(1)) * _TF_UNIT_SECONDS[m.group(2)]


def _resolve_ladder(config: Optional[Dict[str, object]]) -> List[str]:
    """The ordered (finest-first) TF ladder for a config.

    No `config.get("TIMEFRAMES", <magic list>)`: a config that CARRIES the key
    must state a usable ladder or this raises. A config that does not carry it
    at all — and `config is None`, the zero-arg classmethod call — falls back to
    the legacy ladder, which is the documented back-compat path above.
    """
    if config is None or "TIMEFRAMES" not in config:
        ladder = list(_DEFAULT_TIMEFRAMES)
    else:
        ladder = list(config["TIMEFRAMES"])

    if not ladder:
        raise ValueError("TIMEFRAMES must be a non-empty list of timeframes")
    if len(set(ladder)) != len(ladder):
        raise ValueError(f"TIMEFRAMES has duplicate entries: {ladder}")

    seconds = [_tf_seconds(tf) for tf in ladder]
    if any(b <= a for a, b in zip(seconds, seconds[1:])):
        raise ValueError(
            f"TIMEFRAMES must be ordered finest-first by duration, got {ladder} "
            f"({seconds} seconds). Rank 0 (the context leg) is the LAST entry; "
            "a mis-ordered ladder would build every cross-TF regime backwards."
        )
    return ladder


def _tf_at_rank(ladder: Sequence[str], rank: int) -> str:
    """Rank 0 = coarsest (context); the last rank = finest (decision)."""
    if rank < 0 or rank >= len(ladder):
        raise IndexError(
            f"cross-TF regime template needs rank {rank} but the ladder "
            f"{list(ladder)} has only {len(ladder)} timeframes"
        )
    return ladder[len(ladder) - 1 - rank]


def _obf():
    """Lazy accessor for the vendored obfuscation codec (see _obf/codec.py)."""
    from ._obf.codec import default
    return default()


# ── Regime definitions (moved out of the public marvel generator so real names
#    live only in the obfuscated package). generate_regime_stack() returns coded,
#    structure-preserving regime names. `baseline` banned (see CLAUDE.md). ──────
_ORB_BASE_FILTERS = [
    "strong_trend", "ma_momentum", "above_all_mas",
    "low_vol", "high_vol", "strong_candle", "near_ma",
    "rsi_oversold", "rsi_overbought",
    "macd_bullish", "macd_bearish",
    "stoch_bullish", "cci_reversal", "adx_trend",
    "bb_rebound", "mom_positive",
    "low_volume", "high_volume", "vol_breakout",
    "buy_pressure", "sar_aligned",
    "mfi_oversold", "mfi_overbought",
    "bop_bullish", "bop_bearish",
    "roc_positive", "roc_negative",
]
_ORB_SAME_TF_VOL_DIR = [
    ("vol_breakout", "mfi_oversold"), ("vol_breakout", "mfi_overbought"),
    ("high_volume", "bop_bullish"), ("high_volume", "bop_bearish"),
    ("low_volume", "roc_positive"), ("low_volume", "roc_negative"),
]

# ── Cross-TF regime templates, keyed by RANK rather than by timeframe ────────
#
# Until 2026-09-24 sections 2, 4 and 5 of generate_regime_stack() were ~137
# hand-written tuples naming '1d_', '4h_' and '1h_' as literals, so the ONLY
# ladder they could ever describe was 15m/1h/4h/1d. The tuples are unchanged in
# content — every one of them is a (context filter at rank R, directional
# filter at rank R') pair, and the tables below are that pair table with the
# timeframe replaced by its rank. `test_regime_stack_default_ladder.py` pins
# the pre-refactor output so the rewrite cannot quietly change the stack.
#
# Rank 0 = coarsest = CONTEXT; larger rank = finer = DIRECTION (see the ladder
# contract above). The legacy ladder maps rank 0 -> 1d, 1 -> 4h, 2 -> 1h,
# 3 -> 15m. NOTE rank 3 (the finest TF) appears in NO cross-TF template: on the
# legacy ladder 15m is a cross-TF leg nowhere, only a single-TF (§1) and
# same-TF (§3) one. That asymmetry is preserved, not "tidied".
#
# The directional-filter groups are strict prefixes of one another, and the
# differences between them are REAL, not oversights — e.g. `high_vol` context
# crosses 4 directional filters at rank 1 but 6 at rank 2, and never mfi.
_DIR_MACD_RSI = [
    "macd_bullish", "macd_bearish", "rsi_oversold", "rsi_overbought",
]
_DIR_MACD_RSI_BOP = _DIR_MACD_RSI + ["bop_bullish", "bop_bearish"]
_DIR_MACD_RSI_BOP_MFI = _DIR_MACD_RSI_BOP + ["mfi_oversold", "mfi_overbought"]
_DIR_MACD_RSI_BOP_MFI_STOCH = _DIR_MACD_RSI_BOP_MFI + ["stoch_bullish"]
_DIR_MACD = ["macd_bullish", "macd_bearish"]
_DIR_RSI = ["rsi_oversold", "rsi_overbought"]
_DIR_BOP = ["bop_bullish", "bop_bearish"]
_DIR_MFI = ["mfi_oversold", "mfi_overbought"]

# §2 — trend/breakout context + directional signal.
# (context_rank, [context filters], signal_rank, [signal filters])
_CROSS_TF_COMBO_TEMPLATES = [
    (0, ["strong_trend", "ma_momentum", "above_all_mas"], 2, _DIR_MACD_RSI),
    (1, ["strong_trend", "adx_trend"], 2, _DIR_MACD_RSI),
    (1, ["macd_bullish"], 2, _DIR_RSI),
    (0, ["strong_trend"], 1, _DIR_MACD_RSI),
    (0, ["ma_momentum"], 1, _DIR_MACD),
    (0, ["vol_breakout"], 1, _DIR_MACD_RSI),
    (0, ["vol_breakout"], 2, _DIR_MACD_RSI),
    (1, ["vol_breakout"], 2, _DIR_MACD_RSI),
]
# §2b — the two three-leg regimes: coarse context, mid breakout, fine signal.
# (rank0, [filters], rank1, [filters], rank2, [filters])
_CROSS_TF_TRIPLE_TEMPLATES = [
    (0, ["strong_trend", "vol_breakout"], 1, ["vol_breakout"], 2, _DIR_MACD),
]
# §4 — volume / volatility context + directional signal.
_CROSS_TF_VOL_DIR_TEMPLATES = [
    (0, ["low_volume"], 1, _DIR_MACD_RSI_BOP),
    (0, ["low_volume"], 2, _DIR_MACD_RSI_BOP_MFI),
    (0, ["high_volume"], 1, _DIR_MACD_RSI_BOP),
    (0, ["high_volume"], 2, _DIR_MACD_RSI_BOP_MFI),
    (1, ["low_volume"], 2, _DIR_MACD_RSI_BOP_MFI_STOCH),
    (1, ["high_volume"], 2, _DIR_MACD_RSI_BOP_MFI),
    (0, ["low_vol"], 1, _DIR_MACD_RSI_BOP),
    (0, ["low_vol"], 2, _DIR_MACD_RSI_BOP_MFI_STOCH),
    (0, ["high_vol"], 1, _DIR_MACD_RSI),
    (0, ["high_vol"], 2, _DIR_MACD_RSI_BOP),
    (1, ["low_vol"], 2, _DIR_MACD_RSI_BOP_MFI_STOCH),
    (1, ["high_vol"], 2, _DIR_MACD_RSI_BOP),
]
# §5 — TA-lab context + directional signal.
_CROSS_TF_TALAB_TEMPLATES = [
    (0, ["strong_trend"], 2, _DIR_BOP),
    (0, ["vol_breakout"], 2, _DIR_MFI),
    (1, ["vol_breakout"], 2, _DIR_BOP),
    (1, ["vol_breakout"], 2, _DIR_MFI),
]


def _expand_pair_templates(ladder: Sequence[str], templates) -> list[tuple]:
    """(ctx_rank, ctx_filters, sig_rank, sig_filters) -> concrete 2-leg tuples.

    Order is context-filter-major, signal-filter-minor, templates in listed
    order — the order the hand-written tables used, which the dedup in
    generate_regime_stack() then makes observable in the emitted stack.
    """
    out: list[tuple] = []
    for ctx_rank, ctx_filters, sig_rank, sig_filters in templates:
        if ctx_rank >= sig_rank:
            raise ValueError(
                f"cross-TF template has context rank {ctx_rank} at or below "
                f"signal rank {sig_rank}: the context leg must be COARSER "
                "(lower rank) than the directional leg"
            )
        ctx_tf = _tf_at_rank(ladder, ctx_rank)
        sig_tf = _tf_at_rank(ladder, sig_rank)
        for cf in ctx_filters:
            for sf in sig_filters:
                out.append((f"{ctx_tf}_{cf}", f"{sig_tf}_{sf}"))
    return out


def _expand_triple_templates(ladder: Sequence[str], templates) -> list[tuple]:
    """(r0, f0, r1, f1, r2, f2) -> concrete 3-leg tuples, coarsest leg first."""
    out: list[tuple] = []
    for r0, f0, r1, f1, r2, f2 in templates:
        if not (r0 < r1 < r2):
            raise ValueError(
                f"three-leg template ranks must be strictly coarse-to-fine, "
                f"got ({r0}, {r1}, {r2})"
            )
        tf0, tf1, tf2 = (_tf_at_rank(ladder, r) for r in (r0, r1, r2))
        for a in f0:
            for b in f1:
                for c in f2:
                    out.append((f"{tf0}_{a}", f"{tf1}_{b}", f"{tf2}_{c}"))
    return out


def _orb_has_baseline(regime_name: str) -> bool:
    """True if any conjunct of a regime is the banned baseline.

    Delegates to the single shared, suffix-/TF-/_or_-aware codec predicate so the
    baseline-ban logic has one home (see CLAUDE.md, 2026-06-18)."""
    return _obf().has_baseline(regime_name)


class OrbResearch(AgamottoResearch):
    """Cross-timeframe research: merges 4 TF features into one wide matrix."""

    @classmethod
    def generate_regime_stack(cls, config: Optional[Dict[str, object]] = None) -> list[dict]:
        """Coded [{regime, position}] for all Orb cross-TF regimes.

        Builds regimes internally with real atom names (reusing allowed_positions
        for directionality), drops any baseline conjunct, dedups, then returns
        OBFUSCATED, structure-preserving regime names so the public marvel
        generator never handles real names.

        `config` supplies the TF ladder through TIMEFRAMES. Omitted (the
        zero-arg call marvel's generate_orb_regimes.py still makes) it is the
        legacy 15m/1h/4h/1d ladder and the output is byte-identical to the
        shipped 332-row stack — pinned by
        orb_pkg/tests/test_regime_stack_default_ladder.py.
        """
        ladder = _resolve_ladder(config)

        def allowed(filters):
            return cls.allowed_positions("_and_".join(filters))

        regimes: list[dict] = []

        # 1. Single-TF filters on cross-TF data
        for tf in ladder:
            for filt in _ORB_BASE_FILTERS:
                for pos in allowed((f"{tf}_{filt}",)):
                    regimes.append({"regime": f"{tf}_{filt}", "position": pos})

        # 2. Cross-TF compounds (>=1 unidirectional leg)
        cross_tf_combos = (
            _expand_pair_templates(ladder, _CROSS_TF_COMBO_TEMPLATES)
            + _expand_triple_templates(ladder, _CROSS_TF_TRIPLE_TEMPLATES)
        )
        # 3. Same-TF volume x directional cross products
        same_tf = [(f"{tf}_{v}", f"{tf}_{d}") for tf in ladder
                   for (v, d) in _ORB_SAME_TF_VOL_DIR]
        # 4. Cross-TF vol/volatility context + directional signal
        cross_tf_vol_directional = _expand_pair_templates(
            ladder, _CROSS_TF_VOL_DIR_TEMPLATES)
        # 5. Cross-TF TA-lab combos
        cross_tf_talab = _expand_pair_templates(ladder, _CROSS_TF_TALAB_TEMPLATES)

        for combo in cross_tf_combos + same_tf + cross_tf_vol_directional + cross_tf_talab:
            name = "_and_".join(combo)
            for pos in allowed(combo):
                regimes.append({"regime": name, "position": pos})

        # Dedup + enforce no-baseline, then obfuscate
        c = _obf()
        seen, out = set(), []
        for r in regimes:
            if _orb_has_baseline(r["regime"]):
                continue
            key = (r["regime"], r["position"])
            if key in seen:
                continue
            seen.add(key)
            out.append({"regime": c.encode_regime(r["regime"]), "position": r["position"]})
        return out

    def __init__(self, config: Dict[str, object], home_root: str) -> None:
        super().__init__(config, home_root)
        self.timeframes: List[str] = _resolve_ladder(config)
        # BASE_TF / TARGET_TF keep their legacy reads ONLY while the ladder is
        # the legacy one: "15m"/"1h" are meaningless on any other ladder, and
        # silently picking a timeframe the arm never named is the failure this
        # parametrization exists to prevent. Every orb arm on disk states both
        # keys, so nothing deployed reaches either branch's fallback.
        if self.timeframes != _DEFAULT_TIMEFRAMES:
            missing = [k for k in ("BASE_TF", "TARGET_TF") if k not in config]
            if missing:
                raise KeyError(
                    f"TIMEFRAMES={self.timeframes} is not the legacy ladder "
                    f"{_DEFAULT_TIMEFRAMES}, so {missing} must be stated "
                    "explicitly in setting.json (no implicit '15m'/'1h')."
                )
        # DEPRECATED: drop after 2026-12-01 together with _DEFAULT_TIMEFRAMES.
        # NOT validated against `self.timeframes`: a TARGET_TF outside the
        # ladder is a supported shape, exercised by
        # orb_pkg/tests/test_orb_verticalize_dedup.py:164 (step 1 then skips
        # nothing and step 3b finds nothing).
        self.base_tf: str = config.get("BASE_TF", "15m")
        self.target_tf: str = config.get("TARGET_TF", "1h")
        self._tf_instances: dict[str, AgamottoResearch] = {}

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load data for each timeframe serially.

        WHY: the prior ThreadPoolExecutor(max_workers=len(self.timeframes))
        path held 4 multi-symbol kline frames + all of TA-Lib's intermediate
        Series allocations alive simultaneously, peaking at ~65 GB RSS and
        OOM-killing on 29 symbols × 4 TFs × 3.5 years (2026-05-22 smoke).
        Serializing the loop + gc between TFs cuts that 4x to ~0.4 GB peak.
        Per-symbol streaming was tried (2026-05-23 smoke3) but proved much
        slower for ORB-scale data — see commit log for benchmark.
        """
        for tf in self.timeframes:
            tf_config = {**self.config, "TIME_UNIT": tf}
            inst = AgamottoResearch(tf_config, self.home_root)
            inst.load()
            self._tf_instances[tf] = inst
            gc.collect()
        # Parent compatibility: self.raw = base-TF raw data
        self.raw = self._tf_instances[self.base_tf].raw

    def engineer_features(self) -> None:
        """Engineer features for each TF serially, then align. See load() for rationale."""
        for tf, inst in list(self._tf_instances.items()):
            inst.engineer_features()
            # Free raw OHLCV — engineer_features keeps a reference inside .features
            # via the leading copy() but the original .raw is no longer needed.
            inst.raw = None
            gc.collect()
        self._align_timeframes()

    def verticalize(self) -> None:
        """Build vertical features with TF-prefixed derived columns."""
        if self.features is None:
            raise RuntimeError(
                "Call engineer_features() before verticalize().")

        symbols = self.config["SYMBOLS"]
        frames = []

        for sym in symbols:
            native = _symbol_to_native(sym)
            if native is None:
                continue

            sym_cols: dict[str, pd.Series] = {}

            # 1. TF-prefixed DERIVED features (ML features).
            #    TARGET_TF IS SKIPPED — step 3b emits those same source columns
            #    unprefixed, and emitting both put every base-TF feature in
            #    front of the model TWICE (fixed 2026-08-22).
            #
            #    `15m_rsi` and bare `rsi` were the SAME series: this loop reads
            #    `{tf}_{native}_{feat}` for every tf in self.timeframes — which
            #    includes TARGET_TF — and step 3b reads
            #    `{target_tf}_{native}_{feat}`. Verified on the built panel,
            #    `15m_rsi.equals(rsi)` is True. select_feature_columns excludes
            #    step 3's bare raw/MA copies BY NAME (close, mvg1, ...) but has
            #    no rule for bare DERIVED names — agamotto's own model features
            #    are exactly those names — so only the derived pair reached the
            #    model. That asymmetry was the defect. The |IC| ranker puts each
            #    pair adjacently, so the shipped TOPN_ICS = 16 arm spent its 16
            #    slots on 9 DISTINCT features, one of them cross-TF (measured on
            #    15m_r069_long / window_2026_07_31; census in marvel
            #    gauntlet/orb_vs_agamotto_features_20260822.md §1b).
            #
            #    WHY THE UNPREFIXED COPY IS THE ONE KEPT, not the prefixed one:
            #    it is the only choice that leaves every FILTER bit-identical.
            #    _remap_tf_columns overwrites a bare name only when it finds the
            #    prefixed column, so with the TARGET_TF block gone an atomic
            #    `15m_<filter>` falls through to the bare column — which holds
            #    exactly the TARGET_TF values it would have remapped. Other TFs
            #    still have their prefixed blocks and remap as before. It also
            #    matches how stormbreaker names its panel (base TF bare, context
            #    TFs prefixed — core/research.py _get_tf_view).
            #
            #    Dropping the BARE copy instead would have required routing
            #    unprefixed atoms through a TARGET_TF remap, and that breaks:
            #    research_filters.apply_filter_mask splits `_and_` by calling
            #    ITSELF (:225), never OrbResearch._apply_filter_mask, then
            #    strips the TF prefix with a regex (:240) and reads BARE
            #    columns. 160 of the 332 regimes in the shipped stack are
            #    compound. See the note on _apply_filter_mask below.
            for tf in self.timeframes:
                if tf == self.target_tf:
                    continue
                for feat in _DERIVED_FEATURES:
                    src_col = f"{tf}_{native}_{feat}"
                    dst_col = f"{tf}_{feat}"
                    if src_col in self.features.columns:
                        sym_cols[dst_col] = self.features[src_col]

            # 2. TF-prefixed raw/MA columns (for cross-TF filter logic,
            #    excluded by select_feature_columns)
            for tf in self.timeframes:
                for raw in _RAW_COLUMNS:
                    src_col = f"{tf}_{native}_{raw}"
                    dst_col = f"{tf}_{raw}"
                    if src_col in self.features.columns:
                        sym_cols[dst_col] = self.features[src_col]

            # 3. Unprefixed raw/MA from TARGET_TF (for filter logic, excluded)
            for raw in _RAW_COLUMNS:
                src_col = f"{self.target_tf}_{native}_{raw}"
                if src_col in self.features.columns:
                    sym_cols[raw] = self.features[src_col]

            # 3b. Unprefixed derived features from TARGET_TF. THE ONLY COPY of
            #     the TARGET_TF derived block since 2026-08-22 (step 1 skips
            #     that TF) — it serves both the filter logic and the model.
            #     Needed unprefixed so atomic filters like strong_candle
            #     (open_close_pct), low_vol (price_range_pct) resolve without a
            #     TF prefix, and so the `_and_` path in research_filters, which
            #     strips TF prefixes and reads bare columns, keeps working.
            for feat in _DERIVED_FEATURES:
                src_col = f"{self.target_tf}_{native}_{feat}"
                if src_col in self.features.columns:
                    sym_cols[feat] = self.features[src_col]

            # Tripwire for the defect steps 1 and 3b hid (mirrors the agamotto
            # one added in 104d740). Both loops iterate _DERIVED_FEATURES, NOT
            # the frame, so a feature the per-TF AgamottoResearch computed but
            # this list omits is dropped from the panel with nothing in the
            # logs — which is how orb lost all seven scale-free twins between
            # 2026-08-06 and 2026-08-22. Scoped to SCALE_FREE_FEATURES because
            # plenty of other `{tf}_{native}_*` columns are dropped here
            # legitimately (raw OHLCV, TA intermediates, the
            # price_range_pct_q80/q90/q95 filter-only cutoffs). No silent
            # fallback: a twin present in the frame but missing from the panel
            # raises. A TA-Lib failure upstream leaves the source column
            # ABSENT, so a degraded run does not trip this.
            for tf in self.timeframes:
                # TARGET_TF lands unprefixed (step 3b), every other TF prefixed
                # (step 1) — one name each, never both.
                _dropped_sf = [
                    f"{tf}_{native}_{n}" for n in SCALE_FREE_FEATURES
                    if f"{tf}_{native}_{n}" in self.features.columns
                    and (n if tf == self.target_tf else f"{tf}_{n}") not in sym_cols
                ]
                if _dropped_sf:
                    raise KeyError(
                        f"verticalize: {sym} {tf} — engineer_features produced "
                        f"{_dropped_sf} but _DERIVED_FEATURES does not carry "
                        f"them, so they would be dropped from the vertical "
                        f"panel without warning. Add them to _DERIVED_FEATURES."
                    )

            # 4. Returns
            if self.target_tf == self.base_tf:
                # Same TF: pre-computed returns from base TF features
                for ret in _RETURN_COLUMNS:
                    src_col = f"{self.target_tf}_{native}_{ret}"
                    if src_col in self.features.columns:
                        sym_cols[ret] = self.features[src_col]
            else:
                # Cross-TF: return = exit_close / entry_close - 1
                # exit_close  — next target-TF boundary close (same for all base-TF slots)
                # entry_close — current base-TF close (differs per slot → returns differ)
                exit_col = f"{self.target_tf}_{native}_exit_close"
                entry_col = f"{self.base_tf}_{native}_close"
                exit_low_col = f"{self.target_tf}_{native}_exit_low"
                exit_high_col = f"{self.target_tf}_{native}_exit_high"
                if exit_col in self.features.columns and entry_col in self.features.columns:
                    # FEE is required — no magic-number default (CLAUDE.md; the
                    # FEE bug 2026-04-27). AgamottoResearch.engineer_features
                    # already indexes config["FEE"] and orb delegates to it per
                    # TF, so a config reaching here without FEE was already dead.
                    fee = float(self.config["FEE"]) / 10000.0
                    raw = self.features[exit_col] / self.features[entry_col] - 1
                    sym_cols["return"] = raw
                    if (exit_low_col in self.features.columns
                            and exit_high_col in self.features.columns):
                        # Per-leg ladder sizing on the CROSS-TF target
                        # (2026-08-07). This block kept a private copy of the
                        # maths — `min(long_layers, short_layers)`, a hardcoded
                        # 0.0001 step that silently matched only LADDER_BPS == 1.0,
                        # and one shared `LADDER` read through the banned
                        # `get(K, X) or Y` — while the same-TF path above already
                        # inherited the fixed version from AgamottoResearch. Two
                        # copies of the target maths is exactly how the engines
                        # drifted apart; both now route through agamotto/ladder.py.
                        #
                        # exit_low/exit_high are already the exit bar's values, so
                        # unlike the same-TF path they take no shift(-1).
                        ladder_long, ladder_short, step_bps = ladder_params(self.config)
                        close_safe = self.features[entry_col].replace(0, np.nan)
                        exit_low = self.features[exit_low_col]
                        exit_high = self.features[exit_high_col]

                        size_long = compute_ladder_multiplier(
                            close_safe, exit_low, ladder_long, step_bps)
                        # Mirror the short's adverse (upward) move about the entry
                        # close so the same downward-measuring helper serves both.
                        size_short = compute_ladder_multiplier(
                            close_safe, 2.0 * close_safe - exit_high,
                            ladder_short, step_bps)

                        fee_cost = fee * 2.0
                        # SIGN (changed 2026-08-07). `return_short` was negated
                        # here while `return_short_raw` two lines down was not —
                        # the two short columns of the same block disagreed, so
                        # one of them was necessarily wrong. Agamotto and orb's
                        # own same-TF path both keep the short target in
                        # forward-return space (un-negated); that is what a
                        # NEGATIVE threshold under `y_pred < thresh` expects
                        # (CLAUDE.md). Negated, a good short scored HIGH and the
                        # leg selected backwards. Nothing deployed moves: all five
                        # orb settings are BASE_TF == TARGET_TF and take the
                        # same-TF path, so this branch has never run in anger.
                        # Per-rung entry pricing (2026-09-10), the same helper
                        # agamotto's two same-TF copies call — not `raw * rungs`.
                        long_raw = compute_ladder_return(raw, size_long, step_bps, "long")
                        short_raw = compute_ladder_return(raw, size_short, step_bps, "short")
                        sym_cols["return_long"] = long_raw - fee_cost * size_long
                        sym_cols["return_short"] = short_raw + fee_cost * size_short
                        sym_cols["return_long_raw"] = long_raw
                        sym_cols["return_short_raw"] = short_raw
                    else:
                        # No exit low/high aligned -> no ladder, size 1 per leg.
                        # Same un-negated short convention as above.
                        sym_cols["return_long"] = raw - 2 * fee
                        sym_cols["return_short"] = raw + 2 * fee
                        sym_cols["return_long_raw"] = raw.copy()
                        sym_cols["return_short_raw"] = raw.copy()

            # 5. Metadata
            sym_cols["year"] = self.features["year"]
            sym_cols["month"] = self.features["month"]
            sym_cols["symbol"] = sym
            sym_cols["timestamp"] = self.features.index

            subset = pd.DataFrame(sym_cols).reset_index(drop=True)
            frames.append(subset)

        if frames:
            self.vertical_features = pd.concat(
                frames, axis=0, ignore_index=True)
        else:
            self.vertical_features = pd.DataFrame()

    def _apply_filter_mask(
        self,
        df: pd.DataFrame,
        filter_name: str | list,
        position: str,
    ) -> pd.Series:
        """Override: route TF-prefixed filters through column remapping."""
        # List → delegate to super, which splits and re-enters HERE per item.
        if isinstance(filter_name, list):
            return super()._apply_filter_mask(df, filter_name, position)

        if not isinstance(filter_name, str):
            return super()._apply_filter_mask(df, filter_name, position)

        # Compound _and_ / _or_ → delegate to super, which does the decode and
        # the split in ONE place and hands each leg back to this method via the
        # `sub_filter_fn` hook, so the leg gets its own timeframe's columns.
        #
        # FIXED 2026-08-23. Until then the recursion did NOT come back here:
        # research_filters.apply_filter_mask split `_and_`/`_or_` by calling
        # ITSELF, so control skipped this override and reached the
        # `re.sub(r'^(?:15m|1h|4h|1d)_', ...)` strip, which discards the prefix
        # and reads the BARE (TARGET_TF) columns. Every leg of a compound was
        # therefore evaluated on TARGET_TF whatever TF its name said. Measured
        # on a 2,000-row 15m+1h panel: `15m_macd_bullish` 985 rows,
        # `1h_macd_bullish` 956, true conjunction 505 — and
        # `15m_macd_bullish_and_1h_macd_bullish` selected 985, the 15m leg
        # alone. 160 of the 332 regimes in the shipped pred_orb.base.15m_1
        # stack are compound and 136 of those span more than one TF, so every
        # orb research artefact built before this date measures a different
        # regime than its name says.
        #
        # Splitting FIRST (before the TF-prefix loop below) is load-bearing:
        # `15m_a_and_1h_b` starts with `15m_`, so the loop would otherwise
        # remap 15m for the whole compound and leave the 1h leg to be evaluated
        # on a frame that has already been overwritten.
        if "_and_" in filter_name or "_or_" in filter_name:
            return super()._apply_filter_mask(df, filter_name, position)

        # Check for a TF prefix on the (atomic) filter name.
        #
        # ANY TF-shaped prefix, not the hardcoded ("15m_", "1h_", "4h_", "1d_")
        # this matched until 2026-09-24 and NOT `self.timeframes` either. Both
        # of those are wrong, in opposite directions:
        #   * a hardcoded tuple leaves every atom of a 1m/5m/15m/1h arm looking
        #     unprefixed, so it falls through to the bare TARGET_TF columns and
        #     `1m_rsi_oversold` silently reads the target timeframe;
        #   * restricting it to THIS arm's ladder reintroduces the second route
        #     to that same wrong answer — `4h_rsi_oversold` on a 15m+1h panel
        #     would fall through to the bare columns instead of being remapped
        #     to a timeframe that has none and raising in `_require_col`
        #     (orb_pkg/tests/test_cross_tf_compound_dispatch.py:256).
        # Matching the shape and letting `_remap_tf_columns` + `_require_col`
        # adjudicate keeps the loud failure and works on any ladder. Same
        # `^(\d+[smhd])_` shape as obfuscation/codec.py:28.
        _tf_m = _TF_PREFIX_ON_ATOM.match(filter_name)
        if _tf_m is not None:
            tf = _tf_m.group(1)
            base_filter = filter_name[len(tf) + 1:]
            remapped = self._remap_tf_columns(df, tf)
            return super()._apply_filter_mask(
                remapped, base_filter, position)

        # No TF prefix → uses unprefixed TARGET_TF columns directly
        return super()._apply_filter_mask(df, filter_name, position)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _align_timeframes(self) -> None:
        """Forward-fill higher TFs onto the base-TF index."""
        base_inst = self._tf_instances[self.base_tf]
        base_features = base_inst.features
        base_idx = base_features.index

        aligned = pd.DataFrame(index=base_idx)
        meta_cols = {"year", "month"}

        for tf in self.timeframes:
            inst = self._tf_instances[tf]
            tf_features = inst.features

            # Drop metadata columns — added once at the end from base TF
            feature_cols = [c for c in tf_features.columns
                            if c not in meta_cols]
            tf_subset = tf_features[feature_cols]

            # Prefix all columns with TF
            renamed = tf_subset.rename(
                columns={c: f"{tf}_{c}" for c in tf_subset.columns})

            if tf == self.base_tf:
                # Base TF aligns directly (same index)
                aligned = aligned.join(renamed, how="left")
            else:
                close_ts_col = f"{tf}_close_timestamp"
                if close_ts_col not in renamed.columns:
                    logger.warning(
                        f"{close_ts_col} missing — falling back to shift(1)+ffill "
                        f"for {tf}. Run a fresh pipeline to regenerate features."
                    )
                    reindexed = renamed.shift(1).reindex(base_idx, method="ffill")
                    aligned = aligned.join(reindexed, how="left")
                    continue

                # Two-path alignment for higher TFs:
                #
                # FEATURE columns → causal (closed bar only):
                #   Use the most recent bar whose close_timestamp <= T.
                #   Guarantees no lookahead in model inputs.
                #
                # RETURN columns → SKIPPED here.
                #   Cross-TF returns are computed fresh in verticalize() as:
                #   return = exit_close / base_close - 1
                #   where exit_close is the TARGET_TF close aligned by open_time
                #   and base_close is the BASE_TF close at decision time T.
                #   This gives per-slot returns: at T=01:15 entry=close[01:15],
                #   at T=01:30 entry=close[01:30] — same exit, different entries.
                #
                # EXIT_CLOSE (TARGET_TF only) → future exit price:
                #   Use the bar whose open_time <= T. All BASE_TF slots within
                #   one TARGET_TF bar share the same exit_close (e.g. 01:00,
                #   01:15, 01:30, 01:45 all exit at close[02:00]).

                # Skip return columns — only align feature columns
                tf_ret_suffixes = tuple(f"_{r}" for r in _RETURN_COLUMNS
                                        if r != "return_dip" and r != "return_rip")
                tf_ret_suffixes += ("_return_dip", "_return_rip")
                all_cols = list(renamed.columns)
                ret_col_names = [
                    c for c in all_cols
                    if any(c.endswith(s) for s in tf_ret_suffixes)
                ]
                non_ret_col_names = [c for c in all_cols if c not in ret_col_names]

                right_base = renamed.reset_index(names="open_time")
                left = pd.DataFrame({"dt": base_idx})

                # Normalize datetime precision so merge_asof keys have matching
                # dtypes (pandas ≥2.0 distinguishes ms / us / ns).
                left["dt"] = left["dt"].astype("datetime64[us]")
                right_base["open_time"] = right_base["open_time"].astype("datetime64[us]")
                for _ts_col in [c for c in right_base.columns
                                if c.endswith("_close_timestamp")]:
                    right_base[_ts_col] = right_base[_ts_col].astype("datetime64[us]")

                # Feature alignment: backward merge on close_timestamp
                right_feat = right_base[["open_time"] + non_ret_col_names].sort_values(close_ts_col)
                merged_feat = pd.merge_asof(
                    left,
                    right_feat,
                    left_on="dt",
                    right_on=close_ts_col,
                    direction="backward",
                )
                result = merged_feat[non_ret_col_names].copy()
                result.index = base_idx

                # Exit-close alignment (TARGET_TF only): backward merge on open_time.
                # Stores the future exit price per symbol so verticalize() can
                # compute return = exit_close / base_close - 1 per BASE_TF slot.
                if tf == self.target_tf:
                    close_cols = [
                        c for c in non_ret_col_names
                        if c.endswith("_close") and not c.endswith("_close_timestamp")
                    ]
                    if close_cols:
                        right_exit = right_base[["open_time"] + close_cols].sort_values("open_time")
                        merged_exit = pd.merge_asof(
                            left,
                            right_exit,
                            left_on="dt",
                            right_on="open_time",
                            direction="backward",
                        )
                        for col in close_cols:
                            exit_col = col[:-len("_close")] + "_exit_close"
                            result[exit_col] = merged_exit[col].values

                    # Also align low/high for the exit bar so verticalize() can
                    # compute the ladder multiplier on cross-TF returns.
                    lh_cols = [
                        c for c in non_ret_col_names
                        if c.endswith("_low") or c.endswith("_high")
                    ]
                    if lh_cols:
                        right_exit_lh = right_base[["open_time"] + lh_cols].sort_values("open_time")
                        merged_exit_lh = pd.merge_asof(
                            left,
                            right_exit_lh,
                            left_on="dt",
                            right_on="open_time",
                            direction="backward",
                        )
                        for col in lh_cols:
                            if col.endswith("_low"):
                                exit_lh_col = col[:-len("_low")] + "_exit_low"
                            else:
                                exit_lh_col = col[:-len("_high")] + "_exit_high"
                            result[exit_lh_col] = merged_exit_lh[col].values

                aligned = aligned.join(result, how="left")

        # Add metadata from base TF
        aligned["year"] = base_features["year"]
        aligned["month"] = base_features["month"]

        self.features = aligned

    # _compute_ladder_returns is INHERITED from AgamottoResearch (2026-08-06).
    # It used to be duplicated here with `size = min(long_layers, short_layers)`
    # and a hardcoded 0.0001 step. Two copies of the target maths is exactly how
    # the engines drifted apart; see agamotto/ladder.py for the one
    # implementation and tests/test_kline_ladder_sizing.py for the parity test.
    #
    # The cross-TF target in verticalize() carried the SAME duplicate and was
    # folded onto agamotto/ladder.py on 2026-08-07, so both of orb's target
    # paths — same-TF and cross-TF — now honour LADDER_LONG / LADDER_SHORT.

    def _remap_tf_columns(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """Remap TF-prefixed columns to unprefixed for parent filter logic.

        E.g. if tf='1d', maps '1d_close' -> 'close', '1d_mvg1' -> 'mvg1',
        '1d_rsi' -> 'rsi', etc.
        Returns a copy with remapped columns added (overwriting any existing
        unprefixed columns so the parent filter sees this TF's values).

        A filter column this TF does NOT carry is DROPPED rather than left
        showing through (2026-08-23). The bare columns hold TARGET_TF values
        (verticalize step 3/3b), so leaving one in place lets `<tf>_<atom>` read
        the target timeframe and call it `<tf>` — the same silent wrong answer
        the compound-dispatch fix above removes, arriving by a second route
        (an atom naming a TF that is not in TIMEFRAMES, or a column one TF
        computed and another did not). Dropped, `_require_col` raises and names
        the column instead. TARGET_TF is exempt because the bare columns ARE
        its values: after the verticalize de-duplication there is no
        `{target_tf}_<derived>` copy left to remap from, and an atomic
        `15m_<atom>` is meant to fall through to the bare column.
        """
        # All columns the parent's _apply_filter_mask can access
        filter_cols = set(_RAW_COLUMNS) | set(_DERIVED_FEATURES)

        remapped = df.copy()
        tf_prefix = f"{tf}_"
        seen: set[str] = set()
        for col in df.columns:
            if col.startswith(tf_prefix):
                base_name = col[len(tf_prefix):]
                if base_name in filter_cols:
                    remapped[base_name] = df[col]
                    seen.add(base_name)
        if tf != self.target_tf:
            stale = [c for c in filter_cols if c in remapped.columns
                     and c not in seen]
            if stale:
                remapped = remapped.drop(columns=stale)
        return remapped
