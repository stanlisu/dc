"""Orb research: cross-timeframe feature alignment on top of Agamotto."""

from __future__ import annotations

from typing import Dict, List
import gc
import logging

import numpy as np
import pandas as pd

from agamotto import AgamottoResearch
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

# Raw / MA / return columns that filters need but ML should NOT use.
_RAW_COLUMNS = [
    "close", "open", "high", "low", "volume",
    "quote_volume", "number_of_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume",
    "mvg1", "mvg2", "mvg3",
]

_RETURN_COLUMNS = [
    "return", "return_long", "return_short",
    "return_long_raw", "return_short_raw",
]

_TF_PREFIXES = ("15m_", "1h_", "4h_", "1d_")


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
_ORB_TIMEFRAMES = ["15m", "1h", "4h", "1d"]


def _orb_has_baseline(regime_name: str) -> bool:
    """True if any conjunct of a regime is the banned baseline.

    Delegates to the single shared, suffix-/TF-/_or_-aware codec predicate so the
    baseline-ban logic has one home (see CLAUDE.md, 2026-06-18)."""
    return _obf().has_baseline(regime_name)


class OrbResearch(AgamottoResearch):
    """Cross-timeframe research: merges 4 TF features into one wide matrix."""

    @classmethod
    def generate_regime_stack(cls) -> list[dict]:
        """Coded [{regime, position}] for all Orb cross-TF regimes.

        Builds regimes internally with real atom names (reusing allowed_positions
        for directionality), drops any baseline conjunct, dedups, then returns
        OBFUSCATED, structure-preserving regime names so the public marvel
        generator never handles real names.
        """
        def allowed(filters):
            return cls.allowed_positions("_and_".join(filters))

        regimes: list[dict] = []

        # 1. Single-TF filters on cross-TF data
        for tf in _ORB_TIMEFRAMES:
            for filt in _ORB_BASE_FILTERS:
                for pos in allowed((f"{tf}_{filt}",)):
                    regimes.append({"regime": f"{tf}_{filt}", "position": pos})

        # 2. Cross-TF compounds (>=1 unidirectional leg)
        cross_tf_combos = [
            ("1d_strong_trend", "1h_macd_bullish"), ("1d_strong_trend", "1h_macd_bearish"),
            ("1d_strong_trend", "1h_rsi_oversold"), ("1d_strong_trend", "1h_rsi_overbought"),
            ("1d_ma_momentum", "1h_macd_bullish"), ("1d_ma_momentum", "1h_macd_bearish"),
            ("1d_ma_momentum", "1h_rsi_oversold"), ("1d_ma_momentum", "1h_rsi_overbought"),
            ("1d_above_all_mas", "1h_macd_bullish"), ("1d_above_all_mas", "1h_macd_bearish"),
            ("1d_above_all_mas", "1h_rsi_oversold"), ("1d_above_all_mas", "1h_rsi_overbought"),
            ("4h_strong_trend", "1h_macd_bullish"), ("4h_strong_trend", "1h_macd_bearish"),
            ("4h_strong_trend", "1h_rsi_oversold"), ("4h_strong_trend", "1h_rsi_overbought"),
            ("4h_adx_trend", "1h_macd_bullish"), ("4h_adx_trend", "1h_macd_bearish"),
            ("4h_adx_trend", "1h_rsi_oversold"), ("4h_adx_trend", "1h_rsi_overbought"),
            ("4h_macd_bullish", "1h_rsi_oversold"), ("4h_macd_bullish", "1h_rsi_overbought"),
            ("1d_strong_trend", "4h_macd_bullish"), ("1d_strong_trend", "4h_macd_bearish"),
            ("1d_strong_trend", "4h_rsi_oversold"), ("1d_strong_trend", "4h_rsi_overbought"),
            ("1d_ma_momentum", "4h_macd_bullish"), ("1d_ma_momentum", "4h_macd_bearish"),
            ("1d_vol_breakout", "4h_macd_bullish"), ("1d_vol_breakout", "4h_macd_bearish"),
            ("1d_vol_breakout", "4h_rsi_oversold"), ("1d_vol_breakout", "4h_rsi_overbought"),
            ("1d_vol_breakout", "1h_macd_bullish"), ("1d_vol_breakout", "1h_macd_bearish"),
            ("1d_vol_breakout", "1h_rsi_oversold"), ("1d_vol_breakout", "1h_rsi_overbought"),
            ("4h_vol_breakout", "1h_macd_bullish"), ("4h_vol_breakout", "1h_macd_bearish"),
            ("4h_vol_breakout", "1h_rsi_oversold"), ("4h_vol_breakout", "1h_rsi_overbought"),
            ("1d_strong_trend", "4h_vol_breakout", "1h_macd_bullish"),
            ("1d_strong_trend", "4h_vol_breakout", "1h_macd_bearish"),
            ("1d_vol_breakout", "4h_vol_breakout", "1h_macd_bullish"),
            ("1d_vol_breakout", "4h_vol_breakout", "1h_macd_bearish"),
        ]
        # 3. Same-TF volume × directional cross products
        same_tf = [(f"{tf}_{v}", f"{tf}_{d}") for tf in _ORB_TIMEFRAMES
                   for (v, d) in _ORB_SAME_TF_VOL_DIR]
        # 4. Cross-TF vol/volatility context + directional signal
        cross_tf_vol_directional = [
            ("1d_low_volume", "4h_macd_bullish"), ("1d_low_volume", "4h_macd_bearish"),
            ("1d_low_volume", "4h_rsi_oversold"), ("1d_low_volume", "4h_rsi_overbought"),
            ("1d_low_volume", "4h_bop_bullish"), ("1d_low_volume", "4h_bop_bearish"),
            ("1d_low_volume", "1h_macd_bullish"), ("1d_low_volume", "1h_macd_bearish"),
            ("1d_low_volume", "1h_rsi_oversold"), ("1d_low_volume", "1h_rsi_overbought"),
            ("1d_low_volume", "1h_bop_bullish"), ("1d_low_volume", "1h_bop_bearish"),
            ("1d_low_volume", "1h_mfi_oversold"), ("1d_low_volume", "1h_mfi_overbought"),
            ("1d_high_volume", "4h_macd_bullish"), ("1d_high_volume", "4h_macd_bearish"),
            ("1d_high_volume", "4h_rsi_oversold"), ("1d_high_volume", "4h_rsi_overbought"),
            ("1d_high_volume", "4h_bop_bullish"), ("1d_high_volume", "4h_bop_bearish"),
            ("1d_high_volume", "1h_macd_bullish"), ("1d_high_volume", "1h_macd_bearish"),
            ("1d_high_volume", "1h_rsi_oversold"), ("1d_high_volume", "1h_rsi_overbought"),
            ("1d_high_volume", "1h_bop_bullish"), ("1d_high_volume", "1h_bop_bearish"),
            ("1d_high_volume", "1h_mfi_oversold"), ("1d_high_volume", "1h_mfi_overbought"),
            ("4h_low_volume", "1h_macd_bullish"), ("4h_low_volume", "1h_macd_bearish"),
            ("4h_low_volume", "1h_rsi_oversold"), ("4h_low_volume", "1h_rsi_overbought"),
            ("4h_low_volume", "1h_bop_bullish"), ("4h_low_volume", "1h_bop_bearish"),
            ("4h_low_volume", "1h_mfi_oversold"), ("4h_low_volume", "1h_mfi_overbought"),
            ("4h_low_volume", "1h_stoch_bullish"),
            ("4h_high_volume", "1h_macd_bullish"), ("4h_high_volume", "1h_macd_bearish"),
            ("4h_high_volume", "1h_rsi_oversold"), ("4h_high_volume", "1h_rsi_overbought"),
            ("4h_high_volume", "1h_bop_bullish"), ("4h_high_volume", "1h_bop_bearish"),
            ("4h_high_volume", "1h_mfi_oversold"), ("4h_high_volume", "1h_mfi_overbought"),
            ("1d_low_vol", "4h_macd_bullish"), ("1d_low_vol", "4h_macd_bearish"),
            ("1d_low_vol", "4h_rsi_oversold"), ("1d_low_vol", "4h_rsi_overbought"),
            ("1d_low_vol", "4h_bop_bullish"), ("1d_low_vol", "4h_bop_bearish"),
            ("1d_low_vol", "1h_macd_bullish"), ("1d_low_vol", "1h_macd_bearish"),
            ("1d_low_vol", "1h_rsi_oversold"), ("1d_low_vol", "1h_rsi_overbought"),
            ("1d_low_vol", "1h_bop_bullish"), ("1d_low_vol", "1h_bop_bearish"),
            ("1d_low_vol", "1h_mfi_oversold"), ("1d_low_vol", "1h_mfi_overbought"),
            ("1d_low_vol", "1h_stoch_bullish"),
            ("1d_high_vol", "4h_macd_bullish"), ("1d_high_vol", "4h_macd_bearish"),
            ("1d_high_vol", "4h_rsi_oversold"), ("1d_high_vol", "4h_rsi_overbought"),
            ("1d_high_vol", "1h_macd_bullish"), ("1d_high_vol", "1h_macd_bearish"),
            ("1d_high_vol", "1h_rsi_oversold"), ("1d_high_vol", "1h_rsi_overbought"),
            ("1d_high_vol", "1h_bop_bullish"), ("1d_high_vol", "1h_bop_bearish"),
            ("4h_low_vol", "1h_macd_bullish"), ("4h_low_vol", "1h_macd_bearish"),
            ("4h_low_vol", "1h_rsi_oversold"), ("4h_low_vol", "1h_rsi_overbought"),
            ("4h_low_vol", "1h_bop_bullish"), ("4h_low_vol", "1h_bop_bearish"),
            ("4h_low_vol", "1h_mfi_oversold"), ("4h_low_vol", "1h_mfi_overbought"),
            ("4h_low_vol", "1h_stoch_bullish"),
            ("4h_high_vol", "1h_macd_bullish"), ("4h_high_vol", "1h_macd_bearish"),
            ("4h_high_vol", "1h_rsi_oversold"), ("4h_high_vol", "1h_rsi_overbought"),
            ("4h_high_vol", "1h_bop_bullish"), ("4h_high_vol", "1h_bop_bearish"),
        ]
        # 5. Cross-TF TA-lab combos
        cross_tf_talab = [
            ("1d_strong_trend", "1h_bop_bullish"), ("1d_strong_trend", "1h_bop_bearish"),
            ("1d_vol_breakout", "1h_mfi_oversold"), ("1d_vol_breakout", "1h_mfi_overbought"),
            ("4h_vol_breakout", "1h_bop_bullish"), ("4h_vol_breakout", "1h_bop_bearish"),
            ("4h_vol_breakout", "1h_mfi_oversold"), ("4h_vol_breakout", "1h_mfi_overbought"),
        ]
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
        self.timeframes: List[str] = config.get(
            "TIMEFRAMES", ["15m", "1h", "4h", "1d"])
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

            # 1. TF-prefixed DERIVED features (ML features)
            for tf in self.timeframes:
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

            # 3b. Unprefixed derived features from TARGET_TF (for filter logic)
            #     Needed so atomic filters like strong_candle (open_close_pct),
            #     low_vol (price_range_pct) work without a TF prefix.
            for feat in _DERIVED_FEATURES:
                src_col = f"{self.target_tf}_{native}_{feat}"
                if src_col in self.features.columns:
                    sym_cols[feat] = self.features[src_col]

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
                    fee = float(self.config.get("FEE", 0.0)) / 10000.0
                    raw = self.features[exit_col] / self.features[entry_col] - 1
                    sym_cols["return"] = raw
                    if (exit_low_col in self.features.columns
                            and exit_high_col in self.features.columns):
                        # Ladder-adjusted cross-TF returns.
                        # exit_low/exit_high are already the exit bar values (no shift needed).
                        step_size = 0.0001
                        ladder = int(self.config.get("LADDER", 1) or 0)
                        entry_close = self.features[entry_col]
                        close_safe = entry_close.replace(0, np.nan)
                        exit_low = self.features[exit_low_col]
                        exit_high = self.features[exit_high_col]
                        dist_long = ((close_safe - exit_low) / close_safe).clip(lower=0)
                        long_layers = np.floor(dist_long / step_size).clip(0, ladder).fillna(0).astype(int)
                        dist_short = ((exit_high - close_safe) / close_safe).clip(lower=0)
                        short_layers = np.floor(dist_short / step_size).clip(0, ladder).fillna(0).astype(int)
                        # 2026-06-20 refined ladder (parity with mjolnir dc 4fb9eb8): no base
                        # rung + round-trip gate -> size = min(long_layers, short_layers),
                        # same size both sides. <1bps dip OR rise -> 0.
                        size = np.minimum(long_layers, short_layers)
                        fee_cost = fee * 2.0
                        sym_cols["return_long"] = (raw - fee_cost) * size
                        sym_cols["return_short"] = -(raw + fee_cost) * size
                        sym_cols["return_long_raw"] = raw * size
                        sym_cols["return_short_raw"] = raw * size
                    else:
                        sym_cols["return_long"] = raw - 2 * fee
                        sym_cols["return_short"] = -(raw + 2 * fee)
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
        # List → delegate to super (recursive calls come back here)
        if isinstance(filter_name, list):
            return super()._apply_filter_mask(df, filter_name, position)

        if not isinstance(filter_name, str):
            return super()._apply_filter_mask(df, filter_name, position)

        # Compound _and_ / _or_ → delegate to super (splits + recurses here)
        if "_and_" in filter_name or "_or_" in filter_name:
            return super()._apply_filter_mask(df, filter_name, position)

        # Check for TF prefix on the (atomic) filter name
        for prefix in _TF_PREFIXES:
            if filter_name.startswith(prefix):
                tf = prefix.rstrip("_")
                base_filter = filter_name[len(prefix):]
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

    @staticmethod
    def _remap_tf_columns(df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """Remap TF-prefixed columns to unprefixed for parent filter logic.

        E.g. if tf='1d', maps '1d_close' -> 'close', '1d_mvg1' -> 'mvg1',
        '1d_rsi' -> 'rsi', etc.
        Returns a copy with remapped columns added (overwriting any existing
        unprefixed columns so the parent filter sees this TF's values).
        """
        # All columns the parent's _apply_filter_mask can access
        filter_cols = set(_RAW_COLUMNS) | set(_DERIVED_FEATURES)

        remapped = df.copy()
        tf_prefix = f"{tf}_"
        for col in df.columns:
            if col.startswith(tf_prefix):
                base_name = col[len(tf_prefix):]
                if base_name in filter_cols:
                    remapped[base_name] = df[col]
        return remapped
