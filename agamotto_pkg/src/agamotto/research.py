"""Agamotto research library for Binance Futures datasets."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional
import joblib
import glob
import numpy as np
import pandas as pd
import re
import os
import sys
import time
import logging

# Set up logger for Agamotto - ensure it uses root logger's handlers
logger = logging.getLogger(__name__)
# Don't add handlers here - let it propagate to root logger
# This ensures it uses the same handler configuration as the main process

from .research_common import (  # noqa: F401  (re-exported: external
    # callers and gauntlet.paths.kline_research_supports_s3 probe these
    # as attributes of agamotto.research)
    _S3_REGION,
    _is_s3,
    _obf,
    _open_s3fs,
    _s3_key,
)
from .research_features import FeatureEngineeringMixin
from .research_filters import FilterMaskMixin
from .utils import _symbol_to_native, _timeframe_to_seconds


def compute_dual_horizon_target(
    closes_base: pd.Series,
    closes_extra: pd.Series,
    base_tf: str = "1h",
    extra_tf: str = "15m",
) -> pd.Series:
    """Compute the extra-horizon return after the native hold period.

    For each base TF bar at time t:
      native close = closes_base[t+1]  (end of hold period)
      extra close  = closes_extra[t + base_tf + extra_tf]
      return = extra_close / native_close - 1

    Returns a Series aligned to closes_base index with NaN where
    extra data doesn't extend far enough.
    """
    base_seconds = _timeframe_to_seconds(base_tf)
    extra_seconds = _timeframe_to_seconds(extra_tf)
    offset = pd.Timedelta(seconds=base_seconds + extra_seconds)

    result = pd.Series(np.nan, index=closes_base.index, name="return_15m_extra")

    for i in range(len(closes_base) - 1):
        t = closes_base.index[i]
        native_close = closes_base.iloc[i + 1]
        extra_ts = t + offset

        # Find the 15m close at exactly extra_ts
        if extra_ts in closes_extra.index:
            extra_close = closes_extra.loc[extra_ts]
            if native_close != 0:
                result.iloc[i] = extra_close / native_close - 1

    return result


class AgamottoResearch(FeatureEngineeringMixin, FilterMaskMixin):
    # Directionally-biased filters: these only make sense for one side.
    # Enforced at regime_stack.csv generation time so nonsensical combos
    # (e.g. macd_bullish_short) are never created.
    LONG_ONLY_FILTERS = frozenset({
        "macd_bullish", "stoch_bullish", "rsi_oversold",
        "mfi_oversold", "bop_bullish", "roc_positive",
    })
    SHORT_ONLY_FILTERS = frozenset({
        "macd_bearish", "rsi_overbought",
        "mfi_overbought", "bop_bearish", "roc_negative",
    })
    # buy_pressure and sar_aligned are direction-agnostic: condition flips per position

    # Filters that genuinely read close/mvg1/mvg2 in _apply_filter_mask. Only
    # these are gated on the price columns, and a missing column RAISES rather
    # than returning an all-True mask (2026-08-02, no-silent-fallback). Every
    # other atom reads its own column and must dispatch BEFORE that check —
    # mirrors the mjolnir fix (b5ea04a, mjolnir/core/regime_filters.py).
    # `combined_*` composites are handled by prefix, not listed here.
    MVG_DEPENDENT_FILTERS = frozenset({
        "trend_aligned", "strong_trend", "ma_momentum", "above_all_mas",
        "near_ma", "bb_rebound", "sar_aligned",
    })

    @classmethod
    def allowed_positions(cls, filter_name: str) -> list:
        """Return ['long'], ['short'], or ['long','short'] for a regime name.

        Splits compound names on '_and_' and checks each component.
        Contradictory combos (long-only + short-only) return [] (skip).
        """
        # Accept coded regimes (rename rollout): decode code→real, real passes through.
        if isinstance(filter_name, str):
            filter_name = _obf().decode_regime_tolerant(filter_name)
        parts = filter_name.split("_and_") if "_and_" in filter_name else [filter_name]
        needs_long = False
        needs_short = False
        for part in parts:
            # Strip TF prefix (e.g. "4h_macd_bullish" → "macd_bullish")
            base = part.strip()
            tokens = base.split("_")
            if len(tokens) > 1 and tokens[0] in ("15m", "1h", "4h", "1d"):
                base = "_".join(tokens[1:])
            if base in cls.LONG_ONLY_FILTERS:
                needs_long = True
            if base in cls.SHORT_ONLY_FILTERS:
                needs_short = True
        if needs_long and needs_short:
            return []  # contradictory — skip
        if needs_long:
            return ["long"]
        if needs_short:
            return ["short"]
        return ["long", "short"]

    # Canonical base regime list (moved out of the public marvel generator so
    # the real names live only in the obfuscated package). `baseline` excluded —
    # removed 2026-06-18 (see CLAUDE.md). NOTE: `_and_` composites are built from
    # the atoms in _apply_filter_mask; allowed_positions enforces directionality.
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

    # Comprehensive IC-sweep filter set (moved out of the public research_sweep.py
    # so real names live only here). `baseline` excluded (see CLAUDE.md).
    _SWEEP_VOL_FILTERS = ["low_volume", "high_volume", "vol_breakout"]
    _SWEEP_TECH_FILTERS = [
        "above_all_mas", "rsi_oversold", "rsi_overbought",
        "cci_reversal", "bb_rebound", "macd_bullish", "macd_bearish",
        "stoch_bullish", "adx_trend", "mom_positive", "strong_trend",
        "ma_momentum", "near_ma", "strong_candle",
    ]

    @classmethod
    def comprehensive_sweep_regimes(cls) -> list[str]:
        """Coded regime names for the 'comprehensive' alpha sweep: each vol and
        tech atom plus every vol×tech `_and_` combo. Obfuscated (structure
        preserved) so the public sweep driver never holds real names.
        """
        c = _obf()
        names = (
            cls._SWEEP_VOL_FILTERS
            + cls._SWEEP_TECH_FILTERS
            + [f"{v}_and_{t}" for v in cls._SWEEP_VOL_FILTERS
               for t in cls._SWEEP_TECH_FILTERS]
        )
        return [c.encode_regime(n) for n in names]

    @classmethod
    def generate_regime_stack(cls) -> list[dict]:
        """Coded [{regime, position}] for every base regime × allowed position.

        Regime names are returned OBFUSCATED (structure preserved) so the public
        marvel generator that writes regime_stack.csv never handles real names.
        """
        c = _obf()
        rows = []
        for regime in cls.BASE_REGIMES:
            for pos in cls.allowed_positions(regime):
                rows.append({"regime": c.encode_regime(regime), "position": pos})
        return rows

    def __init__(self, config: Dict[str, object], home_root: str) -> None:
        self.config = config
        self.home_root = home_root.rstrip("/")
        self.raw: pd.DataFrame | None = None
        self.features: pd.DataFrame | None = None
        self.filtered_signals: pd.DataFrame | None = None
        self.filtered_signals_long: pd.DataFrame | None = None
        self.filtered_signals_short: pd.DataFrame | None = None

    def predict(self) -> None:
        """
        Optional hook for subclasses (like AgamottoTrading) to inject 
        model predictions into the filtered signals.
        """
        pass

    def load(self) -> None:
        frames: List[pd.DataFrame] = []
        whitelist = {
            _symbol_to_native(sym)
            for sym in self.config.get("SYMBOLS", [])
            if _symbol_to_native(sym)
        }

        timeframe = self.config.get("TIME_UNIT", "1d")
        exchange = self.config.get("EXCHANGE", "BINANCEFUTURES")
        data_family = self.config.get("DATA", "liquid")
        
        # Determine source directory based on exchange type
        if exchange.upper() == "STOCKS":
            # Stocks data structure: data/{data_family}/{timeframe}/
            source_dir = f"{self.home_root}/data/{data_family}/{timeframe}"
        else:
            # BINANCEFUTURES data structure: data/BINANCEFUTURES/{timeframe}/{data_family}
            source_dir = f"{self.home_root}/data/BINANCEFUTURES/{timeframe}/{data_family}"
        
        for symbol_dir in sorted(glob.glob(f"{source_dir}/*")):
            if not os.path.isdir(symbol_dir):
                continue
                
            symbol = os.path.basename(symbol_dir)
            if whitelist and symbol.upper() not in whitelist:
                continue
            
            # Load all monthly CSV files for this symbol
            symbol_frames = []
            for csv_path in sorted(glob.glob(f"{symbol_dir}/*_{timeframe}.csv")):
                logger.debug(f"Loading {csv_path}")
                try:
                    # Read CSV with header to detect actual columns
                    df_header = pd.read_csv(csv_path, nrows=0)
                    actual_columns = df_header.columns.tolist()
                    
                    # Read full CSV
                    df = pd.read_csv(csv_path, header=0)
                    
                    # Ensure open_time_ms exists (required)
                    if "open_time_ms" not in df.columns:
                        raise ValueError(f"Missing required column 'open_time_ms' in {csv_path}")
                    
                    df["timestamp"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
                    df.set_index("timestamp", inplace=True)
                    df = df[~df.index.duplicated(keep="last")]
                    
                    # Required columns (must exist)
                    required_cols = ["open", "high", "low", "close", "volume"]
                    
                    # Optional columns (fill with NaN if missing)
                    optional_cols = [
                        "quote_volume",
                        "number_of_trades",
                        "taker_buy_base_volume",
                        "taker_buy_quote_volume",
                    ]
                    
                    # Check required columns
                    missing_required = [col for col in required_cols if col not in df.columns]
                    if missing_required:
                        raise ValueError(f"Missing required columns: {missing_required}")
                    
                    # Select and convert to float
                    existing_cols = [col for col in required_cols + optional_cols if col in df.columns]
                    df = df[existing_cols].astype(float)
                    symbol_frames.append(df)
                except Exception as exc:
                    logger.warning(f"Failed to load {csv_path}: {exc}")
                    continue
            
            if symbol_frames:
                # Combine all monthly data for this symbol
                symbol_df = pd.concat(symbol_frames).sort_index()
                symbol_df = symbol_df[~symbol_df.index.duplicated(keep="last")]
                symbol_df.columns = [f"{symbol}_{col}" for col in symbol_df.columns]
                frames.append(symbol_df)
        if not frames:
            raise RuntimeError(f"No CSV files matched in {source_dir}")

        combined = pd.concat(frames, axis=1).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.index = combined.index.tz_convert(None)
        self.raw = combined

    def create(self) -> str:
        if self.features is None:
             self.engineer_features()
             
        # Drop the last row (NaN targets) for Training/Research output
        if self.features is not None and not self.features.empty:
             self.features = self.features.iloc[:-1]

        self.verticalize()

        # Resolve output directory based on VERSION
        version = self.config.get("VERSION")
        if not version:
             raise ValueError("VERSION missing from config")
             
        # Use OUTPUT_DIR from config if available (injected by runner)
        if "OUTPUT_DIR" in self.config:
            out_dir = self.config["OUTPUT_DIR"]
        else:
            # Construct path: gauntlet/pred_{version}
            out_dir = os.path.join("gauntlet", f"pred_{version}")
        
        # Route B: an s3:// OUTPUT_DIR is already absolute (bucket-qualified) and
        # has no directories to create. os.path.isabs() returns False for it, so
        # WITHOUT this branch the URI is joined onto home_root and makedirs()
        # creates a local dir named "s3:" — the silent mis-write this guards.
        out_is_s3 = _is_s3(out_dir)
        if not out_is_s3:
            # Ensure absolute path using home_root if relative
            if not os.path.isabs(out_dir):
                out_dir = os.path.join(self.home_root, out_dir)

            os.makedirs(out_dir, exist_ok=True)

        # Save vertical features
        if hasattr(self, 'vertical_features') and self.vertical_features is not None:
            if out_is_s3:
                # Small metadata, but it belongs under OUTPUT_DIR so downstream
                # resolves it the same way for both routes — stream the CSV
                # bytes straight to the object.
                v_key = f"{_s3_key(out_dir)}/vertical_features.csv"
                s3fs = _open_s3fs()
                with s3fs.open_output_stream(v_key) as sink:
                    sink.write(self.vertical_features.to_csv(index=False).encode())
                logger.info("Route B: wrote vertical_features.csv to s3://%s", v_key)
            else:
                v_out_path = os.path.join(out_dir, "vertical_features.csv")
                self.vertical_features.to_csv(v_out_path, index=False)


        # Load regime stack — REGIME_STACK_PATH must be set in config, no fallback
        if "REGIME_STACK_PATH" not in self.config:
            raise ValueError("REGIME_STACK_PATH not set in config — cannot run research without a regime list")

        candidate = self.config["REGIME_STACK_PATH"]
        if os.path.exists(candidate):
            stack_path = candidate
        elif os.path.exists(os.path.join(self.home_root, candidate)):
            stack_path = os.path.join(self.home_root, candidate)
        else:
            raise FileNotFoundError(f"REGIME_STACK_PATH not found: {candidate}")

        if not stack_path.endswith(".csv"):
            raise ValueError(f"REGIME_STACK_PATH must be a .csv file, got: {stack_path}")
        import csv as _csv
        with open(stack_path, newline="") as f:
            regime_stack = list(_csv.DictReader(f))

        # Skip metadata rows (e.g. __summary__ from filter_regime_stacks.py)
        regime_stack = [r for r in regime_stack if not str(r.get("regime", "")).startswith("__")]
        logger.info(f"Loaded {len(regime_stack)} regimes from {stack_path}")
        for regime in regime_stack:
            self.filter_signals(regime, save=True, out_dir=out_dir)
            

        return out_dir

    def verticalize(self, df: pd.DataFrame | None = None) -> None:
        """
        Unify features by removing symbol prefixes and stacking them vertically.
        Populates self.vertical_features.
        """
        if df is None:
            df = self.features
        if df is None:
            raise RuntimeError("Call engineer_features() before verticalize().")

        frames = []
        for sym in self.config["SYMBOLS"]:
            native = _symbol_to_native(sym)
            if native is None:
                continue
            prefix = native
            
            # Base rename map
            base_rename_map = {
                f"{prefix}_close": "close",
                f"{prefix}_mvg1": "mvg1",
                f"{prefix}_mvg2": "mvg2",
                f"{prefix}_mvg3": "mvg3",
                f"{prefix}_price_range": "price_range",
                f"{prefix}_price_range_pct": "price_range_pct",
                f"{prefix}_price_range_pct_q50": "price_range_pct_q50",
                f"{prefix}_open_close_diff": "open_close_diff",
                f"{prefix}_open_close_pct": "open_close_pct",
                f"{prefix}_high_open_pct": "high_open_pct",
                f"{prefix}_low_open_pct": "low_open_pct",
                # Lagged returns
                f"{prefix}_ret_lag1": "ret_lag1",
                f"{prefix}_ret_lag2": "ret_lag2",
                f"{prefix}_ret_lag3": "ret_lag3",
                # TA-Lib
                f"{prefix}_rsi": "rsi",
                f"{prefix}_rsi_7": "rsi_7",
                f"{prefix}_rsi_28": "rsi_28",
                f"{prefix}_macd": "macd",
                f"{prefix}_macdhist": "macdhist",
                f"{prefix}_stoch_k": "stoch_k",
                f"{prefix}_stoch_d": "stoch_d",
                f"{prefix}_cci": "cci",
                f"{prefix}_adx": "adx",
                f"{prefix}_dx": "dx",
                f"{prefix}_plus_di": "plus_di",
                f"{prefix}_minus_di": "minus_di",
                f"{prefix}_mom": "mom",
                f"{prefix}_roc": "roc",
                f"{prefix}_willr": "willr",
                f"{prefix}_cmo": "cmo",
                f"{prefix}_trix": "trix",
                f"{prefix}_ultosc": "ultosc",
                f"{prefix}_stochrsi_k": "stochrsi_k",
                f"{prefix}_stochrsi_d": "stochrsi_d",
                f"{prefix}_obv": "obv",
                f"{prefix}_ad": "ad",
                f"{prefix}_mfi": "mfi",
                f"{prefix}_bop": "bop",
                f"{prefix}_atr": "atr",
                f"{prefix}_natr": "natr",
                f"{prefix}_parkinson_vol": "parkinson_vol",
                f"{prefix}_bb_upper": "bb_upper",
                f"{prefix}_bb_lower": "bb_lower",
                f"{prefix}_sar": "sar",
                # Rolling stats
                f"{prefix}_std": "std",
                f"{prefix}_skew": "skew",
                f"{prefix}_kurt": "kurt",
                f"{prefix}_acf_lag1": "acf_lag1",
                # Volume features
                f"{prefix}_quote_vol_ratio": "quote_vol_ratio",
                f"{prefix}_vol_ratio": "vol_ratio",
                f"{prefix}_buy_pressure": "buy_pressure",
                f"{prefix}_trade_intensity": "trade_intensity",
                f"{prefix}_vol_ret_lag1": "vol_ret_lag1",
                f"{prefix}_vol_ret_lag2": "vol_ret_lag2",
                f"{prefix}_vol_ret_lag3": "vol_ret_lag3",
                # Compound classification label targets (Vomir)
                f"{prefix}_return_dip": "return_dip",
                f"{prefix}_return_rip": "return_rip",
            }

            # Check for laddered returns
            long_return_col = f"{prefix}_return_long"
            short_return_col = f"{prefix}_return_short"
            long_return_raw_col = f"{prefix}_return_long_raw"
            short_return_raw_col = f"{prefix}_return_short_raw"
            base_return_col = f"{prefix}_return"

            # Logic for default "long" verticalization
            rename_map = base_rename_map.copy()
            if long_return_col in df.columns:
                rename_map[long_return_col] = "return_long"
            if short_return_col in df.columns:
                rename_map[short_return_col] = "return_short"
            if long_return_raw_col in df.columns:
                rename_map[long_return_raw_col] = "return_long_raw"
            if short_return_raw_col in df.columns:
                rename_map[short_return_raw_col] = "return_short_raw"
            if base_return_col in df.columns:
                rename_map[base_return_col] = "return"

            # Dual-horizon 2-bar target columns (present only for DH experiments)
            for suffix in ("ret_2bar", "return_long_2bar", "return_short_2bar",
                           "return_long_2bar_raw", "return_short_2bar_raw"):
                col2 = f"{prefix}_{suffix}"
                if col2 in df.columns:
                    rename_map[col2] = suffix

            valid_cols = [c for c in rename_map if c in df.columns]

            subset = df.loc[:, valid_cols + ["year", "month"]].copy()
            subset.rename(columns={orig: rename_map[orig] for orig in valid_cols}, inplace=True)
            
            subset["symbol"] = sym
            subset["timestamp"] = subset.index
            
            frames.append(subset)
            
        if frames:
            self.vertical_features = pd.concat(frames, axis=0, ignore_index=True)
        else:
            self.vertical_features = pd.DataFrame()

    @staticmethod
    def _volume_ratio(df: pd.DataFrame, filter_name: str) -> pd.Series:
        """Volume-ratio series backing low_volume/high_volume/vol_breakout.

        Prefers quote_vol_ratio (quote_vol / 7d MA) over vol_ratio
        (vol / 7d MA), matching the longstanding priority. Raises when neither
        column exists — the old `df.get("quote_vol_ratio", df.get("vol_ratio",
        1.0))` collapsed to the scalar 1.0 there, so the three comparisons
        returned a plain Python bool instead of a per-row mask and
        `features_df[mask]` degenerated (KeyError standalone, silent all-False
        under `_and_`, silently dropped under `_or_`). engineer_features always
        builds vol_ratio (raw `volume` is a required load column), so a real
        feature frame never hits this.
        """
        for col in ("quote_vol_ratio", "vol_ratio"):
            if col in df.columns:
                return df[col]
        raise ValueError(
            f"Filter {filter_name!r} requires a volume-ratio column; frame is "
            "missing both 'quote_vol_ratio' and 'vol_ratio'. A scalar 1.0 "
            "fallback here returns a bool instead of a per-row mask — "
            "failing loud instead. See CLAUDE.md no-silent-fallback rules.")

    def filter_signals(self, regime: dict, limit_timestamp: Optional[pd.Timestamp] = None, save: bool = False, out_dir: str = None) -> pd.DataFrame:
        """
        Filter signals for a SINGLE regime.
        
        Args:
            regime: Regime dictionary containing 'regime' and 'position'.
            limit_timestamp: If provided, only process data from this timestamp onwards.
            save: If True, save filtered signals to disk.
            out_dir: Directory to save signals if save is True.
            
        Returns:
            pd.DataFrame: Filtered signals for this regime.
        """
        if not hasattr(self, 'vertical_features') or self.vertical_features is None:
             raise RuntimeError("Call verticalize() before filter_signals().")

        features_df = self.vertical_features

        if "regime" not in regime:
            raise KeyError("regime row missing required 'regime' key (no baseline default)")
        regime_name = regime["regime"]
        position = regime.get("position", "long")

        # Convert list regime to string for storage (e.g. ["high_volume", "&", "cci_reversal"] -> "high_volume_and_cci_reversal")
        regime_name_str = regime_name
        if isinstance(regime_name, list):
            regime_name_str = "_and_".join(p for p in regime_name if p not in ("|", "&"))

        mask = self._apply_filter_mask(features_df, regime_name, position)

        filtered_subset = features_df[mask].copy()
        if filtered_subset.empty:
            return pd.DataFrame()

        filtered_subset["position"] = position
        filtered_subset["regime"] = regime_name_str
        
        # Assign 'ret' column based on position
        if position == "long":
             if "return_long" in filtered_subset.columns:
                 filtered_subset["ret"] = filtered_subset["return_long"]
             else:
                 raise KeyError(f"Position 'long' requires 'return_long' column in vertical features.")
             if "return_long_raw" in filtered_subset.columns:
                 filtered_subset["ret_raw"] = filtered_subset["return_long_raw"]
        elif position == "short":
             if "return_short" in filtered_subset.columns:
                 filtered_subset["ret"] = filtered_subset["return_short"]
             else:
                 raise KeyError(f"Position 'short' requires 'return_short' column in vertical features.")
             if "return_short_raw" in filtered_subset.columns:
                 filtered_subset["ret_raw"] = filtered_subset["return_short_raw"]
                 
        if save and out_dir:
            # Save to pred_agamotto*_1/filter/filter_{regime}
            
            # Clean regime_name first to remove any pre-existing position suffix
            clean_regime = regime_name_str.replace("_long", "").replace("_short", "")
            
            # Construct simplified filename: filter_{regime}_{position}.csv
            # This avoids filter_regime_long_long.csv
            safe_name = f"{clean_regime}_{position}"
            
            # Cleaning filename characters
            safe_name = "".join([c if c.isalnum() or c in ['_'] else '_' for c in safe_name])
            
            # Route B: write the bulk filter parquet DIRECTLY to S3 when out_dir
            # is an s3:// URI. S3 has no directories, so save_dir is a bucket-
            # qualified key prefix and there is nothing to mkdir.
            save_is_s3 = _is_s3(out_dir)
            if save_is_s3:
                save_dir = f"{_s3_key(out_dir)}/filter"
                save_path = f"{save_dir}/filter_{safe_name}.parquet"
            else:
                save_dir = os.path.join(out_dir, "filter")
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, f"filter_{safe_name}.parquet")
            try:
                # Obfuscation: persist feature columns under opaque codes (real
                # name -> code, TF prefix preserved). Targets / metadata / OHLCV
                # are not in the feature map and pass through unchanged. Done at
                # write only — the in-memory frame keeps real names. Downstream
                # (select_feature_columns, training, meta.pkl, preds) inherits the
                # coded schema, so the public marvel repo never sees real names.
                _to_save = filtered_subset.rename(
                    columns=_obf().encode_columns(filtered_subset.columns))
                # Narrow float64 feature columns to float32 (the rolling trainer
                # casts to float32 on load anyway) + zstd — roughly halves the
                # filter parquet with zero training impact. Only float64 columns
                # are cast, so integer/timestamp columns keep full precision.
                _f64_to_f32 = {
                    c: "float32" for c in _to_save.columns
                    if _to_save[c].dtype == "float64"
                }
                _narrowed = _to_save.astype(_f64_to_f32)
                if save_is_s3:
                    # pandas.to_parquet cannot address an S3FileSystem key, so
                    # go through pyarrow directly. pq.write_table publishes the
                    # object only on a successful close, so a torn write leaves
                    # no readable object (no .tmp staging needed — and no s3fs
                    # rename, which is what breaks the FUSE-mount route).
                    import pyarrow as pa
                    import pyarrow.parquet as pq
                    pq.write_table(
                        pa.Table.from_pandas(_narrowed, preserve_index=False),
                        save_path, compression="zstd", filesystem=_open_s3fs(),
                    )
                else:
                    _narrowed.to_parquet(
                        save_path, index=False, compression="zstd",
                    )
                # logger.info(f"Saved filtered signals for {regime_id} to {save_path}")
            except Exception as e:
                # Route B must fail LOUD: a swallowed S3 error would leave a
                # silently incomplete filter tree that Step 2 then trains on,
                # producing a partial book with no warning. The local path keeps
                # its historical log-and-continue behaviour so this change is
                # confined to the new route.
                if save_is_s3:
                    raise RuntimeError(
                        f"Route B: failed to write filter parquet to s3://{save_path} "
                        f"for {safe_name}: {e!r}. Refusing to continue with an "
                        "incomplete filter tree (CLAUDE.md: no silent fallbacks)."
                    ) from e
                logger.error(f"Failed to save filtered signals for {safe_name}: {e}")

        return filtered_subset
