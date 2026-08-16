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

def _obf():
    """Lazy accessor for the vendored obfuscation codec (see _obf/codec.py)."""
    from ._obf.codec import default
    return default()


# Use relative import for internal module
try:
    from .lib_binance import fetch_futures_klines, klines_to_dataframe
except ImportError:
    # Fallback to assume it's in the path or installed
    try:
        from lib_binance import fetch_futures_klines, klines_to_dataframe
    except ImportError:
        pass

from .features_scalefree import SCALE_FREE_FEATURES, scale_free_levels
from .ladder import compute_ladder_multiplier, ladder_params
from .mm_target import (
    MINUTE_TIMEFRAME,
    TARGET_MODE_MM,
    compute_mm_target,
    target_mode,
)
from .utils import _symbol_to_native, _timeframe_to_seconds

# ── Trailing volatility-quantile ENTRY gates (2026-08-16) ────────────────────
# `price_range_pct` is the per-bar range; its TRAILING quantiles are the cutoffs
# the `high_vol_q*` regime atoms compare each bar against (research_filters.
# _VOL_QUANTILE_ATOMS). Cost is a fixed round trip and edge scales
# super-proportionally with bar volatility, so gating ENTRY on forecast range is
# cost avoidance, not new alpha.
#
# VOL_Q_WINDOW is an OPERATOR-CHOSEN bar count (integer bars, never a duration —
# CLAUDE.md), matching the existing price_range_pct_q50 window so the new cutoffs
# and the incumbent median share one lookback. It doubles as `min_periods` so the
# gate FAILS CLOSED during warmup: at min_periods=1 the rolling quantile of a
# 1-length window is the value itself, so a nominal-20% q80 gate fires on ~30% of
# the first 50 bars; at min_periods=VOL_Q_WINDOW it fires on 0% of them (the
# cutoff is NaN and `x > NaN` is False).
VOL_Q_WINDOW = 700
VOL_Q_LEVELS = (0.80, 0.90, 0.95)
# MUST stay a LITERAL list with a `_FEATURES` suffix: obfuscation/
# extract_inventory.py (:33 _FEATURE_LIST_SUFFIXES, :90 _collect_str_members)
# AST-scans list literals. A comprehension over VOL_Q_LEVELS is invisible to the
# extractor, so these columns would get no feature code, codec.encode_columns
# would silently omit them, and the REAL column names would leak into the filter
# parquet. Order must match VOL_Q_LEVELS (asserted in tests).
VOL_QUANTILE_FEATURES = [
    "price_range_pct_q80",
    "price_range_pct_q90",
    "price_range_pct_q95",
]

# Import filter definitions and evaluation logic from sub-module
from .research_filters import (
    _require_col,
    _volume_ratio,
    LONG_ONLY_FILTERS,
    SHORT_ONLY_FILTERS,
    MVG_DEPENDENT_FILTERS,
    BASE_REGIMES,
    _SWEEP_VOL_FILTERS,
    _SWEEP_TECH_FILTERS,
    allowed_positions,
    comprehensive_sweep_regimes,
    generate_regime_stack,
    apply_filter_mask,
)


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


class AgamottoResearch:
    LONG_ONLY_FILTERS = LONG_ONLY_FILTERS
    SHORT_ONLY_FILTERS = SHORT_ONLY_FILTERS
    MVG_DEPENDENT_FILTERS = MVG_DEPENDENT_FILTERS
    BASE_REGIMES = BASE_REGIMES
    _SWEEP_VOL_FILTERS = _SWEEP_VOL_FILTERS
    _SWEEP_TECH_FILTERS = _SWEEP_TECH_FILTERS

    allowed_positions = staticmethod(allowed_positions)
    comprehensive_sweep_regimes = staticmethod(comprehensive_sweep_regimes)
    generate_regime_stack = staticmethod(generate_regime_stack)
    _volume_ratio = staticmethod(_volume_ratio)

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
            source_dir = f"{self.home_root}/data/{data_family}/{timeframe}"
        else:
            source_dir = f"{self.home_root}/data/BINANCEFUTURES/{timeframe}/{data_family}"
        
        for symbol_dir in sorted(glob.glob(f"{source_dir}/*")):
            if not os.path.isdir(symbol_dir):
                continue
                
            symbol = os.path.basename(symbol_dir)
            if whitelist and symbol.upper() not in whitelist:
                continue
            
            symbol_frames = []
            for csv_path in sorted(glob.glob(f"{symbol_dir}/*_{timeframe}.csv")):
                logger.debug(f"Loading {csv_path}")
                try:
                    df_header = pd.read_csv(csv_path, nrows=0)
                    actual_columns = df_header.columns.tolist()
                    
                    df = pd.read_csv(csv_path, header=0)
                    
                    if "open_time_ms" not in df.columns:
                        raise ValueError(f"Missing required column 'open_time_ms' in {csv_path}")
                    
                    df["timestamp"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
                    df.set_index("timestamp", inplace=True)
                    df = df[~df.index.duplicated(keep="last")]
                    
                    required_cols = ["open", "high", "low", "close", "volume"]
                    optional_cols = [
                        "quote_volume",
                        "number_of_trades",
                        "taker_buy_base_volume",
                        "taker_buy_quote_volume",
                    ]
                    
                    missing_required = [col for col in required_cols if col not in df.columns]
                    if missing_required:
                        raise ValueError(f"Missing required columns: {missing_required}")
                    
                    existing_cols = [col for col in required_cols + optional_cols if col in df.columns]
                    df = df[existing_cols].astype(float)
                    symbol_frames.append(df)
                except Exception as exc:
                    logger.warning(f"Failed to load {csv_path}: {exc}")
                    continue
            
            if symbol_frames:
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

    def _load_minute_bars(self, symbol: str) -> pd.DataFrame:
        """1m OHLC for ONE symbol — the sub-bar grid the MM target resolves on.

        Loaded per symbol rather than as one wide frame: the combined 1m panel for a 28-symbol
        arm is ~1.85M minutes x 140 columns, and only one symbol is needed at a time.

        Mirrors `load()`'s layout and parsing exactly (same source_dir shape, same
        `open_time_ms` -> UTC index, same duplicate policy, same `tz_convert(None)` so the
        index matches `self.raw`). A symbol with no 1m data RAISES — silently dropping it
        would train that symbol on nothing while the rest of the arm carried MM economics.
        """
        exchange = self.config.get("EXCHANGE", "BINANCEFUTURES")
        if exchange.upper() == "STOCKS":
            raise ValueError(
                "TARGET_MODE='mm' is crypto-only: the MM lifecycle is a 24/7 post-only "
                "ladder, and the STOCKS data layout carries no 1m tree.")
        data_family = self.config.get("DATA", "liquid")
        sym_dir = (f"{self.home_root}/data/BINANCEFUTURES/"
                   f"{MINUTE_TIMEFRAME}/{data_family}/{symbol}")
        paths = sorted(glob.glob(f"{sym_dir}/*_{MINUTE_TIMEFRAME}.csv"))
        if not paths:
            raise FileNotFoundError(
                f"no {MINUTE_TIMEFRAME} CSVs for {symbol} under {sym_dir}. TARGET_MODE='mm' "
                f"resolves the ladder and the exit minute-by-minute inside bar T+1 and "
                f"cannot fall back to 15m extremes. Sync them with "
                f"gauntlet/sync_klines.sh.")

        frames = []
        for csv_path in paths:
            d = pd.read_csv(csv_path, header=0)
            if "open_time_ms" not in d.columns:
                raise ValueError(
                    f"Missing required column 'open_time_ms' in {csv_path}")
            missing = [c for c in ("high", "low", "close") if c not in d.columns]
            if missing:
                raise ValueError(f"{csv_path} is missing columns {missing}")
            d["timestamp"] = pd.to_datetime(d["open_time_ms"], unit="ms", utc=True)
            frames.append(
                d.set_index("timestamp")[["high", "low", "close"]].astype(float))

        out = pd.concat(frames).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        out.index = out.index.tz_convert(None)
        return out

    def _compute_ladder_returns(
        self,
        df: pd.DataFrame,
        close_col: str,
        low_col: str,
        high_col: str,
        minute_bars: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Ladder-adjusted target columns for a single symbol's OHLC frame.

        The one implementation for every kline engine — OrbResearch,
        ScepterResearch and AetherResearch inherit it. Each leg is sized by its
        OWN adverse excursion via `agamotto.ladder.compute_ladder_multiplier`;
        neither leg gates the other. Mirrors the sizing inlined in
        `engineer_features`, which walks multi-timeframe `{prefix}_close` columns
        rather than a single symbol frame.

        Returns:
            DataFrame with return_long, return_short, return_long_raw,
            return_short_raw — indexed like `df`.
        """
        # TARGET_MODE='mm' prices what knull's MarketMaker actually earns — a
        # post-only entry ladder, a resting close at avg_cost +/- MM_PROFIT_AIM,
        # and a crossing exit when the passive rung does not fill in time —
        # instead of a close-to-close return scaled by rung count. 15m only; see
        # agamotto/mm_target.py. Absent TARGET_MODE keeps the ladder target, so
        # every pre-existing arm is untouched.
        if target_mode(self.config) == TARGET_MODE_MM:
            return compute_mm_target(df, close_col, low_col, high_col,
                                     self.config, minute_bars=minute_bars)

        ladder_long, ladder_short, step_bps = ladder_params(self.config)
        fee_rate = float(self.config["FEE"]) / 10000.0

        close = df[close_col]
        close_safe = close.replace(0, np.nan)
        price_return = close.pct_change(fill_method=None).shift(-1)
        low_next = df[low_col].shift(-1)
        high_next = df[high_col].shift(-1)

        size_long = compute_ladder_multiplier(
            close_safe, low_next, ladder_long, step_bps)
        # Mirror the short's adverse (upward) move about the close so the same
        # downward-measuring helper serves both legs.
        size_short = compute_ladder_multiplier(
            close_safe, 2.0 * close_safe - high_next, ladder_short, step_bps)

        fee_cost = fee_rate * 2.0
        return pd.DataFrame({
            "return_long": (price_return - fee_cost) * size_long,
            "return_short": (price_return + fee_cost) * size_short,
            "return_long_raw": price_return * size_long,
            "return_short_raw": price_return * size_short,
        }, index=df.index)

    def engineer_features(self) -> None:
        if self.raw is None:
            raise RuntimeError("Call load() before engineer_features().")

        df = self.raw
        engineered_frames = [df]
        
        # Both required — no magic-number defaults, no `get(K,X) or Y`
        # (CLAUDE.md; the FEE bug 2026-04-27). `step_size` was hardcoded to
        # 0.0001, which silently matched only configs with LADDER_BPS == 1.0.
        ladder_long, ladder_short, step_bps = ladder_params(self.config)
        fee_rate = float(self.config["FEE"]) / 10000.0
        dual_horizon = bool(self.config.get("DUAL_HORIZON"))
        # Resolved ONCE, not per symbol: this and `_compute_ladder_returns` are
        # two copies of the same arithmetic, and the design's whole reason for a
        # shared module is to stop them diverging. Both must switch together.
        mm_mode = target_mode(self.config) == TARGET_MODE_MM

        for col in df.columns:
            if col.endswith("_close"):
                base = col[:-6]
                close = df[col]
                open_col = f"{base}_open"
                high_col = f"{base}_high"
                low_col = f"{base}_low"

                open_series = df.get(open_col, close)
                high_series = df.get(high_col, close)
                low_series = df.get(low_col, close)

                price_range = (high_series - low_series).rename(f"{base}_price_range")
                price_range_pct = ((high_series - low_series) / (open_series + 1e-8)).rename(f"{base}_price_range_pct")
                price_range_pct_q50 = price_range_pct.rolling(700, min_periods=1).quantile(0.5).rename(f"{base}_price_range_pct_q50")
                # Trailing vol-quantile cutoffs — built exactly like
                # price_range_pct_q50 above (per SYMBOL, on the WIDE frame, so a
                # cutoff can never be computed across a symbol boundary), but
                # with min_periods=VOL_Q_WINDOW instead of 1: warmup must fail
                # CLOSED (see the VOL_Q_WINDOW comment). The first
                # VOL_Q_WINDOW-1 bars per symbol are NaN by construction and the
                # `>` comparison in research_filters makes that a non-firing bar.
                vol_quantile_cols = [
                    price_range_pct.rolling(
                        VOL_Q_WINDOW, min_periods=VOL_Q_WINDOW
                    ).quantile(level).rename(f"{base}_{name}")
                    for level, name in zip(VOL_Q_LEVELS, VOL_QUANTILE_FEATURES)
                ]
                open_close_diff = (close - open_series).rename(f"{base}_open_close_diff")
                open_close_pct = (open_close_diff / (open_series + 1e-8)).rename(f"{base}_open_close_pct")
                high_open_pct = ((high_series - open_series) / (open_series + 1e-8)).rename(f"{base}_high_open_pct")
                low_open_pct = ((low_series - open_series) / (open_series + 1e-8)).rename(f"{base}_low_open_pct")
                
                hist_return = close.pct_change(fill_method=None)
                price_return = hist_return.shift(-1)

                ret_lag1 = hist_return.shift(1).rename(f"{base}_ret_lag1")
                ret_lag2 = hist_return.shift(2).rename(f"{base}_ret_lag2")
                ret_lag3 = hist_return.shift(3).rename(f"{base}_ret_lag3")

                low_next = low_series.shift(-1)
                high_next = high_series.shift(-1)
                close_safe = close.replace(0, np.nan)

                # Per-leg ladder sizing (2026-08-06). Each leg is sized by its OWN
                # adverse excursion; neither gates the other. The previous
                # `size = min(long_layers, short_layers)` additionally required a
                # FAVOURABLE excursion, which nothing in the executor does — see
                # agamotto/ladder.py and tests/test_kline_ladder_sizing.py. It
                # zeroed exactly the dip-and-keep-falling bars, whose losses are
                # real and are realized live at the next non-BUY decision.
                size_long = compute_ladder_multiplier(
                    close_safe, low_next, ladder_long, step_bps)
                # Mirror the short's adverse move (upward) about the close so the
                # same downward-measuring helper applies to both legs.
                size_short = compute_ladder_multiplier(
                    close_safe, 2.0 * close_safe - high_next, ladder_short, step_bps)

                if mm_mode:
                    # Same four names, MM economics. The fee is absorbed per
                    # branch inside the target (taker pays MM_TAKER_FEE_BPS,
                    # maker pays nothing), so there is no separate fee-adjusted
                    # variant — `return_long` IS `return_long_raw` here.
                    mm_cols = compute_mm_target(
                        pd.DataFrame({"close": close, "low": low_series,
                                      "high": high_series}),
                        "close", "low", "high", self.config,
                        minute_bars=self._load_minute_bars(base))
                    price_return_long = mm_cols["return_long"].rename(
                        f"{base}_return_long")
                    price_return_short = mm_cols["return_short"].rename(
                        f"{base}_return_short")
                    price_return_long_raw = mm_cols["return_long_raw"].rename(
                        f"{base}_return_long_raw")
                    price_return_short_raw = mm_cols["return_short_raw"].rename(
                        f"{base}_return_short_raw")
                else:
                    fee_cost = fee_rate * 2.0
                    long_per_layer_return = price_return - fee_cost
                    price_return_long = (long_per_layer_return * size_long).rename(f"{base}_return_long")

                    short_raw_per_layer = price_return + fee_cost
                    price_return_short = (short_raw_per_layer * size_short).rename(f"{base}_return_short")

                    price_return_long_raw = (price_return * size_long).rename(f"{base}_return_long_raw")
                    price_return_short_raw = (price_return * size_short).rename(f"{base}_return_short_raw")

                if dual_horizon:
                    price_return_2bar = (close.shift(-2) / close_safe - 1)
                    low_min2 = pd.concat(
                        [low_series.shift(-1), low_series.shift(-2)], axis=1).min(axis=1)
                    high_max2 = pd.concat(
                        [high_series.shift(-1), high_series.shift(-2)], axis=1).max(axis=1)

                    # Same per-leg sizing as the 1-bar target, over the widened
                    # 2-bar fill window. Was `min(long_layers2, short_layers2)`.
                    size_long2 = compute_ladder_multiplier(
                        close_safe, low_min2, ladder_long, step_bps)
                    size_short2 = compute_ladder_multiplier(
                        close_safe, 2.0 * close_safe - high_max2, ladder_short, step_bps)

                    ret_2bar = price_return_2bar.rename(f"{base}_ret_2bar")
                    return_long_2bar = ((price_return_2bar - fee_cost) * size_long2).rename(f"{base}_return_long_2bar")
                    return_short_2bar = ((price_return_2bar + fee_cost) * size_short2).rename(f"{base}_return_short_2bar")
                    return_long_2bar_raw = (price_return_2bar * size_long2).rename(f"{base}_return_long_2bar_raw")
                    return_short_2bar_raw = (price_return_2bar * size_short2).rename(f"{base}_return_short_2bar_raw")

                return_dip = (low_next / close_safe - 1).rename(f"{base}_return_dip")
                return_rip = (high_next / close_safe - 1).rename(f"{base}_return_rip")

                price_return_combined = price_return.rename(f"{base}_return")
                
                ma_periods = self.config.get("MA_PERIODS", [7, 25, 99])
                if not isinstance(ma_periods, list) or len(ma_periods) != 3:
                    raise ValueError(f"MA_PERIODS must be a list of 3 integers, got: {ma_periods}")
                
                ma1_period, ma2_period, ma3_period = ma_periods
                ma1 = close.rolling(int(ma1_period), min_periods=1).mean().rename(f"{base}_ma{ma1_period}")
                ma2 = close.rolling(int(ma2_period), min_periods=1).mean().rename(f"{base}_ma{ma2_period}")
                ma3 = close.rolling(int(ma3_period), min_periods=1).mean().rename(f"{base}_ma{ma3_period}")

                volume_features = []
                
                if f"{base}_volume" in df.columns:
                    vol = df[f"{base}_volume"]
                    vol_ma = vol.rolling(7, min_periods=1).mean()
                    vol_ratio = (vol / (vol_ma + 1e-8)).rename(f"{base}_vol_ratio")
                    volume_features.append(vol_ratio)
                    vol_ret = vol.pct_change(fill_method=None)
                    volume_features.append(vol_ret.shift(1).rename(f"{base}_vol_ret_lag1"))
                    volume_features.append(vol_ret.shift(2).rename(f"{base}_vol_ret_lag2"))
                    volume_features.append(vol_ret.shift(3).rename(f"{base}_vol_ret_lag3"))

                if f"{base}_quote_volume" in df.columns:
                    quote_vol = df[f"{base}_quote_volume"]
                    quote_vol_ma = quote_vol.rolling(7, min_periods=1).mean()
                    quote_vol_ratio = (quote_vol / (quote_vol_ma + 1e-8)).rename(f"{base}_quote_vol_ratio")
                    volume_features.append(quote_vol_ratio)

                    taker_buy_col = f"{base}_taker_buy_quote_volume"
                    if taker_buy_col in df.columns:
                        taker_buy = df[taker_buy_col]
                        buy_pressure = (taker_buy / (quote_vol + 1e-8)).rename(f"{base}_buy_pressure")
                        volume_features.append(buy_pressure)
                    
                    trades_col = f"{base}_number_of_trades"
                    if trades_col in df.columns:
                        num_trades = df[trades_col]
                        trades_ma = num_trades.rolling(7, min_periods=1).mean()
                        trade_intensity = (num_trades / (trades_ma + 1e-8)).rename(f"{base}_trade_intensity")
                        volume_features.append(trade_intensity)

                ta_features = []
                try:
                    import talib
                    c_vals = close.values.astype(float)
                    h_vals = high_series.values.astype(float)
                    l_vals = low_series.values.astype(float)
                    
                    ta_features.append(pd.Series(talib.RSI(c_vals, timeperiod=14), index=df.index, name=f"{base}_rsi"))
                    ta_features.append(pd.Series(talib.RSI(c_vals, timeperiod=7), index=df.index, name=f"{base}_rsi_7"))
                    ta_features.append(pd.Series(talib.RSI(c_vals, timeperiod=28), index=df.index, name=f"{base}_rsi_28"))
                    macd, macdsignal, macdhist = talib.MACD(c_vals, fastperiod=12, slowperiod=26, signalperiod=9)
                    ta_features.append(pd.Series(macd, index=df.index, name=f"{base}_macd"))
                    ta_features.append(pd.Series(macdhist, index=df.index, name=f"{base}_macdhist"))
                    
                    slowk, slowd = talib.STOCH(h_vals, l_vals, c_vals, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
                    ta_features.append(pd.Series(slowk, index=df.index, name=f"{base}_stoch_k"))
                    ta_features.append(pd.Series(slowd, index=df.index, name=f"{base}_stoch_d"))
                    
                    ta_features.append(pd.Series(talib.CCI(h_vals, l_vals, c_vals, timeperiod=14), index=df.index, name=f"{base}_cci"))
                    ta_features.append(pd.Series(talib.ADX(h_vals, l_vals, c_vals, timeperiod=14), index=df.index, name=f"{base}_adx"))
                    ta_features.append(pd.Series(talib.DX(h_vals, l_vals, c_vals, timeperiod=14), index=df.index, name=f"{base}_dx"))
                    ta_features.append(pd.Series(talib.PLUS_DI(h_vals, l_vals, c_vals, timeperiod=14), index=df.index, name=f"{base}_plus_di"))
                    ta_features.append(pd.Series(talib.MINUS_DI(h_vals, l_vals, c_vals, timeperiod=14), index=df.index, name=f"{base}_minus_di"))
                    ta_features.append(pd.Series(talib.MOM(c_vals, timeperiod=10), index=df.index, name=f"{base}_mom"))
                    ta_features.append(pd.Series(talib.ROC(c_vals, timeperiod=10), index=df.index, name=f"{base}_roc"))
                    ta_features.append(pd.Series(talib.WILLR(h_vals, l_vals, c_vals, timeperiod=14), index=df.index, name=f"{base}_willr"))
                    ta_features.append(pd.Series(talib.CMO(c_vals, timeperiod=14), index=df.index, name=f"{base}_cmo"))
                    ta_features.append(pd.Series(talib.TRIX(c_vals, timeperiod=30), index=df.index, name=f"{base}_trix"))
                    ta_features.append(pd.Series(talib.ULTOSC(h_vals, l_vals, c_vals, timeperiod1=7, timeperiod2=14, timeperiod3=28), index=df.index, name=f"{base}_ultosc"))

                    fastk, fastd = talib.STOCHRSI(c_vals, timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
                    ta_features.append(pd.Series(fastk, index=df.index, name=f"{base}_stochrsi_k"))
                    ta_features.append(pd.Series(fastd, index=df.index, name=f"{base}_stochrsi_d"))

                    v_vals = df[f"{base}_volume"].values.astype(float)
                    obv_raw = pd.Series(talib.OBV(c_vals, v_vals), index=df.index)
                    ad_raw = pd.Series(talib.AD(h_vals, l_vals, c_vals, v_vals), index=df.index)
                    ta_features.append(obv_raw.diff(14).fillna(0.0).rename(f"{base}_obv"))
                    ta_features.append(ad_raw.diff(14).fillna(0.0).rename(f"{base}_ad"))
                    ta_features.append(pd.Series(talib.MFI(h_vals, l_vals, c_vals, v_vals, timeperiod=14), index=df.index, name=f"{base}_mfi"))
                    ta_features.append(pd.Series(talib.BOP(open_series.values.astype(float), h_vals, l_vals, c_vals), index=df.index, name=f"{base}_bop"))

                    ta_features.append(pd.Series(talib.ATR(h_vals, l_vals, c_vals, timeperiod=14), index=df.index, name=f"{base}_atr"))
                    ta_features.append(pd.Series(talib.NATR(h_vals, l_vals, c_vals, timeperiod=14), index=df.index, name=f"{base}_natr"))
                    parkinson_vol = np.sqrt(
                        1.0 / (4.0 * np.log(2)) * (np.log(high_series / low_series) ** 2)
                    ).rolling(14).mean().rename(f"{base}_parkinson_vol")
                    ta_features.append(parkinson_vol)
                    upper, middle, lower = talib.BBANDS(c_vals, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
                    ta_features.append(pd.Series(upper, index=df.index, name=f"{base}_bb_upper"))
                    ta_features.append(pd.Series(lower, index=df.index, name=f"{base}_bb_lower"))

                    ta_features.append(pd.Series(talib.SAR(h_vals, l_vals, acceleration=0.02, maximum=0.2), index=df.index, name=f"{base}_sar"))
                except Exception as e:
                    logger.warning(f"TA-Lib error for {base}: {e}")

                stats_window = int(self.config.get("STATS_WINDOW", 14))
                # `_acf_lag1` below is a (stats_window - 1)-wide rolling
                # correlation, so it needs at least 3 pairs to be meaningful and
                # `_kurt` needs 4 points. The retired per-window lambda returned
                # a constant 0.0 when the window held fewer than 4 points; rather
                # than silently turning that degenerate all-zero column into real
                # numbers, fail loud. All 775 production setting.json files use 14.
                if stats_window < 4:
                    raise ValueError(
                        "STATS_WINDOW must be >= 4 (rolling kurt/acf_lag1 are "
                        f"degenerate below that); got {stats_window}")
                rolling_stats = []
                rolling_stats.append(hist_return.rolling(window=stats_window).std().rename(f"{base}_std"))
                rolling_stats.append(hist_return.rolling(window=stats_window).skew().rename(f"{base}_skew"))
                rolling_stats.append(hist_return.rolling(window=stats_window).kurt().rename(f"{base}_kurt"))

                # Vectorized rolling lag-1 autocorrelation. `pd.Series.autocorr(lag=1)`
                # over a w-wide window is BY DEFINITION corr(x[:-1], x[1:]) — i.e. a
                # (w-1)-wide rolling correlation of the series against its lag-1 shift.
                # The retired form was
                #     .rolling(w).apply(lambda x: pd.Series(x).autocorr(1), raw=False)
                # which materialized two pd.Series per window (687 windows x 28 symbols
                # = ~19k throwaway objects per live cycle, ~0.78 s/symbol measured).
                # This form is ~500x faster and numerically identical — verified to
                # <= 1.1e-13 on real 15m klines and across NaN / inf / constant-run /
                # short-series edge cases by tests/test_acf_lag1_vectorized.py.
                #
                # min_periods parity is exact, not approximate: rolling(w) needs w
                # non-NaN values, rolling(w-1).corr needs w-1 non-NaN PAIRS, and a NaN
                # (or inf, which pandas coerces to NaN) at index k invalidates exactly
                # rows k..k+w-1 under both forms.
                rolling_stats.append(
                    hist_return.rolling(window=stats_window - 1)
                    .corr(hist_return.shift(1))
                    .fillna(0.0)
                    .rename(f"{base}_acf_lag1")
                )

                engineered_frames.extend([
                    price_range,
                    price_range_pct,
                    price_range_pct_q50,
                    open_close_diff,
                    open_close_pct,
                    high_open_pct,
                    low_open_pct,
                    price_return_combined,
                    price_return_long,
                    price_return_short,
                    price_return_long_raw,
                    price_return_short_raw,
                    return_dip,
                    return_rip,
                    ret_lag1,
                    ret_lag2,
                    ret_lag3,
                    ma1.rename(f"{base}_mvg1"),
                    ma2.rename(f"{base}_mvg2"),
                    ma3.rename(f"{base}_mvg3"),
                ] + vol_quantile_cols + volume_features + ta_features + rolling_stats)

                # Scale-free forms of the seven raw price/volume levels. Pooling
                # symbols makes a price-unit column act as a symbol ID; measured
                # on 10 symbols x 360k rows of 15m, the raw levels are
                # SIGN-INVERTED by pooling (sar reads +0.0042 pooled vs -0.0388
                # within-symbol, 3/10 symbols disagreeing) while every transform
                # scores 0/10 sign flips. See agamotto/features_scalefree.py.
                #
                # The raw columns STAY — research_filters gates regimes on them
                # (`close > sar` :330, `close < bb_lower` :327, `macdhist > 0`
                # :213). They are kept out of the MODEL by
                # gauntlet/rolling_predict_returns.select_feature_columns.
                _ta_by_name = {s.name: s for s in ta_features}
                _src = {f"{base}_close": close, f"{base}_volume": df[f"{base}_volume"]}
                for _n in ("sar", "bb_upper", "bb_lower", "macd", "macdhist", "obv", "ad"):
                    if f"{base}_{_n}" in _ta_by_name:
                        _src[f"{base}_{_n}"] = _ta_by_name[f"{base}_{_n}"]
                _need = [f"{base}_{n}" for n in
                         ("close", "sar", "bb_upper", "bb_lower", "macd",
                          "macdhist", "obv", "ad", "volume")]
                if all(k in _src for k in _need):
                    engineered_frames.extend([
                        s for _, s in scale_free_levels(
                            pd.DataFrame(_src),
                            prefix=f"{base}_",
                            # agamotto stores obv/ad ALREADY differenced
                            # (obv_raw.diff(14) above), so normalise only —
                            # differencing again is a silent second difference.
                            obv_is_cumulative=False,
                        ).items()
                    ])
                else:
                    # WHY safe: the TA-Lib block above swallows its own errors
                    # with a warning, so these sources are legitimately absent on
                    # a TA-Lib failure. Raising here would turn a degraded run
                    # into a dead one. The warning names the gap explicitly.
                    logger.warning(
                        "%s: skipping scale-free level features, missing %s",
                        base, [k for k in _need if k not in _src])

                if dual_horizon:
                    engineered_frames.extend([
                        ret_2bar,
                        return_long_2bar,
                        return_short_2bar,
                        return_long_2bar_raw,
                        return_short_2bar_raw,
                    ])

        feature_df = pd.concat(engineered_frames, axis=1)

        feature_df["year"] = feature_df.index.year
        feature_df["month"] = feature_df.index.month

        tf_secs = _timeframe_to_seconds(self.config.get("TIME_UNIT", "1d"))
        feature_df["close_timestamp"] = (
            feature_df.index + pd.Timedelta(seconds=tf_secs)
        )

        self.features = feature_df

    def create(self) -> str:
        if self.features is None:
             self.engineer_features()
             
        if self.features is not None and not self.features.empty:
             self.features = self.features.iloc[:-1]

        self.verticalize()

        version = self.config.get("VERSION")
        if not version:
             raise ValueError("VERSION missing from config")
             
        if "OUTPUT_DIR" in self.config:
            out_dir = self.config["OUTPUT_DIR"]
        else:
            out_dir = os.path.join("gauntlet", f"pred_{version}")
        
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(self.home_root, out_dir)
            
        os.makedirs(out_dir, exist_ok=True)
        
        # WRITE_VERTICAL_FEATURES_CSV — genuinely optional, ABSENT MEANS TRUE.
        #
        # Not a magic default for a required key (CLAUDE.md): this flag can only
        # skip an ARTIFACT, never change a computed value — the filter parquets,
        # the regime masks and every downstream number are bit-identical either
        # way. Defaulting it False instead would silently break the marvel-explore
        # panel workflows, whose panel path IS this file (gauntlet's
        # regime_discover.py defaults to
        # ~/marvel-explore/<algo>_perwindow_<tf>_<market>/vertical_features.csv),
        # and every one of the ~775 existing setting.json files predates the key.
        # Same shape as the optional MAX_ROWS / WORKERS reads in
        # marvel's gauntlet/regime_ic_rank.py:114-119.
        #
        # Why it exists: nothing in the gauntlet pipeline itself reads this file,
        # and it is expensive. Measured on shield 2026-08-07 for
        # pred_orb.base.15m_1 — 4,996,355 rows x 324 cols = 18.9 GB at ~11 MB/s to
        # the NAS-backed local mirror, i.e. ~26 min of a 9h29m Step 1.
        write_vertical_csv = self.config.get("WRITE_VERTICAL_FEATURES_CSV", True)
        if (write_vertical_csv
                and hasattr(self, 'vertical_features')
                and self.vertical_features is not None):
            v_out_path = os.path.join(out_dir, "vertical_features.csv")
            self.vertical_features.to_csv(v_out_path, index=False)
        elif not write_vertical_csv:
            logger.info(
                "WRITE_VERTICAL_FEATURES_CSV=false — skipping vertical_features.csv "
                "(filter parquets are unaffected)"
            )

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

        regime_stack = [r for r in regime_stack if not str(r.get("regime", "")).startswith("__")]
        logger.info(f"Loaded {len(regime_stack)} regimes from {stack_path}")
        for regime in regime_stack:
            self.filter_signals(regime, save=True, out_dir=out_dir)

        return out_dir

    def verticalize(self, df: pd.DataFrame | None = None) -> None:
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
            
            base_rename_map = {
                f"{prefix}_close": "close",
                f"{prefix}_mvg1": "mvg1",
                f"{prefix}_mvg2": "mvg2",
                f"{prefix}_mvg3": "mvg3",
                f"{prefix}_price_range": "price_range",
                f"{prefix}_price_range_pct": "price_range_pct",
                f"{prefix}_price_range_pct_q50": "price_range_pct_q50",
                # THREE EXPLICIT entries, not a comprehension over
                # VOL_QUANTILE_FEATURES: the drop tripwire below (:827-837) is
                # scoped to SCALE_FREE_FEATURES only, so a rename entry missing
                # here drops the column SILENTLY (valid_cols iterates the map,
                # not the frame) and every high_vol_q* regime then raises on a
                # column the panel quietly lacks.
                f"{prefix}_price_range_pct_q80": "price_range_pct_q80",
                f"{prefix}_price_range_pct_q90": "price_range_pct_q90",
                f"{prefix}_price_range_pct_q95": "price_range_pct_q95",
                f"{prefix}_open_close_diff": "open_close_diff",
                f"{prefix}_open_close_pct": "open_close_pct",
                f"{prefix}_high_open_pct": "high_open_pct",
                f"{prefix}_low_open_pct": "low_open_pct",
                f"{prefix}_ret_lag1": "ret_lag1",
                f"{prefix}_ret_lag2": "ret_lag2",
                f"{prefix}_ret_lag3": "ret_lag3",
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
                f"{prefix}_std": "std",
                f"{prefix}_skew": "skew",
                f"{prefix}_kurt": "kurt",
                f"{prefix}_acf_lag1": "acf_lag1",
                f"{prefix}_quote_vol_ratio": "quote_vol_ratio",
                f"{prefix}_vol_ratio": "vol_ratio",
                f"{prefix}_buy_pressure": "buy_pressure",
                f"{prefix}_trade_intensity": "trade_intensity",
                f"{prefix}_vol_ret_lag1": "vol_ret_lag1",
                f"{prefix}_vol_ret_lag2": "vol_ret_lag2",
                f"{prefix}_vol_ret_lag3": "vol_ret_lag3",
                f"{prefix}_return_dip": "return_dip",
                f"{prefix}_return_rip": "return_rip",
            }

            # engineer_features() computes the scale-free level replacements
            # (SCALE_FREE_FEATURES) into the wide frame; they must be carried
            # through to the vertical panel or the trainer never sees them.
            # Driven off the canonical list, never a second hardcoded copy —
            # the two drifting apart is exactly the defect this replaces.
            for _sf_name in SCALE_FREE_FEATURES:
                base_rename_map[f"{prefix}_{_sf_name}"] = _sf_name

            long_return_col = f"{prefix}_return_long"
            short_return_col = f"{prefix}_return_short"
            long_return_raw_col = f"{prefix}_return_long_raw"
            short_return_raw_col = f"{prefix}_return_short_raw"
            base_return_col = f"{prefix}_return"

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

            for suffix in ("ret_2bar", "return_long_2bar", "return_short_2bar",
                           "return_long_2bar_raw", "return_short_2bar_raw"):
                col2 = f"{prefix}_{suffix}"
                if col2 in df.columns:
                    rename_map[col2] = suffix

            valid_cols = [c for c in rename_map if c in df.columns]

            # Tripwire for the defect this loop replaces: a feature computed by
            # engineer_features but absent from rename_map was dropped here
            # SILENTLY (valid_cols iterates the map, not the frame), so the
            # trainer saw a panel that quietly lacked it. Scoped to the
            # scale-free names because plenty of {prefix}_* columns — raw OHLCV,
            # TA intermediates — are dropped legitimately. No silent fallback:
            # a twin present in the frame but missing from the map raises.
            _dropped_sf = [
                f"{prefix}_{n}" for n in SCALE_FREE_FEATURES
                if f"{prefix}_{n}" in df.columns and f"{prefix}_{n}" not in valid_cols
            ]
            if _dropped_sf:
                raise KeyError(
                    f"verticalize: {sym} — engineer_features produced "
                    f"{_dropped_sf} but rename_map does not carry them, so they "
                    f"would be dropped from the vertical panel without warning. "
                    f"Add them to base_rename_map."
                )

            subset = df.loc[:, valid_cols + ["year", "month"]].copy()
            subset.rename(columns={orig: rename_map[orig] for orig in valid_cols}, inplace=True)
            
            subset["symbol"] = sym
            subset["timestamp"] = subset.index
            
            frames.append(subset)
            
        if frames:
            self.vertical_features = pd.concat(frames, axis=0, ignore_index=True)
        else:
            self.vertical_features = pd.DataFrame()

    def _apply_filter_mask(self, df: pd.DataFrame, filter_name: str | list, position: str) -> pd.Series:
        return apply_filter_mask(
            df,
            filter_name,
            position,
            strict_filters=getattr(self, '_strict_filters', True),
            allowed_positions_fn=type(self).allowed_positions,
        )

    def filter_signals(self, regime: dict, limit_timestamp: Optional[pd.Timestamp] = None, save: bool = False, out_dir: str = None) -> pd.DataFrame:
        if not hasattr(self, 'vertical_features') or self.vertical_features is None:
             raise RuntimeError("Call verticalize() before filter_signals().")

        features_df = self.vertical_features

        if "regime" not in regime:
            raise KeyError("regime row missing required 'regime' key (no baseline default)")
        regime_name = regime["regime"]
        position = regime.get("position", "long")

        regime_name_str = regime_name
        if isinstance(regime_name, list):
            regime_name_str = "_and_".join(p for p in regime_name if p not in ("|", "&"))

        mask = self._apply_filter_mask(features_df, regime_name, position)

        filtered_subset = features_df[mask].copy()
        if filtered_subset.empty:
            return pd.DataFrame()

        filtered_subset["position"] = position
        filtered_subset["regime"] = regime_name_str
        
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
            clean_regime = regime_name_str.replace("_long", "").replace("_short", "")
            safe_name = f"{clean_regime}_{position}"
            safe_name = "".join([c if c.isalnum() or c in ['_'] else '_' for c in safe_name])
            
            save_dir = os.path.join(out_dir, "filter")
            os.makedirs(save_dir, exist_ok=True)
            
            save_path = os.path.join(save_dir, f"filter_{safe_name}.parquet")
            try:
                _to_save = filtered_subset.rename(
                    columns=_obf().encode_columns(filtered_subset.columns))
                _f64_to_f32 = {
                    c: "float32" for c in _to_save.columns
                    if _to_save[c].dtype == "float64"
                }
                _to_save.astype(_f64_to_f32).to_parquet(
                    save_path, index=False, compression="zstd",
                )
            except Exception as e:
                logger.error(f"Failed to save filtered signals for {safe_name}: {e}")

        return filtered_subset
