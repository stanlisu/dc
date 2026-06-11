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

# Use relative import for internal module
try:
    from .lib_binance import fetch_futures_klines, klines_to_dataframe
except ImportError:
    # Fallback to assume it's in the path or installed
    try:
        from lib_binance import fetch_futures_klines, klines_to_dataframe
    except ImportError:
        pass

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


class AgamottoResearch:
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

    @classmethod
    def allowed_positions(cls, filter_name: str) -> list:
        """Return ['long'], ['short'], or ['long','short'] for a regime name.

        Splits compound names on '_and_' and checks each component.
        Contradictory combos (long-only + short-only) return [] (skip).
        """
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

    def engineer_features(self) -> None:
        if self.raw is None:
            raise RuntimeError("Call load() before engineer_features().")

        # WHY: drop the self.raw.copy() — the loop below only reads df[col] to
        # build new Series; it never mutates df in place. The duplicate cost
        # ~one wide multi-symbol OHLCV frame (e.g. 15m × 29 syms × 3.5y =
        # ~300 MB) during engineer_features, contributing to the 2026-05-22
        # OOM peak. self.raw is freed by the caller (OrbResearch.engineer_features
        # sets inst.raw = None after) so retaining the reference here is safe.
        df = self.raw
        engineered_frames = [df]
        
        # Get ladder config for position sizing
        ladder = self.config.get("LADDER", 1)  # Default to 1 if not specified

        ladder = int(self.config.get("LADDER", 1) or 0)
        # FEE is required (no fallback): a missing key must fail loudly so
        # misconfiguration cannot silently inject a phantom fee into the target.
        # The historical `or 0.0` collapsed any falsy FEE (incl. 0) to the
        # default — only safe by coincidence here, banned per CLAUDE.md.
        fee_rate = float(self.config["FEE"]) / 10000.0
        step_size = 0.0001
        # Dual-horizon: when enabled, also emit a 2-bar-horizon target set
        # (ret_2bar + laddered/fee return_{long,short}_2bar) so a second model
        # can be trained as a same-direction confirmation gate. Off by default —
        # only DH experiments set DUAL_HORIZON, so base experiments are unaffected.
        dual_horizon = bool(self.config.get("DUAL_HORIZON"))

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
                open_close_diff = (close - open_series).rename(f"{base}_open_close_diff")
                open_close_pct = (open_close_diff / (open_series + 1e-8)).rename(f"{base}_open_close_pct")
                high_open_pct = ((high_series - open_series) / (open_series + 1e-8)).rename(f"{base}_high_open_pct")
                low_open_pct = ((low_series - open_series) / (open_series + 1e-8)).rename(f"{base}_low_open_pct")
                
                # Calculate historical return (for features) and target return (shifted)
                hist_return = close.pct_change(fill_method=None)
                price_return = hist_return.shift(-1)

                # Lagged returns (past returns as features)
                ret_lag1 = hist_return.shift(1).rename(f"{base}_ret_lag1")
                ret_lag2 = hist_return.shift(2).rename(f"{base}_ret_lag2")
                ret_lag3 = hist_return.shift(3).rename(f"{base}_ret_lag3")

                # Apply ladder multiplier based on next candle's low/high
                # Get next period's low and high (t+1)
                low_next = low_series.shift(-1)
                high_next = high_series.shift(-1)
                close_safe = close.replace(0, np.nan)

                distance_long = ((close_safe - low_next) / close_safe).replace([np.inf, -np.inf], np.nan)
                long_layers = np.floor(distance_long / step_size)
                long_layers = long_layers.clip(lower=0, upper=ladder)
                long_layers = long_layers.fillna(0).astype(int)
                # Total layers = 1 initial + N ladders
                total_long_layers = 1 + long_layers

                distance_short = ((high_next - close_safe) / close_safe).replace([np.inf, -np.inf], np.nan)
                short_layers = np.floor(distance_short / step_size)
                short_layers = short_layers.clip(lower=0, upper=ladder)
                short_layers = short_layers.fillna(0).astype(int)
                # Total layers = 1 initial + N ladders
                total_short_layers = 1 + short_layers

                fee_cost = (fee_rate * 2.0) if fee_rate else 0.0
                long_per_layer_return = price_return - fee_cost
                price_return_long = (long_per_layer_return * total_long_layers).rename(f"{base}_return_long")

                short_raw_per_layer = price_return + fee_cost
                # (price_return + fee_cost) * layers, then short_subset flips it to -(...)
                price_return_short = (short_raw_per_layer * total_short_layers).rename(f"{base}_return_short")

                # Raw returns (no fee) for downstream Sharpe calculation
                price_return_long_raw = (price_return * total_long_layers).rename(f"{base}_return_long_raw")
                price_return_short_raw = (price_return * total_short_layers).rename(f"{base}_return_short_raw")

                # Dual-horizon 2-bar target set — mirror of the 1-bar block above
                # over a 2-bar forward hold. ret_2bar is the plain cumulative
                # return the 2-bar confirmation model trains on (analogous to the
                # 1-bar `ret`); return_{long,short}_2bar are the laddered+fee
                # PnL analogs (analogous to return_long/return_short). Ladder fill
                # uses the worst price over the next two bars (min low / max high).
                if dual_horizon:
                    price_return_2bar = (close.shift(-2) / close_safe - 1)
                    low_min2 = pd.concat(
                        [low_series.shift(-1), low_series.shift(-2)], axis=1).min(axis=1)
                    high_max2 = pd.concat(
                        [high_series.shift(-1), high_series.shift(-2)], axis=1).max(axis=1)

                    distance_long2 = ((close_safe - low_min2) / close_safe).replace([np.inf, -np.inf], np.nan)
                    long_layers2 = np.floor(distance_long2 / step_size)
                    long_layers2 = long_layers2.clip(lower=0, upper=ladder).fillna(0).astype(int)
                    total_long_layers2 = 1 + long_layers2

                    distance_short2 = ((high_max2 - close_safe) / close_safe).replace([np.inf, -np.inf], np.nan)
                    short_layers2 = np.floor(distance_short2 / step_size)
                    short_layers2 = short_layers2.clip(lower=0, upper=ladder).fillna(0).astype(int)
                    total_short_layers2 = 1 + short_layers2

                    ret_2bar = price_return_2bar.rename(f"{base}_ret_2bar")
                    return_long_2bar = ((price_return_2bar - fee_cost) * total_long_layers2).rename(f"{base}_return_long_2bar")
                    return_short_2bar = ((price_return_2bar + fee_cost) * total_short_layers2).rename(f"{base}_return_short_2bar")
                    return_long_2bar_raw = (price_return_2bar * total_long_layers2).rename(f"{base}_return_long_2bar_raw")
                    return_short_2bar_raw = (price_return_2bar * total_short_layers2).rename(f"{base}_return_short_2bar_raw")

                # Dip/rip target columns for compound classification label (Vomir)
                # return_dip = low[T+1]/close[T] - 1  (how far price dips next bar)
                # return_rip = high[T+1]/close[T] - 1  (how far price rips next bar)
                return_dip = (low_next / close_safe - 1).rename(f"{base}_return_dip")
                return_rip = (high_next / close_safe - 1).rename(f"{base}_return_rip")

                # Store both versions (will be used based on position type during filtering)
                price_return_combined = price_return.rename(f"{base}_return")
                
                # Get MA periods from config, default to [7, 25, 99]
                ma_periods = self.config.get("MA_PERIODS", [7, 25, 99])
                if not isinstance(ma_periods, list) or len(ma_periods) != 3:
                    raise ValueError(f"MA_PERIODS must be a list of 3 integers, got: {ma_periods}")
                
                ma1_period, ma2_period, ma3_period = ma_periods
                ma1 = close.rolling(int(ma1_period), min_periods=1).mean().rename(f"{base}_ma{ma1_period}")
                ma2 = close.rolling(int(ma2_period), min_periods=1).mean().rename(f"{base}_ma{ma2_period}")
                ma3 = close.rolling(int(ma3_period), min_periods=1).mean().rename(f"{base}_ma{ma3_period}")

                # Volume-based features
                volume_features = []
                
                # 1. Base Volume
                if f"{base}_volume" in df.columns:
                    vol = df[f"{base}_volume"]
                    vol_ma = vol.rolling(7, min_periods=1).mean()
                    vol_ratio = (vol / (vol_ma + 1e-8)).rename(f"{base}_vol_ratio")
                    volume_features.append(vol_ratio)
                    # Volume return lags — captures volume momentum (surge/fade patterns)
                    vol_ret = vol.pct_change(fill_method=None)
                    volume_features.append(vol_ret.shift(1).rename(f"{base}_vol_ret_lag1"))
                    volume_features.append(vol_ret.shift(2).rename(f"{base}_vol_ret_lag2"))
                    volume_features.append(vol_ret.shift(3).rename(f"{base}_vol_ret_lag3"))

                # 2. Quote Volume
                if f"{base}_quote_volume" in df.columns:
                    quote_vol = df[f"{base}_quote_volume"]
                    quote_vol_ma = quote_vol.rolling(7, min_periods=1).mean()
                    quote_vol_ratio = (quote_vol / (quote_vol_ma + 1e-8)).rename(f"{base}_quote_vol_ratio")
                    volume_features.append(quote_vol_ratio)

                    # Buy pressure (taker buy / total volume)
                    taker_buy_col = f"{base}_taker_buy_quote_volume"
                    if taker_buy_col in df.columns:
                        taker_buy = df[taker_buy_col]
                        buy_pressure = (taker_buy / (quote_vol + 1e-8)).rename(f"{base}_buy_pressure")
                        volume_features.append(buy_pressure)
                    
                    # Trade intensity (trades / 7-period MA trades)
                    trades_col = f"{base}_number_of_trades"
                    if trades_col in df.columns:
                        num_trades = df[trades_col]
                        trades_ma = num_trades.rolling(7, min_periods=1).mean()
                        trade_intensity = (num_trades / (trades_ma + 1e-8)).rename(f"{base}_trade_intensity")
                        volume_features.append(trade_intensity)

                # TA-Lib Indicators
                ta_features = []
                try:
                    import talib
                    # Convert to double precision for TA-Lib if needed
                    c_vals = close.values.astype(float)
                    h_vals = high_series.values.astype(float)
                    l_vals = low_series.values.astype(float)
                    
                    # Momentum Indicators
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

                    # Volume Indicators
                    # OBV and AD are cumulative — use diff(14) instead of raw values
                    # so the result is history-length independent (net change over 14 periods).
                    v_vals = df[f"{base}_volume"].values.astype(float)
                    obv_raw = pd.Series(talib.OBV(c_vals, v_vals), index=df.index)
                    ad_raw = pd.Series(talib.AD(h_vals, l_vals, c_vals, v_vals), index=df.index)
                    ta_features.append(obv_raw.diff(14).fillna(0.0).rename(f"{base}_obv"))
                    ta_features.append(ad_raw.diff(14).fillna(0.0).rename(f"{base}_ad"))
                    ta_features.append(pd.Series(talib.MFI(h_vals, l_vals, c_vals, v_vals, timeperiod=14), index=df.index, name=f"{base}_mfi"))
                    ta_features.append(pd.Series(talib.BOP(open_series.values.astype(float), h_vals, l_vals, c_vals), index=df.index, name=f"{base}_bop"))

                    # Volatility Indicators
                    ta_features.append(pd.Series(talib.ATR(h_vals, l_vals, c_vals, timeperiod=14), index=df.index, name=f"{base}_atr"))
                    ta_features.append(pd.Series(talib.NATR(h_vals, l_vals, c_vals, timeperiod=14), index=df.index, name=f"{base}_natr"))
                    parkinson_vol = np.sqrt(
                        1.0 / (4.0 * np.log(2)) * (np.log(high_series / low_series) ** 2)
                    ).rolling(14).mean().rename(f"{base}_parkinson_vol")
                    ta_features.append(parkinson_vol)
                    upper, middle, lower = talib.BBANDS(c_vals, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
                    ta_features.append(pd.Series(upper, index=df.index, name=f"{base}_bb_upper"))
                    ta_features.append(pd.Series(lower, index=df.index, name=f"{base}_bb_lower"))

                    # Trend Indicators
                    ta_features.append(pd.Series(talib.SAR(h_vals, l_vals, acceleration=0.02, maximum=0.2), index=df.index, name=f"{base}_sar"))
                except Exception as e:
                    logger.warning(f"TA-Lib error for {base}: {e}")

                # Statistical Features (Use historical return, not shifted target return)
                stats_window = int(self.config.get("STATS_WINDOW", 14))
                rolling_stats = []
                rolling_stats.append(hist_return.rolling(window=stats_window).std().rename(f"{base}_std"))
                rolling_stats.append(hist_return.rolling(window=stats_window).skew().rename(f"{base}_skew"))
                rolling_stats.append(hist_return.rolling(window=stats_window).kurt().rename(f"{base}_kurt"))

                # Rolling autocorrelation of returns at lag 1 — captures momentum/mean-reversion regime.
                # Positive = momentum (returns persist), Negative = mean-reversion (returns reverse).
                rolling_stats.append(
                    hist_return.rolling(window=stats_window).apply(
                        lambda x: float(pd.Series(x).autocorr(lag=1)) if len(x) >= 4 else 0.0,
                        raw=False,
                    ).fillna(0.0).rename(f"{base}_acf_lag1")
                )

                engineered_frames.extend([
                    price_range,
                    price_range_pct,
                    price_range_pct_q50,
                    open_close_diff,
                    open_close_pct,
                    high_open_pct,
                    low_open_pct,
                    price_return_combined,  # Original return (for compatibility)
                    price_return_long,  # Long-specific laddered return
                    price_return_short,  # Short-specific laddered return
                    price_return_long_raw,  # Raw long return (no fee)
                    price_return_short_raw,  # Raw short return (no fee)
                    return_dip,  # next bar low/close - 1 (compound label)
                    return_rip,  # next bar high/close - 1 (compound label)
                    ret_lag1,
                    ret_lag2,
                    ret_lag3,
                    ma1.rename(f"{base}_mvg1"),
                    ma2.rename(f"{base}_mvg2"),
                    ma3.rename(f"{base}_mvg3"),
                ] + volume_features + ta_features + rolling_stats)

                if dual_horizon:
                    engineered_frames.extend([
                        ret_2bar,
                        return_long_2bar,
                        return_short_2bar,
                        return_long_2bar_raw,
                        return_short_2bar_raw,
                    ])

        feature_df = pd.concat(engineered_frames, axis=1)

        # WHY: pd.concat with axis=1 in pandas ≥2.0 returns an owning DataFrame
        # (not a view), so the immediately-following column assignments
        # (feature_df["year"] = ..., ["month"] = ..., ["close_timestamp"] = ...)
        # do not need a defensive .copy() to avoid SettingWithCopyWarning.
        # Dropping the copy saves a full feature-matrix duplication
        # (~1.5 GB peak for 15m × 29 syms × 3.5y × ~55 engineered cols),
        # which was a major contributor to the 2026-05-22 OOM.
        feature_df["year"] = feature_df.index.year
        feature_df["month"] = feature_df.index.month

        # close_timestamp: when this bar closes = open time + TF duration.
        # Used downstream by OrbResearch._align_timeframes to enforce causal
        # alignment — a higher-TF bar is only used once its close_timestamp
        # is <= the decision time (base-TF bar open).
        from .utils import _timeframe_to_seconds
        tf_secs = _timeframe_to_seconds(self.config.get("TIME_UNIT", "1d"))
        feature_df["close_timestamp"] = (
            feature_df.index + pd.Timedelta(seconds=tf_secs)
        )

        self.features = feature_df

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
        
        # Ensure absolute path using home_root if relative
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(self.home_root, out_dir)
            
        os.makedirs(out_dir, exist_ok=True)
        
        # Save vertical features
        if hasattr(self, 'vertical_features') and self.vertical_features is not None:
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

    def _apply_filter_mask(self, df: pd.DataFrame, filter_name: str | list, position: str) -> pd.Series:
        """
        Applies a named filter or list of filters to the DataFrame.
        Returns a boolean Series (mask) where True = Keep signal.
        """
        # --- List Support (Recursive) ---
        if isinstance(filter_name, list):
            mask = None
            current_op = None
            for item in filter_name:
                item = str(item).strip()
                if item in ["|", "&"]:
                    current_op = item
                else:
                    sub_mask = self._apply_filter_mask(df, item, position)
                    if mask is None:
                        mask = sub_mask
                    else:
                        if current_op == "|":
                            mask = mask | sub_mask
                        elif current_op == "&":
                            mask = mask & sub_mask
                        else:
                            # Default to AND
                            mask = mask & sub_mask
            return mask if mask is not None else pd.Series(True, index=df.index)

        # --- Base Filters ---
        if isinstance(filter_name, str):
            filter_name = filter_name.lower().strip()

            # Skip metadata rows (e.g. __summary__ from filter_regime_stacks.py)
            if filter_name.startswith("__"):
                return pd.Series(True, index=df.index)

            # Support complex strings like "filterA_and_filterB" or "filterA_or_filterB"
            if "_and_" in filter_name:
                parts = filter_name.split("_and_")
                mask = None
                for part in parts:
                    sub_mask = self._apply_filter_mask(df, part.strip(), position)
                    mask = sub_mask if mask is None else (mask & sub_mask)
                return mask if mask is not None else pd.Series(True, index=df.index)
            
            if "_or_" in filter_name:
                parts = filter_name.split("_or_")
                mask = None
                for part in parts:
                    sub_mask = self._apply_filter_mask(df, part.strip(), position)
                    mask = sub_mask if mask is None else (mask | sub_mask)
                return mask if mask is not None else pd.Series(True, index=df.index)

        if df.empty:
            return pd.Series(dtype=bool)

        # Strip suffixes from filter name
        if isinstance(filter_name, str):
            filter_name = filter_name.replace("_long", "").replace("_short", "")
            # Strip TF prefix from ORB regime names (e.g. "15m_stoch_bullish" → "stoch_bullish")
            filter_name = re.sub(r'^(?:15m|1h|4h|1d)_', '', filter_name)

        # Enforce LONG_ONLY / SHORT_ONLY constraints via allowed_positions()
        _allowed = type(self).allowed_positions(filter_name) if isinstance(filter_name, str) else None
        if _allowed and position not in _allowed:
            return pd.Series(False, index=df.index)

        # Ensure we have common required columns
        required_base = ["close", "mvg1", "mvg2"]
        if not all(col in df.columns for col in required_base):
            logger.warning(f"Filter {filter_name} skipped: missing required columns {required_base}")
            return pd.Series(True, index=df.index)

        # Common trend conditions
        up_trend = (df["close"] > df["mvg1"]) & (df["mvg1"] > df["mvg2"])
        down_trend = (df["close"] < df["mvg1"]) & (df["mvg1"] < df["mvg2"])
        
        if position == "long":
            if filter_name == "baseline" or filter_name == "trend_aligned":
                return up_trend
            if filter_name == "strong_trend":
                return up_trend & (df["mvg2"] > df["mvg3"]) if "mvg3" in df.columns else up_trend
            if filter_name == "ma_momentum":
                return (df["mvg1"] > df["mvg2"]) & (df["mvg1"] > df["mvg3"]) if "mvg3" in df.columns else (df["mvg1"] > df["mvg2"])
            if filter_name == "above_all_mas":
                mask = (df["close"] > df["mvg1"]) & (df["close"] > df["mvg2"])
                if "mvg3" in df.columns:
                    mask &= (df["close"] > df["mvg3"])
                return mask
            if filter_name == "low_vol":
                q50 = df["price_range_pct_q50"] if "price_range_pct_q50" in df.columns else df["price_range_pct"].rolling(700, min_periods=1).quantile(0.5)
                return df["price_range_pct"] < q50
            if filter_name == "high_vol":
                q50 = df["price_range_pct_q50"] if "price_range_pct_q50" in df.columns else df["price_range_pct"].rolling(700, min_periods=1).quantile(0.5)
                return df["price_range_pct"] > q50
            if filter_name == "strong_candle":
                return (df["open_close_pct"] > 0.005)
            if filter_name == "near_ma":
                return ((df["close"] - df["mvg1"]) / df["mvg1"] < 0.02)

            # TA-Lib based
            if filter_name == "rsi_oversold":
                return df["rsi"] < 30 if "rsi" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "macd_bullish":
                return df["macdhist"] > 0 if "macdhist" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "stoch_bullish":
                return (df["stoch_k"] > df["stoch_d"]) if "stoch_k" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "cci_reversal":
                return df["cci"] > 100 if "cci" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "adx_trend":
                return (df["adx"] > 25) if "adx" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "bb_rebound":
                return df["close"] < df["bb_lower"] if "bb_lower" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "mom_positive":
                return df["mom"] > 0 if "mom" in df.columns else pd.Series(True, index=df.index)

            # Volume-based (priority to quote_vol_ratio if available, else vol_ratio)
            v_ratio = df.get("quote_vol_ratio", df.get("vol_ratio", 1.0))

            if filter_name == "low_volume": return (v_ratio < 1.0)
            if filter_name == "high_volume": return (v_ratio > 1.0)
            if filter_name == "vol_breakout": return (v_ratio > 2.0)

            # New TA-lab filters
            if filter_name == "buy_pressure":
                return df["buy_pressure"] > 0.55 if "buy_pressure" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "mfi_oversold":
                return df["mfi"] < 30 if "mfi" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "sar_aligned":
                return df["close"] > df["sar"] if "sar" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "bop_bullish":
                return df["bop"] > 0.1 if "bop" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "roc_positive":
                return df["roc"] > 0 if "roc" in df.columns else pd.Series(True, index=df.index)
        else:  # short
            if filter_name == "baseline" or filter_name == "trend_aligned":
                return down_trend
            if filter_name == "strong_trend":
                return down_trend & (df["mvg2"] < df["mvg3"]) if "mvg3" in df.columns else down_trend
            if filter_name == "ma_momentum":
                return (df["mvg1"] < df["mvg2"]) & (df["mvg1"] < df["mvg3"]) if "mvg3" in df.columns else (df["mvg1"] < df["mvg2"])
            if filter_name == "above_all_mas":
                mask = (df["close"] < df["mvg1"]) & (df["close"] < df["mvg2"])
                if "mvg3" in df.columns:
                    mask &= (df["close"] < df["mvg3"])
                return mask
            if filter_name == "low_vol":
                q50 = df["price_range_pct_q50"] if "price_range_pct_q50" in df.columns else df["price_range_pct"].rolling(700, min_periods=1).quantile(0.5)
                return df["price_range_pct"] < q50
            if filter_name == "high_vol":
                q50 = df["price_range_pct_q50"] if "price_range_pct_q50" in df.columns else df["price_range_pct"].rolling(700, min_periods=1).quantile(0.5)
                return df["price_range_pct"] > q50
            if filter_name == "strong_candle":
                return (df["open_close_pct"] < -0.005)
            if filter_name == "near_ma":
                return ((df["mvg1"] - df["close"]) / df["mvg1"] < 0.02)

            # TA-Lib based
            if filter_name == "rsi_overbought":
                return df["rsi"] > 70 if "rsi" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "macd_bearish":
                return df["macdhist"] < 0 if "macdhist" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "stoch_bullish":
                return (df["stoch_k"] > df["stoch_d"]) if "stoch_k" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "cci_reversal":
                return df["cci"] < -100 if "cci" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "adx_trend":
                return (df["adx"] > 25) if "adx" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "bb_rebound":
                return df["close"] > df["bb_upper"] if "bb_upper" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "mom_positive":
                return df["mom"] < 0 if "mom" in df.columns else pd.Series(True, index=df.index)

            # Volume-based (priority to quote_vol_ratio if available, else vol_ratio)
            # quote_vol_ratio = quote_vol / 7d_ma
            # vol_ratio = vol / 7d_ma

            # Helper to get either ratio
            v_ratio = df.get("quote_vol_ratio", df.get("vol_ratio", 1.0))

            if filter_name == "low_volume": return (v_ratio < 1.0)
            if filter_name == "high_volume": return (v_ratio > 1.0)
            if filter_name == "vol_breakout": return (v_ratio > 2.0)

            # New TA-lab filters
            if filter_name == "buy_pressure":
                return df["buy_pressure"] < 0.45 if "buy_pressure" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "mfi_overbought":
                return df["mfi"] > 70 if "mfi" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "sar_aligned":
                return df["close"] < df["sar"] if "sar" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "bop_bearish":
                return df["bop"] < -0.1 if "bop" in df.columns else pd.Series(True, index=df.index)
            if filter_name == "roc_negative":
                return df["roc"] < 0 if "roc" in df.columns else pd.Series(True, index=df.index)

            # Combined Strategies
            if filter_name.startswith("combined_"):
                # Component masks
                _q50 = df["price_range_pct_q50"] if "price_range_pct_q50" in df.columns else df["price_range_pct"].rolling(700, min_periods=1).quantile(0.5)
                low_vol_mask = down_trend & (df["price_range_pct"] < _q50)
                vol_breakout_mask = down_trend & (df.get("quote_vol_ratio", 0.0) > 2.0)
                
                if filter_name == "combined_union":
                    # Trade if EITHER Low Vol OR Volume Breakout
                    return low_vol_mask | vol_breakout_mask
                
                if filter_name == "combined_intersection":
                    # Trade only if BOTH Low Vol AND Volume Breakout
                    return low_vol_mask & vol_breakout_mask
                
                # New Combinations
                above_all_mask = (df["close"] < df["mvg1"]) & (df["close"] < df["mvg2"])
                if "mvg3" in df.columns: above_all_mask &= (df["close"] < df["mvg3"])
                
                if filter_name == "combined_union_plus":
                     # Low Vol OR Vol Breakout OR Above All MAs
                     return low_vol_mask | vol_breakout_mask | above_all_mask
                     
                if filter_name == "combined_union_alt":
                     # Low Vol OR Above All MAs
                     return low_vol_mask | above_all_mask

        
        # No fallback: raise or return False mask depending on context
        if getattr(self, '_strict_filters', True):
            raise ValueError(f"Unknown filter name: {filter_name}")
        logger.debug("Unknown filter '%s' for position '%s' — treating as no-match",
                      filter_name, position)
        return pd.Series(False, index=df.index)

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

        regime_name = regime.get("regime", "baseline")
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
            
            save_dir = os.path.join(out_dir, "filter")
            os.makedirs(save_dir, exist_ok=True)
            
            save_path = os.path.join(save_dir, f"filter_{safe_name}.parquet")
            try:
                # Narrow float64 feature columns to float32 (the rolling trainer
                # casts to float32 on load anyway) + zstd — roughly halves the
                # filter parquet with zero training impact. Only float64 columns
                # are cast, so integer/timestamp columns keep full precision.
                _f64_to_f32 = {
                    c: "float32" for c in filtered_subset.columns
                    if filtered_subset[c].dtype == "float64"
                }
                filtered_subset.astype(_f64_to_f32).to_parquet(
                    save_path, index=False, compression="zstd",
                )
                # logger.info(f"Saved filtered signals for {regime_id} to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save filtered signals for {safe_name}: {e}")

        return filtered_subset
