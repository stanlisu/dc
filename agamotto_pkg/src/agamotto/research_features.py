"""agamotto research: feature engineering.

Split out of research.py to keep every script under the PyArmor trial
per-script ceiling (research.py built clean at 58,842 B and failed at
63,418 B; see dc deploy.log). The method body below is VERBATIM from
research.py — this split must not change behaviour, and byte-parity of the
emitted filter parquets is the gate on it.
"""
from __future__ import annotations

import logging
import re
import time

import numpy as np
import pandas as pd

from .utils import _timeframe_to_seconds

logger = logging.getLogger(__name__)


class FeatureEngineeringMixin:
    """Provides :meth:`engineer_features` to AgamottoResearch."""

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

                distance_short = ((high_next - close_safe) / close_safe).replace([np.inf, -np.inf], np.nan)
                short_layers = np.floor(distance_short / step_size)
                short_layers = short_layers.clip(lower=0, upper=ladder)
                short_layers = short_layers.fillna(0).astype(int)

                # 2026-06-20 refined ladder fill (parity with mjolnir dc 4fb9eb8): no free
                # base rung (size = n, NOT 1 + n) AND a round-trip gate — a unit is realized
                # only if it BOTH filled on entry AND could exit on the opposite excursion
                # (LONG enters on the dip / exits on the rise; SHORT mirrors), so both sides
                # use size = min(long_layers, short_layers). A <1bps dip OR <1bps rise -> 0
                # (no always-on position). The old `1 + n` base rung + forward-excursion
                # look-ahead manufactured a direction-agnostic phantom edge on a
                # close-to-close return.
                size = np.minimum(long_layers, short_layers)

                fee_cost = (fee_rate * 2.0) if fee_rate else 0.0
                long_per_layer_return = price_return - fee_cost
                price_return_long = (long_per_layer_return * size).rename(f"{base}_return_long")

                short_raw_per_layer = price_return + fee_cost
                # (price_return + fee_cost) * size, then short_subset flips it to -(...)
                price_return_short = (short_raw_per_layer * size).rename(f"{base}_return_short")

                # Raw returns (no fee) for downstream Sharpe calculation
                price_return_long_raw = (price_return * size).rename(f"{base}_return_long_raw")
                price_return_short_raw = (price_return * size).rename(f"{base}_return_short_raw")

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

                    distance_short2 = ((high_max2 - close_safe) / close_safe).replace([np.inf, -np.inf], np.nan)
                    short_layers2 = np.floor(distance_short2 / step_size)
                    short_layers2 = short_layers2.clip(lower=0, upper=ladder).fillna(0).astype(int)

                    # Refined ladder (same as 1-bar block): no base rung + round-trip gate.
                    size2 = np.minimum(long_layers2, short_layers2)

                    ret_2bar = price_return_2bar.rename(f"{base}_ret_2bar")
                    return_long_2bar = ((price_return_2bar - fee_cost) * size2).rename(f"{base}_return_long_2bar")
                    return_short_2bar = ((price_return_2bar + fee_cost) * size2).rename(f"{base}_return_short_2bar")
                    return_long_2bar_raw = (price_return_2bar * size2).rename(f"{base}_return_long_2bar_raw")
                    return_short_2bar_raw = (price_return_2bar * size2).rename(f"{base}_return_short_2bar_raw")

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

