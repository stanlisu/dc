"""MjolnirFeatures: microstructure + price features on aligned bar DataFrames."""

from __future__ import annotations

import logging
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .features_scalefree import scale_free_levels

logger = logging.getLogger(__name__)

# Default rolling windows (in bars)
DEFAULT_WINDOWS: List[int] = [30, 60, 300, 900]
# TA-Lib MA periods (mirrors Agamotto default)
MA_PERIODS = [7, 25, 99]
# Bar-TF → seconds mapping (used for boundary-aligned target computation)
_TF_SECONDS: dict = {"5s": 5, "15s": 15, "30s": 30, "1m": 60, "5m": 300, "15m": 900}
# Reverse map, built once. Every value in _TF_SECONDS is distinct, so a measured
# bar spacing names exactly one TF or none at all (in which case we raise).
_SECONDS_TO_TF: Dict[int, str] = {v: k for k, v in _TF_SECONDS.items()}
assert len(_SECONDS_TO_TF) == len(_TF_SECONDS), "_TF_SECONDS values must be distinct"


def infer_bar_seconds(index: pd.Index) -> Optional[float]:
    """Modal spacing of *index*, in SECONDS, or ``None`` when undecidable.

    ``None`` means "too few rows to judge" — the caller must keep looking or
    raise, never treat it as agreement.

    The MODE, not the mean: zero-trade bars, gaps at the seams between daily
    parquets and duplicate-dropped rows all drag the mean, while the mode is the
    grid step itself.

    Duration arithmetic goes through ``Timedelta.total_seconds()`` and NEVER a
    raw int64 view (``.asi8`` / ``.astype("int64")`` / ``.view("int64")``). Those
    return the index's OWN unit, which is ``[us]`` for anything that round-trips
    through parquet — exactly the trap that made ``_window_days`` read 5s bars as
    0.005s (CLAUDE.md "Duration arithmetic"; dc ``25eb57a``, 2026-08-04). Bar
    parquets are the very frames that hit this, so it is not hypothetical here.
    """
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 10:
        return None
    deltas = pd.Series(index.sort_values()).diff().dropna()
    deltas = deltas[deltas > pd.Timedelta(0)]
    if deltas.empty:
        return None
    return float(deltas.mode().iloc[0].total_seconds())


def resolve_bar_grid(
    config: Mapping,
    symbol_bars: Mapping[str, pd.DataFrame],
) -> Tuple[str, int]:
    """``(bar_tf, horizon_bars)`` MEASURED from the bars actually loaded.

    THE ONE derivation. Both ``core/research.py`` sites (``engineer_features``,
    which hands ``bar_tf`` to :class:`MjolnirFeatures`, and ``verticalize``,
    which hands ``horizon_bars`` to the ladder) and ``core/streaming.py``'s two
    equivalents call this and nothing else, so they cannot drift apart.

    WHAT THIS REPLACES. All four sites carried the identical expression::

        bar_tf = "5s" if cfg.get("TRAIN_BARS_DIR") else time_unit
        horizon_bars = max(1, _TF_SECONDS[time_unit] // _TF_SECONDS[bar_tf])

    ``bar_tf`` was inferred from WHETHER ``TRAIN_BARS_DIR`` was set, never from
    WHAT IT POINTED AT. Every arm that had one pointed at a ``base.5s_1/bars``
    directory, so the guess held — but only by convention, and nothing checked
    it. Point ``TRAIN_BARS_DIR`` at 15s bars and a 1m target still computed
    ``horizon_bars = 60/5 = 12`` when the truth is ``60/15 = 4``: a 3x wrong
    forward window on the ladder low/high lookahead AND a 3x wrong
    ``_bar_seconds`` inside the boundary-aligned target, silently, with no error.
    That blocks coarser base grids (the cheap way to shrink a 195 GB panel) and
    is a live correctness hazard for any non-5s base.

    Measurement is the ground truth, so no new config key is required and no
    ``setting.json`` has to change. ``TRAIN_BARS_DIR`` was only ever a proxy for
    "the base bars are finer than TIME_UNIT"; the measured spacing subsumes it,
    which is why the flag is not read here at all.

    ``max(1, ...)`` is gone. It clamped a quantity DERIVED from other config,
    banned by name in CLAUDE.md with two prior incidents. A ``TIME_UNIT`` finer
    than the bars is a config error — the target boundary cannot land inside a
    bar — and became a plausible 1-bar ladder instead of a raise.

    Args:
        config:      Parsed setting.json. Only ``TIME_UNIT`` is read.
        symbol_bars: Per-symbol bar frames, DatetimeIndex'd, as ``load()``
                     produces them.

    Returns:
        ``(bar_tf, horizon_bars)`` — the measured base-bar TF, and how many of
        those bars span one ``TIME_UNIT`` target boundary.

    Raises:
        KeyError:   ``TIME_UNIT`` names a TF outside ``_TF_SECONDS``.
        ValueError: bars are empty / too short to measure, symbols disagree on
                    spacing, the spacing is not a known TF, ``TIME_UNIT`` is
                    finer than the bars, or the ratio is not a whole number.
    """
    time_unit = config.get("TIME_UNIT", "5s")
    if time_unit not in _TF_SECONDS:
        raise KeyError(
            f"TIME_UNIT={time_unit!r} is not a known timeframe — known: "
            f"{sorted(_TF_SECONDS)}"
        )

    measured = {}
    for sym, bars in symbol_bars.items():
        secs = infer_bar_seconds(getattr(bars, "index", None))
        if secs is not None:
            measured[sym] = secs
    if not measured:
        raise ValueError(
            "cannot measure base-bar spacing: no loaded symbol had >= 10 bars "
            f"with a positive time delta (symbols: {sorted(symbol_bars)}). "
            "Refusing to guess the bar timeframe — every feature window, the "
            "target boundary and the ladder horizon depend on it."
        )

    distinct = sorted(set(measured.values()))
    if len(distinct) > 1:
        raise ValueError(
            f"symbols disagree on base-bar spacing: {measured!r}. One "
            "experiment cannot mix bar grids — the target horizon is a single "
            "scalar shared by every symbol."
        )

    bar_seconds = distinct[0]
    if bar_seconds != int(bar_seconds) or int(bar_seconds) not in _SECONDS_TO_TF:
        raise ValueError(
            f"measured base-bar spacing {bar_seconds:g}s is not a known "
            f"timeframe — known: "
            f"{[f'{tf}={s}s' for s, tf in sorted(_SECONDS_TO_TF.items())]}. "
            "These bars were built on a grid mjolnir cannot describe; rebuild "
            "them with build_bars.py at a supported TIME_UNIT."
        )
    bar_tf = _SECONDS_TO_TF[int(bar_seconds)]

    target_seconds = _TF_SECONDS[time_unit]
    bar_sec = _TF_SECONDS[bar_tf]
    if target_seconds < bar_sec:
        raise ValueError(
            f"TIME_UNIT={time_unit!r} ({target_seconds}s) is FINER than the "
            f"measured base bars ({bar_tf}, {bar_sec}s): the target boundary "
            "would fall inside a single bar, so there is no forward window to "
            "predict. Point TRAIN_BARS_DIR at bars at or below TIME_UNIT, or "
            "raise TIME_UNIT to at least the bar TF. (This used to floor to a "
            "1-bar ladder via max(1, ...) and train on a target nothing "
            "produces.)"
        )
    if target_seconds % bar_sec:
        raise ValueError(
            f"TIME_UNIT={time_unit!r} ({target_seconds}s) is not a whole "
            f"multiple of the measured base bars ({bar_tf}, {bar_sec}s): the "
            "target boundary does not land on a bar close, so the forward "
            "window would straddle bars. Choose a TIME_UNIT divisible by the "
            "bar TF."
        )
    return bar_tf, int(target_seconds // bar_sec)


class MjolnirFeatures:
    """Compute microstructure and price features on aligned bar DataFrames.

    Feature groups:
    - Trade flow:   trade_imbalance, dollar_tsi, trade_intensity, kyle_lambda
    - Order book:   depth_imbalance_L1/L3/L5, relative_spread, microprice_vs_mid
    - OFI:          Multi-level Order Flow Imbalance (Cont et al. 2023)
    - Derivative:   basis_pct, funding_rate
    - Liquidation:  liq_burst_ratio, liq_directional_imbalance, pre_funding
    - Price:        Same TA-Lib indicators as Agamotto (MA, RSI, ATR, MACD, BB)
    - Rolling stats on key signals at each window in feature_windows
      (constant; see mjolnir/core/research.py:_DEFAULT_FEATURE_WINDOWS)

    Args:
        feature_windows: Rolling window sizes in bars (default [30,60,300,900]).
        target_horizon:  Number of bars ahead for the forward return target.
                         Ignored when target_tf != bar_tf (boundary-aligned mode).
        fee_rate:        Round-trip fee as a fraction (e.g. 0.00045).
        prefix:          Optional string prefix for all feature columns (not target
                         columns).  Used for multi-resolution merging so higher-TF
                         features are named e.g. ``15s_trade_imbalance_w30``.
                         Default ``None`` — no prefix, backward-compatible.
        bar_tf:          Time resolution of the base bars (e.g. "5s"). Used to
                         determine bars-per-boundary for variable-horizon targets.
        target_tf:       Target time resolution (e.g. "1m").  When target_tf !=
                         bar_tf, each row's target is the return to the next
                         target_tf boundary, not a fixed-horizon shift.
    """

    # Columns that must NOT be prefixed regardless of the prefix setting.
    # These are target/metadata columns produced by _compute_targets and
    # _compute_temporal.
    _META_COLS: frozenset = frozenset({
        "return", "return_long", "return_short",
        "return_long_raw", "return_short_raw",
        "year", "month",
    })

    # Point-in-time book / snapshot / derivative feature columns that carry
    # bar-CLOSE microstructure state but are stamped at the bar-OPEN timestamp
    # (the aligner takes the LAST tick in each bar, or ffills the derivative
    # ticker, then stamps the result at the bar-open index — see
    # aligner.py _last_value / _last_value_snapshot / _ffill_derivative and
    # _make_bar_index/floor). The forward-return target window ALSO opens at the
    # bar-open timestamp, so a value that reflects the END of bar T leaks the
    # first part of bar T's target window → look-ahead. These columns are shifted
    # forward by exactly one base bar in compute() so the value at row T is the
    # state known at/before T (the prior bar's close), mirroring the cross-TF
    # +tf_seconds shift in multi_tf_merge.py.
    #
    # NOTE: this set is purely point-in-time book/snapshot/derivative state and
    # its direct contemporaneous derivatives (incl. OFI, a depth difference).
    # It intentionally EXCLUDES already-causal columns (pct_change/diff/rolling-
    # of-history, ret_lagN, *_roll*; basis_pct's source is OK to shift as
    # point-in-time), trade/OHLCV bar aggregates, and all target/meta columns.
    _POINT_IN_TIME_PREFIXES: tuple = (
        "bids_", "asks_",        # raw snapshot levels (price/qty)
        "depth_bid_L", "depth_ask_L", "depth_imbalance_L",  # depth + imbalance
        "ofi_L", "ofi_agg",      # order-flow imbalance (depth diff)
    )
    _POINT_IN_TIME_EXACT: frozenset = frozenset({
        # book_ticker last-in-bar raw columns
        "bid_price", "bid_amount", "ask_price", "ask_amount",
        # contemporaneous book derivatives
        "spread", "mid_price", "relative_spread", "microprice", "microprice_vs_mid",
        # derivative_ticker ffilled raw state
        "mark_price", "index_price", "open_interest",
        "funding_rate", "next_funding_time", "predicted_funding_rate",
        "basis_pct",
    })

    def _point_in_time_columns(self, df: pd.DataFrame) -> List[str]:
        """Columns whose value at row T reflects bar T's CLOSE and must shift +1."""
        cols: List[str] = []
        for c in df.columns:
            if c in self._POINT_IN_TIME_EXACT or c.startswith(self._POINT_IN_TIME_PREFIXES):
                cols.append(c)
        return cols

    def __init__(
        self,
        feature_windows: Optional[List[int]] = None,
        target_horizon: int = 60,
        fee_rate: float = 0.00045,
        prefix: Optional[str] = None,
        bar_tf: str = "5s",
        target_tf: str = "5s",
    ) -> None:
        self.feature_windows = feature_windows or DEFAULT_WINDOWS
        self.target_horizon = target_horizon
        self.fee_rate = fee_rate
        self.prefix = prefix
        self.bar_tf = bar_tf
        self.target_tf = target_tf
        self._bar_seconds = _TF_SECONDS.get(bar_tf, 5)
        self._target_seconds = _TF_SECONDS.get(target_tf, self._bar_seconds)
        self._boundary_mode = self._target_seconds > self._bar_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Compute all features on the aligned bar DataFrame.

        Args:
            bars: Output of StreamAligner.align() — one row per bar with a
                  UTC DatetimeIndex.  May span multiple days.

        Returns:
            DataFrame with all feature columns + target return columns.
            Same index as ``bars``.
        """
        df = bars.copy()

        df = self._compute_book_features(df)
        df = self._compute_ofi(df)
        df = self._compute_derivative_features(df)

        # WHY: base-TF point-in-time book/snapshot/derivative features carry the
        # bar-CLOSE state (last tick in bar, or ffilled derivative) but are
        # stamped at the bar-OPEN timestamp, where the forward-return target
        # window also opens. So the contemporaneous value at row T "sees" the
        # first part of bar T's target window → look-ahead leakage (empirically:
        # corr(depth_imbalance_L1, target) peaks when the feature is pushed ~15s
        # into the FUTURE). The cross-TF merge already corrects the analogous
        # higher-TF case (multi_tf_merge.py shifts the higher-TF index forward by
        # tf_seconds); apply the SAME correction to base-TF point-in-time columns
        # by shifting them forward exactly ONE base bar so row T holds bar T-1's
        # CLOSED state — known at/before T, the instant the target window opens.
        # Done BEFORE rolling stats (so their rolling windows inherit causal
        # timing) and BEFORE targets (computed from the UNSHIFTED mid below) so
        # the target itself is never shifted. Trade/OHLC bar aggregates and
        # already-causal columns (pct_change/diff/rolling-of-history) are NOT
        # shifted; see _point_in_time_columns / _POINT_IN_TIME_* for the set.
        target_mid = df.get("mid_price", df.get("close"))
        if target_mid is not None:
            target_mid = target_mid.copy()
        pit_cols = self._point_in_time_columns(df)
        if pit_cols:
            df[pit_cols] = df[pit_cols].shift(1)

        df = self._compute_liquidation_features(df)
        df = self._compute_trade_flow_features(df)
        df = self._compute_price_features(df)
        df = self._compute_rolling_features(df)
        df = self._compute_targets(df, target_mid=target_mid)
        # Defragment after many column assignments before final columns
        df = df.copy()
        df = self._compute_temporal(df)
        # Inject intra-cycle position features when training at finer granularity
        # than the prediction target boundary (e.g. 5s base → 1m target).
        if self._boundary_mode:
            df = self._compute_cycle_features(df)

        # Sanitise microstructure features: inf/NaN on sparse bars (zero-trade
        # 5s bars) is replaced with 0.0, which is a valid "no activity" value.
        # Target columns (return, return_long, return_short, *_raw) are excluded
        # so NaN at the tail (incomplete forward window) is preserved for the
        # training pipeline to drop correctly.
        feature_cols = [c for c in df.columns if c not in self._META_COLS]
        df[feature_cols] = (
            df[feature_cols]
            .replace([np.inf, -np.inf], 0.0)
            .fillna(0.0)
        )

        # Apply column prefix for multi-resolution merging.  Only feature
        # columns are renamed — target/metadata columns (in _META_COLS) keep
        # their original names so the training pipeline can locate them.
        if self.prefix:
            rename_map = {
                c: f"{self.prefix}_{c}"
                for c in df.columns
                if c not in self._META_COLS
            }
            df = df.rename(columns=rename_map)

        return df

    # ------------------------------------------------------------------
    # Book features
    # ------------------------------------------------------------------

    def _compute_book_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Spread, microprice, depth imbalance."""
        # Spread & microprice from book_ticker last-value columns
        if "bid_price" in df.columns and "ask_price" in df.columns:
            df["spread"] = df["ask_price"] - df["bid_price"]
            df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2
            df["relative_spread"] = df["spread"] / (df["mid_price"] + 1e-10)

            if "bid_amount" in df.columns and "ask_amount" in df.columns:
                bid_q = df["bid_amount"].clip(lower=0)
                ask_q = df["ask_amount"].clip(lower=0)
                total_q = bid_q + ask_q + 1e-10
                df["microprice"] = (df["bid_price"] * ask_q + df["ask_price"] * bid_q) / total_q
                df["microprice_vs_mid"] = df["microprice"] - df["mid_price"]

        # Depth imbalance at multiple levels
        for n in (1, 3, 5):
            bid_col = f"depth_bid_L{n}"
            ask_col = f"depth_ask_L{n}"
            if bid_col in df.columns and ask_col in df.columns:
                denom = df[bid_col] + df[ask_col] + 1e-10
                df[f"depth_imbalance_L{n}"] = (df[bid_col] - df[ask_col]) / denom

        return df

    # ------------------------------------------------------------------
    # OFI (Order Flow Imbalance) — Cont et al. 2023
    # ------------------------------------------------------------------

    def _compute_ofi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Multi-level Order Flow Imbalance (approximated from bar snapshots).

        OFI_k ≈ Δbid_qty_k − Δask_qty_k between consecutive bars.
        Aggregate: ofi_agg = sum(OFI_k, k=1..5).
        """
        ofi_cols = []
        for n in (1, 3, 5):
            bid_col = f"depth_bid_L{n}"
            ask_col = f"depth_ask_L{n}"
            if bid_col in df.columns and ask_col in df.columns:
                ofi = df[bid_col].diff() - df[ask_col].diff()
                col_name = f"ofi_L{n}"
                df[col_name] = ofi
                ofi_cols.append(col_name)

        if ofi_cols:
            df["ofi_agg"] = df[ofi_cols].sum(axis=1)

        return df

    # ------------------------------------------------------------------
    # Derivative features
    # ------------------------------------------------------------------

    def _compute_derivative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basis, funding, OI velocity."""
        if "mark_price" in df.columns and "index_price" in df.columns:
            df["basis_pct"] = (
                (df["mark_price"] - df["index_price"]) / (df["index_price"] + 1e-10) * 100
            )

        # oi_velocity/oi_acceleration REMOVED 2026-07-24: pct_change of the
        # REST-polled OI counter is not replicable live — live (5s poll) vs
        # offline (Tardis's poller) step timings disagree in SIGN on ~10% of
        # 30s bars (corr as low as 0.42 per symbol, measured 2026-07-22).
        # Raw open_interest LEVEL stays: live-vs-built swap-rescore of the
        # deployed model = 0 threshold flips (marvel tesseract/
        # portfolio_2026-07-22/oi_swap_summary_2026-07-22.json).

        # Pre-funding-settlement flag: 0–2h before 00:00 / 08:00 / 16:00 UTC
        if df.index.tz is not None:
            hour = df.index.hour
            minute = df.index.minute
            seconds_in_day = hour * 3600 + minute * 60
            # Settlement epochs: 0, 8*3600, 16*3600
            settlement_seconds = np.array([0, 8 * 3600, 16 * 3600])
            pre_fund = np.zeros(len(df), dtype=bool)
            for epoch in settlement_seconds:
                window_start = (epoch - 2 * 3600) % 86400
                if window_start < epoch:
                    pre_fund |= (seconds_in_day >= window_start) & (seconds_in_day < epoch)
                else:
                    # Wraps midnight
                    pre_fund |= (seconds_in_day >= window_start) | (seconds_in_day < epoch)
            df["pre_funding"] = pre_fund.astype(float)

        return df

    # ------------------------------------------------------------------
    # Liquidation features
    # ------------------------------------------------------------------

    def _compute_liquidation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Liquidation burst ratio and directional imbalance."""
        has_long = "liq_long_notional" in df.columns
        has_short = "liq_short_notional" in df.columns

        if has_long and has_short:
            df["liq_total_notional"] = df["liq_long_notional"].fillna(0) + df["liq_short_notional"].fillna(0)
            total = df["liq_total_notional"]
            roll60 = total.rolling(60, min_periods=1).mean()
            df["liq_burst_ratio"] = total / (roll60 + 1e-10)
            df["liq_directional_imbalance"] = (
                (df["liq_long_notional"].fillna(0) - df["liq_short_notional"].fillna(0))
                / (df["liq_total_notional"] + 1e-10)
            )
        return df

    # ------------------------------------------------------------------
    # Trade flow features
    # ------------------------------------------------------------------

    def _compute_trade_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dollar TSI, trade intensity, Kyle's lambda."""
        close = df.get("close", df.get("mid_price"))
        if close is None:
            return df

        if "trade_imbalance" in df.columns and "volume" in df.columns:
            df["dollar_tsi"] = df["trade_imbalance"] * df["volume"] * close

        if "n_trades" in df.columns:
            roll = df["n_trades"].rolling(60, min_periods=1).mean()
            df["trade_intensity"] = df["n_trades"] / (roll + 1e-10)

        if "volume" in df.columns and "open" in df.columns:
            price_move = (close - df["open"]).abs()
            df["kyle_lambda"] = price_move / (df["volume"] + 1e-10)

        return df

    # ------------------------------------------------------------------
    # Price / TA-Lib features (mirrors Agamotto)
    # ------------------------------------------------------------------

    def _compute_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standard price indicators using TA-Lib (or numpy fallback)."""
        close = df.get("close")
        if close is None or close.isna().all():
            return df

        high = df.get("high", close)
        low = df.get("low", close)
        volume = df.get("volume", pd.Series(np.nan, index=df.index))
        open_ = df.get("open", close)

        hist_ret = close.pct_change(fill_method=None)
        open_close_diff = close - open_
        new_cols = {}
        for i, period in enumerate(MA_PERIODS, start=1):
            new_cols[f"mvg{i}"] = close.rolling(period, min_periods=1).mean()
        new_cols["ret_lag1"] = hist_ret.shift(1)
        new_cols["ret_lag2"] = hist_ret.shift(2)
        new_cols["ret_lag3"] = hist_ret.shift(3)
        new_cols["price_range"] = high - low
        new_cols["price_range_pct"] = (high - low) / (open_ + 1e-10)
        new_cols["open_close_diff"] = open_close_diff
        new_cols["open_close_pct"] = open_close_diff / (open_ + 1e-10)
        new_cols["high_open_pct"] = (high - open_) / (open_ + 1e-10)
        new_cols["low_open_pct"] = (low - open_) / (open_ + 1e-10)
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        df = self._compute_talib(df, close, high, low, volume)

        # Scale-free forms of the seven raw price/volume levels. verticalize()
        # pools symbols, so a price-unit column acts partly as a symbol ID;
        # measured on 10 symbols x 360k rows of 15m, pooling SIGN-INVERTS them
        # (sar reads +0.0042 pooled vs -0.0388 within-symbol, 3/10 symbols
        # disagreeing) while every transform scores 0/10 sign flips. See
        # mjolnir/core/features_scalefree.py.
        #
        # The raw columns STAY — regime filters gate on them. They are kept out
        # of the MODEL by gauntlet/rolling_predict_returns.select_feature_columns.
        need = ["close", "sar", "bb_upper", "bb_lower", "macd", "macdhist",
                "obv", "ad", "volume"]
        if all(c in df.columns for c in need):
            # mjolnir stores obv/ad as raw cumulative talib.OBV/AD, so the
            # default (differencing) is correct here — unlike agamotto, which
            # already diffs them and must pass obv_is_cumulative=False.
            df = pd.concat([df, scale_free_levels(df)], axis=1)
        else:
            # WHY safe: _compute_talib falls back to _numpy_indicators when
            # TA-Lib is unavailable, which does not emit every column. Raising
            # would turn a degraded run into a dead one; the warning names the
            # gap so it cannot pass unnoticed.
            logger.warning("skipping scale-free level features, missing %s",
                           [c for c in need if c not in df.columns])
        return df

    def _compute_talib(
        self,
        df: pd.DataFrame,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
    ) -> pd.DataFrame:
        """Apply TA-Lib indicators; fall back to numpy implementations."""
        try:
            import talib  # noqa: F401
            df = self._talib_indicators(df, close, high, low, volume)
        except ImportError:
            logger.debug("TA-Lib not available; using numpy fallbacks.")
            df = self._numpy_indicators(df, close, high, low)
        return df

    def _talib_indicators(self, df, close, high, low, volume):
        import talib
        c = close.values.astype(float)
        h = high.values.astype(float)
        lo = low.values.astype(float)
        v = volume.fillna(0).values.astype(float)

        macd_vals, _, macdhist_vals = talib.MACD(c)
        stoch_k_vals, stoch_d_vals = talib.STOCH(h, lo, c)
        bb_upper, _, bb_lower = talib.BBANDS(c)
        new_cols = {
            "rsi":      talib.RSI(c, timeperiod=14),
            "rsi_7":    talib.RSI(c, timeperiod=7),
            "rsi_28":   talib.RSI(c, timeperiod=28),
            "macd":     macd_vals,
            "macdhist": macdhist_vals,
            "stoch_k":  stoch_k_vals,
            "stoch_d":  stoch_d_vals,
            "cci":      talib.CCI(h, lo, c, timeperiod=14),
            "adx":      talib.ADX(h, lo, c, timeperiod=14),
            "dx":       talib.DX(h, lo, c, timeperiod=14),
            "plus_di":  talib.PLUS_DI(h, lo, c, timeperiod=14),
            "minus_di": talib.MINUS_DI(h, lo, c, timeperiod=14),
            "mom":      talib.MOM(c, timeperiod=10),
            "roc":      talib.ROC(c, timeperiod=10),
            "willr":    talib.WILLR(h, lo, c, timeperiod=14),
            "cmo":      talib.CMO(c, timeperiod=14),
            "atr":      talib.ATR(h, lo, c, timeperiod=14),
            "natr":     talib.NATR(h, lo, c, timeperiod=14),
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "sar":      talib.SAR(h, lo),
            "obv":      talib.OBV(c, v),
            "ad":       talib.AD(h, lo, c, v),
            "mfi":      talib.MFI(h, lo, c, v, timeperiod=14),
            "std":      talib.STDDEV(c, timeperiod=14),
        }
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    def _numpy_indicators(self, df, close, high, low):
        """Minimal numpy fallbacks when TA-Lib is unavailable."""
        c = close

        def _rsi(series, period):
            delta = series.diff()
            gain = delta.clip(lower=0).rolling(period, min_periods=1).mean()
            loss = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
            rs = gain / (loss + 1e-10)
            return 100 - 100 / (1 + rs)

        macd_s = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
        ema_signal = macd_s.ewm(span=9, adjust=False).mean()
        atr = (high - low).rolling(14, min_periods=1).mean()
        roll20 = c.rolling(20, min_periods=1)
        new_cols = {
            "rsi":      _rsi(c, 14),
            "rsi_7":    _rsi(c, 7),
            "rsi_28":   _rsi(c, 28),
            "macd":     macd_s,
            "macdhist": macd_s - ema_signal,
            "atr":      atr,
            "natr":     atr / (close + 1e-10) * 100,
            "std":      c.rolling(14, min_periods=1).std(),
            "bb_upper": roll20.mean() + 2 * roll20.std(),
            "bb_lower": roll20.mean() - 2 * roll20.std(),
            "mom":      c.diff(10),
            "roc":      c.pct_change(10, fill_method=None) * 100,
        }
        # Stubs for TA-Lib-only indicators
        for col in ["stoch_k", "stoch_d", "cci", "adx", "dx", "plus_di",
                    "minus_di", "willr", "cmo", "sar", "obv", "ad", "mfi"]:
            new_cols[col] = np.nan
        return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # ------------------------------------------------------------------
    # Rolling statistics on key microstructure signals
    # ------------------------------------------------------------------

    def _compute_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling mean + std for key signals at each configured window."""
        key_signals = [
            "trade_imbalance", "ofi_agg", "depth_imbalance_L5",
            "relative_spread", "dollar_tsi", "kyle_lambda",
        ]
        new_cols = {}
        for col in key_signals:
            if col not in df.columns:
                continue
            for w in self.feature_windows:
                roll = df[col].rolling(w, min_periods=max(1, w // 4))
                new_cols[f"{col}_roll{w}_mean"] = roll.mean()
                new_cols[f"{col}_roll{w}_std"] = roll.std()

        # Volume ratio (relative to 7-bar MA)
        if "volume" in df.columns:
            ma7_vol = df["volume"].rolling(7, min_periods=1).mean()
            new_cols["vol_ratio"] = df["volume"] / (ma7_vol + 1e-10)

        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df

    # ------------------------------------------------------------------
    # Target returns
    # ------------------------------------------------------------------

    def _compute_targets(
        self, df: pd.DataFrame, target_mid: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Compute forward mid-price returns.

        When target_tf == bar_tf (same resolution), uses a fixed horizon shift
        of TARGET_HORIZON_BARS bars ahead.

        When target_tf > bar_tf (boundary-aligned mode, e.g. 5s base predicting
        1m close), computes variable-horizon returns to the next target_tf
        boundary.  At 05:00:05 with target_tf=1m, the target is the close at
        05:01:00; at 05:00:55 it is still the 05:01:00 close.

        ``target_mid`` MUST be the UNSHIFTED contemporaneous mid (captured in
        compute() before the point-in-time feature shift). The shifted
        ``mid_price`` column in ``df`` is a FEATURE and must never define the
        forward-return target, or the shift would bleed into the label.
        """
        mid = target_mid if target_mid is not None else df.get("mid_price", df.get("close"))
        if mid is None:
            return df

        fee = self.fee_rate
        if self._boundary_mode:
            forward_return = self._boundary_aligned_return(mid)
        else:
            h = self.target_horizon
            forward_return = mid.pct_change(h, fill_method=None).shift(-h)

        # Targets are UNSIGNED market returns — NOT the trade's P&L. A short is
        # profitable when these are NEGATIVE; the consumer applies the direction
        # (marvel generate_daily_pnl books `signal * y_true_raw`, signal = -1 for a
        # short). Fee is subtracted for the long and ADDED for the short: a long
        # needs the move above +fee, a short below -fee. Identical to agamotto
        # (agamotto/research.py:370-380) so the SHARED marvel PnL engine is correct
        # for both algos. Until 2026-07-29 the short rows here were negated, making
        # mjolnir's target position-signed under the same column names agamotto uses
        # unsigned; the engine signed it again and every short leg booked
        # +forward_return. See tasks/lessons.md 2026-07-29.
        new_cols = pd.DataFrame({
            "return": forward_return,
            "return_long": forward_return - fee,
            "return_short": forward_return + fee,
            "return_long_raw": forward_return,
            "return_short_raw": forward_return,
        }, index=df.index)
        return pd.concat([df, new_cols], axis=1)

    def _boundary_aligned_return(self, mid: pd.Series) -> pd.Series:
        """Variable-horizon forward return to the next target_tf boundary.

        For each row, computes (close_at_next_boundary / close_now) - 1.
        At an exact boundary (secs_in_cycle == 0), the next boundary is one
        full target cycle ahead.
        """
        idx = mid.index
        total_secs = idx.hour * 3600 + idx.minute * 60 + idx.second
        secs_in_cycle = total_secs % self._target_seconds
        bars_remaining = np.where(
            secs_in_cycle == 0,
            self._target_seconds // self._bar_seconds,
            (self._target_seconds - secs_in_cycle) // self._bar_seconds,
        ).astype(int)

        mid_arr = mid.values.astype(float)
        n = len(mid_arr)
        i_arr = np.arange(n)
        j_arr = i_arr + bars_remaining
        valid = j_arr < n
        j_clamped = np.clip(j_arr, 0, n - 1)
        returns = np.where(valid, mid_arr[j_clamped] / mid_arr[i_arr] - 1, np.nan)
        return pd.Series(returns, index=idx, dtype=float)

    # ------------------------------------------------------------------
    # BTC cross-asset features
    # ------------------------------------------------------------------

    # The ONLY columns of the BTC feature frame (compute() output) that
    # add_btc_cross_features reads. MUST stay in sync with the column guards
    # in that method. Live inference ships exactly this slim frame to the
    # non-BTC predict workers (compute-once-per-cycle, 2026-07-26) instead of
    # the raw 1000-bar BTC buffer — a new consumer column added below without
    # extending this tuple would silently vanish from live cross-features.
    BTC_CROSS_INPUT_COLS = (
        "mid_price",
        "bid_amount",
        "ask_amount",
        "trade_imbalance",
        "volume",
        "ofi_L1",
        "relative_spread",
        "liq_directional_imbalance",
    )

    def add_btc_cross_features(
        self,
        df: pd.DataFrame,
        btc_features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge BTC microstructure signals into a non-BTC symbol DataFrame.

        Uses already-computed BTC feature columns (output of compute()) so that
        mid_price, relative_spread, ofi_L1, and liq_directional_imbalance are
        available directly.

        No-op when btc_features_df is None or empty.

        Args:
            df:              Feature DataFrame for the target symbol (from compute()).
            btc_features_df: Feature DataFrame for BTC (from compute() on BTC bars).

        Returns:
            df with BTC cross-asset columns appended (in-place copy).
        """
        if btc_features_df is None or btc_features_df.empty:
            return df

        # Align by timestamp index; keep only matching bars
        btc = btc_features_df.reindex(df.index)

        # --- Point-in-time BTC signals ---
        if "mid_price" in btc.columns:
            btc_mid = btc["mid_price"]
            mid_ret = btc_mid.pct_change(fill_method=None)
            df = df.copy()
            df["btc_mid_return_lag1"] = mid_ret.shift(1)
            df["btc_mid_return_lag4"] = btc_mid.pct_change(4, fill_method=None).shift(1)
        else:
            df = df.copy()

        if "bid_amount" in btc.columns and "ask_amount" in btc.columns:
            bid_q = btc["bid_amount"].clip(lower=0)
            ask_q = btc["ask_amount"].clip(lower=0)
            df["btc_book_imbalance_L1"] = (bid_q - ask_q) / (bid_q + ask_q + 1e-10)

        if "trade_imbalance" in btc.columns and "volume" in btc.columns:
            buy_vol = ((btc["trade_imbalance"] + 1) / 2 * btc["volume"]).clip(0, btc["volume"])
            df["btc_trade_imbalance"] = (buy_vol / (btc["volume"] + 1e-10)).clip(0, 1)

        if "ofi_L1" in btc.columns:
            df["btc_ofi_L1"] = btc["ofi_L1"]

        if "relative_spread" in btc.columns and "relative_spread" in df.columns:
            df["btc_spread_ratio"] = btc["relative_spread"] / (df["relative_spread"] + 1e-10)

        if "liq_directional_imbalance" in btc.columns:
            df["btc_liq_directional"] = btc["liq_directional_imbalance"]

        # --- Rolling stats for key BTC cross-signals ---
        cross_roll_cols = [
            col for col in ["btc_book_imbalance_L1", "btc_trade_imbalance", "btc_ofi_L1"]
            if col in df.columns
        ]
        if cross_roll_cols:
            roll_new = {}
            for col in cross_roll_cols:
                for w in self.feature_windows:
                    roll = df[col].rolling(w, min_periods=max(1, w // 4))
                    roll_new[f"{col}_roll{w}_mean"] = roll.mean()
                    roll_new[f"{col}_roll{w}_std"] = roll.std()
            df = pd.concat([df, pd.DataFrame(roll_new, index=df.index)], axis=1)

        return df

    # ------------------------------------------------------------------
    # Temporal metadata
    # ------------------------------------------------------------------

    def _compute_temporal(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add year/month columns required by rolling_predict_returns.py."""
        if df.index.tz is not None:
            df["year"] = df.index.year
            df["month"] = df.index.month
        return df

    def _compute_cycle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intra-cycle position features for boundary-aligned prediction.

        Tells the model where it is within the current target_tf cycle so it
        can correctly calibrate its variable-horizon prediction.

        Added columns (not prefixed — treated as features, sanitised to 0 on NaN):
          cycle_progress   — fraction of current cycle elapsed [0, 1)
          secs_to_boundary — seconds until the next target_tf boundary
        """
        idx = df.index
        total_secs = idx.hour * 3600 + idx.minute * 60 + idx.second
        secs_in_cycle = total_secs % self._target_seconds
        df = df.copy()
        df["cycle_progress"] = secs_in_cycle / self._target_seconds
        df["secs_to_boundary"] = (self._target_seconds - secs_in_cycle) % self._target_seconds
        return df
