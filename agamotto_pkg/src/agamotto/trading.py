"""Agamotto research library for Binance Futures datasets."""

from __future__ import annotations
from .utils import _symbol_to_native, _timeframe_to_seconds, _round_down_to_timeframe_boundary
from .research import AgamottoResearch, _obf

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
import joblib
import glob
import pandas as pd
import os
import time
import json
import logging

# Set up logger for Agamotto - ensure it uses root logger's handlers
logger = logging.getLogger(__name__)
# Don't add handlers here - let it propagate to root logger
# This ensures it uses the same handler configuration as the main process


def dual_gate_filter(sym_preds: "pd.DataFrame"):
    """Split one symbol's regime predictions into firing longs and shorts,
    applying the dual-horizon agreement gate.

    A row fires only when the 1-bar prediction clears ``opt_threshold`` in its
    direction AND — when the regime stack carries a 2-bar threshold for that row
    (``opt_threshold_2bar`` present and non-null with a ``prediction_2bar``) —
    the 2-bar prediction clears ``opt_threshold_2bar`` in the SAME direction
    (long: both above their thresholds; short: both below). Rows lacking a 2-bar
    threshold fall back to 1-bar-only (preserves non-DH behavior). Symmetric for
    long and short.

    Returns (longs_df, shorts_df).
    """
    has_dual = (
        "opt_threshold_2bar" in sym_preds.columns
        and "prediction_2bar" in sym_preds.columns
    )

    long_base = (
        (sym_preds["position"] == "long") &
        (sym_preds["prediction"] > sym_preds["opt_threshold"])
    )
    short_base = (
        (sym_preds["position"] == "short") &
        (sym_preds["prediction"] < sym_preds["opt_threshold"])
    )

    if has_dual:
        has_2bar = sym_preds["opt_threshold_2bar"].notna()
        long_dual_ok = ~has_2bar | (sym_preds["prediction_2bar"] > sym_preds["opt_threshold_2bar"])
        short_dual_ok = ~has_2bar | (sym_preds["prediction_2bar"] < sym_preds["opt_threshold_2bar"])
        longs = sym_preds[long_base & long_dual_ok]
        shorts = sym_preds[short_base & short_dual_ok]
    else:
        longs = sym_preds[long_base]
        shorts = sym_preds[short_base]

    return longs, shorts

# Use relative import for internal module
try:
    from .lib_binance import fetch_futures_klines, klines_to_dataframe
except ImportError:
    # Fallback to assume it's in the path or installed
    try:
        from lib_binance import fetch_futures_klines, klines_to_dataframe
    except ImportError:
        pass


def _closes_at_timestamp(raw, symbols, target_ts) -> dict:
    """``{symbol: close}`` read off ``raw`` at ``target_ts``, selected by LABEL.

    ``target_ts`` is ``vertical_features["timestamp"].max()`` — the row
    ``predict()`` actually ran on. Pricing anywhere else means the order is sent
    at a different bar than the one the model saw.

    **Why label and not position.** This replaced ``self.raw[col].iloc[-2]``,
    whose comment ("not the incomplete current candle whose close is NaN") was
    written for the frame ``_fetch_and_prepare_data`` BUILDS. That frame does end
    with the in-flight candle — the REST fetch includes it and
    ``knull/kline_stream.py`` appends it on the WS path on purpose so the two
    match. But ``self.raw`` is assigned exactly once, in ``_process_combined``,
    AFTER ``combined = combined.iloc[:-1]`` has already removed it. So ``iloc[-1]``
    was the just-closed candle and ``iloc[-2]`` was a full ``TIME_UNIT`` older.

    Measured live on hydra 2026-08-15 03:00:14 (``tesseract/probe_agamotto_bars.py``):
    ``raw`` held ``[02:15, 02:30, 02:45]`` and 699 rows at ``limit=700`` — one
    row dropped, the in-progress 03:00 candle absent entirely — while
    ``vertical_features["timestamp"].max()`` was ``02:45``. Across 550 live
    signals every emitted price matched the ``02:30``-equivalent bar, costing a
    median +23.8 bps per entry on ltp and +29.6 on sumo.

    Two independent layers each believed they owned the "drop the incomplete
    bar" step and neither could see the other, because a positional index has no
    way to state which bar it means. A label can, so the disagreement now raises
    instead of silently pricing a stale bar.

    A NaN at ``target_ts`` falls back to the last valid close AT OR BEFORE it —
    never a later one, which would be lookahead and is reachable whenever ``raw``
    still holds rows past the target. All-NaN or an absent column gives 0.0, which
    the caller's ``close > 0`` guard turns into "use the init-time size".
    """
    symbols = list(symbols or [])
    if raw is None or raw.empty:
        return {sym: 0.0 for sym in symbols}
    # Precondition, stated rather than assumed. `_process_combined` dedupes
    # (`~combined.index.duplicated`) before the only `self.raw =`, so this is
    # unreachable on the live path — but a duplicated label makes `raw.at[...]`
    # return a Series, and the `pd.isna(val)` below then dies with "The truth
    # value of a Series is ambiguous", a crash far from its cause.
    if not raw.index.is_unique:
        dupes = raw.index[raw.index.duplicated()].unique().tolist()
        raise ValueError(
            f"raw has duplicate index labels {dupes[:5]} — cannot resolve a "
            f"single close per timestamp. Dedupe before pricing.")
    if target_ts not in raw.index:
        raise KeyError(
            f"predict() targeted {target_ts!r} but it is not present in raw "
            f"(raw spans {raw.index.min()}..{raw.index.max()}, {len(raw)} rows). "
            f"The prediction row and the pricing row have diverged — refusing to "
            f"price at a different bar than the model saw."
        )
    out = {}
    for sym in symbols:
        col = f"{_symbol_to_native(sym)}_close"
        if col not in raw.columns:
            out[sym] = 0.0
            continue
        val = raw.at[target_ts, col]
        if pd.isna(val):
            logger.warning(
                "NaN close for %s at %s — falling back to the last valid close "
                "at or before it", sym, target_ts)
            # Boolean mask + idxmax, NOT `raw.loc[:target_ts]`. Label slicing on a
            # non-monotonic index does NOT raise — pandas falls back to POSITIONAL
            # slicing and silently includes rows dated AFTER the target, which is
            # the very lookahead this helper exists to prevent, reintroduced inside
            # the fallback. A mask is order-independent by construction.
            prior = raw[col][raw.index <= target_ts].dropna()
            val = prior.loc[prior.index.max()] if len(prior) else 0.0
        out[sym] = float(val)
    return out


class AgamottoTrading(AgamottoResearch):
    def __init__(
        self,
        config: Dict[str, object],
        home_root: str,
        period: Optional[str],
        skip_load: bool = False
    ) -> None:
        super().__init__(config, home_root)
        # In trading, unknown filters are gracefully skipped (not errors)
        self._strict_filters = False
        # Use period from config or parameter (config takes precedence)
        self.period = self.config.get("WEIGHTS_PERIOD") or period
        self.trading_mode = self.config.get(
            "TRADING_MODE", "both")  # both, long_only, short_only
        self.models: dict[str, dict[str, dict[str, object]]] = {}
        self.latest_predictions: pd.DataFrame | None = None
        self.latest_predictions_path: Optional[str] = None
        self.decisions: dict[str, tuple[str, float, float]] = {}

        # Prediction Thresholds
        self.long_pred_threshold = float(
            self.config.get("LONG_PRED_THRESHOLD", 0.0))
        self.short_pred_threshold = float(
            self.config.get("SHORT_PRED_THRESHOLD", 0.0))

        # Strict configuration: REGIME_STACK_PATH is the primary source of truth.
        self.regime_stack_path = self.config.get("REGIME_STACK_PATH")
        
        if self.regime_stack_path:
            if not os.path.isabs(self.regime_stack_path):
                self.regime_stack_path = os.path.join(
                    home_root, self.regime_stack_path)
            self._load_regime_stack()
            if not self.regime_stack:
                 logger.error(f"Regime stack at {self.regime_stack_path} loaded NO valid models!")
        else:
            raise ValueError("STRATEGY_STACK_PATH not provided in configuration.")

        # Calculate SIZES: CAPITAL / yesterday_close for each symbol
        self._calculate_sizes()

        # Load initial data to be ready for immediate trading
        if not skip_load:
            try:
                self.load_data(limit=700)
            except Exception as e:
                logger.warning(f"Failed to load initial data in __init__: {e}")

        logger.info(
            f"Agamotto initiated V5 (STRATEGY_STACK_PATH fix) - Period: {self.period}, Trading Mode: {self.trading_mode}")

    def _calculate_sizes(self) -> None:
        """Calculate SIZES via the unified symbiote.sizing.compute_sizes helper.

        Result is a Dict[str, float] keyed by symbol (replaces the legacy
        List[float] aligned to SYMBOLS index — see symbiote/sizing.py).
        Per CLAUDE.md no-fallback rule: missing yesterday_close, missing
        step_size, or non-positive CAPITAL raises rather than collapsing
        to a zero-rung config.
        """
        if "SIZES" in self.config and self.config["SIZES"]:
            n = (len(self.config["SIZES"]) if hasattr(self.config["SIZES"], "__len__")
                 else "?")
            logger.info(f"SIZES already provided in config: {n} symbols")
            return

        # Import path mirrors agamotto.trading top-of-file: prefer the agamotto
        # relative module, fall back to the bare lib_binance shim. Failure now
        # raises (no silent [0.0]*N fallback per CLAUDE.md).
        from agamotto.lib_binance import (
            fetch_latest_closes,
            fetch_price_lot_sizes,
        )
        from symbiote.sizing import compute_sizes

        symbols = self.config.get("SYMBOLS", [])
        if not symbols:
            logger.warning("No SYMBOLS in config, SIZES not calculated")
            self.config["SIZES"] = {}
            return

        if "CAPITAL" not in self.config:
            raise KeyError(
                "CAPITAL is required to compute SIZES; pass an explicit "
                "USDT-per-rung value in config"
            )
        capital = float(self.config["CAPITAL"])

        logger.info(
            f"Calculating SIZES via compute_sizes for {len(symbols)} symbols "
            f"(CAPITAL={capital})..."
        )

        closes = fetch_latest_closes(symbols, interval="1d")
        logger.info(f"  fetch_latest_closes returned {len(closes)} items")
        lot_sizes = fetch_price_lot_sizes(symbols)
        logger.info(f"  fetch_price_lot_sizes returned {len(lot_sizes)} items")

        step_sizes = {
            sym: float(lot_sizes[sym]["step_size"])
            for sym in symbols
            if sym in lot_sizes and "step_size" in lot_sizes[sym]
        }

        sizes = compute_sizes(
            capital=capital,
            symbols=symbols,
            yesterday_closes=closes,
            step_sizes=step_sizes,
        )
        self.config["SIZES"] = sizes
        self.config["LOT_SIZES"] = lot_sizes
        logger.info(
            f"Calculated SIZES for {len(sizes)} symbols "
            f"(dict keyed by symbol)"
        )

    def _load_regime_stack(self) -> None:
        """Load multiple regimes and their models from the CSV optimization results."""
        from utils.lib import load_regime_stack
        stack_data = load_regime_stack(self.regime_stack_path)

        self.regime_stack = []
        # We also need self.models for backward compatibility/internal hooks
        self.models = {"long": {}, "short": {}}

        for row in stack_data:
            target_dir = row["directory"]
            position = row["position"]
            model_name = row["model"]
            threshold = float(row["optimal_threshold"])
            threshold_2bar_raw = row.get("optimal_threshold_2bar")
            threshold_2bar = float(threshold_2bar_raw) if threshold_2bar_raw not in (None, "", "nan") else float("nan")

            # Resolve research directory
            research_path = os.path.join(
                self.home_root, "gauntlet", target_dir)
            if not os.path.isdir(research_path):
                raise FileNotFoundError(
                    f"Research directory not found: {research_path}")

            # Load the specific model for this strategy
            weights_dir = os.path.join(
                research_path, "weights", self.period)

            if not os.path.isdir(weights_dir):
                raise FileNotFoundError(f"Missing weights directory for period {self.period} in {research_path}")

            # Load model artifact
            regime_name = row.get("regime")
            if not regime_name:
                raise ValueError(f"Missing regime name in stack entry for {target_dir}")

            pos_dir = os.path.join(weights_dir, regime_name)

            if not os.path.isdir(pos_dir):
                raise FileNotFoundError(f"Regime folder {regime_name} not found in {weights_dir}")

            # Try loading with lowercase model name first (common convention)
            low_name = model_name.lower()
            meta_path = os.path.join(pos_dir, f"{low_name}_meta.pkl")
            model_path = os.path.join(pos_dir, f"{low_name}_model.pkl")
            scaler_path = os.path.join(pos_dir, f"{low_name}_scaler.pkl")

            # Fallback to original case if lowercase not found
            if not os.path.exists(model_path):
                meta_path = os.path.join(pos_dir, f"{model_name}_meta.pkl")
                model_path = os.path.join(pos_dir, f"{model_name}_model.pkl")
                scaler_path = os.path.join(pos_dir, f"{model_name}_scaler.pkl")

            if not (os.path.exists(meta_path) and os.path.exists(
                    model_path) and os.path.exists(scaler_path)):
                logger.warning(
                    f"Skipping regime {regime_name}/{model_name}: "
                    f"missing model files in {pos_dir}")
                continue

            artifact = {
                "model": joblib.load(model_path),
                "scaler": joblib.load(scaler_path),
                "metadata": joblib.load(meta_path),
            }

            # Dual-bar: load 2-bar model from {regime}_2bar/ if threshold_2bar is set
            import math as _math
            artifact_2bar = None
            if not _math.isnan(threshold_2bar):
                pos_dir_2bar = os.path.join(weights_dir, f"{regime_name}_2bar")
                if os.path.isdir(pos_dir_2bar):
                    meta_2bar = os.path.join(pos_dir_2bar, f"{low_name}_meta.pkl")
                    model_2bar = os.path.join(pos_dir_2bar, f"{low_name}_model.pkl")
                    scaler_2bar = os.path.join(pos_dir_2bar, f"{low_name}_scaler.pkl")
                    if not os.path.exists(model_2bar):
                        meta_2bar = os.path.join(pos_dir_2bar, f"{model_name}_meta.pkl")
                        model_2bar = os.path.join(pos_dir_2bar, f"{model_name}_model.pkl")
                        scaler_2bar = os.path.join(pos_dir_2bar, f"{model_name}_scaler.pkl")
                    if os.path.exists(meta_2bar) and os.path.exists(model_2bar) and os.path.exists(scaler_2bar):
                        artifact_2bar = {
                            "model": joblib.load(model_2bar),
                            "scaler": joblib.load(scaler_2bar),
                            "metadata": joblib.load(meta_2bar),
                        }
                    else:
                        logger.warning(f"Dual-bar: 2-bar model files missing for {regime_name} in {pos_dir_2bar}")
                else:
                    logger.warning(f"Dual-bar: _2bar weights dir not found: {pos_dir_2bar}")

            # Load config for this strategy to get its specific LONG_FILTER /
            # SHORT_FILTER
            strat_config_path = os.path.join(research_path, "setting.json")
            with open(strat_config_path, "r") as f:
                strat_config = json.load(f)

            filter_val = strat_config.get(
                "LONG_FILTER" if position == "long" else "SHORT_FILTER")

            strat_entry = {
                "id": f"{target_dir}_{position}_{model_name}",
                "directory": target_dir,
                "regime": row.get("regime", "unknown"),
                "position": position,
                "model_name": model_name,
                "threshold": threshold,
                "threshold_2bar": threshold_2bar,
                "artifact": artifact,
                "artifact_2bar": artifact_2bar,
                "filter": filter_val,
                "config": strat_config
            }
            self.regime_stack.append(strat_entry)
            logger.info(
                f"Loaded regime: {
                    strat_entry['id']} (Threshold: {threshold})")

    def _load_latest_weights(self, base_dir: str) -> None:
        logger.info(f"Loading single-directory weights from {base_dir}")
        if not os.path.isdir(base_dir):
            raise FileNotFoundError(
                f"Weights directory '{base_dir}' does not exist")

        # Resolve weights directory (support nested Gauntlet structure)
        weights_root = base_dir
        # Case A: Passed the regime folder. Look for 'weights' inside.
        if os.path.isdir(os.path.join(base_dir, "weights")):
            weights_root = os.path.join(base_dir, "weights")

        # Case B: Resolve window_* subfolders (latest)
        subdirs = glob.glob(os.path.join(weights_root, "window_*"))
        if subdirs:
            # Use the latest window if self.period is None, otherwise specific period
            if self.period:
                specific = os.path.join(weights_root, self.period)
                if os.path.isdir(specific):
                    weights_root = specific
                else:
                    target = sorted(subdirs)[-1]
                    logger.warning(f"Period {self.period} not found in {weights_root}. Using latest: {target}")
                    weights_root = target
            else:
                weights_root = sorted(subdirs)[-1]

        # Initialize regime stack for single-model mode
        self.regime_stack = []

        candidates = {
            "long": os.path.join(weights_root, "long"),
            "short": os.path.join(weights_root, "short")
        }
        for label, dir_path in candidates.items():
            if not os.path.isdir(dir_path):
                continue
            for meta_path in glob.glob(os.path.join(dir_path, "*_meta.pkl")):
                name = os.path.basename(meta_path).replace("_meta.pkl", "")
                artifact = {
                    "model": joblib.load(
                        os.path.join(
                            dir_path,
                            f"{name}_model.pkl")),
                    "scaler": joblib.load(
                        os.path.join(
                            dir_path,
                            f"{name}_scaler.pkl")),
                    "metadata": joblib.load(meta_path),
                }
                self.regime_stack.append({
                    "id": f"base_{label}_{name}",
                    "position": label,
                    "model_name": name,
                    "threshold": self.long_pred_threshold if label == "long" else self.short_pred_threshold,
                    "artifact": artifact,
                    "filter": self.config.get("LONG_FILTER" if label == "long" else "SHORT_FILTER")
                })

    def load_data(self, limit: int = 700) -> None:
        """Fetch and prepare market data with retry logic for staleness."""
        self._fetch_and_prepare_data(limit=limit)

        feature_tf = self.config.get("FEATURE_TF") or self.config.get("TIME_UNIT", "1d")
        tf_seconds = _timeframe_to_seconds(feature_tf)

        self._data_fresh = False
        max_retries = 10
        for i in range(max_retries):
            last_ts = self.raw.index.max()
            now_unix = int(pd.Timestamp.utcnow().tz_localize(None).timestamp())
            current_floor_unix = (now_unix // tf_seconds) * tf_seconds
            expected_last = pd.Timestamp(current_floor_unix - tf_seconds, unit='s')
            if last_ts >= expected_last:
                logger.info(
                    f"Data is fresh (last bar {last_ts} >= expected {expected_last})")
                self._data_fresh = True
                break
            if i < max_retries - 1:
                logger.warning(
                    f"Data stale (last bar {last_ts}, expected >= {expected_last}). "
                    f"Retry {i + 1}/{max_retries}...")
                time.sleep(1)
                self._fetch_and_prepare_data(limit=limit)
            else:
                logger.error(
                    f"Data stale after {max_retries} retries "
                    f"(last bar {last_ts}, expected >= {expected_last}). "
                    f"Decisions will be CLOSE (all zeros).")

    def _process_combined(self, combined: pd.DataFrame,
                          limit: int) -> None:
        """Shared post-processing for both REST and WS paths."""
        combined = combined[~combined.index.duplicated(keep="last")]
        if combined.index.tz is not None:
            combined.index = combined.index.tz_localize(None)
        combined = combined.tail(limit)

        # Drop the current incomplete bar so features use only closed bars,
        # matching the training pipeline behaviour.
        if len(combined) > 1:
            combined = combined.iloc[:-1]

        self.raw = combined
        self.engineer_features()
        self.verticalize()

        # Dump the just-closed candle to CSV for tesseract comparison.
        if (hasattr(self, 'vertical_features')
                and self.vertical_features is not None
                and not self.vertical_features.empty):
            now_utc = pd.Timestamp.now(tz="UTC").tz_localize(None)
            timeframe_str = (self.config.get("FEATURE_TF")
                             or self.config.get("TIME_UNIT", "15m"))
            current_candle_start = _round_down_to_timeframe_boundary(
                now_utc, timeframe_str)
            target_ts = current_candle_start - pd.Timedelta(
                seconds=_timeframe_to_seconds(timeframe_str))
            latest_row = self.vertical_features[
                self.vertical_features["timestamp"] == target_ts]
            project = self.config.get("PROJECT", "debug")
            version = self.config.get("VERSION", "")
            ts_str = current_candle_start.strftime("%Y%m%d_%H%M")
            if version:
                debug_name = f"debug_features_{version}_{ts_str}.csv"
            else:
                algo = self.config.get("STRATEGY", "agamotto")
                debug_name = (f"debug_features_{algo}"
                              f"_{timeframe_str}_{ts_str}.csv")
            features_csv_path = os.path.join(
                self.home_root, project, debug_name)
            if os.path.exists(features_csv_path):
                logger.info(
                    f"Skipping debug_features dump — "
                    f"{debug_name} already exists (ts={target_ts})")
            else:
                latest_row = latest_row.copy()
                latest_row["timestamp"] = latest_row["timestamp"].apply(
                    lambda t: t.strftime("%Y-%m-%d %H:%M:%S")
                    if hasattr(t, "strftime") else str(t))
                latest_row.to_csv(features_csv_path, index=False)

    def _fetch_and_prepare_data(self, limit: int = 700) -> None:
        """Internal helper to fetch data without retry/recursion logic."""
        symbols = self.config["SYMBOLS"]
        if not symbols:
            raise RuntimeError("No symbols configured for trading.")

        feature_tf = (self.config.get("FEATURE_TF")
                      or self.config.get("TIME_UNIT", "1d"))

        # --- WS buffer path: read from kline_buffer if available ---
        kline_buffer = getattr(self, '_kline_buffer', None)
        if kline_buffer is not None:
            native_symbols = [
                _symbol_to_native(s) for s in symbols
                if _symbol_to_native(s)
            ]
            if kline_buffer.is_ready(native_symbols, feature_tf):
                frames = []
                unusable: List[str] = []
                for native in native_symbols:
                    df = kline_buffer.get_dataframe(native, feature_tf)
                    if df is None or df.empty:
                        unusable.append(native)
                        continue
                    # Rename from {sym}_{tf}_{col} to {sym}_{col}
                    rename_map = {
                        c: c.replace(f"_{feature_tf}_", "_", 1)
                        for c in df.columns
                        if f"_{feature_tf}_" in c
                    }
                    if rename_map:
                        df = df.rename(columns=rename_map)
                    frames.append(df)
                if len(frames) == len(native_symbols):
                    combined = pd.concat(frames, axis=1).sort_index()
                    self._process_combined(combined, limit)
                    return
                # A caller that wired up a WS buffer believes this cycle is
                # cheap; falling back to REST costs ~10s inside the decision
                # window. Name the symbols so the fault is actionable, never
                # just "incomplete".
                logger.warning(
                    "WS kline buffer read INCOMPLETE for tf=%s — FALLING BACK "
                    "TO REST (slow path): %d/%d symbols usable, "
                    "empty/missing=%s",
                    feature_tf, len(frames), len(native_symbols),
                    unusable[:20])
            else:
                missing = [
                    n for n in native_symbols
                    if kline_buffer.get_dataframe(n, feature_tf) is None
                ]
                logger.warning(
                    "WS kline buffer NOT READY for tf=%s — FALLING BACK TO "
                    "REST (slow path): %d/%d symbols absent from the buffer: "
                    "%s",
                    feature_tf, len(missing), len(native_symbols),
                    missing[:20])

        # --- REST path (original) ---
        end_ts = pd.Timestamp.utcnow()
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        else:
            end_ts = end_ts.tz_convert("UTC")

        # Fetch up to now (includes the current incomplete candle).
        # make_decision() drops the last row (incomplete) before filtering and
        # predicting, so the decision is always based on the just-closed,
        # fully-settled candle — no extra latency.
        end_ms = int(end_ts.timestamp() * 1000)

        # FEATURE_TF allows multi-horizon bots to fetch a finer-grained timeframe
        # (e.g. 15m features) while the executor cycle runs on the target horizon
        # (e.g. 1h via TIME_UNIT).  Falls back to TIME_UNIT when not set.
        feature_tf = self.config.get("FEATURE_TF") or self.config.get("TIME_UNIT", "1d")
        timeframe_seconds = _timeframe_to_seconds(feature_tf)
        start_ts = end_ts - pd.Timedelta(seconds=timeframe_seconds * limit)
        start_ts = start_ts.tz_convert(
            "UTC") if start_ts.tzinfo else start_ts.tz_localize("UTC")
        start_ms = int(start_ts.timestamp() * 1000)

        frames: List[pd.DataFrame] = []
        time_unit = feature_tf

        def _fetch_symbol(sym: str) -> pd.DataFrame | None:
            native = _symbol_to_native(sym)
            if not native:
                return None
            rows = fetch_futures_klines(
                native, time_unit, start_ms, end_ms, limit=limit)
            df = klines_to_dataframe(rows, native, time_unit)
            if df.empty:
                return None
            rename_map = {
                f"{native}_{time_unit}_{col}": f"{native}_{col}"
                for col in [
                    "open", "high", "low", "close", "volume",
                    "quote_volume", "number_of_trades",
                    "taker_buy_base_volume", "taker_buy_quote_volume",
                ]
            }
            df.rename(columns=rename_map, inplace=True)
            return df

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_sym = {
                executor.submit(_fetch_symbol, sym): sym
                for sym in symbols
            }
            for future in as_completed(future_to_sym):
                sym = future_to_sym[future]
                try:
                    df = future.result()
                    if df is not None:
                        frames.append(df)
                except Exception as e:
                    logger.error(f"Failed to fetch klines for {sym}: {e}")

        if not frames:
            raise RuntimeError("Failed to fetch trading data via REST API.")

        combined = pd.concat(frames, axis=1).sort_index()
        self._process_combined(combined, limit)



    def predict(self, filtered_signals: pd.DataFrame, regime: dict) -> pd.DataFrame:
        """
        Run inference for a single regime on the latest settled kline.

        make_decision() already drops the incomplete candle from
        self.features before verticalize/filter.  We predict on the
        settled candle (max timestamp in vertical_features), NOT on the
        max timestamp within filtered_signals — the latter can be stale
        if the current candle doesn't pass the regime's filter.
        """
        if filtered_signals.empty:
            return pd.DataFrame()

        # Target is the settled candle — the latest timestamp in the full
        # vertical_features (after make_decision's incomplete-candle drop).
        target_ts = self.vertical_features["timestamp"].max()

        # Filter for exactly the target timestamp
        target_row = filtered_signals[filtered_signals["timestamp"] == target_ts].copy()
        
        if target_row.empty:
            # logger.debug(f"No data found for target timestamp {target_ts} in regime {regime.get('id')}")
            return pd.DataFrame()

        # Obfuscation: live features are computed with REAL names; the model's
        # meta.feature_columns + scaler are CODED (or REAL for pre-rollout
        # weights). Add coded aliases so the selection + scaler name-check match
        # either namespace.
        target_row = _obf().add_feature_aliases(target_row)

        # 2. Run prediction
        try:
            artifact = regime["artifact"]
            f_cols = artifact["metadata"]["feature_columns"]
            
            # Ensure columns exist
            missing = [c for c in f_cols if c not in target_row.columns]
            if missing:
                logger.warning(f"Regime {regime['id']} missing features: {missing}")
                return pd.DataFrame()
                
            X = target_row[f_cols].copy()
            if X.isnull().values.any():
                nan_cols = [c for c in f_cols if X[c].isnull().any()]
                logger.warning(
                    f"Regime {regime['id']}: NaN in {len(nan_cols)} feature columns "
                    f"(filling with 0.0): {nan_cols}")
                X.fillna(0.0, inplace=True)
                
            X_scaled = artifact["scaler"].transform(X)
            preds = artifact["model"].predict(X_scaled)

            # Guard against degenerate models (e.g. near-zero IQR in RobustScaler
            # causing division-by-~0 → predictions on the order of 1e14+).
            # Returns on any single bar cannot exceed 100%, so clip to [-1, 1].
            import numpy as np
            if np.abs(preds).max() > 1.0:
                logger.warning(
                    f"Regime {regime['id']}: prediction overflow "
                    f"({preds}) — clipping to [-1, 1]")
                preds = np.clip(preds, -1.0, 1.0)

            target_row["prediction"] = preds
            target_row["strategy_id"] = regime["id"]
            target_row["regime"] = regime.get("regime", "unknown")
            target_row["opt_threshold"] = regime["threshold"]

            # Dual-bar: run 2-bar model if loaded
            artifact_2bar = regime.get("artifact_2bar")
            threshold_2bar = regime.get("threshold_2bar", float("nan"))
            if artifact_2bar is not None:
                f_cols_2bar = artifact_2bar["metadata"]["feature_columns"]
                missing_2bar = [c for c in f_cols_2bar if c not in target_row.columns]
                if missing_2bar:
                    logger.warning(f"Regime {regime['id']} 2-bar missing features: {missing_2bar}")
                    target_row["prediction_2bar"] = float("nan")
                else:
                    X_2bar = target_row[f_cols_2bar].copy()
                    X_2bar.fillna(0.0, inplace=True)
                    X_scaled_2bar = artifact_2bar["scaler"].transform(X_2bar)
                    preds_2bar = artifact_2bar["model"].predict(X_scaled_2bar)
                    if np.abs(preds_2bar).max() > 1.0:
                        preds_2bar = np.clip(preds_2bar, -1.0, 1.0)
                    target_row["prediction_2bar"] = preds_2bar
                target_row["opt_threshold_2bar"] = threshold_2bar
            else:
                target_row["prediction_2bar"] = float("nan")
                target_row["opt_threshold_2bar"] = float("nan")

            return target_row
            
        except Exception as e:
            logger.error(f"Prediction failed for {regime.get('id')}: {e}")
            return pd.DataFrame()

    def make_decision(self, label: str = "both") -> dict[str, list[float, float]]:
        """
        Orchestrator for generating decisions using regime stack.
        """
        # 1. Initialize decisions
        self.decisions = {sym: [0.0, 0.0] for sym in self.config.get("SYMBOLS", [])}

        if not getattr(self, "_data_fresh", True):
            logger.warning(
                "Data not fresh — returning CLOSE (all zeros) for all symbols.")
            return self.decisions

        if self.features is None:
            self.engineer_features()

        self.verticalize()
        all_predictions = []

        # Use regime stack loaded by __init__ via REGIME_STACK_PATH — no fallback
        stack_to_use = self.regime_stack if hasattr(self, "regime_stack") and self.regime_stack else []

        if not stack_to_use:
            logger.error("No regime stack available for decision making.")
            return self.decisions

        # 3. Iterate Regimes
        for regime in stack_to_use:
            try:
                # Filter signals for this regime
                # We do NOT save outputs during decision making
                # filter_signals returns DataFrame
                # We need to pass 'out_dir' only if saving, but here we don't save
                sig = self.filter_signals(regime, save=False)
                
                # Predict
                pred_df = self.predict(sig, regime)
                
                if not pred_df.empty:
                    all_predictions.append(pred_df)
                    
            except Exception as e:
                logger.error(f"Error processing regime {regime.get('id')}: {e}")

        if not all_predictions:
            logger.info("No predictions generated.")
            return self.decisions
            
        # 4. Aggregate Decisions
        combined_preds = pd.concat(all_predictions, axis=0, ignore_index=True)
        
        # Close prices for sizing AND for the price the executor anchors on.
        # Taken from the SAME row predict() ran on, selected by LABEL — see
        # _closes_at_timestamp for why this must never be positional again.
        latest_closes = _closes_at_timestamp(
            getattr(self, "raw", None),
            self.config.get("SYMBOLS", []),
            target_ts=self.vertical_features["timestamp"].max(),
        )

        # Recalculate sizes using fresh closes so notional ≈ CAPITAL
        from decimal import Decimal
        capital = self.config.get("CAPITAL", 100)
        lot_sizes = self.config.get("LOT_SIZES", {})
        size_map = {}
        for sym in self.config["SYMBOLS"]:
            close = latest_closes.get(sym, 0.0)
            lot_info = lot_sizes.get(sym, {})
            step = lot_info.get("step_size", 0)
            if close > 0 and step > 0:
                qty = capital / close
                prec = max(0, int(round(-Decimal(str(step)).log10()))) if step < 1 else 0
                qty = round(qty / step) * step
                qty = round(qty, prec)
                # Bump to meet exchange min_notional (same logic as _calculate_sizes)
                min_notional = lot_info.get("min_notional", 0.0)
                if min_notional > 0 and qty * close < min_notional:
                    import math as _math
                    qty = _math.ceil(min_notional / close / step) * step
                    qty = round(qty, prec)
                size_map[sym] = qty
            else:
                # Fallback to init-time size
                idx = self.config["SYMBOLS"].index(sym)
                init_sizes = self.config.get("SIZES", [])
                size_map[sym] = init_sizes[idx] if idx < len(init_sizes) else 0.0

        reverse = int(self.config.get("REVERSE", 1))
        for sym in self.config["SYMBOLS"]:
            # Filter predictions for this symbol
            sym_preds = combined_preds[combined_preds["symbol"] == sym]

            if sym_preds.empty:
                continue

            # Dual-horizon agreement gate (symmetric long+short). See
            # dual_gate_filter() for the rule.
            longs, shorts = dual_gate_filter(sym_preds)

            long_count = len(longs)
            short_count = len(shorts)

            # Logging
            if long_count > 0 or short_count > 0:
                specs = []
                if long_count > 0:
                    specs.extend([f"{r} long" for r in longs["regime"].unique()])
                if short_count > 0:
                    specs.extend([f"{r} short" for r in shorts["regime"].unique()])
                logger.info(f"Decision for {sym}: {' + '.join(sorted(list(set(specs))))}")

            # Net Quantity — position size = CAPITAL * |net_count|
            net_count = long_count - short_count
            base_size = float(size_map.get(sym, 0.0))
            final_qty = base_size * net_count * reverse
            
            price = latest_closes.get(sym, 0.0)
            self.decisions[sym] = [price, final_qty]

        # Telegram notification is handled by Symbiote._send_decisions_telegram()
        # to avoid duplicate messages.

        return self.decisions

    def clean(self) -> dict[str, list[float, float]]:
        symbols = self.config.get("SYMBOLS", [])
        self.decisions = {sym: [0.0, 0.0] for sym in symbols}
        return self.decisions
