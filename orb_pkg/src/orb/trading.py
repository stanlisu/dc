"""Orb trading: cross-timeframe live inference on top of OrbResearch."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional
import joblib
import json
import logging
import os
import time

import pandas as pd

from agamotto.utils import (
    _symbol_to_native,
    _timeframe_to_seconds,
    _round_down_to_timeframe_boundary,
)
from agamotto.research import _obf

try:
    from agamotto.lib_binance import fetch_futures_klines, klines_to_dataframe
except ImportError:
    try:
        from lib_binance import fetch_futures_klines, klines_to_dataframe
    except ImportError:
        pass

from .research import OrbResearch

logger = logging.getLogger(__name__)


# ``_closes_at_timestamp`` — re-exported, ONE implementation, in
# ``agamotto/trading.py``. It reads ``{symbol: close}`` off ``raw`` at
# ``target_ts`` by LABEL and raises if that label is absent, replacing the
# ``self.raw[col].iloc[-2]`` orb inherited from ``AgamottoTrading`` along with the
# premise "the last row is the incomplete candle".
#
# WHY THE PREMISE IS FALSE IN ORB, on BOTH data paths and for two different
# reasons: the REST path fetches ``limit + 1`` bars and drops the in-flight one
# (``_fetch_and_prepare_data``, :496-497) before assigning ``self.raw`` (:513),
# while the WS path never holds the in-flight bar at all (:385-386, "WS buffer
# only contains closed bars — do NOT drop the last bar") and assigns ``self.raw``
# at :399. Those two assignments are the ONLY ones, and both sit downstream of
# the drop — so ``iloc[-1]`` IS the just-closed bar and ``iloc[-2]`` is a full
# ``BASE_TF`` older.
#
# WHAT ``predict()`` TARGETS, verified for orb rather than assumed — orb is
# cross-TF, so the target row needed its own check. ``predict`` selects
# ``vertical_features["timestamp"].max()`` (:559); ``OrbResearch.verticalize``
# sets that column to ``self.features.index`` (research.py:392);
# ``_align_timeframes`` builds ``self.features`` on ``base_idx``, the BASE_TF
# features index (research.py:441-443, :577); and ``engineer_features`` preserves
# ``raw.index`` row-for-row. Higher TFs are ``merge_asof``'d ONTO that index and
# never widen it. So the target equals ``self.raw.index.max()`` — the settled bar
# that ``iloc[-2]`` misses by one, exactly as in agamotto.
#
# orb is not live, so this cost nothing; the same pairing did cost agamotto a
# median +23.8 bps per entry on ltp (dc ``f47d588``, hydra 2026-08-15).
#
# WHY IMPORTED AND NOT COPIED (2026-08-17). ``74ce8a1`` ported the helper into
# this module by duplication, and it promptly drifted: ``cb454fd`` hardened
# agamotto's copy the same day — an order-independent NaN fallback (label slicing
# a non-monotonic index silently falls back to POSITIONAL slicing and reaches
# FORWARD in time, rebuilding the lookahead inside the repair) plus a
# duplicate-label guard — and neither reached this copy. Two copies of the same
# rule is how engines drift apart; there is now one.
from agamotto.trading import _closes_at_timestamp  # noqa: F401


class OrbTrading(OrbResearch):
    """Live cross-timeframe inference — mirrors AgamottoTrading's interface."""

    def __init__(
        self,
        config: Dict[str, object],
        home_root: str,
        period: Optional[str] = None,
        skip_load: bool = False,
    ) -> None:
        super().__init__(config, home_root)
        self.period = self.config.get("WEIGHTS_PERIOD") or period
        self.trading_mode = self.config.get("TRADING_MODE", "both")
        self.models: dict = {}
        self.latest_predictions: pd.DataFrame | None = None
        self.latest_predictions_path: Optional[str] = None
        self.decisions: dict[str, list] = {}

        self.long_pred_threshold = float(
            self.config.get("LONG_PRED_THRESHOLD", 0.0))
        self.short_pred_threshold = float(
            self.config.get("SHORT_PRED_THRESHOLD", 0.0))

        self.regime_stack_path = self.config.get("REGIME_STACK_PATH")
        if self.regime_stack_path:
            if not os.path.isabs(self.regime_stack_path):
                self.regime_stack_path = os.path.join(
                    home_root, self.regime_stack_path)
            self._load_regime_stack()
            if not self.regime_stack:
                logger.error(
                    f"Regime stack at {self.regime_stack_path} "
                    "loaded NO valid models!")
        else:
            raise ValueError(
                "REGIME_STACK_PATH not provided in configuration.")

        self._calculate_sizes()

        if not skip_load:
            try:
                self.load_data(limit=700)
            except Exception as e:
                logger.warning(
                    f"Failed to load initial data in __init__: {e}")

        logger.info(
            f"Orb initiated - Period: {self.period}, "
            f"Trading Mode: {self.trading_mode}, "
            f"Timeframes: {self.timeframes}")

    # ------------------------------------------------------------------
    # Regime stack loading (same logic as AgamottoTrading)
    # ------------------------------------------------------------------

    def _load_regime_stack(self) -> None:
        from utils.lib import load_regime_stack
        stack_data = load_regime_stack(self.regime_stack_path)

        self.regime_stack: list[dict] = []
        self.models = {"long": {}, "short": {}}

        for row in stack_data:
            # Skip rows that have no model or directory
            if not row.get("model") or not row.get("directory"):
                continue
            target_dir = row["directory"]
            position = row["position"]
            model_name = row["model"]
            threshold = float(row["optimal_threshold"])

            research_path = os.path.join(
                self.home_root, "gauntlet", target_dir)
            if not os.path.isdir(research_path):
                research_path = os.path.join(self.home_root, target_dir)

            weights_dir = os.path.join(
                research_path, "weights", self.period)
            if not os.path.isdir(weights_dir):
                raise FileNotFoundError(
                    f"Missing weights directory for period "
                    f"{self.period} in {research_path}")

            regime_name = row.get("regime")
            if not regime_name:
                raise ValueError(
                    f"Missing regime name in stack entry for {target_dir}")

            pos_dir = os.path.join(weights_dir, regime_name)
            if not os.path.isdir(pos_dir):
                raise FileNotFoundError(
                    f"Regime folder {regime_name} not found in {weights_dir}")

            low_name = model_name.lower()
            meta_path = os.path.join(pos_dir, f"{low_name}_meta.pkl")
            model_path = os.path.join(pos_dir, f"{low_name}_model.pkl")
            scaler_path = os.path.join(pos_dir, f"{low_name}_scaler.pkl")

            if not os.path.exists(model_path):
                meta_path = os.path.join(
                    pos_dir, f"{model_name}_meta.pkl")
                model_path = os.path.join(
                    pos_dir, f"{model_name}_model.pkl")
                scaler_path = os.path.join(
                    pos_dir, f"{model_name}_scaler.pkl")

            if not (os.path.exists(meta_path)
                    and os.path.exists(model_path)
                    and os.path.exists(scaler_path)):
                try:
                    files = os.listdir(pos_dir)
                    logger.error(f"Files in {pos_dir}: {files}")
                except Exception:
                    pass
                raise FileNotFoundError(
                    f"Missing model files for {model_name} in {pos_dir}")

            artifact = {
                "model": joblib.load(model_path),
                "scaler": joblib.load(scaler_path),
                "metadata": joblib.load(meta_path),
            }

            strat_config_path = os.path.join(
                research_path, "setting.json")
            with open(strat_config_path, "r") as f:
                strat_config = json.load(f)

            filter_val = strat_config.get(
                "LONG_FILTER" if position == "long" else "SHORT_FILTER")

            self.regime_stack.append({
                "id": f"{target_dir}_{position}_{model_name}",
                "directory": target_dir,
                "regime": row.get("regime", "unknown"),
                "position": position,
                "model_name": model_name,
                "threshold": threshold,
                "artifact": artifact,
                "filter": filter_val,
                "config": strat_config,
            })
            logger.info(
                f"Loaded regime: "
                f"{target_dir}_{position}_{model_name} "
                f"(Threshold: {threshold})")

    # ------------------------------------------------------------------
    # Size calculation (same logic as AgamottoTrading)
    # ------------------------------------------------------------------

    def _calculate_sizes(self) -> None:
        if "SIZES" in self.config and self.config["SIZES"]:
            logger.info(
                f"SIZES already provided in config: "
                f"{len(self.config['SIZES'])} symbols")
            return

        try:
            from lib_binance import fetch_latest_closes, fetch_price_lot_sizes
        except ImportError:
            try:
                from agamotto.lib_binance import (
                    fetch_latest_closes,
                    fetch_price_lot_sizes,
                )
            except ImportError:
                logger.warning(
                    "Cannot import lib_binance, SIZES not calculated")
                self.config["SIZES"] = [0.0] * len(
                    self.config.get("SYMBOLS", []))
                return

        from decimal import Decimal

        symbols = self.config.get("SYMBOLS", [])
        capital = self.config.get("CAPITAL", 100)
        if not symbols:
            self.config["SIZES"] = []
            return

        logger.info(
            f"Calculating SIZES for {len(symbols)} symbols "
            f"(CAPITAL={capital})...")

        closes = fetch_latest_closes(symbols, interval="1d")
        lot_sizes = fetch_price_lot_sizes(symbols)

        sizes = []
        for symbol in symbols:
            close = closes.get(symbol)
            lot_info = lot_sizes.get(symbol, {})
            step_size = lot_info.get("step_size", 0)
            if close and close > 0 and step_size > 0:
                qty = capital / close
                precision = max(
                    0,
                    int(round(-Decimal(str(step_size)).log10()))
                ) if step_size < 1 else 0
                rounded_qty = round(qty / step_size) * step_size
                rounded_qty = round(rounded_qty, precision)
                sizes.append(rounded_qty)
            else:
                sizes.append(0.0)

        self.config["SIZES"] = sizes
        self.config["LOT_SIZES"] = lot_sizes
        logger.info(f"Calculated SIZES for {len(sizes)} symbols")

    # ------------------------------------------------------------------
    # Data loading — fetches ALL timeframes
    # ------------------------------------------------------------------

    def load_data(self, limit: int = 700) -> None:
        self._fetch_and_prepare_data(limit=limit)

        tf_seconds = _timeframe_to_seconds(
            self.config.get("TIME_UNIT", "1d"))

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

    def _fetch_and_prepare_data(self, limit: int = 700) -> None:
        """Fetch klines for ALL timeframes, then align."""
        from agamotto import AgamottoResearch

        symbols = self.config["SYMBOLS"]
        if not symbols:
            raise RuntimeError("No symbols configured for trading.")

        # --- WS buffer path ---
        kline_buffer = getattr(self, '_kline_buffer', None)
        if kline_buffer is not None:
            native_symbols = [
                _symbol_to_native(s) for s in symbols
                if _symbol_to_native(s)
            ]
            if kline_buffer.is_ready_multi_tf(
                    native_symbols, self.timeframes):
                self._tf_instances = {}
                all_ok = True
                for tf in self.timeframes:
                    frames = []
                    for native in native_symbols:
                        df = kline_buffer.get_dataframe(native, tf)
                        if df is None or df.empty:
                            all_ok = False
                            break
                        # Rename {sym}_{tf}_{col} -> {sym}_{col}
                        rename_map = {
                            c: c.replace(f"_{tf}_", "_", 1)
                            for c in df.columns if f"_{tf}_" in c
                        }
                        if rename_map:
                            df = df.rename(columns=rename_map)
                        frames.append(df)
                    if not all_ok:
                        break

                    combined = pd.concat(frames, axis=1).sort_index()
                    combined = combined[
                        ~combined.index.duplicated(keep="last")]
                    if combined.index.tz is not None:
                        combined.index = combined.index.tz_localize(None)
                    combined = combined.tail(limit)
                    # WS buffer only contains closed bars — do NOT drop
                    # the last bar (unlike REST which includes the open bar)
                    tf_secs = _timeframe_to_seconds(tf)
                    combined["close_timestamp"] = (
                        combined.index + pd.Timedelta(seconds=tf_secs))
                    tf_config = {**self.config, "TIME_UNIT": tf}
                    inst = AgamottoResearch(tf_config, self.home_root)
                    inst.raw = combined
                    self._tf_instances[tf] = inst

                if all_ok and len(self._tf_instances) == len(
                        self.timeframes):
                    self.raw = self._tf_instances[self.base_tf].raw
                    # Verify WS data is fresh before committing.
                    # WS buffer may not yet have the new bar's open event,
                    # leaving the last closed bar one step behind.
                    # If stale, fall through to REST.
                    _base_tf_sec = _timeframe_to_seconds(
                        self.config.get("TIME_UNIT", "15m"))
                    _now_unix = int(
                        pd.Timestamp.utcnow().tz_localize(None).timestamp())
                    _floor_unix = (_now_unix // _base_tf_sec) * _base_tf_sec
                    _expected = pd.Timestamp(
                        _floor_unix - _base_tf_sec, unit='s')
                    if self.raw.index.max() >= _expected:
                        self.engineer_features()
                        self.verticalize()
                        self._dump_debug_features()
                        return
                    logger.warning(
                        f"WS buffer stale (last bar {self.raw.index.max()}, "
                        f"expected >= {_expected}), falling back to REST")
                else:
                    logger.warning(
                        "WS buffer incomplete for orb multi-TF, "
                        "falling back to REST")

        # --- REST path (original) ---
        end_ts = pd.Timestamp.utcnow()
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("UTC")
        else:
            end_ts = end_ts.tz_convert("UTC")
        end_ms = int(end_ts.timestamp() * 1000)

        # Fetch each TF and build an AgamottoResearch-like instance
        self._tf_instances = {}

        for tf in self.timeframes:
            tf_seconds = _timeframe_to_seconds(tf)
            # Fetch one extra bar so that after dropping the open/incomplete bar
            # we still have exactly `limit` closed bars.  rolling(limit) on the
            # last bar then uses the same window as the research pipeline (which
            # has full history), giving identical price_range_pct_q50 values.
            rest_limit = limit + 1
            start_ts = end_ts - pd.Timedelta(seconds=tf_seconds * rest_limit)
            if start_ts.tzinfo is None:
                start_ts = start_ts.tz_localize("UTC")
            start_ms = int(start_ts.timestamp() * 1000)

            frames: list[pd.DataFrame] = []

            def _fetch_symbol(sym, tf=tf, start_ms=start_ms, _rlimit=rest_limit):
                native = _symbol_to_native(sym)
                if not native:
                    return None
                rows = fetch_futures_klines(
                    native, tf, start_ms, end_ms, limit=_rlimit)
                df = klines_to_dataframe(rows, native, tf)
                if df.empty:
                    return None
                rename_map = {
                    f"{native}_{tf}_{col}": f"{native}_{col}"
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
                        logger.error(
                            f"Failed to fetch {tf} klines for {sym}: {e}")

            if not frames:
                raise RuntimeError(
                    f"Failed to fetch trading data for TF={tf}.")

            combined = pd.concat(frames, axis=1).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.index = combined.index.tz_localize(None)
            combined = combined.tail(rest_limit)

            # Drop the current incomplete bar so all TFs use only closed bars,
            # matching the training pipeline (AgamottoTrading) behaviour.
            # After this drop we have exactly `limit` closed bars.
            if len(combined) > 1:
                combined = combined.iloc[:-1]

            # Stamp close_timestamp on each row: open_time + tf_duration.
            # engineer_features() carries this into features, where
            # _align_timeframes uses it for causal merge_asof alignment.
            tf_secs = _timeframe_to_seconds(tf)
            combined["close_timestamp"] = (
                combined.index + pd.Timedelta(seconds=tf_secs)
            )

            tf_config = {**self.config, "TIME_UNIT": tf}
            inst = AgamottoResearch(tf_config, self.home_root)
            inst.raw = combined
            self._tf_instances[tf] = inst

        # Set self.raw for parent compatibility (base TF)
        self.raw = self._tf_instances[self.base_tf].raw

        # Engineer + align
        self.engineer_features()
        self.verticalize()
        self._dump_debug_features()

        last_ts = self.raw.index.max()
        logger.info(
            f"Loaded cross-TF data up to {last_ts} "
            f"({len(self.raw)} rows, TFs={self.timeframes})")

    def _dump_debug_features(self) -> None:
        """Dump latest features for tesseract comparison."""
        if (not hasattr(self, 'vertical_features')
                or self.vertical_features is None
                or self.vertical_features.empty):
            return
        now_utc = pd.Timestamp.now(tz="UTC").tz_localize(None)
        tf_str = self.config.get("TIME_UNIT", "15m")
        current_start = _round_down_to_timeframe_boundary(
            now_utc, tf_str)
        target_ts = current_start - pd.Timedelta(
            seconds=_timeframe_to_seconds(tf_str))
        latest_row = self.vertical_features[
            self.vertical_features["timestamp"] == target_ts]
        project = self.config.get("PROJECT", "debug")
        algo = self.config.get("STRATEGY", "orb")
        ts_str = current_start.strftime("%Y%m%d_%H%M")
        csv_path = os.path.join(
            self.home_root, project,
            f"debug_features_{algo}_{tf_str}_{ts_str}.csv")
        latest_row.to_csv(csv_path, index=False)
        logger.info(f"Dumped {len(latest_row)} rows to {csv_path}")

    # ------------------------------------------------------------------
    # Prediction + decision (same logic as AgamottoTrading)
    # ------------------------------------------------------------------

    def predict(
        self,
        filtered_signals: pd.DataFrame,
        regime: dict,
    ) -> pd.DataFrame:
        if filtered_signals.empty:
            return pd.DataFrame()

        target_ts = self.vertical_features["timestamp"].max()
        target_row = filtered_signals[
            filtered_signals["timestamp"] == target_ts].copy()
        if target_row.empty:
            return pd.DataFrame()

        # Obfuscation: add coded aliases so the meta.feature_columns selection +
        # scaler name-check match either coded (new) or real (old) weights.
        target_row = _obf().add_feature_aliases(target_row)

        try:
            artifact = regime["artifact"]
            f_cols = artifact["metadata"]["feature_columns"]
            missing = [c for c in f_cols if c not in target_row.columns]
            if missing:
                logger.warning(
                    f"Regime {regime['id']} missing features: {missing}")
                return pd.DataFrame()

            X = target_row[f_cols].copy()
            if X.isnull().values.any():
                X.fillna(0.0, inplace=True)

            X_scaled = artifact["scaler"].transform(X)
            preds = artifact["model"].predict(X_scaled)

            target_row["prediction"] = preds
            target_row["strategy_id"] = regime["id"]
            target_row["regime"] = regime.get("regime", "unknown")
            target_row["opt_threshold"] = regime["threshold"]
            return target_row

        except Exception as e:
            logger.error(
                f"Prediction failed for {regime.get('id')}: {e}")
            return pd.DataFrame()

    def make_decision(
        self,
        label: str = "both",
    ) -> dict[str, list]:
        self.decisions = {
            sym: [0.0, 0.0]
            for sym in self.config.get("SYMBOLS", [])
        }

        if not getattr(self, "_data_fresh", True):
            logger.warning(
                "Data not fresh — returning CLOSE (all zeros) for all symbols.")
            return self.decisions

        if self.features is None:
            self.engineer_features()

        self.verticalize()
        all_predictions: list[pd.DataFrame] = []

        stack_to_use = getattr(self, "regime_stack", [])
        if not stack_to_use:
            logger.error("No regime stack available for decision making.")
            return self.decisions

        for regime in stack_to_use:
            try:
                sig = self.filter_signals(regime, save=False)
                pred_df = self.predict(sig, regime)
                if not pred_df.empty:
                    all_predictions.append(pred_df)
            except Exception as e:
                logger.error(
                    f"Error processing regime {regime.get('id')}: {e}")

        if not all_predictions:
            logger.info("No predictions generated.")
            return self.decisions

        combined_preds = pd.concat(
            all_predictions, axis=0, ignore_index=True)

        # Close prices for sizing AND for the price the executor anchors on.
        # Taken from the SAME row predict() ran on, selected by LABEL — see
        # _closes_at_timestamp for why this must never be positional again.
        latest_closes = _closes_at_timestamp(
            getattr(self, "raw", None),
            self.config.get("SYMBOLS", []),
            target_ts=self.vertical_features["timestamp"].max(),
        )

        from decimal import Decimal
        capital = self.config.get("CAPITAL", 100)
        lot_sizes = self.config.get("LOT_SIZES", {})
        size_map: dict[str, float] = {}
        for sym in self.config["SYMBOLS"]:
            close = latest_closes.get(sym, 0.0)
            lot_info = lot_sizes.get(sym, {})
            step = lot_info.get("step_size", 0)
            if close > 0 and step > 0:
                qty = capital / close
                prec = max(
                    0,
                    int(round(-Decimal(str(step)).log10()))
                ) if step < 1 else 0
                qty = round(qty / step) * step
                qty = round(qty, prec)
                size_map[sym] = qty
            else:
                idx = self.config["SYMBOLS"].index(sym)
                init_sizes = self.config.get("SIZES", [])
                size_map[sym] = (
                    init_sizes[idx] if idx < len(init_sizes) else 0.0)

        retain_threshold = self.config.get("RETAIN_THRESHOLD")
        if retain_threshold is not None:
            retain_threshold = float(retain_threshold)

        for sym in self.config["SYMBOLS"]:
            sym_preds = combined_preds[combined_preds["symbol"] == sym]
            if sym_preds.empty:
                continue

            longs = sym_preds[
                (sym_preds["position"] == "long")
                & (sym_preds["prediction"] > sym_preds["opt_threshold"])
            ]
            shorts = sym_preds[
                (sym_preds["position"] == "short")
                & (sym_preds["prediction"] < sym_preds["opt_threshold"])
            ]

            long_count = len(longs)
            short_count = len(shorts)

            if long_count > 0 or short_count > 0:
                specs = []
                if long_count > 0:
                    specs.extend(
                        [f"{r} long" for r in longs["regime"].unique()])
                if short_count > 0:
                    specs.extend(
                        [f"{r} short" for r in shorts["regime"].unique()])
                logger.info(
                    f"Decision for {sym}: "
                    f"{' + '.join(sorted(set(specs)))}")

            net_count = long_count - short_count
            base_size = float(size_map.get(sym, 0.0))
            final_qty = base_size * net_count
            price = latest_closes.get(sym, 0.0)
            self.decisions[sym] = [price, final_qty]

            # Retain threshold: close an open position early when conviction
            # drops below retain_threshold (asymmetric close signal).
            if retain_threshold is not None:
                venom = getattr(self, "venom", None)
                trade = venom.get(sym) if venom is not None else None
                if trade is not None and trade.status.value == "OPEN":
                    # Compute the net score as average prediction across all
                    # long regimes minus average across all short regimes.
                    long_scores = sym_preds[sym_preds["position"] == "long"]["prediction"]
                    short_scores = sym_preds[sym_preds["position"] == "short"]["prediction"]
                    avg_long = float(long_scores.mean()) if not long_scores.empty else 0.0
                    avg_short = float(short_scores.mean()) if not short_scores.empty else 0.0
                    # net_score > 0 → long signal, < 0 → short signal
                    net_score = avg_long - avg_short

                    if trade.position_qty > 0:  # currently LONG
                        if net_score < retain_threshold:
                            logger.info(
                                f"Closing {sym} early: score {net_score:.4f} "
                                f"below retain threshold {retain_threshold:.4f}")
                            self.decisions[sym] = [0.0, 0.0]
                    else:  # currently SHORT
                        if net_score > -retain_threshold:
                            logger.info(
                                f"Closing {sym} early: score {net_score:.4f} "
                                f"above retain threshold {-retain_threshold:.4f}")
                            self.decisions[sym] = [0.0, 0.0]

        return self.decisions

    def clean(self) -> dict[str, list]:
        symbols = self.config.get("SYMBOLS", [])
        self.decisions = {sym: [0.0, 0.0] for sym in symbols}
        return self.decisions
