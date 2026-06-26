"""MjolnirTrading: live inference engine for 5-second bar ML predictions.

Architecture:
- Loads trained models from weights/ (latest window directory)
- Loads optimal_regime_stack.json for regime filters and thresholds
- Maintains per-symbol rolling bar buffer (deque)
- On predict(): assembles DataFrame, computes MjolnirFeatures, runs models
- Returns ('long'/'short', threshold) when ≥ MIN_SIGNAL_COUNT models agree
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from mjolnir.core.features import MjolnirFeatures, _TF_SECONDS
from mjolnir.core.multi_tf_merge import merge_cross_tf_features
from mjolnir.core.research import MjolnirResearch, _DEFAULT_FEATURE_WINDOWS
from mjolnir.core.utils import normalize_symbol

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PredictDiag:
    """Diagnostic payload from predict(). Frozen — immutable after creation."""
    side: str            # "long" or "short"
    threshold: float     # aggregate best threshold (min long / max short)
    y_pred: float        # raw model score of the strongest-signal model
    y_pred_thresh: float  # per-model threshold of the strongest-signal model
    regime: str          # regime name of the strongest-signal model
    model_name: str      # model key (e.g. "r1_long_lightgbm")
    n_triggered: int = 1  # regimes that triggered the winning side (conviction)

_BTC = "BINANCE_PERP_BTC_USDT"
_BTC_NATIVE = normalize_symbol(_BTC)

# Dedup set for the regime-filter silent-except warning in predict().
# Keyed by (regime_name, position, exception_class_name). Lives for the
# process lifetime — long enough to surface the failure in offline replay
# and any live bridge restart cycle, short enough to not need explicit
# clearing.
_LOGGED_FILTER_FAILURES: set = set()


class MjolnirTrading:
    """Live inference engine: predict on 5s bars using trained Mjolnir models.

    Args:
        config:    Dict loaded from setting.json. Must contain SYMBOLS,
                   TARGET_HORIZON_BARS, FEE, OUTPUT_DIR. Feature windows are
                   constants (mjolnir/core/research.py:_DEFAULT_FEATURE_WINDOWS).
        home_root: Repository root directory (for resolving relative paths).
    """

    BUFFER_MAXLEN: int = 1000   # bars to retain per symbol (~83 minutes)
    MIN_WARMUP_BARS: int = 120  # minimum bars before prediction starts (10 minutes)

    def __init__(self, config: dict, home_root: str) -> None:
        self.config = config
        self.home_root = home_root
        # Per-TF buffers keyed as self._tf_buffers[tf][symbol] (Bug 2 — live
        # cross-TF merge needs separate higher-TF bar streams). The base TF
        # is the one named in cfg["TIME_UNIT"] when set, else "5s" for back
        # compat with older tests that don't carry TIME_UNIT.
        self._base_tf: str = config.get("TIME_UNIT", "5s")
        self._multi_tfs: List[str] = list(config.get("MULTI_TF_BARS", []))
        all_tfs = [self._base_tf] + [tf for tf in self._multi_tfs
                                      if tf != self._base_tf]
        self._tf_buffers: Dict[str, Dict[str, Deque]] = {tf: {} for tf in all_tfs}
        self._tf_timestamps: Dict[str, Dict[str, Deque]] = {tf: {} for tf in all_tfs}
        # Aliases preserved for back compat with code that touches base buffers
        # directly (e.g. tests asserting on _buffers / _timestamps).
        self._buffers: Dict[str, Deque] = self._tf_buffers[self._base_tf]
        self._timestamps: Dict[str, Deque] = self._tf_timestamps[self._base_tf]

        if "FEATURE_WINDOWS" in config:
            raise ValueError(
                "FEATURE_WINDOWS is deprecated; remove it from setting.json — "
                "windows are constants now (mjolnir/core/research.py)"
            )
        self._feat_engine = MjolnirFeatures(
            feature_windows=list(_DEFAULT_FEATURE_WINDOWS),
            target_horizon=int(config.get("TARGET_HORIZON_BARS", 1)),
            fee_rate=float(config.get("FEE", 2.0)) / 10000.0,
            bar_tf=self._base_tf,
            target_tf=self._base_tf,
        )
        # One trivial-window engine per cross-TF (mirrors research.py:287-297).
        self._tf_engines: Dict[str, MjolnirFeatures] = {
            tf: MjolnirFeatures(
                feature_windows=[1],
                target_horizon=int(config.get("TARGET_HORIZON_BARS", 1)),
                fee_rate=float(config.get("FEE", 2.0)) / 10000.0,
                prefix=tf,
                bar_tf=tf,
                target_tf=tf,
            )
            for tf in self._multi_tfs
        }

        # Load regime stack + models
        out_dir = config.get("OUTPUT_DIR", home_root)
        self._regime_stack: List[dict] = self._load_regime_stack(out_dir)
        self._models: Dict[str, object] = {}    # key: "{regime_dir_name}_{model_lower}"
        self._scalers: Dict[str, object] = {}
        self._metas: Dict[str, dict] = {}
        self._load_models(out_dir)

        # Research instance for regime filter logic (reuse _apply_filter_mask)
        self._research = MjolnirResearch(config=config, home_root=home_root)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_bar(
        self,
        symbol: str,
        bar: dict,
        timestamp: pd.Timestamp,
        tf: Optional[str] = None,
    ) -> None:
        """Append a new completed bar to the symbol's rolling buffer.

        ``tf`` selects which timeframe buffer to append to; defaults to the
        configured base TF. Cross-TF builders push their completed bars in
        with ``tf="15s"`` / ``tf="30s"`` / etc. so live cross-TF feature
        merge has real higher-TF bars available at predict time.
        """
        if tf is None:
            tf = self._base_tf
        if tf not in self._tf_buffers:
            raise KeyError(
                f"add_bar: tf={tf!r} not in configured buffers "
                f"{sorted(self._tf_buffers)!r} — TIME_UNIT or MULTI_TF_BARS "
                "missing this TF"
            )
        bufs = self._tf_buffers[tf]
        tss = self._tf_timestamps[tf]
        if symbol not in bufs:
            bufs[symbol] = deque(maxlen=self.BUFFER_MAXLEN)
            tss[symbol] = deque(maxlen=self.BUFFER_MAXLEN)
        bufs[symbol].append(bar)
        tss[symbol].append(timestamp)

    def snapshot_inputs(self, symbol: str) -> Optional[dict]:
        """Grab an immutable, picklable copy of every buffer ``predict`` reads
        for *symbol*: its own base buffer + the BTC buffer (cross-features) +
        each cross-TF buffer. MUST be called on the bridge's event loop — that
        is the only race-free point vs ``add_bar``'s deque mutation. Returns
        ``None`` if the symbol's own buffer is below warmup.

        Bar dicts are appended immutably (never mutated in place), so copying
        their references is safe; a process pool deep-copies them at pickle
        time anyway. The conditional BTC / cross-TF inclusion gates mirror the
        original ``predict`` exactly.
        """
        if symbol not in self._buffers:
            return None
        buf = self._buffers[symbol]
        if len(buf) < self.MIN_WARMUP_BARS:
            return None
        inp: dict = {
            "buf": list(buf),
            "ts": list(self._timestamps[symbol]),
            "btc_buf": None,
            "btc_ts": None,
            "tf": {},
        }
        if (symbol != _BTC
                and _BTC in self._buffers
                and len(self._buffers[_BTC]) >= self.MIN_WARMUP_BARS):
            inp["btc_buf"] = list(self._buffers[_BTC])
            inp["btc_ts"] = list(self._timestamps[_BTC])
        for cross_tf in self._multi_tfs:
            tf_buf = self._tf_buffers[cross_tf].get(symbol)
            tf_ts = self._tf_timestamps[cross_tf].get(symbol)
            if not tf_buf or len(tf_buf) < 2:
                continue
            inp["tf"][cross_tf] = (list(tf_buf), list(tf_ts))
        return inp

    def predict(self, symbol: str) -> Optional[PredictDiag]:
        """Snapshot + compute in one call (back-compat / single-process path).

        Equivalent to ``predict_from_inputs(symbol, snapshot_inputs(symbol))``.
        The bridge splits the two so the snapshot runs on the event loop
        (race-free vs ``add_bar``) while ``predict_from_inputs`` runs in a
        worker pool.
        """
        inp = self.snapshot_inputs(symbol)
        if inp is None:
            return None
        return self.predict_from_inputs(symbol, inp)

    def predict_from_inputs(
        self, symbol: str, inputs: dict,
    ) -> Optional[PredictDiag]:
        """Pure compute half of ``predict``: build features from an ``inputs``
        snapshot (from :meth:`snapshot_inputs`) and run the regime stack.

        Reads ONLY read-only model / feature-engine / research / config state,
        so it is safe to run in a worker process whose own buffers are empty —
        all bar data arrives via ``inputs``.
        """
        # Build DataFrame from the snapshot
        df = pd.DataFrame(inputs["buf"])
        idx = pd.DatetimeIndex(inputs["ts"], name=None)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        df.index = idx

        # Compute features
        try:
            feats = self._feat_engine.compute(df)
        except Exception as exc:
            logger.error("Feature computation failed for %s: %s", symbol, exc)
            return None

        # Inject BTC cross-features for non-BTC symbols. The warmup / non-BTC
        # gates live in snapshot_inputs (btc_buf is only set when eligible).
        if inputs.get("btc_buf"):
            btc_df = pd.DataFrame(inputs["btc_buf"])
            btc_idx = pd.DatetimeIndex(inputs["btc_ts"])
            if btc_idx.tz is None:
                btc_idx = btc_idx.tz_localize("UTC")
            btc_df.index = btc_idx
            try:
                btc_feats = self._feat_engine.compute(btc_df)
                feats = self._feat_engine.add_btc_cross_features(feats, btc_feats)
            except Exception as exc:
                logger.warning("BTC cross-features failed for %s: %s", symbol, exc)

        # Bug 2 (audit A1) — merge cross-TF features into the base frame using
        # the EXACT same helper research uses. Without this, models trained on
        # MULTI_TF_BARS see only base-TF columns at inference and either
        # silently drop them (pre-Bug 5) or raise (post-Bug 5). Cross-TF bars
        # are pushed into per-TF buffers by the bridge.
        tf_feats_map: Dict[str, pd.DataFrame] = {}
        for cross_tf, (tf_buf, tf_ts) in inputs.get("tf", {}).items():
            tf_df = pd.DataFrame(tf_buf)
            tf_idx = pd.DatetimeIndex(tf_ts)
            if tf_idx.tz is None:
                tf_idx = tf_idx.tz_localize("UTC")
            tf_df.index = tf_idx
            try:
                tf_feats_map[cross_tf] = self._tf_engines[cross_tf].compute(tf_df)
            except Exception as exc:
                logger.warning(
                    "Cross-TF feature compute failed for %s TF=%s: %s",
                    symbol, cross_tf, exc,
                )
        if tf_feats_map:
            feats = merge_cross_tf_features(feats, tf_feats_map)

        # Predict on the last complete bar (second-to-last row: -2)
        if len(feats) < 2:
            return None
        row = feats.iloc[-2]

        # Run each regime entry in the stack
        long_count = 0
        short_count = 0
        best_long_thresh: Optional[float] = None
        best_short_thresh: Optional[float] = None
        # Track the strongest-signal model per side: (pred, thresh, regime, model_key)
        best_long_diag: Optional[Tuple[float, float, str, str]] = None
        best_short_diag: Optional[Tuple[float, float, str, str]] = None

        for entry in self._regime_stack:
            regime_name = entry.get("regime", "")
            position = entry.get("position", "long")
            model_name = entry.get("model", "LightGBM").lower()
            threshold = float(entry.get("threshold", 0.001))

            # Check regime condition over the FULL feature window, then read the
            # last-complete-bar row (-2) — NOT on a one-row DataFrame.
            # Quantile-based filters (wide_spread, tight_spread,
            # high/low_liquidation_pressure) are panel-relative, e.g.
            # `relative_spread > relative_spread.quantile(0.5)`. Evaluated on a
            # single row, quantile(0.5) == that value, so `x > x` is ALWAYS
            # False and the regime never activates live — silently disabling
            # every quantile regime (the 2026-06-15 all-short bug: the only-long
            # entry was wide_spread_long, so long_count was pinned at 0). The
            # backtest passes the full panel (research.py:546/884), so it worked
            # there. Passing the window restores that panel-relative semantics;
            # we select iloc[-2] to mirror `row` above.
            # Instrumentation: log first occurrence of each (regime, position,
            # exception_class) so offline-replay "None for every bar" failures
            # surface their root cause without flooding logs.
            try:
                mask = self._research._apply_filter_mask(feats, regime_name, position)
                if not bool(mask.iloc[-2]):
                    continue
            except Exception as exc:
                key = (regime_name, position, type(exc).__name__)
                if key not in _LOGGED_FILTER_FAILURES:
                    _LOGGED_FILTER_FAILURES.add(key)
                    logger.warning(
                        "regime filter raised — regime=%s position=%s: %s: %s "
                        "(first occurrence; suppressing duplicates)",
                        regime_name, position, type(exc).__name__, exc,
                    )
                continue

            # Load model
            dir_key = f"{regime_name}_{position}"
            model_key = f"{dir_key}_{model_name}"
            if model_key not in self._models:
                continue

            model = self._models[model_key]
            scaler = self._scalers[model_key]
            meta = self._metas[model_key]
            feat_cols = meta.get("feature_columns", [])

            # Bug 5 (audit B2) — fail fast if the live feats DataFrame is
            # missing any feature the model was trained on. Silently dropping
            # to `available` produced a feature-count mismatch at scaler.transform
            # which was swallowed at debug, causing the regime to emit zero
            # signals forever with no operator-visible warning.
            missing = [c for c in feat_cols if c not in feats.columns]
            assert not missing, (
                f"regime {regime_name!r} ({position}) missing features: "
                f"{sorted(missing)} (model expected {len(feat_cols)} cols, "
                f"live feats has {len(feats.columns)} cols)"
            )
            X = row[feat_cols].values.reshape(1, -1).astype(float)
            X = np.where(np.isinf(X), np.nan, X)
            X = np.nan_to_num(X, nan=0.0)
            X_scaled = scaler.transform(X)
            pred = float(model.predict(X_scaled)[0])

            if position == "long" and pred > threshold:
                long_count += 1
                best_long_thresh = (threshold if best_long_thresh is None
                                    else min(best_long_thresh, threshold))
                margin = pred - threshold
                if best_long_diag is None or margin > (best_long_diag[0] - best_long_diag[1]):
                    best_long_diag = (pred, threshold, regime_name, model_key)
            elif position == "short" and pred < threshold:
                short_count += 1
                best_short_thresh = (threshold if best_short_thresh is None
                                     else max(best_short_thresh, threshold))
                margin = threshold - pred
                if best_short_diag is None or margin > (best_short_diag[1] - best_short_diag[0]):
                    best_short_diag = (pred, threshold, regime_name, model_key)

        # SHORT-drought diagnostic — log per-regime vote counts BEFORE the
        # gate so ops can grep "predict votes" to distinguish (a) shorts
        # never being counted (predict() bug) from (b) shorts counted but
        # losing `long_count > short_count` (stack-composition issue).
        # bridge result lines alone can't tell these apart. Cheap; one INFO
        # per symbol per decision.
        logger.info(
            "predict votes %s: long_count=%d short_count=%d "
            "best_long=%s best_short=%s",
            symbol, long_count, short_count,
            best_long_thresh, best_short_thresh,
        )

        # Bug 7 (audit D3) — required key, no silent default per CLAUDE.md.
        if "MIN_SIGNAL_COUNT" not in self.config:
            raise KeyError(
                "MIN_SIGNAL_COUNT is required in setting.json (typically 1)"
            )
        min_count = int(self.config["MIN_SIGNAL_COUNT"])
        result: Optional[PredictDiag] = None
        # Bug 6 (audit B5/D5) — explicit None-check, NOT `or 0.001`. A regime
        # with threshold=0.0 is legitimate; the falsy collapse would silently
        # rewrite it to 0.001 (FEE-bug pattern). If the count gate triggered
        # but the matching best_*_thresh is still None, that is a code bug —
        # raise rather than fabricate a magic value.
        if long_count >= min_count and long_count > short_count:
            if best_long_thresh is None:
                raise RuntimeError(
                    f"long_count={long_count} but best_long_thresh is None — "
                    "regime stack accounting bug"
                )
            if best_long_diag is None:
                raise RuntimeError(
                    f"long_count={long_count} but best_long_diag is None — "
                    "regime stack accounting bug"
                )
            pred_val, pred_thresh, regime, mkey = best_long_diag
            result = PredictDiag(
                side="long", threshold=best_long_thresh,
                y_pred=pred_val, y_pred_thresh=pred_thresh,
                regime=regime, model_name=mkey,
                n_triggered=long_count,
            )
        elif short_count >= min_count and short_count > long_count:
            if best_short_thresh is None:
                raise RuntimeError(
                    f"short_count={short_count} but best_short_thresh is None — "
                    "regime stack accounting bug"
                )
            if best_short_diag is None:
                raise RuntimeError(
                    f"short_count={short_count} but best_short_diag is None — "
                    "regime stack accounting bug"
                )
            pred_val, pred_thresh, regime, mkey = best_short_diag
            result = PredictDiag(
                side="short", threshold=best_short_thresh,
                y_pred=pred_val, y_pred_thresh=pred_thresh,
                regime=regime, model_name=mkey,
                n_triggered=short_count,
            )

        # Apply REVERSE: -1 flips long→short and short→long
        reverse = int(self.config.get("REVERSE", 1))
        if result is not None and reverse == -1:
            result = PredictDiag(
                side="short" if result.side == "long" else "long",
                threshold=result.threshold,
                y_pred=result.y_pred, y_pred_thresh=result.y_pred_thresh,
                regime=result.regime, model_name=result.model_name,
                n_triggered=result.n_triggered,
            )

        # Note: IPC send moved to MjolnirBridge._inference_loop after Phase 6
        # so the bridge can attach a per-symbol qty (computed from
        # CAPITAL + exchange-info step/precision) to the payload.
        return result

    # ------------------------------------------------------------------
    # Private: model loading
    # ------------------------------------------------------------------

    def _load_regime_stack(self, out_dir: str) -> List[dict]:
        """Load regime stack CSV from REGIME_STACK_PATH in config.

        Raises KeyError immediately if REGIME_STACK_PATH is not configured.
        Raises FileNotFoundError if the configured path does not exist.

        Maps optimal_threshold → threshold so that filtered_optimal_regime_stack.csv
        (output of filter_regime_stacks.py) can be used directly for live inference.

        Resolution order for relative REGIME_STACK_PATH:
          1. ``home_root / REGIME_STACK_PATH`` (legacy local convention)
          2. ``OUTPUT_DIR / filtered_optimal_regime_stack.csv`` (NAS-routed `_1`
             experiments — setting.json holds a relative pointer but the file
             lives next to the rest of the OUTPUT_DIR artifacts on NAS)
        Absolute paths are tried as-is. If no candidate exists, raise
        FileNotFoundError with EVERY tried path so the operator can debug.
        """
        if "REGIME_STACK_PATH" not in self.config:
            raise KeyError("REGIME_STACK_PATH is required in config but not set")
        cfg_path = Path(self.config["REGIME_STACK_PATH"])
        candidates: List[Path] = []
        if cfg_path.is_absolute():
            candidates.append(cfg_path)
        else:
            candidates.append(Path(self.home_root) / cfg_path)
            # WHY: `_1` experiments keep the canonical regime stack in OUTPUT_DIR
            # on NAS even though setting.json carries a relative pointer.
            candidates.append(Path(out_dir) / "filtered_optimal_regime_stack.csv")
        for cand in candidates:
            if cand.exists():
                path = cand
                break
        else:
            raise FileNotFoundError(
                "REGIME_STACK_PATH not found. Tried:\n  - "
                + "\n  - ".join(str(c) for c in candidates)
            )
        df = pd.read_csv(path)
        # filtered_optimal_regime_stack.csv uses 'optimal_threshold'; map to 'threshold'
        if "optimal_threshold" in df.columns and "threshold" not in df.columns:
            df = df.rename(columns={"optimal_threshold": "threshold"})
        # Skip summary rows (produced by filter_regime_stacks.py)
        if "regime" in df.columns:
            df = df[df["regime"] != "__summary__"]
        stack = df.to_dict(orient="records")
        logger.info("Loaded %d regime entries from %s", len(stack), path)
        return stack

    def _load_models(self, out_dir: str) -> None:
        """Load model/scaler/meta pkl files from the latest weights window."""
        weights_root = Path(out_dir) / "weights"
        if not weights_root.exists():
            logger.warning("No weights/ directory found at %s", weights_root)
            return

        # Find latest window directory
        window_dirs = sorted(
            d for d in weights_root.iterdir()
            if d.is_dir() and d.name.startswith("window_")
        )
        if not window_dirs:
            logger.warning("No window directories in %s", weights_root)
            return

        latest = window_dirs[-1]
        logger.info("Loading models from %s", latest)

        loaded = 0
        for entry in self._regime_stack:
            regime_name = entry.get("regime", "")
            position = entry.get("position", "long")
            model_name = entry.get("model", "LightGBM").lower()

            dir_key = f"{regime_name}_{position}"
            regime_dir = latest / dir_key
            if not regime_dir.exists():
                # Some pipelines name it just {regime_name}
                regime_dir = latest / regime_name
                if not regime_dir.exists():
                    logger.debug("No weights dir for %s", dir_key)
                    continue

            model_path = regime_dir / f"{model_name}_model.pkl"
            scaler_path = regime_dir / f"{model_name}_scaler.pkl"
            meta_path = regime_dir / f"{model_name}_meta.pkl"

            if not model_path.exists():
                logger.debug("Model file missing: %s", model_path)
                continue

            try:
                model_key = f"{dir_key}_{model_name}"
                self._models[model_key] = joblib.load(model_path)
                self._scalers[model_key] = (joblib.load(scaler_path)
                                            if scaler_path.exists()
                                            else _IdentityScaler())
                self._metas[model_key] = (joblib.load(meta_path)
                                          if meta_path.exists()
                                          else {})
                loaded += 1
            except Exception as exc:
                logger.error("Failed to load model %s: %s", model_path, exc)

        logger.info("Loaded %d/%d models", loaded, len(self._regime_stack))


class _IdentityScaler:
    """No-op scaler for missing scaler files."""
    def transform(self, X):
        return X
