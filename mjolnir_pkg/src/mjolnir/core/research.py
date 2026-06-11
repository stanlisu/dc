"""MjolnirResearch: top-level pipeline class mirroring AgamottoResearch."""

from __future__ import annotations

import gc
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# pyarrow is REQUIRED for the streaming filter writer. Per CLAUDE.md no-silent-
# fallbacks: if pyarrow is missing we want a hard ImportError, not a silent
# fall-back to the pandas-concat path (which is precisely what the streaming
# refactor exists to avoid).
import pyarrow as pa  # noqa: F401  -- imported here so missing pyarrow fails at import time
import pyarrow.parquet as pq

from .features import MjolnirFeatures, _TF_SECONDS
from .loader import MjolnirLoader
from .aligner import StreamAligner
from .multi_tf_merge import merge_cross_tf_features
from .utils import normalize_symbol, get_dates_in_range

logger = logging.getLogger(__name__)


def _filter_parquet_filenames(
    files: List[str],
    start: Optional[str],
    end: Optional[str],
) -> List[str]:
    """Filter `YYYYMMDD.parquet` filenames by inclusive date range.

    `start` / `end` are `YYYYMMDD` strings (or None). Lex-compare works because
    stem length is fixed. Non-`.parquet` filenames are passed through unchanged.
    """
    if start and end and start > end:
        raise RuntimeError(f"LOAD_START {start} > LOAD_END {end}")
    if not start and not end:
        return list(files)
    out = []
    for f in files:
        stem = f[:-len(".parquet")] if f.endswith(".parquet") else f
        if start and stem < start:
            continue
        if end and stem > end:
            continue
        out.append(f)
    return out


# Columns excluded from ML feature selection (mirrors rolling_predict_returns.py)
_META_COLS = {
    "ret", "ret_raw", "return", "return_long", "return_short",
    "return_long_raw", "return_short_raw",
    "position", "regime", "timestamp", "symbol", "year", "month",
}

# Base-TF rolling-stat windows (in bars). Constant — every mjolnir setting.json
# previously hardcoded the same list, so the FEATURE_WINDOWS knob was dead
# weight. See mjolnir/trading.py for the live-inference twin.
_DEFAULT_FEATURE_WINDOWS = (30, 60, 300, 900)

# Row-group size for streaming filter parquets. Matches
# gauntlet/rolling_predict_returns.py::_rewrite_parquet_fine_rowgroups
# default batch_size (100_000). The rolling step's needs_rewrite check
# triggers when num_rows / num_row_groups > 300_000, so writing at
# 100_000 here means newly-produced filter parquets skip the rewrite
# step entirely. Keep these two constants aligned.
_FILTER_ROW_GROUP_SIZE = 100_000


def _widen_ints_to_float(table: "pa.Table") -> "pa.Table":
    """Promote every integer column in `table` to float64.

    Streaming writes lock the writer's schema on the first row group. If sym A
    has no NaNs in column X (pandas → int64) but sym B has NaNs (pandas →
    float64), the writer would reject sym B. Promoting all int columns to
    float64 up-front matches the offline merge_filter_batches.py widening
    rule and costs ~2x storage on those columns (snappy-compressed, modest).
    """
    int_types = {pa.int8(), pa.int16(), pa.int32(), pa.int64(),
                 pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64()}
    new_fields = []
    needs_cast = False
    for f in table.schema:
        if f.type in int_types:
            new_fields.append(pa.field(f.name, pa.float64()))
            needs_cast = True
        else:
            new_fields.append(f)
    if not needs_cast:
        return table
    target = pa.schema(new_fields)
    return table.cast(target, safe=False)


def _narrow_floats_to_float32(table: "pa.Table") -> "pa.Table":
    """Narrow every float64 column to float32 before writing filter parquets.

    The rolling trainer (`_read_pq` in rolling_predict_returns.py) already casts
    every float64 feature column to float32 on load, so float64 on disk is pure
    wasted storage (~2x on the feature matrix, which is ~90% of the columns).
    Narrowing at write halves the filter parquet with ZERO effect on training —
    the trainer sees float32 either way.

    Applied BEFORE `_widen_ints_to_float`, so genuine integer columns (later
    promoted to float64 only for streaming-schema consistency) keep full
    precision and are never narrowed — only true float64 features are cast.
    Cross-symbol dtype drift is reconciled downstream by the writer's
    `table.cast(state["schema"])` step, same as for int widening.
    """
    new_fields = []
    needs_cast = False
    for f in table.schema:
        if f.type == pa.float64():
            new_fields.append(pa.field(f.name, pa.float32()))
            needs_cast = True
        else:
            new_fields.append(f)
    if not needs_cast:
        return table
    target = pa.schema(new_fields)
    return table.cast(target, safe=False)


class MjolnirResearch:
    """Research pipeline for Mjolnir tick-data ML.

    Mirrors AgamottoResearch in API so that all downstream Gauntlet scripts
    (rolling_predict_returns, generate_daily_pnl, optimize_thresholds,
    filter_regime_stacks) can be reused without modification.

    Args:
        config:    Dictionary loaded from setting.json.
        home_root: Repository root directory.
    """

    def __init__(self, config: Dict, home_root: str) -> None:
        self.config = config
        self.home_root = home_root.rstrip("/")
        self.vertical_features: Optional[pd.DataFrame] = None
        # per-symbol bar DataFrames, keyed by native ticker (base TF)
        self._symbol_bars: Dict[str, pd.DataFrame] = {}
        # per-TF per-symbol bar DataFrames for MULTI_TF_BARS
        # {tf: {native_symbol: bars_df}}
        self._multi_tf_bars: Dict[str, Dict[str, pd.DataFrame]] = {}

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load pre-built bar parquet files from bars/ directory.

        Expects files at:
            {out_dir}/bars/{symbol}/{YYYYMMDD}.parquet

        If ``BARS_DIR`` is set in config, that directory is used instead,
        allowing multiple horizon experiments to share one set of bar files.
        """
        out_dir = self._resolve_out_dir()
        # TRAIN_BARS_DIR: optional override for the training base bars directory.
        # Used when the experiment trains at a finer resolution than TIME_UNIT
        # (e.g., 1m_1 trains on 5s bars to produce boundary-aligned 1m targets).
        # Falls back to BARS_DIR, then to {out_dir}/bars.
        train_bars_dir_cfg = self.config.get("TRAIN_BARS_DIR") or self.config.get("BARS_DIR")
        if train_bars_dir_cfg:
            bars_root = train_bars_dir_cfg if os.path.isabs(train_bars_dir_cfg) else os.path.join(self.home_root, train_bars_dir_cfg)
        else:
            bars_root = os.path.join(out_dir, "bars")
        if not os.path.isdir(bars_root):
            raise RuntimeError(
                f"bars/ directory not found at {bars_root}. "
                "Run build_bars.py first."
            )

        load_start = self.config.get("LOAD_START")
        load_end = self.config.get("LOAD_END")
        symbols = self.config.get("SYMBOLS", [])
        for sym in symbols:
            native = normalize_symbol(sym)
            sym_dir = os.path.join(bars_root, native)
            if not os.path.isdir(sym_dir):
                logger.warning("No bar directory for %s at %s", native, sym_dir)
                continue

            files = _filter_parquet_filenames(
                sorted(f for f in os.listdir(sym_dir) if f.endswith(".parquet")),
                load_start, load_end,
            )
            if not files:
                logger.warning("No parquet files for %s", native)
                continue

            frames = []
            for fname in files:
                try:
                    frames.append(pd.read_parquet(os.path.join(sym_dir, fname)))
                except Exception as exc:
                    logger.warning("Failed to load %s/%s: %s", sym_dir, fname, exc)

            if frames:
                combined = pd.concat(frames).sort_index()
                combined = combined[~combined.index.duplicated(keep="last")]
                # Forward-fill OHLCV for zero-trade bars (no trades in a 5s window
                # → NaN OHLCV, which cascades into all TA features and breaks
                # Ridge/ElasticNet). Standard convention: carry last traded price.
                if "close" in combined.columns:
                    combined["close"] = combined["close"].ffill()
                    no_trade = combined["n_trades"].isna() if "n_trades" in combined.columns \
                        else pd.Series(False, index=combined.index)
                    for col in ["open", "high", "low", "vwap"]:
                        if col in combined.columns:
                            combined.loc[no_trade, col] = combined.loc[no_trade, "close"]
                    for col in ["volume", "n_trades", "buy_vol", "sell_vol", "trade_imbalance"]:
                        if col in combined.columns:
                            combined[col] = combined[col].fillna(0.0)
                self._symbol_bars[native] = combined
                logger.info("Loaded %d bars for %s", len(combined), native)

        if not self._symbol_bars:
            raise RuntimeError("No bar files found. Run build_bars.py first.")

        # Load additional TF bars for multi-resolution features.
        # Each TF's bars live at the BARS_DIR of the corresponding _1 experiment:
        #   /mnt/tardis-data-archive/mjolnir/pred_mjolnir.base.{tf}_1/bars/
        # The MULTI_TF_BARS_DIRS config key (optional) allows explicit overrides
        # as a dict {tf: bars_path}.  If absent, we auto-derive from the archive.
        multi_tfs = self.config.get("MULTI_TF_BARS", [])
        if multi_tfs:
            tf_dirs_override = self.config.get("MULTI_TF_BARS_DIRS", {})
            for tf in multi_tfs:
                if tf in tf_dirs_override:
                    tf_bars_root = tf_dirs_override[tf]
                    if not os.path.isabs(tf_bars_root):
                        tf_bars_root = os.path.join(self.home_root, tf_bars_root)
                else:
                    # Auto-derive: same archive root, pred_mjolnir.base.{tf}_1/bars/
                    # bars_root is e.g. /mnt/.../pred_mjolnir.base.5s_1/bars
                    # Need two dirname() calls: strip /bars then strip /pred_mjolnir.base.5s_1
                    archive_base = os.path.dirname(os.path.dirname(bars_root.rstrip("/")))
                    tf_norm = tf.replace("s", "s").replace("m", "m")
                    tf_bars_root = os.path.join(
                        archive_base, f"pred_mjolnir.base.{tf_norm}_1", "bars"
                    )
                if not os.path.isdir(tf_bars_root):
                    # No silent skip — a missing cross-TF bars dir would produce
                    # a filter parquet with only base-TF cols and the model would
                    # train on a much narrower feature matrix than the spec
                    # prescribes. Either build the missing bars or remove the
                    # TF from MULTI_TF_BARS in setting.json.
                    # See researcher_data_integrity_report.md item 5 and
                    # researcher_label_quality_report.md §3.3.
                    raise RuntimeError(
                        f"MULTI_TF_BARS: bars directory for TF {tf!r} not found "
                        f"at {tf_bars_root!r}. Either build bars for that TF or "
                        f"remove {tf!r} from MULTI_TF_BARS in setting.json. "
                        "Refusing to silently produce an incomplete filter parquet."
                    )

                tf_sym_bars: Dict[str, pd.DataFrame] = {}
                for sym in symbols:
                    native = normalize_symbol(sym)
                    sym_dir = os.path.join(tf_bars_root, native)
                    if not os.path.isdir(sym_dir):
                        logger.warning(
                            "MULTI_TF_BARS: no bar dir for %s at %s", native, sym_dir
                        )
                        continue
                    files = _filter_parquet_filenames(
                        sorted(f for f in os.listdir(sym_dir) if f.endswith(".parquet")),
                        load_start, load_end,
                    )
                    if not files:
                        logger.warning("MULTI_TF_BARS: no parquet files for %s TF=%s", native, tf)
                        continue
                    frames = []
                    for fname in files:
                        try:
                            frames.append(pd.read_parquet(os.path.join(sym_dir, fname)))
                        except Exception as exc:
                            logger.warning(
                                "MULTI_TF_BARS: failed to load %s/%s: %s", sym_dir, fname, exc
                            )
                    if frames:
                        combined_tf = pd.concat(frames).sort_index()
                        combined_tf = combined_tf[~combined_tf.index.duplicated(keep="last")]
                        if "close" in combined_tf.columns:
                            combined_tf["close"] = combined_tf["close"].ffill()
                            no_trade_tf = combined_tf["n_trades"].isna() \
                                if "n_trades" in combined_tf.columns \
                                else pd.Series(False, index=combined_tf.index)
                            for col in ["open", "high", "low", "vwap"]:
                                if col in combined_tf.columns:
                                    combined_tf.loc[no_trade_tf, col] = \
                                        combined_tf.loc[no_trade_tf, "close"]
                            for col in ["volume", "n_trades", "buy_vol",
                                        "sell_vol", "trade_imbalance"]:
                                if col in combined_tf.columns:
                                    combined_tf[col] = combined_tf[col].fillna(0.0)
                        tf_sym_bars[native] = combined_tf
                        logger.info(
                            "Loaded %d bars for %s TF=%s", len(combined_tf), native, tf
                        )
                self._multi_tf_bars[tf] = tf_sym_bars
                logger.info("Loaded multi-TF bars for TF=%s (%d symbols)", tf, len(tf_sym_bars))

    def engineer_features(self) -> None:
        """Apply MjolnirFeatures to loaded bars; populates per-symbol feature DFs.

        Processes BTC first so that BTC cross-asset features can be injected into
        ETH/SOL/AVAX via add_btc_cross_features().
        """
        if not self._symbol_bars:
            raise RuntimeError("Call load() before engineer_features().")

        cfg = self.config
        # FEE is required (no fallback): a missing key should fail loudly so
        # misconfiguration cannot silently inject a phantom fee into the target.
        # The historical `or 4.5` collapsed `FEE: 0.0` (falsy) to 4.5 bps —
        # poisoning every filter parquet with a -4.5e-4 offset on `ret`.
        # Per CLAUDE.md "no fallback defaults": read directly and KeyError on
        # missing. See researcher_label_quality_report.md §3.1 and
        # researcher_pipeline_audit_report.md "Bug B".
        if "FEE" not in cfg:
            raise KeyError(
                "FEE missing from setting.json — required (in bps, e.g. 0.0 or 4.5). "
                "No default permitted; please set explicitly."
            )
        fee_rate = float(cfg["FEE"]) / 10000.0
        target_horizon = int(cfg.get("TARGET_HORIZON_BARS", 60))
        if "FEATURE_WINDOWS" in cfg:
            raise ValueError(
                "FEATURE_WINDOWS is deprecated; remove it from setting.json — "
                "windows are constants now (mjolnir/core/research.py)"
            )
        feature_windows = list(_DEFAULT_FEATURE_WINDOWS)
        time_unit = cfg.get("TIME_UNIT", "5s")
        # bar_tf: the actual resolution of the base bars being loaded.
        # When TRAIN_BARS_DIR is set, we are loading 5s bars as the training
        # base regardless of TIME_UNIT (which controls the target boundary).
        bar_tf = "5s" if cfg.get("TRAIN_BARS_DIR") else time_unit

        feat_engine = MjolnirFeatures(
            feature_windows=feature_windows,
            target_horizon=target_horizon,
            fee_rate=fee_rate,
            bar_tf=bar_tf,
            target_tf=time_unit,
        )

        # BTC must be processed first so cross-features are available for other symbols
        _BTC = "BINANCE_PERP_BTC_USDT"
        ordered = sorted(self._symbol_bars.keys(), key=lambda s: (s != _BTC, s))

        # Build per-TF feature engines for multi-resolution merging.
        # Multi-TF (higher-TF) engines always use native bar resolution as bar_tf.
        # They produce features for context only — their targets are dropped.
        # Cross-TF engines use a single trivial window to avoid feature redundancy
        # with base-TF FEATURE_WINDOWS rolling stats. Without this, e.g. 30s base
        # w60 (30 min) and 1m cross-TF w30 (30 min) encode the same horizon via
        # different paths, which inflates feature collinearity and hurts Ridge OOS.
        multi_tfs = list(self._multi_tf_bars.keys())
        tf_engines = {
            tf: MjolnirFeatures(
                feature_windows=[1],
                target_horizon=target_horizon,
                fee_rate=fee_rate,
                prefix=tf,
                bar_tf=tf,
                target_tf=tf,
            )
            for tf in multi_tfs
        }

        self._symbol_features: Dict[str, pd.DataFrame] = {}
        btc_feats: Optional[pd.DataFrame] = None
        for native in ordered:
            bars = self._symbol_bars[native]
            logger.info("Engineering features for %s (%d bars)...", native, len(bars))
            tf_feats_map: Dict[str, pd.DataFrame] = {}
            try:
                feats = feat_engine.compute(bars)
                if native == _BTC:
                    btc_feats = feats
                elif btc_feats is not None:
                    feats = feat_engine.add_btc_cross_features(feats, btc_feats)

                # Compute higher-TF feature frames for this symbol. The merge
                # itself runs OUTSIDE this try so that a helper-raised
                # RuntimeError (e.g. unknown TF in MULTI_TF_BARS) propagates
                # cleanly instead of being caught and silently dropped.
                for tf in multi_tfs:
                    tf_bars_for_sym = self._multi_tf_bars[tf].get(native)
                    if tf_bars_for_sym is None or tf_bars_for_sym.empty:
                        logger.warning(
                            "No %s bars for %s — skipping TF merge", tf, native
                        )
                        continue
                    try:
                        tf_feats_map[tf] = tf_engines[tf].compute(tf_bars_for_sym)
                        logger.info(
                            "Computed %d %s-TF feature cols for %s",
                            len(tf_feats_map[tf].columns), tf, native,
                        )
                    except Exception as exc:
                        logger.error(
                            "Multi-TF feature engineering failed for %s TF=%s: %s",
                            native, tf, exc,
                        )
            except Exception as exc:
                logger.error("Feature engineering failed for %s: %s", native, exc)
                continue

            # Bug 2 fix: shared helper used by research AND live. The helper
            # raises on unknown TF / config mismatch — let it propagate.
            feats = merge_cross_tf_features(feats, tf_feats_map)
            self._symbol_features[native] = feats

        # Sanity guard: if MULTI_TF_BARS was configured, every per-symbol feature
        # frame must contain at least one column prefixed by each requested TF.
        # Without this guard, the merge above can silently no-op (e.g. an inner
        # exception swallowed by the per-TF try/except, or a missing per-symbol
        # bars df) and downstream filter parquets carry only base-TF columns —
        # the exact failure that produced 5s_1 / 15s_1's stale-feature corruption.
        # See researcher_label_quality_report.md §3.3 and
        # researcher_feature_quality_report.md §4.
        if multi_tfs and self._symbol_features:
            tf_prefixes = [f"{tf}_" for tf in multi_tfs]
            missing_per_tf = {tf: 0 for tf in multi_tfs}
            for feats in self._symbol_features.values():
                cols = list(feats.columns)
                for tf, pref in zip(multi_tfs, tf_prefixes):
                    if not any(c.startswith(pref) for c in cols):
                        missing_per_tf[tf] += 1
            n_syms = len(self._symbol_features)
            zero_tfs = [tf for tf, n in missing_per_tf.items() if n == n_syms]
            if zero_tfs:
                raise RuntimeError(
                    "Cross-TF feature merge produced zero "
                    f"columns for TF(s) {zero_tfs!r} across all "
                    f"{n_syms} symbols, despite MULTI_TF_BARS={multi_tfs!r} "
                    "in setting.json. The merge silently failed — likely an "
                    "inner exception or a missing per-symbol bars frame. "
                    "Refusing to produce a filter parquet without the cross-TF "
                    "context features the spec prescribes."
                )

    def verticalize(self) -> None:
        """Stack per-symbol feature DataFrames into a single vertical DataFrame.

        Adds 'symbol' and 'timestamp' columns (timestamp comes from index).
        Drops rows with NaN targets (last target_horizon bars per symbol).
        """
        if not hasattr(self, "_symbol_features") or not self._symbol_features:
            raise RuntimeError("Call engineer_features() before verticalize().")

        frames = []
        symbols = self.config.get("SYMBOLS", [])
        # Map native -> canonical symbol name
        native_to_sym = {normalize_symbol(s): s for s in symbols}

        # Pop from _symbol_features to free originals as we go (saves ~50% memory)
        import gc
        natives = list(self._symbol_features.keys())
        for native in natives:
            feats = self._symbol_features.pop(native)
            sym = native_to_sym.get(native, native)
            feats["symbol"] = sym
            feats["timestamp"] = feats.index
            # Apply ladder-adjusted return columns. The horizon (in bars)
            # matches the prediction window: native-mode (bar_tf ==
            # TIME_UNIT) uses 1 bar; boundary-aligned mode (e.g. 5s bars
            # predicting 30s closes) uses TIME_UNIT_seconds /
            # bar_tf_seconds so the low/high lookahead spans the full
            # prediction horizon. Without this the boundary-mode target
            # collapses to a frictionless close-to-close return, which
            # the maker-first executor cannot trade — every signal where
            # price gaps the predicted direction immediately gets credit
            # in research but never fills live.
            time_unit = self.config.get("TIME_UNIT", "5s")
            bar_tf = "5s" if self.config.get("TRAIN_BARS_DIR") else time_unit
            horizon_bars = max(
                1, _TF_SECONDS[time_unit] // _TF_SECONDS[bar_tf])
            if all(c in feats.columns for c in ("close", "low", "high")):
                ladder_cols = self._compute_ladder_returns(
                    feats, "close", "low", "high", horizon_bars=horizon_bars)
                for col in ladder_cols.columns:
                    feats[col] = ladder_cols[col].values
            # Drop rows with NaN in return columns (last target_horizon bars)
            if "return" in feats.columns:
                feats = feats[feats["return"].notna()]
            frames.append(feats.reset_index(drop=True))
        gc.collect()

        if frames:
            self.vertical_features = pd.concat(frames, axis=0, ignore_index=True)
            logger.info(
                "Vertical features: %d rows × %d cols",
                len(self.vertical_features),
                len(self.vertical_features.columns),
            )
        else:
            self.vertical_features = pd.DataFrame()

        # _symbol_features already emptied during verticalize (pop above)

    def filter_signals(
        self,
        regime: Dict,
        limit_timestamp=None,
        save: bool = False,
        out_dir: Optional[str] = None,
    ) -> pd.DataFrame:
        """Filter vertical features for a single regime.

        Args:
            regime:          Dict with 'regime' and 'position' keys.
            limit_timestamp: If set, only rows from this timestamp onwards.
            save:            Save result to out_dir/filter/.
            out_dir:         Output directory.

        Returns:
            Filtered DataFrame with 'ret', 'ret_raw', 'position', 'regime' added.
        """
        if self.vertical_features is None:
            raise RuntimeError("Call verticalize() before filter_signals().")

        features_df = self.vertical_features
        if limit_timestamp is not None:
            features_df = features_df[features_df["timestamp"] >= limit_timestamp]

        regime_name = regime.get("regime", "baseline")
        position = regime.get("position", "long")

        regime_name_str = regime_name
        if isinstance(regime_name, list):
            regime_name_str = "_and_".join(
                p for p in regime_name if p not in ("|", "&")
            )

        mask = self._apply_filter_mask(features_df, regime_name, position)
        n_selected = mask.sum()

        if n_selected == 0:
            return pd.DataFrame()

        # REVERSE=-1 trades the opposite direction: swap return columns
        reverse = int(self.config.get("REVERSE", 1))
        effective_position = position
        if reverse == -1:
            effective_position = "short" if position == "long" else "long"

        # Determine ret/ret_raw column names — no copy needed
        if effective_position == "long":
            ret_col, ret_raw_col = "return_long", "return_long_raw"
        else:
            ret_col, ret_raw_col = "return_short", "return_short_raw"

        if save and out_dir:
            clean = regime_name_str.replace("_long", "").replace("_short", "")
            safe_name = "".join(
                c if c.isalnum() or c == "_" else "_"
                for c in f"{clean}_{position}"
            )
            save_dir = os.path.join(out_dir, "filter")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"filter_{safe_name}.parquet")
            try:
                # Write in chunks to avoid allocating the full filtered copy at once.
                # For large regimes (baseline = all rows), a single copy would OOM.
                import pyarrow as pa
                import pyarrow.parquet as pq
                indices = features_df.index[mask]
                chunk_size = 500_000
                writer = None
                for start in range(0, len(indices), chunk_size):
                    chunk_idx = indices[start:start + chunk_size]
                    chunk = features_df.loc[chunk_idx].copy()
                    chunk["position"] = position
                    chunk["regime"] = regime_name_str
                    chunk["ret"] = chunk[ret_col]
                    if ret_raw_col in chunk.columns:
                        chunk["ret_raw"] = chunk[ret_raw_col]
                    table = pa.Table.from_pandas(chunk, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(save_path, table.schema)
                    # row_group_size subdivides the input Table into ~100K-row
                    # row groups inside this single write_table call (pyarrow
                    # auto-slices). Mirrors the streaming path at line 920 so
                    # filter parquets produced via filter_signals(save=True)
                    # — the path used by gauntlet/regenerate_filter.py — also
                    # skip the rolling step's _rewrite_parquet_fine_rowgroups
                    # pass (triggered when num_rows / num_row_groups > 300_000).
                    writer.write_table(table, row_group_size=_FILTER_ROW_GROUP_SIZE)
                    del chunk, table
                if writer is not None:
                    writer.close()
                logger.info("Saved filter %s (%d rows)", safe_name, n_selected)
            except Exception as exc:
                logger.error("Failed to save %s: %s", safe_name, exc)

        # Return empty DataFrame for the save-only path (create() doesn't use the return)
        return pd.DataFrame()

    def create(self) -> str:
        """Orchestrate the full research pipeline.

        Memory-bounded streaming variant: per-symbol feature engineering and
        per-(regime, position) filter writes happen inside one symbol-loop,
        so peak RAM is bounded by ONE symbol's engineered features rather than
        the whole 29-symbol × multi-TF accumulation that previously OOM'd at
        ~150 GB.

        Returns:
            Output directory path.
        """
        if not self._symbol_bars:
            self.load()

        out_dir = self._resolve_out_dir()
        os.makedirs(out_dir, exist_ok=True)

        # Load regime stack BEFORE engineering — fail fast on misconfig.
        regime_stack = self._load_regime_stack(out_dir)
        logger.info("Processing %d regimes via per-symbol streaming...", len(regime_stack))

        self._stream_filter_parquets(out_dir, regime_stack)

        # Drop residual bar dicts — engineer/stream consumed them already but
        # be defensive in case an early-return left state behind.
        self._symbol_bars.clear()
        self._multi_tf_bars.clear()
        gc.collect()
        return out_dir

    def _stream_filter_parquets(
        self,
        out_dir: str,
        regime_stack: List[Dict],
    ) -> None:
        """Per-symbol streaming write of (regime, position) filter parquets.

        Iterates SYMBOLS one at a time. For each symbol:
          1. Engineer base-TF + cross-TF features (one symbol's worth in RAM).
          2. Apply ladder returns and drop NaN-target rows.
          3. For every (regime, position) in the regime stack, compute the
             filter mask, write the slice as a row group to that regime's
             ``ParquetWriter``.
          4. Drop the symbol's feature frame, gc.

        Writers are opened lazily on first non-empty slice, written to a
        ``*.tmp`` path, fsync'd best-effort, and renamed to the canonical
        ``filter_*.parquet`` on success.

        BTC features are kept resident across the loop so cross-asset
        features can be injected into ETH/SOL/AVAX (mirrors the historical
        engineer_features ordering — BTC first, others second).
        """
        cfg = self.config
        # FEE is required (no fallback). Mirrors engineer_features.
        if "FEE" not in cfg:
            raise KeyError(
                "FEE missing from setting.json — required (in bps, e.g. 0.0 or 4.5). "
                "No default permitted; please set explicitly."
            )
        fee_rate = float(cfg["FEE"]) / 10000.0
        target_horizon = int(cfg.get("TARGET_HORIZON_BARS", 60))
        if "FEATURE_WINDOWS" in cfg:
            raise ValueError(
                "FEATURE_WINDOWS is deprecated; remove it from setting.json — "
                "windows are constants now (mjolnir/core/research.py)"
            )
        feature_windows = list(_DEFAULT_FEATURE_WINDOWS)
        time_unit = cfg.get("TIME_UNIT", "5s")
        bar_tf = "5s" if cfg.get("TRAIN_BARS_DIR") else time_unit

        feat_engine = MjolnirFeatures(
            feature_windows=feature_windows,
            target_horizon=target_horizon,
            fee_rate=fee_rate,
            bar_tf=bar_tf,
            target_tf=time_unit,
        )

        multi_tfs = list(self._multi_tf_bars.keys())
        tf_engines = {
            tf: MjolnirFeatures(
                feature_windows=[1],
                target_horizon=target_horizon,
                fee_rate=fee_rate,
                prefix=tf,
                bar_tf=tf,
                target_tf=tf,
            )
            for tf in multi_tfs
        }

        # BTC must be processed first so cross-features are available for other symbols.
        _BTC = "BINANCE_PERP_BTC_USDT"
        ordered = sorted(self._symbol_bars.keys(), key=lambda s: (s != _BTC, s))

        # Map native -> canonical symbol name (preserves SYMBOLS spelling for the
        # `symbol` column in output parquets).
        symbols_cfg = self.config.get("SYMBOLS", [])
        native_to_sym = {normalize_symbol(s): s for s in symbols_cfg}

        reverse = int(self.config.get("REVERSE", 1))

        save_dir = os.path.join(out_dir, "filter")
        os.makedirs(save_dir, exist_ok=True)

        # Per-(regime, position) writer state. Keys are (regime_name_str, position).
        # Value is a dict: {writer, schema, tmp_path, final_path, n_rows}.
        writers: Dict[Tuple[str, str], Dict] = {}

        # Cross-TF guard counters — populated as we go; checked at the end.
        tf_cols_seen = {tf: 0 for tf in multi_tfs}

        btc_feats: Optional[pd.DataFrame] = None
        n_symbols_processed = 0
        success = False
        try:
            for native in ordered:
                bars = self._symbol_bars.pop(native, None)
                if bars is None:
                    continue
                logger.info(
                    "Streaming features for %s (%d bars)...", native, len(bars),
                )
                feats = None
                try:
                    feats = feat_engine.compute(bars)
                except Exception as exc:
                    logger.error("Feature engineering failed for %s: %s", native, exc)
                # Free this symbol's base bars now that features are materialised
                # (or were attempted — on failure we still want bars off the heap).
                bars = None
                if feats is None:
                    gc.collect()
                    continue

                if native == _BTC:
                    # Stash BTC features for cross-asset enrichment of other symbols.
                    # We deep-copy because feats will be mutated in place below.
                    btc_feats = feats.copy()
                elif btc_feats is not None:
                    feats = feat_engine.add_btc_cross_features(feats, btc_feats)

                # Cross-TF merge per symbol.
                tf_feats_map: Dict[str, pd.DataFrame] = {}
                for tf in multi_tfs:
                    tf_bars_for_sym = self._multi_tf_bars.get(tf, {}).pop(native, None)
                    if tf_bars_for_sym is None or tf_bars_for_sym.empty:
                        logger.warning(
                            "No %s bars for %s — skipping TF merge", tf, native,
                        )
                        continue
                    try:
                        tf_feats_map[tf] = tf_engines[tf].compute(tf_bars_for_sym)
                        logger.info(
                            "Computed %d %s-TF feature cols for %s",
                            len(tf_feats_map[tf].columns), tf, native,
                        )
                    except Exception as exc:
                        logger.error(
                            "Multi-TF feature engineering failed for %s TF=%s: %s",
                            native, tf, exc,
                        )
                    del tf_bars_for_sym
                # merge_cross_tf_features raises on unknown TF / config mismatch.
                feats = merge_cross_tf_features(feats, tf_feats_map)
                del tf_feats_map

                # Track which cross-TFs contributed columns so we can run the
                # post-loop "every TF carried at least one column somewhere" guard.
                for tf in multi_tfs:
                    pref = f"{tf}_"
                    if any(c.startswith(pref) for c in feats.columns):
                        tf_cols_seen[tf] += 1

                # Stamp symbol + timestamp columns (matches verticalize() output).
                sym_canonical = native_to_sym.get(native, native)
                feats["symbol"] = sym_canonical
                feats["timestamp"] = feats.index

                # Apply ladder returns. horizon_bars matches the
                # prediction window so the low/high lookahead spans the
                # same span as the price-return horizon. In native mode
                # (bar_tf == TIME_UNIT) this is 1; in boundary-aligned
                # mode (e.g. mjolnir.base.30s_1 = 5s bars predicting 30s
                # boundary closes) it is TIME_UNIT_seconds /
                # bar_tf_seconds, so a 5s/30s experiment uses horizon=6.
                horizon_bars = max(
                    1, _TF_SECONDS[time_unit] // _TF_SECONDS[bar_tf])
                if all(
                    c in feats.columns for c in ("close", "low", "high")
                ):
                    ladder_cols = self._compute_ladder_returns(
                        feats, "close", "low", "high",
                        horizon_bars=horizon_bars,
                    )
                    for col in ladder_cols.columns:
                        feats[col] = ladder_cols[col].values
                    del ladder_cols

                # Drop rows with NaN target — matches verticalize() semantics.
                if "return" in feats.columns:
                    feats = feats[feats["return"].notna()]
                if feats.empty:
                    logger.warning("Feature frame for %s empty after NaN-drop; skipping", native)
                    del feats
                    gc.collect()
                    continue

                # reset_index so the row-group has a clean integer index.
                feats = feats.reset_index(drop=True)

                # Write one slice per (regime, position) into the lazy writers.
                self._write_symbol_to_filters(
                    feats=feats,
                    regime_stack=regime_stack,
                    writers=writers,
                    save_dir=save_dir,
                    reverse=reverse,
                )

                del feats
                n_symbols_processed += 1
                # gc every symbol — these are 1+ GB frames; we want the heap
                # to hand pages back to the OS before the next symbol starts.
                gc.collect()

            # Cross-TF column-presence guard (mirrors engineer_features post-loop guard).
            if multi_tfs and n_symbols_processed > 0:
                zero_tfs = [tf for tf, n in tf_cols_seen.items() if n == 0]
                if zero_tfs:
                    raise RuntimeError(
                        "Cross-TF feature merge produced zero "
                        f"columns for TF(s) {zero_tfs!r} across all "
                        f"{n_symbols_processed} symbols, despite MULTI_TF_BARS={multi_tfs!r} "
                        "in setting.json. The merge silently failed — likely an "
                        "inner exception or a missing per-symbol bars frame. "
                        "Refusing to produce a filter parquet without the cross-TF "
                        "context features the spec prescribes."
                    )
            success = True
        finally:
            # Always finalise writers — close + rename .tmp -> final on success,
            # or close + delete .tmp on exception so a partial mid-write parquet
            # does not get picked up by downstream globs as real data. Per
            # CLAUDE.md "no silent fallbacks": a torn write must fail loudly
            # (delete the tmp) rather than masquerade as a complete artifact.
            self._finalise_writers(writers, success=success)

    def _write_symbol_to_filters(
        self,
        feats: pd.DataFrame,
        regime_stack: List[Dict],
        writers: Dict[Tuple[str, str], Dict],
        save_dir: str,
        reverse: int,
    ) -> None:
        """For one symbol's vertical features, append a row group per regime.

        Lazily opens a ParquetWriter per (regime_name_str, position) on first
        non-empty slice. Schema is captured from that first table; subsequent
        symbols MUST have the same schema or we raise (no silent column drift).
        """
        for regime in regime_stack:
            regime_name = regime.get("regime", "baseline")
            position = regime.get("position", "long")

            regime_name_str = regime_name
            if isinstance(regime_name, list):
                regime_name_str = "_and_".join(
                    p for p in regime_name if p not in ("|", "&")
                )

            mask = self._apply_filter_mask(feats, regime_name, position)
            n_selected = int(mask.sum())
            if n_selected == 0:
                continue

            effective_position = position
            if reverse == -1:
                effective_position = "short" if position == "long" else "long"
            if effective_position == "long":
                ret_col, ret_raw_col = "return_long", "return_long_raw"
            else:
                ret_col, ret_raw_col = "return_short", "return_short_raw"

            chunk = feats.loc[mask].copy()
            chunk["position"] = position
            chunk["regime"] = regime_name_str
            chunk["ret"] = chunk[ret_col]
            if ret_raw_col in chunk.columns:
                chunk["ret_raw"] = chunk[ret_raw_col]

            table = pa.Table.from_pandas(chunk, preserve_index=False)
            del chunk
            # Narrow genuine float64 feature columns to float32 (the trainer
            # casts to float32 on load anyway) — halves the filter parquet.
            # Done BEFORE widening ints so integer columns keep full precision.
            table = _narrow_floats_to_float32(table)
            # Pre-promote: widen any integer columns to float64 BEFORE opening
            # the writer. Reason: pandas may infer int64 on the first symbol
            # (no NaNs in that slice) but float64 on a later symbol (where
            # NaNs appeared and forced promotion). The streaming writer's
            # schema is fixed at first write, so a later float64 column would
            # be rejected by .cast() — pre-widening here mirrors the offline
            # merge_filter_batches.py:_unified_schema pattern.
            table = _widen_ints_to_float(table)

            key = (regime_name_str, position)
            state = writers.get(key)
            if state is None:
                clean = regime_name_str.replace("_long", "").replace("_short", "")
                safe_name = "".join(
                    c if c.isalnum() or c == "_" else "_"
                    for c in f"{clean}_{position}"
                )
                final_path = os.path.join(save_dir, f"filter_{safe_name}.parquet")
                tmp_path = final_path + ".tmp"
                # If a prior aborted run left a stale .tmp, clear it so the
                # ParquetWriter starts fresh.
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                writer = pq.ParquetWriter(tmp_path, table.schema, compression="zstd")
                state = {
                    "writer": writer,
                    "schema": table.schema,
                    "tmp_path": tmp_path,
                    "final_path": final_path,
                    "n_rows": 0,
                    "safe_name": safe_name,
                }
                writers[key] = state
            else:
                # Schema reconciliation: column SET must match (fail-fast on
                # drift); column ORDER and dtype-promotion (int64→float64 from
                # NaN insertion in one symbol but not another) are reconciled
                # by casting onto the writer's schema. Mirrors
                # mjolnir/gauntlet/merge_filter_batches.py:_unified_schema +
                # tbl.cast() pattern.
                want_names = set(state["schema"].names)
                got_names = set(table.schema.names)
                missing = want_names - got_names
                extra = got_names - want_names
                if missing or extra:
                    raise RuntimeError(
                        f"Schema drift detected for filter {state['safe_name']!r} "
                        "between symbols — engineered feature column SET differs. "
                        f"Missing in this symbol: {sorted(missing)!r}; "
                        f"extra in this symbol: {sorted(extra)!r}. "
                        "This indicates non-deterministic feature engineering "
                        "across symbols and would corrupt the parquet."
                    )
                if not table.schema.equals(state["schema"]):
                    # Column set matches; cast to reconcile column order +
                    # dtype-promotion (e.g. int64→float64 from NaN promotion).
                    # safe=False allows numeric widening; will still raise on a
                    # truly incompatible cast (e.g. string→int).
                    table = table.select(state["schema"].names).cast(
                        state["schema"], safe=False,
                    )

            # row_group_size splits the input Table into multiple ~100k-row
            # row groups inside this single write_table call (pyarrow auto-
            # slices). Keeps the file's num_rows / num_row_groups ratio under
            # the rolling step's 300_000 rewrite threshold so the downstream
            # _rewrite_parquet_fine_rowgroups pass is a no-op for new files.
            state["writer"].write_table(table, row_group_size=_FILTER_ROW_GROUP_SIZE)
            state["n_rows"] += table.num_rows
            del table

    def _finalise_writers(
        self,
        writers: Dict[Tuple[str, str], Dict],
        success: bool,
    ) -> None:
        """Close every open ParquetWriter and finalise its .tmp file.

        Called from the streaming loop's ``finally`` so no tmp files leak.

        On ``success=True``:
          - Close each writer.
          - Rename the .tmp to its canonical ``filter_*.parquet`` path.
          - Drop the .tmp if zero rows were ever written (legitimate empty filter).

        On ``success=False`` (the streaming loop raised mid-write):
          - Close each writer (best-effort; a close failure here MUST NOT mask
            the original exception the caller is about to re-raise).
          - Delete every .tmp instead of renaming. A partially-written parquet
            renamed to its canonical name would be picked up by downstream
            globs (e.g. ``gauntlet/rolling_predict_returns.py``,
            ``optimize_thresholds.py``) as real data. Per CLAUDE.md
            "no silent fallbacks", a torn write must fail loudly — and the
            simplest "loud" signal is for the canonical file to simply not exist.
        """
        for key, state in list(writers.items()):
            try:
                state["writer"].close()
            except Exception as exc:
                # WHY: best-effort cleanup; if we are in the failure branch the
                # original exception is what callers care about, and even on
                # the success branch a close failure here just leaves a .tmp
                # we will handle below. Logging at warning is enough — do not
                # re-raise (we still need to process the rest of the writers).
                logger.warning(
                    "Failed to close ParquetWriter for %s: %s",
                    state["safe_name"], exc,
                )

            tmp_path = state["tmp_path"]
            final_path = state["final_path"]

            if not success:
                # Failure path: discard partial tmp, never rename to canonical.
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError as exc:
                        # WHY: cleanup is best-effort; the upstream exception
                        # is what matters and will re-raise after this returns.
                        logger.warning(
                            "Failed to remove partial tmp %s: %s",
                            tmp_path, exc,
                        )
                logger.warning(
                    "crashed mid-write — discarded partial filter %s",
                    tmp_path,
                )
                continue

            if state["n_rows"] == 0:
                # No rows ever written — drop the tmp file.
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                logger.info(
                    "Filter %s produced 0 rows; not writing parquet.",
                    state["safe_name"],
                )
                continue
            try:
                # Best-effort fsync of the parent dir for atomic rename guarantees.
                os.replace(tmp_path, final_path)
                logger.info(
                    "Saved filter %s (%d rows)", state["safe_name"], state["n_rows"],
                )
            except OSError as exc:
                logger.error(
                    "Failed to rename %s -> %s: %s",
                    tmp_path, final_path, exc,
                )

    # ------------------------------------------------------------------
    # Regime filter masks
    # ------------------------------------------------------------------

    def _apply_filter_mask(
        self,
        df: pd.DataFrame,
        filter_name,
        position: str,
    ) -> pd.Series:
        """Return a boolean Series: True = include in filter.

        Supports:
        - String filter names with _and_ / _or_ composition
        - List filters: ["filter_a", "&", "filter_b"]
        - Mjolnir-specific microstructure filters
        - Standard Agamotto price filters (baseline, rsi, macd, etc.)
        """
        # --- List support (recursive) ---
        if isinstance(filter_name, list):
            mask = None
            op = None
            for item in filter_name:
                item = str(item).strip()
                if item in ("|", "&"):
                    op = item
                else:
                    sub = self._apply_filter_mask(df, item, position)
                    if mask is None:
                        mask = sub
                    elif op == "|":
                        mask = mask | sub
                    else:
                        mask = mask & sub
            return mask if mask is not None else pd.Series(True, index=df.index)

        # --- String composition ---
        if isinstance(filter_name, str):
            name = filter_name.lower().strip()
            if "_and_" in name:
                parts = name.split("_and_")
                mask = None
                for p in parts:
                    sub = self._apply_filter_mask(df, p.strip(), position)
                    mask = sub if mask is None else (mask & sub)
                return mask if mask is not None else pd.Series(True, index=df.index)
            if "_or_" in name:
                parts = name.split("_or_")
                mask = None
                for p in parts:
                    sub = self._apply_filter_mask(df, p.strip(), position)
                    mask = sub if mask is None else (mask | sub)
                return mask if mask is not None else pd.Series(True, index=df.index)
        else:
            name = str(filter_name).lower().strip()

        if df.empty:
            return pd.Series(dtype=bool)

        # Strip trailing _long / _short from the filter name
        name = name.replace("_long", "").replace("_short", "")

        return self._named_filter(df, name, position)

    def _named_filter(self, df: pd.DataFrame, name: str, position: str) -> pd.Series:
        """Dispatch to a named filter implementation."""
        true = pd.Series(True, index=df.index)

        # ---- Microstructure filters (Mjolnir-specific) ----

        if name == "baseline":
            return true

        if name == "high_liquidation_pressure":
            col = "liq_burst_ratio"
            if col not in df.columns:
                return true
            return df[col] > df[col].quantile(0.75)

        if name == "low_liquidation_pressure":
            col = "liq_burst_ratio"
            if col not in df.columns:
                return true
            return df[col] < df[col].quantile(0.25)

        if name == "funding_positive":
            col = "funding_rate"
            if col not in df.columns:
                return true
            return df[col] > 0

        if name == "funding_negative":
            col = "funding_rate"
            if col not in df.columns:
                return true
            return df[col] < 0

        if name == "deep_book":
            col = "depth_imbalance_L5"
            if col not in df.columns:
                return true
            if position == "long":
                return df[col] > df[col].quantile(0.6)
            return df[col] < df[col].quantile(0.4)

        if name == "trade_imbalance":
            col = "trade_imbalance"
            if col not in df.columns:
                return true
            if position == "long":
                return df[col] > 0
            return df[col] < 0

        if name == "basis_premium":
            col = "basis_pct"
            if col not in df.columns:
                return true
            return df[col] > 0

        if name == "basis_discount":
            col = "basis_pct"
            if col not in df.columns:
                return true
            return df[col] < 0

        if name == "pre_funding_settlement":
            col = "pre_funding"
            if col not in df.columns:
                return true
            return df[col] > 0

        if name == "oi_expansion":
            col = "oi_velocity"
            if col not in df.columns:
                return true
            return df[col] > 0

        if name == "oi_contraction":
            col = "oi_velocity"
            if col not in df.columns:
                return true
            return df[col] < 0

        if name == "ofi_positive":
            col = "ofi_agg"
            if col not in df.columns:
                return true
            if position == "long":
                return df[col] > 0
            return df[col] < 0

        if name == "tight_spread":
            col = "relative_spread"
            if col not in df.columns:
                return true
            return df[col] < df[col].quantile(0.5)

        if name == "wide_spread":
            col = "relative_spread"
            if col not in df.columns:
                return true
            return df[col] > df[col].quantile(0.5)

        # ---- Standard price filters (shared with Agamotto) ----

        mvg1 = df.get("mvg1")
        mvg2 = df.get("mvg2")
        close = df.get("close", df.get("mid_price"))

        if mvg1 is None or mvg2 is None or close is None:
            # Return true if base price columns missing
            return true

        up_trend = (close > mvg1) & (mvg1 > mvg2)
        down_trend = (close < mvg1) & (mvg1 < mvg2)

        if name in ("trend_aligned",):
            return up_trend if position == "long" else down_trend

        if name == "strong_trend":
            mvg3 = df.get("mvg3")
            if position == "long":
                return up_trend & (mvg2 > mvg3) if mvg3 is not None else up_trend
            return down_trend & (mvg2 < mvg3) if mvg3 is not None else down_trend

        if name == "ma_momentum":
            mvg3 = df.get("mvg3")
            if position == "long":
                return (mvg1 > mvg2) & (mvg1 > mvg3) if mvg3 is not None else (mvg1 > mvg2)
            return (mvg1 < mvg2) & (mvg1 < mvg3) if mvg3 is not None else (mvg1 < mvg2)

        if name == "rsi_oversold":
            col = "rsi"
            if col not in df.columns:
                return true
            return df[col] < 30

        if name == "rsi_overbought":
            col = "rsi"
            if col not in df.columns:
                return true
            return df[col] > 70

        if name == "macd_bullish":
            col = "macdhist"
            if col not in df.columns:
                return true
            return df[col] > 0 if position == "long" else df[col] < 0

        if name == "macd_bearish":
            col = "macdhist"
            if col not in df.columns:
                return true
            return df[col] < 0 if position == "long" else df[col] > 0

        if name == "adx_trend":
            col = "adx"
            if col not in df.columns:
                return true
            return df[col] > 25

        if name == "bb_rebound":
            if position == "long":
                col = "bb_lower"
                if col not in df.columns:
                    return true
                return close < df[col]
            else:
                col = "bb_upper"
                if col not in df.columns:
                    return true
                return close > df[col]

        if name in ("high_volume", "vol_breakout"):
            col = "vol_ratio"
            if col not in df.columns:
                return true
            threshold = 2.0 if name == "vol_breakout" else 1.0
            return df[col] > threshold

        if name == "low_volume":
            col = "vol_ratio"
            if col not in df.columns:
                return true
            return df[col] < 1.0

        if name == "high_vol":
            col = "price_range_pct"
            if col not in df.columns:
                return true
            return df[col] > df[col].quantile(0.5)

        if name == "low_vol":
            col = "price_range_pct"
            if col not in df.columns:
                return true
            return df[col] < df[col].quantile(0.5)

        if name == "mom_positive":
            col = "mom"
            if col not in df.columns:
                return true
            return df[col] > 0 if position == "long" else df[col] < 0

        raise ValueError(f"Unknown filter: {name!r}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_ladder_returns(
        self,
        df: pd.DataFrame,
        close_col: str,
        low_col: str,
        high_col: str,
        horizon_bars: int = 1,
    ) -> pd.DataFrame:
        """Compute ladder-adjusted return columns for a single symbol.

        Mirrors the agamotto.research.AgamottoResearch.engineer_features ladder
        logic (same step_size = 1 bps, same clipping, same column names),
        generalized to a variable forward horizon so the fill window matches
        the prediction window.

        At horizon_bars == 1 the behaviour is identical to the original
        1-bar shift, preserving native-mode parity (TIME_UNIT == bar
        resolution, e.g. mjolnir.base.5s_1). For boundary-aligned
        experiments (e.g. mjolnir.base.30s_1: 5s bars predicting the next
        30s boundary close) callers should pass
        horizon_bars = TIME_UNIT_seconds / bar_tf_seconds so the low/high
        lookahead spans the full prediction horizon, instead of only the
        next single bar (which under-counted ladder fills and produced
        a frictionless close-to-close target).

        Args:
            df:           DataFrame with at least close_col, low_col, high_col.
            close_col:    Name of the close price column.
            low_col:      Name of the low price column.
            high_col:     Name of the high price column.
            horizon_bars: Forward horizon used both for the price return
                          (close[t+h] / close[t] - 1) and for the
                          low/high min/max lookahead window. Must be >= 1.

        Returns:
            DataFrame with columns: return_long, return_short,
            return_long_raw, return_short_raw.
        """
        if horizon_bars < 1:
            raise ValueError(
                f"horizon_bars must be >= 1, got {horizon_bars}")

        step_size = 0.0001
        ladder = int(self.config.get("LADDER", 1) or 0)
        # FEE is required (no fallback): see commit 0500d8fa for rationale.
        # The historical `or 0.0` collapsed any falsy FEE to the default.
        fee_rate = float(self.config["FEE"]) / 10000.0

        close = df[close_col]
        low_series = df[low_col]
        high_series = df[high_col]

        # Forward h-bar price return: close[t+h] / close[t] - 1.
        price_return = close.pct_change(
            horizon_bars, fill_method=None).shift(-horizon_bars)
        close_safe = close.replace(0, np.nan)

        # Forward-rolling min low / max high over (t, t+horizon_bars]:
        # shift(-1) so position t holds low[t+1], then reverse-rolling so
        # the window covers the next horizon_bars bars (exclusive of t).
        # At horizon_bars=1 this reduces to low.shift(-1) / high.shift(-1).
        low_shifted = low_series.shift(-1)
        high_shifted = high_series.shift(-1)
        low_window = (
            low_shifted.iloc[::-1]
            .rolling(horizon_bars, min_periods=1)
            .min()
            .iloc[::-1]
        )
        high_window = (
            high_shifted.iloc[::-1]
            .rolling(horizon_bars, min_periods=1)
            .max()
            .iloc[::-1]
        )

        distance_long = ((close_safe - low_window) / close_safe).replace(
            [np.inf, -np.inf], np.nan)
        long_layers = np.floor(distance_long / step_size).clip(
            lower=0, upper=ladder).fillna(0).astype(int)
        total_long_layers = 1 + long_layers

        distance_short = ((high_window - close_safe) / close_safe).replace(
            [np.inf, -np.inf], np.nan)
        short_layers = np.floor(distance_short / step_size).clip(
            lower=0, upper=ladder).fillna(0).astype(int)
        total_short_layers = 1 + short_layers

        fee_cost = (fee_rate * 2.0) if fee_rate else 0.0
        return_long = ((price_return - fee_cost) * total_long_layers).rename("return_long")
        # Short profits when price falls: negate price_return (and fee, which is paid either side).
        # Matches features.py:449 `"return_short": -(forward_return + fee)`.
        return_short = ((-(price_return + fee_cost)) * total_short_layers).rename("return_short")
        return_long_raw = (price_return * total_long_layers).rename("return_long_raw")
        return_short_raw = ((-price_return) * total_short_layers).rename("return_short_raw")

        return pd.concat(
            [return_long, return_short, return_long_raw, return_short_raw], axis=1)

    def _resolve_out_dir(self) -> str:
        version = self.config.get("VERSION")
        if not version:
            raise ValueError("VERSION missing from config.")
        if "OUTPUT_DIR" in self.config:
            out_dir = self.config["OUTPUT_DIR"]
        else:
            project = self.config.get("PROJECT", "mjolnir/gauntlet")
            out_dir = os.path.join(project, f"pred_{version}")
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(self.home_root, out_dir)
        return out_dir

    def _load_regime_stack(self, out_dir: str) -> List[Dict]:
        """Load regime stack from REGIME_STACK_PATH in config. CSV-only, no fallbacks.

        CSV rows have regime/position/model columns; returns deduplicated
        list of {regime, position} dicts (one entry per unique pair).

        Raises:
            KeyError: if REGIME_STACK_PATH is not set in config.
            ValueError: if path does not end in .csv.
            FileNotFoundError: if the configured path does not exist.
        """
        from utils.lib import load_regime_stack

        # Resolve relative paths before calling shared loader
        if "REGIME_STACK_PATH" not in self.config:
            raise KeyError("REGIME_STACK_PATH is required in config but not set")
        candidate = self.config["REGIME_STACK_PATH"]
        if not os.path.isabs(candidate) and not os.path.exists(candidate):
            resolved = os.path.join(self.home_root, candidate)
            if os.path.exists(resolved):
                candidate = resolved

        raw_stack = load_regime_stack(candidate)

        # Deduplicate to unique (regime, position) pairs
        seen = set()
        stack = []
        for row in raw_stack:
            key = (row["regime"], row["position"])
            if key not in seen:
                seen.add(key)
                stack.append({"regime": row["regime"], "position": row["position"]})
        logger.info("Deduplicated to %d unique regime/position pairs", len(stack))
        return stack
