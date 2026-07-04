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
from .regime_filters import apply_filter_mask
from .ladder import compute_ladder_returns
from .streaming import (
    stream_filter_parquets,
    _FILTER_ROW_GROUP_SIZE,
)

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
            # Drop rows with NaN in return columns (last target_horizon bars).
            # limit_then_taker looks 2h ahead, so its return_long/short are NaN
            # for the final 2h bars while "return" (horizon h) is not — gate on
            # the target columns too so NaN targets can never reach training.
            if "return" in feats.columns:
                keep = feats["return"].notna()
                if str(self.config.get("LADDER_FILL_MODE", "ladder")).lower() == "limit_then_taker":
                    for _tcol in ("return_long", "return_short"):
                        if _tcol in feats.columns:
                            keep &= feats[_tcol].notna()
                feats = feats[keep]
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

        if "regime" not in regime:
            raise KeyError("regime row missing required 'regime' key (no baseline default)")
        regime_name = regime["regime"]
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
            # Route B (s3://) is implemented ONLY on the per-symbol streaming
            # path (stream_filter_parquets), which is what create() — and thus
            # the sharded regen — uses. This chunked filter_signals(save=True)
            # path is the separate gauntlet/regenerate_filter.py route and is
            # local-FS only. Fail LOUD rather than silently mis-writing a
            # local-FS parquet to an s3:// key (CLAUDE.md: no silent fallbacks).
            if str(out_dir).startswith("s3://"):
                raise NotImplementedError(
                    "filter_signals(save=True) does not support s3:// out_dir; "
                    "use the streaming create() path (run_research.py) for Route B "
                    f"S3 writes. Got out_dir={out_dir!r}."
                )
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
                    # auto-slices). Mirrors the streaming path so
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
        the whole 29-symbol x multi-TF accumulation that previously OOM'd at
        ~150 GB.

        Returns:
            Output directory path.
        """
        if not self._symbol_bars:
            self.load()

        out_dir = self._resolve_out_dir()
        # Route B: an s3:// OUTPUT_DIR has no directories to create — the
        # streaming writer opens objects directly via S3FileSystem. Only mkdir
        # for local paths (prod ubuntu-on-mount is unchanged).
        if not str(out_dir).startswith("s3://"):
            os.makedirs(out_dir, exist_ok=True)

        # Load regime stack BEFORE engineering — fail fast on misconfig.
        regime_stack = self._load_regime_stack(out_dir)
        logger.info("Processing %d regimes via per-symbol streaming...", len(regime_stack))

        stream_filter_parquets(
            config=self.config,
            symbol_bars=self._symbol_bars,
            multi_tf_bars=self._multi_tf_bars,
            out_dir=out_dir,
            regime_stack=regime_stack,
            apply_mask_fn=self._apply_filter_mask,
        )

        # Drop residual bar dicts — engineer/stream consumed them already but
        # be defensive in case an early-return left state behind.
        self._symbol_bars.clear()
        self._multi_tf_bars.clear()
        gc.collect()
        return out_dir

    # ------------------------------------------------------------------
    # Thin delegation methods (preserve private-method API for test compat)
    # ------------------------------------------------------------------

    def _apply_filter_mask(
        self,
        df: pd.DataFrame,
        filter_name,
        position: str,
    ) -> pd.Series:
        """Delegate to standalone apply_filter_mask()."""
        return apply_filter_mask(df, filter_name, position)

    def _compute_ladder_returns(
        self,
        df: pd.DataFrame,
        close_col: str,
        low_col: str,
        high_col: str,
        horizon_bars: int = 1,
    ) -> pd.DataFrame:
        """Delegate to standalone compute_ladder_returns()."""
        return compute_ladder_returns(
            self.config, df, close_col, low_col, high_col, horizon_bars)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_out_dir(self) -> str:
        version = self.config.get("VERSION")
        if not version:
            raise ValueError("VERSION missing from config.")
        if "OUTPUT_DIR" in self.config:
            out_dir = self.config["OUTPUT_DIR"]
        else:
            project = self.config.get("PROJECT", "mjolnir/gauntlet")
            out_dir = os.path.join(project, f"pred_{version}")
        # Route B: an s3:// URI is already absolute (bucket-qualified) — do NOT
        # prepend home_root (os.path.isabs("s3://...") is False, which would
        # otherwise mangle it into "<home_root>/s3://..." and drop it back onto
        # the local FS).
        if not str(out_dir).startswith("s3://") and not os.path.isabs(out_dir):
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
