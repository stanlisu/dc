"""Streaming filter-parquet writer for MjolnirResearch.create().

Handles per-symbol streaming writes of (regime, position) filter parquets
so that peak RAM is bounded by ONE symbol's engineered features.

Extracted from research.py to keep each module under ~700 lines for
PyArmor trial compatibility.
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.fs as pafs
import pyarrow.parquet as pq

from .features import MjolnirFeatures, _TF_SECONDS
from .multi_tf_merge import merge_cross_tf_features
from .regime_filters import apply_filter_mask
from .ladder import compute_ladder_returns, resolve_fill_mode


def _obf():
    """Lazy accessor for the vendored obfuscation codec (see _obf/codec.py)."""
    from .._obf.codec import default
    return default()
from .utils import normalize_symbol

logger = logging.getLogger(__name__)

# Row-group size for streaming filter parquets. Matches
# gauntlet/rolling_predict_returns.py::_rewrite_parquet_fine_rowgroups
# default batch_size (100_000). The rolling step's needs_rewrite check
# triggers when num_rows / num_row_groups > 300_000, so writing at
# 100_000 here means newly-produced filter parquets skip the rewrite
# step entirely. Keep these two constants aligned.
_FILTER_ROW_GROUP_SIZE = 100_000

# Bucket region for s3:// filter writes (Route B). The `tardis-stan-data`
# bucket lives in ap-northeast-1; overridable via env for a moved/mirrored
# bucket. This is a bucket-location constant, not a strategy/PnL config, so a
# named default here does not violate the CLAUDE.md "no magic-number config
# defaults" rule (which targets FEE/CAPITAL/thresholds).
_S3_REGION = os.environ.get("MJOLNIR_S3_REGION", "ap-northeast-1")


def _is_s3(path) -> bool:
    """True if ``path`` is an ``s3://`` URI (Route B direct-to-S3 filter write)."""
    return str(path).startswith("s3://")


def _s3_key(uri: str) -> str:
    """``s3://bucket/prefix`` -> ``bucket/prefix``.

    pyarrow's ``S3FileSystem`` addresses objects by a bucket-qualified key
    (``bucket/key``), NOT a bare key — mirrors stan_sim/scripts/build_bars.py.
    """
    return uri[len("s3://"):].rstrip("/")


def _open_s3fs() -> "pafs.S3FileSystem":
    """Open an ``S3FileSystem`` using the DEFAULT AWS credential chain.

    No creds file (unlike build_bars.py's ``s3_fs()``): the ``stan`` user's
    default AWS creds (``us-s3-rw-user``) already grant write to
    ``tardis-stan-data``. Fails LOUDLY on any init/credential/region error —
    per CLAUDE.md there is NO fallback to the local filesystem for an s3://
    request; a broken S3 config must abort the run, not silently write local.
    """
    try:
        fs = pafs.S3FileSystem(region=_S3_REGION)
    except Exception as exc:
        raise RuntimeError(
            f"failed to init S3FileSystem(region={_S3_REGION!r}) for s3:// filter "
            f"writes via the default AWS credential chain: {exc!r}. Refusing to "
            "fall back to local (CLAUDE.md: no silent fallbacks)."
        ) from exc
    return fs


def _s3_delete_quiet(s3fs: "pafs.S3FileSystem", key: str) -> None:
    """Best-effort delete of an S3 object (used only on the abort/empty paths).

    Guarded so a missing object is a no-op. WHY (swallow is safe): this runs
    only in cleanup — either the streaming loop already raised (that exception
    is what callers must see, not a secondary delete failure) or we are dropping
    a legitimately-empty 0-row filter. A delete failure here leaves at most a
    stray object; it never masks the primary error.
    """
    try:
        info = s3fs.get_file_info(key)
        if info.type != pafs.FileType.NotFound:
            s3fs.delete_file(key)
    except Exception as exc:
        logger.warning("failed to delete s3 object %s: %s", key, exc)


def _widen_ints_to_float(table: pa.Table) -> pa.Table:
    """Promote every integer column in `table` to float64.

    Streaming writes lock the writer's schema on the first row group. If sym A
    has no NaNs in column X (pandas -> int64) but sym B has NaNs (pandas ->
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


def _narrow_floats_to_float32(table: pa.Table) -> pa.Table:
    """Narrow every float64 column to float32 before writing filter parquets.

    The rolling trainer (`_read_pq` in rolling_predict_returns.py) already casts
    every float64 feature column to float32 on load, so float64 on disk is pure
    wasted storage (~2x on the feature matrix, which is ~90% of the columns).
    Narrowing at write halves the filter parquet with ZERO effect on training --
    the trainer sees float32 either way.

    Applied BEFORE `_widen_ints_to_float`, so genuine integer columns (later
    promoted to float64 only for streaming-schema consistency) keep full
    precision and are never narrowed -- only true float64 features are cast.
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


def stream_filter_parquets(
    config: Dict,
    symbol_bars: Dict[str, pd.DataFrame],
    multi_tf_bars: Dict[str, Dict[str, pd.DataFrame]],
    out_dir: str,
    regime_stack: List[Dict],
    apply_mask_fn: Callable,
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
    engineer_features ordering -- BTC first, others second).

    Args:
        config:         Dictionary loaded from setting.json.
        symbol_bars:    Per-symbol bar DataFrames (will be mutated — popped).
        multi_tf_bars:  Per-TF per-symbol bar DataFrames (will be mutated — popped).
        out_dir:        Output directory path.
        regime_stack:   List of {regime, position} dicts.
        apply_mask_fn:  Callable(df, filter_name, position) -> boolean Series.
    """
    cfg = config
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
    # Import here to avoid circular — _DEFAULT_FEATURE_WINDOWS is a constant
    # defined in research.py.
    from .research import (_DEFAULT_FEATURE_WINDOWS,
                           resolve_fill_zero_trade,
                           resolve_ta_price_source)
    feature_windows = list(_DEFAULT_FEATURE_WINDOWS)
    time_unit = cfg.get("TIME_UNIT", "5s")
    bar_tf = "5s" if cfg.get("TRAIN_BARS_DIR") else time_unit
    # Required, no default. Bars built without the zero-trade price ffill carry
    # genuine NaN prices; a blanket fillna(0.0) over those pins 21 TA columns
    # to exactly 0.0 for the rest of the series. Mirrors engineer_features.
    zero_fill_prices = resolve_fill_zero_trade(cfg)
    # Required, no default — see resolve_ta_price_source. Mirrors
    # engineer_features; the TARGET's price source is unaffected.
    ta_price_source = resolve_ta_price_source(cfg)

    feat_engine = MjolnirFeatures(
        feature_windows=feature_windows,
        target_horizon=target_horizon,
        fee_rate=fee_rate,
        bar_tf=bar_tf,
        target_tf=time_unit,
        zero_fill_prices=zero_fill_prices,
        ta_price_source=ta_price_source,
    )

    multi_tfs = list(multi_tf_bars.keys())
    tf_engines = {
        tf: MjolnirFeatures(
            feature_windows=[1],
            target_horizon=target_horizon,
            fee_rate=fee_rate,
            prefix=tf,
            bar_tf=tf,
            target_tf=tf,
            zero_fill_prices=zero_fill_prices,
            ta_price_source=ta_price_source,
        )
        for tf in multi_tfs
    }

    # BTC must be processed first so cross-features are available for other symbols.
    _BTC = "BINANCE_PERP_BTC_USDT"
    ordered = sorted(symbol_bars.keys(), key=lambda s: (s != _BTC, s))

    # Map native -> canonical symbol name (preserves SYMBOLS spelling for the
    # `symbol` column in output parquets).
    symbols_cfg = config.get("SYMBOLS", [])
    native_to_sym = {normalize_symbol(s): s for s in symbols_cfg}

    reverse = int(config.get("REVERSE", 1))

    # Route B: write filter parquets DIRECTLY to S3 via S3FileSystem when
    # out_dir is an s3:// URI (bypasses the read-only s3fs mount). Local paths
    # keep the byte-for-byte-unchanged local-FS path below.
    if _is_s3(out_dir):
        s3fs = _open_s3fs()
        # S3 has no directories — nothing to mkdir. save_dir is a bucket-
        # qualified key prefix ("bucket/prefix/filter").
        save_dir = f"{_s3_key(out_dir)}/filter"
        logger.info("Route B: streaming filters directly to s3://%s", save_dir)
    else:
        s3fs = None
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
            bars = symbol_bars.pop(native, None)
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
                tf_bars_for_sym = multi_tf_bars.get(tf, {}).pop(native, None)
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
                ladder_cols = compute_ladder_returns(
                    config, feats, "close", "low", "high",
                    horizon_bars=horizon_bars,
                )
                for col in ladder_cols.columns:
                    feats[col] = ladder_cols[col].values
                del ladder_cols

            # Drop rows with NaN target — matches verticalize() semantics.
            # limit_then_taker looks 2h ahead; gate on target cols too.
            if "return" in feats.columns:
                keep = feats["return"].notna()
                if resolve_fill_mode(config) == "limit_then_taker":
                    for _tcol in ("return_long", "return_short"):
                        if _tcol in feats.columns:
                            keep &= feats[_tcol].notna()
                feats = feats[keep]
            if feats.empty:
                logger.warning("Feature frame for %s empty after NaN-drop; skipping", native)
                del feats
                gc.collect()
                continue

            # reset_index so the row-group has a clean integer index.
            feats = feats.reset_index(drop=True)

            # Write one slice per (regime, position) into the lazy writers.
            write_symbol_to_filters(
                feats=feats,
                regime_stack=regime_stack,
                writers=writers,
                save_dir=save_dir,
                reverse=reverse,
                apply_mask_fn=apply_mask_fn,
                config=config,
                s3fs=s3fs,
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
        finalise_writers(writers, success=success)


def write_symbol_to_filters(
    feats: pd.DataFrame,
    regime_stack: List[Dict],
    writers: Dict[Tuple[str, str], Dict],
    save_dir: str,
    reverse: int,
    apply_mask_fn: Callable,
    config: Dict,
    s3fs: Optional["pafs.S3FileSystem"] = None,
) -> None:
    """For one symbol's vertical features, append a row group per regime.

    Lazily opens a ParquetWriter per (regime_name_str, position) on first
    non-empty slice. Schema is captured from that first table; subsequent
    symbols MUST have the same schema or we raise (no silent column drift).

    When ``s3fs`` is provided (Route B), ``save_dir`` is a bucket-qualified S3
    key prefix and each writer targets the FINAL key directly (S3 multipart
    upload is atomic — the object only becomes visible on ``close()``), so the
    ``.tmp``->rename dance used for the local FS is unnecessary.
    """
    for regime in regime_stack:
        if "regime" not in regime:
            raise KeyError("regime row missing required 'regime' key (no baseline default)")
        regime_name = regime["regime"]
        position = regime.get("position", "long")

        regime_name_str = regime_name
        if isinstance(regime_name, list):
            regime_name_str = "_and_".join(
                p for p in regime_name if p not in ("|", "&")
            )

        mask = apply_mask_fn(feats, regime_name, position)
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

        # Obfuscation: persist feature columns under opaque codes (TF prefix +
        # rolling-stat suffix preserved). Targets/metadata/OHLCV aren't in the
        # feature map and pass through unchanged. Coded schema flows to meta/preds.
        chunk = chunk.rename(columns=_obf().encode_columns(chunk.columns))

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
            if s3fs is not None:
                # Write DIRECTLY to the final S3 key. The multipart upload is
                # atomic: the object appears only on a successful close(), so no
                # partial/torn parquet can be picked up mid-write and no .tmp
                # staging is needed. Abort/empty cleanup deletes the final key
                # (see finalise_writers).
                final_path = f"{save_dir}/filter_{safe_name}.parquet"
                tmp_path = None
                writer = pq.ParquetWriter(
                    final_path, table.schema, compression="zstd", filesystem=s3fs,
                )
            else:
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
                "s3fs": s3fs,
            }
            writers[key] = state
        else:
            # Schema reconciliation: column SET must match (fail-fast on
            # drift); column ORDER and dtype-promotion (int64->float64 from
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
                # dtype-promotion (e.g. int64->float64 from NaN promotion).
                # safe=False allows numeric widening; will still raise on a
                # truly incompatible cast (e.g. string->int).
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


def finalise_writers(
    writers: Dict[Tuple[str, str], Dict],
    success: bool,
) -> None:
    """Close every open ParquetWriter and finalise its .tmp file.

    Called from the streaming loop's ``finally`` so no tmp files leak.

    On ``success=True``:
      - Close each writer. A ``close()`` failure here is FATAL: on S3 it means
        the CompleteMultipartUpload never published the object (so n_rows>0 but
        nothing landed); on local it means the ``.tmp`` lacks a valid footer.
        Publishing/renaming either would promote a failed write to canonical and
        make the downstream merge silently miss that regime's rows. So a
        success-path close() failure is collected, the artifact is best-effort
        deleted (never published), and a RuntimeError is re-raised after
        finalising the remaining writers so ``run_research`` exits non-zero and
        the sharded driver aborts before merge.
      - Rename the .tmp to its canonical ``filter_*.parquet`` path.
      - Drop the .tmp if zero rows were ever written (legitimate empty filter).

    On ``success=False`` (the streaming loop raised mid-write):
      - Close each writer (best-effort; a close failure here MUST NOT mask
        the original exception the caller is about to re-raise).
      - Delete every .tmp instead of renaming. A partially-written parquet
        renamed to its canonical name would be picked up by downstream
        globs (e.g. ``gauntlet/rolling_predict_returns.py``,
        ``optimize_thresholds.py``) as real data. Per CLAUDE.md
        "no silent fallbacks", a torn write must fail loudly -- and the
        simplest "loud" signal is for the canonical file to simply not exist.
    """
    # Success-path close() failures collected here and re-raised at the end so
    # every writer is finalised first (no leaked open handles) but the run still
    # fails loud.
    close_failures: List[Tuple[str, Exception]] = []
    for key, state in list(writers.items()):
        close_ok = True
        try:
            state["writer"].close()
        except Exception as exc:
            close_ok = False
            if success:
                # SUCCESS PATH: close() failure is FATAL — the object was NOT
                # published (S3 multipart not completed) / the local footer was
                # not written. Do NOT log "Saved", do NOT rename/publish. Record
                # it, drop any partial artifact below, and re-raise after the loop.
                logger.error(
                    "FATAL: ParquetWriter.close() failed for %s on the success "
                    "path — refusing to publish this filter (unpublished/torn "
                    "write): %s",
                    state["safe_name"], exc,
                )
                close_failures.append((state["safe_name"], exc))
            else:
                # WHY: abort branch — the original exception is what callers care
                # about; we delete the (unpublished/torn) artifact below anyway,
                # so a secondary close failure must NOT mask the primary error.
                logger.warning(
                    "Failed to close ParquetWriter for %s: %s",
                    state["safe_name"], exc,
                )

        tmp_path = state["tmp_path"]
        final_path = state["final_path"]
        s3fs = state.get("s3fs")

        if success and not close_ok:
            # Success path but this writer's close() failed: never publish. Best-
            # effort delete of any partial/unpublished artifact so a torn write
            # can't masquerade as real data, then move on (we re-raise below).
            if s3fs is not None:
                _s3_delete_quiet(s3fs, final_path)
            elif tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError as exc:
                    # WHY: cleanup is best-effort; the RuntimeError we re-raise
                    # at the end of the loop is what fails the run loud.
                    logger.warning(
                        "Failed to remove partial tmp %s: %s", tmp_path, exc,
                    )
            continue

        if not success:
            # Failure path: never leave a canonical filter behind.
            if s3fs is not None:
                # close() above completed the multipart, so a truncated-but-
                # valid parquet now exists at the final key — delete it so a
                # torn write can't masquerade as real data (CLAUDE.md: no
                # silent fallbacks; a torn write must fail loudly / not exist).
                _s3_delete_quiet(s3fs, final_path)
            elif os.path.exists(tmp_path):
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
                final_path if s3fs is not None else tmp_path,
            )
            continue

        if state["n_rows"] == 0:
            # No rows ever written — drop the empty artifact.
            if s3fs is not None:
                # close() wrote an empty (0-row) parquet object; remove it.
                _s3_delete_quiet(s3fs, final_path)
            elif os.path.exists(tmp_path):
                os.remove(tmp_path)
            logger.info(
                "Filter %s produced 0 rows; not writing parquet.",
                state["safe_name"],
            )
            continue

        if s3fs is not None:
            # S3 multipart already finalised the object atomically on close();
            # nothing to rename.
            logger.info(
                "Saved filter %s (%d rows) -> s3://%s",
                state["safe_name"], state["n_rows"], final_path,
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

    if close_failures:
        names = ", ".join(n for n, _ in close_failures)
        raise RuntimeError(
            f"ParquetWriter.close() failed on the success path for "
            f"{len(close_failures)} filter(s) [{names}] — these regimes were NOT "
            "published (S3 CompleteMultipartUpload not completed / local parquet "
            "footer not written) and their partial artifacts were deleted. "
            "Failing the run so run_research exits non-zero and the sharded "
            "driver aborts before merge (CLAUDE.md: no silent fallbacks — a "
            "failed write must never be reported as a saved filter)."
        ) from close_failures[0][1]
