# Mjolnir

Tick-data ML research pipeline. Applies the same regime + rolling-ML methodology as Agamotto to microsecond tick data from Binance futures, extracting microstructure signals invisible at candle resolution.

## What It Does

Reads 5 raw data streams (trades, book ticker, order book L25, derivative ticker, liquidations) from Tardis Binance archives, aligns them into time bars (default 5s, configurable to 15s/30s/1m/5m), engineers 100+ microstructure features (order flow imbalance, Kyle's lambda, depth ratios, funding rate, liquidation cascades), then produces `filter/*.parquet` files in the same format as Agamotto -- allowing all downstream gauntlet scripts to be reused verbatim.

> **Live trading lives in `knull/`** (LTP, Binance PAPI, OKX executors). Mjolnir's job is research + bar building only. See [`knull/README.md`](../knull/README.md) for the executor architecture, WebSocket loops, and per-symbol state machines.

## Architecture

```
Data Layer:  MjolnirLoader -> StreamAligner -> 5s bars (17,280/day)
Feature Layer:  MjolnirFeatures -> 100+ microstructure + price features
Research Layer:  MjolnirResearch -> 13 regimes -> filter parquets
Trading Layer:  MjolnirTrading -> emits signals via UDS bridge -> knull executor
```

## Key Files

| File | Description |
|------|-------------|
| `core/loader.py` | `MjolnirLoader`: reads daily Tardis gzip shards per symbol/stream |
| `core/aligner.py` | `StreamAligner`: merges 5 streams into uniform time-bar DataFrames |
| `core/features.py` | `MjolnirFeatures`: microstructure + price features at [30, 60, 300, 900] bar windows |
| `core/research.py` | `MjolnirResearch`: top-level pipeline class (Agamotto-compatible output format) |
| `trading.py` | `MjolnirTrading`: live inference with rolling bar buffer, ensemble voting (min signal count); produces signals consumed by `knull/` |
| `gauntlet/build_bars.py` | Step 1: raw shards -> parquet bars (parallel via joblib) |
| `gauntlet/run_research.py` | Step 2: engineer features, filter by 13 regimes |
| `tests/` | `test_aligner`, `test_bridge_qty`, `test_features`, `test_loader`, `test_research_date_filter`, `test_research_reverse`, `test_rolling`, `test_trading`, `test_trading_reverse` |

## Microstructure Features (100+)

| Category | Features |
|----------|----------|
| **Book** | Relative spread, microprice vs mid, depth imbalance (L1/L3/L5) |
| **Trade flow** | Trade imbalance, dollar trade intensity, Kyle's lambda |
| **OFI** | Multi-level order flow imbalance (Cont et al. 2023) |
| **Derivatives** | Basis %, funding rate, OI velocity/acceleration |
| **Liquidation** | Burst ratio, directional imbalance, pre-funding signals |
| **Price** | TA-Lib indicators (MA, RSI, ATR, MACD, Bollinger Bands) |
| **Cross-asset** | BTC correlation features for non-BTC symbols |
| **Rolling** | All above computed at [30, 60, 300, 900] bar windows |

## Regime Filters (13)

baseline, liquidation_pressure, high_funding, low_funding, depth_imbalance_bid, depth_imbalance_ask, oi_expansion, oi_contraction, high_ofi, low_ofi, wide_spread, narrow_spread, high_trade_intensity

## Stream Alignment

| Stream | Aggregation |
|--------|-------------|
| Trades | OHLCV, buy/sell volume, trade imbalance, VWAP, n_trades |
| Book ticker | Last-value bid/ask price/amount per bar |
| Order book L25 | Depth at 25 levels, aggregated to L1/L3/L5 |
| Derivative ticker | Forward-filled mark price, index price, funding rate, open interest |
| Liquidations | Accumulated sum per side (long/short notional per bar) |

## Live Trading

Live execution is a two-process pipeline. Mjolnir runs the **bridge** (loads weights, builds live bars from WebSocket ticks, emits signals over a Unix domain socket); `knull/` runs the **executor** (per-symbol state machine, order placement, venue API). Both processes read the same merged `setting.json`. See [`knull/README.md`](../knull/README.md) for executor internals.

```bash
# Bridge (signal source)
bash knull/run_mjolnir_bridge.sh mjolnir/gauntlet/pred_mjolnir.base.5s_1

# Executor (consumes signals over SOCKET_PATH)
KNULL_PID_NAME=knull_ltp_mjolnir5s KNULL_LOG_NAME=knull_ltp_mjolnir5s.log \
    bash knull/run_knull.sh mjolnir/gauntlet/pred_mjolnir.base.5s_1/setting.json
```

## Usage (Research)

```bash
# Step 1: Build time bars from raw shards
PYTHONPATH="agamotto_pkg/src:." python mjolnir/gauntlet/build_bars.py \
  -c mjolnir/gauntlet/pred_mjolnir.base.5s_1 --start 20240629 --end 20250101 --workers 8

# Step 2: Engineer features + regime filtering
PYTHONPATH="agamotto_pkg/src:." python mjolnir/gauntlet/run_research.py \
  -c mjolnir/gauntlet/pred_mjolnir.base.5s_1

# Steps 3-7: drive the full chain via the canonical pipeline script
bash mjolnir/gauntlet/run_mjolnir_pipeline.sh pred_mjolnir.base.5s_1
```

## Configuration (`setting.json`)

The merged setting carries both research and live-execution config. All path conventions (LOCAL vs NAS) and per-artifact routing are documented canonically in [`mjolnir/gauntlet/README.md`](gauntlet/README.md). Key descriptions below; type, what it points to, producer (who writes / hand-edits), and consumer (who reads).

### Identity

| Key | Type | Value / Meaning | Producer | Consumer |
|---|---|---|---|---|
| `VERSION` | str | Experiment label, format `{algo}.{base\|dh}.{TIME_UNIT}_{1\|2}` (e.g. `mjolnir.base.5s_1`) | hand | `mjolnir/core/research.py`, `gauntlet/experiment_brief.py`, `gauntlet/optimize_aether_thresholds.py`, `gauntlet/validate_pipeline_output.py` |
| `STRATEGY` | str | Always `mjolnir` for this family | hand | `gauntlet/regenerate_filter.py`, `gauntlet/run_*_research.py` |
| `EXCHANGE` | str | `BINANCE_FUTURES_TICK` | hand | informational |
| `TIME_UNIT` | str | Bar frequency (`5s`, `15s`, `30s`, `1m`, `5m`). Replaces legacy `BAR_FREQ`. | hand | `mjolnir/core/research.py`, `mjolnir/gauntlet/build_bars.py`, `daily_pnl_by_symbol.py`, `gauntlet/optimize_thresholds.py`, `gauntlet/generate_daily_pnl.py`, `gauntlet/research_sweep.py` |

### Data inputs

| Key | Type | Value / Meaning | Producer | Consumer |
|---|---|---|---|---|
| `TARDIS_INPUT_DIR` | str path | Raw Tardis Binance shard root (`/mnt/tardis-data-archive/binance`, s3fs). Replaces `TARDIS_ROOT` (legacy alias removed from settings 2026-04-29; loader still accepts it for backward compat). | hand | `mjolnir/gauntlet/build_bars.py:111` |
| `STREAMS` | list[str] | `["trades", "book_ticker", "book_snapshot_25", "derivative_ticker", "liquidations"]` | hand | `mjolnir/gauntlet/build_bars.py` |
| `SYMBOLS` | list[str] | Universe (BINANCE_PERP_*_USDT) | hand | `mjolnir/core/research.py`, `mjolnir/gauntlet/build_bars.py`, `run_mjolnir_pipeline.sh` |

### Bar / feature output paths

| Key | Type | Value / Meaning | Producer | Consumer |
|---|---|---|---|---|
| `BARS_DIR` | str path | Per-experiment bar parquets (NAS, e.g. `/opt/stan_data/tardis-data-archive/mjolnir/pred_mjolnir.base.5s_1/bars`). Output of step 1, input of steps 2+. | step 1 (build_bars) | step 2 (run_research), `run_mjolnir_pipeline.sh:52` |
| `TRAIN_BARS_DIR` | str path | Optional: finer-resolution bars used for training only (e.g. point 15s_1 / 30s_1 at the 5s bar mount). When set, research loads features from this dir instead of `BARS_DIR`. | hand | `mjolnir/core/research.py` |
| `OUTPUT_DIR` | str path | Per-experiment output root for everything except `bars/`. Routes `filter/`, `regime_stack.csv`, `ic_sweep.csv`, `stats/`, `daily_pnl/`, `weights/`, `*_regime_stack.csv` artifacts. **5s_1 currently uses `/mnt/tardis-data-archive/...`** while 15s_1/30s_1 use `/opt/stan_data/...` — see [`gauntlet/README.md`](gauntlet/README.md) exception note. | hand | `mjolnir/trading.py`, `mjolnir/core/research.py`, `mjolnir/gauntlet/run_research.py`, `daily_pnl_by_symbol.py`, `gauntlet/paths.py`, `gauntlet/optimize_thresholds.py`, `gauntlet/filter_regime_stacks.py`, `gauntlet/rolling_predict_returns.py`, `gauntlet/generate_daily_pnl.py`, `gauntlet/experiment_brief.py` |

### Feature engineering

| Key | Type | Value / Meaning | Producer | Consumer |
|---|---|---|---|---|
| `FEATURE_WINDOWS` | list[int] | Rolling window sizes in bars (`[30, 60, 300, 900]`). **Applies to the BASE TF only** — cross-TF (`MULTI_TF_BARS`) feature engines are pinned to a single trivial window (`[1]`) since 2026-04-29 to avoid horizon-collinearity with the base-TF rolling stats. See [feature design note](#feature-design-cross-tf-trivial-windows). | hand | `mjolnir/trading.py`, `mjolnir/core/research.py` |
| `TARGET_HORIZON_BARS` | int | Prediction horizon in bars (1 = one-bar-ahead) | hand | `mjolnir/trading.py`, `mjolnir/core/research.py`, `daily_pnl_by_symbol.py`, `gauntlet/generate_daily_pnl.py`, `gauntlet/optimize_thresholds.py` |
| `MULTI_TF_BARS` | list[str] | Auxiliary TF bar dirs to merge into features (e.g. `["15s","30s"]` for the 5s base). Empty list disables. **Each base TF should list the OTHER TFs in scope** — listing the base TF in its own `MULTI_TF_BARS` re-duplicates base features. **Cross-TF features run with `feature_windows=[1]` (point-in-time only, no rolling expansion)** since 2026-04-29. | hand | `mjolnir/core/research.py:157` |
| `MULTI_TF_BARS_DIRS` | dict[str,str] | Optional override of the auto-derived per-TF bar dir (TF -> path). | hand | `mjolnir/core/research.py:159` |

### Training window / model sweep

| Key | Type | Value / Meaning | Producer | Consumer |
|---|---|---|---|---|
| `TRAIN_START` | str (YYYYMMDD) | First date of training window | hand | `run_mjolnir_pipeline.sh:70` |
| `TRAIN_END` | str (YYYYMMDD) | Last date of training window | hand | `run_mjolnir_pipeline.sh:71` |
| `TRAIN_DAYS` | int | Training window length in days for rolling re-fit | hand | `run_mjolnir_pipeline.sh:72` |
| `TEST_DAYS` | int | Holdout/test window length in days for rolling re-fit | hand | `run_mjolnir_pipeline.sh:73` |
| `WINDOW_SIZE` | int | Rolling window step multiplier (forwarded to `gauntlet/rolling_predict_returns.py`) | hand | rolling stage |
| `SWEEP_MODELS` | list[str] | Model classes to sweep. Currently `["LightGBM", "Ridge"]`. | hand | `mjolnir/gauntlet/run_research.py:144` |
| `REGIME_STACK_PATH` | str path | Filtered regime stack CSV (one row per (regime, model, threshold)). Hand-seeded; auto-overwritten by `run_research.py:146` and step 7 (`filter_regime_stacks.py`). | hand + step 7 | `mjolnir/trading.py`, `mjolnir/core/research.py`, `gauntlet/regenerate_filter.py`, `gauntlet/experiment_brief.py`, `gauntlet/run_*_research.py` |

### Threshold optimization

The legacy `OPTIMIZE_STEP`, `OPTIMIZE_MAX_THRESH`, and `OPTIMIZE_MIN_TPM` config
keys were removed (2026-04-27). Threshold and TPM are now decided arbitrarily
per run via `gauntlet/optimize_thresholds.py` CLI flags (`--step`, `--max-thresh`).

| Key | Type | Value / Meaning | Producer | Consumer |
|---|---|---|---|---|
| `LADDER` | int | Number of laddered exits | hand | `mjolnir/core/research.py:818` |
| `FEE` | float (bps) | Round-trip fee assumption. Currently `0.0` (fills priced at maker). | hand | `mjolnir/trading.py`, `mjolnir/core/research.py`, `daily_pnl_by_symbol.py` |

### PnL / sizing

| Key | Type | Value / Meaning | Producer | Consumer |
|---|---|---|---|---|
| `CAPITAL` | int | Per-symbol notional cap used in PnL accounting | hand | `mjolnir/gauntlet/append_merged_totals.py`, `daily_pnl_by_symbol.py` |
| `LEVERAGE` | int | Account-level leverage used in PnL math (live exec leverage lives under `executor.LEVERAGE`) | hand | `mjolnir/tests/test_bridge_qty.py` |
| `REVERSE` | int (1 or -1) | `1` = use predicted sign; `-1` = invert (mean-reversion variant) | hand | `mjolnir/trading.py:191`, `mjolnir/core/research.py:420`, `append_merged_totals.py`, `daily_pnl_by_symbol.py` |
| `TRAILING_TRIGGER_ATR_MULT` | float | ATR-multiple at which trailing stop activates | hand | live executor (`knull/`) |
| `TRAILING_DISTANCE_ATR_MULT` | float | Trailing stop distance in ATR multiples | hand | live executor (`knull/`) |

### Live execution

| Key | Type | Value / Meaning | Producer | Consumer |
|---|---|---|---|---|
| `SOCKET_PATH` | str path | Bridge -> knull IPC socket (e.g. `/tmp/knull_ltp_mjolnir5s.sock`). Only set on settings actively wired for live trading. | hand | `knull/bridge_runner.py`, `knull/mjolnir_bridge.py`, `knull/run_knull.py`, `knull/run_vibranium.py` |
| `MIN_SIGNAL_COUNT` | int | Minimum number of model-votes required to take a trade | hand | `mjolnir/trading.py:183` |
| `FORCE_CLOSE_ALL` | bool | Live kill-switch (executor-side) | hand | live executor |
| `executor` | object (sub-dict) | Per-experiment executor config block (only populated on settings wired live, currently `5s_1`). | hand | `knull/run_knull.py` |
| `executor.EXEC_VENUE` | str | One of `ltp`, `binance`, `okx` | hand | `knull/run_knull.py:68,91` |
| `executor.LTP_API_KEY` / `executor.LTP_API_SECRET` | str | LTP venue creds | hand | `knull/ltp_executor.py:46` |
| `executor.LEVERAGE` | int | Live executor leverage (separate from the top-level `LEVERAGE` accounting knob) | hand | `knull/*` |
| `executor.ENTRY_TIMEOUT_SEC` | int | Limit-order entry timeout before cancel/retry | hand | `knull/base_executor.py:130-132` |
| `executor.REPRICE_OPEN_SEC` | int | Reprice cadence for entry / ladder rungs (seconds) | hand | `knull/base_executor.py` |
| `executor.REPRICE_CLOSE_SEC` | int | Reprice cadence for exit rungs (seconds) | hand | `knull/base_executor.py` |
| `executor.PASSIVE_SEC` | int | Passive (post-only maker) phase duration in seconds | hand | `knull/base_executor.py` |
| `executor.CROSSING_SEC` | int | Crossing-taker phase duration in seconds; hard-close threshold = `PASSIVE_SEC + CROSSING_SEC` (derived; `MAX_EXIT_SEC` knob is deprecated) | hand | `knull/base_executor.py` |

## Feature design: cross-TF trivial windows

Since 2026-04-29, the cross-TF feature engines built inside
`mjolnir/core/research.py::engineer_features` are constructed with
`feature_windows=[1]` instead of inheriting `FEATURE_WINDOWS` from
`setting.json`. The base-TF engine still receives `FEATURE_WINDOWS` as
configured.

**Why.** When both engines expanded the same `[30, 60, 300, 900]` window list,
the rolling stats encoded the same horizon via two paths. Example: 30s base
`w60` (30 min mean of `trade_imbalance`) and 1m cross-TF `w30` (30 min mean of
`trade_imbalance`) are the same horizon. With multiple TFs and multiple key
signals, this produces dozens of near-collinear feature pairs that inflate
ridge condition number and degrade Ridge OOS IC. LightGBM was less affected
but still saw redundant splits. Pinning cross-TF engines to `[1]` makes
cross-TF features purely point-in-time context (last-value forward-fill from
the higher-TF bar), and reserves rolling expansion to the base TF where
window choice is meaningful.

**Set `MULTI_TF_BARS` to the OTHER TFs.** Each base TF should list cross-TFs
that are *not* itself, otherwise base features are recomputed and merged in as
prefixed duplicates.

## Important

- The rolling worker count is auto-tuned by `mjolnir/gauntlet/run_mjolnir_pipeline.sh` (starts at `WORKERS_INIT=24`, halves on OOM down to `MIN_WORKERS`). Do not pass `--workers` manually.
- Data available from 2024-06-29 for Binance futures via Tardis.
- Bar building takes 20min-1hr depending on date range.
- Feature latency: <10ms per symbol at inference time (small trees + linear regressors).
- ~120 bar warmup required for rolling statistics before predictions are valid.
- All authenticated venue endpoints in `knull/` use Binance papi (not fapi). All orders are LIMIT — market orders are forbidden.
