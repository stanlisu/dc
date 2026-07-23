# Stormbreaker

Tick-native cross-timeframe ML research system. Extends Mjolnir by applying ORB-style context-TF regime stacking to microstructure signals — each signal bar is filtered using conditions evaluated on coarser bar frequencies (15s, 1m, 5m, 15m), enabling multi-resolution regime detection from a single tick-data source.

## What It Does

Reads the same 5 Tardis streams as Mjolnir, builds bars at both the signal frequency (5s/15s/30s/1m) and context frequencies (via `CONTEXT_BAR_FREQS`), engineers 100+ microstructure features at each frequency, then filters using 800 tick-native cross-TF regime conditions. Outputs `filter/*.parquet` files compatible with all downstream gauntlet scripts.

## Architecture

```
Tick Data (5 streams) → StreamAligner → signal bars + context bars
                                          ↓
MjolnirFeatures(prefix=tf) → {tf}_{col} columns merged into signal frame
                                          ↓
StormBreakerResearch → _apply_filter_mask → apply_filter(tf_view, base_filter)
                                          ↓
filter/*.parquet → rolling_predict → daily_pnl → optimize → filter_regime_stacks
```

## Key Differences from Mjolnir

| | Mjolnir | Stormbreaker |
|---|---|---|
| Regime conditions | 13 microstructure filters | 800 tick-native cross-TF regimes |
| Context bars | Optional (MULTI_TF_BARS) | Required (CONTEXT_BAR_FREQS) |
| Filter taxonomy | Position-agnostic | LONG-only / SHORT-only / BOTH |
| Cross-TF pairing | Single-TF only | Higher-TF context + lower-TF signal |

## Key Files

| File | Description |
|------|-------------|
| `core/filters.py` | 12 tick-native filter conditions, `apply_filter()`, `allowed_positions()` |
| `core/research.py` | `StormBreakerResearch`: resolves CONTEXT_BAR_FREQS, routes `{tf}_{filter}` names |
| `gauntlet/generate_stormbreaker_regimes.py` | 800 regimes (400L/400S) across 4 sections |
| `gauntlet/run_research.py` | Step 2: generate regime stack + run StormBreakerResearch |
| `gauntlet/run_stormbreaker_pipeline.sh` | Steps 1–6 for a single experiment |
| `gauntlet/run_stormbreaker_queue.sh` | Full queue: all TFs in context-dependency order |

## Tick-Native Filters (12)

| Direction | Filter | Condition |
|-----------|--------|-----------|
| LONG | `buy_flow` | `trade_imbalance > 0.2` |
| LONG | `bid_heavy` | `depth_imbalance_L5 > 0.2` |
| LONG | `ofi_positive` | `ofi_agg > 0` |
| LONG | `short_liq_spike` | `liq_burst_ratio > 2× avg` AND `liq_directional_imbalance > 0` |
| SHORT | `sell_flow` | `trade_imbalance < -0.2` |
| SHORT | `ask_heavy` | `depth_imbalance_L5 < -0.2` |
| SHORT | `ofi_negative` | `ofi_agg < 0` |
| SHORT | `long_liq_spike` | `liq_burst_ratio > 2× avg` AND `liq_directional_imbalance < 0` |
| BOTH | `liq_spike` | `liq_burst_ratio > 2× avg` |
| BOTH | `high_spread` | `relative_spread > rolling_median × 1.5` |
| BOTH | `low_spread` | `relative_spread < rolling_median × 0.7` |

## Regime Generator (800 regimes)

Four sections, all regimes named `{tf}_{filter}` or `{tf}_{filter}_and_{tf}_{filter}`:

| Section | Pattern | Count |
|---------|---------|-------|
| 1. Single-TF | `{tf}_{filter}` at 5s/15s/30s/1m/5m/15m | 96 |
| 2. BOTH context + directional signal | `{ctx_tf}_{BOTH}_and_{sig_tf}_{LONG/SHORT}` | 256 |
| 3. Same-side directional pairs | `{ctx_tf}_{LONG}_and_{sig_tf}_{LONG}` (and SHORT×SHORT) | 192 |
| 4. Directional context + BOTH signal | `{ctx_tf}_{LONG/SHORT}_and_{sig_tf}_{BOTH}` | 256 |

TF pairs (context → signal): 15s→5s, 1m→5s, 1m→15s, 5m→15s, 5m→30s, 15m→30s, 5m→1m, 15m→1m

## Experiment Directories

| Experiment | BAR_FREQ | CONTEXT_BAR_FREQS | TRAIN/TEST |
|------------|----------|-------------------|------------|
| pred_stormbreaker.base.5s_1 | 5s | [15s, 1m] | 3d / 1d |
| pred_stormbreaker.base.5s_2 | 5s | [15s, 1m] | 2d / 1d |
| pred_stormbreaker.base.15s_1 | 15s | [1m, 5m] | 5d / 2d |
| pred_stormbreaker.base.15s_2 | 15s | [1m, 5m] | 3d / 1d |
| pred_stormbreaker.base.30s_1 | 30s | [5m, 15m] | 5d / 2d |
| pred_stormbreaker.base.30s_2 | 30s | [5m, 15m] | 3d / 1d |
| pred_stormbreaker.base.1m_1 | 1m | [5m, 15m] | 7d / 3d |
| pred_stormbreaker.base.1m_2 | 1m | [5m, 15m] | 5d / 2d |
| pred_stormbreaker.base.5m_1 | 5m | [] | bar-provider |
| pred_stormbreaker.base.15m_1 | 15m | [] | bar-provider |

Context bar directories are auto-resolved to `pred_stormbreaker.base.{tf}_1/bars/` regardless of which variant is running (always `_1` for context).

## Usage

```bash
# Full queue (all TFs, both variants — runs for hours)
bash stormbreaker/gauntlet/run_stormbreaker_queue.sh

# Single experiment (Steps 1–6)
bash stormbreaker/gauntlet/run_stormbreaker_pipeline.sh 5s 1

# Individual steps
PYTHONPATH="agamotto_pkg/src:."

# Step 1: build bars
python mjolnir/gauntlet/build_bars.py \
  -c stormbreaker/gauntlet/pred_stormbreaker.base.5s_1 \
  --start 20251001 --end 20260324 --workers 8

# Step 2: research (generates regimes + filter parquets)
python stormbreaker/gauntlet/run_research.py \
  -c stormbreaker/gauntlet/pred_stormbreaker.base.5s_1

# Steps 3–6: standard gauntlet scripts
python gauntlet/rolling_predict_returns.py -c /mnt/tardis-data-archive/stormbreaker/pred_stormbreaker.base.5s_1 --train-days 3 --test-days 1 --workers 60
python gauntlet/generate_daily_pnl.py      -c /mnt/tardis-data-archive/stormbreaker/pred_stormbreaker.base.5s_1
python gauntlet/optimize_thresholds.py     -c /mnt/tardis-data-archive/stormbreaker/pred_stormbreaker.base.5s_1 --step 0.0001 --max-thresh 0.001
python gauntlet/filter_regime_stacks.py    --project /mnt/tardis-data-archive/stormbreaker --min-sharpe 1.0 --min-tpm 50 --top-n 60
```

## Implementation Notes

**`_long`/`_short` stripping fix**: The parent `MjolnirResearch._apply_filter_mask` strips `_long`/`_short` from filter names before dispatch, which would corrupt `1m_long_liq_spike` → `1m_liq_spike`. `StormBreakerResearch` overrides `_apply_filter_mask` to intercept tick-native atoms before this strip, then recurses compound `_and_`/`_or_` names through `self` so the fix applies at every nesting level.

**Context column lookup**: Context TF features are prefixed by `MjolnirFeatures(prefix=tf)` during `engineer_features()`. `_get_tf_view(df, "15s")` strips the `15s_` prefix to produce a bare-column sub-DataFrame that `apply_filter()` can read without modification.
