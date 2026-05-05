# Aether: Cross-TF Pooled Model with Stacked Regime Inference

## Motivation

ORB trains one model per regime condition. This creates a data scarcity problem: strict or niche regime combos (e.g. `vol_breakout AND rsi_oversold AND adx_trend`) may fire fewer than 50 times in a 3-month training window, producing degenerate models that get dropped. This limits regime complexity — you cannot freely stack conditions without risking data starvation.

Aether decouples model training from regime selectivity:

- **Training**: one pooled long model + one pooled short model per base TF, trained on all available data regardless of regime condition
- **Inference**: cross-TF regime stacks as entry filters — arbitrarily strict combos spanning multiple timeframes, applied post-prediction to decide when to trade

This means a regime stack like `1h_vol_breakout AND 4h_macd_bullish AND 1d_above_all_mas` can be tested and deployed with no additional training — it's just a row filter on pre-computed predictions.

---

## Architecture

```
Feature Engineering          Model Training              Backtest / Optimize
───────────────────          ──────────────              ───────────────────
ORB-style cross-TF    ──→   1 long model (per TF)  ──→  for each condition stack C:
(184 features per row)       1 short model (per TF)       - filter rows where C is active
                             trained on ALL rows           - threshold-optimize on holdout
                             (pooled, no regime split)     - score Sharpe/TPM
                                                          keep top-N stacks
```

---

## Key Differences from ORB and Agamotto

| Property | Agamotto | ORB | **Aether** |
|---|---|---|---|
| Feature set | 46 single-TF features | 184 cross-TF features | 184 cross-TF features |
| Models | 1 per regime | 1 per regime | **1 per direction (long/short)** |
| Training data | regime-filtered rows | regime-filtered rows | **all rows, pooled** |
| Regime conditions | single-TF | single-TF | **cross-TF stacks** |
| Rare regime support | blocked (< 50 samples) | blocked (< 50 samples) | **no limit** |
| Regime combinations | ~46 per TF | ~46 per TF | **exponential — tested post-hoc** |

---

## Feature Engineering

Same pipeline as ORB (`orb/research.py` / `gauntlet/run_orb_research.py`):

1. Load klines for all 4 TFs per symbol
2. Engineer 46 AgamottoResearch features per TF
3. Forward-fill align all TFs to BASE_TF granularity
4. Verticalize: one row per (timestamp, symbol), 184 features + regime condition booleans

**Critical addition vs ORB**: keep all regime condition boolean columns in the feature matrix (currently ORB drops them and uses them only for file partitioning). In Aether these become:
- Input features to the model (the model learns when conditions predict returns)
- Inference-time filter keys (used to select rows for cross-TF regime stacks)

Column naming follows ORB convention: `{tf}_{condition}` (e.g. `1h_vol_breakout`, `4h_macd_bullish`, `1d_above_all_mas`).

---

## Model Training

### What changes from ORB

ORB writes one `filter_{regime}.parquet` per regime and trains a separate model on each. Aether writes a **single** `filter_long.parquet` and `filter_short.parquet` — all rows where `return_long > 0` baseline (long) or `return_short > 0` (short), pooled across all regimes and symbols.

### Rolling window

Same rolling window approach as ORB:
- `WINDOW_SIZE = 4` (3 months train, 1 month test)
- `--workers 180`
- Writes `preds_YYYY_MM_long.parquet` and `preds_YYYY_MM_short.parquet`

### Models

Same sweep as ORB: LightGBM, XGBoost, Ridge, HistGBR.

Best model per direction is selected by holdout IC (information coefficient) across all windows.

---

## Cross-TF Condition Stack Generation

After rolling predict, generate all candidate condition stacks to test:

```python
# All single-condition stacks
conditions_1tf = ["1h_vol_breakout", "4h_macd_bullish", "1d_above_all_mas", ...]  # ~46 × 4 TFs = 184

# All 2-way cross-TF combinations (different TFs only — same-TF combos are already in ORB)
conditions_2tf = [(a, b) for a in conditions_1tf for b in conditions_1tf
                  if tf(a) != tf(b) and a < b]  # ~184 × 184 / 2 = ~17k

# Optional 3-way
conditions_3tf = [...]  # ~10M — filter to min occurrence threshold first
```

Filter stacks: only keep combos that fire at least `MIN_OCCURRENCES = 200` times in the full prediction set (across all windows). This prevents testing trivially rare stacks with no statistical power.

---

## Threshold Optimization

For each candidate condition stack:

1. Filter prediction rows where ALL conditions in the stack are active
2. For long: sweep threshold `t` over `[0, max_pred]` — signal = `y_pred > t`
3. For short: sweep threshold over `[-max_pred, 0]` — signal = `y_pred < t`
4. Score each threshold by holdout Sharpe and TPM
5. Record `(stack, direction, model, threshold, sharpe, tpm, pnl_bps, win_rate)`

Output: `optimal_regime_stack.csv` with one row per (stack, direction, model) — same format as existing ORB/Agamotto output, so `filter_regime_stacks.py` works unchanged.

---

## Filter and Deployment

`filter_regime_stacks.py` unchanged — reads `optimal_regime_stack.csv`, applies Sharpe/TPM/top-N filters, writes `filtered_optimal_regime_stack.csv`.

Deployment: same live trading bot structure. At inference time:
1. Check all condition booleans from current bar's features
2. For each regime in filtered stack: check if all conditions in the stack are active
3. If active: use the shared long/short model (not a regime-specific model) to predict return
4. Apply threshold → trade signal

**Weight files**: only 2 model files per TF per direction (e.g. `window_2026_03/long/LightGBM_model.pkl`) instead of hundreds of per-regime files. Weight loading at bot startup is instant.

---

## Proposed Directory Structure

```
gauntlet/pred_aether1h_1/
    setting.json            # BASE_TF=1h, TIMEFRAMES=[15m,1h,4h,1d]
    filter/
        filter_long.parquet     # all long rows (pooled)
        filter_short.parquet    # all short rows (pooled)
    stats/windows/
        preds_2026_03_long.parquet
        preds_2026_03_short.parquet
    weights/
        window_2026_03/
            long/
                LightGBM_model.pkl
                LightGBM_scaler.pkl
            short/
                LightGBM_model.pkl
                LightGBM_scaler.pkl
    optimal_regime_stack.csv
    filtered_optimal_regime_stack.csv
```

---

## Pipeline Script

```
run_aether_pipeline.sh:
  Step 1: research (aether/research.py) — generates filter_long.parquet, filter_short.parquet
  Step 2: rolling_predict — trains pooled long/short model, writes preds
  Step 3: optimize_thresholds_aether.py — generates cross-TF stacks, scores each
  Step 4: filter_regime_stacks.py — standard filter, --weights-month 2026_03
  Step 5: dump window_2026_03 weights
```

---

## Open Design Questions

1. **Train on ALL rows or only rows where any condition is active?**
   All rows is cleaner and avoids training bias. Condition booleans in features let the model learn regime-conditional behavior naturally.

2. **Max stack depth?**
   Start with 2-way cross-TF combos. 3-way is feasible but the combinatorial space is large — gate on `MIN_OCCURRENCES` filter.

3. **Same-TF stacks?**
   Already covered by ORB (`vol_breakout AND rsi_oversold` within same TF). Aether should focus on cross-TF (at least 2 different TFs in the stack) to differentiate.

4. **Separate long/short models or single model with direction as feature?**
   Start with separate long/short models — consistent with existing pipeline and cleaner optimization.

5. **Base TF**
   Start with `BASE_TF=1h` (same as primary agamotto/ORB experiments). Extend to 15m and 4h once 1h is validated.
