# Scepter Algorithm Design

**Date:** 2026-03-31
**Status:** Approved

---

## Overview

Scepter is a new algorithm that models the hierarchical influence of BTC and ETH on altcoin price behavior. Current algorithms (Agamotto, ORB) treat all symbols from the same feature pool — each symbol's model sees only its own indicators. Scepter adds cross-symbol features (lagged returns, rolling correlation, cointegration spread, relative strength, anchor volatility regime) and crosses altcoin own-state regimes with BTC-state conditions to expose structure that is invisible to siloed models.

---

## Architecture

`ScepterResearch` inherits from `OrbResearch`. The full multi-TF pipeline (15m/1h/4h/1d alignment, `merge_asof`, causal constraints) is inherited unchanged. The only new method is `add_anchor_features(df, tf)`, which runs after ORB's standard feature engineering and before `verticalize()`. All downstream pipeline steps (`rolling_predict_returns.py`, `optimize_thresholds.py`, `filter_regime_stacks.py`) are unchanged.

BTC and ETH are **anchor symbols** — always loaded, never verticalized as prediction targets. Each altcoin row in `vertical_features.csv` gets a new block of cross-symbol columns prefixed `btc_` and `eth_`.

```
scepter/
  __init__.py
  research.py              # ScepterResearch(OrbResearch)

gauntlet/
  run_scepter_research.py  # entry point (mirrors run_orb_research.py)
  pred_scepter15m_1/setting.json
  pred_scepter1h_1/setting.json
  pred_scepter4h_1/setting.json
  pred_scepter1d_1/setting.json
```

---

## Cross-Symbol Features

For each altcoin row at timestamp `t` and timeframe `tf`, `add_anchor_features()` joins the following columns from BTC and ETH (prefixed `btc_` / `eth_`). All features are computed causally — only past data at each bar.

### Lagged Returns (lead-lag signal)
- `btc_ret_lag1`, `btc_ret_lag2`, `btc_ret_lag3` — BTC bar return at t-1, t-2, t-3
- Same for ETH

### Rolling Correlation
- `btc_corr_14`, `btc_corr_28` — Pearson correlation of ALT vs BTC returns over 14 and 28 bars
- Same for ETH

### Cointegration Spread (mean-reversion signal)
- `btc_spread` — `ALT_close − β·BTC_close` where β is the rolling OLS coefficient (28-bar window). Positive = ALT overpriced vs BTC.
- Same for ETH

### Relative Strength
- `btc_rel_strength` — `ALT_return_14 − BTC_return_14` (14-bar cumulative return difference)
- Same for ETH

### Anchor Volatility Regime
- `btc_atr_ratio` — BTC ATR / 28-bar MA of BTC ATR. >1 = elevated vol, <1 = compressed.
- Same for ETH

Window sizes (14, 28) are configurable via `ANCHOR_WINDOWS` in `setting.json`.

---

## Regime Definitions

Scepter regimes are **crossed**: altcoin own-state condition × BTC-state condition. Each (altcoin, timestamp) pair belongs to exactly one combined regime.

### Own-State Conditions (altcoin's own indicators — same as Agamotto)
- `above_all_mas` — close > mvg1 > mvg2 > mvg3
- `high_volume` — vol_ratio > threshold
- `adx_trend` — ADX > 25
- `vol_breakout` — ATR above rolling mean

### BTC-State Conditions (from anchor features)
- `btc_trending_up` — BTC close > BTC mvg1 > BTC mvg2
- `btc_trending_down` — BTC close < BTC mvg1 < BTC mvg2
- `btc_high_vol` — `btc_atr_ratio` > 1.2
- `btc_low_vol` — `btc_atr_ratio` < 0.8

### Crossed Regime Examples
- `high_volume_and_above_all_mas_and_btc_trending_up_long`
- `high_volume_and_above_all_mas_and_btc_trending_down_short`
- `adx_trend_and_btc_high_vol_long`

BTC-state conditions are defined in `setting.json` as `ANCHOR_REGIMES` — configurable without code changes. `filter_signals()` is extended to include `btc_state` as an additional filter column. `optimize_thresholds.py` treats each crossed regime independently.

---

## Pipeline Integration

### `setting.json` additions
```json
{
  "STRATEGY": "scepter",
  "ANCHOR_SYMBOLS": ["BINANCE_PERP_BTC_USDT", "BINANCE_PERP_ETH_USDT"],
  "ANCHOR_WINDOWS": [14, 28],
  "ANCHOR_REGIMES": {
    "btc_trending_up":   {"col": "btc_close_vs_ma", "op": ">", "val": 0},
    "btc_trending_down": {"col": "btc_close_vs_ma", "op": "<", "val": 0},
    "btc_high_vol":      {"col": "btc_atr_ratio",   "op": ">", "val": 1.2},
    "btc_low_vol":       {"col": "btc_atr_ratio",   "op": "<", "val": 0.8}
  },
  "WINDOW_SIZE": 6
}
```

### Full Pipeline (downstream unchanged)
```
run_scepter_research.py       → vertical_features.csv + filter/*.parquet
rolling_predict_returns.py    → stats/windows/preds_*.parquet
optimize_thresholds.py        → optimal_regime_stack.csv
filter_regime_stacks.py       → filtered_optimal_regime_stack.csv
```

### Data Requirements
`ANCHOR_SYMBOLS` must be present in the raw kline data. BTC and ETH are already downloaded for ORB experiments — no new download step needed. `ScepterResearch.__init__()` raises `KeyError` immediately if any anchor symbol is missing. No fallback.

---

## Key Design Constraints

- **No fallbacks** — missing anchor data raises immediately; never silently loads wrong data
- **Causal only** — all cross-symbol features use only data available at prediction time
- **No downstream changes** — `rolling_predict_returns.py` and everything after are untouched
- **BTC/ETH never predicted** — anchor symbols are feature inputs only, not in `vertical_features.csv` as targets
