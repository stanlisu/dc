# Orb

Cross-timeframe feature alignment system. Merges features from all 4 timeframes (15m, 1h, 4h, 1d) into a single wide feature matrix, producing 184 ML features per row vs Agamotto's 46.

## What It Does

Agamotto trains single-TF models. Orb enables patterns like "1h RSI oversold WHILE 1d trend is bullish" by forward-fill aligning higher TFs onto the target TF index using causal `merge_asof` (no lookahead), then verticalizing with TF-prefixed columns (`15m_rsi`, `1h_macd`, `4h_adx`, `1d_cci`). This produces structurally different signals for genuine portfolio diversification.

## Architecture

```
OrbTrading (live trading)
  +-- OrbResearch (research/backtesting)
      +-- AgamottoResearch (single-TF base)
      +-- 4x AgamottoResearch instances (one per TF: 15m, 1h, 4h, 1d)
```

## Key Files

| File | Description |
|------|-------------|
| `research.py` | `OrbResearch(AgamottoResearch)`: composes 4 AgamottoResearch instances, `_align_timeframes()` with causal merge_asof, `_apply_filter_mask()` with TF-prefixed column remapping |
| `trading.py` | `OrbTrading(OrbResearch)`: live inference, fetches klines for all 4 TFs concurrently (WS buffer + REST fallback), regime stack loading, position sizing |
| `tests/test_orb_research.py` | 20+ tests: alignment, verticalization, filter masking, close_timestamp causality, cross-TF returns |
| `tests/test_ws_buffer_orb.py` | WS buffer multi-TF readiness, REST fallback, backward compatibility |

## Key Design Decisions

- **Causality protection**: Uses `close_timestamp` and backward merge_asof to ensure no lookahead -- features at time T only use bars that have closed
- **Per-slot returns**: Cross-TF returns differ per base-TF slot within a target-TF bar (different entry prices)
- **Feature categorization**: Derived features (RSI, MACD) are TF-prefixed for ML; raw/MA columns from TARGET_TF are unprefixed for filter logic
- **Filter routing**: Unprefixed filter names use TARGET_TF columns; TF-prefixed filters (e.g., `"1h_baseline"`) remap that TF's columns; compound filters (e.g., `"1h_baseline_and_15m_high_vol"`) chain multiple TF filters

## Usage

```bash
# Research pipeline
PYTHONPATH="agamotto_pkg/src:." python gauntlet/run_orb_research.py \
  -c gauntlet/pred_orb1h_1/setting.json

# Training (reuses gauntlet scripts)
PYTHONPATH="agamotto_pkg/src:." python gauntlet/rolling_predict_returns.py \
  --setting-dir gauntlet/pred_orb1h_1

# Tests
PYTHONPATH="agamotto_pkg/src:." pytest orb/tests/ -v
```

```python
# Live trading (called by trade_execution.py)
from orb import OrbTrading
orb = OrbTrading(config, home_root, period="window_2026_03")
decisions = orb.make_decision()
# Returns: {symbol: [price, qty]} -- same interface as AgamottoTrading
```

## Configuration

Orb-specific `setting.json` keys (in addition to standard Agamotto config):

| Key | Example | Description |
|-----|---------|-------------|
| `STRATEGY` | `"orb"` | Must be set to `"orb"` |
| `TARGET_TF` | `"1h"` | Timeframe for returns (prediction target) |
| `TIMEFRAMES` | `["15m","1h","4h","1d"]` | All TFs to merge |
| `BASE_TF` | `"1h"` | Must equal TARGET_TF (prevents data leakage) |
| `WINDOW_SIZE` | `4` | Rolling window size (not 12 like base Agamotto) |
| `TRADING_MODE` | `"both"` | `"long"`, `"short"`, or `"both"` |
| `RETAIN_THRESHOLD` | (float) | Early close signal threshold (asymmetric) |
| `LONG_PRED_THRESHOLD` | `0.0` | Threshold for long predictions |
| `SHORT_PRED_THRESHOLD` | `0.0` | Threshold for short predictions |
