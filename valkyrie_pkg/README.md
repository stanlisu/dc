# Valkyrie

Deribit weekly options classification pipeline. Predicts DOWN/FLAT/UP spot moves from weekday entry to Friday 08:00 UTC settlement, then evaluates short option strategies.

## What It Does

Builds daily features from Tardis Deribit options chain data (IV OHLC, skew, term structure, Greeks, OI), constructs weekly samples (Mon-Thu entries to Friday settlement), trains 3-class classifiers (LightGBM, XGBoost, LogisticRegression), and optimizes probability thresholds. Strategy is short options only -- the classifier identifies high-confidence FLAT weeks and structurally overpriced legs.

Assets: BTC (inverse), ETH (inverse), SOL_USDC (linear).

## Architecture

```
Tardis CSV.gz -> ValkyrieLoader -> DailyFeatureBuilder -> ValkyrieResearch
                                                           |
                                        verticalize (Mon-Thu -> Friday)
                                           |            |           |
                                     short_call    short_put   short_straddle
                                           |
                                   3-class target (DOWN/FLAT/UP)
                                           |
                                   regime filtering (18+ regimes)
                                           |
                              rolling_predict_classify (11mo/1mo windows)
                                           |
                              optimize_thresholds_classify
```

## Key Files

| File | Description |
|------|-------------|
| `core/loader.py` | `ValkyrieLoader`: parse Deribit options chain CSV.gz from Tardis |
| `core/features.py` | `DailyFeatureBuilder`: ATM IV OHLC, 25d/10d skew, term structure ratio, OI, Greeks, rolling features (50+ per day) |
| `core/research.py` | `ValkyrieResearch`: daily -> weekly samples, option P&L model (short call/put/straddle), stop-loss, regime filtering (~736 lines) |
| `core/reentry.py` | `DailyReentrySimulation`: signal-flip re-entry within the week (close at flip day, re-enter) |
| `core/utils.py` | ATM selection, delta-strike lookup, IV rank (252d), expiry bucketing |
| `gauntlet/run_research.py` | CLI: build features, verticalize, filter |
| `gauntlet/rolling_predict_classify.py` | Rolling 11mo/1mo classification windows (~568 lines) |
| `gauntlet/optimize_thresholds_classify.py` | Sweep probability thresholds for DOWN class (~384 lines) |
| `tests/` | 56 tests across 5 files |

## Features (50+ per Day)

| Category | Features |
|----------|----------|
| **Spot** | Open, close, settlement prices |
| **ATM IV** | Call/put/avg IV OHLC |
| **Skew** | 25d risk reversal, 10d RR, 25d butterfly |
| **Term Structure** | Weekly IV / Monthly IV ratio (contango vs backwardation) |
| **Greeks** | Gamma, vega, theta, delta_call (at ATM) |
| **OI** | Total, call, put, put/call ratio |
| **Rolling** | IV rank (252d), spot returns (5/10/20/60d), HV (10/20d), RSI-7, IV vs HV ratio |
| **Temporal** | Day of week, days to Friday, week number, days to expiry |

## Regime Filters (18+)

baseline, high_iv_rank, low_iv_rank, backwardation, contango, positive_skew, negative_skew, rsi_oversold, rsi_overbought, above_ma20, below_ma20, momentum_up, momentum_down, iv_premium, iv_discount, vol_expansion, vol_compression, oi_expansion, high_hv, low_hv

## Option P&L Model

- **Settlement**: Inverse (BTC/ETH) = `intrinsic / spot_friday`; Linear (SOL_USDC) = `intrinsic`
- **Entry**: ATM call/put premium at entry day's close
- **Stop-loss**: If daily high > `entry_premium * (1 + STOPLOSS_PCT)`, closed at -STOPLOSS_PCT
- **3-class target**: DOWN (spot <= entry * (1 - threshold)), FLAT (middle), UP (spot >= entry * (1 + threshold))

## Usage

```bash
# 1. Build features + filter parquets
PYTHONPATH="agamotto_pkg/src:." python valkyrie/gauntlet/run_research.py \
  --setting valkyrie/gauntlet/pred_valkyrie1w_1/ \
  --start 20230101 --end 20251020

# 2. Train rolling classifiers
PYTHONPATH="agamotto_pkg/src:." python valkyrie/gauntlet/rolling_predict_classify.py \
  --setting valkyrie/gauntlet/pred_valkyrie1w_1/ --workers 4

# 3. Optimize thresholds
PYTHONPATH="agamotto_pkg/src:." python valkyrie/gauntlet/optimize_thresholds_classify.py \
  --setting valkyrie/gauntlet/pred_valkyrie1w_1/

# Full pipeline (chains to Vomir afterward)
bash valkyrie/run_valkyrie_then_vomir.sh

# Tests
pytest valkyrie/tests/ -v
```

## Configuration

`pred_valkyrie1w_1/setting.json`:

| Key | Description |
|-----|-------------|
| `ASSETS` | `["BTC", "ETH", "SOL_USDC"]` |
| `TARDIS_ROOT` | Path to Tardis Deribit data (`/mnt/tardis/deribit`) |
| `TARGET_THRESHOLD` | Spot change for UP/DOWN classification (0.10 = 10%) |
| `STOPLOSS_PCT` | Stop-loss on option premium (2.0 = 200 bps) |
| `SETTLEMENT_HOUR_UTC` | Deribit Friday settlement hour (8) |
| `MAX_WEEKLY_DAYS` | Days-to-expiry for "weekly" bucket (7) |
| `TRAIN_MONTHS` | Rolling window train length (11) |
| `TEST_MONTHS` | Rolling window test length (1) |
| `DAILY_REENTRY` | Enable intra-week signal-flip re-entry |

## Status

Research-only. No live Deribit execution exists.
