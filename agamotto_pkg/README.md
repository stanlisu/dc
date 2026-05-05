# Agamotto

ML engine for the Marvel trading ecosystem. Handles feature engineering, model training (LightGBM/XGBoost/Ridge/ElasticNet), and live inference via regime-stacked predictions.

## What It Does

Agamotto has two modes:

- **`AgamottoResearch`** -- Offline backtesting/training: ingests raw OHLCV data, engineers 50+ technical features (TA-Lib indicators), trains rolling-window models, and generates verticalized feature sets for regime-based model optimization.
- **`AgamottoTrading`** -- Live inference: loads a regime stack (portfolio of specialized models), fetches fresh kline data from Binance Futures REST API or WebSocket buffers, runs predictions through every active regime per symbol, and aggregates long/short votes into a final position decision.

Both modes share the same `engineer_features()` method to guarantee feature parity between training and production.

## Key Files

| File | Description |
|------|-------------|
| `src/agamotto/research.py` | `AgamottoResearch`: data ingestion, 50+ technical features (TA-Lib), verticalization, signal filtering, `compute_dual_horizon_target()` |
| `src/agamotto/trading.py` | `AgamottoTrading`: live kline fetch (REST + WS buffer), regime stack loading, inference, vote aggregation, position sizing |
| `src/agamotto/utils.py` | Symbol parsing (`_symbol_to_native`), timeframe conversion (`_timeframe_to_seconds`), boundary alignment |
| `src/agamotto/lib_binance.py` | Binance REST API: `fetch_futures_klines()`, `klines_to_dataframe()`, exchangeInfo with 60s TTL cache + backoff |
| `src/agamotto/telegram.py` | `send_telegram_message()`, `format_decisions_for_telegram()` |
| `tests/` | Unit tests |

## Feature Engineering (50+ Features)

| Category | Features |
|----------|----------|
| **Price/Returns** | Open-close diff, high/low %, returns (lagged 1/2/3), laddered returns (long/short with fee), range, return_dip, return_rip |
| **Moving Averages** | 3 configurable periods (default 7, 25, 99) |
| **Momentum** | RSI (14/7/28), MACD/MACDHIST, Stochastic (fast/slow), CCI, ADX, DX, PLUS_DI, MINUS_DI, MOM, ROC, Williams %R, CMO, TRIX, UltimateOsc, StochRSI |
| **Volume** | OBV, AD, MFI, BOP, volume ratio, buy pressure, trade intensity |
| **Volatility** | ATR, NATR, Bollinger Bands, SAR, Parkinson Vol |
| **Statistical** | Std dev, skewness, kurtosis, autocorrelation (14-period default) |
| **Ladder** | Long/short layers based on next-bar price movement, fee-adjusted |

## Usage

```bash
# Research (via gauntlet pipeline)
PYTHONPATH="agamotto_pkg/src:." python gauntlet/run_agamotto_research.py \
  -c gauntlet/pred_agamotto1h_1/setting.json

# Live inference (called by trade_execution.py in ltp/optimus/sumo)
from agamotto.trading import AgamottoTrading
trader = AgamottoTrading(config=..., home_root=...)
decisions = trader.make_decision()
# Returns: {'BTCUSDT': [price, target_qty], ...}

# Tests
PYTHONPATH="agamotto_pkg/src:." pytest agamotto_pkg/tests/ -v
```

## Configuration

Key `setting.json` fields consumed by Agamotto:

| Key | Description |
|-----|-------------|
| `SYMBOLS` | List of traded symbols (e.g., `BINANCE_PERP_BTC_USDT`) |
| `TIME_UNIT` | Bar timeframe (`15m`, `1h`, `4h`, `1d`) |
| `FEATURE_TF` | Optional fine-grained feature fetch TF (defaults to TIME_UNIT) |
| `REGIME_STACK_PATH` | Path to CSV regime stack (required, no fallbacks) |
| `WEIGHTS_PATH` | Directory containing `.pkl` model files |
| `WEIGHTS_PERIOD` | Period folder (e.g., `window_2026_03`) |
| `SWEEP_MODELS` | Models to train: `LightGBM`, `XGBoost`, `Ridge`, `ElasticNet` |
| `CAPITAL` | Position sizing base (qty = CAPITAL / price, default 100) |
| `FEE` | Fee rate in bps (e.g., 2.0 = 0.02%) |
| `TRADING_MODE` | `"both"`, `"long_only"`, `"short_only"` |
| `LONG_PRED_THRESHOLD` | Min prediction to open long (default 0.0) |
| `SHORT_PRED_THRESHOLD` | Max prediction to open short (default 0.0) |
| `MA_PERIODS` | Moving average periods (default [7, 25, 99]) |
| `STATS_WINDOW` | Rolling statistics window (default 14) |

## Decision Flow (Live Trading)

1. Load regime stack from `REGIME_STACK_PATH` (CSV, required)
2. Calculate position sizes: `CAPITAL / yesterday_close`, rounded to exchange step_size
3. Fetch 700 recent klines via REST or WS buffer (concurrently per symbol)
4. Run `engineer_features()` and `verticalize()`
5. For each regime: `filter_signals()` -> `predict()` -> aggregate votes
6. Long if prediction > threshold + filters pass; short if prediction < threshold
7. Final qty = `(long_count - short_count) * base_size`
8. Return `{'symbol': [price, qty], ...}`

## Extended By

- **`OrbResearch`** / **`OrbTrading`** (`orb/`) -- Cross-TF feature alignment (184 features)
- **`MjolnirResearch`** (`mjolnir/`) -- Tick-data microstructure features

## Approved Models

Only models with parallel training (`n_jobs`) and early stopping: LightGBM, XGBoost, HistGBR, Ridge, ElasticNet. Never add MLPRegressor, RandomForest, or PyTorch models to `SWEEP_MODELS`.
