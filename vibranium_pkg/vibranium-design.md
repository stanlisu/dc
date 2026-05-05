# Vibranium Design — Stationary Portfolio Mean-Reversion

## Core Idea

Find portfolio weights **w** over liquid crypto perpetuals such that the portfolio's cumulative log-value decomposes into:

```
V(t) = Σ wᵢ · log(Pᵢ(t)) = intercept + α·t + S(t)
```

where **α > 0** is a positive drift and **S(t)** is a stationary, mean-reverting process (Ornstein-Uhlenbeck). We trade the S(t) component:

- **Long** when S(t) is below its mean (spread is cheap, expect reversion upward)
- **Short** when S(t) is above its mean (spread is expensive, expect reversion downward)
- **Flat** when S(t) is near its mean (no edge)

The positive drift α means holding the portfolio has a natural tailwind. The mean-reversion of S(t) provides the trading signal. Together: trend + mean-reversion on the deviation from trend.

---

## Mathematical Model

### 1. Portfolio Spread

Given N symbols with log-prices log(P₁), ..., log(Pₙ) and weights w₁, ..., wₙ (normalized so Σ|wᵢ| = 1):

```
V(t) = Σ wᵢ · log(Pᵢ(t))
```

Weights can be positive (long) or negative (short). The portfolio is a linear combination of log-prices — geometrically, a ratio of power-weighted prices.

### 2. Trend + Stationary Decomposition

Fit a linear trend on the lookback window via OLS:

```
V(t) = intercept + α·t + S(t)
```

- **intercept**: level of V at t=0 of the lookback window
- **α**: drift per bar (positive = portfolio appreciates over time)
- **S(t)**: residual after removing the trend — this must be stationary

The intercept is critical for out-of-sample use. When we apply the same trend line to the forward (trading) window, we need both the slope (α) and the intercept to correctly position the detrended spread. Without the intercept, S(t) drifts by hundreds of sigma out-of-sample (this was a bug we fixed).

### 3. Ornstein-Uhlenbeck Model for S(t)

The detrended spread S(t) is modeled as an OU process:

```
dS = κ(μ - S) dt + σ dW
```

- **κ** (kappa): mean-reversion speed. Higher = faster snap-back to mean.
- **μ** (mu): long-run mean of S(t). Should be near zero after detrending.
- **σ** (sigma): volatility of the innovations driving S(t).
- **half-life** = ln(2) / κ: bars until half the deviation from μ is absorbed.

**Estimation** via OLS on the discrete-time equivalent:

```
ΔS(t) = a + b·S(t) + ε(t)

κ = -b       (per bar; b must be negative for mean-reversion)
μ = -a / b
σ = std(ε)
```

This is equivalent to the Dickey-Fuller regression. If b ≥ 0, the spread is not mean-reverting and we reject the portfolio.

### 4. Z-Score

The equilibrium standard deviation of an OU process is:

```
σ_eq = σ / √(2κ)
```

The z-score at each bar:

```
z(t) = (S(t) - μ) / σ_eq
```

Under the OU model, z(t) is approximately standard normal in steady state. Values far from zero indicate profitable mean-reversion opportunities.

---

## Pipeline

### Step 1: Pre-screen Symbols

From the full liquid universe (20-30 symbols), select the top `n_assets` (default 8) by average pairwise cointegration score. For each pair (A, B), run the Engle-Granger cointegration test and compute -log(p-value). Each symbol's score = average of its pairwise scores. Higher score = more cointegrated with the universe = better candidate for a stationary portfolio.

**Why pre-screen?** Johansen cointegration is O(n³) in the number of assets. Reducing from 30 to 8 symbols makes it tractable. We keep symbols that "play well together" — i.e., have strong long-run statistical relationships.

### Step 2: Johansen Eigenvectors (Seeds)

Run the Johansen cointegration test on the selected symbols' log-prices. This solves a generalized eigenvalue problem on the error-correction matrix, producing eigenvectors that define stationary linear combinations of the input series.

```
coint_johansen(log_prices, det_order=0, k_ar_diff=1)
→ eigenvectors e₁, e₂, ..., eₖ   (sorted by eigenvalue, descending)
```

Each eigenvector eᵢ is a candidate weight vector. The first eigenvector corresponds to the "most stationary" linear combination. We take the top 3 as seed candidates.

**Key insight**: Johansen maximizes stationarity (eigenvalue of the error-correction matrix), NOT profitability or mean-reversion speed. The most stationary combination may have negative drift or very slow reversion. That's why we need Step 3.

### Step 3: Optimization (Optional)

Refine each Johansen seed via differential evolution to maximize a better objective:

```
maximize   κ / σ        (mean-reversion speed / noise)
subject to α > 0        (positive drift)
           |wᵢ| ≤ 0.25  (concentration limit per symbol)
           Σ|wᵢ| = 1    (normalization)
```

The objective κ/σ is the signal-to-noise ratio of the mean-reversion. Higher κ means faster snap-back; lower σ means less noise. Together, κ/σ measures how "tradeable" the spread is.

**Solver**: `scipy.optimize.differential_evolution` — a global optimizer that handles non-convex, non-smooth objectives. Works well in 6-12 dimensions. The Johansen eigenvector is passed as `x0` (initial guess) to warm-start the search.

**Regularization**: L2 penalty on weights (λ · Σwᵢ²) prevents the optimizer from finding degenerate solutions with extreme weights on low-liquidity assets.

**Note**: This step is computationally expensive (~minutes per refit on 8640 bars). Can be skipped with `optimize=False` for fast backtesting.

### Step 4: Validate

Reject portfolios where:
- κ ≤ 0 (spread is not mean-reverting)
- half-life > `max_halflife` (reversion too slow to be tradeable; default 500 bars = ~5 days)

Among remaining candidates, select by score = κ/σ (with 1.5x bonus if α > 0).

---

## Signal Generation (Per Bar)

Given a fitted portfolio (weights, OU params, trend intercept + slope):

```
1. Compute raw spread:        V(t) = Σ wᵢ · log(Pᵢ(t))
2. Detrend using saved fit:   S(t) = V(t) - (intercept + α · t)
3. Z-score:                   z(t) = (S(t) - μ) / σ_eq
4. State machine:
     if flat  and z < -entry_z  → long  (+1)    [spread below trend]
     if flat  and z > +entry_z  → short (-1)    [spread above trend]
     if in position and |z| < exit_z → flat (0) [reverted to mean]
```

**Position semantics**: signal=+1 means "buy the portfolio" — long symbols with wᵢ > 0, short symbols with wᵢ < 0. signal=-1 is the reverse.

---

## Walk-Forward Backtest

```
for each rebalance window (default 672 bars = 7 days):
    1. Fit portfolio on lookback window (default 8640 bars = 90 days)
       → weights, OU params, trend intercept + slope
    2. Generate signals on the next 672 bars using the fitted model
    3. Compute PnL per bar:
       pnl(t) = prev_signal · Σ(wᵢ · log_return_i(t))
       Subtract transaction cost (0.05% per leg) on signal changes
    4. Advance to next rebalance window
```

This is strictly walk-forward: the model only sees past data. No lookahead.

---

## PnL Accounting

Each bar, the portfolio return is:

```
R_portfolio(t) = Σ wᵢ · [log(Pᵢ(t)) - log(Pᵢ(t-1))]
```

PnL = previous bar's signal × R_portfolio. Transaction cost (0.05% maker fee × number of legs) is charged whenever the signal changes (entry or exit).

---

## Hyperparameters

| Parameter | Default | Role |
|---|---|---|
| `lookback_bars` | 8640 (90d) | Window for fitting weights + OU params |
| `rebalance_bars` | 672 (7d) | How often to refit the portfolio |
| `n_assets` | 8 | Number of symbols in the portfolio |
| `max_weight` | 0.25 | Max absolute weight per symbol |
| `entry_z` | 2.0 | Enter when \|z\| exceeds this |
| `exit_z` | 0.5 | Exit when \|z\| falls below this |
| `max_halflife` | 500 | Reject spreads slower than this (bars) |
| `fee_rate` | 0.0005 | 0.05% per side maker fee |
| `l2_lambda` | 0.1 | L2 regularization on weights |
| `optimize` | True | Whether to run differential evolution |

---

## Architecture

```
vibranium/
  ou_estimator.py       — fit_ou(), detrend_spread(), spread_zscore()
  portfolio_builder.py  — prescreen → Johansen → optimize → validate
  signal.py             — per-bar z-score + entry/exit state machine
  backtest.py           — walk-forward simulation with multi-leg PnL
  research.py           — hyperparameter sweep, summary tables
  data.py               — 15m kline loader (CSV → aligned DataFrame)
  run_research.py       — CLI entry point
```

---

## Known Issues and Next Steps

### Why initial backtest showed negative Sharpe

1. **Top-6 majors are too correlated.** BTC, ETH, SOL, BNB, XRP, DOGE all move as one factor. Johansen finds "stationary" combinations, but they're dominated by tiny residual noise with no predictive structure. Crypto stat arb needs 20+ altcoins with diverse idiosyncratic behavior.

2. **Johansen maximizes stationarity, not profit.** The most stationary eigenvector can have negative drift or systematically lose money. The optimizer (Step 3) is designed to fix this, but it's currently too slow for production use on long lookback windows.

3. **Sign ambiguity.** Johansen eigenvectors are unique up to sign. The negative of any eigenvector is equally stationary. Current code doesn't try both signs — it should pick the sign that gives positive α.

### Planned improvements

- **Sign flip**: try both +w and -w for each Johansen eigenvector, keep the one with α > 0
- **Faster optimizer**: subsample the spread to ~2000 bars inside the objective function, or use a proxy objective (ADF statistic) instead of full OU fit
- **Broader universe**: 20-30 mid/small-cap altcoins for better diversification and stronger cointegration relationships
- **Adaptive rebalancing**: refit when OU half-life exceeds 2× in-sample estimate, rather than on a fixed schedule
- **Multi-timeframe**: fit on 1h bars (fewer samples, faster), trade on 15m bars (finer entry/exit)
