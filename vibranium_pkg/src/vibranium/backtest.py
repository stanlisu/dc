"""
vibranium/backtest.py — Walk-forward backtest for the Vibranium stationary portfolio strategy.

Every rebalance_period:
  1. Fit portfolio weights on lookback window
  2. Generate signals on the next rebalance window
  3. Compute multi-leg PnL with transaction costs

Output: daily PnL, Sharpe, drawdown, per-year stats.
"""
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vibranium.portfolio_builder import build_portfolio, PortfolioResult, _log_prices
from vibranium.spread_signal import generate_signals, SignalConfig

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Backtest parameters."""
    lookback_bars: int = 8640     # ~90 days of 15m bars
    rebalance_bars: int = 672     # ~7 days of 15m bars
    n_assets: int = 8
    max_weight: float = 0.25
    entry_z: float = 2.0
    exit_z: float = 0.5
    max_halflife: float = 500.0
    fee_rate: float = 0.0005     # 0.05% maker fee per side
    optimize: bool = True
    l2_lambda: float = 0.1


@dataclass
class BacktestResult:
    """Walk-forward backtest results."""
    daily_pnl: pd.Series
    sharpe: float
    max_drawdown: float
    total_return: float
    n_rebalances: int
    avg_halflife: float
    avg_n_trades: float
    per_year: pd.DataFrame = field(default_factory=pd.DataFrame)
    portfolio_log: list = field(default_factory=list)  # per-rebalance metadata


def run_backtest(prices: pd.DataFrame, config: BacktestConfig) -> BacktestResult:
    """
    Walk-forward backtest.

    Parameters
    ----------
    prices : pd.DataFrame
        Close prices for the full symbol universe (15m bars).
    config : BacktestConfig
        All hyperparameters.

    Returns
    -------
    BacktestResult
    """
    log_p = _log_prices(prices)
    n_bars = len(prices)
    signal_config = SignalConfig(
        entry_z=config.entry_z,
        exit_z=config.exit_z,
        max_halflife=config.max_halflife,
    )

    all_pnl = []
    portfolio_log = []
    halflives = []
    trade_counts = []

    i = config.lookback_bars
    while i < n_bars:
        # Lookback window
        lb_start = i - config.lookback_bars
        lb_end = i
        lookback_prices = prices.iloc[lb_start:lb_end]

        # Fit portfolio
        portfolio = build_portfolio(
            lookback_prices,
            n_assets=config.n_assets,
            max_weight=config.max_weight,
            l2_lambda=config.l2_lambda,
            optimize=config.optimize,
            max_halflife=config.max_halflife,
        )

        # Forward window
        fwd_end = min(i + config.rebalance_bars, n_bars)
        fwd_prices = prices.iloc[i:fwd_end]
        fwd_log_p = log_p.iloc[i:fwd_end]

        if portfolio is None:
            # No valid portfolio — stay flat
            flat_pnl = pd.Series(0.0, index=fwd_prices.index)
            all_pnl.append(flat_pnl)
            portfolio_log.append({
                "bar_start": i, "bar_end": fwd_end,
                "status": "no_portfolio",
            })
            i = fwd_end
            continue

        halflives.append(portfolio.ou.half_life)

        # Generate signals on forward window
        signals_df = generate_signals(
            fwd_log_p,
            weights=portfolio.weights,
            symbols=portfolio.symbols,
            ou=portfolio.ou,
            alpha_per_bar=portfolio.alpha_per_bar,
            intercept=portfolio.intercept,
            config=signal_config,
            lookback_start_idx=config.lookback_bars,
        )

        # Compute PnL per bar
        bar_pnl = _compute_bar_pnl(
            fwd_prices, signals_df, portfolio, config.fee_rate,
        )
        all_pnl.append(bar_pnl)

        # Count trades
        sig_changes = np.diff(signals_df["signal"].values, prepend=0)
        n_trades = int(np.sum(np.abs(sig_changes) > 0))
        trade_counts.append(n_trades)

        portfolio_log.append({
            "bar_start": i,
            "bar_end": fwd_end,
            "symbols": portfolio.symbols,
            "weights": portfolio.to_dict(),
            "halflife": portfolio.ou.half_life,
            "kappa": portfolio.ou.kappa,
            "alpha": portfolio.alpha_per_bar,
            "method": portfolio.method,
            "n_trades": n_trades,
        })

        i = fwd_end

    if not all_pnl:
        return BacktestResult(
            daily_pnl=pd.Series(dtype=float),
            sharpe=0.0, max_drawdown=0.0, total_return=0.0,
            n_rebalances=0, avg_halflife=0.0, avg_n_trades=0.0,
        )

    # Combine all bar-level PnL
    bar_pnl_all = pd.concat(all_pnl)
    bar_pnl_all = bar_pnl_all[~bar_pnl_all.index.duplicated(keep="first")]

    # Aggregate to daily
    daily_pnl = bar_pnl_all.resample("D").sum()
    daily_pnl = daily_pnl[daily_pnl != 0]  # drop days with no data

    # Metrics
    std = daily_pnl.std()
    sharpe = float(daily_pnl.mean() / std * np.sqrt(365)) if std > 0 else 0.0

    cum = daily_pnl.cumsum()
    max_dd = float((cum - cum.cummax()).min()) if len(cum) > 0 else 0.0

    total_return = float(daily_pnl.sum())

    # Per-year
    per_year = pd.DataFrame()
    if not daily_pnl.empty:
        per_year = daily_pnl.groupby(daily_pnl.index.year).agg(
            total_pnl="sum",
            sharpe=lambda x: float(x.mean() / x.std() * np.sqrt(365)) if x.std() > 0 else 0.0,
        )

        def _max_dd(x):
            c = x.cumsum()
            return float((c - c.cummax()).min()) if len(c) > 0 else 0.0
        per_year["max_drawdown"] = daily_pnl.groupby(daily_pnl.index.year).apply(_max_dd)

    return BacktestResult(
        daily_pnl=daily_pnl,
        sharpe=sharpe,
        max_drawdown=max_dd,
        total_return=total_return,
        n_rebalances=len(portfolio_log),
        avg_halflife=float(np.mean(halflives)) if halflives else 0.0,
        avg_n_trades=float(np.mean(trade_counts)) if trade_counts else 0.0,
        per_year=per_year,
        portfolio_log=portfolio_log,
    )


def _compute_bar_pnl(
    prices: pd.DataFrame,
    signals_df: pd.DataFrame,
    portfolio: PortfolioResult,
    fee_rate: float,
) -> pd.Series:
    """
    Compute bar-level PnL from signals and portfolio weights.

    PnL = signal * Σ(w_i * log_return_i) per bar.
    Fee charged on signal changes.
    """
    log_p = _log_prices(prices)
    symbols = portfolio.symbols
    weights = portfolio.weights

    # Log returns per symbol
    avail = [s for s in symbols if s in log_p.columns]
    w_idx = [symbols.index(s) for s in avail]
    w_sub = np.array([weights[i] for i in w_idx])
    norm = np.sum(np.abs(w_sub))
    if norm > 0:
        w_sub = w_sub / norm

    log_ret = log_p[avail].diff()

    # Portfolio return per bar
    port_ret = log_ret.dot(w_sub)

    # PnL = previous signal * portfolio return (hold into this bar)
    sig = signals_df["signal"].values
    prev_sig = np.concatenate([[0], sig[:-1]])
    pnl = prev_sig * port_ret.values

    # Transaction cost on signal changes
    n_legs = len(avail)
    fee = np.where(sig != prev_sig, fee_rate * n_legs, 0.0)
    pnl = pnl - fee

    return pd.Series(pnl, index=prices.index, name="pnl")
