"""
vibranium/pairs.py — 2-leg pairs mean-reversion engine.

PairEngine: fit hedge ratio, detrend spread, estimate OU parameters.
PairsBacktest: walk-forward backtest for a single pair.
PairsResearch: sweep all pairs x hyperparameters, rank by Sharpe.
"""
import itertools
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from vibranium.ou_estimator import OUParams, fit_ou, detrend_spread
logger = logging.getLogger(__name__)


def _apply_state_machine_v2(z_arr, entry_z, exit_z, stop_z=0.0, max_hold=0):
    """
    Entry/exit state machine with stop-loss and max holding period.

    +1 when z < -entry_z, -1 when z > +entry_z, 0 when |z| < exit_z.
    Force exit if |z| > stop_z (stop-loss) or held > max_hold bars.
    stop_z=0 or max_hold=0 disables the respective check.
    """
    n = len(z_arr)
    signals = np.zeros(n, dtype=np.int8)
    position = 0
    bars_held = 0

    for i in range(n):
        z = z_arr[i]
        if np.isnan(z):
            signals[i] = position
            if position != 0:
                bars_held += 1
            continue

        if position == 0:
            bars_held = 0
            if z < -entry_z:
                position = 1
                bars_held = 1
            elif z > entry_z:
                position = -1
                bars_held = 1
        else:
            bars_held += 1
            # Normal exit: mean reverted
            if abs(z) < exit_z:
                position = 0
                bars_held = 0
            # Stop-loss: spread blew out
            elif stop_z > 0 and abs(z) > stop_z:
                position = 0
                bars_held = 0
            # Max hold: force exit
            elif max_hold > 0 and bars_held > max_hold:
                position = 0
                bars_held = 0

        signals[i] = position

    return signals


def _apply_state_machine_ladder(z_arr, entry_z, exit_z, stop_z=0.0,
                                max_hold=0, n_rungs=5, rung_step=0.5):
    """
    Ladder entry state machine: add 1/n_rungs position every rung_step z.

    Rung 1: |z| > entry_z           → 1/n_rungs
    Rung 2: |z| > entry_z + step    → 2/n_rungs
    ...
    Rung N: |z| > entry_z + (N-1)*step → N/N_rungs (full position)

    Exit when |z| < exit_z. Returns float array of position sizes in [-1, 1].
    """
    n = len(z_arr)
    signals = np.zeros(n, dtype=np.float64)
    direction = 0  # +1 or -1
    bars_held = 0

    for i in range(n):
        z = z_arr[i]
        if np.isnan(z):
            signals[i] = signals[i - 1] if i > 0 else 0.0
            if direction != 0:
                bars_held += 1
            continue

        if direction == 0:
            # Not in a position — check for entry
            bars_held = 0
            if z < -entry_z:
                direction = 1
                bars_held = 1
            elif z > entry_z:
                direction = -1
                bars_held = 1
            else:
                signals[i] = 0.0
                continue

        if direction != 0:
            bars_held += 1
            az = abs(z)

            # Exit conditions
            if az < exit_z:
                direction = 0
                bars_held = 0
                signals[i] = 0.0
                continue
            if stop_z > 0 and az > stop_z:
                direction = 0
                bars_held = 0
                signals[i] = 0.0
                continue
            if max_hold > 0 and bars_held > max_hold:
                direction = 0
                bars_held = 0
                signals[i] = 0.0
                continue

            # Compute ladder size: how many rungs are filled?
            rungs_filled = 0
            for r in range(n_rungs):
                threshold = entry_z + r * rung_step
                if az > threshold:
                    rungs_filled = r + 1
            size = rungs_filled / n_rungs
            signals[i] = direction * size

    return signals


def _apply_state_machine_agg_ladder(z_arr, entry_z, exit_z, stop_z=0.0,
                                    max_hold=0, n_rungs=10, rung_step=0.5):
    """
    Aggressive ladder: rung k adds k units. Sizes up harder at extremes.

    Rung 1: |z| > entry_z           → weight 1
    Rung 2: |z| > entry_z + step    → weight 1+2 = 3
    Rung 3: |z| > entry_z + 2*step  → weight 1+2+3 = 6
    ...
    Rung N: |z| > entry_z + (N-1)*step → weight N*(N+1)/2

    Normalized: position = sum(1..k) / sum(1..N) = k*(k+1) / (N*(N+1))
    At full ladder, position = 1.0.
    """
    n = len(z_arr)
    signals = np.zeros(n, dtype=np.float64)
    direction = 0
    bars_held = 0
    total_weight = n_rungs * (n_rungs + 1) / 2.0  # sum(1..N)

    for i in range(n):
        z = z_arr[i]
        if np.isnan(z):
            signals[i] = signals[i - 1] if i > 0 else 0.0
            if direction != 0:
                bars_held += 1
            continue

        if direction == 0:
            bars_held = 0
            if z < -entry_z:
                direction = 1
                bars_held = 1
            elif z > entry_z:
                direction = -1
                bars_held = 1
            else:
                signals[i] = 0.0
                continue

        if direction != 0:
            bars_held += 1
            az = abs(z)

            if az < exit_z:
                direction = 0
                bars_held = 0
                signals[i] = 0.0
                continue
            if stop_z > 0 and az > stop_z:
                direction = 0
                bars_held = 0
                signals[i] = 0.0
                continue
            if max_hold > 0 and bars_held > max_hold:
                direction = 0
                bars_held = 0
                signals[i] = 0.0
                continue

            # Aggressive ladder: rung k adds k units
            rungs_filled = 0
            for r in range(n_rungs):
                threshold = entry_z + r * rung_step
                if az > threshold:
                    rungs_filled = r + 1
            # Cumulative weight: sum(1..k) = k*(k+1)/2
            cum_weight = rungs_filled * (rungs_filled + 1) / 2.0
            size = cum_weight / total_weight
            signals[i] = direction * size

    return signals


class KalmanHedge:
    """Online Kalman filter for dynamic hedge ratio estimation.

    Observation model: log(A) = h * log(B) + noise
    State: hedge ratio h (scalar)
    """

    def __init__(self, delta=1e-4, ve=1e-3):
        self.delta = delta  # Process noise (how fast h can change)
        self.ve = ve        # Observation noise
        self.hedge_ratio = 0.0
        self.P = 1.0       # State covariance
        self.n_updates = 0

    def update(self, log_a: float, log_b: float) -> float:
        """Update hedge ratio with new observation."""
        P_pred = self.P + self.delta
        y = log_a - self.hedge_ratio * log_b
        S = log_b * log_b * P_pred + self.ve
        K = P_pred * log_b / S
        self.hedge_ratio += K * y
        self.P = (1 - K * log_b) * P_pred
        self.n_updates += 1
        return self.hedge_ratio


class PairEngine:
    """
    Fit and evaluate a 2-leg cointegrated pair.

    spread = log(A) - h * log(B), where h is OLS hedge ratio.
    """

    def __init__(self, sym_a: str, sym_b: str):
        self.sym_a = sym_a
        self.sym_b = sym_b
        self.hedge_ratio: float = 1.0
        self.ou: Optional[OUParams] = None
        self.alpha_per_bar: float = 0.0
        self.intercept: float = 0.0
        self.spread: Optional[pd.Series] = None

    def fit(self, prices: pd.DataFrame) -> "PairEngine":
        """Fit hedge ratio and OU parameters on a price window."""
        log_a = np.log(prices[self.sym_a]).values
        log_b = np.log(prices[self.sym_b]).values
        mask = np.isfinite(log_a) & np.isfinite(log_b)
        X = np.column_stack([np.ones(mask.sum()), log_b[mask]])
        beta, _, _, _ = np.linalg.lstsq(X, log_a[mask], rcond=None)
        self.hedge_ratio = float(beta[1])
        spread_raw = pd.Series(
            log_a - self.hedge_ratio * log_b, index=prices.index
        )
        self.spread = spread_raw
        detrended, self.alpha_per_bar, self.intercept = detrend_spread(
            spread_raw
        )
        self.ou = fit_ou(detrended.values)
        return self

    def zscore(self, prices, lookback_start_idx=0):
        """Z-score of spread using fitted OU params. Safe for OOS data."""
        log_a = np.log(prices[self.sym_a]).values
        log_b = np.log(prices[self.sym_b]).values
        spread = log_a - self.hedge_ratio * log_b
        t = np.arange(len(spread), dtype=float) + lookback_start_idx
        trend = self.intercept + self.alpha_per_bar * t
        s_detrended = spread - trend
        sigma_eq = (self.ou.sigma / np.sqrt(2 * self.ou.kappa)
                    if self.ou.kappa > 0 and self.ou.sigma > 0
                    else np.nan)
        if np.isnan(sigma_eq) or sigma_eq <= 0:
            return pd.Series(np.full(len(prices), np.nan),
                             index=prices.index)
        z = (s_detrended - self.ou.mu) / sigma_eq
        return pd.Series(z, index=prices.index)

    def generate_signals(self, prices, entry_z=2.0, exit_z=0.5,
                         stop_z=0.0, max_hold=0,
                         n_rungs=1, rung_step=0.5,
                         ladder_mode="linear",
                         max_halflife=500.0, lookback_start_idx=0):
        """
        Generate entry/exit signals.

        Parameters
        ----------
        stop_z : float
            Stop-loss: force exit if |z| exceeds this. 0 = disabled.
        max_hold : int
            Max bars to hold a position. 0 = disabled.
        n_rungs : int
            Ladder rungs. 1 = all-in (original). >1 = scale in.
        rung_step : float
            Z-score distance between rungs.
        ladder_mode : str
            "linear" = equal size per rung. "agg" = rung k adds k units.

        Returns DataFrame: spread, z_score, signal.
        """
        log_a = np.log(prices[self.sym_a]).values
        log_b = np.log(prices[self.sym_b]).values
        spread = log_a - self.hedge_ratio * log_b
        z = self.zscore(prices, lookback_start_idx=lookback_start_idx)
        if (self.ou is None or self.ou.kappa <= 0
                or self.ou.half_life > max_halflife):
            return pd.DataFrame({
                "spread": spread, "z_score": z.values,
                "signal": np.zeros(len(prices), dtype=float),
            }, index=prices.index)
        if n_rungs > 1 and ladder_mode == "agg":
            signals = _apply_state_machine_agg_ladder(
                z.values, entry_z, exit_z, stop_z, max_hold,
                n_rungs, rung_step)
        elif n_rungs > 1:
            signals = _apply_state_machine_ladder(
                z.values, entry_z, exit_z, stop_z, max_hold,
                n_rungs, rung_step)
        else:
            signals = _apply_state_machine_v2(
                z.values, entry_z, exit_z, stop_z, max_hold
            ).astype(float)
        return pd.DataFrame({
            "spread": spread, "z_score": z.values,
            "signal": signals,
        }, index=prices.index)


# ---------------------------------------------------------------------------
# PnL computation
# ---------------------------------------------------------------------------

def compute_pair_pnl(signals, port_ret, fee):
    """
    Bar-level PnL: pnl[t] = prev_signal * port_ret[t] - fee on changes.

    port_ret = log_ret_A - h * log_ret_B (the spread return).
    signal=+1 means long the spread (long A, short B).
    Supports fractional signals (ladder sizing).
    Fee scales with the size of position change.
    """
    signals = np.asarray(signals, dtype=float)
    prev_sig = np.concatenate([[0.0], signals[:-1]])
    pnl = prev_sig * port_ret
    # Fee proportional to change in position size
    delta = np.abs(signals - prev_sig)
    pnl = pnl - delta * fee
    return pnl


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

@dataclass
class PairBacktestResult:
    """Result from a single-pair walk-forward backtest."""
    pair: str
    daily_pnl: pd.Series
    sharpe: float
    total_return: float
    max_drawdown: float
    avg_halflife: float
    n_trades: int
    n_rebalances: int
    entry_z: float
    exit_z: float


class PairsBacktest:
    """Walk-forward backtest for a single pair."""

    def __init__(self, lookback_bars=5760, rebalance_bars=672,
                 entry_z=2.0, exit_z=0.5, stop_z=0.0, max_hold=0,
                 n_rungs=1, rung_step=0.5, ladder_mode="linear",
                 max_halflife=500.0, fee_rate=0.0005):
        self.lookback_bars = lookback_bars
        self.rebalance_bars = rebalance_bars
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.max_hold = max_hold
        self.n_rungs = n_rungs
        self.rung_step = rung_step
        self.ladder_mode = ladder_mode
        self.max_halflife = max_halflife
        self.fee_rate = fee_rate

    def run(self, sym_a, sym_b, prices):
        """Run walk-forward backtest on one pair."""
        n = len(prices)
        all_pnl = []
        halflives = []
        n_trades = 0

        i = self.lookback_bars
        while i < n:
            lb_prices = prices.iloc[i - self.lookback_bars:i]
            engine = PairEngine(sym_a, sym_b)
            engine.fit(lb_prices)

            if engine.ou is None or engine.ou.kappa <= 0:
                i += self.rebalance_bars
                continue

            halflives.append(engine.ou.half_life)
            fwd_end = min(i + self.rebalance_bars, n)
            fwd_prices = prices.iloc[i:fwd_end]

            sigs_df = engine.generate_signals(
                fwd_prices, entry_z=self.entry_z, exit_z=self.exit_z,
                stop_z=self.stop_z, max_hold=self.max_hold,
                n_rungs=self.n_rungs, rung_step=self.rung_step,
                ladder_mode=self.ladder_mode,
                max_halflife=self.max_halflife,
                lookback_start_idx=self.lookback_bars,
            )

            # Spread return: log_ret_A - h * log_ret_B
            log_ret_a = np.log(fwd_prices[sym_a]).diff().fillna(0).values
            log_ret_b = np.log(fwd_prices[sym_b]).diff().fillna(0).values
            port_ret = log_ret_a - engine.hedge_ratio * log_ret_b

            signals = sigs_df["signal"].values
            pnl = compute_pair_pnl(signals, port_ret, self.fee_rate * 2)

            n_trades += int(
                np.sum(np.abs(np.diff(signals, prepend=0)) > 0)
            )
            all_pnl.append(pd.Series(pnl, index=fwd_prices.index))
            i = fwd_end

        if not all_pnl:
            return PairBacktestResult(
                pair=f"{sym_a}/{sym_b}",
                daily_pnl=pd.Series(dtype=float),
                sharpe=0.0, total_return=0.0, max_drawdown=0.0,
                avg_halflife=0.0, n_trades=0, n_rebalances=0,
                entry_z=self.entry_z, exit_z=self.exit_z,
            )

        pnl_all = pd.concat(all_pnl)
        pnl_all = pnl_all[~pnl_all.index.duplicated(keep="first")]
        daily = pnl_all.resample("D").sum()
        daily = daily[daily != 0]

        std = daily.std()
        sharpe = (float(daily.mean() / std * np.sqrt(365))
                  if std > 0 else 0.0)
        cum = daily.cumsum()
        maxdd = (float((cum - cum.cummax()).min())
                 if len(cum) > 0 else 0.0)

        return PairBacktestResult(
            pair=f"{sym_a}/{sym_b}",
            daily_pnl=daily,
            sharpe=sharpe,
            total_return=float(daily.sum()),
            max_drawdown=maxdd,
            avg_halflife=(float(np.mean(halflives))
                          if halflives else 0.0),
            n_trades=n_trades,
            n_rebalances=len(halflives),
            entry_z=self.entry_z,
            exit_z=self.exit_z,
        )


# ---------------------------------------------------------------------------
# Research sweep
# ---------------------------------------------------------------------------

class PairsResearch:
    """Sweep all pairs x hyperparameters, rank by Sharpe."""

    def __init__(self, **bt_kwargs):
        self.bt_kwargs = bt_kwargs

    def sweep(self, prices, symbols=None, entry_z_values=None):
        """
        Run all pairs x entry_z x sign-flip. Returns sorted results.

        Sign-flip: tries both A/B and B/A orderings, keeps better Sharpe.
        """
        if symbols is None:
            symbols = list(prices.columns)
        if entry_z_values is None:
            entry_z_values = [1.5, 2.0, 2.5]

        pairs = list(itertools.combinations(symbols, 2))
        results = []

        for sym_a, sym_b in pairs:
            best = None
            for ez in entry_z_values:
                for a, b in [(sym_a, sym_b), (sym_b, sym_a)]:
                    bt = PairsBacktest(entry_z=ez, **self.bt_kwargs)
                    r = bt.run(a, b, prices)
                    if r.n_rebalances == 0:
                        continue
                    if best is None or r.sharpe > best.sharpe:
                        best = r
            if best is not None:
                results.append(best)

        results.sort(key=lambda r: r.sharpe, reverse=True)
        return results

    @staticmethod
    def print_summary(results):
        """Print ranked results table."""
        if not results:
            print("No results.")
            return

        sep = "-" * 78
        print(sep)
        print("VIBRANIUM PAIRS RESEARCH")
        print(sep)
        print(
            f"{'#':>3} {'Pair':<25} {'ez':>4} {'Sharpe':>7} "
            f"{'Return':>9} {'MaxDD':>9} {'HL':>5} {'Trds':>5}"
        )
        print(sep)

        for i, r in enumerate(results):
            sign = "+" if r.total_return >= 0 else ""
            print(
                f"{i+1:>3} {r.pair:<25} {r.entry_z:>4.1f} "
                f"{r.sharpe:>+7.2f} {sign}{r.total_return:>8.4f} "
                f"{r.max_drawdown:>+9.4f} {r.avg_halflife:>5.0f} "
                f"{r.n_trades:>5d}"
            )

        print(sep)
        pos = sum(1 for r in results if r.sharpe > 0)
        print(f"Positive Sharpe: {pos}/{len(results)}")
        print(sep)
