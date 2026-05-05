"""
vibranium/break_recover.py — Break-Recover FSM for 2-leg pairs.

Tracks cointegration state per pair: INACTIVE -> COINTEGRATED -> BROKEN -> RECOVERED.
Only pairs in RECOVERED state are traded (mean-reversion after proven durability).
"""
import enum
import itertools
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class State(enum.Enum):
    INACTIVE = "INACTIVE"
    COINTEGRATED = "COINTEGRATED"
    BROKEN = "BROKEN"
    RECOVERED = "RECOVERED"


def compute_sharpe_weight(sharpe: float, max_weight: float = 3.0) -> float:
    """Convert rolling Sharpe to position weight. Negative -> 0."""
    if sharpe <= 0:
        return 0.0
    return min(sharpe, max_weight)


def check_momentum_confirm(z_history, direction, confirm_bars=2):
    """Check if z-score is reverting toward zero for N bars before entry.

    direction=+1 (long): z should be increasing (less negative)
    direction=-1 (short): z should be decreasing (less positive)

    Returns True (allow entry) if confirm_bars <= 0 or insufficient history.
    """
    if confirm_bars <= 0 or len(z_history) < confirm_bars + 1:
        return True
    recent = z_history[-(confirm_bars + 1):]
    for i in range(1, len(recent)):
        if direction > 0 and recent[i] <= recent[i - 1]:
            return False
        if direction < 0 and recent[i] >= recent[i - 1]:
            return False
    return True


def apply_momentum_filter(signals, z_scores, confirm_bars=2):
    """Post-process signals: zero out entries where z isn't reverting.

    signals: array of -1, 0, +1 (or floats for ladder)
    z_scores: array of z-score values (same length)
    """
    if confirm_bars <= 0:
        return signals
    filtered = signals.copy()
    for i in range(1, len(signals)):
        prev = signals[i - 1] if i > 0 else 0
        curr = signals[i]
        # Detect entry (0 -> nonzero)
        if prev == 0 and curr != 0:
            direction = 1 if curr > 0 else -1
            start = max(0, i - confirm_bars)
            z_hist = list(z_scores[start:i + 1])
            if not check_momentum_confirm(z_hist, direction,
                                          confirm_bars):
                filtered[i] = 0
    return filtered


class PairFSM:
    """
    4-state FSM tracking cointegration break-recover cycles.

    Transitions based on ADF p-value vs threshold:
      INACTIVE      -- p < thresh --> COINTEGRATED
      COINTEGRATED  -- p >= thresh --> BROKEN
      BROKEN        -- p < thresh --> RECOVERED  (entry trigger)
      RECOVERED     -- p >= thresh --> BROKEN    (exit trigger)
      RECOVERED     -- p < thresh --> RECOVERED  (stay, keep trading)
    """

    def __init__(self, adf_threshold: float = 0.05):
        self.adf_threshold = adf_threshold
        self.state = State.INACTIVE
        self.history: List[Tuple[State, State]] = []

    def update(self, adf_pvalue: float) -> State:
        """Advance FSM based on latest ADF p-value. Returns new state."""
        old = self.state
        cointegrated = adf_pvalue < self.adf_threshold

        if self.state == State.INACTIVE:
            if cointegrated:
                self.state = State.COINTEGRATED
        elif self.state == State.COINTEGRATED:
            if not cointegrated:
                self.state = State.BROKEN
        elif self.state == State.BROKEN:
            if cointegrated:
                self.state = State.RECOVERED
        elif self.state == State.RECOVERED:
            if not cointegrated:
                self.state = State.BROKEN

        if old != self.state:
            self.history.append((old, self.state))

        return self.state

    @property
    def is_tradeable(self) -> bool:
        return self.state == State.RECOVERED


class MultiTFFilter:
    """Gate trades by requiring both primary (15m) and secondary (1h)
    FSMs to be RECOVERED."""

    def __init__(self, adf_threshold: float = 0.05):
        self.primary = PairFSM(adf_threshold=adf_threshold)
        self.secondary = PairFSM(adf_threshold=adf_threshold)

    def update_primary(self, adf_pvalue: float) -> State:
        return self.primary.update(adf_pvalue)

    def update_secondary(self, adf_pvalue: float) -> State:
        return self.secondary.update(adf_pvalue)

    @property
    def is_tradeable(self) -> bool:
        return self.primary.is_tradeable and self.secondary.is_tradeable


# ---------------------------------------------------------------------------
# Walk-forward backtest with break-recover FSM
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from vibranium.pairs import PairEngine, KalmanHedge


class BreakRecoverBacktest:
    """
    Walk-forward backtest using break-recover FSM.

    Only trades mean-reversion when a pair enters RECOVERED state
    (cointegration broke then re-established).

    Sizing: base leg = capital USDT, hedge leg = capital * |h| USDT.
    """

    def __init__(self, lookback_bars=8640, rebalance_bars=672,
                 capital=110.0, fee_rate=0.0005, adf_threshold=0.05,
                 entry_z=2.0, exit_z=0.5, max_halflife=500.0,
                 filter_min_pass=0, filter_lookback_n=0,
                 n_rungs=1, rung_step=0.5,
                 sharpe_weight=False, confirm_bars=0,
                 use_multi_tf=False, secondary_prices=None,
                 use_kalman=False):
        self.lookback_bars = lookback_bars
        self.rebalance_bars = rebalance_bars
        self.capital = capital
        self.fee_rate = fee_rate
        self.adf_threshold = adf_threshold
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.max_halflife = max_halflife
        self.filter_min_pass = filter_min_pass
        self.filter_lookback_n = filter_lookback_n
        self.n_rungs = n_rungs
        self.rung_step = rung_step
        self.sharpe_weight = sharpe_weight
        self.confirm_bars = confirm_bars
        self.use_multi_tf = use_multi_tf
        self.secondary_prices = secondary_prices
        self.use_kalman = use_kalman

    def _passes_filter(self, window_passes):
        """Check if pair passes rolling scorecard filter.

        Returns True if filter is disabled, no history yet (first window),
        or enough recent windows were positive.
        """
        if self.filter_min_pass <= 0 or self.filter_lookback_n <= 0:
            return True  # Filter disabled
        if not window_passes:
            return True  # First window always trades
        recent = window_passes[-self.filter_lookback_n:]
        return sum(recent) >= self.filter_min_pass

    def _run_adf(self, spread_values):
        """Run ADF test, return p-value. Returns 1.0 on failure."""
        clean = spread_values[np.isfinite(spread_values)]
        if len(clean) < 20:
            return 1.0
        try:
            result = adfuller(clean, autolag="AIC")
            return float(result[1])
        except Exception:
            return 1.0

    def run(self, sym_a, sym_b, prices):
        """
        Run walk-forward break-recover backtest on one pair.

        Returns dict with: daily_pnl, sharpe, total_return, max_drawdown,
        n_trades, n_rebalances, state_log.
        """
        n = len(prices)
        all_pnl = []
        state_log = []
        n_trades = 0
        n_filtered = 0
        window_passes = []  # Rolling scorecard: True/False per traded window
        rolling_sharpes = []  # Per-window Sharpe for weighting

        # Multi-TF: use MultiTFFilter; single-TF: use PairFSM
        if self.use_multi_tf:
            fsm = MultiTFFilter(adf_threshold=self.adf_threshold)
        else:
            fsm = PairFSM(adf_threshold=self.adf_threshold)

        # Kalman filter per pair (persists across rebalance windows)
        kalman = KalmanHedge() if self.use_kalman else None

        i = self.lookback_bars
        while i < n:
            lb_start = i - self.lookback_bars
            lb_prices = prices.iloc[lb_start:i]

            # Fit hedge ratio and spread
            engine = PairEngine(sym_a, sym_b)
            engine.fit(lb_prices)

            # ADF on the spread (primary TF)
            spread_vals = (engine.spread.values
                           if engine.spread is not None
                           else np.array([]))
            adf_p = self._run_adf(spread_vals)

            # Advance FSM
            if self.use_multi_tf:
                old_state_p = fsm.primary.state
                fsm.update_primary(adf_pvalue=adf_p)
                # Secondary TF: fit on 1h data and run ADF
                sec_adf_p = 1.0
                if (self.secondary_prices is not None
                        and sym_a in self.secondary_prices.columns
                        and sym_b in self.secondary_prices.columns):
                    sec_prices = self.secondary_prices[[sym_a, sym_b]]
                    # Find the 1h bar closest to current 15m timestamp
                    ts = prices.index[min(i, n - 1)]
                    sec_mask = sec_prices.index <= ts
                    if sec_mask.sum() > 100:
                        sec_lb = sec_prices[sec_mask].iloc[-self.lookback_bars // 4:]
                        if len(sec_lb) > 50:
                            sec_engine = PairEngine(sym_a, sym_b)
                            sec_engine.fit(sec_lb)
                            sec_spread = (sec_engine.spread.values
                                          if sec_engine.spread is not None
                                          else np.array([]))
                            sec_adf_p = self._run_adf(sec_spread)
                fsm.update_secondary(adf_pvalue=sec_adf_p)
                new_state = fsm.primary.state
                if old_state_p != new_state:
                    state_log.append({
                        "bar": i,
                        "time": (str(prices.index[i])
                                 if i < n else None),
                        "from": old_state_p.value,
                        "to": new_state.value,
                        "adf_p": round(adf_p, 4),
                    })
            else:
                old_state = fsm.state
                fsm.update(adf_pvalue=adf_p)
                if old_state != fsm.state:
                    state_log.append({
                        "bar": i,
                        "time": (str(prices.index[i])
                                 if i < n else None),
                        "from": old_state.value,
                        "to": fsm.state.value,
                        "adf_p": round(adf_p, 4),
                    })

            fwd_end = min(i + self.rebalance_bars, n)

            # Only trade in RECOVERED state
            if (fsm.is_tradeable
                    and engine.ou is not None
                    and engine.ou.kappa > 0):
                if engine.ou.half_life <= self.max_halflife:
                    # Check rolling scorecard filter
                    if not self._passes_filter(window_passes):
                        n_filtered += 1
                        i = fwd_end
                        continue

                    fwd_prices = prices.iloc[i:fwd_end]
                    sigs_df = engine.generate_signals(
                        fwd_prices,
                        entry_z=self.entry_z,
                        exit_z=self.exit_z,
                        max_halflife=self.max_halflife,
                        lookback_start_idx=self.lookback_bars,
                        n_rungs=self.n_rungs,
                        rung_step=self.rung_step,
                    )
                    signals = sigs_df["signal"].values

                    # Momentum confirmation filter
                    if self.confirm_bars > 0:
                        z_vals = sigs_df["z_score"].values
                        signals = apply_momentum_filter(
                            signals, z_vals, self.confirm_bars
                        )

                    # Update Kalman filter on forward window if enabled
                    if kalman is not None:
                        for j in range(len(fwd_prices)):
                            la = np.log(fwd_prices[sym_a].iloc[j])
                            lb = np.log(fwd_prices[sym_b].iloc[j])
                            if np.isfinite(la) and np.isfinite(lb):
                                kalman.update(la, lb)

                    # PnL in USDT with asymmetric sizing
                    log_ret_a = (np.log(fwd_prices[sym_a])
                                 .diff().fillna(0).values)
                    log_ret_b = (np.log(fwd_prices[sym_b])
                                 .diff().fillna(0).values)
                    h = (abs(kalman.hedge_ratio)
                         if kalman is not None and kalman.n_updates > 100
                         else abs(engine.hedge_ratio))
                    port_ret_usdt = (self.capital * log_ret_a
                                     - self.capital * h * log_ret_b)

                    # Fee on total notional
                    total_notional = self.capital + self.capital * h
                    fee_per_trade = self.fee_rate * total_notional

                    prev_sig = np.concatenate([[0.0], signals[:-1]])
                    pnl = prev_sig * port_ret_usdt
                    delta = np.abs(signals - prev_sig)
                    pnl = pnl - delta * fee_per_trade

                    # Apply Sharpe weight if enabled
                    if self.sharpe_weight:
                        if not rolling_sharpes:
                            sw = 1.0  # First window: no history
                        else:
                            sw = compute_sharpe_weight(
                                rolling_sharpes[-1])
                        pnl = pnl * sw

                    # Score this window for the rolling scorecard
                    window_pnl = pd.Series(pnl, index=fwd_prices.index)
                    daily_w = window_pnl.resample("D").sum()
                    daily_w = daily_w[daily_w != 0]
                    if len(daily_w) > 1:
                        w_std = daily_w.std()
                        w_sharpe = (float(daily_w.mean() / w_std)
                                    if w_std > 0 else 0.0)
                        window_passes.append(w_sharpe > 0)
                    else:
                        w_sharpe = (float(daily_w.sum())
                                    if len(daily_w) > 0 else 0.0)
                        window_passes.append(daily_w.sum() > 0)
                    rolling_sharpes.append(w_sharpe)

                    n_trades += int(np.sum(delta > 0))
                    all_pnl.append(window_pnl)

            i = fwd_end

        # Aggregate results
        if not all_pnl:
            return {
                "daily_pnl": pd.Series(dtype=float),
                "sharpe": 0.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "n_trades": 0,
                "n_rebalances": len(state_log),
                "n_filtered": n_filtered,
                "state_log": state_log,
            }

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

        return {
            "daily_pnl": daily,
            "sharpe": sharpe,
            "total_return": float(daily.sum()),
            "max_drawdown": maxdd,
            "n_trades": n_trades,
            "n_rebalances": len(state_log),
            "n_filtered": n_filtered,
            "state_log": state_log,
        }


# ---------------------------------------------------------------------------
# Multi-pair sweep + aggregation
# ---------------------------------------------------------------------------


class BreakRecoverSweep:
    """Sweep all pairs in a price DataFrame through break-recover backtest."""

    def __init__(self, **bt_kwargs):
        self.bt_kwargs = bt_kwargs

    def run(self, prices, symbols=None):
        """
        Run break-recover backtest on all pair combinations.
        Returns list of result dicts sorted by Sharpe descending.
        """
        if symbols is None:
            symbols = list(prices.columns)

        pairs = list(itertools.combinations(symbols, 2))
        results = []

        for sym_a, sym_b in pairs:
            # Try both orderings, keep better Sharpe
            best = None
            for a, b in [(sym_a, sym_b), (sym_b, sym_a)]:
                bt = BreakRecoverBacktest(**self.bt_kwargs)
                r = bt.run(a, b, prices)
                r["pair"] = f"{a}/{b}"
                if best is None or r["sharpe"] > best["sharpe"]:
                    best = r
            if best is not None:
                results.append(best)

        results.sort(key=lambda r: r["sharpe"], reverse=True)
        return results

    def aggregate(self, prices, symbols=None):
        """
        Run sweep and aggregate daily PnL across all pairs.
        Returns dict with aggregate daily_pnl, sharpe, per_pair results.
        """
        per_pair = self.run(prices, symbols)
        all_daily = []

        for r in per_pair:
            if len(r["daily_pnl"]) > 0:
                all_daily.append(r["daily_pnl"].rename(r["pair"]))

        if not all_daily:
            return {
                "daily_pnl": pd.Series(dtype=float),
                "sharpe": 0.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "per_pair": per_pair,
            }

        combined = pd.concat(all_daily, axis=1).fillna(0).sum(axis=1)
        daily = combined[combined != 0]
        std = daily.std()
        sharpe = float(daily.mean() / std * np.sqrt(365)) if std > 0 else 0.0
        cum = daily.cumsum()
        maxdd = float((cum - cum.cummax()).min()) if len(cum) > 0 else 0.0

        return {
            "daily_pnl": daily,
            "sharpe": sharpe,
            "total_return": float(daily.sum()),
            "max_drawdown": maxdd,
            "per_pair": per_pair,
        }
