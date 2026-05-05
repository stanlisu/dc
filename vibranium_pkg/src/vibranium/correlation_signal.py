"""
vibranium/correlation_signal.py — Correlation-divergence mean-reversion signal.

Strategy:
  For each altcoin paired against an anchor (BTC or ETH):
  1. Rolling Pearson correlation of log-returns over corr_window bars.
  2. Moving average of that correlation over ma_window bars (the "baseline").
  3. Entry when corr < baseline - threshold:
       direction = +1 if spread_z < 0 (anchor cheap vs alt) → long anchor/short alt
       direction = -1 if spread_z > 0 (anchor expensive vs alt) → short anchor/long alt
  4. Exit when corr >= baseline.
  5. Threshold T is optimised on the train period by sweeping and maximising Sharpe.

Performance notes:
  - compute_correlation_series / compute_baseline / compute_spread are vectorised
    (pandas rolling + numpy cumsum OLS) — O(n) with no Python loops.
  - The state-machine in _apply_state_machine is unavoidably sequential but operates
    only on raw numpy arrays for minimal overhead.
  - optimise_threshold caches (corr, baseline, spread_z) once per pair and only
    re-runs the state-machine for each threshold value in the sweep.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ThresholdResult:
    """Result from optimise_threshold."""
    best_threshold: float
    train_sharpe: float
    sweep: pd.DataFrame = field(default_factory=pd.DataFrame)  # threshold -> sharpe curve


class CorrelationSignal:
    """
    Correlation-divergence mean-reversion signal for a single anchor/altcoin pair.

    Parameters
    ----------
    anchor : str
        Anchor symbol (e.g. "BTCUSDT" or "ETHUSDT").
    corr_window : int
        Number of bars for the rolling Pearson correlation (default 48 = 12h of 15m).
    ma_window : int
        Number of bars for the moving average of the correlation (default 48).
    entry_z : float
        Threshold T: entry when corr < baseline - T.
    exit_mode : str
        "cross_ma" — exit when corr crosses back above baseline.
    """

    def __init__(
        self,
        anchor: str = "BTCUSDT",
        corr_window: int = 48,
        ma_window: int = 48,
        entry_z: float = 0.0,
        exit_mode: str = "cross_ma",
    ):
        self.anchor = anchor
        self.corr_window = corr_window
        self.ma_window = ma_window
        self.entry_z = entry_z
        self.exit_mode = exit_mode

    # ------------------------------------------------------------------
    # Core computations (all vectorised — no Python loops)
    # ------------------------------------------------------------------

    def compute_correlation_series(self, prices: pd.DataFrame, target: str) -> pd.Series:
        """
        Rolling Pearson correlation between log-returns of anchor and target.

        corr[t] = pearsonr(anchor_returns[t-W:t], target_returns[t-W:t])

        Returns pd.Series in [-1, 1], NaN for the first corr_window bars.
        """
        log_ret = np.log(prices[[self.anchor, target]]).diff()
        corr = log_ret[self.anchor].rolling(self.corr_window).corr(log_ret[target])
        corr.name = f"corr_{self.anchor}_{target}"
        return corr

    def compute_baseline(self, corr: pd.Series) -> pd.Series:
        """
        Baseline = rolling mean of the correlation series over ma_window bars.

        baseline[t] = corr.rolling(ma_window).mean()
        """
        baseline = corr.rolling(self.ma_window).mean()
        baseline.name = f"baseline_{corr.name}"
        return baseline

    def compute_spread(self, prices: pd.DataFrame, target: str) -> pd.Series:
        """
        Expanding OLS hedge ratio (vectorised via cumulative sums); spread z-score.

        spread = log(anchor) - h * log(target)
        spread_z = (spread - spread.rolling(48).mean()) / spread.rolling(48).std()

        The expanding OLS slope h[t] is estimated analytically via cumulative sums —
        O(n) with no Python loops.

        Returns pd.Series (spread z-score), same index as prices.
        """
        log_anchor = np.log(prices[self.anchor]).values.astype(float)
        log_target = np.log(prices[target]).values.astype(float)
        idx = prices.index

        valid = np.isfinite(log_anchor) & np.isfinite(log_target)
        y = np.where(valid, log_anchor, 0.0)
        x = np.where(valid, log_target, 0.0)
        cnt = np.cumsum(valid.astype(float))
        sum_x = np.cumsum(x)
        sum_y = np.cumsum(y)
        sum_xx = np.cumsum(x * x)
        sum_xy = np.cumsum(x * y)

        with np.errstate(divide="ignore", invalid="ignore"):
            denom = cnt * sum_xx - sum_x ** 2
            hedge = np.where(
                (cnt >= 10) & (denom != 0),
                (cnt * sum_xy - sum_x * sum_y) / denom,
                1.0,
            )

        # Suppress hedge until warm-up window is complete
        min_bars = self.corr_window + self.ma_window
        hedge[:min_bars] = np.nan

        hedge_s = pd.Series(hedge, index=idx).ffill()
        spread_raw = pd.Series(log_anchor, index=idx) - hedge_s * pd.Series(log_target, index=idx)
        spread_mean = spread_raw.rolling(48).mean()
        spread_std = spread_raw.rolling(48).std()
        spread_z = (spread_raw - spread_mean) / spread_std.replace(0, np.nan)
        spread_z.name = f"spread_z_{self.anchor}_{target}"
        return spread_z

    # ------------------------------------------------------------------
    # State machine (sequential by nature — operates on numpy arrays)
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_state_machine(
        corr_arr: np.ndarray,
        baseline_arr: np.ndarray,
        spread_z_arr: np.ndarray,
        threshold: float,
    ) -> np.ndarray:
        """
        Apply the entry/exit state machine on pre-computed arrays.

        Entry: corr < baseline - threshold
          signal = +1 if spread_z < 0, -1 if spread_z > 0
        Exit (cross_ma): corr >= baseline → signal = 0

        Returns int array of signals (same length as inputs).
        """
        n = len(corr_arr)
        signals = np.zeros(n, dtype=np.int8)
        position = 0

        for i in range(n):
            c = corr_arr[i]
            b = baseline_arr[i]
            sz = spread_z_arr[i]

            if np.isnan(c) or np.isnan(b) or np.isnan(sz):
                signals[i] = position
                continue

            if position == 0:
                if c < b - threshold:
                    if sz > 0:
                        position = -1
                    elif sz < 0:
                        position = 1
            else:
                if c >= b:
                    position = 0

            signals[i] = position

        return signals

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_signals(
        self,
        prices: pd.DataFrame,
        target: str,
        threshold: float,
    ) -> pd.DataFrame:
        """
        Generate trading signals for the anchor/target pair.

        Entry: corr < baseline - threshold
          signal = +1 (long anchor/short alt)  when spread_z < 0 (anchor cheap)
          signal = -1 (short anchor/long alt)  when spread_z > 0 (anchor expensive)
        Exit: corr >= baseline (cross_ma mode).

        Returns pd.DataFrame with columns: corr, baseline, spread_z, signal.
        """
        corr = self.compute_correlation_series(prices, target)
        baseline = self.compute_baseline(corr)
        spread_z = self.compute_spread(prices, target)

        signals = self._apply_state_machine(
            corr.values, baseline.values, spread_z.values, threshold
        )

        return pd.DataFrame({
            "corr": corr.values,
            "baseline": baseline.values,
            "spread_z": spread_z.values,
            "signal": signals.astype(int),
        }, index=prices.index)

    def compute_sharpe(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        target: str,
        fee: float = 0.0005,
    ) -> float:
        """
        Compute annualised Sharpe from signals (vectorised — no Python loop).

        PnL per bar:
          signal=+1: anchor_return - target_return
          signal=-1: target_return - anchor_return
          fee charged on bars where signal changes (entry/exit).

        Sharpe = mean(daily_pnl) / std(daily_pnl) * sqrt(365), from daily aggregation.
        Returns 0.0 when all signals are flat or daily std == 0.
        """
        log_ret_anchor = np.log(prices[self.anchor]).diff().values
        log_ret_target = np.log(prices[target]).diff().values
        sig = signals["signal"].values.astype(int)
        idx = prices.index

        # PnL from holding the previous bar's position into the current bar
        prev_sig = np.concatenate([[0], sig[:-1]])
        pnl_hold = np.where(
            prev_sig == 1,
            log_ret_anchor - log_ret_target,
            np.where(prev_sig == -1, log_ret_target - log_ret_anchor, 0.0),
        )

        # Fee on bars where signal changes
        fee_cost = np.where(sig != prev_sig, fee, 0.0)
        bar_pnl = pnl_hold - fee_cost

        pnl_series = pd.Series(bar_pnl, index=idx)
        daily_pnl = pnl_series.resample("D").sum()
        daily_pnl = daily_pnl[daily_pnl.index >= signals.index[0].normalize()]

        std = daily_pnl.std()
        if std == 0 or np.isnan(std):
            return 0.0
        return float(daily_pnl.mean() / std * np.sqrt(365))

    def optimise_threshold(
        self,
        prices: pd.DataFrame,
        target: str,
        train_end: pd.Timestamp,
        threshold_range: Optional[np.ndarray] = None,
    ) -> ThresholdResult:
        """
        Sweep threshold T on the training period and return the best T.

        Optimisation: corr, baseline, spread_z are computed ONCE; only the
        state machine is re-run for each threshold value in the sweep.

        Parameters
        ----------
        prices : pd.DataFrame
            Full price history (train + test).
        target : str
            Altcoin symbol.
        train_end : pd.Timestamp
            Last timestamp (inclusive) of the training window.
        threshold_range : np.ndarray or None
            Thresholds to sweep. Defaults to np.arange(0.0, 0.31, 0.01).

        Returns
        -------
        ThresholdResult
        """
        if threshold_range is None:
            threshold_range = np.arange(0.0, 0.31, 0.01)

        train_prices = prices[prices.index <= train_end]

        # Pre-compute rolling features once (expensive pandas rolling ops)
        corr = self.compute_correlation_series(train_prices, target)
        baseline = self.compute_baseline(corr)
        spread_z = self.compute_spread(train_prices, target)

        corr_arr = corr.values
        baseline_arr = baseline.values
        spread_z_arr = spread_z.values

        # Pre-compute log returns for Sharpe calculation
        log_ret_anchor = np.log(train_prices[self.anchor]).diff().values
        log_ret_target = np.log(train_prices[target]).diff().values
        train_idx = train_prices.index

        rows = []
        best_threshold = 0.0
        best_sharpe = -np.inf

        for t in threshold_range:
            try:
                sig_arr = self._apply_state_machine(corr_arr, baseline_arr, spread_z_arr, float(t))

                # Inline Sharpe computation (avoids rebuilding DataFrames)
                prev_sig = np.concatenate([[0], sig_arr[:-1]])
                pnl_hold = np.where(
                    prev_sig == 1,
                    log_ret_anchor - log_ret_target,
                    np.where(prev_sig == -1, log_ret_target - log_ret_anchor, 0.0),
                )
                fee_cost = np.where(sig_arr != prev_sig, 0.0005, 0.0)
                bar_pnl = pnl_hold - fee_cost

                pnl_series = pd.Series(bar_pnl, index=train_idx)
                daily_pnl = pnl_series.resample("D").sum()

                std = daily_pnl.std()
                sharpe = float(daily_pnl.mean() / std * np.sqrt(365)) if (std > 0 and not np.isnan(std)) else 0.0
            except Exception as exc:
                logger.debug("threshold=%.3f failed: %s", t, exc)
                sharpe = 0.0

            rows.append({"threshold": float(t), "train_sharpe": sharpe})
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_threshold = float(t)

        sweep_df = pd.DataFrame(rows)
        return ThresholdResult(
            best_threshold=best_threshold,
            train_sharpe=best_sharpe,
            sweep=sweep_df,
        )
