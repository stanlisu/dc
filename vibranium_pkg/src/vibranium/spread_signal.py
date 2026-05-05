"""
vibranium/signal.py — Signal generation for the Vibranium stationary portfolio strategy.

Each bar:
  1. Compute S_t = Σ w_i · log(P_i,t)
  2. Detrend: S̃_t = S_t - α̂·t  (remove drift estimated on lookback)
  3. Z-score using OU equilibrium σ_eq = σ / sqrt(2κ)
  4. Entry/exit rules on z-score
"""
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from vibranium.ou_estimator import OUParams, spread_zscore

logger = logging.getLogger(__name__)


@dataclass
class SignalConfig:
    """Signal generation parameters."""
    entry_z: float = 2.0        # enter when |z| > entry_z
    exit_z: float = 0.5         # exit when |z| < exit_z
    max_halflife: float = 500.0  # skip if OU half-life exceeds this (bars)


def generate_signals(
    log_prices: pd.DataFrame,
    weights: np.ndarray,
    symbols: list,
    ou: OUParams,
    alpha_per_bar: float,
    intercept: float,
    config: SignalConfig,
    lookback_start_idx: int = 0,
) -> pd.DataFrame:
    """
    Generate trading signals from portfolio spread z-score.

    Parameters
    ----------
    log_prices : pd.DataFrame
        Log-prices for all symbols.
    weights : np.ndarray
        Portfolio weights (Σ|w| = 1).
    symbols : list[str]
        Symbols corresponding to weights.
    ou : OUParams
        Fitted OU parameters.
    alpha_per_bar : float
        Drift rate per bar.
    intercept : float
        Trend intercept from the lookback window fit.
    config : SignalConfig
        Entry/exit thresholds.
    lookback_start_idx : int
        Index offset for detrending (bar 0 of lookback = t=0 for drift).

    Returns
    -------
    pd.DataFrame
        Columns: spread, spread_detrended, z_score, signal
    """
    # Compute raw spread
    avail = [s for s in symbols if s in log_prices.columns]
    if len(avail) < 2:
        return _empty_signals(log_prices.index)

    w_idx = [symbols.index(s) for s in avail]
    w_sub = np.array([weights[i] for i in w_idx])
    norm = np.sum(np.abs(w_sub))
    if norm > 0:
        w_sub = w_sub / norm

    spread = log_prices[avail].dot(w_sub)

    # Detrend using the same trend line from the lookback fit:
    # trend(t) = intercept + alpha * t, where t is relative to lookback start
    n = len(spread)
    t = np.arange(n, dtype=float) + lookback_start_idx
    trend = intercept + alpha_per_bar * t
    spread_detrended = spread - trend

    # Z-score using OU params
    z = spread_zscore(spread_detrended, ou)

    # Half-life kill switch
    if ou.half_life > config.max_halflife or ou.kappa <= 0:
        return pd.DataFrame({
            "spread": spread,
            "spread_detrended": spread_detrended,
            "z_score": z,
            "signal": 0,
        }, index=log_prices.index)

    # State machine: entry/exit
    signals = _apply_state_machine(z.values, config.entry_z, config.exit_z)

    return pd.DataFrame({
        "spread": spread,
        "spread_detrended": spread_detrended,
        "z_score": z,
        "signal": signals,
    }, index=log_prices.index)


def _apply_state_machine(
    z_arr: np.ndarray,
    entry_z: float,
    exit_z: float,
) -> np.ndarray:
    """
    Entry/exit state machine on z-score array.

    +1 (long portfolio)  when z < -entry_z  (spread below trend)
    -1 (short portfolio) when z > +entry_z  (spread above trend)
     0 (flat)           when |z| < exit_z
    """
    n = len(z_arr)
    signals = np.zeros(n, dtype=np.int8)
    position = 0

    for i in range(n):
        z = z_arr[i]
        if np.isnan(z):
            signals[i] = position
            continue

        if position == 0:
            if z < -entry_z:
                position = 1   # long: spread is cheap
            elif z > entry_z:
                position = -1  # short: spread is expensive
        else:
            if abs(z) < exit_z:
                position = 0   # exit

        signals[i] = position

    return signals


def _empty_signals(index: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(index)
    return pd.DataFrame({
        "spread": np.full(n, np.nan),
        "spread_detrended": np.full(n, np.nan),
        "z_score": np.full(n, np.nan),
        "signal": np.zeros(n, dtype=int),
    }, index=index)
