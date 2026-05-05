"""
vibranium/ou_estimator.py — Ornstein-Uhlenbeck parameter estimation and spread decomposition.

Given a spread S_t (portfolio log-value), fit:
    dS = κ(μ - S)dt + σ dW

Parameters estimated via OLS:  ΔS = a + b·S + ε
    κ = -b / Δt
    μ = -a / b
    σ = std(ε) / sqrt(Δt)
    half_life = ln(2) / κ
"""
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OUParams:
    """Fitted Ornstein-Uhlenbeck parameters."""
    kappa: float        # mean-reversion speed (higher = faster)
    mu: float           # long-run mean
    sigma: float        # volatility of the OU process
    half_life: float    # bars until half of deviation is absorbed
    alpha: float        # drift rate of the portfolio (per bar)
    adf_stat: float     # ADF test statistic (more negative = more stationary)
    intercept: float = 0.0  # trend intercept from detrending fit


def fit_ou(spread: np.ndarray, dt: float = 1.0) -> OUParams:
    """
    Fit OU parameters via OLS regression on ΔS vs S.

    Parameters
    ----------
    spread : np.ndarray
        The stationary spread (S_t) series.
    dt : float
        Time step in bars (default 1.0 = per bar).

    Returns
    -------
    OUParams
    """
    spread = np.asarray(spread, dtype=float)
    # Remove NaN
    mask = np.isfinite(spread[:-1]) & np.isfinite(spread[1:])
    s = spread[:-1][mask]
    ds = np.diff(spread)[mask]

    if len(s) < 20:
        return OUParams(kappa=0.0, mu=0.0, sigma=0.0,
                        half_life=np.inf, alpha=0.0, adf_stat=0.0)

    # OLS: ΔS = a + b·S + ε
    X = np.column_stack([np.ones(len(s)), s])
    beta, residuals, _, _ = np.linalg.lstsq(X, ds, rcond=None)
    a, b = beta[0], beta[1]

    eps = ds - X @ beta
    sigma_eps = float(np.std(eps))

    # OU params
    kappa = -b / dt
    if kappa <= 0:
        # Not mean-reverting
        return OUParams(kappa=kappa, mu=0.0, sigma=sigma_eps / np.sqrt(dt),
                        half_life=np.inf, alpha=float(a / dt), adf_stat=b)

    mu = -a / b
    sigma = sigma_eps / np.sqrt(dt)
    half_life = np.log(2) / kappa

    return OUParams(
        kappa=kappa,
        mu=mu,
        sigma=sigma,
        half_life=half_life,
        alpha=float(a / dt),
        adf_stat=b,
    )


def decompose_spread(
    log_prices: pd.DataFrame,
    weights: np.ndarray,
    symbols: list,
) -> pd.Series:
    """
    Compute portfolio spread: S_t = Σ w_i · log(P_i,t).

    Parameters
    ----------
    log_prices : pd.DataFrame
        Log-prices with columns for each symbol.
    weights : np.ndarray
        Portfolio weights (Σ|w| = 1).
    symbols : list[str]
        Symbols corresponding to the weights.

    Returns
    -------
    pd.Series
        The spread series, indexed by log_prices.index.
    """
    return log_prices[symbols].dot(weights)


def detrend_spread(spread: pd.Series) -> tuple:
    """
    Remove linear drift from spread: S_t = intercept + α·t + S̃_t.

    Returns
    -------
    (detrended: pd.Series, alpha_per_bar: float, intercept: float)
        Detrended spread, drift rate per bar, and intercept.
    """
    vals = spread.values
    n = len(vals)
    t = np.arange(n, dtype=float)

    mask = np.isfinite(vals)
    if mask.sum() < 20:
        return spread, 0.0, 0.0

    # OLS: S = a + α·t
    X = np.column_stack([np.ones(mask.sum()), t[mask]])
    beta, _, _, _ = np.linalg.lstsq(X, vals[mask], rcond=None)
    intercept, alpha = beta[0], beta[1]

    trend = intercept + alpha * t
    detrended = pd.Series(vals - trend, index=spread.index)

    return detrended, float(alpha), float(intercept)


def spread_zscore(
    spread: pd.Series,
    ou: OUParams,
) -> pd.Series:
    """
    Z-score the spread using OU parameters: z = (S - μ) / σ_eq.

    σ_eq = σ / sqrt(2κ) is the equilibrium standard deviation of the OU process.
    """
    if ou.kappa <= 0 or ou.sigma <= 0:
        # Fallback to rolling z-score
        mu = spread.rolling(96).mean()
        std = spread.rolling(96).std()
        return (spread - mu) / std.replace(0, np.nan)

    sigma_eq = ou.sigma / np.sqrt(2 * ou.kappa)
    if sigma_eq <= 0:
        return pd.Series(np.zeros(len(spread)), index=spread.index)

    return (spread - ou.mu) / sigma_eq
