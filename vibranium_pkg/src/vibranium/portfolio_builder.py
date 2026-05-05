"""
vibranium/portfolio_builder.py — Hybrid Johansen-seeded optimization for stationary portfolios.

Pipeline:
  1. Pre-screen symbols by pairwise cointegration score → top n_assets
  2. Johansen eigenvectors as seed candidates
  3. Differential evolution to maximize κ/σ subject to α>0, |w_i|<max_weight, Σ|w|=1

The result is a weight vector whose portfolio spread is stationary with positive drift.
"""
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from vibranium.ou_estimator import OUParams, fit_ou, decompose_spread, detrend_spread

logger = logging.getLogger(__name__)


@dataclass
class PortfolioResult:
    """Result from portfolio construction."""
    symbols: list           # ordered symbol names
    weights: np.ndarray     # portfolio weights (Σ|w| = 1)
    ou: OUParams            # OU params of the detrended spread
    alpha_per_bar: float    # drift rate per bar
    intercept: float        # trend intercept from detrending fit
    method: str             # "johansen", "optimized", or "johansen_only"
    fit_date: Optional[str] = None

    def to_dict(self) -> dict:
        return {s: float(w) for s, w in zip(self.symbols, self.weights)}


def _log_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices.replace(0, np.nan))


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    """Normalize so Σ|w| = 1."""
    norm = np.sum(np.abs(w))
    return w / norm if norm > 0 else w


# ---------------------------------------------------------------------------
# Step 1: Pre-screen symbols
# ---------------------------------------------------------------------------

def prescreen_symbols(
    log_prices: pd.DataFrame,
    n_assets: int = 8,
) -> list:
    """
    Select top n_assets symbols by average pairwise cointegration score.

    For each symbol, compute average -log(pvalue) across all pairs.
    Higher score = more cointegrated with the universe.
    """
    symbols = list(log_prices.columns)
    if len(symbols) <= n_assets:
        return symbols

    # Sample pairs to keep it fast (all pairs if n <= 15, else random subset)
    n = len(symbols)
    scores = {s: 0.0 for s in symbols}
    count = {s: 0 for s in symbols}

    for i in range(n):
        for j in range(i + 1, n):
            a, b = symbols[i], symbols[j]
            ya, yb = log_prices[a].dropna(), log_prices[b].dropna()
            common = ya.index.intersection(yb.index)
            if len(common) < 100:
                continue
            try:
                # Use fixed maxlag for speed (AIC is O(n*maxlag) and very slow)
                maxlag = min(20, len(common) // 20)
                _, pval, _ = coint(ya.loc[common].values, yb.loc[common].values,
                                   maxlag=maxlag, autolag=None)
                score = -np.log(max(pval, 1e-10))
                scores[a] += score
                scores[b] += score
                count[a] += 1
                count[b] += 1
            except Exception:
                pass

    # Average score
    avg_scores = {}
    for s in symbols:
        if count[s] > 0:
            avg_scores[s] = scores[s] / count[s]
        else:
            avg_scores[s] = 0.0

    ranked = sorted(avg_scores, key=avg_scores.get, reverse=True)
    selected = ranked[:n_assets]
    logger.info("Pre-screened %d → %d symbols: %s", len(symbols), n_assets, selected)
    return selected


# ---------------------------------------------------------------------------
# Step 2: Johansen eigenvectors
# ---------------------------------------------------------------------------

def johansen_seeds(
    log_prices: pd.DataFrame,
    max_vectors: int = 3,
) -> list:
    """
    Run Johansen cointegration test and return eigenvectors as seed weight vectors.

    Returns list of (weights, eigenvalue) tuples, sorted by eigenvalue descending.
    """
    data = log_prices.dropna().values
    if data.shape[0] < 30 or data.shape[1] < 2:
        logger.warning("Insufficient data for Johansen: %s", data.shape)
        return []

    try:
        res = coint_johansen(data, det_order=0, k_ar_diff=1)
    except Exception as exc:
        logger.warning("Johansen failed: %s", exc)
        return []

    seeds = []
    n_vectors = min(max_vectors, res.evec.shape[1])
    eigenvalues = res.lr1[:, 0] if res.lr1.ndim > 1 else res.eig

    for i in range(n_vectors):
        w = res.evec[:, i].copy()
        w = _normalize_weights(w)
        ev = float(eigenvalues[i]) if i < len(eigenvalues) else 0.0
        seeds.append((w, ev))

    seeds.sort(key=lambda x: x[1], reverse=True)
    return seeds


# ---------------------------------------------------------------------------
# Step 3: Optimization
# ---------------------------------------------------------------------------

def _objective(
    w_raw: np.ndarray,
    log_prices_arr: np.ndarray,
    mean_returns: np.ndarray,
    l2_lambda: float,
) -> float:
    """
    Objective for differential_evolution: minimize negative κ/σ.

    w_raw is unconstrained; we normalize internally.
    Penalize: negative drift, very slow reversion, extreme weights.
    """
    w = w_raw / np.sum(np.abs(w_raw)) if np.sum(np.abs(w_raw)) > 0 else w_raw

    # Portfolio spread
    spread = log_prices_arr @ w

    # Detrend
    n = len(spread)
    t = np.arange(n, dtype=float)
    mask = np.isfinite(spread)
    if mask.sum() < 50:
        return 1e6

    X = np.column_stack([np.ones(mask.sum()), t[mask]])
    beta, _, _, _ = np.linalg.lstsq(X, spread[mask], rcond=None)
    alpha = beta[1]  # drift per bar

    # Drift penalty: we want α > 0
    drift_penalty = 0.0
    if alpha <= 0:
        drift_penalty = 100.0 * abs(alpha)

    # Detrended spread
    trend = beta[0] + alpha * t
    s_detrended = spread - trend
    s_clean = s_detrended[mask]

    # OU fit on detrended spread
    ds = np.diff(s_clean)
    s_lag = s_clean[:-1]
    if len(s_lag) < 30:
        return 1e6

    X_ou = np.column_stack([np.ones(len(s_lag)), s_lag])
    beta_ou, _, _, _ = np.linalg.lstsq(X_ou, ds, rcond=None)
    b = beta_ou[1]

    kappa = -b  # per bar
    if kappa <= 0:
        return 1e6  # not mean-reverting

    eps = ds - X_ou @ beta_ou
    sigma = float(np.std(eps))
    if sigma <= 0:
        return 1e6

    # Objective: maximize κ/σ → minimize -κ/σ
    score = kappa / sigma

    # L2 regularization on weights
    l2_pen = l2_lambda * np.sum(w ** 2)

    return -score + drift_penalty + l2_pen


def optimize_weights(
    log_prices: pd.DataFrame,
    seed_weights: Optional[np.ndarray] = None,
    max_weight: float = 0.25,
    l2_lambda: float = 0.1,
    maxiter: int = 300,
    popsize: int = 25,
) -> tuple:
    """
    Optimize portfolio weights via differential evolution to maximize κ/σ.

    Parameters
    ----------
    log_prices : pd.DataFrame
        Log-prices for the selected symbols.
    seed_weights : np.ndarray or None
        Johansen eigenvector as initial seed.
    max_weight : float
        Maximum absolute weight per symbol.
    l2_lambda : float
        L2 regularization strength.
    maxiter : int
        Max iterations for differential evolution.
    popsize : int
        Population size multiplier.

    Returns
    -------
    (weights: np.ndarray, success: bool)
    """
    n_assets = log_prices.shape[1]
    arr = log_prices.dropna().values

    # Mean returns for drift check
    returns = np.diff(arr, axis=0)
    mean_returns = np.mean(returns, axis=0)

    bounds = [(-max_weight, max_weight)] * n_assets

    # Clip seed to bounds so x0 is feasible
    x0 = None
    if seed_weights is not None:
        x0 = _normalize_weights(seed_weights)
        x0 = np.clip(x0, -max_weight, max_weight)
        # Re-normalize after clipping
        x0 = _normalize_weights(x0)
        x0 = np.clip(x0, -max_weight, max_weight)

    try:
        result = differential_evolution(
            _objective,
            bounds=bounds,
            args=(arr, mean_returns, l2_lambda),
            maxiter=maxiter,
            popsize=popsize,
            seed=42,
            tol=1e-6,
            x0=x0,
        )
        w_opt = _normalize_weights(result.x)
        return w_opt, result.success
    except Exception as exc:
        logger.warning("Optimization failed: %s", exc)
        if seed_weights is not None:
            return seed_weights, False
        return np.ones(n_assets) / n_assets, False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_portfolio(
    prices: pd.DataFrame,
    n_assets: int = 8,
    max_weight: float = 0.25,
    l2_lambda: float = 0.1,
    optimize: bool = True,
    max_halflife: float = 500.0,
) -> Optional[PortfolioResult]:
    """
    Full pipeline: prescreen → Johansen seeds → optimize → validate.

    Parameters
    ----------
    prices : pd.DataFrame
        Close prices (not log) for the full symbol universe.
    n_assets : int
        Number of symbols in the portfolio.
    max_weight : float
        Max absolute weight per symbol.
    l2_lambda : float
        L2 regularization on weights.
    optimize : bool
        If True, refine Johansen seed via differential evolution.
        If False, use best Johansen eigenvector directly.
    max_halflife : float
        Maximum OU half-life in bars. Reject portfolios above this.

    Returns
    -------
    PortfolioResult or None if no valid portfolio found.
    """
    log_p = _log_prices(prices).dropna()
    if log_p.shape[0] < 100 or log_p.shape[1] < 2:
        logger.warning("Insufficient data: %d rows x %d cols", log_p.shape[0], log_p.shape[1])
        return None

    fit_date = str(log_p.index.max().date()) if not log_p.empty else None

    # Step 1: prescreen
    selected = prescreen_symbols(log_p, n_assets=n_assets)
    log_p = log_p[selected]

    # Step 2: Johansen seeds
    seeds = johansen_seeds(log_p)
    if not seeds:
        logger.warning("No Johansen seeds found")
        return None

    best_result = None
    best_score = -np.inf

    for seed_w, eigenval in seeds:
        if optimize:
            # Step 3: optimize
            w_opt, success = optimize_weights(
                log_p, seed_weights=seed_w,
                max_weight=max_weight, l2_lambda=l2_lambda,
            )
            method = "optimized"
        else:
            w_opt = seed_w
            method = "johansen_only"

        # Evaluate
        spread = decompose_spread(log_p, w_opt, selected)
        detrended, alpha, intercept = detrend_spread(spread)
        ou = fit_ou(detrended.values)
        ou.intercept = intercept

        # Validate
        if ou.kappa <= 0:
            continue
        if ou.half_life > max_halflife:
            logger.debug("Half-life %.1f > max %.1f, skipping", ou.half_life, max_halflife)
            continue

        score = ou.kappa / ou.sigma if ou.sigma > 0 else 0.0
        if alpha > 0:
            score *= 1.5  # bonus for positive drift

        if score > best_score:
            best_score = score
            best_result = PortfolioResult(
                symbols=selected,
                weights=w_opt,
                ou=ou,
                alpha_per_bar=alpha,
                intercept=intercept,
                method=method,
                fit_date=fit_date,
            )

    if best_result is None:
        logger.warning("No valid portfolio found (all half-lives too long or κ <= 0)")

    return best_result
