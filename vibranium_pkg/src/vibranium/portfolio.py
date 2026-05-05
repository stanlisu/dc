"""
vibranium/portfolio.py — Cointegration-based stationary portfolio construction and backtesting.

Key classes:
  CointResult       — result from an Engle-Granger pair test
  PortfolioWeights  — fitted weights + stationarity metadata
  BacktestResult    — PnL metrics
  StationaryPortfolio — main engine: fit, signal, backtest
"""
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CointResult:
    """Engle-Granger cointegration result for a pair of assets."""
    asset_a: str
    asset_b: str
    pvalue: float
    hedge_ratio: float          # OLS coefficient: asset_a ~ hedge_ratio * asset_b
    adf_pvalue: float           # ADF on the residual spread


@dataclass
class PortfolioWeights:
    """Weights vector defining the stationary portfolio."""
    symbols: list               # ordered list of symbols
    weights: np.ndarray         # weight for each symbol (sum(|w|) == 1)
    adf_pvalue: float           # ADF p-value of the spread on the fit window
    method: str                 # "johansen" or "pairs"
    fit_date: Optional[str] = None   # last date of the lookback window used

    def to_dict(self) -> dict:
        return {sym: float(w) for sym, w in zip(self.symbols, self.weights)}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class BacktestResult:
    """Summary statistics from a walk-forward backtest."""
    daily_pnl: pd.Series                    # date-indexed daily PnL
    sharpe: float
    max_drawdown: float
    avg_holding_days: float
    total_return: float
    per_year: pd.DataFrame = field(default_factory=pd.DataFrame)  # yearly stats


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _log_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Return log-price DataFrame (log of close prices)."""
    return np.log(prices.replace(0, np.nan))


def _compute_spread(log_prices: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Compute portfolio spread: sum(w_i * log(P_i)) for each date."""
    wv = np.asarray(weights)
    return log_prices.dot(wv)


def _zscore(spread: pd.Series, mean: float, std: float) -> pd.Series:
    """Standardise a spread given pre-computed mean and std."""
    if std == 0:
        return pd.Series(np.zeros(len(spread)), index=spread.index)
    return (spread - mean) / std


def _adf_pvalue(series: pd.Series) -> float:
    """Return ADF p-value for stationarity of *series*."""
    clean = series.dropna()
    if len(clean) < 20:
        return 1.0
    try:
        result = adfuller(clean, autolag="AIC", regression="c")
        return float(result[1])
    except Exception:
        return 1.0


def _ols_hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    """OLS hedge ratio: y ~ beta * x (no intercept term in the ratio)."""
    # Use np.polyfit for speed
    mask = y.notna() & x.notna()
    if mask.sum() < 10:
        return 1.0
    coeffs = np.polyfit(x[mask], y[mask], 1)
    return float(coeffs[0])


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class StationaryPortfolio:
    """
    Cointegration-based stationary portfolio for liquid crypto perpetuals.

    Parameters
    ----------
    lookback_days : int
        Number of calendar days used to estimate portfolio weights.
    n_assets : int
        Maximum number of assets in the Johansen basket.
    entry_z : float
        Z-score threshold to enter a position (|z| > entry_z → trade).
    exit_z : float
        Z-score threshold to exit a position (|z| < exit_z while in position).
    rebalance_days : int
        How many days between weight re-estimations.
    """

    def __init__(
        self,
        lookback_days: int = 90,
        n_assets: int = 5,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        rebalance_days: int = 7,
    ):
        self.lookback_days = lookback_days
        self.n_assets = n_assets
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.rebalance_days = rebalance_days

    # ------------------------------------------------------------------
    # Cointegration testing
    # ------------------------------------------------------------------

    def find_cointegrated_pairs(self, prices: pd.DataFrame) -> list:
        """
        Run Engle-Granger cointegration test on all pairs.

        Returns
        -------
        list[CointResult]
            Sorted by p-value ascending (most stationary first).
        """
        log_p = _log_prices(prices).dropna()
        symbols = list(log_p.columns)
        results = []

        for i, a in enumerate(symbols):
            for b in symbols[i + 1:]:
                ya, yb = log_p[a], log_p[b]
                try:
                    score, pval, _ = coint(ya, yb, autolag="AIC")
                    hedge = _ols_hedge_ratio(ya, yb)
                    residual = ya - hedge * yb
                    adf_p = _adf_pvalue(residual)
                    results.append(CointResult(
                        asset_a=a, asset_b=b,
                        pvalue=float(pval),
                        hedge_ratio=hedge,
                        adf_pvalue=adf_p,
                    ))
                except Exception as exc:
                    logger.debug("coint(%s, %s) failed: %s", a, b, exc)

        results.sort(key=lambda r: r.pvalue)
        return results

    # ------------------------------------------------------------------
    # Portfolio fitting
    # ------------------------------------------------------------------

    def fit_portfolio(self, prices: pd.DataFrame, method: str = "johansen") -> PortfolioWeights:
        """
        Fit portfolio weights that produce a stationary spread.

        Parameters
        ----------
        prices : pd.DataFrame
            Log-prices aligned on a common date index (lookback window only).
        method : str
            "johansen" — Johansen trace test on up to n_assets assets.
            "pairs"    — OLS hedge ratio for the best Engle-Granger pair.

        Returns
        -------
        PortfolioWeights
            Weights normalised so sum(|weights|) == 1.
        """
        log_p = _log_prices(prices).dropna()
        fit_date = str(log_p.index.max().date()) if not log_p.empty else None

        if method == "johansen":
            return self._fit_johansen(log_p, fit_date)
        elif method == "pairs":
            return self._fit_pairs(log_p, fit_date)
        else:
            raise ValueError(f"Unknown method: {method!r}. Use 'johansen' or 'pairs'.")

    def _fit_johansen(self, log_p: pd.DataFrame, fit_date: str) -> PortfolioWeights:
        """Johansen cointegration — returns eigenvector of the first cointegrating relation."""
        symbols = list(log_p.columns)
        # Limit to n_assets by variance-explained heuristic (take most volatile)
        if len(symbols) > self.n_assets:
            vols = log_p.std()
            symbols = vols.nlargest(self.n_assets).index.tolist()
            log_p = log_p[symbols]

        data = log_p.values
        if data.shape[0] < 20 or data.shape[1] < 2:
            logger.warning("Johansen: not enough data (%d rows, %d cols)", data.shape[0], data.shape[1])
            # Fallback: equal weights
            w = np.ones(len(symbols)) / len(symbols)
            return PortfolioWeights(symbols=symbols, weights=w, adf_pvalue=1.0, method="johansen", fit_date=fit_date)

        try:
            res = coint_johansen(data, det_order=0, k_ar_diff=1)
        except Exception as exc:
            logger.warning("Johansen test failed: %s. Falling back to pairs.", exc)
            return self._fit_pairs(log_p, fit_date)

        # evec columns correspond to cointegrating vectors; first column = max eigenvalue vector
        evec = res.evec[:, 0]

        # Normalise so sum(|w|) == 1
        norm = np.sum(np.abs(evec))
        if norm == 0:
            evec = np.ones(len(symbols)) / len(symbols)
        else:
            evec = evec / norm

        spread = _compute_spread(log_p, evec)
        adf_p = _adf_pvalue(spread)
        logger.debug("Johansen portfolio ADF p=%.4f  symbols=%s", adf_p, symbols)

        return PortfolioWeights(
            symbols=symbols,
            weights=evec,
            adf_pvalue=adf_p,
            method="johansen",
            fit_date=fit_date,
        )

    def _fit_pairs(self, log_p: pd.DataFrame, fit_date: str) -> PortfolioWeights:
        """Find the best cointegrated pair and return [1, -hedge_ratio] weights."""
        pairs = self.find_cointegrated_pairs(log_p)
        if not pairs:
            logger.warning("No cointegrated pairs found; using first two columns.")
            symbols = list(log_p.columns[:2])
            w = np.array([0.5, -0.5])
            return PortfolioWeights(symbols=symbols, weights=w, adf_pvalue=1.0, method="pairs", fit_date=fit_date)

        best = pairs[0]
        symbols = [best.asset_a, best.asset_b]
        h = best.hedge_ratio
        raw = np.array([1.0, -h])
        norm = np.sum(np.abs(raw))
        weights = raw / norm if norm > 0 else raw

        spread = _compute_spread(log_p[symbols], weights)
        adf_p = _adf_pvalue(spread)

        return PortfolioWeights(
            symbols=symbols,
            weights=weights,
            adf_pvalue=adf_p,
            method="pairs",
            fit_date=fit_date,
        )

    # ------------------------------------------------------------------
    # Spread computation
    # ------------------------------------------------------------------

    def compute_spread(self, prices: pd.DataFrame, pw: PortfolioWeights) -> pd.Series:
        """
        Compute the portfolio spread z-score.

        Parameters
        ----------
        prices : pd.DataFrame
            Full price history (columns must include pw.symbols).
        pw : PortfolioWeights
            Fitted weights from fit_portfolio().

        Returns
        -------
        pd.Series
            Z-score of the spread (mean/std computed over the entire series).
        """
        log_p = _log_prices(prices[pw.symbols])
        spread = _compute_spread(log_p, pw.weights)
        mean = spread.mean()
        std = spread.std()
        return _zscore(spread, mean, std)

    # ------------------------------------------------------------------
    # Signal generation (walk-forward)
    # ------------------------------------------------------------------

    def _reindex_weights(self, pw: PortfolioWeights, avail: list):
        """Return sub-weights for *avail* symbols, normalised so sum(|w|)==1."""
        idx_map = {s: j for j, s in enumerate(pw.symbols)}
        sub_w = np.array([pw.weights[idx_map[s]] for s in avail])
        norm = np.sum(np.abs(sub_w))
        return sub_w / norm if norm > 0 else sub_w

    def _update_signal_state(self, position: int, z: float) -> tuple:
        """Apply entry/exit rules and return (new_position, signal)."""
        if position == 0:
            if z < -self.entry_z:
                position = 1
            elif z > self.entry_z:
                position = -1
        else:
            if abs(z) < self.exit_z:
                position = 0
        return position, position

    def _null_record(self, date, weights_json: str = "{}") -> dict:
        """Return a flat record with NaN spread/z and zero signal."""
        return {"date": date, "spread": np.nan, "z_score": np.nan, "signal": 0, "weights_json": weights_json}

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling walk-forward signal generation.

        Every *rebalance_days* calendar days, re-fit the portfolio on the past
        *lookback_days* rows.  Compute the z-score of the spread and generate
        entry/exit signals.

        Signal convention
        -----------------
        +1  "buy spread"   → spread is cheap; buy the portfolio as specified by weights
        -1  "sell spread"  → spread is expensive; sell the portfolio
         0  flat / exit

        Returns
        -------
        pd.DataFrame
            Columns: spread, z_score, signal, weights_json
        """
        log_p = _log_prices(prices)
        dates = log_p.index
        n = len(dates)

        records = []
        current_weights: Optional[PortfolioWeights] = None
        last_rebalance_idx = -self.rebalance_days  # force fit on first bar
        position = 0  # current position: +1, -1, or 0

        for i in range(self.lookback_days, n):
            date = dates[i]

            # Re-fit if rebalance window has elapsed
            if i - last_rebalance_idx >= self.rebalance_days:
                window = prices.iloc[i - self.lookback_days: i]
                valid_cols = window.columns[window.notna().all()].tolist()
                if len(valid_cols) >= 2:
                    current_weights = self.fit_portfolio(window[valid_cols], method="johansen")
                    last_rebalance_idx = i

            if current_weights is None:
                records.append(self._null_record(date))
                continue

            # Compute spread for this day using lookback mean/std
            lookback_slice = log_p.iloc[i - self.lookback_days: i]
            avail = [s for s in current_weights.symbols if s in lookback_slice.columns]
            if len(avail) < 2:
                records.append(self._null_record(date, current_weights.to_json()))
                continue

            sub_weights = self._reindex_weights(current_weights, avail)
            spread_history = lookback_slice[avail].dot(sub_weights)
            today_log = log_p.iloc[i][avail]
            if today_log.isna().any():
                records.append(self._null_record(date, current_weights.to_json()))
                continue

            today_spread = float(today_log.dot(sub_weights))
            z = _zscore(pd.Series([today_spread]), spread_history.mean(), spread_history.std()).iloc[0]
            position, signal = self._update_signal_state(position, z)
            w_dict = {s: float(sub_weights[j]) for j, s in enumerate(avail)}
            records.append({
                "date": date,
                "spread": today_spread,
                "z_score": float(z),
                "signal": signal,
                "weights_json": json.dumps(w_dict),
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.set_index("date")
        return df

    # ------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------

    def backtest(self, prices: pd.DataFrame, signals: pd.DataFrame) -> BacktestResult:
        """
        Simulate PnL from signals.

        Position sizing: equal notional (1 unit of the portfolio value) per leg.
        Transaction cost: 0.05% per side (maker fee) applied at each trade.

        Parameters
        ----------
        prices : pd.DataFrame
            Full price history aligned with signals index.
        signals : pd.DataFrame
            Output from generate_signals().

        Returns
        -------
        BacktestResult
        """
        fee_rate = 0.0005  # 0.05% per trade

        log_p = _log_prices(prices)
        # Align on signals index
        log_p = log_p.reindex(signals.index)

        prev_signal = 0
        prev_weights: Optional[dict] = None
        prev_log_p: Optional[pd.Series] = None
        daily_pnl = []

        for date, row in signals.iterrows():
            signal = int(row["signal"])
            try:
                weights_dict = json.loads(row["weights_json"])
            except Exception:
                weights_dict = {}

            if not weights_dict:
                daily_pnl.append({"date": date, "pnl": 0.0})
                prev_signal = signal
                prev_weights = weights_dict
                prev_log_p = log_p.loc[date] if date in log_p.index else None
                continue

            current_log_p = log_p.loc[date] if date in log_p.index else None

            # PnL from holding previous position
            pnl = 0.0
            if prev_signal != 0 and prev_weights and prev_log_p is not None and current_log_p is not None:
                for sym, w in prev_weights.items():
                    if sym in prev_log_p.index and sym in current_log_p.index:
                        log_ret = current_log_p[sym] - prev_log_p[sym]
                        pnl += prev_signal * w * log_ret

            # Transaction cost: charged when signal changes
            tc = 0.0
            if signal != prev_signal and weights_dict:
                n_legs = len(weights_dict)
                tc = fee_rate * n_legs  # rough: one leg per symbol traded

            daily_pnl.append({"date": date, "pnl": pnl - tc})

            prev_signal = signal
            prev_weights = weights_dict
            prev_log_p = current_log_p

        pnl_df = pd.DataFrame(daily_pnl).set_index("date")["pnl"]
        pnl_df = pnl_df.fillna(0.0)

        # Sharpe (annualised, daily returns)
        std = pnl_df.std()
        sharpe = (pnl_df.mean() / std * np.sqrt(252)) if std > 0 else 0.0

        # Max drawdown
        cum = pnl_df.cumsum()
        roll_max = cum.cummax()
        dd = (cum - roll_max)
        max_dd = float(dd.min())

        # Average holding period
        position_changes = signals["signal"].diff().fillna(0).abs()
        n_trades = int((position_changes > 0).sum())
        in_position = (signals["signal"] != 0).sum()
        avg_hold = float(in_position / n_trades) if n_trades > 0 else 0.0

        total_return = float(pnl_df.sum())

        # Per-year breakdown
        yearly = pnl_df.groupby(pnl_df.index.year).agg(
            total_pnl="sum",
            sharpe=lambda x: (x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0.0,
        )
        # Max drawdown per year

        def _max_dd(x):
            c = x.cumsum()
            return float((c - c.cummax()).min())
        yearly["max_drawdown"] = pnl_df.groupby(pnl_df.index.year).apply(_max_dd)

        return BacktestResult(
            daily_pnl=pnl_df,
            sharpe=float(sharpe),
            max_drawdown=max_dd,
            avg_holding_days=avg_hold,
            total_return=total_return,
            per_year=yearly,
        )
