"""
vibranium/pair_selector.py — Walk-forward pair selection for Vibranium.

Tracks pair metrics across consecutive rebalance windows. Selects pairs
that pass stability filters: ADF consistency, OU quality, hedge stability,
rolling Sharpe floor.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class WindowRecord:
    """Metrics from one rebalance window."""
    adf_pvalue: float
    kappa: float
    sigma: float
    kappa_sigma: float
    hedge_ratio: float
    sharpe: float


class PairHistory:
    """
    Tracks metric history for one pair across rebalance windows.
    Decides if the pair is tradeable based on stability filters.
    """

    def __init__(self, pair_id: str, n_consistent: int = 3,
                 min_kappa_sigma: float = 0.5, max_h_drift: float = 0.20,
                 min_sharpe: float = 0.0, max_windows: int = 20):
        self.pair_id = pair_id
        self.n_consistent = n_consistent
        self.min_kappa_sigma = min_kappa_sigma
        self.max_h_drift = max_h_drift
        self.min_sharpe = min_sharpe
        self.max_windows = max_windows
        self.windows: List[WindowRecord] = []

    def record(self, adf_pvalue: float, kappa: float, sigma: float,
               hedge_ratio: float, sharpe: float):
        """Record metrics from one rebalance window."""
        ks = kappa / sigma if sigma > 0 else 0.0
        self.windows.append(WindowRecord(
            adf_pvalue=adf_pvalue, kappa=kappa, sigma=sigma,
            kappa_sigma=ks, hedge_ratio=hedge_ratio, sharpe=sharpe))
        # Cap history
        if len(self.windows) > self.max_windows:
            self.windows = self.windows[-self.max_windows:]

    def is_tradeable(self) -> bool:
        """Check all stability filters on the last n_consistent windows."""
        if len(self.windows) < self.n_consistent:
            return False

        recent = self.windows[-self.n_consistent:]

        # 1. ADF consistency: all recent windows must have p < 0.05
        if any(w.adf_pvalue >= 0.05 for w in recent):
            return False

        # 2. OU quality: latest window kappa/sigma above threshold
        if recent[-1].kappa_sigma < self.min_kappa_sigma:
            return False

        # 3. Hedge ratio stability: check drift between consecutive windows
        for i in range(1, len(recent)):
            h_prev = recent[i - 1].hedge_ratio
            h_curr = recent[i].hedge_ratio
            if h_prev == 0:
                return False
            drift = abs(h_curr - h_prev) / abs(h_prev)
            if drift > self.max_h_drift:
                return False

        # 4. Rolling Sharpe floor: latest window
        if recent[-1].sharpe < self.min_sharpe:
            return False

        return True

    @property
    def latest_kappa_sigma(self) -> float:
        """Latest window's kappa/sigma for ranking."""
        if not self.windows:
            return 0.0
        return self.windows[-1].kappa_sigma


class PairSelector:
    """
    Manages PairHistory for all candidate pairs.
    Call update() each rebalance, then select() to get tradeable pairs.
    """

    def __init__(self, n_consistent: int = 3, min_kappa_sigma: float = 0.5,
                 max_h_drift: float = 0.20, min_sharpe: float = 0.0):
        self.n_consistent = n_consistent
        self.min_kappa_sigma = min_kappa_sigma
        self.max_h_drift = max_h_drift
        self.min_sharpe = min_sharpe
        self._histories: Dict[str, PairHistory] = {}

    def _get_history(self, pair_id: str) -> PairHistory:
        if pair_id not in self._histories:
            self._histories[pair_id] = PairHistory(
                pair_id, n_consistent=self.n_consistent,
                min_kappa_sigma=self.min_kappa_sigma,
                max_h_drift=self.max_h_drift,
                min_sharpe=self.min_sharpe)
        return self._histories[pair_id]

    def update(self, pair_id: str, adf_pvalue: float, kappa: float,
               sigma: float, hedge_ratio: float, sharpe: float):
        """Record one window's metrics for a pair."""
        self._get_history(pair_id).record(
            adf_pvalue=adf_pvalue, kappa=kappa, sigma=sigma,
            hedge_ratio=hedge_ratio, sharpe=sharpe)

    def update_from_engine(self, pair_id: str, engine, sharpe: float):
        """Extract metrics from a fitted PairEngine and record."""
        from statsmodels.tsa.stattools import adfuller
        # Compute ADF p-value from the spread
        spread = engine.spread
        if spread is not None and len(spread.dropna()) > 20:
            try:
                adf_result = adfuller(spread.dropna(), autolag="AIC")
                adf_pvalue = float(adf_result[1])
            except Exception:
                adf_pvalue = 1.0
        else:
            adf_pvalue = 1.0

        self.update(
            pair_id=pair_id,
            adf_pvalue=adf_pvalue,
            kappa=engine.ou.kappa if engine.ou else 0.0,
            sigma=engine.ou.sigma if engine.ou else 0.0,
            hedge_ratio=engine.hedge_ratio,
            sharpe=sharpe)

    def select(self, top_n: int = 0) -> list:
        """
        Return list of tradeable pair_ids, ranked by kappa/sigma descending.
        If top_n > 0, return at most top_n pairs.
        """
        tradeable = [
            (pid, h.latest_kappa_sigma)
            for pid, h in self._histories.items()
            if h.is_tradeable()
        ]
        tradeable.sort(key=lambda x: x[1], reverse=True)
        result = [pid for pid, _ in tradeable]
        if top_n > 0:
            result = result[:top_n]
        return result
