"""
vibranium/research.py — Research pipeline for Vibranium stationary portfolio strategy.

Runs walk-forward backtests across hyperparameter combinations and prints a summary table.
"""
import itertools
import logging
import os
from dataclasses import dataclass

import pandas as pd

from vibranium.data import VibraniumData
from vibranium.backtest import run_backtest, BacktestConfig, BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class SweepResult:
    """Result of one hyperparameter combination."""
    config: BacktestConfig
    result: BacktestResult
    label: str


class VibraniumResearch:
    """
    Vibranium research pipeline: hyperparameter sweep over walk-forward backtests.

    Parameters
    ----------
    data_dir : str
        Path to kline data directory.
    symbols : list[str] or None
        Override symbol list. If None, loads all available.
    start_date : str or None
        ISO date to start data.
    end_date : str or None
        ISO date to end data.
    """

    def __init__(
        self,
        data_dir: str = "data/BINANCEFUTURES/15m/liquid/",
        symbols: list = None,
        start_date: str = None,
        end_date: str = None,
    ):
        self.data_dir = data_dir
        self._symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self._loader = VibraniumData(data_dir=data_dir)
        self._prices = None

    def _load_prices(self) -> pd.DataFrame:
        """Load and cache prices."""
        if self._prices is None:
            self._prices = self._loader.load_prices(
                symbols=self._symbols,
                start_date=self.start_date,
                end_date=self.end_date,
            )
        return self._prices

    def run_single(self, config: BacktestConfig) -> BacktestResult:
        """Run a single backtest with the given config."""
        prices = self._load_prices()
        return run_backtest(prices, config)

    def run_sweep(
        self,
        lookback_bars: list = None,
        n_assets: list = None,
        rebalance_bars: list = None,
        entry_z: list = None,
        exit_z: list = None,
        max_halflife: list = None,
        optimize: list = None,
    ) -> list:
        """
        Sweep over hyperparameter grid.

        Each parameter is a list of values to try. All combinations are tested.
        Defaults to a sensible grid if not specified.

        Returns
        -------
        list[SweepResult]
            Sorted by Sharpe descending.
        """
        if lookback_bars is None:
            lookback_bars = [5760, 8640]        # 60d, 90d
        if n_assets is None:
            n_assets = [6, 8]
        if rebalance_bars is None:
            rebalance_bars = [672]               # 7d
        if entry_z is None:
            entry_z = [1.5, 2.0, 2.5]
        if exit_z is None:
            exit_z = [0.5]
        if max_halflife is None:
            max_halflife = [500.0]
        if optimize is None:
            optimize = [True, False]

        prices = self._load_prices()
        grid = list(itertools.product(
            lookback_bars, n_assets, rebalance_bars,
            entry_z, exit_z, max_halflife, optimize,
        ))

        logger.info("Running %d configurations", len(grid))
        results = []

        for i, (lb, na, rb, ez, xz, mh, opt) in enumerate(grid):
            cfg = BacktestConfig(
                lookback_bars=lb,
                rebalance_bars=rb,
                n_assets=na,
                entry_z=ez,
                exit_z=xz,
                max_halflife=mh,
                optimize=opt,
            )
            label = (
                f"lb={lb} na={na} rb={rb} "
                f"ez={ez} xz={xz} mh={mh} opt={opt}"
            )
            logger.info("[%d/%d] %s", i + 1, len(grid), label)

            try:
                bt = run_backtest(prices, cfg)
                results.append(SweepResult(config=cfg, result=bt, label=label))
            except Exception as exc:
                logger.warning("Config failed: %s — %s", label, exc)

        results.sort(key=lambda r: r.result.sharpe, reverse=True)
        self._print_summary(results)
        return results

    def _print_summary(self, results: list):
        """Print a summary table of sweep results."""
        if not results:
            print("No results.")
            return

        sep = "-" * 110
        print(sep)
        print("VIBRANIUM RESEARCH — Stationary Portfolio Sweep")
        print(sep)
        header = (
            f"{'#':>3} {'Sharpe':>7} {'Return':>8} {'MaxDD':>8} "
            f"{'AvgHL':>6} {'Trades':>6} {'Rebal':>5} {'Config'}"
        )
        print(header)
        print(sep)

        for i, sr in enumerate(results):
            r = sr.result
            sign = "+" if r.total_return >= 0 else ""
            row = (
                f"{i+1:>3} {r.sharpe:>+7.2f} {sign}{r.total_return:>7.4f} "
                f"{r.max_drawdown:>+8.4f} {r.avg_halflife:>6.0f} "
                f"{r.avg_n_trades:>6.1f} {r.n_rebalances:>5d} {sr.label}"
            )
            print(row)

        print(sep)
        if results:
            best = results[0]
            print(
                f"Best: Sharpe={best.result.sharpe:+.2f}  "
                f"Return={best.result.total_return:+.4f}  "
                f"Config: {best.label}"
            )
        print(sep)

    def save(self, results: list, output_dir: str = "vibranium/output/"):
        """Save sweep results to CSV."""
        os.makedirs(output_dir, exist_ok=True)

        rows = []
        for sr in results:
            r = sr.result
            c = sr.config
            rows.append({
                "lookback_bars": c.lookback_bars,
                "n_assets": c.n_assets,
                "rebalance_bars": c.rebalance_bars,
                "entry_z": c.entry_z,
                "exit_z": c.exit_z,
                "max_halflife": c.max_halflife,
                "optimize": c.optimize,
                "sharpe": r.sharpe,
                "total_return": r.total_return,
                "max_drawdown": r.max_drawdown,
                "avg_halflife": r.avg_halflife,
                "avg_n_trades": r.avg_n_trades,
                "n_rebalances": r.n_rebalances,
            })

        df = pd.DataFrame(rows)
        path = os.path.join(output_dir, "sweep_results.csv")
        df.to_csv(path, index=False)
        print(f"Saved sweep results to {path}")
