"""create() must SKIP a regime whose source column was never built for this
dataset, rather than crash the whole run on the first one.

Real incident: equities OHLCV data (adamantium/us_stocks) has no
quote_volume/taker_buy_quote_volume/number_of_trades columns (unlike Binance
klines), so buy_pressure/trade_intensity are never computed for it. The crypto-
oriented regime stack still names them, and every equities run of
run_orb_research.py crashed on the first such regime with
`ValueError: Filter 'buy_pressure' requires column 'buy_pressure'` — even
though 100+ other regimes in the same stack were perfectly evaluable.
_require_col's own message says the remedy is to drop the regime from the
stack, not abort the run; this locks that behavior in.

The skip must be NARROW: only MissingFilterColumnError is swallowed. Any other
exception type must still propagate and fail the run, so a real bug elsewhere
in filter evaluation is never silently hidden alongside the legitimate skip.
"""
from __future__ import annotations

import csv
import sys
from unittest.mock import patch

sys.path.insert(0, "agamotto_pkg/src")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import pytest

from agamotto.research import AgamottoResearch
from agamotto.research_filters import MissingFilterColumnError

# Raw column prefix must be the NATIVE symbol (what _symbol_to_native maps
# SYMBOL down to), not the full exchange-qualified config entry — mirrors
# test_vertical_features_csv_flag.py's SYMBOL="BTCUSDT" / SYMBOLS=["BINANCE_
# PERP_BTC_USDT"] split exactly; using the same string for both leaves
# engineer_features unable to find any raw column for the configured symbol.
NATIVE = "BTCUSDT"
SYMBOL = "BINANCE_PERP_BTC_USDT"


def _make_raw(n: int = 400) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(3)
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame(
        {
            f"{NATIVE}_open": close - 0.05,
            f"{NATIVE}_high": close + 0.5,
            f"{NATIVE}_low": close - 0.5,
            f"{NATIVE}_close": close,
            f"{NATIVE}_volume": rng.integers(100, 1000, n).astype(float),
        },
        index=idx,
    )


def _stack_csv(tmp_path, regimes: list[str]):
    p = tmp_path / "regime_stack.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["regime", "position", "model"])
        w.writeheader()
        for r in regimes:
            w.writerow({"regime": r, "position": "long", "model": "Ridge"})
    return p


def _build(tmp_path, regimes: list[str]) -> AgamottoResearch:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    cfg = {
        "SYMBOLS": [SYMBOL],
        "EXCHANGE": "BINANCE",
        "DATA": "liquid",
        "TIME_UNIT": "1h",
        "LADDER": 1,
        "LADDER_BPS": 1.0,
        "MA_PERIODS": [7, 25, 99],
        "STATS_WINDOW": 14,
        "FEE": 2.25,
        "VERSION": "test.skip_missing_column",
        "OUTPUT_DIR": str(out_dir),
        "REGIME_STACK_PATH": str(_stack_csv(tmp_path, regimes)),
    }
    research = AgamottoResearch(cfg, str(tmp_path))
    research.raw = _make_raw()
    return research


def test_evaluable_regime_is_not_skipped_alongside_an_unevaluable_one(tmp_path, caplog):
    """buy_pressure needs quote_volume-derived columns this raw OHLCV frame
    never gets (no quote_volume column at all) and must be skipped; rsi_oversold
    needs only close-derived `rsi`, always present, and must NOT be skipped —
    i.e. the skip is per-regime, not an all-or-nothing fallback for the run."""
    import logging
    with caplog.at_level(logging.WARNING, logger="agamotto.research"):
        research = _build(tmp_path, ["buy_pressure", "rsi_oversold"])
        research.create()

    skip_lines = [r.message for r in caplog.records if r.message.startswith("Skipping regime")]
    assert any("buy_pressure" in m for m in skip_lines), "buy_pressure should have been skipped"
    assert not any("rsi_oversold" in m for m in skip_lines), (
        "rsi_oversold has every column it needs and must not be skipped"
    )


def test_missing_column_regime_does_not_abort_the_run(tmp_path):
    """The actual regression: before the fix, this raised and create() never
    returned. Now it must return the out_dir normally."""
    research = _build(tmp_path, ["buy_pressure"])
    out_dir = research.create()
    assert out_dir is not None


def test_other_exception_types_still_propagate(tmp_path, monkeypatch):
    """The catch must be narrow: a non-MissingFilterColumnError failure in
    filter evaluation must still crash the run, not be swallowed alongside
    the legitimate skip."""
    research = _build(tmp_path, ["rsi_oversold"])

    def _boom(self, regime, save=False, out_dir=None):
        raise RuntimeError("unrelated real bug")

    with patch.object(AgamottoResearch, "filter_signals", _boom):
        with pytest.raises(RuntimeError, match="unrelated real bug"):
            research.create()


def test_missing_filter_column_error_is_a_value_error_subclass():
    """Existing `except ValueError` call sites elsewhere must be unaffected."""
    assert issubclass(MissingFilterColumnError, ValueError)
