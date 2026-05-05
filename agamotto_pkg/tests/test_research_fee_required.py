"""FEE is required in AgamottoResearch.engineer_features — no silent fallback.

Regression for the FEE-or-falsy-collapse bug class. The historical line
    fee_rate = float(self.config.get("FEE", 0.0) or 0.0) / 10000.0
collapsed FEE=0 (Python falsy) to the default, hiding misconfiguration.
Per CLAUDE.md "no fallback defaults", a missing FEE must raise.

This is the cross-callsite companion to mjolnir/tests/test_research_fee_zero.py;
the fee-loading line is identical across agamotto, mjolnir and orb research
modules so the same contract must hold.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "agamotto_pkg/src")
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import pytest

from agamotto.research import AgamottoResearch


def _make_raw(n: int = 50, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Minimal OHLCV frame consumable by engineer_features."""
    idx = pd.date_range("2025-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame(
        {
            f"{symbol}_open": close - 0.05,
            f"{symbol}_high": close + 0.5,
            f"{symbol}_low": close - 0.5,
            f"{symbol}_close": close,
            f"{symbol}_volume": rng.integers(100, 1000, n).astype(float),
        },
        index=idx,
    )


def _base_config(fee_value):
    cfg = {
        "SYMBOLS": ["BINANCE_PERP_BTC_USDT"],
        "EXCHANGE": "BINANCE",
        "DATA": "liquid",
        "TIME_UNIT": "1h",
        "LADDER": 1,
        "MA_PERIODS": [7, 25, 99],
        "STATS_WINDOW": 14,
    }
    if fee_value is not None:
        cfg["FEE"] = fee_value
    return cfg


def test_fee_missing_raises_key_error():
    """Per CLAUDE.md 'no fallback defaults', missing FEE must raise."""
    research = AgamottoResearch(_base_config(fee_value=None), "/tmp/fake_root")
    research.raw = _make_raw()
    with pytest.raises(KeyError, match="FEE"):
        research.engineer_features()


def test_fee_zero_does_not_silently_inject_phantom_fee():
    """FEE=0.0 must produce fee_rate=0.0 (and not collapse to a default)."""
    research = AgamottoResearch(_base_config(fee_value=0.0), "/tmp/fake_root")
    research.raw = _make_raw()
    # Should not raise. We're not asserting on output values here — only that
    # the FEE=0 path executes successfully (the buggy `or 0.0` form would
    # also pass, but the matched test_fee_missing_raises_key_error above
    # locks in the contract that distinguishes good from bad code.)
    research.engineer_features()
    assert research.features is not None
