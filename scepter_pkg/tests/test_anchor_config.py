"""ScepterResearch config keys are required, with no silent defaults.

ANCHOR_WINDOWS defaulted to [14, 28] via config.get until 2026-09-29.
"""
from __future__ import annotations

import pytest

from scepter.research import ScepterResearch

_BASE = {
    "SYMBOLS": ["BINANCE_PERP_SOL_USDT"],
    "ANCHOR_SYMBOLS": ["BINANCE_PERP_BTC_USDT"],
    "ANCHOR_WINDOWS": [14, 28],
    "EXCHANGE": "BINANCE",
    "DATA": "liquid",
    "TIMEFRAMES": ["15m"],
    "BASE_TF": "15m",
    "TARGET_TF": "15m",
    "TIME_UNIT": "15m",
    "LADDER": 1,
    "LADDER_BPS": 1.0,
    "FEE": 0,
    "MA_PERIODS": [7, 25, 99],
    "STATS_WINDOW": 14,
}


def _without(key: str) -> dict:
    return {k: v for k, v in _BASE.items() if k != key}


def test_anchor_windows_used_as_given():
    assert ScepterResearch({**_BASE, "ANCHOR_WINDOWS": [5, 10]}, "/tmp/fake_root").anchor_windows == [5, 10]


def test_missing_anchor_windows_raises():
    with pytest.raises(KeyError, match="ANCHOR_WINDOWS"):
        ScepterResearch(_without("ANCHOR_WINDOWS"), "/tmp/fake_root")


def test_empty_anchor_windows_raises():
    with pytest.raises(ValueError, match="ANCHOR_WINDOWS"):
        ScepterResearch({**_BASE, "ANCHOR_WINDOWS": []}, "/tmp/fake_root")


def test_missing_anchor_symbols_still_raises_first():
    with pytest.raises(KeyError, match="ANCHOR_SYMBOLS"):
        ScepterResearch(_without("ANCHOR_SYMBOLS"), "/tmp/fake_root")
