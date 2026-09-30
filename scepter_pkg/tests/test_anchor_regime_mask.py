"""ScepterResearch._apply_filter_mask must fail loud on a missing anchor column.

Until 2026-09-29 a missing ANCHOR_REGIMES `col` logged a warning and returned
an all-True mask, so the anchor half of every `{own}_and_{anchor}` regime
vanished and the regime traded unconditioned. Reachable when an arm's
ANCHOR_REGIMES was copied from an arm with a different anchor (BTC vs QQQ), or
when the anchor's klines never loaded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agamotto.research_filters import MissingFilterColumnError
from scepter._obf.codec import default
from scepter.research import ScepterResearch

_BTC_REGIMES = {
    "btc_trending_up": {"col": "btc_close_vs_ma", "op": ">", "val": 0},
    "btc_trending_down": {"col": "btc_close_vs_ma", "op": "<", "val": 0},
}


def _research(anchor_symbols, anchor_regimes) -> ScepterResearch:
    return ScepterResearch({
        "SYMBOLS": ["BINANCE_PERP_SOL_USDT"],
        "ANCHOR_SYMBOLS": anchor_symbols,
        "ANCHOR_WINDOWS": [14, 28],
        "ANCHOR_REGIMES": anchor_regimes,
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
    }, "/tmp/fake_root")


def _frame(**cols) -> pd.DataFrame:
    return pd.DataFrame(cols, index=range(len(next(iter(cols.values())))))


@pytest.fixture
def qqq_arm_with_btc_regimes() -> ScepterResearch:
    """The copy-paste case: a QQQ-anchored arm carrying a BTC arm's ANCHOR_REGIMES."""
    return _research(["QQQ"], _BTC_REGIMES)


def test_missing_anchor_column_raises_naming_part_col_and_anchors(qqq_arm_with_btc_regimes):
    df = _frame(qqq_close_vs_ma=[1.0, -1.0, 2.0])
    with pytest.raises(ValueError) as exc:
        qqq_arm_with_btc_regimes._apply_filter_mask(df, "btc_trending_up", "long")
    msg = str(exc.value)
    assert "'btc_trending_up'" in msg
    assert "'btc_close_vs_ma'" in msg
    assert "['QQQ']" in msg
    assert "['qqq']" in msg


def test_missing_anchor_column_is_not_the_skippable_error(qqq_arm_with_btc_regimes):
    # create() catches MissingFilterColumnError and SKIPS the regime. Every
    # scepter regime has an anchor leg, so that type would silently skip the
    # whole stack instead of failing the run.
    df = _frame(qqq_close_vs_ma=[1.0])
    with pytest.raises(ValueError) as exc:
        qqq_arm_with_btc_regimes._apply_filter_mask(df, "btc_trending_up", "long")
    assert not isinstance(exc.value, MissingFilterColumnError)


def test_compound_raises_instead_of_dropping_the_anchor_leg(qqq_arm_with_btc_regimes):
    df = _frame(rsi=[10.0, 50.0, 20.0], qqq_close_vs_ma=[1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match="btc_close_vs_ma"):
        qqq_arm_with_btc_regimes._apply_filter_mask(
            df, "rsi_oversold_and_btc_trending_down", "long")


def test_coded_regime_name_raises_too(qqq_arm_with_btc_regimes):
    coded = default().encode_regime("rsi_oversold_and_btc_trending_up")
    df = _frame(rsi=[10.0], qqq_close_vs_ma=[1.0])
    with pytest.raises(ValueError, match="btc_close_vs_ma"):
        qqq_arm_with_btc_regimes._apply_filter_mask(df, coded, "long")


def test_coded_anchor_regime_keys_raise_too():
    # Committed setting.json files key ANCHOR_REGIMES by CODE (e.g. "r080").
    c = default()
    coded = {c.encode_regime(k): v for k, v in _BTC_REGIMES.items()}
    research = _research(["QQQ"], coded)
    with pytest.raises(ValueError, match="btc_close_vs_ma"):
        research._apply_filter_mask(
            _frame(qqq_close_vs_ma=[1.0]), "btc_trending_up", "long")


def test_present_anchor_column_masks_and_composes():
    research = _research(["BINANCE_PERP_BTC_USDT"], _BTC_REGIMES)
    df = _frame(rsi=[10.0, 50.0, 20.0, 15.0],
                btc_close_vs_ma=[1.0, 1.0, -1.0, np.nan])
    anchor = research._apply_filter_mask(df, "btc_trending_up", "long")
    assert anchor.tolist() == [True, True, False, False]  # NaN -> False
    own = research._apply_filter_mask(df, "rsi_oversold", "long")
    both = research._apply_filter_mask(df, "rsi_oversold_and_btc_trending_up", "long")
    assert both.tolist() == (own & anchor).tolist()


def test_empty_frame_returns_empty_mask_like_the_base(qqq_arm_with_btc_regimes):
    # Mirrors agamotto's apply_filter_mask `df.empty` guard: length 0, so it
    # cannot make a regime fire.
    for df in (pd.DataFrame(), pd.DataFrame(index=range(5))):
        mask = qqq_arm_with_btc_regimes._apply_filter_mask(df, "btc_trending_up", "long")
        assert len(mask) == 0
