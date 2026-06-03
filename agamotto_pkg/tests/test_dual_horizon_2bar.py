"""Tests for the generalized dual-horizon (2-bar) design.

DH = predict a 1-bar AND a 2-bar horizon return; fire on the current bar only
when BOTH horizons clear threshold in the SAME direction. Two pieces:
  1. research.engineer_features emits a 2-bar target set (ret_2bar + laddered/fee
     return_{long,short}_2bar) when config["DUAL_HORIZON"] is set — at any TF.
  2. trading.dual_gate_filter applies the symmetric long+short agreement gate.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))


# ---------------------------------------------------------------------------
# 1. ret_2bar target generation in engineer_features
# ---------------------------------------------------------------------------
def _synthetic_raw(prefix="BTCUSDT", n=160):
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    rng = np.random.RandomState(7)
    close = pd.Series(100 + np.cumsum(rng.randn(n) * 0.3), index=idx)
    high = close * (1 + np.abs(rng.randn(n)) * 0.001)
    low = close * (1 - np.abs(rng.randn(n)) * 0.001)
    open_ = close.shift(1).fillna(close.iloc[0])
    vol = pd.Series(1000 + rng.randint(0, 500, n), index=idx)
    return pd.DataFrame({
        f"{prefix}_open": open_, f"{prefix}_high": high,
        f"{prefix}_low": low, f"{prefix}_close": close, f"{prefix}_volume": vol,
    })


def _base_config(dual):
    cfg = {
        "TIME_UNIT": "15m",
        "SYMBOLS": ["BTCUSDT"],
        "LADDER": 0,          # 0 ladder layers => total layers = 1
        "FEE": 0,             # no fee => laddered return == plain cumulative
        "MA_PERIODS": [7, 25, 99],
        "STATS_WINDOW": 14,
    }
    if dual:
        cfg["DUAL_HORIZON"] = True
    return cfg


def test_ret_2bar_is_cumulative_two_bar_return():
    from agamotto.research import AgamottoResearch
    prefix = "BTCUSDT"
    raw = _synthetic_raw(prefix)
    r = AgamottoResearch(_base_config(dual=True), ".")
    r.raw = raw
    r.engineer_features()

    assert f"{prefix}_ret_2bar" in r.features.columns
    close = raw[f"{prefix}_close"]
    expected = close.shift(-2) / close.replace(0, np.nan) - 1
    got = r.features[f"{prefix}_ret_2bar"]
    mask = expected.notna()
    assert np.allclose(got[mask].values, expected[mask].values, atol=1e-12)
    # last two bars have no 2-bar future -> NaN
    assert pd.isna(got.iloc[-1]) and pd.isna(got.iloc[-2])


def test_laddered_2bar_equals_plain_when_no_ladder_no_fee():
    from agamotto.research import AgamottoResearch
    prefix = "BTCUSDT"
    raw = _synthetic_raw(prefix)
    r = AgamottoResearch(_base_config(dual=True), ".")
    r.raw = raw
    r.engineer_features()
    # LADDER=0, FEE=0 => total layers = 1, fee_cost = 0 => return_*_2bar == ret_2bar
    rl = r.features[f"{prefix}_return_long_2bar"]
    rs = r.features[f"{prefix}_return_short_2bar"]
    ret2 = r.features[f"{prefix}_ret_2bar"]
    m = ret2.notna()
    assert np.allclose(rl[m].values, ret2[m].values, atol=1e-12)
    assert np.allclose(rs[m].values, ret2[m].values, atol=1e-12)


def test_no_2bar_columns_without_dual_horizon():
    from agamotto.research import AgamottoResearch
    prefix = "BTCUSDT"
    raw = _synthetic_raw(prefix)
    r = AgamottoResearch(_base_config(dual=False), ".")
    r.raw = raw
    r.engineer_features()
    assert f"{prefix}_ret_2bar" not in r.features.columns
    assert f"{prefix}_return_long_2bar" not in r.features.columns
    # 1-bar targets are still present
    assert f"{prefix}_return_long" in r.features.columns


# ---------------------------------------------------------------------------
# 2. Symmetric dual-horizon agreement gate (trading.dual_gate_filter)
# ---------------------------------------------------------------------------
def _row(position, pred, thr, pred2=None, thr2=None):
    d = {"position": position, "prediction": pred, "opt_threshold": thr}
    if pred2 is not None:
        d["prediction_2bar"] = pred2
    if thr2 is not None:
        d["opt_threshold_2bar"] = thr2
    return d


def test_dual_gate_long_requires_both_above():
    from agamotto.trading import dual_gate_filter
    df = pd.DataFrame([
        _row("long", 0.005, 0.001, pred2=0.004, thr2=0.001),   # both above -> fire
        _row("long", 0.005, 0.001, pred2=-0.002, thr2=0.001),  # 2-bar below -> suppressed
    ])
    longs, shorts = dual_gate_filter(df)
    assert len(longs) == 1 and len(shorts) == 0
    assert longs.iloc[0]["prediction_2bar"] == 0.004


def test_dual_gate_short_requires_both_below():
    from agamotto.trading import dual_gate_filter
    df = pd.DataFrame([
        _row("short", -0.005, -0.001, pred2=-0.004, thr2=-0.001),  # both below -> fire
        _row("short", -0.005, -0.001, pred2=0.002, thr2=-0.001),   # 2-bar above -> suppressed
    ])
    longs, shorts = dual_gate_filter(df)
    assert len(shorts) == 1 and len(longs) == 0
    assert shorts.iloc[0]["prediction_2bar"] == -0.004


def test_dual_gate_falls_back_to_1bar_when_no_2bar_columns():
    from agamotto.trading import dual_gate_filter
    df = pd.DataFrame([
        _row("long", 0.005, 0.001),
        _row("short", -0.005, -0.001),
        _row("long", 0.0005, 0.001),   # 1-bar below -> never fires
    ])
    longs, shorts = dual_gate_filter(df)
    assert len(longs) == 1 and len(shorts) == 1


def test_dual_gate_null_2bar_threshold_falls_back_to_1bar():
    from agamotto.trading import dual_gate_filter
    df = pd.DataFrame([
        # 2-bar columns present but threshold is NaN for this row -> 1-bar only
        _row("long", 0.005, 0.001, pred2=-0.9, thr2=np.nan),
    ])
    longs, shorts = dual_gate_filter(df)
    assert len(longs) == 1  # not suppressed by the (null) 2-bar gate
