"""Regression lock for the base-bar grid: `bar_tf` and `horizon_bars`.

THE DEFECT. Until 2026-08-09 four sites — ``core/research.py`` (``:350``
``engineer_features``, ``:490`` ``verticalize``) and ``core/streaming.py``
(``:211``, ``:340``) — each carried their own copy of::

    bar_tf = "5s" if cfg.get("TRAIN_BARS_DIR") else time_unit
    horizon_bars = max(1, _TF_SECONDS[time_unit] // _TF_SECONDS[bar_tf])

``bar_tf`` came from WHETHER ``TRAIN_BARS_DIR`` was set, never from WHAT IT
POINTED AT. Every production arm's ``TRAIN_BARS_DIR`` happens to point at a
``base.5s_1/bars`` directory, so the guess held by convention alone. Point it at
15s bars and a 1m target still computed ``horizon_bars = 60//5 = 12`` when the
truth is ``60//15 = 4`` — a 3x wrong ladder lookahead AND a 3x wrong
``_bar_seconds`` in the boundary-aligned target, silently, with no error. That
blocks coarser base grids (the cheap way to shrink a 195 GB panel) and is a live
correctness hazard for any non-5s base.

Second defect on the same line: ``max(1, ...)`` clamped a quantity DERIVED from
other config, banned by name in CLAUDE.md. A ``TIME_UNIT`` finer than the bars
became a plausible 1-bar ladder instead of a raise.

WHAT THIS FILE PINS.
1. ``test_production_arm_horizons`` — the exact ``(bar_tf, horizon_bars)`` each
   of the 10 real tick arms resolved to BEFORE the fix. If any of these move,
   every cached filter parquet in production is invalidated.
2. The 15s-base case that used to return 12 and must return 4.
3. The sub-bar case that used to floor to 1 and must raise.
4. That both ``research.py`` consumers and both ``streaming.py`` consumers go
   through the ONE helper — no second copy can reappear.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import mjolnir.core.research as research_mod
from mjolnir.core.features import (
    _TF_SECONDS,
    infer_bar_seconds,
    resolve_bar_grid,
)
from mjolnir.core.research import MjolnirResearch


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _index(n: int, tf: str, unit: str = "ns", tz="UTC") -> pd.DatetimeIndex:
    """A uniform bar index at *tf*.

    The freq is an explicit ``Timedelta``, NOT the TF code: pandas reads ``"1m"``
    as one MONTH, so ``freq="1m"`` would silently build a 31-day grid.
    """
    idx = pd.date_range(
        "2026-01-01", periods=n, freq=pd.Timedelta(seconds=_TF_SECONDS[tf]), tz=tz)
    return idx.astype(f"datetime64[{unit}, {tz}]") if tz else idx.astype(
        f"datetime64[{unit}]")


def _bars(n: int, tf: str, **kw) -> pd.DataFrame:
    """Minimal aligned-bar frame with everything MjolnirFeatures.compute needs."""
    idx = _index(n, tf, **kw)
    bid = 100.0 + np.arange(n) * 0.01
    return pd.DataFrame(
        {
            "open": bid, "high": bid + 0.02, "low": bid - 0.02, "close": bid + 0.01,
            "volume": np.full(n, 3.0), "n_trades": np.full(n, 5.0),
            "trade_imbalance": np.zeros(n), "vwap": bid,
            "buy_vol": np.full(n, 1.5), "sell_vol": np.full(n, 1.5),
            "bid_price": bid, "ask_price": bid + 0.01,
            "bid_amount": np.full(n, 10.0), "ask_amount": np.full(n, 5.0),
            "bids_0_qty": np.full(n, 10.0), "asks_0_qty": np.full(n, 5.0),
            "depth_bid_L1": np.full(n, 10.0), "depth_ask_L1": np.full(n, 5.0),
            "open_interest": 1000.0 + np.arange(n), "mark_price": bid,
            "index_price": bid,
        },
        index=idx,
    )


def _cfg(time_unit: str, train_bars_dir=None) -> dict:
    cfg = {
        "TIME_UNIT": time_unit,
        "FEE": 0.0,
        "TARGET_HORIZON_BARS": 1,
        "LADDER_FILL_MODE": "ladder",
        "SYMBOLS": ["BTCUSDT"],
    }
    if train_bars_dir is not None:
        cfg["TRAIN_BARS_DIR"] = train_bars_dir
    return cfg


# ---------------------------------------------------------------------------
# 1. Production pin — these 10 numbers must never move
# ---------------------------------------------------------------------------

# (label, TIME_UNIT, ACTUAL base-bar TF on disk, TRAIN_BARS_DIR, horizon_bars)
#
# Sources, read 2026-08-09:
#   marvel mjolnir/gauntlet/pred_mjolnir.base.{5s,15s,30s,1m}_1/setting.json
#   marvel stormbreaker/gauntlet/pred_stormbreaker.base.{5s,15s,30s,1m,5m,15m}_1/setting.json
# ``build_bars.py`` builds at ``cfg["TIME_UNIT"]`` and every ``TRAIN_BARS_DIR``
# points at a ``base.5s_1/bars`` directory, so the base bars really are 5s in all
# ten. ``horizon_bars`` is the value the PRE-FIX code produced, captured by
# driving the real call path.
#
# The two ``5m``/``15m`` stormbreaker arms carry ``TIME_UNIT="5s"`` with
# ``BAR_FREQ`` naming a coarser TF. They are knowingly unrepaired
# (``_PENDING_TF_REPAIR`` in marvel stormbreaker/tests/
# test_stormbreaker_setting_json_invariants.py) and are already rejected by
# marvel's ``bar_spacing.assert_config_bar_tf_consistent`` before reaching dc;
# they are pinned here anyway so nothing about them changes by accident.
PRODUCTION_ARMS = [
    ("mjolnir.base.5s_1", "5s", "5s", None, 1),
    ("mjolnir.base.15s_1", "15s", "5s", "/mnt/.../pred_mjolnir.base.5s_1/bars", 3),
    ("mjolnir.base.30s_1", "30s", "5s", "/mnt/.../pred_mjolnir.base.5s_1/bars", 6),
    ("mjolnir.base.1m_1", "1m", "5s", "/mnt/.../pred_mjolnir.base.5s_1/bars", 12),
    ("stormbreaker.base.5s_1", "5s", "5s", None, 1),
    ("stormbreaker.base.15s_1", "15s", "5s", "/mnt/.../base.5s_1/bars", 3),
    ("stormbreaker.base.30s_1", "30s", "5s", "/mnt/.../base.5s_1/bars", 6),
    ("stormbreaker.base.1m_1", "1m", "5s", "/mnt/.../base.5s_1/bars", 12),
    ("stormbreaker.base.5m_1", "5s", "5s", None, 1),
    ("stormbreaker.base.15m_1", "5s", "5s", None, 1),
]


@pytest.mark.parametrize(
    "label,time_unit,bar_tf,train_bars_dir,expected_horizon",
    PRODUCTION_ARMS,
    ids=[a[0] for a in PRODUCTION_ARMS],
)
def test_production_arm_horizons(
    label, time_unit, bar_tf, train_bars_dir, expected_horizon
):
    """Every real tick arm resolves to exactly what it resolved to before.

    All ten train on 5s bars, so the measured derivation reproduces the old
    guess exactly and no cached filter parquet is invalidated.
    """
    got_tf, got_h = resolve_bar_grid(
        _cfg(time_unit, train_bars_dir), {"BTCUSDT": _bars(200, bar_tf)})
    assert (got_tf, got_h) == (bar_tf, expected_horizon)


# ---------------------------------------------------------------------------
# 2. The defect: a non-5s TRAIN_BARS_DIR
# ---------------------------------------------------------------------------

def test_15s_base_with_1m_target_is_four_bars_not_twelve():
    """THE failing case. Pre-fix this returned ("5s", 12); the truth is 60/15."""
    cfg = _cfg("1m", train_bars_dir="/mnt/.../pred_mjolnir.base.15s_1/bars")
    assert resolve_bar_grid(cfg, {"BTCUSDT": _bars(200, "15s")}) == ("15s", 4)


@pytest.mark.parametrize(
    "bar_tf,time_unit,expected,stale",
    [
        ("15s", "1m", 4, 12),     # 3x too long pre-fix
        ("15s", "30s", 2, 6),     # 3x
        ("30s", "5m", 10, 60),    # 6x
        ("1m", "15m", 15, 180),   # 12x
        ("5m", "15m", 3, 180),    # 60x
    ],
)
def test_coarser_base_grids(bar_tf, time_unit, expected, stale):
    """The coarser-grid experiment this unblocks, across the TF vocabulary."""
    cfg = _cfg(time_unit, train_bars_dir=f"/mnt/.../base.{bar_tf}_1/bars")
    got_tf, got_h = resolve_bar_grid(cfg, {"BTCUSDT": _bars(200, bar_tf)})
    assert (got_tf, got_h) == (bar_tf, expected)
    assert got_h != stale, "still computing the horizon as if the bars were 5s"


def test_train_bars_dir_is_not_consulted_at_all():
    """The flag is a proxy the measurement subsumes; setting it changes nothing."""
    bars = {"BTCUSDT": _bars(200, "15s")}
    with_flag = resolve_bar_grid(_cfg("1m", "/anywhere/at/all"), bars)
    without = resolve_bar_grid(_cfg("1m"), bars)
    assert with_flag == without == ("15s", 4)


# ---------------------------------------------------------------------------
# 3. The clamp: a sub-bar TIME_UNIT must raise, not floor to 1
# ---------------------------------------------------------------------------

def test_sub_bar_time_unit_raises_instead_of_flooring_to_one():
    """``max(1, 5//60)`` used to hand back a believable 1-bar ladder."""
    with pytest.raises(ValueError, match="FINER than the measured base bars"):
        resolve_bar_grid(_cfg("5s"), {"BTCUSDT": _bars(200, "1m")})


def test_non_divisible_pair_raises(monkeypatch):
    """A target boundary that does not land on a bar close must raise.

    Every ordered pair inside the shipped TF vocabulary (5/15/30/60/300/900s)
    happens to divide evenly, so the straddle guard is exercised against an
    injected 45s TF — 60 % 45 != 0.
    """
    monkeypatch.setitem(_TF_SECONDS, "45s", 45)
    from mjolnir.core import features as feat_mod
    monkeypatch.setitem(feat_mod._SECONDS_TO_TF, 45, "45s")
    idx = pd.date_range(
        "2026-01-01", periods=200, freq=pd.Timedelta(seconds=45), tz="UTC")
    bars = _bars(200, "5s").set_axis(idx)
    with pytest.raises(ValueError, match="not a whole multiple"):
        resolve_bar_grid(_cfg("1m"), {"BTCUSDT": bars})


# ---------------------------------------------------------------------------
# 4. Measurement itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
@pytest.mark.parametrize("tf", sorted(_TF_SECONDS, key=_TF_SECONDS.get))
def test_spacing_is_unit_independent(unit, tf):
    """The [us] trap: bar parquets read back as ``datetime64[us, UTC]``, and a
    raw ``.asi8`` view would read 5s bars as 0.005s (CLAUDE.md; dc 25eb57a)."""
    assert infer_bar_seconds(_index(200, tf, unit=unit)) == float(_TF_SECONDS[tf])


def test_spacing_survives_gaps_and_duplicated_seams():
    """Modal, not mean: a day-boundary gap must not move the answer."""
    idx = _index(400, "5s")
    idx = idx.delete(range(100, 130))            # a 150s hole
    assert infer_bar_seconds(idx) == 5.0


def test_spacing_undecidable_on_short_input():
    assert infer_bar_seconds(_index(5, "5s")) is None
    assert infer_bar_seconds(pd.Index([1, 2, 3])) is None


def test_unmeasurable_bars_raise_rather_than_defaulting():
    with pytest.raises(ValueError, match="cannot measure base-bar spacing"):
        resolve_bar_grid(_cfg("5s"), {"BTCUSDT": _bars(4, "5s")})


def test_symbols_disagreeing_on_grid_raise():
    bars = {"BTCUSDT": _bars(200, "5s"), "ETHUSDT": _bars(200, "30s")}
    with pytest.raises(ValueError, match="disagree on base-bar spacing"):
        resolve_bar_grid(_cfg("1m"), bars)


def test_unknown_measured_grid_raises():
    idx = pd.date_range(
        "2026-01-01", periods=200, freq=pd.Timedelta(seconds=7), tz="UTC")
    bars = _bars(200, "5s").set_axis(idx)
    with pytest.raises(ValueError, match="not a known timeframe"):
        resolve_bar_grid(_cfg("1m"), {"BTCUSDT": bars})


def test_unknown_time_unit_raises():
    with pytest.raises(KeyError, match="not a known timeframe"):
        resolve_bar_grid(_cfg("7s"), {"BTCUSDT": _bars(200, "5s")})


# ---------------------------------------------------------------------------
# 5. Both consumers go through the ONE helper
# ---------------------------------------------------------------------------

_SRC_DIR = Path(research_mod.__file__).resolve().parent


@pytest.mark.parametrize("fname", ["research.py", "streaming.py"])
def test_no_second_copy_of_the_train_bars_dir_guess(fname):
    """The expression that caused this must not reappear in either module.

    Two independent copies is exactly how the defect survived a review that
    fixed neither.
    """
    src = (_SRC_DIR / fname).read_text()
    code = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#"))
    assert not re.search(r'"5s"\s+if\s+.*TRAIN_BARS_DIR', code), (
        f"{fname} re-derives bar_tf from the TRAIN_BARS_DIR flag — "
        "call features.resolve_bar_grid instead")
    assert not re.search(r"max\(\s*\n?\s*1,\s*_TF_SECONDS", code), (
        f"{fname} clamps a derived horizon with max(1, ...) — banned by "
        "CLAUDE.md; raise instead")


def _spy_run(cfg, bars):
    """Drive the real engineer_features/verticalize path, recording what each
    consumer received."""
    seen = {}
    real = research_mod.MjolnirFeatures

    class _Spy(real):
        def __init__(self, *a, **kw):
            if kw.get("prefix") is None:
                seen["bar_tf"] = kw.get("bar_tf")
            super().__init__(*a, **kw)

    research_mod.MjolnirFeatures = _Spy
    try:
        res = MjolnirResearch(config=cfg, home_root="/tmp")
        res._symbol_bars = {"BTCUSDT": bars}
        res.engineer_features()
        assert res._symbol_features, "feature engineering produced no frames"

        def _spy_ladder(df, close_col, low_col, high_col, horizon_bars=1):
            seen["horizon_bars"] = horizon_bars
            return pd.DataFrame(index=df.index)

        res._compute_ladder_returns = _spy_ladder
        res.verticalize()
    finally:
        research_mod.MjolnirFeatures = real
    return seen


def test_engineer_features_and_verticalize_share_the_measured_grid():
    """End-to-end: the 15s/1m case reaches BOTH consumers as (15s, 4)."""
    cfg = _cfg("1m", train_bars_dir="/mnt/.../base.15s_1/bars")
    seen = _spy_run(cfg, _bars(300, "15s"))
    assert seen["bar_tf"] == "15s"
    assert seen["horizon_bars"] == 4


def test_end_to_end_production_5s_arm_is_unchanged():
    """The 1m production arm still reaches both consumers as (5s, 12)."""
    cfg = _cfg("1m", train_bars_dir="/mnt/.../pred_mjolnir.base.5s_1/bars")
    seen = _spy_run(cfg, _bars(300, "5s"))
    assert seen["bar_tf"] == "5s"
    assert seen["horizon_bars"] == 12


def test_streaming_writer_resolves_the_grid_from_the_bars_too():
    """``stream_filter_parquets`` is the path ``create()`` actually takes
    (research.py:687), and it carried its own copy of the guess. It must reach
    the same helper before writing anything."""
    from mjolnir.core.streaming import stream_filter_parquets

    with pytest.raises(ValueError, match="FINER than the measured base bars"):
        stream_filter_parquets(
            config=_cfg("5s"),
            symbol_bars={"BTCUSDT": _bars(200, "1m")},
            multi_tf_bars={},
            out_dir="/tmp/does-not-matter-raises-first",
            regime_stack=[],
            apply_mask_fn=lambda df, name, pos: None,
        )


def test_verticalize_without_engineer_features_raises():
    res = MjolnirResearch(config=_cfg("1m"), home_root="/tmp")
    res._symbol_features = {"BTCUSDT": _bars(20, "5s")}
    with pytest.raises(RuntimeError, match="resolves the measured bar"):
        res.verticalize()
