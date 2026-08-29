"""orb's panel must carry the seven scale-free twins it already computes.

orb never LACKED them: its per-TF `AgamottoResearch` instances compute
`sar_dist`/`bb_pctb`/… into each TF's wide frame (`agamotto/research.py`
scale-free block) and `_align_timeframes` carries them through as
`{tf}_{native}_sar_dist`. `verticalize()`'s step-1 loop iterates
`_DERIVED_FEATURES`, NOT the frame, and that list omitted all seven — so the
panel silently lost them. Same failure SHAPE as agamotto's pre-`104d740`
rename-map drop, a different mechanism (a list of names vs a rename map).

Measured consequence (`marvel/gauntlet/orb_vs_agamotto_features_20260822.md`
§1a): orb's 15m block was 40 features, a strict SUBSET of agamotto's 47, and
those seven twins were 1,794 of agamotto's exclusive top-5 picks — `sar_dist`
alone in 663 of 1,887 leg-windows.

TA-Lib is required here: the twins are derived from `sar`/`bb_*`/`macd*`/`obv`/
`ad`, and without TA-Lib `engineer_features` logs a warning and emits none of
them. That is a legitimately degraded frame, not the defect under test — see
`test_tripwire_silent_on_a_frame_that_never_had_them`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("talib", reason="scale-free twins are derived from TA-Lib outputs")

from agamotto import AgamottoResearch                       # noqa: E402
from agamotto.features_scalefree import SCALE_FREE_FEATURES  # noqa: E402
from orb import research as orb_research                    # noqa: E402
from orb.research import OrbResearch                        # noqa: E402


NATIVE = "BTCUSDT"
SYMBOL = "BINANCE_PERP_BTC_USDT"
_ROWS_15M = 400
_FREQ = {"15m": ("15min", 1), "1h": ("1h", 4)}


def _make_ohlcv(n: int, start: str, freq: str) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq=freq)
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5)
    return pd.DataFrame({
        f"{NATIVE}_open": close - 0.1,
        f"{NATIVE}_high": close + 0.5,
        f"{NATIVE}_low": close - 0.5,
        f"{NATIVE}_close": close,
        f"{NATIVE}_volume": rng.integers(100, 1000, n).astype(float),
    }, index=idx)


def _config() -> dict:
    return {
        "SYMBOLS": [SYMBOL],
        "EXCHANGE": "BINANCE",
        "DATA": "liquid",
        "TIMEFRAMES": ["15m", "1h"],
        "BASE_TF": "15m",
        "TARGET_TF": "15m",
        "TIME_UNIT": "15m",
        "LADDER": 1,
        "LADDER_BPS": 1.0,
        "FEE": 0,
        "MA_PERIODS": [7, 25, 99],
        "STATS_WINDOW": 14,
    }


def _dest_names(orb: OrbResearch, tf: str, feat: str) -> list[str]:
    """Panel name(s) a per-TF derived feature may legitimately land under.

    The TARGET_TF block is emitted UNPREFIXED (verticalize step 3b) and the
    other TFs prefixed (step 1). Both names are accepted for the TARGET_TF so
    this file stays green either side of the de-duplication commit, which is
    what decides between them; every other TF is pinned to its prefix.
    """
    if tf == orb.target_tf:
        return [feat, f"{tf}_{feat}"]
    return [f"{tf}_{feat}"]


def _panel() -> OrbResearch:
    """Engineer + verticalize a two-TF orb panel with no disk I/O."""
    cfg = _config()
    orb = OrbResearch(cfg, "/tmp/fake_root")
    for tf in cfg["TIMEFRAMES"]:
        freq, div = _FREQ[tf]
        inst = AgamottoResearch({**cfg, "TIME_UNIT": tf}, "/tmp/fake_root")
        inst.raw = _make_ohlcv(_ROWS_15M // div, "2025-01-01", freq)
        orb._tf_instances[tf] = inst
    orb.raw = orb._tf_instances[cfg["BASE_TF"]].raw
    orb.engineer_features()
    orb.verticalize()
    return orb


# ── the seven twins reach the panel, on every timeframe ──────────────────────

def _panel_column(orb: OrbResearch, tf: str, name: str) -> pd.Series:
    cols = orb.vertical_features.columns
    for candidate in _dest_names(orb, tf, name):
        if candidate in cols:
            return orb.vertical_features[candidate]
    raise AssertionError(
        f"{name} absent from the panel for {tf}; looked for "
        f"{_dest_names(orb, tf, name)}")


def test_scale_free_twins_present_for_every_timeframe():
    orb = _panel()
    cols = set(orb.vertical_features.columns)
    missing = [(tf, name)
               for tf in orb.timeframes for name in SCALE_FREE_FEATURES
               if not any(c in cols for c in _dest_names(orb, tf, name))]
    assert not missing, f"scale-free twins absent from the panel: {missing}"


def test_twins_are_the_values_engineer_features_computed():
    """Carried through, not recomputed — the panel column IS the frame column.

    Recomputing here would impose orb's indicator parameters on a series
    agamotto built with its own (`features_scalefree.py`, "DERIVED, NOT
    RECOMPUTED").
    """
    orb = _panel()
    for tf in orb.timeframes:
        for name in SCALE_FREE_FEATURES:
            src = orb.features[f"{tf}_{NATIVE}_{name}"].reset_index(drop=True)
            dst = _panel_column(orb, tf, name).reset_index(drop=True)
            pd.testing.assert_series_equal(src, dst, check_names=False)


def test_twins_are_not_all_nan():
    """A twin that arrives all-NaN is as useless as an absent one."""
    orb = _panel()
    for name in SCALE_FREE_FEATURES:
        col = _panel_column(orb, "15m", name)
        assert col.notna().any(), f"{name} (15m) is entirely NaN"


def test_derived_features_list_is_not_a_second_hardcoded_copy():
    """Driven off `SCALE_FREE_FEATURES` — two copies drifting IS the defect."""
    assert set(SCALE_FREE_FEATURES) <= set(orb_research._DERIVED_FEATURES)


# ── the tripwire: a twin in the frame but not in _DERIVED_FEATURES ───────────

def test_tripwire_raises_when_every_twin_is_dropped(monkeypatch):
    """Restore the pre-fix list and verticalize must FAIL, not quietly ship 40.

    This is the mutation check for the tripwire: without it the same monkeypatch
    produces a panel that merely lacks the columns, which is exactly how the
    defect survived from 2026-08-06 to 2026-08-22.
    """
    pre_fix = [f for f in orb_research._DERIVED_FEATURES
               if f not in set(SCALE_FREE_FEATURES)]
    monkeypatch.setattr(orb_research, "_DERIVED_FEATURES", pre_fix)
    with pytest.raises(KeyError, match="_DERIVED_FEATURES does not carry"):
        _panel()


def test_tripwire_names_the_dropped_column(monkeypatch):
    partial = [f for f in orb_research._DERIVED_FEATURES if f != "sar_dist"]
    monkeypatch.setattr(orb_research, "_DERIVED_FEATURES", partial)
    with pytest.raises(KeyError) as exc:
        _panel()
    assert f"15m_{NATIVE}_sar_dist" in str(exc.value)


def test_tripwire_silent_on_a_frame_that_never_had_them():
    """A TA-Lib failure leaves the source columns ABSENT — degraded, not broken.

    The tripwire must not turn that into a dead run, matching the
    `logger.warning` branch that guards agamotto's scale-free block.
    """
    orb = _panel()
    keep = [c for c in orb.features.columns
            if not any(c.endswith(f"_{n}") for n in SCALE_FREE_FEATURES)]
    orb.features = orb.features[keep]
    orb.verticalize()          # must not raise
    cols = set(orb.vertical_features.columns)
    assert not [c for c in cols
                if any(c.endswith(n) for n in SCALE_FREE_FEATURES)]
