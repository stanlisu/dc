"""vomir prices at the row the classifier scored — pinned against DC SOURCE.

Companion to `test_agamotto_price_row.py` / `test_orb_price_row.py` /
`test_aether_price_row.py`. Two things make this file necessary rather than
redundant.

1. `vomir_pkg/tests/*` DO NOT TEST THIS REPO. They bare-import
   `from vomir.trading import VomirTrading` with no `sys.path` insertion, and this
   env has MARVEL's packages pip-installed, so `vomir.trading` resolves to
   `/home/stan/sandbox/marvel/vomir_pkg/src/vomir/trading.py` — a PyArmor blob
   built from whatever dc source was checked out at build time. Verified
   2026-08-17 by printing `vomir.trading.__file__` from inside that suite. Their
   17 passes say nothing about dc's source, which is why this file inserts the dc
   paths explicitly and asserts on `__file__`.

2. VOMIR NEVER HAD THE `iloc[-2]` DEFECT, so the contract needs stating rather
   than just restoring. `_get_latest_closes` read `raw[col].iloc[-1]`, which IS
   the row the classifier scored: `AgamottoTrading._process_combined` assigns
   `self.raw` then calls `engineer_features()` + `verticalize()` back to back
   (agamotto_pkg/src/agamotto/trading.py:487-489), and `engineer_features`
   preserves `raw.index` row-for-row — verified empirically at 60 bars,
   `features.index.max() == vertical_features["timestamp"].max() ==
   raw.index.max()`. The VALUE was right.

   What was wrong was the NaN fallback, and it was worse than the one `cb454fd`
   fixed in agamotto: `raw[col].dropna().iloc[-1]` takes the last valid close
   ANYWHERE in `raw`, with no at-or-before-target constraint at all. And the
   correct value held only because of that back-to-back coupling — the same "two
   layers each believe they own the bar bookkeeping" arrangement that made
   agamotto's positional read wrong, one refactor away from repeating it.

`VomirTrading` extends `AgamottoTrading` but OVERRIDES `make_decision` and owns
its own `_get_latest_closes`, so `f47d588`'s fix to `AgamottoTrading.make_decision`
never touched it — inheritance was not enough, which is the trap this family of
tests exists to catch.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
# Must precede the marvel install — see the module docstring.
sys.path.insert(0, str(_ROOT / "agamotto_pkg" / "src"))
sys.path.insert(0, str(_ROOT / "vomir_pkg" / "src"))

from vomir.classifier import CLASS_LONG, CLASS_NEUTRAL, CLASS_SHORT  # noqa: E402
from vomir.trading import VomirTrading  # noqa: E402

T_STALE = pd.Timestamp("2026-08-15 02:30:00")
T_SETTLED = pd.Timestamp("2026-08-15 02:45:00")   # the row the classifier scores
BTC = "BINANCE_PERP_BTC_USDT"


def test_this_suite_actually_exercises_dc_source():
    """The whole point of the file. If `vomir.trading` resolves to marvel's
    PyArmor blob, every assertion below is about a build artifact and none of them
    is about this repo."""
    import vomir.trading as vt
    assert Path(vt.__file__).resolve() == (
        _ROOT / "vomir_pkg" / "src" / "vomir" / "trading.py").resolve(), (
        f"imported {vt.__file__} — not dc source; a bare `import vomir` picks up "
        f"the marvel install, which is exactly what vomir_pkg/tests does")


def _artifacts():
    imputer = MagicMock(); imputer.transform.return_value = np.array([[0.1, 0.2]])
    scaler = MagicMock(); scaler.transform.return_value = np.array([[0.3, 0.4]])
    pca = MagicMock(); pca.transform.return_value = np.array([[0.5, 0.6]])
    kmeans = MagicMock(); kmeans.predict.return_value = np.array([0])
    clf = MagicMock()
    clf.classes_ = np.array([CLASS_SHORT, CLASS_NEUTRAL, CLASS_LONG])
    clf.predict_proba.return_value = np.array([[0.1, 0.1, 0.8]])   # LONG
    return {"imputer": imputer, "scaler": scaler, "pca": pca, "kmeans": kmeans,
            "clf": clf,
            "meta": {"feature_cols": ["feat_a", "feat_b"], "n_clusters": 4,
                     "pca_components": 2}}


def _fake_research_init(self, config, home_root):
    self.config = config
    self.home_root = home_root.rstrip("/")
    self.raw = None
    self.features = None
    self.filtered_signals = None
    self.filtered_signals_long = None
    self.filtered_signals_short = None


def _vomir(raw):
    """`VomirTrading` holding the `raw` / `vertical_features` pairing
    `_process_combined` actually produces: `raw` ENDS at the scored row, because
    that method assigns `self.raw` and re-runs `verticalize()` in the same breath.

    Note this differs from `vomir_pkg/tests`, whose fixture pairs a `raw` ending
    2024-12-31 with a 2025-01-01 `vertical_features` timestamp — `raw` a full day
    SHORT of the scored row, which `_process_combined` cannot produce. That
    inconsistency is invisible to a positional read and fatal to a label one; it
    is left alone here because those tests exercise marvel's blob, not this tree.
    """
    config = {
        "TIME_UNIT": "15m",
        "TIMEFRAME_SECONDS": 900,
        "SIZES": [0.01],
        "SYMBOLS": [BTC],
        "CAPITAL": 1000,
        "TRADING_MODE": "both",
        "VOMIR_WEIGHTS_PATH": "/tmp/fake_vomir_weights",
        "VOMIR_LONG_THRESHOLD": 0.4,
        "VOMIR_SHORT_THRESHOLD": 0.4,
        "LOT_SIZES": {BTC: {"step_size": 0.001, "min_notional": 5.0}},
    }
    with patch("vomir.trading.AgamottoResearch.__init__", _fake_research_init), \
         patch("vomir.trading.VomirClassifier.load_model_artifacts",
               return_value=_artifacts()), \
         patch.object(VomirTrading, "_calculate_sizes"), \
         patch.object(VomirTrading, "load_data"):
        inst = VomirTrading(config=config, home_root="/tmp", skip_load=True)

    inst.raw = raw
    inst.vertical_features = pd.DataFrame({
        "timestamp": [T_SETTLED],
        "symbol": [BTC],
        "feat_a": [1.0],
        "feat_b": [2.0],
    })
    inst._data_fresh = True
    return inst


def _raw(**overrides):
    df = pd.DataFrame(
        {"BTCUSDT_close": [63073.3, 63088.9, 63113.6]},
        index=pd.DatetimeIndex(
            [pd.Timestamp("2026-08-15 02:15:00"), T_STALE, T_SETTLED]),
    )
    for ts, val in overrides.items():
        df.loc[pd.Timestamp(ts), "BTCUSDT_close"] = val
    return df


def test_price_is_the_close_of_the_scored_row():
    price, qty = _vomir(_raw()).make_decision()[BTC]
    assert price == pytest.approx(63113.6)
    assert qty == pytest.approx(round(1000 / 63113.6 / 0.001) * 0.001)


def test_a_later_row_in_raw_cannot_drag_the_price_forward():
    """`iloc[-1]` was correct only because `_process_combined` re-verticalizes in
    the same breath as assigning `raw`. Break that coupling — a `raw` holding a
    bar past the scored row — and a positional read silently prices the FUTURE.
    Label selection cannot, and this is lookahead, not staleness: strictly worse
    than the agamotto defect."""
    later = pd.DataFrame(
        {"BTCUSDT_close": [63200.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-08-15 03:00:00")]))
    price, _ = _vomir(pd.concat([_raw(), later])).make_decision()[BTC]
    assert price == pytest.approx(63113.6), (
        "priced 63200.0 — the bar AFTER the one the classifier scored")


def test_raw_that_does_not_reach_the_scored_row_raises():
    """No silent fallback (CLAUDE.md). vomir's old read would have priced the
    2024-12-31-equivalent bar without comment."""
    short_raw = _raw().iloc[:-1]                       # ends at T_STALE
    with pytest.raises(KeyError, match="not present in raw"):
        _vomir(short_raw).make_decision()


def test_the_nan_fallback_cannot_reach_past_the_scored_row():
    """The one behaviour that was genuinely wrong, not merely fragile. The old
    `raw[col].dropna().iloc[-1]` took the last valid close ANYWHERE in `raw`, so
    with the scored row NaN and a later bar present it returned 63200.0 — a
    future price. It must return the 02:30 close instead."""
    raw = _raw()
    raw.loc[T_SETTLED, "BTCUSDT_close"] = np.nan
    later = pd.DataFrame(
        {"BTCUSDT_close": [63200.0]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-08-15 03:00:00")]))
    price, _ = _vomir(pd.concat([raw, later])).make_decision()[BTC]
    assert price == pytest.approx(63088.9), (
        "63200.0 means the unconstrained dropna() fallback is back")


def test_no_close_column_prices_zero_and_falls_back_to_the_init_size():
    """Unchanged behaviour: 0.0 routes through `_compute_size`'s `price > 0`
    guard to the init-time SIZES entry."""
    raw = _raw().rename(columns={"BTCUSDT_close": "SOMETHINGELSE_close"})
    price, qty = _vomir(raw).make_decision()[BTC]
    assert price == 0.0
    assert qty == pytest.approx(0.01)                  # SIZES[0]
