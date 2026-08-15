"""The price a signal is sent at MUST come from the row the model predicted on.

Regression contract for orb's copy of the one-bar-stale defect found 2026-08-15
on agamotto (`tests/test_agamotto_price_row.py`, dc `f47d588`). orb is NOT live,
so this is a port made before it could cost anything — not an incident.

WHAT IS WRONG. `OrbTrading.make_decision` priced every signal at
`self.raw[col].iloc[-2]`, inherited verbatim from `AgamottoTrading` along with
the premise "the last row is the incomplete candle". That premise is false on
BOTH of orb's data paths, and for two different reasons:

  * REST (`_fetch_and_prepare_data`, orb_pkg/src/orb/trading.py:427-431) fetches
    `limit + 1` bars and then drops the in-flight one — "After this drop we have
    exactly `limit` closed bars" — and only THEN assigns
    `self.raw = self._tf_instances[self.base_tf].raw` at :447.
  * WS (:320-322) never has the in-flight bar to begin with — "WS buffer only
    contains closed bars — do NOT drop the last bar (unlike REST which includes
    the open bar)" — and assigns `self.raw` at :333.

`self.raw` is assigned in exactly those two places (nowhere else), and both sit
downstream of the drop. So `iloc[-1]` IS the just-closed bar and `iloc[-2]` is a
full BASE_TF older than the row `predict()` ran on.

WHAT `predict()` TARGETS. orb is cross-TF, so this needed its own check rather
than copying agamotto's expression — but it lands in the same place.
`OrbTrading.predict` (:494) selects `filtered_signals["timestamp"] == target_ts`
with `target_ts = self.vertical_features["timestamp"].max()`. That column is set
in `OrbResearch.verticalize` (orb_pkg/src/orb/research.py:392) as
`self.features.index`; `self.features` is built by `_align_timeframes` (:443,
:575) as `pd.DataFrame(index=base_idx)` where `base_idx` is the BASE_TF
features index; and `AgamottoResearch.engineer_features` concatenates per-symbol
frames indexed by `self.raw.index` without dropping rows. Higher TFs are
merge_asof'd ONTO that index, never widening it. Hence
`vertical_features["timestamp"].max() == self.raw.index.max()` — the settled bar,
which `iloc[-2]` misses by one.

THE INVARIANT THIS FILE PINS. The price must come from the target timestamp
selected by LABEL, not by position — a positional index is what let two layers
each believe they owned the "drop the incomplete bar" step without either
noticing the other. These tests are written against `_closes_at_timestamp`, the
seam extracted so the rule is checkable without weights, a regime stack or a
network.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "agamotto_pkg" / "src"))
sys.path.insert(0, str(_ROOT / "orb_pkg" / "src"))

from orb.trading import OrbTrading, _closes_at_timestamp  # noqa: E402

# The tail of `self.raw` (BASE_TF = 15m) at a 03:00 cycle. The in-progress 03:00
# candle is already gone — dropped on the REST path, never present on the WS one.
T_STALE = pd.Timestamp("2026-08-15 02:30:00")   # closed at 02:45 — iloc[-2]
T_SETTLED = pd.Timestamp("2026-08-15 02:45:00")  # closed at 03:00 — iloc[-1]

BTC = "BINANCE_PERP_BTC_USDT"
LINK = "BINANCE_PERP_LINK_USDT"


def _raw():
    """`self.raw`'s tail at a 03:00 cycle, post incomplete-bar drop."""
    return pd.DataFrame(
        {
            "BTCUSDT_close": [63073.3, 63088.9, 63113.6],
            "LINKUSDT_close": [9.552, 9.645, 9.723],
        },
        index=pd.DatetimeIndex(
            [pd.Timestamp("2026-08-15 02:15:00"), T_STALE, T_SETTLED]),
    )


def test_price_comes_from_the_row_predict_targets():
    """THE regression. The settled row is the LAST row here, so the old
    `iloc[-2]` returns 63088.9 / 9.645 — one full BASE_TF stale — while the
    contract requires the 02:45 row's 63113.6 / 9.723."""
    out = _closes_at_timestamp(_raw(), [BTC, LINK], T_SETTLED)
    assert out[BTC] == pytest.approx(63113.6)
    assert out[LINK] == pytest.approx(9.723)


def test_it_is_not_positional_so_a_second_drop_cannot_shift_it():
    """Append rows AFTER the target timestamp: a positional reader silently
    slides, a label-based one cannot. This is the failure mode that produced the
    bug — orb's REST path drops 'the incomplete bar' at trading.py:431 while the
    reader assumed nobody had."""
    raw = _raw()
    extra = pd.DataFrame(
        {"BTCUSDT_close": [63200.0], "LINKUSDT_close": [9.9]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-08-15 03:00:00")]))
    out = _closes_at_timestamp(pd.concat([raw, extra]), [BTC, LINK], T_SETTLED)
    assert out[BTC] == pytest.approx(63113.6)
    assert out[LINK] == pytest.approx(9.723)


def test_a_missing_target_timestamp_raises_rather_than_guessing():
    """No silent fallback (CLAUDE.md). If the row the model predicted on is not
    in `raw`, the two have genuinely diverged and pricing anything at all would
    hide it. orb has TWO data paths building `raw` under different rules, so the
    divergence this catches is more reachable here than in agamotto."""
    with pytest.raises(KeyError, match="not present in raw"):
        _closes_at_timestamp(_raw(), [BTC], pd.Timestamp("2026-08-15 03:00:00"))


def test_a_symbol_with_no_close_column_prices_zero():
    """Unchanged behaviour: an absent column yields 0.0, which the caller's
    `close > 0` guard turns into 'fall back to the init-time size'."""
    out = _closes_at_timestamp(_raw(), ["BINANCE_PERP_NOTREAL_USDT"], T_SETTLED)
    assert out["BINANCE_PERP_NOTREAL_USDT"] == 0.0


def test_a_nan_close_falls_back_to_the_last_valid_one():
    """Unchanged behaviour, and still needed: a symbol listed but not trading
    can carry NaN at its newest bars. Fall back to the last valid close AT OR
    BEFORE the target — never to a later one, which would be lookahead."""
    raw = _raw()
    raw.loc[T_SETTLED, "BTCUSDT_close"] = np.nan
    out = _closes_at_timestamp(raw, [BTC], T_SETTLED)
    assert out[BTC] == pytest.approx(63088.9)


def test_the_nan_fallback_never_reaches_forward_in_time():
    """A NaN at the target must not be filled from a LATER bar — that would be
    lookahead, and it is reachable whenever `raw` still holds rows past the
    target (see the positional test above)."""
    raw = _raw()
    raw.loc[T_SETTLED, "BTCUSDT_close"] = np.nan
    extra = pd.DataFrame({"BTCUSDT_close": [99999.0], "LINKUSDT_close": [9.9]},
                         index=pd.DatetimeIndex([pd.Timestamp("2026-08-15 03:00:00")]))
    out = _closes_at_timestamp(pd.concat([raw, extra]), [BTC], T_SETTLED)
    assert out[BTC] == pytest.approx(63088.9)


def test_all_nan_prices_zero_rather_than_inventing_a_number():
    raw = _raw()
    raw["BTCUSDT_close"] = np.nan
    assert _closes_at_timestamp(raw, [BTC], T_SETTLED)[BTC] == 0.0


def test_empty_raw_prices_nothing():
    out = _closes_at_timestamp(pd.DataFrame(), [BTC], T_SETTLED)
    assert out == {BTC: 0.0}


# ---------------------------------------------------------------------------
# End-to-end: the price that actually leaves make_decision
# ---------------------------------------------------------------------------


def _orb():
    """`OrbTrading` with the network, weights and feature engineering removed,
    holding exactly the `raw` / `vertical_features` pairing the real
    `_fetch_and_prepare_data` produces (verified by driving the REST path with a
    patched fetcher: `raw` ends at the settled bar, the in-flight bar absent)."""
    config = {
        "TIME_UNIT": "15m",
        "TIMEFRAME_SECONDS": 900,
        "TIMEFRAMES": ["15m"],
        "BASE_TF": "15m",
        "TARGET_TF": "15m",
        "SIZES": [0.01],
        "SYMBOLS": [BTC],
        "CAPITAL": 1000,
        "TRADING_MODE": "both",
        "LONG_PRED_THRESHOLD": 0.0,
        "SHORT_PRED_THRESHOLD": 0.0,
        "REGIME_STACK_PATH": "/tmp/fake_regime_stack.csv",
        "LOT_SIZES": {BTC: {"step_size": 0.001, "min_notional": 5.0}},
    }
    with patch.object(OrbTrading, "_load_regime_stack",
                      lambda s: setattr(s, "regime_stack", [])), \
         patch.object(OrbTrading, "_calculate_sizes"), \
         patch.object(OrbTrading, "load_data"):
        inst = OrbTrading(config=config, home_root="/tmp",
                          period="window_test", skip_load=True)

    inst.engineer_features = MagicMock()
    inst.verticalize = MagicMock()
    inst.vertical_features = pd.DataFrame({
        "timestamp": [T_SETTLED],
        "symbol": [BTC],
        "feat_a": [1.0],
        "feat_b": [2.0],
        "position": ["long"],
    })
    inst.features = inst.vertical_features.copy()
    inst.raw = _raw()[["BTCUSDT_close"]]
    inst._data_fresh = True
    return inst


def _long_regime():
    model = MagicMock()
    model.predict.return_value = [0.01]
    scaler = MagicMock()
    scaler.transform.return_value = [[0.1, 0.2]]
    return {
        "id": "r_long", "regime": "test_regime", "position": "long",
        "model_name": "mock", "threshold": 0.005,
        "artifact": {"model": model, "scaler": scaler,
                     "metadata": {"feature_columns": ["feat_a", "feat_b"]}},
        "filter": None, "config": {},
    }


def test_make_decision_emits_the_settled_close_not_the_one_before_it():
    """The end-to-end regression: it is `decisions[sym][0]` that the executor
    anchors its LIMIT order on. On the old code this returned 63088.9 — the
    02:30 bar — for a signal computed on 02:45."""
    inst = _orb()
    inst.regime_stack = [_long_regime()]
    inst.filter_signals = lambda regime, save=False: inst.vertical_features

    price, qty = inst.make_decision()[BTC]

    assert price == pytest.approx(63113.6), (
        "priced off the wrong bar — 63088.9 means the stale iloc[-2] is back")
    # Sizing reads the same close, so it moves with it.
    assert qty == pytest.approx(round(1000 / 63113.6 / 0.001) * 0.001)
