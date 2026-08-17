"""The price a signal is sent at MUST come from the row the model predicted on.

Regression contract for aether's copy of the one-bar-stale defect found
2026-08-15 on agamotto (`tests/test_agamotto_price_row.py`, dc `f47d588`) and
ported to orb the same day (`tests/test_orb_price_row.py`, dc `74ce8a1`). aether
was the third carrier and the LAST to be fixed: `f47d588`'s own commit message
left it explicitly unverified — "aether has no upstream drop, so its iloc[-2]
may be correct ... needs its own verification, deliberately out of this diff" —
and nothing came back to it until 2026-08-17. aether is not live, so this cost
nothing.

WHY NEITHER EARLIER FIX REACHED IT. `AetherTrading` extends `AetherResearch` ->
`OrbResearch`, so it is NOT a subclass of `AgamottoTrading` and NOT of
`OrbTrading`. Both fixes landed on classes aether does not inherit from, and
`AetherTrading.make_decision` held its own verbatim
`self.raw[col].iloc[-2] if len(self.raw) >= 2 else self.raw[col].iloc[-1]`.

AETHER DOES HAVE AN UPSTREAM DROP, contrary to the note in `f47d588`.
`AetherTrading.load_data` (aether_pkg/src/aether/trading.py:157-160) is a
three-line delegation to `OrbTrading.load_data(self, limit=limit)`, so `self.raw`
is built by orb's `_fetch_and_prepare_data` and assigned in exactly orb's two
places, both downstream of the in-flight-bar drop:

  * REST (orb_pkg/src/orb/trading.py:480) drops it — "After this drop we have
    exactly `limit` closed bars" — and assigns `self.raw` at :496.
  * WS (:370-371) never has it — "WS buffer only contains closed bars — do NOT
    drop the last bar" — and assigns at :382.

So `iloc[-1]` IS the just-closed bar and `iloc[-2]` is a full BASE_TF older.

WHAT AETHER'S FORWARD PASSES TARGET. `AetherResearch` overrides only `__init__`,
`create` and `train_pooled_models` (aether_pkg/src/aether/research.py:22, :25,
:66), so `engineer_features`, `verticalize` and `_align_timeframes` are
`OrbResearch`'s verbatim. `make_decision` sets
`target_ts = self.vertical_features["timestamp"].max()` (aether trading.py:184)
and both pooled forward passes score only
`vertical_features[timestamp == target_ts]`; the regime vote at :231 filters on
the same label. `verticalize` sets that column to `self.features.index` (orb
research.py:392) and `_align_timeframes` (orb research.py:437) builds
`self.features` on the BASE_TF features index (:440-441, assigned :575), which
`engineer_features` carries from `raw.index` row-for-row. Hence
`vertical_features["timestamp"].max() == self.raw.index.max()` — the settled bar,
which `iloc[-2]` missed by one.

Every line number above was re-derived by grep against the post-fix tree
(2026-08-17). An earlier draft of this file carried citations taken from the
pre-fix orb source, stale by 28 lines because deleting orb's duplicated 63-line
helper shifted everything below it — which is why the same-file citations inside
`orb/trading.py` and `aether/trading.py` are now written as SYMBOLS instead.

`create()`'s `self.features = self.features.iloc[:-1]` (aether research.py:37)
does NOT apply: that is the offline research path, and the live path reaches
`engineer_features`/`verticalize` through `load_data`, never through `create`.

THE INVARIANT THIS FILE PINS. The price must come from the target timestamp
selected by LABEL, not by position. aether now imports the ONE
`_closes_at_timestamp` from `agamotto.trading` rather than growing a third copy,
so these tests also pin that the import is wired to the price path.
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
sys.path.insert(0, str(_ROOT / "aether_pkg" / "src"))

from aether.trading import AetherTrading, _closes_at_timestamp  # noqa: E402

# The tail of `self.raw` (BASE_TF = 15m) at a 03:00 cycle. The in-progress 03:00
# candle is already gone — dropped on the REST path, never present on the WS one.
T_STALE = pd.Timestamp("2026-08-15 02:30:00")    # closed at 02:45 — iloc[-2]
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


def test_aether_reuses_the_one_helper_rather_than_a_third_copy():
    """`74ce8a1` ported the helper into orb by duplication and it drifted within
    the hour: `cb454fd` hardened agamotto's copy the same day (order-independent
    NaN fallback + duplicate-label guard) and neither change reached orb's copy.
    aether must not add a third — it imports.

    Asserted on `__module__` rather than object identity between
    `aether.trading` and `agamotto.trading`: this conda env has MARVEL's packages
    pip-installed, so which tree a bare `import agamotto` resolves to depends on
    sys.path order at first import, and an identity check would be testing the
    environment instead of the code. `__module__` pins the invariant either way —
    the function aether prices with is DEFINED in `agamotto.trading`, not here."""
    import aether.trading as aether_trading

    assert (aether_trading._closes_at_timestamp.__module__
            == "agamotto.trading"), (
        "aether grew its own copy of the helper — that is how orb's copy drifted")
    assert _closes_at_timestamp.__module__ == "agamotto.trading"


def test_price_comes_from_the_row_the_forward_pass_targets():
    """THE regression. The settled row is the LAST row here, so the old
    `iloc[-2]` returns 63088.9 / 9.645 — one full BASE_TF stale — while the
    contract requires the 02:45 row's 63113.6 / 9.723."""
    out = _closes_at_timestamp(_raw(), [BTC, LINK], T_SETTLED)
    assert out[BTC] == pytest.approx(63113.6)
    assert out[LINK] == pytest.approx(9.723)


def test_it_is_not_positional_so_a_second_drop_cannot_shift_it():
    """Append rows AFTER the target timestamp: a positional reader silently
    slides, a label-based one cannot. This is the failure mode that produced the
    bug — orb's REST path (which builds aether's `raw`) drops 'the incomplete
    bar' while aether's reader assumed nobody had."""
    extra = pd.DataFrame(
        {"BTCUSDT_close": [63200.0], "LINKUSDT_close": [9.9]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-08-15 03:00:00")]))
    out = _closes_at_timestamp(
        pd.concat([_raw(), extra]), [BTC, LINK], T_SETTLED)
    assert out[BTC] == pytest.approx(63113.6)
    assert out[LINK] == pytest.approx(9.723)


def test_a_missing_target_timestamp_raises_rather_than_guessing():
    """No silent fallback (CLAUDE.md). If the row the forward pass scored is not
    in `raw`, the two have genuinely diverged and pricing anything at all would
    hide it. aether inherits orb's TWO data paths, so the divergence this catches
    is as reachable here as in orb."""
    with pytest.raises(KeyError, match="not present in raw"):
        _closes_at_timestamp(_raw(), [BTC], pd.Timestamp("2026-08-15 03:00:00"))


def test_a_symbol_with_no_close_column_prices_zero():
    """Unchanged behaviour, and load-bearing for aether specifically: the old
    code left such a symbol OUT of `latest_closes` entirely and the sizing loop's
    `latest_closes.get(sym, 0.0)` turned that into 0.0. The helper returns the
    0.0 explicitly — same value, same `close > 0` branch to the init-time size."""
    out = _closes_at_timestamp(_raw(), ["BINANCE_PERP_NOTREAL_USDT"], T_SETTLED)
    assert out["BINANCE_PERP_NOTREAL_USDT"] == 0.0


def test_a_nan_close_falls_back_to_the_last_valid_one():
    """DELIBERATE BEHAVIOUR CHANGE for aether — NOT a preserved behaviour, and the
    only one in this fix. Called out because an earlier draft of this docstring
    labelled it "unchanged", which was simply wrong.

    BEFORE: `float(val) if not pd.isna(val) else 0.0`. A NaN close became **0.0**,
    and `knull/orb_bridge._decisions_to_signals` (marvel, :156-157) maps
    `price <= 0` to `("FLAT", price, 0)` — no order at all. So a NaN close meant
    aether stood down for that symbol on that bar.

    AFTER: the shared helper returns the last valid close AT OR BEFORE the target.
    So where aether previously placed NOTHING it will now place an order at an
    OLDER price — a real change in what reaches an executor, not a refactor.

    Kept anyway, for three reasons: it matches agamotto and orb, so all three
    kline arms fail the same way; it is `logger.warning`-ed on every occurrence,
    so it is visible rather than silent; and the alternative (0.0 -> FLAT) is the
    silent-fallback pattern CLAUDE.md bans — it discards the signal without
    saying so. The blast radius today is zero: aether has no live path (absent
    from `launch_bots.ALGO_REGISTRY` and `orb_bridge._load_strategy`), so this
    lands before aether ever trades, which is the point of fixing it now.

    NOT a change for vomir, whose old fallback was already
    `raw[col].dropna().iloc[-1]` — a close, never 0.0. There the change is only
    WHICH close (now masked to `index <= target_ts`), and vomir IS live."""
    raw = _raw()
    raw.loc[T_SETTLED, "BTCUSDT_close"] = np.nan
    out = _closes_at_timestamp(raw, [BTC], T_SETTLED)
    assert out[BTC] == pytest.approx(63088.9)
    assert out[BTC] != 0.0, "0.0 would be the old FLAT-producing behaviour"


def test_the_nan_fallback_never_reaches_forward_in_time():
    raw = _raw()
    raw.loc[T_SETTLED, "BTCUSDT_close"] = np.nan
    extra = pd.DataFrame(
        {"BTCUSDT_close": [99999.0], "LINKUSDT_close": [9.9]},
        index=pd.DatetimeIndex([pd.Timestamp("2026-08-15 03:00:00")]))
    out = _closes_at_timestamp(pd.concat([raw, extra]), [BTC], T_SETTLED)
    assert out[BTC] == pytest.approx(63088.9)


def test_nan_fallback_is_order_independent():
    """Inherited from `cb454fd`, which aether gets for free by importing rather
    than copying: label slicing a NON-monotonic index does not raise — pandas
    slices POSITIONALLY and silently includes rows dated AFTER the target."""
    raw = pd.DataFrame(
        {"BTCUSDT_close": [63073.3, 63088.9, 99999.0, np.nan]},
        index=pd.DatetimeIndex([
            pd.Timestamp("2026-08-15 02:15:00"),
            T_STALE,                                 # 02:30
            pd.Timestamp("2026-08-15 03:00:00"),     # LATER than the target
            T_SETTLED,                               # 02:45, the target, NaN
        ]),
    )
    assert not raw.index.is_monotonic_increasing     # the precondition under test
    assert _closes_at_timestamp(raw, [BTC], T_SETTLED)[BTC] == pytest.approx(
        63088.9)


def test_duplicate_index_labels_raise_rather_than_exploding_downstream():
    raw = _raw()
    dup = pd.concat([raw, raw.iloc[[-1]]])           # T_SETTLED twice
    with pytest.raises(ValueError, match="duplicate"):
        _closes_at_timestamp(dup, [BTC], T_SETTLED)


def test_all_nan_prices_zero_rather_than_inventing_a_number():
    raw = _raw()
    raw["BTCUSDT_close"] = np.nan
    assert _closes_at_timestamp(raw, [BTC], T_SETTLED)[BTC] == 0.0


def test_empty_raw_prices_nothing():
    assert _closes_at_timestamp(pd.DataFrame(), [BTC], T_SETTLED) == {BTC: 0.0}


# ---------------------------------------------------------------------------
# End-to-end: the price that actually leaves make_decision
# ---------------------------------------------------------------------------


def _aether(tmp_stack: Path):
    """`AetherTrading` with the network, weights and feature engineering removed,
    holding exactly the `raw` / `vertical_features` pairing
    `OrbTrading._fetch_and_prepare_data` produces: `raw` ends at the settled bar
    and the in-flight bar is absent."""
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
        "REGIME_STACK_PATH": str(tmp_stack),
        "LOT_SIZES": {BTC: {"step_size": 0.001, "min_notional": 5.0}},
    }
    with patch.object(AetherTrading, "_load_regime_stack",
                      lambda s, p: setattr(s, "regime_stack", [])), \
         patch.object(AetherTrading, "_load_models"), \
         patch.object(AetherTrading, "load_data"):
        inst = AetherTrading(config=config, home_root="/tmp",
                             period="window_test", skip_load=True)

    inst.engineer_features = MagicMock()
    inst.verticalize = MagicMock()
    inst.vertical_features = pd.DataFrame({
        "timestamp": [T_SETTLED],
        "symbol": [BTC],
        "feat_a": [1.0],
        "feat_b": [2.0],
    })
    inst.features = inst.vertical_features.copy()
    inst.raw = _raw()[["BTCUSDT_close"]]
    inst._data_fresh = True

    model = MagicMock()
    model.predict.return_value = [0.01]
    scaler = MagicMock()
    scaler.transform.return_value = [[0.1, 0.2]]
    inst.models = {"long": {"model": model, "scaler": scaler,
                            "feature_columns": ["feat_a", "feat_b"]}}
    inst.regime_stack = [{
        "regime": "test_regime", "position": "long", "threshold": 0.005,
        "id": "r_long",
    }]
    inst.filter_signals = lambda regime, save=False: inst.vertical_features
    return inst


def test_make_decision_emits_the_settled_close_not_the_one_before_it(tmp_path):
    """The end-to-end regression: it is `decisions[sym][0]` that the executor
    anchors its LIMIT order on. On the old code this returned 63088.9 — the
    02:30 bar — for a vote computed on 02:45."""
    stack = tmp_path / "regime_stack.csv"
    stack.write_text("regime,position,threshold\n")
    inst = _aether(stack)

    price, qty = inst.make_decision()[BTC]

    assert price == pytest.approx(63113.6), (
        "priced off the wrong bar — 63088.9 means the stale iloc[-2] is back")
    # net = +1 long vote, and sizing reads the same close, so qty moves with it.
    assert qty == pytest.approx(round(1000 / 63113.6 / 0.001) * 0.001)
