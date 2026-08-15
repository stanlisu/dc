"""The price a signal is sent at MUST come from the row the model predicted on.

Regression contract for the one-bar-stale live defect found 2026-08-15 on
`agamotto.base.15m_1` (hydra, ltp + sumo, DRY_RUN:false).

WHAT WENT WRONG. `make_decision` priced every signal at `self.raw[col].iloc[-2]`,
commented "not the incomplete current candle whose close is NaN". That premise
is true of the frame `_fetch_and_prepare_data` BUILDS (the REST fetch "includes
the current incomplete candle"; `knull/kline_stream.py` appends the in-flight bar
on the WS path deliberately so the two match) — but `self.raw` is assigned ONCE,
in `_process_combined`, AFTER `combined = combined.iloc[:-1]` has removed it. So
by the time `make_decision` runs, `iloc[-1]` IS the just-closed candle and
`iloc[-2]` is a full TIME_UNIT older.

Measured on hydra at 03:00:14 on 2026-08-15 (`tesseract/probe_agamotto_bars.py`,
read-only, live box, deployed package):

    raw.index[-3:]  ['2026-08-15 02:15:00', '02:30:00', '02:45:00']
    raw rows        699                      <- 700 fetched, exactly ONE dropped
    iloc[-1]        02:45:00  BTC 63113.6    <- Binance: opens 02:45, CLOSES 03:00
    iloc[-2]        02:30:00  BTC 63088.9    <- closed 02:45, one full 15m older
    in-progress     03:00:00                 <- NOT PRESENT in raw at all
    vertical_features['timestamp'].max() = 02:45:00   <- the row predict() targets

Over 550 live signals (2026-08-07..15, 28 symbols) every emitted price matched
`close(T-2)`; none matched `close(T-1)` exclusively. Cost: median +23.8 bps per
entry on ltp / +29.6 on sumo versus the research anchor.

THE INVARIANT THIS FILE PINS. `predict()` targets
`self.vertical_features["timestamp"].max()`. The price must come from THAT
timestamp, selected by LABEL, not by position — a positional index is what let
two layers each believe they owned the "drop the incomplete bar" step without
either noticing the other. These tests are written against
`_closes_at_timestamp`, the seam extracted so the rule is checkable without
weights, a regime stack or a network.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agamotto_pkg" / "src"))

from agamotto.trading import _closes_at_timestamp  # noqa: E402

# The three bars live at the tail of `raw` at a 03:00 cycle, exactly as the probe
# observed them. `raw` has ALREADY had the in-progress 03:00 candle removed.
T_STALE = pd.Timestamp("2026-08-15 02:30:00")   # closed at 02:45 — iloc[-2]
T_SETTLED = pd.Timestamp("2026-08-15 02:45:00")  # closed at 03:00 — iloc[-1]

BTC = "BINANCE_PERP_BTC_USDT"
LINK = "BINANCE_PERP_LINK_USDT"


def _raw():
    """`self.raw`'s tail as it really is at a 03:00 cycle (post incomplete-drop)."""
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
    `iloc[-2]` returns 63088.9 / 9.645 — the values the live bot actually sent —
    while the contract requires the 02:45 row's 63113.6 / 9.723."""
    out = _closes_at_timestamp(_raw(), [BTC, LINK], T_SETTLED)
    assert out[BTC] == pytest.approx(63113.6)
    assert out[LINK] == pytest.approx(9.723)


def test_it_is_not_positional_so_a_second_drop_cannot_shift_it():
    """Append rows AFTER the target timestamp: a positional reader silently
    slides, a label-based one cannot. This is the failure mode that produced the
    bug — two layers each dropping 'the incomplete bar'."""
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
    hide it. The live bug was invisible for exactly this reason — every layer
    had a plausible answer."""
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
